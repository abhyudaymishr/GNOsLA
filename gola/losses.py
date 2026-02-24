from __future__ import annotations

import torch
import torch.nn.functional as F


def field_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def _as_batched(u: torch.Tensor) -> torch.Tensor:
    if u.ndim == 2:
        return u.unsqueeze(0)
    if u.ndim == 3:
        return u
    raise ValueError(f"expected [N, C] or [B, N, C], got {tuple(u.shape)}")


def divergence_penalty_2d(u: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    h, w = grid_shape
    u = _as_batched(u)
    if u.shape[-1] < 2:
        raise ValueError("u must contain at least two channels: vx, vy")
    if u.shape[1] != h * w:
        raise ValueError("grid_shape does not match flattened node dimension")

    vel = u[..., :2].reshape(u.shape[0], h, w, 2)
    vx = vel[..., 0]
    vy = vel[..., 1]

    dx = 1.0 / max(w - 1, 1)
    dy = 1.0 / max(h - 1, 1)

    dvx_dx = (torch.roll(vx, shifts=-1, dims=2) - torch.roll(vx, shifts=1, dims=2)) / (2.0 * dx)
    dvy_dy = (torch.roll(vy, shifts=-1, dims=1) - torch.roll(vy, shifts=1, dims=1)) / (2.0 * dy)
    div = dvx_dx + dvy_dy
    return (div.square()).mean()


def kinetic_energy_consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    velocity_dims: tuple[int, int] = (0, 1),
) -> torch.Tensor:
    pred = _as_batched(pred)
    target = _as_batched(target)
    vxp, vyp = pred[..., velocity_dims[0]], pred[..., velocity_dims[1]]
    vxt, vyt = target[..., velocity_dims[0]], target[..., velocity_dims[1]]
    energy_pred = 0.5 * (vxp.square() + vyp.square()).mean(dim=1)
    energy_tgt = 0.5 * (vxt.square() + vyt.square()).mean(dim=1)
    return F.mse_loss(energy_pred, energy_tgt)


def enstrophy_error_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    grid_shape: tuple[int, int],
    velocity_dims: tuple[int, int] = (0, 1),
) -> torch.Tensor:
    h, w = grid_shape
    pred = _as_batched(pred)
    target = _as_batched(target)

    if pred.shape[1] != h * w or target.shape[1] != h * w:
        raise ValueError("grid_shape does not match flattened node dimension")

    pred_vel = pred[..., list(velocity_dims)].reshape(pred.shape[0], h, w, 2)
    tgt_vel = target[..., list(velocity_dims)].reshape(target.shape[0], h, w, 2)

    def curl_z(v: torch.Tensor) -> torch.Tensor:
        vx = v[..., 0]
        vy = v[..., 1]
        dx = 1.0 / max(w - 1, 1)
        dy = 1.0 / max(h - 1, 1)
        dvy_dx = (torch.roll(vy, shifts=-1, dims=2) - torch.roll(vy, shifts=1, dims=2)) / (2.0 * dx)
        dvx_dy = (torch.roll(vx, shifts=-1, dims=1) - torch.roll(vx, shifts=1, dims=1)) / (2.0 * dy)
        return dvy_dx - dvx_dy

    vort_pred = curl_z(pred_vel)
    vort_tgt = curl_z(tgt_vel)
    ens_pred = 0.5 * vort_pred.square().mean(dim=(1, 2))
    ens_tgt = 0.5 * vort_tgt.square().mean(dim=(1, 2))
    return F.mse_loss(ens_pred, ens_tgt)
