from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def make_uniform_grid(height: int, width: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, steps=height, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, steps=width, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def _canonicalize_fields_layout(fields: torch.Tensor | np.ndarray, layout: str) -> torch.Tensor | np.ndarray:
    layout = layout.upper()

    def _expand_0(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0)
        return np.expand_dims(x, axis=0)

    def _permute_stchw_to_sthwc(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.permute(0, 1, 3, 4, 2).contiguous()
        return np.transpose(x, (0, 1, 3, 4, 2))

    def _permute_tchw_to_thwc(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.permute(0, 2, 3, 1).contiguous()
        return np.transpose(x, (0, 2, 3, 1))

    if layout == "STHWC":
        if fields.ndim != 5:
            raise ValueError("layout STHWC expects rank-5 tensor [S, T, H, W, C]")
        return fields

    if layout == "STCHW":
        if fields.ndim != 5:
            raise ValueError("layout STCHW expects rank-5 tensor [S, T, C, H, W]")
        return _permute_stchw_to_sthwc(fields)

    if layout == "THWC":
        if fields.ndim != 4:
            raise ValueError("layout THWC expects rank-4 tensor [T, H, W, C]")
        return _expand_0(fields)

    if layout == "TCHW":
        if fields.ndim != 4:
            raise ValueError("layout TCHW expects rank-4 tensor [T, C, H, W]")
        return _expand_0(_permute_tchw_to_thwc(fields))

    if layout != "AUTO":
        raise ValueError(f"unsupported layout '{layout}'")

    if fields.ndim == 5:
        if fields.shape[-1] <= 16:
            return fields
        if fields.shape[2] <= 16:
            return _permute_stchw_to_sthwc(fields)
        raise ValueError(
            "could not infer 5D layout automatically. "
            "Please set layout explicitly to STHWC or STCHW."
        )

    if fields.ndim == 4:
        if fields.shape[-1] <= 16:
            return _expand_0(fields)
        if fields.shape[1] <= 16:
            return _expand_0(_permute_tchw_to_thwc(fields))
        raise ValueError(
            "could not infer 4D layout automatically. "
            "Please set layout explicitly to THWC or TCHW."
        )

    raise ValueError("fields must have rank 4 or 5 for supported layouts")


class PreGenNavierStokes2DDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        fields: torch.Tensor | np.ndarray,
        *,
        delta_t_steps: int = 1,
        layout: str = "auto",
    ) -> None:
        super().__init__()

        fields = _canonicalize_fields_layout(fields, layout=layout)
        if isinstance(fields, torch.Tensor):
            fields = fields.float()
            self._torch_backend = True
            self._dtype = fields.dtype
            self._numpy_resident_bytes = 0
        else:
            self._torch_backend = False
            self._dtype = torch.float32
            self._numpy_resident_bytes = 0 if isinstance(fields, np.memmap) else int(fields.nbytes)

        s, t, h, w, c = fields.shape
        if delta_t_steps <= 0:
            raise ValueError("delta_t_steps must be > 0")
        if t - delta_t_steps <= 0:
            raise ValueError("time_steps must be larger than delta_t_steps")

        self.fields = fields
        self.delta_t_steps = delta_t_steps
        self.num_trajectories = s
        self.time_steps = t
        self.height = h
        self.width = w
        self.channels = c

        self.x = make_uniform_grid(h, w, dtype=torch.float32)
        cell_area = 1.0 / max((h - 1) * (w - 1), 1)
        self.node_weight = torch.full((h * w,), cell_area, dtype=torch.float32)

        self.pairs_per_trajectory = t - delta_t_steps
        self.total_pairs = s * self.pairs_per_trajectory

    @classmethod
    def from_npz(cls, path: str | Path, field_key: str = "field", **kwargs: Any) -> "PreGenNavierStokes2DDataset":
        path = Path(path)
        data = np.load(path)
        if field_key not in data:
            raise KeyError(f"field_key '{field_key}' not in {path}")
        return cls(data[field_key], **kwargs)

    @classmethod
    def from_npy(
        cls,
        path: str | Path,
        *,
        mmap: bool = True,
        **kwargs: Any,
    ) -> "PreGenNavierStokes2DDataset":
        path = Path(path)
        data = np.load(path, mmap_mode="r" if mmap else None)
        return cls(data, **kwargs)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        field_key: str = "field",
        mmap: bool = True,
        **kwargs: Any,
    ) -> "PreGenNavierStokes2DDataset":
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            return cls.from_npz(path, field_key=field_key, **kwargs)
        if suffix == ".npy":
            return cls.from_npy(path, mmap=mmap, **kwargs)
        raise ValueError(f"unsupported data file extension '{suffix}', expected .npz or .npy")

    @property
    def resident_fields_bytes(self) -> int:
        if self._torch_backend:
            return int(self.fields.numel() * self.fields.element_size())
        return self._numpy_resident_bytes

    def __len__(self) -> int:
        return self.total_pairs

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        traj = index // self.pairs_per_trajectory
        t = index % self.pairs_per_trajectory
        t_next = t + self.delta_t_steps

        if self._torch_backend:
            u_t = self.fields[traj, t].reshape(-1, self.channels).float()
            u_t_next = self.fields[traj, t_next].reshape(-1, self.channels).float()
        else:
            # np.memmap slices can be read-only; copy=True guarantees writable storage for torch.from_numpy.
            u_t_np = np.array(self.fields[traj, t], dtype=np.float32, copy=True, order="C")
            u_t_next_np = np.array(self.fields[traj, t_next], dtype=np.float32, copy=True, order="C")
            u_t = torch.from_numpy(u_t_np.reshape(-1, self.channels))
            u_t_next = torch.from_numpy(u_t_next_np.reshape(-1, self.channels))

        return {
            "u_t": u_t,
            "u_t_next": u_t_next,
            "x": self.x,
            "node_weight": self.node_weight,
        }
