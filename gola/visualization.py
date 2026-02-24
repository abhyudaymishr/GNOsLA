from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class AttentionVisualizationConfig:
    nx: int = 96
    ny: int = 96
    holes: tuple[tuple[float, float, float], ...] = ((0.45, 0.52, 0.14), (0.63, 0.45, 0.11))
    query_point: tuple[float, float] = (0.26, 0.50)
    radius: float = 0.34
    sigma: float = 0.10
    max_edges: int = 140


@dataclass
class DatasetAttentionConfig:
    data_path: Path | None = None
    hf_repo_id: str | None = None
    hf_filename: str | None = None
    hf_repo_type: str = "dataset"
    hf_cache_dir: Path | None = None
    hf_local_files_only: bool = False
    field_key: str = "field"
    layout: str = "auto"
    sample_idx: int = 0
    time_idx: int = 0
    query_point: tuple[float, float] = (0.26, 0.50)
    radius: float = 0.34
    sigma: float = 0.10
    max_edges: int = 140
    obstacle_channel: int | None = None
    obstacle_threshold: float = 0.5
    obstacle_mode: str = "auto"
    x_channel: int | None = None
    y_channel: int | None = None
    velocity_channels: tuple[int, int] = (0, 1)
    zero_velocity_eps: float = 1e-6
    max_time_for_mask: int = 20


def _build_domain(config: AttentionVisualizationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, config.nx)
    y = np.linspace(0.0, 1.0, config.ny)
    xx, yy = np.meshgrid(x, y)
    mask = np.ones((config.ny, config.nx), dtype=bool)
    for cx, cy, r in config.holes:
        mask &= (xx - cx) ** 2 + (yy - cy) ** 2 >= r**2
    return xx, yy, mask


def _normalize_channel(idx: int, channels: int) -> int:
    if idx < 0:
        idx = channels + idx
    if idx < 0 or idx >= channels:
        raise ValueError(f"channel index {idx} out of bounds for channels={channels}")
    return idx


def _nearest_valid_index(
    xx: np.ndarray,
    yy: np.ndarray,
    mask: np.ndarray,
    query_point: tuple[float, float],
) -> tuple[int, int]:
    qx, qy = query_point
    dist2 = (xx - qx) ** 2 + (yy - qy) ** 2
    dist2 = np.where(mask, dist2, np.inf)
    flat = int(np.argmin(dist2))
    iy, ix = np.unravel_index(flat, mask.shape)
    return int(iy), int(ix)


def _geodesic_distance(
    mask: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    query_idx: tuple[int, int],
    radius: float,
) -> np.ndarray:
    ny, nx = mask.shape
    qy, qx = query_idx
    dist = np.full((ny, nx), np.inf, dtype=np.float64)
    dist[qy, qx] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, qy, qx)]

    while heap:
        d, y, x = heapq.heappop(heap)
        if d != dist[y, x]:
            continue
        if d > radius:
            continue

        if y > 0 and mask[y - 1, x]:
            nd = d + float(np.hypot(xx[y - 1, x] - xx[y, x], yy[y - 1, x] - yy[y, x]))
            if nd < dist[y - 1, x] and nd <= radius:
                dist[y - 1, x] = nd
                heapq.heappush(heap, (nd, y - 1, x))
        if y + 1 < ny and mask[y + 1, x]:
            nd = d + float(np.hypot(xx[y + 1, x] - xx[y, x], yy[y + 1, x] - yy[y, x]))
            if nd < dist[y + 1, x] and nd <= radius:
                dist[y + 1, x] = nd
                heapq.heappush(heap, (nd, y + 1, x))
        if x > 0 and mask[y, x - 1]:
            nd = d + float(np.hypot(xx[y, x - 1] - xx[y, x], yy[y, x - 1] - yy[y, x]))
            if nd < dist[y, x - 1] and nd <= radius:
                dist[y, x - 1] = nd
                heapq.heappush(heap, (nd, y, x - 1))
        if x + 1 < nx and mask[y, x + 1]:
            nd = d + float(np.hypot(xx[y, x + 1] - xx[y, x], yy[y, x + 1] - yy[y, x]))
            if nd < dist[y, x + 1] and nd <= radius:
                dist[y, x + 1] = nd
                heapq.heappush(heap, (nd, y, x + 1))

    return dist


def _attention_edges(
    geodesic_distance: np.ndarray,
    geodesic_local: np.ndarray,
    query_idx: tuple[int, int],
    sigma: float,
    max_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    qy, qx = query_idx
    coords = np.argwhere(geodesic_local)
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

    keep = (coords[:, 0] != qy) | (coords[:, 1] != qx)
    coords = coords[keep]
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

    d = geodesic_distance[coords[:, 0], coords[:, 1]]
    order = np.argsort(d)
    coords = coords[order]
    d = d[order]

    if max_edges > 0 and len(coords) > max_edges:
        coords = coords[:max_edges]
        d = d[:max_edges]

    scale = max(float(sigma), 1e-8)
    logits = -(d**2) / (2.0 * scale * scale)
    weights = np.exp(logits - logits.max())
    weights = weights / weights.sum()
    return coords, weights


def _draw_base(ax: Any, xx: np.ndarray, yy: np.ndarray, mask: np.ndarray) -> None:
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#f9f3ea", "#e1ecf7"])
    ax.pcolormesh(xx, yy, mask.astype(float), shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.contour(xx, yy, mask.astype(float), levels=[0.5], colors="#243b53", linewidths=1.2)
    ax.set_xlim(float(np.nanmin(xx)), float(np.nanmax(xx)))
    ax.set_ylim(float(np.nanmin(yy)), float(np.nanmax(yy)))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_attention_panels(
    xx: np.ndarray,
    yy: np.ndarray,
    mask: np.ndarray,
    query_point: tuple[float, float],
    radius: float,
    sigma: float,
    max_edges: int,
    title_suffix: str = "",
) -> "Figure":
    import matplotlib.pyplot as plt

    qy, qx = _nearest_valid_index(xx, yy, mask, query_point)
    xq = float(xx[qy, qx])
    yq = float(yy[qy, qx])

    geod = _geodesic_distance(mask, xx, yy, (qy, qx), radius)
    euclid = mask & (((xx - xq) ** 2 + (yy - yq) ** 2) <= radius**2)
    geodesic_local = mask & (geod <= radius)
    euclid_only = euclid & (~geodesic_local)
    edge_coords, alpha = _attention_edges(geod, geodesic_local, (qy, qx), sigma, max_edges)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.reshape(-1)

    _draw_base(ax0, xx, yy, mask)
    ax0.scatter(xx[mask], yy[mask], s=3, color="#4776b4", alpha=0.20, linewidths=0.0)
    ax0.scatter([xq], [yq], s=130, color="#c53030", marker="*", zorder=5)
    ax0.set_title("Domain Manifold Geometry", fontsize=12)

    _draw_base(ax1, xx, yy, mask)
    ax1.scatter(xx[euclid], yy[euclid], s=8, color="#f6ad55", alpha=0.8, linewidths=0.0)
    if np.any(euclid_only):
        ax1.scatter(xx[euclid_only], yy[euclid_only], s=12, color="#e53e3e", alpha=0.9, linewidths=0.0)
    ax1.add_patch(plt.Circle((xq, yq), radius, fill=False, linestyle="--", linewidth=2.0, color="#b83280"))
    ax1.scatter([xq], [yq], s=130, color="#c53030", marker="*", zorder=5)
    ax1.set_title("Euclidean Local Ball", fontsize=12)

    _draw_base(ax2, xx, yy, mask)
    ax2.scatter(xx[geodesic_local], yy[geodesic_local], s=8, color="#2b6cb0", alpha=0.85, linewidths=0.0)
    ax2.scatter([xq], [yq], s=130, color="#c53030", marker="*", zorder=5)
    ax2.set_title("Geodesic Local Support", fontsize=12)

    _draw_base(ax3, xx, yy, mask)
    if len(edge_coords) > 0:
        alpha_max = float(np.max(alpha))
        denom = alpha_max if alpha_max > 0 else 1.0
        for (iy, ix), w in zip(edge_coords, alpha):
            strength = float(w) / denom
            ax3.plot(
                [xq, float(xx[iy, ix])],
                [yq, float(yy[iy, ix])],
                color="#2f855a",
                alpha=0.10 + 0.85 * strength,
                linewidth=0.5 + 2.8 * strength,
            )
        xs = xx[edge_coords[:, 0], edge_coords[:, 1]]
        ys = yy[edge_coords[:, 0], edge_coords[:, 1]]
        ax3.scatter(xs, ys, c=alpha, cmap="viridis", s=14 + 170 * alpha, linewidths=0.0)
    ax3.scatter([xq], [yq], s=130, color="#c53030", marker="*", zorder=6)
    ax3.set_title("Sparse Geometric Attention Weights", fontsize=12)

    title = (
        "Sparse geometric attention emerges because the Navier-Stokes operator is a geodesically local integral "
        "operator defined on a manifold with holes."
    )
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    fig.suptitle(title, fontsize=13)
    return fig


def _resolve_data_path(config: DatasetAttentionConfig) -> Path:
    if config.data_path is not None:
        return Path(config.data_path)

    if config.hf_repo_id and config.hf_filename:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required for hf_repo_id/hf_filename") from exc

        file_path = hf_hub_download(
            repo_id=config.hf_repo_id,
            filename=config.hf_filename,
            repo_type=config.hf_repo_type,
            cache_dir=None if config.hf_cache_dir is None else str(config.hf_cache_dir),
            local_files_only=config.hf_local_files_only,
        )
        return Path(file_path)

    raise ValueError("provide data_path or hf_repo_id+hf_filename")


def _load_array(path: Path, field_key: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if suffix == ".npz":
        npz = np.load(path)
        if field_key not in npz:
            raise KeyError(f"field_key '{field_key}' not in {path}")
        return npz[field_key]
    raise ValueError(f"unsupported data file extension '{suffix}', expected .npy or .npz")


def _infer_layout(shape: tuple[int, ...], layout: str) -> str:
    key = layout.upper()
    valid = {"AUTO", "STHWC", "STCHW", "THWC", "TCHW", "HWC", "CHW", "HW"}
    if key not in valid:
        raise ValueError(f"unsupported layout '{layout}'")

    if key != "AUTO":
        return key

    if len(shape) == 5:
        if shape[-1] <= 32:
            return "STHWC"
        if shape[2] <= 32:
            return "STCHW"
    if len(shape) == 4:
        if shape[-1] <= 32:
            return "THWC"
        if shape[1] <= 32:
            return "TCHW"
    if len(shape) == 3:
        if shape[-1] <= 32:
            return "HWC"
        if shape[0] <= 32:
            return "CHW"
    if len(shape) == 2:
        return "HW"

    raise ValueError(f"could not infer layout from shape {shape}")


def _extract_frame(
    arr: np.ndarray,
    layout: str,
    sample_idx: int,
    time_idx: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    if layout == "STHWC":
        s = arr.shape[0]
        t = arr.shape[1]
        si = min(max(int(sample_idx), 0), s - 1)
        ti = min(max(int(time_idx), 0), t - 1)
        frame = np.asarray(arr[si, ti], dtype=np.float64)
        return frame, (si, ti)

    if layout == "STCHW":
        s = arr.shape[0]
        t = arr.shape[1]
        si = min(max(int(sample_idx), 0), s - 1)
        ti = min(max(int(time_idx), 0), t - 1)
        frame = np.asarray(arr[si, ti], dtype=np.float64).transpose(1, 2, 0)
        return frame, (si, ti)

    if layout == "THWC":
        t = arr.shape[0]
        ti = min(max(int(time_idx), 0), t - 1)
        frame = np.asarray(arr[ti], dtype=np.float64)
        return frame, (0, ti)

    if layout == "TCHW":
        t = arr.shape[0]
        ti = min(max(int(time_idx), 0), t - 1)
        frame = np.asarray(arr[ti], dtype=np.float64).transpose(1, 2, 0)
        return frame, (0, ti)

    if layout == "HWC":
        frame = np.asarray(arr, dtype=np.float64)
        return frame, (0, 0)

    if layout == "CHW":
        frame = np.asarray(arr, dtype=np.float64).transpose(1, 2, 0)
        return frame, (0, 0)

    if layout == "HW":
        frame = np.asarray(arr, dtype=np.float64)[..., None]
        return frame, (0, 0)

    raise ValueError(f"unsupported layout '{layout}'")


def _extract_velocity_sequence(
    arr: np.ndarray,
    layout: str,
    sample_idx: int,
    velocity_channels: tuple[int, int],
    max_time_for_mask: int,
) -> np.ndarray | None:
    if layout in {"HWC", "CHW", "HW"}:
        return None

    if layout == "STHWC":
        si = min(max(int(sample_idx), 0), arr.shape[0] - 1)
        t = min(arr.shape[1], max(max_time_for_mask, 1))
        c0 = _normalize_channel(velocity_channels[0], arr.shape[-1])
        c1 = _normalize_channel(velocity_channels[1], arr.shape[-1])
        return np.asarray(arr[si, :t, :, :, [c0, c1]], dtype=np.float64)

    if layout == "STCHW":
        si = min(max(int(sample_idx), 0), arr.shape[0] - 1)
        t = min(arr.shape[1], max(max_time_for_mask, 1))
        c0 = _normalize_channel(velocity_channels[0], arr.shape[2])
        c1 = _normalize_channel(velocity_channels[1], arr.shape[2])
        seq = np.asarray(arr[si, :t, [c0, c1], :, :], dtype=np.float64)
        return seq.transpose(0, 2, 3, 1)

    if layout == "THWC":
        t = min(arr.shape[0], max(max_time_for_mask, 1))
        c0 = _normalize_channel(velocity_channels[0], arr.shape[-1])
        c1 = _normalize_channel(velocity_channels[1], arr.shape[-1])
        return np.asarray(arr[:t, :, :, [c0, c1]], dtype=np.float64)

    if layout == "TCHW":
        t = min(arr.shape[0], max(max_time_for_mask, 1))
        c0 = _normalize_channel(velocity_channels[0], arr.shape[1])
        c1 = _normalize_channel(velocity_channels[1], arr.shape[1])
        seq = np.asarray(arr[:t, [c0, c1], :, :], dtype=np.float64)
        return seq.transpose(0, 2, 3, 1)

    return None


def _boundary_connected(mask: np.ndarray) -> np.ndarray:
    ny, nx = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []

    for x in range(nx):
        if mask[0, x]:
            stack.append((0, x))
        if mask[ny - 1, x]:
            stack.append((ny - 1, x))
    for y in range(ny):
        if mask[y, 0]:
            stack.append((y, 0))
        if mask[y, nx - 1]:
            stack.append((y, nx - 1))

    while stack:
        y, x = stack.pop()
        if visited[y, x] or not mask[y, x]:
            continue
        visited[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < ny:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < nx:
            stack.append((y, x + 1))

    return visited


def _obstacle_from_channel(values: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    mode_key = mode.lower()
    solid_hi = values >= threshold
    solid_lo = values < threshold

    if mode_key == "solid_is_one":
        return solid_hi
    if mode_key == "fluid_is_one":
        return solid_lo
    if mode_key != "auto":
        raise ValueError(f"unsupported obstacle_mode '{mode}'")

    interior_hi = solid_hi & (~_boundary_connected(solid_hi))
    interior_lo = solid_lo & (~_boundary_connected(solid_lo))
    if interior_hi.sum() > interior_lo.sum():
        return solid_hi
    if interior_lo.sum() > interior_hi.sum():
        return solid_lo

    ratio_hi = float(solid_hi.mean())
    ratio_lo = float(solid_lo.mean())
    target = 0.15
    if 0.001 < ratio_hi < 0.85 and 0.001 < ratio_lo < 0.85:
        return solid_hi if abs(ratio_hi - target) <= abs(ratio_lo - target) else solid_lo
    if 0.001 < ratio_hi < 0.85:
        return solid_hi
    if 0.001 < ratio_lo < 0.85:
        return solid_lo
    return np.zeros_like(values, dtype=bool)


def _frame_coordinates(
    frame: np.ndarray,
    x_channel: int | None,
    y_channel: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, c = frame.shape

    if x_channel is not None and y_channel is not None:
        xc = _normalize_channel(x_channel, c)
        yc = _normalize_channel(y_channel, c)
        xx = frame[..., xc]
        yy = frame[..., yc]
        valid = np.isfinite(xx) & np.isfinite(yy)
        if np.any(valid):
            return xx, yy, valid

    x = np.linspace(0.0, 1.0, w)
    y = np.linspace(0.0, 1.0, h)
    xx, yy = np.meshgrid(x, y)
    valid = np.ones((h, w), dtype=bool)
    return xx, yy, valid


def _frame_mask(
    frame: np.ndarray,
    arr: np.ndarray,
    layout: str,
    sample_idx: int,
    config: DatasetAttentionConfig,
) -> np.ndarray:
    h, w, c = frame.shape
    mask = np.isfinite(frame).all(axis=-1)

    if config.obstacle_channel is not None:
        oc = _normalize_channel(config.obstacle_channel, c)
        obstacle = _obstacle_from_channel(frame[..., oc], config.obstacle_mode, config.obstacle_threshold)
        if np.any(obstacle):
            inner = obstacle & (~_boundary_connected(obstacle))
            if np.any(inner):
                mask &= ~inner
            else:
                mask &= ~obstacle
        return mask

    if c >= 2:
        seq = _extract_velocity_sequence(
            arr,
            layout,
            sample_idx,
            config.velocity_channels,
            config.max_time_for_mask,
        )
        if seq is not None and seq.size > 0:
            speed = np.linalg.norm(seq, axis=-1)
            static = np.max(speed, axis=0) <= config.zero_velocity_eps
            if np.any(static):
                inner = static & (~_boundary_connected(static))
                if np.any(inner):
                    mask &= ~inner

    return mask


def create_sparse_geometric_attention_figure(
    config: AttentionVisualizationConfig | None = None,
) -> "Figure":
    config = AttentionVisualizationConfig() if config is None else config
    xx, yy, mask = _build_domain(config)
    return _plot_attention_panels(
        xx=xx,
        yy=yy,
        mask=mask,
        query_point=config.query_point,
        radius=config.radius,
        sigma=config.sigma,
        max_edges=config.max_edges,
        title_suffix="Synthetic manifold geometry",
    )


def create_dataset_sparse_geometric_attention_figure(
    config: DatasetAttentionConfig,
) -> "Figure":
    path = _resolve_data_path(config)
    arr = _load_array(path, config.field_key)
    layout = _infer_layout(arr.shape, config.layout)
    frame, (si, ti) = _extract_frame(arr, layout, config.sample_idx, config.time_idx)
    if frame.ndim != 3:
        raise ValueError(f"expected extracted frame rank=3, got {frame.ndim}")

    xx, yy, coord_valid = _frame_coordinates(frame, config.x_channel, config.y_channel)
    mask = _frame_mask(frame, arr, layout, si, config) & coord_valid
    if not np.any(mask):
        raise ValueError("no valid fluid nodes available for visualization")

    invalid = ~mask
    holes = invalid & (~_boundary_connected(invalid))
    hole_ratio = float(holes.mean())
    title_suffix = (
        f"Dataset={path.name} layout={layout} sample={si} time={ti} interior-hole-ratio={hole_ratio:.4f}"
    )

    return _plot_attention_panels(
        xx=xx,
        yy=yy,
        mask=mask,
        query_point=config.query_point,
        radius=config.radius,
        sigma=config.sigma,
        max_edges=config.max_edges,
        title_suffix=title_suffix,
    )


def save_sparse_geometric_attention(
    output_path: str | Path,
    config: AttentionVisualizationConfig | None = None,
    dpi: int = 220,
) -> Path:
    import matplotlib.pyplot as plt

    fig = create_sparse_geometric_attention_figure(config=config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_dataset_sparse_geometric_attention(
    output_path: str | Path,
    config: DatasetAttentionConfig,
    dpi: int = 220,
) -> Path:
    import matplotlib.pyplot as plt

    fig = create_dataset_sparse_geometric_attention_figure(config=config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_hole_values(raw: list[str]) -> tuple[tuple[float, float, float], ...]:
    holes: list[tuple[float, float, float]] = []
    for value in raw:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"invalid hole value '{value}', expected cx,cy,r")
        cx, cy, r = float(parts[0]), float(parts[1]), float(parts[2])
        holes.append((cx, cy, r))
    return tuple(holes)


def _parse_velocity_channels(raw: str) -> tuple[int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("velocity-channels must have format c0,c1")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/sparse_geometric_attention.png"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--mode", type=str, choices=["auto", "synthetic", "dataset"], default="auto")

    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--hole", action="append", default=["0.45,0.52,0.14", "0.63,0.45,0.11"])

    parser.add_argument("--query-x", type=float, default=0.26)
    parser.add_argument("--query-y", type=float, default=0.50)
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--max-edges", type=int, default=140)

    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--field-key", type=str, default="field")
    parser.add_argument("--layout", type=str, default="auto")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--time-idx", type=int, default=0)
    parser.add_argument("--obstacle-channel", type=int, default=None)
    parser.add_argument("--obstacle-threshold", type=float, default=0.5)
    parser.add_argument("--obstacle-mode", type=str, choices=["auto", "solid_is_one", "fluid_is_one"], default="auto")
    parser.add_argument("--x-channel", type=int, default=None)
    parser.add_argument("--y-channel", type=int, default=None)
    parser.add_argument("--velocity-channels", type=str, default="0,1")
    parser.add_argument("--zero-velocity-eps", type=float, default=1e-6)
    parser.add_argument("--max-time-for-mask", type=int, default=20)

    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-filename", type=str, default=None)
    parser.add_argument("--hf-repo-type", type=str, choices=["dataset", "model", "space"], default="dataset")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-local-files-only", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode
    if mode == "auto":
        if args.data is not None or (args.hf_repo_id and args.hf_filename):
            mode = "dataset"
        else:
            mode = "synthetic"

    if mode == "synthetic":
        config = AttentionVisualizationConfig(
            nx=args.nx,
            ny=args.ny,
            holes=_parse_hole_values(args.hole),
            query_point=(args.query_x, args.query_y),
            radius=args.radius,
            sigma=args.sigma,
            max_edges=args.max_edges,
        )
        output = save_sparse_geometric_attention(args.output, config=config, dpi=args.dpi)
        print(output)
        return

    data_config = DatasetAttentionConfig(
        data_path=args.data,
        hf_repo_id=args.hf_repo_id,
        hf_filename=args.hf_filename,
        hf_repo_type=args.hf_repo_type,
        hf_cache_dir=args.hf_cache_dir,
        hf_local_files_only=args.hf_local_files_only,
        field_key=args.field_key,
        layout=args.layout,
        sample_idx=args.sample_idx,
        time_idx=args.time_idx,
        query_point=(args.query_x, args.query_y),
        radius=args.radius,
        sigma=args.sigma,
        max_edges=args.max_edges,
        obstacle_channel=args.obstacle_channel,
        obstacle_threshold=args.obstacle_threshold,
        obstacle_mode=args.obstacle_mode,
        x_channel=args.x_channel,
        y_channel=args.y_channel,
        velocity_channels=_parse_velocity_channels(args.velocity_channels),
        zero_velocity_eps=args.zero_velocity_eps,
        max_time_for_mask=args.max_time_for_mask,
    )
    output = save_dataset_sparse_geometric_attention(args.output, config=data_config, dpi=args.dpi)
    print(output)


if __name__ == "__main__":
    main()
