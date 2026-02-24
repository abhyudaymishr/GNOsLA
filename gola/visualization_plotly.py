from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.graph_objects import Figure

from .visualization import (
    AttentionVisualizationConfig,
    DatasetAttentionConfig,
    _build_domain,
    _extract_frame,
    _frame_coordinates,
    _frame_mask,
    _infer_layout,
    _load_array,
    _nearest_valid_index,
    _resolve_data_path,
)


@dataclass
class PlotlyKernelVisualizationConfig:
    mode: str = "auto"
    output_html: Path = Path("artifacts/navier_stokes_kernel_3d.html")
    nu: float = 0.01
    dt: float = 1.0
    query_x: float | None = None
    query_y: float | None = None
    random_query: bool = True
    seed: int = 0
    sdf_channel: int | None = None
    normalize_surfaces: bool = True
    geometry_scale: float = 1.0
    kernel_scale: float = 1.0
    show: bool = False
    synthetic: AttentionVisualizationConfig = field(default_factory=AttentionVisualizationConfig)
    dataset: DatasetAttentionConfig = field(default_factory=DatasetAttentionConfig)


def _normalize_channel(idx: int, channels: int) -> int:
    if idx < 0:
        idx = channels + idx
    if idx < 0 or idx >= channels:
        raise ValueError(f"channel index {idx} out of bounds for channels={channels}")
    return idx


def _choose_query(
    xx: np.ndarray,
    yy: np.ndarray,
    mask: np.ndarray,
    query_x: float | None,
    query_y: float | None,
    random_query: bool,
    seed: int,
) -> tuple[int, int]:
    if query_x is not None and query_y is not None:
        return _nearest_valid_index(xx, yy, mask, (float(query_x), float(query_y)))

    if random_query:
        points = np.argwhere(mask)
        if points.size == 0:
            raise ValueError("no valid fluid cells available for query selection")
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, len(points)))
        iy, ix = points[idx]
        return int(iy), int(ix)

    return _nearest_valid_index(xx, yy, mask, (float(np.nanmean(xx)), float(np.nanmean(yy))))


def _bfs_geodesic(mask: np.ndarray, qx: int, qy: int) -> torch.Tensor:
    h, w = mask.shape
    dist = torch.full((h, w), float("inf"), dtype=torch.float32)
    queue: deque[tuple[int, int]] = deque()
    dist[qy, qx] = 0.0
    queue.append((qx, qy))

    while queue:
        x, y = queue.popleft()
        base = float(dist[y, x].item())

        nx = x + 1
        if nx < w and mask[y, nx]:
            nd = base + 1.0
            if nd < float(dist[y, nx].item()):
                dist[y, nx] = nd
                queue.append((nx, y))

        nx = x - 1
        if nx >= 0 and mask[y, nx]:
            nd = base + 1.0
            if nd < float(dist[y, nx].item()):
                dist[y, nx] = nd
                queue.append((nx, y))

        ny = y + 1
        if ny < h and mask[ny, x]:
            nd = base + 1.0
            if nd < float(dist[ny, x].item()):
                dist[ny, x] = nd
                queue.append((x, ny))

        ny = y - 1
        if ny >= 0 and mask[ny, x]:
            nd = base + 1.0
            if nd < float(dist[ny, x].item()):
                dist[ny, x] = nd
                queue.append((x, ny))

    return dist


def _boundary_cells(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            v = mask[y, x]
            if y > 0 and mask[y - 1, x] != v:
                boundary[y, x] = True
            elif y + 1 < h and mask[y + 1, x] != v:
                boundary[y, x] = True
            elif x > 0 and mask[y, x - 1] != v:
                boundary[y, x] = True
            elif x + 1 < w and mask[y, x + 1] != v:
                boundary[y, x] = True
    return boundary


def _distance_to_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    queue: deque[tuple[int, int]] = deque()
    seeds = _boundary_cells(mask)
    seed_points = np.argwhere(seeds)
    for y, x in seed_points:
        dist[y, x] = 0.0
        queue.append((int(x), int(y)))

    if not queue:
        return np.zeros((h, w), dtype=np.float32)

    while queue:
        x, y = queue.popleft()
        base = dist[y, x]

        nx = x + 1
        if nx < w and dist[y, nx] > base + 1.0:
            dist[y, nx] = base + 1.0
            queue.append((nx, y))

        nx = x - 1
        if nx >= 0 and dist[y, nx] > base + 1.0:
            dist[y, nx] = base + 1.0
            queue.append((nx, y))

        ny = y + 1
        if ny < h and dist[ny, x] > base + 1.0:
            dist[ny, x] = base + 1.0
            queue.append((x, ny))

        ny = y - 1
        if ny >= 0 and dist[ny, x] > base + 1.0:
            dist[ny, x] = base + 1.0
            queue.append((x, ny))

    return dist


def _signed_distance_from_mask(mask: np.ndarray) -> np.ndarray:
    d = _distance_to_boundary(mask)
    signed = np.where(mask, d, -d)
    return signed.astype(np.float32)


def _extract_geometry_surface(
    frame: np.ndarray,
    mask: np.ndarray,
    sdf_channel: int | None,
) -> np.ndarray:
    h, w, c = frame.shape
    if sdf_channel is not None:
        ch = _normalize_channel(sdf_channel, c)
        return np.asarray(frame[..., ch], dtype=np.float32)
    if c == 1:
        return np.asarray(frame[..., 0], dtype=np.float32)
    return _signed_distance_from_mask(mask)


def _build_surfaces(
    xx: np.ndarray,
    yy: np.ndarray,
    sdf: np.ndarray,
    mask: np.ndarray,
    qx: int,
    qy: int,
    nu: float,
    dt: float,
    normalize_surfaces: bool,
    geometry_scale: float,
    kernel_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nu <= 0 or dt <= 0:
        raise ValueError("nu and dt must be > 0")

    h, w = mask.shape
    yy_idx, xx_idx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    euclid_dist = torch.sqrt((xx_idx - float(qx)) ** 2 + (yy_idx - float(qy)) ** 2)
    geo_dist = _bfs_geodesic(mask, qx=qx, qy=qy)

    geo_kernel = torch.exp(-((geo_dist**2) / (4.0 * nu * dt)))
    geo_kernel[~torch.from_numpy(mask)] = 0.0
    global_kernel = torch.exp(-((euclid_dist**2) / (4.0 * nu * dt)))

    z_geom = sdf.astype(np.float32)
    z_geo = geo_kernel.numpy().astype(np.float32)
    z_global = global_kernel.numpy().astype(np.float32)

    if normalize_surfaces:
        gmax = float(np.max(np.abs(z_geom)))
        if gmax > 0:
            z_geom = z_geom / gmax
        kmax = max(float(np.max(np.abs(z_geo))), float(np.max(np.abs(z_global))), 1e-12)
        z_geo = z_geo / kmax
        z_global = z_global / kmax

    z_geom = geometry_scale * z_geom
    z_geo = kernel_scale * z_geo
    z_global = kernel_scale * z_global
    return z_geom, z_geo, z_global


def _dataset_geometry(config: DatasetAttentionConfig, sdf_channel: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    path = _resolve_data_path(config)
    arr = _load_array(path, config.field_key)
    layout = _infer_layout(arr.shape, config.layout)
    frame, (si, ti) = _extract_frame(arr, layout, config.sample_idx, config.time_idx)
    if frame.ndim != 3:
        raise ValueError(f"expected extracted frame rank=3, got {frame.ndim}")
    xx, yy, coord_valid = _frame_coordinates(frame, config.x_channel, config.y_channel)
    mask = _frame_mask(frame, arr, layout, si, config) & coord_valid
    if not np.any(mask):
        raise ValueError("no valid fluid cells available for dataset visualization")
    sdf = _extract_geometry_surface(frame, mask, sdf_channel=sdf_channel)
    label = f"Dataset={path.name} layout={layout} sample={si} time={ti}"
    return xx, yy, sdf, mask, label


def _synthetic_geometry(config: AttentionVisualizationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    xx, yy, mask = _build_domain(config)
    sdf = _signed_distance_from_mask(mask)
    return xx, yy, sdf, mask, "Synthetic manifold geometry"


def create_plotly_kernel_figure(config: PlotlyKernelVisualizationConfig) -> Figure:
    mode = config.mode.lower()
    if mode == "auto":
        if config.dataset.data_path is not None or (config.dataset.hf_repo_id and config.dataset.hf_filename):
            mode = "dataset"
        else:
            mode = "synthetic"

    if mode == "dataset":
        xx, yy, sdf, mask, subtitle = _dataset_geometry(config.dataset, config.sdf_channel)
    elif mode == "synthetic":
        xx, yy, sdf, mask, subtitle = _synthetic_geometry(config.synthetic)
    else:
        raise ValueError(f"unsupported mode '{config.mode}'")

    qy, qx = _choose_query(
        xx=xx,
        yy=yy,
        mask=mask,
        query_x=config.query_x,
        query_y=config.query_y,
        random_query=config.random_query,
        seed=config.seed,
    )

    z_geom, z_geo, z_global = _build_surfaces(
        xx=xx,
        yy=yy,
        sdf=sdf,
        mask=mask,
        qx=qx,
        qy=qy,
        nu=config.nu,
        dt=config.dt,
        normalize_surfaces=config.normalize_surfaces,
        geometry_scale=config.geometry_scale,
        kernel_scale=config.kernel_scale,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=z_geom,
            colorscale="RdBu",
            opacity=0.58,
            name="Geometry (SDF)",
            showscale=False,
        )
    )
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=z_geo,
            colorscale="Viridis",
            opacity=0.92,
            name="True Geodesic Kernel",
            showscale=False,
        )
    )
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=z_global,
            colorscale="Plasma",
            opacity=0.46,
            name="Invalid Global Attention",
            showscale=False,
        )
    )

    z_marker = float(max(z_geom[qy, qx], z_geo[qy, qx], z_global[qy, qx]))
    fig.add_trace(
        go.Scatter3d(
            x=[float(xx[qy, qx])],
            y=[float(yy[qy, qx])],
            z=[z_marker],
            mode="markers",
            marker={"size": 6, "color": "black"},
            name="Query Point",
        )
    )

    title = "Navier-Stokes Operator Locality vs Global Attention"
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle} | q=({qx},{qy})</sup>",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Value",
        },
        width=1080,
        height=860,
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
    )
    return fig


def save_plotly_kernel_figure(config: PlotlyKernelVisualizationConfig) -> Path:
    fig = create_plotly_kernel_figure(config)
    output = Path(config.output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn", full_html=True)
    if config.show:
        fig.show()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["auto", "synthetic", "dataset"], default="auto")
    parser.add_argument("--output-html", type=Path, default=Path("artifacts/navier_stokes_kernel_3d.html"))
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--query-x", type=float, default=None)
    parser.add_argument("--query-y", type=float, default=None)
    parser.add_argument("--random-query", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf-channel", type=int, default=None)
    parser.add_argument("--no-normalize-surfaces", action="store_true")
    parser.add_argument("--geometry-scale", type=float, default=1.0)
    parser.add_argument("--kernel-scale", type=float, default=1.0)
    parser.add_argument("--show", action="store_true")

    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--hole", action="append", default=["0.45,0.52,0.14", "0.63,0.45,0.11"])

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


def _parse_hole_values(raw: list[str]) -> tuple[tuple[float, float, float], ...]:
    holes: list[tuple[float, float, float]] = []
    for value in raw:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"invalid hole value '{value}', expected cx,cy,r")
        holes.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return tuple(holes)


def _parse_velocity_channels(raw: str) -> tuple[int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("velocity-channels must have format c0,c1")
    return int(parts[0]), int(parts[1])


def _build_config(args: argparse.Namespace) -> PlotlyKernelVisualizationConfig:
    dataset = DatasetAttentionConfig(
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
        query_point=(args.query_x if args.query_x is not None else 0.26, args.query_y if args.query_y is not None else 0.50),
        radius=0.34,
        sigma=0.10,
        max_edges=140,
        obstacle_channel=args.obstacle_channel,
        obstacle_threshold=args.obstacle_threshold,
        obstacle_mode=args.obstacle_mode,
        x_channel=args.x_channel,
        y_channel=args.y_channel,
        velocity_channels=_parse_velocity_channels(args.velocity_channels),
        zero_velocity_eps=args.zero_velocity_eps,
        max_time_for_mask=args.max_time_for_mask,
    )
    synthetic = AttentionVisualizationConfig(
        nx=args.nx,
        ny=args.ny,
        holes=_parse_hole_values(args.hole),
        query_point=(args.query_x if args.query_x is not None else 0.26, args.query_y if args.query_y is not None else 0.50),
        radius=0.34,
        sigma=0.10,
        max_edges=140,
    )
    return PlotlyKernelVisualizationConfig(
        mode=args.mode,
        output_html=args.output_html,
        nu=args.nu,
        dt=args.dt,
        query_x=args.query_x,
        query_y=args.query_y,
        random_query=args.random_query or (args.query_x is None or args.query_y is None),
        seed=args.seed,
        sdf_channel=args.sdf_channel,
        normalize_surfaces=not args.no_normalize_surfaces,
        geometry_scale=args.geometry_scale,
        kernel_scale=args.kernel_scale,
        show=args.show,
        synthetic=synthetic,
        dataset=dataset,
    )


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    output = save_plotly_kernel_figure(config)
    print(output)


if __name__ == "__main__":
    main()
