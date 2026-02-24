from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseGeometryGraph:
    edge_index: torch.Tensor
    rel_pos: torch.Tensor
    distance: torch.Tensor


def _filter_self_edges(dst: torch.Tensor, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    keep = dst != src
    return dst[keep], src[keep]


def radius_graph(
    x: torch.Tensor,
    radius: float,
    *,
    max_neighbors: int | None = None,
    include_self: bool = False,
    chunk_size: int = 1024,
) -> SparseGeometryGraph:
    if x.ndim != 2:
        raise ValueError(f"x must have shape [N, dim], got {tuple(x.shape)}")
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    n = x.shape[0]
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        queries = x[start:stop]
        dist = torch.cdist(queries, x, p=2)

        if max_neighbors is not None:
            k = min(max_neighbors + (1 if include_self else 0), n)
            dist_k, idx_k = torch.topk(dist, k=k, dim=1, largest=False)
            keep = dist_k <= radius
            local_dst = torch.arange(start, stop, device=x.device).unsqueeze(1).expand_as(idx_k)
            dst = local_dst[keep]
            src = idx_k[keep]
        else:
            keep = dist <= radius
            local_i, src = keep.nonzero(as_tuple=True)
            dst = local_i + start

        if not include_self:
            dst, src = _filter_self_edges(dst, src)

        if dst.numel() == 0:
            continue

        dst_parts.append(dst)
        src_parts.append(src)

    if not dst_parts:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        rel_pos = torch.empty((0, x.shape[1]), dtype=x.dtype, device=x.device)
        distance = torch.empty((0, 1), dtype=x.dtype, device=x.device)
        return SparseGeometryGraph(edge_index=edge_index, rel_pos=rel_pos, distance=distance)

    dst = torch.cat(dst_parts, dim=0)
    src = torch.cat(src_parts, dim=0)

    order = torch.argsort(dst * n + src)
    dst = dst[order]
    src = src[order]

    rel_pos = x[dst] - x[src]
    distance = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    edge_index = torch.stack([dst, src], dim=0)
    return SparseGeometryGraph(edge_index=edge_index, rel_pos=rel_pos, distance=distance)


def knn_graph(
    x: torch.Tensor,
    k: int,
    *,
    include_self: bool = False,
    chunk_size: int = 1024,
) -> SparseGeometryGraph:
    if x.ndim != 2:
        raise ValueError(f"x must have shape [N, dim], got {tuple(x.shape)}")
    if k <= 0:
        raise ValueError("k must be > 0")

    n = x.shape[0]
    k_eff = min(k + (1 if include_self else 0), n)

    dst_parts: list[torch.Tensor] = []
    src_parts: list[torch.Tensor] = []

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        queries = x[start:stop]
        dist = torch.cdist(queries, x, p=2)
        _, idx_k = torch.topk(dist, k=k_eff, dim=1, largest=False)

        local_dst = torch.arange(start, stop, device=x.device).unsqueeze(1).expand_as(idx_k)
        dst = local_dst.reshape(-1)
        src = idx_k.reshape(-1)

        if not include_self:
            dst, src = _filter_self_edges(dst, src)

        dst_parts.append(dst)
        src_parts.append(src)

    dst = torch.cat(dst_parts, dim=0)
    src = torch.cat(src_parts, dim=0)
    order = torch.argsort(dst * n + src)
    dst = dst[order]
    src = src[order]

    rel_pos = x[dst] - x[src]
    distance = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    edge_index = torch.stack([dst, src], dim=0)
    return SparseGeometryGraph(edge_index=edge_index, rel_pos=rel_pos, distance=distance)
