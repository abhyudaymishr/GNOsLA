from __future__ import annotations

import torch
from torch import nn


def edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int, eps: float = 1e-12) -> torch.Tensor:
    max_per_dst = torch.full((num_nodes,), -torch.inf, dtype=scores.dtype, device=scores.device)
    max_per_dst.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)

    shifted = scores - max_per_dst[dst]
    exp_scores = torch.exp(shifted)
    den = torch.zeros((num_nodes,), dtype=scores.dtype, device=scores.device)
    den.scatter_add_(0, dst, exp_scores)
    return exp_scores / (den[dst] + eps)


class GOLALayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        *,
        score_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = (2 * hidden_dim) + spatial_dim + 1
        self.score_mlp = nn.Sequential(
            nn.Linear(input_dim, score_hidden_dim),
            nn.SiLU(),
            nn.Linear(score_hidden_dim, score_hidden_dim),
            nn.SiLU(),
            nn.Linear(score_hidden_dim, 1),
        )
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        rel_pos: torch.Tensor,
        distance: torch.Tensor,
        node_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h.ndim != 2:
            raise ValueError(f"h must have shape [N, hidden_dim], got {tuple(h.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")

        num_nodes = h.shape[0]
        dst, src = edge_index[0], edge_index[1]

        edge_feat = torch.cat([h[dst], h[src], rel_pos, distance], dim=-1)
        scores = self.score_mlp(edge_feat).squeeze(-1)
        alpha = edge_softmax(scores, dst, num_nodes)

        msg = self.value_proj(h[src]) * alpha.unsqueeze(-1)
        if node_weight is not None:
            msg = msg * node_weight[src].unsqueeze(-1)

        out = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, h.shape[-1]), msg)
        return h + out
