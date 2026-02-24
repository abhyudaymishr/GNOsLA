from __future__ import annotations

from torch import nn
import torch

from .layers import GOLALayer


class GOLAOperator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        spatial_dim: int,
        *,
        out_channels: int | None = None,
        score_hidden_dim: int = 128,
        residual_output: bool = True,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_output = residual_output

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GOLALayer(
                    hidden_dim=hidden_dim,
                    spatial_dim=spatial_dim,
                    score_hidden_dim=score_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(
        self,
        u_t: torch.Tensor,
        edge_index: torch.Tensor,
        rel_pos: torch.Tensor,
        distance: torch.Tensor,
        *,
        node_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if u_t.ndim != 2:
            raise ValueError(f"u_t must have shape [N, C], got {tuple(u_t.shape)}")

        h = self.input_proj(u_t)
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, rel_pos=rel_pos, distance=distance, node_weight=node_weight)

        pred = self.output_proj(h)
        if self.residual_output and pred.shape[-1] == u_t.shape[-1]:
            pred = u_t + pred
        return pred
