from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from .data import PreGenNavierStokes2DDataset
from .graph import SparseGeometryGraph, knn_graph, radius_graph
from .losses import (
    divergence_penalty_2d,
    enstrophy_error_2d,
    field_mse_loss,
    kinetic_energy_consistency_loss,
)
from .model import GOLAOperator


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    radius: float | None = 0.02
    k_neighbors: int | None = None
    max_neighbors: int | None = 64
    graph_chunk_size: int = 1024
    lambda_div: float = 0.0
    lambda_energy: float = 0.0
    lambda_enstrophy: float = 0.0
    device: str = "cpu"
    log_every: int = 50


def build_graph(
    x: torch.Tensor,
    config: TrainConfig,
) -> SparseGeometryGraph:
    if config.radius is not None and config.k_neighbors is not None:
        raise ValueError("set either radius or k_neighbors, not both")
    if config.radius is None and config.k_neighbors is None:
        raise ValueError("set one of radius or k_neighbors")

    if config.radius is not None:
        return radius_graph(
            x,
            radius=config.radius,
            max_neighbors=config.max_neighbors,
            chunk_size=config.graph_chunk_size,
        )

    return knn_graph(
        x,
        k=config.k_neighbors or 16,
        chunk_size=config.graph_chunk_size,
    )


def train_operator(
    model: GOLAOperator,
    dataset: PreGenNavierStokes2DDataset,
    config: TrainConfig,
) -> list[dict[str, float]]:
    device = torch.device(config.device)
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    x = dataset.x.to(device)
    graph = build_graph(x, config)
    edge_index = graph.edge_index.to(device)
    rel_pos = graph.rel_pos.to(device)
    distance = graph.distance.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        running = {
            "loss": 0.0,
            "field": 0.0,
            "div": 0.0,
            "energy": 0.0,
            "enstrophy": 0.0,
        }

        for step, batch in enumerate(loader, start=1):
            u_t = batch["u_t"].to(device)
            u_t_next = batch["u_t_next"].to(device)
            node_weight = batch["node_weight"].to(device)

            batch_loss = torch.zeros((), device=device)
            field_acc = torch.zeros((), device=device)
            div_acc = torch.zeros((), device=device)
            energy_acc = torch.zeros((), device=device)
            enstrophy_acc = torch.zeros((), device=device)

            for b in range(u_t.shape[0]):
                pred = model(
                    u_t[b],
                    edge_index=edge_index,
                    rel_pos=rel_pos,
                    distance=distance,
                    node_weight=node_weight[b],
                )

                field = field_mse_loss(pred, u_t_next[b])
                div = torch.zeros((), device=device)
                energy = torch.zeros((), device=device)
                enstrophy = torch.zeros((), device=device)

                sample_loss = field
                if config.lambda_div:
                    div = divergence_penalty_2d(pred, (dataset.height, dataset.width))
                    sample_loss = sample_loss + config.lambda_div * div
                if config.lambda_energy:
                    energy = kinetic_energy_consistency_loss(pred, u_t_next[b])
                    sample_loss = sample_loss + config.lambda_energy * energy
                if config.lambda_enstrophy:
                    enstrophy = enstrophy_error_2d(pred, u_t_next[b], (dataset.height, dataset.width))
                    sample_loss = sample_loss + config.lambda_enstrophy * enstrophy

                batch_loss = batch_loss + sample_loss
                field_acc = field_acc + field
                div_acc = div_acc + div
                energy_acc = energy_acc + energy
                enstrophy_acc = enstrophy_acc + enstrophy

            batch_loss = batch_loss / u_t.shape[0]
            field_acc = field_acc / u_t.shape[0]
            div_acc = div_acc / u_t.shape[0]
            energy_acc = energy_acc / u_t.shape[0]
            enstrophy_acc = enstrophy_acc / u_t.shape[0]

            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()

            running["loss"] += float(batch_loss.item())
            running["field"] += float(field_acc.item())
            running["div"] += float(div_acc.item())
            running["energy"] += float(energy_acc.item())
            running["enstrophy"] += float(enstrophy_acc.item())

            if step % config.log_every == 0:
                print(
                    f"[epoch {epoch + 1:03d} | step {step:04d}] "
                    f"loss={batch_loss.item():.6f} field={field_acc.item():.6f}"
                )

        denom = len(loader)
        epoch_stats = {k: v / denom for k, v in running.items()}
        history.append(epoch_stats)
        print(
            f"[epoch {epoch + 1:03d}] "
            f"loss={epoch_stats['loss']:.6f} field={epoch_stats['field']:.6f} "
            f"div={epoch_stats['div']:.6f} energy={epoch_stats['energy']:.6f} "
            f"enstrophy={epoch_stats['enstrophy']:.6f}"
        )

    return history
