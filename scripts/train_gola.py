from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Allow running as `python scripts/train_gola.py` from repository root without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gola.data import PreGenNavierStokes2DDataset
from gola.model import GOLAOperator
from gola.train import TrainConfig, train_operator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GOLA on pre-generated Navier-Stokes 2D data.")
    parser.add_argument("--data", type=Path, default=None, help="Path to local .npz/.npy data file")
    parser.add_argument("--field-key", type=str, default="field")
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=["auto", "STHWC", "STCHW", "THWC", "TCHW"],
        help="Axis layout for .npy/.npz field tensor",
    )
    parser.add_argument("--hf-repo-id", type=str, default=None, help="Hugging Face dataset repo_id")
    parser.add_argument("--hf-filename", type=str, default=None, help="File path inside Hugging Face dataset repo")
    parser.add_argument("--hf-repo-type", type=str, default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--no-mmap", action="store_true", help="Disable mmap for .npy local files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--k-neighbors", type=int, default=None)
    parser.add_argument("--max-neighbors", type=int, default=64)
    parser.add_argument("--graph-chunk-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lambda-div", type=float, default=0.0)
    parser.add_argument("--lambda-energy", type=float, default=0.0)
    parser.add_argument("--lambda-enstrophy", type=float, default=0.0)
    parser.add_argument("--low-memory", action="store_true", help="Enable <8GB RAM training profile")
    parser.add_argument("--ram-budget-gb", type=float, default=8.0)
    parser.add_argument("--lowmem-hidden-dim", type=int, default=96)
    parser.add_argument("--lowmem-layers", type=int, default=3)
    parser.add_argument("--lowmem-max-neighbors", type=int, default=24)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--delta-t-steps", type=int, default=1)
    return parser.parse_args()


def resolve_data_path(args: argparse.Namespace) -> Path:
    if args.data is not None:
        return args.data

    if args.hf_repo_id and args.hf_filename:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for --hf-repo-id/--hf-filename. "
                "Install with: pip install huggingface_hub"
            ) from exc

        file_path = hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=args.hf_filename,
            repo_type=args.hf_repo_type,
            cache_dir=None if args.hf_cache_dir is None else str(args.hf_cache_dir),
            local_files_only=args.hf_local_files_only,
        )
        return Path(file_path)

    raise ValueError("provide --data OR both --hf-repo-id and --hf-filename")


def apply_cli_low_memory_overrides(args: argparse.Namespace) -> None:
    if not args.low_memory:
        return

    args.batch_size = 1
    args.num_workers = 0
    args.graph_chunk_size = min(max(int(args.graph_chunk_size), 64), 256)

    if args.hidden_dim > args.lowmem_hidden_dim:
        print(f"[low-memory] hidden_dim clamped {args.hidden_dim} -> {args.lowmem_hidden_dim}")
        args.hidden_dim = args.lowmem_hidden_dim
    if args.layers > args.lowmem_layers:
        print(f"[low-memory] layers clamped {args.layers} -> {args.lowmem_layers}")
        args.layers = args.lowmem_layers

    if args.k_neighbors is not None:
        if args.k_neighbors > args.lowmem_max_neighbors:
            print(f"[low-memory] k-neighbors clamped {args.k_neighbors} -> {args.lowmem_max_neighbors}")
            args.k_neighbors = args.lowmem_max_neighbors
        args.radius = None
    else:
        if args.max_neighbors > args.lowmem_max_neighbors:
            print(f"[low-memory] max-neighbors clamped {args.max_neighbors} -> {args.lowmem_max_neighbors}")
            args.max_neighbors = args.lowmem_max_neighbors


def resolve_graph_parameters(args: argparse.Namespace) -> tuple[float | None, int | None]:
    if args.k_neighbors is not None:
        if args.radius is not None:
            print("[graph] both --radius and --k-neighbors provided; using k-neighbors and disabling radius.")
        return None, args.k_neighbors
    return args.radius, None


def main() -> None:
    args = parse_args()
    apply_cli_low_memory_overrides(args)
    data_path = resolve_data_path(args)

    dataset = PreGenNavierStokes2DDataset.from_file(
        data_path,
        field_key=args.field_key,
        delta_t_steps=args.delta_t_steps,
        layout=args.layout,
        mmap=not args.no_mmap,
    )

    radius, k_neighbors = resolve_graph_parameters(args)

    model = GOLAOperator(
        in_channels=dataset.channels,
        out_channels=dataset.channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        spatial_dim=dataset.x.shape[-1],
        residual_output=True,
    )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        radius=radius,
        k_neighbors=k_neighbors,
        max_neighbors=args.max_neighbors,
        graph_chunk_size=args.graph_chunk_size,
        low_memory_mode=args.low_memory,
        ram_budget_gb=args.ram_budget_gb,
        lambda_div=args.lambda_div,
        lambda_energy=args.lambda_energy,
        lambda_enstrophy=args.lambda_enstrophy,
        device=args.device,
    )

    train_operator(model, dataset, config)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
