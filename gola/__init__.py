from .data import PreGenNavierStokes2DDataset, make_uniform_grid
from .graph import SparseGeometryGraph, knn_graph, radius_graph
from .losses import (
    divergence_penalty_2d,
    enstrophy_error_2d,
    field_mse_loss,
    kinetic_energy_consistency_loss,
)
from .model import GOLAOperator
from .train import TrainConfig, train_operator

__all__ = [
    "AttentionVisualizationConfig",
    "DatasetAttentionConfig",
    "GOLAOperator",
    "PlotlyKernelVisualizationConfig",
    "PreGenNavierStokes2DDataset",
    "SparseGeometryGraph",
    "TrainConfig",
    "create_dataset_sparse_geometric_attention_figure",
    "create_plotly_kernel_figure",
    "create_sparse_geometric_attention_figure",
    "divergence_penalty_2d",
    "enstrophy_error_2d",
    "field_mse_loss",
    "kinetic_energy_consistency_loss",
    "knn_graph",
    "make_uniform_grid",
    "radius_graph",
    "save_dataset_sparse_geometric_attention",
    "save_plotly_kernel_figure",
    "save_sparse_geometric_attention",
    "train_operator",
]


def __getattr__(name: str):
    if name in {
        "AttentionVisualizationConfig",
        "DatasetAttentionConfig",
        "create_sparse_geometric_attention_figure",
        "create_dataset_sparse_geometric_attention_figure",
        "save_sparse_geometric_attention",
        "save_dataset_sparse_geometric_attention",
        "PlotlyKernelVisualizationConfig",
        "create_plotly_kernel_figure",
        "save_plotly_kernel_figure",
    }:
        from .visualization import (
            AttentionVisualizationConfig,
            DatasetAttentionConfig,
            create_dataset_sparse_geometric_attention_figure,
            create_sparse_geometric_attention_figure,
            save_dataset_sparse_geometric_attention,
            save_sparse_geometric_attention,
        )
        from .visualization_plotly import (
            PlotlyKernelVisualizationConfig,
            create_plotly_kernel_figure,
            save_plotly_kernel_figure,
        )

        exports = {
            "AttentionVisualizationConfig": AttentionVisualizationConfig,
            "DatasetAttentionConfig": DatasetAttentionConfig,
            "PlotlyKernelVisualizationConfig": PlotlyKernelVisualizationConfig,
            "create_dataset_sparse_geometric_attention_figure": create_dataset_sparse_geometric_attention_figure,
            "create_plotly_kernel_figure": create_plotly_kernel_figure,
            "create_sparse_geometric_attention_figure": create_sparse_geometric_attention_figure,
            "save_dataset_sparse_geometric_attention": save_dataset_sparse_geometric_attention,
            "save_plotly_kernel_figure": save_plotly_kernel_figure,
            "save_sparse_geometric_attention": save_sparse_geometric_attention,
        }
        return exports[name]
    raise AttributeError(name)
