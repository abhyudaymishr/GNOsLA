# GOLA Repository Documentation

## 1. Overview

`GOLA` implements a Geometric Neural Operator with Sparse Local Attention for one-step PDE operator learning:

\[
\mathcal{T}_{\Delta t}: u(\cdot, t) \mapsto u(\cdot, t + \Delta t)
\]

The project is focused on 2D Navier-Stokes style data, with geometry-aware sparse neighborhoods and physics-aware optional losses.

## 2. Repository Layout

```text
gola/
  __init__.py
  data.py
  graph.py
  layers.py
  losses.py
  model.py
  train.py
  visualization.py
  visualization_plotly.py
scripts/
  train_gola.py
notebooks/
  gola_colab_demo.ipynb
README.md
pyproject.toml
```

## 3. Core Pipeline

1. Load field trajectories (`.npy` or `.npz`) into `PreGenNavierStokes2DDataset`.
2. Flatten spatial grids to node tokens `(x_i, u_i)`.
3. Build static geometry graph (`radius_graph` or `knn_graph`).
4. Apply stacked `GOLALayer` updates with edge-softmax attention over local neighbors.
5. Predict `u_hat(t+Δt)` and optimize field MSE (+ optional physics penalties).

## 4. Module Reference

### 4.1 `gola/data.py`

- `PreGenNavierStokes2DDataset`
  - Supports layouts: `STHWC`, `STCHW`, `THWC`, `TCHW`, and `auto`.
  - Returns per-sample dict:
    - `u_t`: `[N, C]`
    - `u_t_next`: `[N, C]`
    - `x`: `[N, 2]`
    - `node_weight`: `[N]`
  - `.npy` can be loaded with mmap (`mmap=True` default).
  - Tracks resident data bytes via `resident_fields_bytes`.

- `make_uniform_grid(height, width)` creates normalized `[0,1] x [0,1]` coordinates.

### 4.2 `gola/graph.py`

- `radius_graph(x, radius, max_neighbors, chunk_size, include_self)`
- `knn_graph(x, k, chunk_size, include_self)`

Both return `SparseGeometryGraph`:

- `edge_index`: `[2, E]` (`dst`, `src`)
- `rel_pos`: `[E, spatial_dim]`
- `distance`: `[E, 1]`

Graph construction is chunked to control memory.

### 4.3 `gola/layers.py`

- `edge_softmax(scores, dst, num_nodes)` computes destination-grouped softmax.
- `GOLALayer`
  - Edge feature: `[h_i, h_j, x_i - x_j, ||x_i-x_j||]`
  - Score MLP -> sparse attention weights
  - Weighted message aggregation via `scatter_add_`
  - Residual node update `h + out`

### 4.4 `gola/model.py`

- `GOLAOperator`
  - `input_proj` -> `num_layers` of `GOLALayer` -> `output_proj`
  - Optional residual output (`pred = u_t + pred` when channels match)

### 4.5 `gola/losses.py`

- `field_mse_loss`
- `divergence_penalty_2d`
- `kinetic_energy_consistency_loss`
- `enstrophy_error_2d`

These can be composed during training via scalar lambdas.

### 4.6 `gola/train.py`

- `TrainConfig`: contains optimizer, graph, memory, and physics-loss settings.
- `build_graph`: dispatches radius/kNN graph building.
- `estimate_peak_ram_gb`: rough memory estimator.
- `apply_low_memory_profile`: enforces memory-safe settings when enabled.
- `train_operator`: end-to-end train loop.

#### Low-Memory Profile Behavior

When `low_memory_mode=True`:

- `batch_size=1`
- `num_workers=0`
- graph chunk clamped to `[64, 256]`
- neighbors capped to 24 (`k_neighbors` or `max_neighbors`)

### 4.7 `gola/visualization.py` (2D panels)

- Synthetic or dataset-backed 2D manifold/attention visualization.
- Composite export:
  - `save_sparse_geometric_attention(...)`
  - `save_dataset_sparse_geometric_attention(...)`

- Separate panel export APIs:
  - `save_sparse_geometric_attention_components(...)`
  - `save_dataset_sparse_geometric_attention_components(...)`

These save four files:

- `domain_manifold_geometry.png`
- `euclidean_local_ball.png`
- `geodesic_local_support.png`
- `sparse_geometric_attention_weights.png`

### 4.8 `gola/visualization_plotly.py` (interactive 3D)

- `PlotlyKernelVisualizationConfig`
- `create_plotly_kernel_figure(...)`
- `save_plotly_kernel_figure(...)`

3D surfaces rendered:

- Geometry (SDF-like surface)
- True geodesic kernel
- Global Euclidean kernel (invalid baseline)

### 4.9 `scripts/train_gola.py`

CLI training entry point with:

- local file or Hugging Face path resolution
- layout selection
- graph mode selection (radius or kNN)
- low-memory CLI controls (`--low-memory`, `--ram-budget-gb`, etc.)
- mmap control for local `.npy` (`--no-mmap`)

## 5. Installation

```bash
pip install -e .
```

or:

```bash
pip install git+https://github.com/abhyudaymishr/GNOsLA.git
```

## 6. Common Workflows

### 6.1 Train from local file

```bash
python scripts/train_gola.py \
  --data /path/to/data.npy \
  --layout auto \
  --epochs 20 \
  --radius 0.02
```

### 6.2 Train from Hugging Face

```bash
python scripts/train_gola.py \
  --hf-repo-id sage-lab/PreGen-NavierStokes-2D \
  --hf-filename Geometry_Axis/FPO_Geometry_Easy_NoObstacle.npy \
  --hf-repo-type dataset \
  --layout auto \
  --low-memory \
  --ram-budget-gb 8
```

### 6.3 2D visualization (single composite figure)

```bash
python -m gola.visualization \
  --mode dataset \
  --data /path/to/data.npy \
  --output artifacts/sparse_geometric_attention.png
```

### 6.4 2D visualization (four separate panels, Python API)

```python
from pathlib import Path
from gola.visualization import (
    DatasetAttentionConfig,
    save_dataset_sparse_geometric_attention_components,
)

paths = save_dataset_sparse_geometric_attention_components(
    output_dir=Path("artifacts/attention_components_dataset"),
    config=DatasetAttentionConfig(
        data_path=Path("/path/to/data.npy"),
        layout="auto",
        sample_idx=0,
        time_idx=0,
    ),
    dpi=220,
)
print(paths)
```

### 6.5 Interactive 3D Plotly

```bash
python -m gola.visualization_plotly \
  --mode dataset \
  --data /path/to/data.npy \
  --layout auto \
  --output-html artifacts/navier_stokes_kernel_3d.html
```

## 7. Colab Notebook

Notebook:

- `notebooks/gola_colab_demo.ipynb`

Includes:

- installation from GitHub
- low-memory training smoke run
- synthetic + dataset visualization
- real Hugging Face download flow (non-dry-run)
- interactive Plotly output

## 8. Performance and Memory Notes

- Graph build complexity: approximately `O(N * k)` for sparse neighborhoods.
- Dense attention baseline complexity: `O(N^2)`.
- `.npy` mmap reduces host RAM pressure for large files.
- Actual memory usage depends on:
  - `N`, `k`, `hidden_dim`, `layers`
  - graph chunk size
  - runtime backend and allocator overhead

## 9. Current Caveats

1. The file `FPO_Geometry_Easy_NoObstacle.npy` is a no-obstacle variant, so geodesic-vs-Euclidean blocking effects can be limited.
2. For stronger geometry-hole demonstrations, prefer obstacle variants in the same dataset family.
3. Plotly kernel visibility depends strongly on `nu` and grid scale; very small `nu` can collapse support to near-delta.

## 10. Artifacts

Generated outputs are usually stored under `artifacts/`:

- 2D composite PNGs
- 2D separate panel PNG folders
- 3D Plotly HTML files
