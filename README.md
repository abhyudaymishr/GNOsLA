# GOLA: Geometric Neural Operator with Sparse Local Attention

This repository provides a formal implementation scaffold for learning one-step PDE time-advance operators:

$$
\mathcal{T}_{\Delta t}: u(\cdot, t) \mapsto u(\cdot, t + \Delta t)
$$

using geometry-induced sparse attention.

## 1) Dataset formalization

`PreGenNavierStokes2DDataset` expects a tensor/array:

- Shape: `[num_trajectories, time_steps, H, W, channels]`
- Navier-Stokes default channels: `(v_x, v_y, p)`

Each sample returns:

- `u_t` with shape `[N, C]`
- `u_t_next` with shape `[N, C]`
- `x` with shape `[N, 2]` (uniform grid coordinates)
- `node_weight` with shape `[N]` (quadrature weight)

where `N = H * W`.

## 2) Tokenization

Tokens are geometric, not sequential:

$$
\tau_i^t = (x_i, u_i^t)
$$

No positional encoding, no causal masks, no padding.

## 3) Geometry-induced graph

`gola.graph` builds a static geometry graph:

- `radius_graph(x, r)` for \(N(i) = \{j \mid \lVert x_i - x_j \rVert \le r\}\)
- `knn_graph(x, k)` for irregular meshes / adaptive neighborhoods

The graph stores:

- `edge_index = [dst=i, src=j]`
- `rel_pos = x_i - x_j`
- `distance = ||x_i - x_j||`

## 4) Sparse-attention kernel approximation

`GOLALayer` implements:

$$
u_i^{(l+1)} = u_i^{(l)} + \sum_{j \in N(i)} \alpha_{ij}^{(l)} W^{(l)}u_j^{(l)} w_j
$$

with

$$
\alpha_{ij}^{(l)} = \mathrm{softmax}_{j \in N(i)}\left(\phi_\theta(u_i^{(l)}, u_j^{(l)}, x_i-x_j, \|x_i-x_j\|)\right)
$$

This is a learned local quadrature rule for a nonlinear integral operator.

## 5) Time-stepping operator

`GOLAOperator` stacks `L` layers:

$$
\hat{u}_{t+\Delta t} = \mathcal{G}_\theta(u_t)
$$

and supports residual output prediction.

## 6) Losses

Implemented losses in `gola.losses`:

- `field_mse_loss` (primary operator learning objective)
- `divergence_penalty_2d`
- `kinetic_energy_consistency_loss`
- `enstrophy_error_2d`

## 7) Training

`train_operator` executes:

1. Build geometric graph from coordinates.
2. Run stacked GOLA layers.
3. Compute field + optional physics-aware penalties.
4. Backpropagate through \(\phi_\theta\) and \(W\).

Run with:

```bash
python scripts/train_gola.py \
  --data /path/to/navier_stokes.npz \
  --field-key field \
  --epochs 20 \
  --layers 4 \
  --hidden-dim 128 \
  --radius 0.02 \
  --max-neighbors 64 \
  --lambda-div 0.0 \
  --lambda-energy 0.0 \
  --lambda-enstrophy 0.0
```

Low-memory profile (<8 GB RAM target):

```bash
python scripts/train_gola.py \
  --hf-repo-id sage-lab/PreGen-NavierStokes-2D \
  --hf-filename Geometry_Axis/FPO_Geometry_Easy_NoObstacle.npy \
  --hf-repo-type dataset \
  --layout auto \
  --low-memory \
  --ram-budget-gb 8 \
  --epochs 10 \
  --device cpu
```

This profile enforces memory-safe defaults: `batch_size=1`, `num_workers=0`, chunked graph build, bounded neighbors, and optional model-size clamps.

Using Hugging Face Hub directly (your PreGen dataset file):

```python
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="sage-lab/PreGen-NavierStokes-2D",
    filename="Geometry_Axis/FPO_Geometry_Easy_NoObstacle.npy",
    repo_type="dataset",
)
```

or from CLI:

```bash
python scripts/train_gola.py \
  --hf-repo-id sage-lab/PreGen-NavierStokes-2D \
  --hf-filename Geometry_Axis/FPO_Geometry_Easy_NoObstacle.npy \
  --hf-repo-type dataset \
  --layout auto \
  --epochs 20 \
  --layers 4 \
  --hidden-dim 128 \
  --radius 0.02
```

## Notes

- Complexity is \(O(Nk)\) with \(k = |N(i)|\), versus dense attention \(O(N^2)\).
- `radius_graph` and `knn_graph` are chunked to keep memory bounded on larger meshes.
- For `.npy` datasets, mmap loading is enabled by default to avoid loading the full tensor into RAM (`--no-mmap` disables it).

## Visualization

Generate a figure that demonstrates:

`Sparse geometric attention emerges because the Navier-Stokes operator is a geodesically local integral operator defined on a manifold with holes.`

Dataset-backed visualization (recommended):

```bash
python -m gola.visualization \
  --mode dataset \
  --data /path/to/navier_stokes.npy \
  --layout auto \
  --output artifacts/sparse_geometric_attention.png
```

Dataset-backed via Hugging Face:

```bash
python -m gola.visualization \
  --mode dataset \
  --hf-repo-id sage-lab/PreGen-NavierStokes-2D \
  --hf-filename Geometry_Axis/FPO_Geometry_Easy_NoObstacle.npy \
  --hf-repo-type dataset \
  --output artifacts/sparse_geometric_attention.png
```

Synthetic fallback:

```bash
python -m gola.visualization --mode synthetic --output artifacts/sparse_geometric_attention.png
```

Interactive 3D Plotly visualization (rotate/zoom/inspect):

```bash
pip install plotly

python -m gola.visualization_plotly \
  --mode dataset \
  --data /path/to/navier_stokes.npy \
  --layout auto \
  --output-html artifacts/navier_stokes_kernel_3d.html
```

Interactive 3D via Hugging Face:

```bash
python -m gola.visualization_plotly \
  --mode dataset \
  --hf-repo-id sage-lab/PreGen-NavierStokes-2D \
  --hf-filename Geometry_Axis/FPO_Geometry_Easy_NoObstacle.npy \
  --hf-repo-type dataset \
  --output-html artifacts/navier_stokes_kernel_3d.html
```

## Google Colab

[Open in Colab](https://colab.research.google.com/github/abhyudaymishr/GNOsLA/blob/main/notebooks/gola_colab_demo.ipynb)

Colab notebook:

- `notebooks/gola_colab_demo.ipynb`

The first code cell installs the latest module directly from this repository and sets up plotting.
