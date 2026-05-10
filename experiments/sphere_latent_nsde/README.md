# Sphere Latent NSDE Activity Classification

JAX/Equinox/georax reimplementation of the HumanActivity latent SDE benchmark
from the PyTorch submodule at `experiments/sphere_latent_sde/`.

The latent state lives on `S^{z_dim-1}`. The model uses the georax manifold
implementation in `geometry.py`; frame coordinates match the original
lower-triangular `so(n)` convention used by the PyTorch code.

## Paper Results

| Paper item | Code/result |
|---|---|
| Sphere latent SDE memory scaling | PyTorch submodule memory sweep and `experiments/sphere_latent_sde/figures/memory.pdf` |
| HumanActivity accuracy at compute parity | `experiments/sphere_latent_sde/results/compute_parity.csv` |
| Sphere memory raw data | submodule memory-sweep CSVs |

This JAX version is the parity/reimplementation path used for georax solver
experimentation; the committed paper memory figures are produced by the
PyTorch submodule.

## Setup

```bash
uv pip install -e ".[sphere-jax]"
```

## Data

By default, configs look for the UCI Human Activity data at:

```text
experiments/sphere_latent_sde/data_dir
```

That directory is ignored by git. The loader first tries the processed Torch
artifact `PersonActivity/processed/data.pt` if `torch` is importable; otherwise
it rebuilds arrays from `PersonActivity/raw/ConfLongDemo_JSI.txt`.

To download and preprocess the UCI Human Activity file with the PyTorch
provider:

```bash
cd experiments/sphere_latent_sde
python -c "from data.activity_provider import HumanActivityProvider; HumanActivityProvider('data_dir', download=True)"
```

This fetches `ConfLongDemo_JSI.txt` from the UCI Machine Learning Repository and
writes both the raw file and the processed `data.pt` artifact under
`data_dir/PersonActivity/`. For a JAX-only setup, it is enough to place the raw
file at
`experiments/sphere_latent_sde/data_dir/PersonActivity/raw/ConfLongDemo_JSI.txt`;
the raw URL is
`https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt`.

## Checks And Runs

```bash
# Compare preprocessing against the original PyTorch repo layout
uv run python experiments/sphere_latent_nsde/scripts/check_original_parity.py

# One-batch CPU smoke run
uv run python -m experiments.sphere_latent_nsde.train_activity \
  experiments/sphere_latent_nsde/configs/activity/smoke.toml

# Full solver/seed sweep
uv run python -m experiments.sphere_latent_nsde.train_activity \
  experiments/sphere_latent_nsde/configs/activity/sweep.toml

# One sweep entry
uv run python -m experiments.sphere_latent_nsde.train_activity \
  experiments/sphere_latent_nsde/configs/activity/sweep.toml --index 0
```

The original output grid has 228 points. For faster JAX development runs, set
`num_timepoints = 64` or `32`; reconstruction and classification time IDs are
remapped onto that grid.

`nfe_budget` counts forward drift evaluations. Geometric Euler costs 1 FE per
step, CG2 costs 2 FEs per step, and CFEES25 costs 3 FEs per step. If
`nfe_budget` is omitted, the config chooses the largest common multiple of 6 not
exceeding `num_timepoints - 1`.

Each run writes to
`experiments/sphere_latent_nsde/results/activity__<solver>_<adjoint>__seed<N>__<timestamp>/`
with `config.toml`, `nsde.eqx`, `history.json`, and `metrics.json`.
