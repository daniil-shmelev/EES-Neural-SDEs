# Sphere Latent NSDE Activity Classification

JAX/Equinox reimplementation of the HumanActivity experiment from
`/home/luke/EES-LatentSDEonHS/activity_classification.py`.

The latent state lives on `S^{z_dim-1}` and is integrated through a georax
`Manifold` implementation in `geometry.py`. Its frame coordinates match the
original PyTorch lower-triangular `so(n)` convention. Training uses cheap
Cayley/Taylor charts selected by the georax solver; the parity script uses the
exact exponential chart to reproduce the baseline homogeneous-space action.

## Data

The cyreal dataset reads the PersonActivity data from:

```bash
/home/luke/EES-LatentSDEonHS/data_dir
```

It first tries the processed Torch `data.pt` if `torch` is importable. In the
JAX-only environment it rebuilds the same arrays from the raw text file. The
parity script confirms that this raw NumPy preprocessing is exactly equal to
the original processed artifact. Split strategy defaults to `auto`: exact Torch
`random_split` when `torch` is importable, otherwise a deterministic NumPy split
with the same split sizes.

## Checks

Compare behavior against the original Torch repo:

```bash
uv run python experiments/sphere_latent_nsde/scripts/check_original_parity.py
```

Run a small end-to-end smoke train:

```bash
uv run python -m experiments.sphere_latent_nsde.train_activity \
  experiments/sphere_latent_nsde/configs/activity/smoke.toml
```

Run the full three-entry solver sweep:

```bash
uv run python -m experiments.sphere_latent_nsde.train_activity \
  experiments/sphere_latent_nsde/configs/activity/sweep.toml
```

Run one sweep entry:

```bash
uv run python -m experiments.sphere_latent_nsde.train_activity \
  experiments/sphere_latent_nsde/configs/activity/sweep.toml --index 0
```

The exact original output grid has 228 points. For faster JAX development runs,
set `num_timepoints = 64` or `num_timepoints = 32` in a TOML config; the
dataloader remaps reconstruction/classification time IDs onto that grid.

The SDE solve is NFE-normalised across solvers: `nfe_budget` counts forward
drift evaluations, geometric Euler costs 1 FE per solver step, CG2 costs 2 FEs
per solver step, and CFEES25 costs 3 FEs per solver step. If `nfe_budget` is
omitted, the largest common multiple of 6 not exceeding `num_timepoints - 1` is
used, and the latent path is interpolated back to the configured output grid for
reconstruction and classification losses.

Each run writes a timestamped directory under
`experiments/sphere_latent_nsde/results/activity__<solver>_<adjoint>__seed<N>__<timestamp>/`
containing `config.toml`, `nsde.eqx`, `history.json`, and `metrics.json`.

Training selects the best checkpoint by validation accuracy, then evaluates the
test split once at the end. That final test accuracy is recorded as
`test_acc_at_best_val_pct` in `metrics.json`.
