# Sphere Latent NSDE Activity Classification

JAX/Equinox reimplementation of the HumanActivity experiment from `/home/luke/EES-LatentSDEonHS/activity_classification.py`.

The latent state lives on `S^{z_dim-1}` and uses Georax's public batched `Sphere`. Its generator coordinates match the original PyTorch lower-triangular `so(n)` convention. All four methods now use `diffeqsolve`, the library's charts and the selected Diffrax adjoint. The former custom CF-EES adjoint, Euler stepping loop and SRKMK wrapper have been removed. The parity script checks the action using the library's high-order chart.

Existing results retain their original implementation and Brownian-driver provenance. New runs use the pinned Georax revision, including CF-EES(2,5;1/10), and must be measured again before replacing manuscript results.

## Data

The cyreal dataset reads the PersonActivity data from:

```bash
/home/luke/EES-LatentSDEonHS/data_dir
```

It first tries the processed Torch `data.pt` if `torch` is importable. In the JAX-only environment it rebuilds the same arrays from the raw text file. The parity script confirms that this raw NumPy preprocessing is exactly equal to the original processed artifact. Split strategy defaults to `auto`: exact Torch `random_split` when `torch` is importable, otherwise a deterministic NumPy split with the same split sizes.

## Checks

Compare behavior against the original Torch repo:

```bash
uv run --extra sphere python experiments/sphere_latent_nsde/scripts/check_original_parity.py
```

Check the migrated solver paths and gradients:

```bash
uv run --extra sphere --extra dev python -m pytest experiments/sphere_latent_nsde/test_georax_migration.py -q
```

Run the solver sweep from the repository root:

```bash
./run sphere
./run sphere --index 0
```

The first command runs all entries in [configs/activity/sweep.toml](configs/activity/sweep.toml); the second selects one. Use `--config PATH` to select another configuration and `--help` for options. The launcher installs the required uv dependencies.

The exact original output grid has 228 points. For faster JAX development runs, set `num_timepoints = 64` or `num_timepoints = 32` in a TOML config; the dataloader remaps reconstruction/classification time IDs onto that grid.

The SDE solve is NFE-normalised across solvers: `nfe_budget` counts forward drift evaluations, geometric Euler costs 1 FE per solver step, CG2 and SRKMK-GeneralShARK cost 2 FEs per solver step, and CFEES25 costs 3 FEs per step. If `nfe_budget` is omitted, the largest common multiple of 6 not exceeding `num_timepoints - 1` is used, and the latent path is interpolated back to the configured output grid for reconstruction and classification losses.

Each run writes a timestamped directory under `experiments/sphere_latent_nsde/results/runs/<timestamp>/activity__<solver>_<adjoint>__seed<N>__<timestamp>/` containing `config.toml`, `nsde.eqx`, `history.json`, and `metrics.json`.

Training selects the best checkpoint by validation accuracy, then evaluates the test split once at the end. That final test accuracy is recorded as `test_acc_at_best_val_pct` in `metrics.json`.
