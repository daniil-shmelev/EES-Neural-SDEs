# Sphere Latent NSDE Activity Classification

JAX/Equinox reimplementation of the HumanActivity experiment from `/home/luke/EES-LatentSDEonHS/activity_classification.py`.

The latent state lives on `S^{z_dim-1}` and uses Georax's public batched `Sphere`. Its generator coordinates match the original PyTorch lower-triangular `so(n)` convention. All four methods now use `diffeqsolve`, the library's charts and the selected Diffrax adjoint. The former custom CF-EES adjoint, Euler stepping loop and SRKMK wrapper have been removed. The parity script checks the action using the library's high-order chart.

Existing results retain their original implementation and Brownian-driver provenance. New runs use Georax `bf5c3554bfa8199427d4a3200fa58b69b657aa98`, including CF-EES(2,5;1/10), and must be measured again before replacing manuscript results. The default sweep replaces Itô Geometric Euler with Stratonovich `GeometricEulerHeun()` on `Sphere`, using the `geometric_euler_heun_direct` and `geometric_euler_heun_checkpoint_full` methods. Old Euler results are not Euler–Heun measurements.

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

Fixed-step SDE solves are normalised by forward drift evaluations: `nfe_budget` counts drift evaluations, Geometric Euler–Heun costs 1 per solver step, CG2 and SRKMK-GeneralShARK cost 2 per step, and CFEES25 costs 3 per step. Euler–Heun also evaluates diffusion twice per step; matching drift evaluations does not match total field evaluations or runtime. The metrics separately record forward drift and diffusion evaluation counts. If `nfe_budget` is omitted, the largest common multiple of 6 not exceeding `num_timepoints - 1` is used, and the latent path is interpolated back to the configured output grid for reconstruction and classification losses.

## Adaptive CF-EES

The default sweep includes `cfees25_adaptive_reversible` as well as the fixed-step methods. To run only this entry with the supplied sweep:

```bash
./run sphere --index 6
```

A single-run TOML can instead set `method = "cfees25_adaptive_reversible"`, or set `solver = "cfees25"`, `adjoint = "reversible"` and `adaptive = true`. The adaptive settings are `adaptive_atol = 1e-3`, `adaptive_dtmin = 1e-4` and `adaptive_max_steps = 4096`. Euler–Heun remains fixed-step. Tolerances can be varied in separate configurations for an accuracy–time comparison.

The controller uses CF-EES's final-increment embedded estimate. Its error measure is the largest chordal endpoint difference over all examples and Monte Carlo samples; it is invariant under rotations and does not dilute one difficult trajectory into a batch average. One PI controller governs the shared batch mesh, so batch size is part of the adaptive protocol. Its initial and maximum step are `0.99 / (nfe_budget / 3)`. The loss quadrature grid and output interpolation are the same as fixed-step CF-EES at that reference budget; accepted integration times are free to vary between them. Thus `nfe_budget` sets a reference grid and maximum step, not an adaptive evaluation budget.

The Brownian tree has tolerance `adaptive_dtmin / 10` and is retained through rejected attempts. Sphere rotation fields need not commute, so `ClipStepSizeController` stores rejected endpoints and forces later steps to visit them, following [Diffrax's adaptive SDE guidance](https://docs.kidger.site/diffrax/api/stepsize_controller/). Reversible gradients use the ordinary `ReversibleAdjoint` and reconstruct on the accepted mesh. The minimum-step policy matches Kuramoto: `force_dtmin=True` permits accepting a step at the minimum size even if the estimated error exceeds tolerance. The maximum-attempt limit still fails the solve when exceeded.

`history.json` records mean attempted, accepted and rejected steps and forward drift/diffusion evaluations for each training and evaluation epoch, plus per-epoch and elapsed wall time. Counts include rejected attempts and refer to one batched solve on the shared mesh. `metrics.json` records `adaptive = true`, the controller settings and measured `test_solver_work`; fixed-budget fields such as `actual_forward_nfe` and `solve_n_steps` are `null` for adaptive runs. `reference_nfe_budget` retains the output/quadrature-grid setting. `train_time_s` includes compilation, training and evaluation. Compare held-out accuracy against measured time rather than treating adaptive runs as fixed-NFE results.

Verify adaptive rejection handling, accepted-mesh replay and reversible gradients against checkpointed gradients:

```bash
uv run --extra sphere --extra dev python -m pytest experiments/sphere_latent_nsde/test_adaptive.py -q
```

Each run writes a timestamped directory under `experiments/sphere_latent_nsde/results/runs/<timestamp>/activity__<method>__seed<N>__<timestamp>/` containing `config.toml`, `nsde.eqx`, `history.json`, and `metrics.json`.

Training selects the best checkpoint by validation accuracy, then evaluates the test split once at the end. Validation and test metrics are weighted by valid example counts, including a partial final batch; solver-work counts remain averages per batched solve. That final test accuracy is recorded as `test_acc_at_best_val_pct` in `metrics.json`.
