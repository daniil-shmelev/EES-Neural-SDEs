# Sphere Latent NSDE Activity Classification

JAX/Equinox reimplementation of the HumanActivity experiment from `/home/luke/EES-LatentSDEonHS/activity_classification.py`.

The latent state lives on `S^{z_dim-1}` and uses Georax's public batched `Sphere`. Its generator coordinates match the original PyTorch lower-triangular `so(n)` convention. All four methods now use `diffeqsolve`, the library's charts and the selected Diffrax adjoint. The former custom CF-EES adjoint, Euler stepping loop and SRKMK wrapper have been removed. The parity script checks the action using the library's high-order chart.

Existing results retain their original implementation and Brownian-driver provenance. New runs use Georax `20b401c9ffc83dc449c0340f4bc7b1289761c2b1` and cosine EES, and must be measured again before replacing manuscript results. The default sweep replaces Itô Geometric Euler with Stratonovich `GeometricEulerHeun()` on `Sphere`, using direct and full adjoints. Old Euler results are not Euler–Heun measurements.

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

The first command runs all entries in [config.toml](config.toml); the second selects one. As in Kuramoto, `[base]` holds shared settings, `[sweep]` selects seeds, and `[integrators.<name>]` lists each integrator's adjoints and step settings. Adjoint names are `direct`, `full`, `recursive` and `reversible`; reversible reconstruction requires cosine EES. Change run parameters in that TOML; there are no CLI overrides for them. Use `--config PATH` to select another configuration and `--help` for options. The launcher installs the required uv dependencies. For example:

```toml
[integrators.geometric_euler_heun]
adjoints = ["direct", "full"]
n_steps = 30
```

Each listed adjoint runs with the same integrator settings. Stepping defaults to `["fixed"]`, and output method labels are generated automatically.

The exact original output grid has 228 points. For faster JAX development runs, set `num_timepoints = 64` or `num_timepoints = 32` in a TOML config; the dataloader remaps reconstruction/classification time IDs onto that grid.

Set `n_steps` under each integrator. The supplied fixed-step counts match 30 forward drift evaluations: 30 Euler–Heun steps, 15 CG2 or SRKMK-GeneralShARK steps, and 10 cosine EES steps. Euler–Heun also evaluates diffusion twice per step; matching drift evaluations does not match total field evaluations or runtime. The metrics derive the drift budget from `n_steps` and separately record forward drift and diffusion evaluation counts. The latent path is interpolated back to the configured output grid for reconstruction and classification losses.

The cosine integrator table sets `reversible_order = 5`, giving three stages per step. Any odd order at least three is supported; its stage count is `(reversible_order + 1) // 2`. Fixed steps use `embedded=False`; adaptive steps set it to `True`. Saved configurations and metrics record both settings, and method labels include the order (for example `cf_cosine_ees_p5_reversible`). Stage budgets exclude the companion, while adaptive work statistics include its additional source-level drift and diffusion calls per attempted step. JIT may share these calls with the first stage.

## Adaptive cosine EES

The default sweep includes `cf_cosine_ees_p5_adaptive_reversible` as well as the fixed-step methods. To run only this entry with the supplied sweep:

```bash
./run sphere --index 6
```

The `[integrators.cf_cosine_ees]` table selects `adjoints = ["reversible"]` and `stepping = ["fixed", "adaptive"]`. Its nested `[integrators.cf_cosine_ees.adaptive]` table sets `atol = 1e-2`, `dtmin = 1e-4` and `max_steps = 4096`. Adaptive runs inherit `n_steps = 10` from the parent table; adding `n_steps` to the nested table overrides it only for adaptive runs. Euler–Heun remains fixed-step. Tolerances can be varied in separate configurations for an accuracy–time comparison.

The controller uses cosine EES's optional Lie–Euler embedded estimate (`embedded=True`). Its error measure is the largest chordal endpoint difference over all examples and Monte Carlo samples; it is invariant under rotations and does not dilute one difficult trajectory into a batch average. One PI controller governs the shared batch mesh, so batch size is part of the adaptive protocol. Its initial and maximum step are `0.99 / n_steps`. The loss quadrature grid and output interpolation use the same reference step count as fixed-step cosine EES; accepted integration times are free to vary between them. For adaptive runs, `n_steps` therefore sets the reference grid and maximum step size; actual work is measured during the solve.

The Brownian tree has tolerance `adaptive_dtmin / 10` and is retained through rejected attempts. Sphere rotation fields need not commute, so `ClipStepSizeController` stores rejected endpoints and forces later steps to visit them, following [Diffrax's adaptive SDE guidance](https://docs.kidger.site/diffrax/api/stepsize_controller/). Reversible gradients use the ordinary `ReversibleAdjoint` and reconstruct on the accepted mesh. The minimum-step policy matches Kuramoto: `force_dtmin=True` permits accepting a step at the minimum size even if the estimated error exceeds tolerance. The maximum-attempt limit still fails the solve when exceeded.

`history.json` records mean attempted, accepted and rejected steps and forward drift/diffusion evaluations for each training and evaluation epoch, plus per-epoch and elapsed wall time. Counts include rejected attempts and refer to one batched solve on the shared mesh. `metrics.json` records `adaptive = true`, the controller settings and measured `test_solver_work`; fixed-budget fields such as `actual_forward_nfe` and `solve_n_steps` are `null` for adaptive runs. `reference_nfe_budget` retains the output/quadrature-grid setting. `train_time_s` includes compilation, training and evaluation. Compare held-out accuracy against measured time rather than treating adaptive runs as fixed-NFE results.

Verify adaptive rejection handling, accepted-mesh replay and reversible gradients against checkpointed gradients:

```bash
uv run --extra sphere --extra dev python -m pytest experiments/sphere_latent_nsde/test_adaptive.py -q
```

Every configuration runs in a fresh process and records `peak_gpu_memory_bytes` alongside `train_time_s` in `metrics.json`. Memory is the peak JAX GPU allocation from process startup through training and evaluation, including resident data, model, optimiser and temporary buffers. It is read from the allocator’s `peak_bytes_in_use`, so short-lived peaks are included. CPU runs or unsupported allocator statistics produce `null`. This uses the same definition as Kuramoto.

Each run writes a timestamped directory under `experiments/sphere_latent_nsde/results/runs/<timestamp>/activity__<method>__seed<N>__<timestamp>/` containing `config.toml`, `nsde.eqx`, `history.json`, and `metrics.json`.

Training selects the best checkpoint by validation accuracy, then evaluates the test split once at the end. Validation and test metrics are weighted by valid example counts, including a partial final batch; solver-work counts remain averages per batched solve. That final test accuracy is recorded as `test_acc_at_best_val_pct` in `metrics.json`.

## Visualising the task

Plot one sequence using a saved run's `config.toml` and `nsde.eqx`:

```bash
uv run --extra sphere python -m experiments.sphere_latent_nsde.visualize \
  /path/to/cosine-run \
  --output-dir experiments/sphere_latent_nsde/results/visualizations
```

The figure shows all four tags' observed x/y/z positions and reconstructed means, true and predicted activity bands, class probabilities, and sampled latent paths in a shared 3D PCA projection. Missing observations are masked. Predictions use the same output-time indices and averaging of logits as the evaluation code. Inference conditions on the complete observed sequence; this is activity classification within that sequence.

The default selects the first test window containing an activity transition, independently of model predictions, and draws eight posterior samples on CPU. Use `--sample-index N`, `--split train|val|test`, `--mc-samples N`, `--seed N`, or `--device gpu` to change these choices. The saved split strategy is reused; for `auto`, match the training environment's Torch availability to reproduce its split. Adaptive inference uses a single-example batch, so its shared adaptive mesh can differ from evaluation with larger batches.

Outputs are `activity_example.png`, `activity_example.pdf`, the plotted arrays in `activity_example.npz`, and selection/solver metadata in `activity_example.json`. Without `--output-dir`, these go into the run's `visualizations/` directory. The displayed accuracy describes only the selected sequence. For the default `z_dim = 16`, the latent paths lie on `S^{15}`; the 3D PCA view reports its explained variance and does not preserve spherical geometry.
