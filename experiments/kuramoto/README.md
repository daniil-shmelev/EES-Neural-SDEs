# Kuramoto learning experiment

The active experiment trains one geometric neural SDE on the tangent torus with 1000 oscillators. Its drift enforces `dtheta/dt = omega` and learns a phase-dependent mean-field force plus scalar damping `2*sigmoid(damping_logit)` in `(0, 2)`, initialized at 1. This includes the cached dataset's true unit damping. The force uses a shared per-oscillator network with trigonometric moments and learned oscillator embeddings; it has no velocity inputs. The diffusion remains a learned neural field. The same architecture is used for CG2, CF-EES, Geometric Euler–Heun and SRKMK–GeneralShARK. This changes the model assumption from the earlier unconstrained drift, whose backward reconstruction amplified numerical errors. The cached dataset has 1500/300/300 training/validation/test trajectories, 100 observations and horizon 5. Training uses the multi-horizon energy score with four stochastic predictions per example.

## One configuration

[config.toml](config.toml) contains all training settings. `[base]` holds shared settings, `[sweep]` selects seeds, `[integrators.<name>]` lists each integrator's adjoints and step settings. Adjoint names are `direct`, `full`, `recursive` and `reversible`, subject to the combinations supported by each benchmark. The fixed-step training comparison uses these solver--adjoint combinations:

| Solver | Adjoint | Steps | Forward drift evaluations | Forward diffusion evaluations |
| --- | --- | ---: | ---: | ---: |
| CG2 | Full | 75 | 150 | 150 |
| CG2 | Recursive | 75 | 150 | 150 |
| CF-EES(2,5) | Reconstruction without checkpoints | 50 | 150 | 150 |
| Geometric Euler–Heun | Full | 150 | 150 | 300 |
| Geometric Euler–Heun | Recursive | 150 | 150 | 300 |
| SRKMK–GeneralShARK | Full | 75 | 150 | 225 |
| SRKMK–GeneralShARK | Recursive | 75 | 150 | 225 |

Geometric Euler–Heun uses Georax's `GeometricEulerHeun()` directly on `TangentTorus`, with one drift and two diffusion evaluations per step. It uses the same Stratonovich convention as CG2 and CF-EES, so no drift conversion is needed. Each method is trained independently from the same initialization, architecture and loss. Euler–Heun supports fixed steps and full or recursive checkpointing; it has no embedded error estimate or reversible adjoint. The defaults match drift evaluations, not total field evaluations or runtime. For accuracy–time comparisons, vary each method's step count and use measured wall-clock time.

SRKMK wraps Diffrax’s `GeneralShARK()` in Georax’s tangent-torus charts. Its Brownian tree supplies the required space-time integrals. It uses two drift and three diffusion evaluations per step, with the same full and recursive checkpoint options as the other fixed-step baselines.

Run commands from the repository root:

```bash
./run kuramoto
```

The training command expands every selected seed, integrator, stepping mode and adjoint. The supplied configuration uses seeds 0, 1 and 2. To run only Euler–Heun for seed 0, keep this selection in `config.toml` and remove the other integrator tables:

```toml
[sweep]
seed = [0]

[integrators.geometric_euler_heun]
adjoints = ["full", "recursive"]
n_steps = 150
```

Set solver step counts under their integrator table. `stepping` defaults to `["fixed"]`; CF-EES selects `["fixed", "adaptive"]`. Output method labels are generated automatically. Model dimensions, training epochs and batch size belong in `[base]`.

Each invocation gets a fresh directory under `results/unified/runs/`. Training prints progress and saves a log per method. `--config PATH` selects another TOML, and `--index N` selects one training sweep entry, in seed-then-method order. Every run records `peak_gpu_memory_bytes` alongside `train_walltime_s` in `metrics.json`; the comparison summary includes both. Memory is the peak JAX GPU allocation from process startup through training and evaluation, including resident data, model, optimiser and temporary buffers. It is captured before demonstration forecasts and solver-work diagnostics. Each configuration runs in a fresh process, and CPU runs or unsupported allocator statistics produce `null`.

Training, validation and test batches print progress and elapsed time; the first training and evaluation batches include JAX compilation. Evaluation computes loss and MAEs from the same forecast ensemble and counts each held-out trajectory once, masking the padded final batch and weighting batch averages by their valid example counts. Diffusion applies its diagonal entries directly instead of materialising a dense matrix, and loss calculations save only their four requested observation times. With `save_every_epoch = true`, epoch weights and the latest optimiser state are saved before evaluation, so an interruption during evaluation preserves that epoch's training weights. `--resume-from` accepts final checkpoints or the last fully evaluated epoch with its matching latest optimiser state. If evaluation was interrupted after a newer optimiser was saved, resume refuses to pair that optimiser with the older evaluated model.

## Adaptive CF-EES

The default sweep includes `cfees25_adaptive_reversible`, with the final-increment embedded estimator, dynamic step sizes and checkpoint-free reversible reconstruction gradients. To select just its seed-0 entry in the supplied TOML:

```bash
./run kuramoto --index 7
```

The `[integrators.cfees25.adaptive]` table sets `n_steps = 10`, `dtmin = 1e-4`, `atol = 3e-2` and `max_steps = 4096`. These override the parent integrator's settings only for adaptive runs. The tolerance controls the estimated RMS difference between the two endpoints, using wrapped phase differences and ordinary velocity differences. Diffrax's standard `PIDController`, configured as a PI controller with `force_dtmin=True`, adjusts the step size, capped at `T/n_steps` (0.5 for this configuration). At `dtmin`, steps are accepted even if the estimated error exceeds tolerance. The Brownian-tree tolerance is ten times smaller than `dtmin`. `max_steps` limits attempted steps per trajectory. All of these options are read from the TOML.

Batched adaptive reconstruction currently needs a Diffrax fix: the pinned fork reads a padded time after one batch member finishes its reverse loop. The tested fix is in `/home/luke/diffrax/.worktrees/adaptive-reversible`, on branch `codex/adaptive-reversible-vmap`; the launcher selects that worktree automatically whenever the selected TOML runs include adaptive CF-EES. The original pin remains unchanged. The verified SIAM result directory retains the exact patch and dependency provenance.

The Brownian tree is retained through rejection. The learned diffusion depends on the full state and need not commute, so the standard `ClipStepSizeController` wrapper stores rejected endpoints and forces subsequent steps to revisit them. CF-EES uses the ordinary `ReversibleAdjoint` and reconstructs states on the accepted mesh without trajectory checkpoints. The accepted-time buffer is separate from the reconstructed state. No inverse-correction loop or checkpointed gradient is substituted. The previous unconstrained drift remains available as `drift_kind = "indexed_mean_field"` for reading historical checkpoints and reproducing the failure; new runs use `second_order_mean_field`. The earlier checkpointed speed comparison does not establish reversible-training performance. A single-run diagnostic TOML can explicitly set `adjoint = "checkpoint_recursive"` and `adaptive = true`. Adaptive runs have variable evaluation counts; they are additional experiments to the fixed 150-forward-drift-evaluation comparison. `metrics.json` records accepted/rejected steps and forward generator evaluations on eight fixed validation trajectories using the best validation checkpoint.

## Verification

Tests compare reversible gradients with reference gradients, check adaptive mesh replay, and verify exact phase kinematics, velocity-independent forcing and bounded damping. Forward stability alone does not guarantee accurate backward reconstruction; gradient comparisons on learned fields are part of the numerical validation.

For short iterations, keep the production network and reduce the batch, number of forecast samples and data subset:

```bash
PYTHONPATH=/home/luke/diffrax/.worktrees/adaptive-reversible:$PWD \
  .venv/bin/python -m experiments.kuramoto.scripts.check_reversible_training \
  --output-dir /tmp/kuramoto-reversible-check --steps 600
```

This compares CG2 with checkpoints and CF-EES with ordinary reversible reconstruction from the same initialization, using two examples and two forecasts per update. `--modes ees` runs just the candidate. `--initial-checkpoint PATH` instead starts both methods at the same saved weights with fresh, identical optimisers; the checkpoint must match the configured architecture. The script records matched validation losses, timings and a final same-weights gradient comparison; reference gradients are used only in that diagnostic. Its default two-sample losses must not be compared numerically with the full four-sample results.

The existing simulator checks and same-noise Lyapunov diagnostic remain available. scripts/run_m1.py defaults match the cached production dataset; --smoke uses a small dataset for testing.

## Historical results and dependencies

Historical models, presets, pilot scripts and documentation are in `/home/luke/nsde_archive/experiments/kuramoto/legacy/`. The current SIAM learning comparison, manuscript datasets, historical memory sweep and simulator diagnostics remain here. NeurIPS fixed-step runtime records are now in `../../neurips/kuramoto/results/`; superseded trial runs were deleted as recorded in `../../docs/cleanup_20261027.json`. The archived memory curve used a global MLP with 1,204,792 parameters; the second-order mean-field training model has 582,251. These isolated-gradient measurements have a different scope from the new per-run training memory metric.

New runs use Georax `bf5c3554bfa8199427d4a3200fa58b69b657aa98`, its public `TangentTorus` geometry, CF-EES(2,5;1/10) and Geometric Euler–Heun. Historical learning and memory records used the earlier implementation with parameter 1/4. The protocol marker is now `kuramoto_second_order_mean_field_georaxbf5c355_v4`, so the plotting script rejects records from earlier revisions, including the former Itô Euler baseline. Historical results retain their original solver and dependency provenance and cannot be relabelled as Euler–Heun measurements. The completed full-data training comparison is documented in [the verified results](results/unified/reversible_second_order_verified_20260925/README.md): reversible CF-EES reached test loss 361.03 in 6.68 minutes, and matched-model CG2 reached 361.19 in 11.51 minutes (15 epochs, seed 0).
