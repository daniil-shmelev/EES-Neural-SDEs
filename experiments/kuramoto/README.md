# Kuramoto learning experiment

The active experiment trains one geometric neural SDE on the tangent torus with 1000 oscillators. Its drift enforces `dtheta/dt = omega` and learns a phase-dependent mean-field force plus scalar damping `2*sigmoid(damping_logit)` in `(0, 2)`, initialized at 1. This includes the cached dataset's true unit damping. The force uses a shared per-oscillator network with trigonometric moments and learned oscillator embeddings; it has no velocity inputs. The diffusion remains a learned neural field. The same architecture is used for CG2, CF-EES and Geometric Euler–Heun. This changes the model assumption from the earlier unconstrained drift, whose backward reconstruction amplified numerical errors. The cached dataset has 1500/300/300 training/validation/test trajectories, 100 observations and horizon 5. Training uses the multi-horizon energy score with four stochastic predictions per example.

## One configuration

[configs/kuramoto.toml](configs/kuramoto.toml) defines the model, dataset, loss and training settings. Training and isolated memory measurements both use the same model factory and loss. The fixed-step training comparison and memory measurements use these solver--adjoint combinations:

| Solver | Adjoint | Steps | Forward drift evaluations | Forward diffusion evaluations |
| --- | --- | ---: | ---: | ---: |
| CG2 | Full | 75 | 150 | 150 |
| CG2 | Recursive | 75 | 150 | 150 |
| CF-EES(2,5) | Reconstruction without checkpoints | 50 | 150 | 150 |
| Geometric Euler–Heun | Full | 150 | 150 | 300 |
| Geometric Euler–Heun | Recursive | 150 | 150 | 300 |

Geometric Euler–Heun uses Georax's `GeometricEulerHeun()` directly on `TangentTorus`, with one drift and two diffusion evaluations per step. It uses the same Stratonovich convention as CG2 and CF-EES, so no drift conversion is needed. Each method is trained independently from the same initialization, architecture and loss. Euler–Heun supports fixed steps and full or recursive checkpointing; it has no embedded error estimate or reversible adjoint. The defaults match drift evaluations, not total field evaluations or runtime. For accuracy–time comparisons, vary each method's step count and use measured wall-clock time.

Run commands from the repository root:

```bash
./run kuramoto
./run kuramoto --memory
```

Both commands include the two Euler–Heun variants by default. To train only Euler–Heun for one seed:

```bash
./run kuramoto --methods geometric_euler_heun_full geometric_euler_heun_recursive --seeds 0
```

Use `--n-steps-geometric-euler-heun-full` and `--n-steps-geometric-euler-heun-recursive` to override their respective 150-step defaults. For an Euler–Heun-only memory sweep, pass `--pairs geometric_euler_heun:checkpoint_full geometric_euler_heun:checkpoint_recursive` to `./run kuramoto --memory`. Individual training runs accept `--solver geometric_euler_heun --adjoint checkpoint_full --n-steps 150` (or `checkpoint_recursive`) through `python -m experiments.kuramoto.experiment.train_kuramoto`.

Each invocation gets a fresh directory under `results/unified/runs/`. Training prints progress and saves a log per method; `--memory` writes both `memory.json` and `memory.pdf`. Use `--seeds 0` for one training seed, `--config PATH` for a configuration override, and `--help` for all options. Memory sweeps vary the solver step count while keeping the model, batch, observations and loss fixed. They measure an isolated gradient calculation on one real training batch, excluding the optimiser and the full resident datasets. Training histories separately record process peaks during training and evaluation. These are distinct memory measurements of the same model.

Training, validation and test batches print progress and elapsed time; the first training and evaluation batches include JAX compilation. Evaluation computes loss and MAEs from the same forecast ensemble and counts each held-out trajectory once, masking the padded final batch and weighting batch averages by their valid example counts. Diffusion applies its diagonal entries directly instead of materialising a dense matrix, and loss calculations save only their four requested observation times. With `save_every_epoch = true`, epoch weights and the latest optimiser state are saved before evaluation, so an interruption during evaluation preserves that epoch's training weights. `--resume-from` accepts final checkpoints or the last fully evaluated epoch with its matching latest optimiser state. If evaluation was interrupted after a newer optimiser was saved, resume refuses to pair that optimiser with the older evaluated model.

The plot displays actual peak allocations, without flattening or baseline subtraction. It rejects historical data without the unified protocol marker and rejects sweeps that mix model or loss settings. The isolated memory sweep has not been rerun for the second-order model; the previous memory figure cannot be relabelled as a measurement of this model.

## Adaptive CF-EES

Train with the final-increment embedded estimator, dynamic step sizes and checkpoint-free reversible reconstruction gradients:

```bash
./run kuramoto --adaptive --seed 0
```

The launcher uses `reversible`, `n_steps = 10` and `adaptive_dtmin = 1e-4` for this mode; explicit CLI flags override the preset. The shared TOML supplies `adaptive_atol = 3e-2`. The tolerance controls the estimated RMS difference between the two endpoints, using wrapped phase differences and ordinary velocity differences. Diffrax's standard `PIDController`, configured as a PI controller with `force_dtmin=True`, adjusts the step size, capped at `T/n_steps` (0.5 for this configuration). At `adaptive_dtmin`, steps are accepted even if the estimated error exceeds tolerance. The Brownian-tree tolerance is ten times smaller than `adaptive_dtmin`. `adaptive_max_steps` defaults to 4096 attempted steps per trajectory. These options can be set in the shared TOML configuration or with the corresponding CLI flags.

Batched adaptive reconstruction currently needs a Diffrax fix: the pinned fork reads a padded time after one batch member finishes its reverse loop. The tested fix is in `/home/luke/diffrax/.worktrees/adaptive-reversible`, on branch `codex/adaptive-reversible-vmap`; the launcher selects that worktree automatically for `--adaptive`. The original pin remains unchanged. The verified SIAM result directory retains the exact patch and dependency provenance.

The Brownian tree is retained through rejection. The learned diffusion depends on the full state and need not commute, so the standard `ClipStepSizeController` wrapper stores rejected endpoints and forces subsequent steps to revisit them. CF-EES uses the ordinary `ReversibleAdjoint` and reconstructs states on the accepted mesh without trajectory checkpoints. The accepted-time buffer is separate from the reconstructed state. No inverse-correction loop or checkpointed gradient is substituted. The previous unconstrained drift remains available as `drift_kind = "indexed_mean_field"` for reading historical checkpoints and reproducing the failure; new runs use `second_order_mean_field`. The earlier checkpointed speed comparison does not establish reversible-training performance. `--adjoint checkpoint_recursive` remains an explicit diagnostic alternative. Adaptive runs have variable evaluation counts; they are additional experiments to the fixed 150-forward-drift-evaluation comparison. `metrics.json` records accepted/rejected steps and forward generator evaluations on eight fixed validation trajectories using the best validation checkpoint.

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

Historical models, presets, pilot scripts and documentation are in `/home/luke/nsde_archive/experiments/kuramoto/legacy/`. The current SIAM learning comparison, manuscript datasets, historical memory sweep and simulator diagnostics remain here. NeurIPS fixed-step runtime records are now in `../../neurips/kuramoto/results/`; superseded trial runs were deleted as recorded in `../../docs/cleanup_20261027.json`. The archived memory curve used a global MLP with 1,204,792 parameters; the second-order mean-field training model has 582,251. These results must not be merged into the new memory sweep.

New runs use Georax `bf5c3554bfa8199427d4a3200fa58b69b657aa98`, its public `TangentTorus` geometry, CF-EES(2,5;1/10) and Geometric Euler–Heun. Historical learning and memory records used the earlier implementation with parameter 1/4. The protocol marker is now `kuramoto_second_order_mean_field_georaxbf5c355_v4`, so the plotting script rejects records from earlier revisions, including the former Itô Euler baseline. Historical results retain their original solver and dependency provenance and cannot be relabelled as Euler–Heun measurements. The completed full-data training comparison is documented in [the verified results](results/unified/reversible_second_order_verified_20260925/README.md): reversible CF-EES reached test loss 361.03 in 6.68 minutes, and matched-model CG2 reached 361.19 in 11.51 minutes (15 epochs, seed 0). Isolated memory measurements have not been repeated.
