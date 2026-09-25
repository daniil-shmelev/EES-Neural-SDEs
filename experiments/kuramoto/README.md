# Kuramoto learning experiment

The active experiment trains one geometric neural SDE on the tangent torus with 1000 oscillators. Its drift enforces `dtheta/dt = omega` and learns a phase-dependent mean-field force plus scalar damping `2*sigmoid(damping_logit)` in `(0, 2)`, initialized at 1. This includes the cached dataset's true unit damping. The force uses a shared per-oscillator network with trigonometric moments and learned oscillator embeddings; it has no velocity inputs. The diffusion remains a learned neural field. The same model is used for CG2 and CF-EES. This changes the model assumption from the earlier unconstrained drift, whose backward reconstruction amplified numerical errors. The cached dataset has 1500/300/300 training/validation/test trajectories, 100 observations and horizon 5. Training uses the multi-horizon energy score with four stochastic predictions per example.

## One configuration

[configs/kuramoto.toml](configs/kuramoto.toml) defines the model, dataset, loss and training settings. Training and isolated memory measurements both use the same model factory and loss. The fixed-step training comparison and memory measurements use these solver--adjoint combinations:

| Solver | Adjoint | Steps | Forward generator evaluations |
| --- | --- | ---: | ---: |
| CG2 | Full | 75 | 150 |
| CG2 | Recursive | 75 | 150 |
| CF-EES(2,5) | Reconstruction without checkpoints | 50 | 150 |

Run commands from the repository root:

```bash
./run kuramoto
./run kuramoto --memory
```

Each invocation gets a fresh directory under `results/unified/runs/`. Training prints progress and saves a log per method; `--memory` writes both `memory.json` and `memory.pdf`. Use `--seeds 0` for one training seed, `--config PATH` for a configuration override, and `--help` for all options. Memory sweeps vary the solver step count while keeping the model, batch, observations and loss fixed. They measure an isolated gradient calculation on one real training batch, excluding the optimiser and the full resident datasets. Training histories separately record process peaks during training and evaluation. These are distinct memory measurements of the same model.

Training, validation and test batches print progress and elapsed time; the first training and evaluation batches include JAX compilation. Evaluation computes loss and MAEs from the same forecast ensemble and counts each held-out trajectory once, masking the padded final batch and weighting batch averages by their valid example counts. Diffusion applies its diagonal entries directly instead of materialising a dense matrix, and loss calculations save only their four requested observation times. With `save_every_epoch = true`, epoch weights and the latest optimiser state are saved before evaluation, so an interruption during evaluation preserves that epoch's training weights. `--resume-from` accepts final checkpoints or the last fully evaluated epoch with its matching latest optimiser state. If evaluation was interrupted after a newer optimiser was saved, resume refuses to pair that optimiser with the older evaluated model.

The plot displays actual peak allocations, without flattening or baseline subtraction. It rejects historical data without the unified protocol marker and rejects sweeps that mix model or loss settings. The isolated memory sweep has not been rerun for the second-order model; the previous memory figure cannot be relabelled as a measurement of this model.

## Adaptive CF-EES

Train with the final-increment embedded estimator, dynamic step sizes and checkpoint-free reversible reconstruction gradients:

```bash
./run kuramoto --adaptive --seed 0
```

The launcher uses `reversible`, `n_steps = 10` and `adaptive_dtmin = 1e-4` for this mode; explicit CLI flags override the preset. The shared TOML supplies `adaptive_atol = 3e-2`. The tolerance controls the estimated RMS difference between the two endpoints, using wrapped phase differences and ordinary velocity differences. Diffrax's PI controller adjusts the step size, capped at `T/n_steps` (0.5 for this configuration). The Brownian-tree tolerance is ten times smaller than `adaptive_dtmin`; a step that cannot meet tolerance at the minimum size fails. `adaptive_max_steps` defaults to 4096 attempted steps per trajectory. These options can be set in the shared TOML configuration or with the corresponding CLI flags.

Batched adaptive reconstruction currently needs a Diffrax fix: the pinned fork reads a padded time after one batch member finishes its reverse loop. The tested fix is in `/home/luke/diffrax/.worktrees/adaptive-reversible`, on branch `codex/adaptive-reversible-vmap`; the launcher selects that worktree automatically for `--adaptive`. The original pin remains unchanged. The earlier adaptive attempt retains the exact patch and dependency provenance.

The Brownian tree is retained through rejection. CF-EES uses the ordinary `ReversibleAdjoint` and reconstructs states on the accepted mesh without trajectory checkpoints. The accepted-time buffer is separate from the reconstructed state. No inverse-correction loop or checkpointed gradient is substituted. The previous unconstrained drift remains available as `drift_kind = "indexed_mean_field"` for reading historical checkpoints and reproducing the failure; new runs use `second_order_mean_field`. The earlier checkpointed speed comparison does not establish reversible-training performance. `--adjoint checkpoint_recursive` remains an explicit diagnostic alternative. Adaptive runs have variable evaluation counts; they are additional experiments to the fixed 150-forward-evaluation comparison. `metrics.json` records accepted/rejected steps and forward generator evaluations on eight fixed validation trajectories using the best validation checkpoint.

The endpoint controller permits an accepted final step shorter than `adaptive_dtmin` when it has been clipped to the remaining interval. It preserves minimum-step failures before completion and on rejected steps. This avoids the pinned PID controller reporting failure solely because the proposed next step is too short after integration has already finished.

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

Historical models, presets, pilot scripts and documentation are in `/home/luke/nsde_archive/experiments/kuramoto/legacy/`. Manuscript datasets and the other measured results remain in this repository. The archived memory curve used a global MLP with 1,204,792 parameters; the second-order mean-field training model has 582,251. These results must not be merged into the new memory sweep.

New runs use Georax `8cbae326c64eccd0a2c37d1ea7767a14cc29b8ae`, its public `TangentTorus` geometry and CF-EES(2,5;1/10). Historical learning and memory records used the earlier implementation with parameter 1/4. The protocol marker is now `kuramoto_second_order_mean_field_georax8cbae32_v3`, so the plotting script rejects old records. The protocol revision records the second-order model, so earlier timings and memory measurements cannot be relabelled as measurements of this implementation. The completed full-data training comparison is documented in [the verified results](results/unified/reversible_second_order_verified_20260925/README.md): reversible CF-EES reached test loss 361.03 in 6.68 minutes, and matched-model CG2 reached 361.19 in 11.51 minutes (15 epochs, seed 0). Isolated memory measurements have not been repeated.
