# Chaotic n-link Pendulum Neural SDE on $T\mathbb{T}^n$

Replacement for the RNA torsion experiment. Trains a neural SDE on the
product Lie group $T\mathbb{T}^n = \mathbb{T}^n \times \mathbb{R}^n$ to
forecast trajectories of a chaotic n-link planar pendulum driven by an
underdamped Langevin SDE on the cotangent bundle.

Plan: `~/.claude/plans/ok-carefully-plan-how-sequential-cerf.md` (locally on
the dev machine).

## Workflow

Heavy compute (data generation, GPU sweeps, training) runs on a separate,
more powerful machine. Implementation happens locally. Each milestone ends
with a portable orchestrator script. Results return as JSON / NPZ; plotting
and analysis run locally on returned artifacts.

| Milestone | Status | Hand-off |
| --- | --- | --- |
| M1: simulator + verification | implemented | yes |
| M2: model + training scaffolding | pending | no |
| M3: pilot diagnostic gate | pending | yes |
| M4: memory + scaling sweeps | pending | yes |
| M5: hero training | pending | yes |
| M6: plots + manuscript edits | pending | no |

## M1 — data simulator and verification

Implemented:

- `datasets/lagrangian.py` — closed-form $L(\theta, \dot\theta)$, $T$, $V$,
  $H$ for an n-link planar pendulum. Mass matrix and gradients via JAX
  autodiff. Drift functions for the underdamped Langevin SDE.
- `datasets/simulator.py` — diffrax `Heun` simulator on
  $(\theta, p) \in \mathbb{R}^{2n}$ with `MultiTerm(ODETerm,
  ControlTerm)` and `VirtualBrownianTree`. Vectorised batch simulator.
  Includes a symplectic-Verlet sub-routine for $\sigma=0$ sanity checks.
- `datasets/visualize.py` — matplotlib animation helper that renders an
  n-link pendulum trajectory to a GIF (used by the data generator to
  produce a sanity-check animation of the first test trajectory).
- `datasets/pipeline.py` — JIT-compiled batched simulator + persistence
  helpers (`simulate_batch_jit`, `simulate_split`, `save_split`,
  `generate_one_n`, `library_versions`).
- `datasets/verification.py` — `hamiltonian_drift_study`,
  `stationarity_study`, and `run_verification` driver.
- `scripts/run_m1.py` — single CLI orchestrator that runs verification
  then generates data for `--n-list` (default `2 4 8`).
- `tests/test_lagrangian.py` — unit tests on the JAX-derived mass matrix
  ($n=1$ scalar formula and $n=2$ analytic formula) and a smoke test of
  the SDE simulator.

Choice of solver: diffrax has first-class `UnderdampedLangevinDriftTerm` /
`UnderdampedLangevinDiffusionTerm` and Foster–Langevin SRKs (`ALIGN`,
`ShOULD`, `SPaRK`, `QUICSORT`), but those assume a *constant* diagonal
inverse-mass $u$. Our $M(\theta)$ is configuration-dependent, so we use a
generic `MultiTerm(ODETerm, ControlTerm)` with `Heun` at fine $dt$ instead.
This is the same pattern the RNA NSDE uses
(`experiments/rna/models/torus_nsde.py:308-322`).

### Local smoke

The local Windows environment currently has a broken JAX / jaxlib pin
(jaxlib 0.9.2 vs equinox 0.11.12 expecting `jaxlib.xla_extension`). `uv sync`
also fails on the current `requires-python` ≥ 3.10 vs the diffrax fork
needing 3.11. Until that's untangled, local validation is restricted to
`py_compile` (which has been run and passes for all M1 modules). The
remote machine is the source of truth for functional testing.

If you fix the local env first, the smoke command is:

```
python -m experiments.pendulum.scripts.run_m1 --smoke
```

### Remote hand-off (single command)

On the powerful machine, with the project installed
(`pip install -e .[jax-base]` should suffice; pyproject pulls in the diffrax
+ georax forks):

```bash
cd /path/to/EES-Neural-SDEs
git pull
python -m experiments.pendulum.scripts.run_m1
```

That's it — one orchestrator runs the simulator verification for $n=2$ and
generates training data for $n \in \{2, 4, 8\}$. A GIF of the first test
trajectory at the smallest $n$ in the list is rendered automatically into
the data directory.

The orchestrator's defaults match the plan
(`--sigma 0.3 --gamma 0.05 --T 2.0 --n-fine 16384 --n-obs 200
--n-train 5000 --n-val 1000 --n-test 1000 --seed 0`); override any subset
with the matching CLI flag. To customise the link counts:
`--n-list 2 4 8` (space-separated, `nargs='+'`). To skip verification:
`--skip-verify`. To disable the GIF: `--no-gif`.

Expected wall-clock on a single A100 / H100: a few minutes for verification,
~30 min total for the three data splits, ~30 s for the GIF render. NPZ sizes
~3-15 MiB each, GIF ~1-3 MiB.

### What to send back

- `experiments/pendulum/results/simulator_verification_n2.{json,npz}` (small)
- `experiments/pendulum/data/pendulum_n{2,4,8}_seed0.{npz,json}` (large; rsync / scp)
- `experiments/pendulum/data/pendulum_n2_seed0.gif` (small, useful for the
  manuscript / slide visuals)

### Pass criterion

- The `Heun` curve in the verification JSON drifts at most ~$10^{-3}$
  relative energy at $n_{\text{fine}}=16384$ over $T=2$s; the Verlet
  reference stays at machine epsilon. If `Heun` drift exceeds $10^{-2}$,
  bump $n_{\text{fine}}$ and re-run before generating data.
- The stationarity study should show the per-trajectory energy mean
  saturating to a roughly constant value over time (the stationary
  distribution); the spread should not blow up.
- The GIF should show recognisable (chaotic) double-pendulum motion: a
  visibly random swing trajectory, not all-zeros, not blowing up.

If all three pass, M2 (model + training scaffolding) starts on the dev box
and will not need remote compute again until M3.
