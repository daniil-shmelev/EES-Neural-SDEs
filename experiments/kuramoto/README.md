# Stochastic 2nd-order Kuramoto Neural SDE on $T\mathbb{T}^N$

Replacement for the RNA torsion experiment. Trains a neural SDE on the
product Lie group $T\mathbb{T}^N = \mathbb{T}^N \times \mathbb{R}^N$ to
forecast trajectories of a stochastic Kuramoto-with-inertia network — the
canonical model of power-grid frequency stability (Filatrella, Nielsen &
Pedersen 2008; Schäfer et al. 2018; Olmi & Torcini 2024).

Plan: `~/.claude/plans/ok-carefully-plan-how-sequential-cerf.md`.

## Dynamics

We integrate the canonical second-order Kuramoto SDE (Olmi & Torcini 2024
eq. (1) with $K_2 = 0$; deterministic part from Filatrella, Nielsen &
Pedersen 2008; noise term as in Schmietendorf et al. 2014 / Schäfer et al.
2018):

$$m\,\ddot{\theta}_i = -\dot{\theta}_i + \Omega_i
    + \frac{K}{N}\sum_j \sin(\theta_j - \theta_i) + \xi_i(t),
    \qquad \langle\xi_i(t)\xi_j(s)\rangle = 2D\,\delta_{ij}\delta(t-s).$$

Bimodal natural frequencies $\Omega_i \in \{+P, -P\}$ (Filatrella generator
/ consumer balance). The state for SDE integration is
$(\theta, \omega) \in \mathbb{R}^{2N}$ on the cotangent bundle
$T\mathbb{T}^N$.

## Workflow

Heavy compute (data generation, GPU sweeps, training) runs on a separate,
more powerful machine. Implementation happens locally. Each milestone ends
with a portable orchestrator script.

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

- `datasets/kuramoto.py` — `KuramotoParams` (`eqx.Module`), drift function
  (vectorised mean-field $\sin(\theta_j-\theta_i)$ coupling), Brownian
  diffusion factory, complex order parameter.
- `datasets/simulator.py` — diffrax `Heun` simulator on $(\theta, \omega)
  \in \mathbb{R}^{2N}$ with `MultiTerm(ODETerm, ControlTerm)` and
  `VirtualBrownianTree`. Reuses `wrap_to_pi` from `experiments/rna/models/torus.py`.
- `datasets/pipeline.py` — JIT-compiled batched simulator
  (`simulate_batch_jit`), chunked simulation (`simulate_split`),
  persistence (`save_split`), and per-$N$ orchestration (`generate_one_n`).
- `datasets/verification.py` — analytic two-oscillator phase-lock study
  (deterministic limit, $D=0$) and SDE order-parameter stationarity study.
- `datasets/visualize.py` — phase-circle GIF: $N$ oscillators on the unit
  circle plus the complex order-parameter arrow $r e^{i\Psi}$.
- `scripts/run_m1.py` — single CLI orchestrator.
- `tests/test_kuramoto.py` — unit tests on `KuramotoParams`, the drift
  function, the analytic 2-oscillator phase-lock, and the order parameter.

Choice of solver: `Heun` at fine $dt$ (Stratonovich strong order 0.5,
weak order 1). Diffusion is constant additive noise on $\omega$, so
Stratonovich and Itô coincide and `Heun` gives a clean reference
trajectory at sub-second wall-clock per chunk.

### Local smoke

The local Windows environment currently has a broken JAX / jaxlib pin;
local validation is restricted to `py_compile`. Functional testing
happens on the remote machine.

If you fix the local env:

```
python -m experiments.kuramoto.scripts.run_m1 --smoke
```

### Remote hand-off (single command)

```bash
cd /path/to/EES-Neural-SDEs
git pull
python -m experiments.kuramoto.scripts.run_m1
```

Defaults match the plan: $N \in \{2, 4, 8\}$, $P=0.5$, $K=2.0$
(deterministic 2-oscillator phase-lock $\arcsin(2P/K) \approx 0.524$),
$m=1$, $D=0.05$, $T=5\,\text{s}$, $n_\text{fine}=16384$,
$n_\text{obs}=200$, 5000/1000/1000 train/val/test.
Override any subset with the matching CLI flag. To customise the network
sizes: `--n-list 2 4 8`. To skip verification: `--skip-verify`. To
disable the GIF: `--no-gif`.

Expected wall-clock on a single A100 / H100: ~3 min verification, ~30 min
total data gen, ~30 s GIF render. NPZ sizes ~3-15 MiB each, GIF ~1-3 MiB.

### What to send back

- `experiments/kuramoto/results/simulator_verification.{json,npz}` (small)
- `experiments/kuramoto/data/kuramoto_N{2,4,8}_seed0.{npz,json}` (large)
- `experiments/kuramoto/data/kuramoto_N2_seed0.gif` (small, demo)

### Pass criterion

- Phase-lock relative error in the verification JSON drops below ~$10^{-3}$
  by $n_\text{fine} = 16384$ (`Heun` converges to the analytic
  $\Delta\theta_\infty = \arcsin(2P/K)$).
- Order-parameter stationary mean is finite and the std doesn't grow
  with time (no integration blow-up).
- The GIF shows partial-sync dynamics: $N$ phase points on the unit
  circle, the order-parameter arrow $re^{i\Psi}$ wandering with
  intermediate length $r \in (0.3, 0.9)$.

If all three pass, M2 (model + training scaffolding) starts on the dev
box and will not need remote compute again until M3.
