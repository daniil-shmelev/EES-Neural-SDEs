# ODE convergence on SO(3)

Run from the repository root in a Python 3.13+ environment with Diffrax, georax, diffrax-lowstorage, JAX, NumPy, SciPy, and Matplotlib:

```sh
python -m experiments.convergence_ode.convergence
```

The experiment solves `R' = AR + RB`, `R(0) = I`, to time one, where `A = hat((0.4, -0.2, 0.7))` and `B = hat((-0.3, 0.8, 0.1))`. The exact solution is `exp(tA) exp(tB)`. A `GeometricTerm` supplies the right-action generator `R.T @ A @ R + B`; its only increment is the time step. Forward and reverse trajectories use `diffrax.diffeqsolve` with `georax.CFEES25` and `georax.CFEES27`, uniform steps, and float64 arithmetic. The selected SO(3) action uses a degree-eight Taylor exponential followed by QR.

The two-by-two figure uses the shared `experiments.convergence_plotting.plot_error_curve` helper and the shared STIX settings from `experiments.plotting`. Rows correspond to CF-EES(2,5) and CF-EES(2,7); columns show forward and recovery errors. Crimson crosses are measurements; blue lines have prescribed slopes. All measurements remain plotted, including those near roundoff.

Outputs in `results/` are `cfees_ode_convergence.pdf`, `ode_convergence.csv`, and `ode_validation.json`. The forward error is the maximum Frobenius error at mesh points. Recovery is the distance from the initial point after integrating backward from the numerical endpoint. The CSV also records a single forward–reverse defect, the embedded discrepancy, and the maximum orthogonality defect. The embedded diagnostic expresses the paper's normalized final-increment companion as a generic georax commutator-free tableau and checks its principal endpoint against the low-storage solver. It does not use CFEES25's built-in penultimate-stage companion.

Reported slopes fit the finest three measurements above `100 * eps(float64)`; the JSON records each fitting grid. The defaults are 2 through 128 steps. Options are `--min-power`, `--max-power`, `--terminal-time`, and `--output-dir`. Use `--plot-only` to redraw the saved CSV without rerunning the solvers.

The recorded run used georax `3e3b84ce3598627b2c2098cc1d3d8d8e2d030c13`, diffrax-lowstorage `5adac497d51ef424588ddb9252bbc52f8ce870c9`, and the sammccallum/Diffrax fork at `aeb1335b5a6278e85a270231e0f97b8db4453ae6`. The fork supplies `ReversibleAdjoint`. The local georax checkout had existing export changes in its two `__init__.py` files; its solver and SO(3) implementation files were unchanged. Package versions are recorded in the results JSON.

Install the unified solver stack with `uv pip install -e ".[convergence-ode]"`. The version hashes above describe the historical local run; new runs use the repository pins and record their installed package versions.
