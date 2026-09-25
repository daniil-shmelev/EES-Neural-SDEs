# Stability cross-sections (paper §3, Fig. `fig:ees_stoch_stability`)

Plots cross-sections of the mean-square stability domains for `RK4` and `EES(2,5)` applied to the linear SDE test equation `dy = λ y dt + μ y dW`, with the published ODE stability set for `Reversible Heun` overlaid in red.

For `EES(2,5)` the solution is mean-square stable iff
$$\mathbb E\bigl[\,|1 + \rho + \tfrac{1}{2}\rho^2 + \tfrac{1}{8}\rho^3|^2\,\bigr] < 1, \qquad \rho = \lambda\,dt + \mu\,dW.$$
The 4D domain is sliced along four 2D cross-sections (real, imaginary, two diagonal).

## Run

```bash
./run stability
```

Run from the repository root. The launcher selects uv dependencies and writes `stoch_stability.{pdf,eps,png}` in a fresh `results/runs/<timestamp>/` directory. Use `--help` for settings; `--resolution` defaults to 500.
