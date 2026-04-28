# Stability cross-sections (paper §3, Fig. `fig:ees_stoch_stability`)

Plots cross-sections of the mean-square stability domains for `RK3`, `RK4`, and `EES(2,5)` applied to the linear SDE test equation `dy = λ y dt + μ y dW`.

For `EES(2,5)` the solution is mean-square stable iff
$$\mathbb E\bigl[\,|1 + \rho + \tfrac{1}{2}\rho^2 + \tfrac{1}{8}\rho^3|^2\,\bigr] < 1, \qquad \rho = \lambda\,dt + \mu\,dW.$$
The 4D domain is sliced along four 2D cross-sections (real, imaginary, two diagonal).

## Run

```bash
python experiments/stability/stability.py
```

Writes `results/stoch_stability.{pdf,eps,png}`.

## Dependencies

Just the core extras: `pip install -e .` (numpy + matplotlib).
