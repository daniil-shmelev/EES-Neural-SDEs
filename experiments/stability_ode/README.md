# ODE stability regions (paper §3, Fig. `fig:ees25_stability`)

Reproduces Figure 1 of the manuscript: stability domain for $\mathrm{EES}(2,5;1/10)$ compared to RK4, the McCallum–Foster reversible variants of Euler and Midpoint, and the Reversible Heun method on the linear test ODE $\mathrm{d}y = \lambda y\,\mathrm{d}t$.

For an explicit Runge–Kutta method $(A, b)$, the stability function is
$$R(z) = \frac{\det(I - z A + z \mathbf 1 b^\top)}{\det(I - z A)}, \qquad z = \lambda h,$$
and the stability region is $\{z \in \mathbb C : |R(z)| < 1\}$.

For a McCallum–Foster reversible scheme based on a method with propagating function $R$, the linearised iteration on the test ODE is a $2\times 2$ matrix; its A-stability region is the set on which the spectral radius is $< 1$. We use the closed-form eigenvalue expressions verbatim from [`sammccallum/reversible/experiments/stability/plot.py`](https://github.com/sammccallum/reversible/blob/master/experiments/stability/plot.py) — the same formulas that produced Figure 1 of [McCallum & Foster (2024)](https://arxiv.org/abs/2410.11648). The script uses the published $\lambda = 0.8$.

The Reversible Heun scheme is mean-square stable only on the imaginary segment $[-i, i]$ (Kidger et al. 2021, Theorem D.19) and is drawn as a bold coloured segment on the imaginary axis since it has empty interior.

EES(2,5;1/10) and RK4 contours are computed via the determinant formula in [kauri](https://github.com/daniil-shmelev/kauri), mirroring `examples/ees_schemes/stability.ipynb`.

## Run

```bash
python experiments/stability_ode/stability_regions.py
```

Writes `results/stability_regions_1.pdf`.

## Setup

```bash
uv pip install -e ".[stability-ode]"
```

Pulls `kauri` (for the EES25 / RK3 / RK4 Butcher tableaux).

## Results committed

| File | Backs |
|---|---|
| `results/stability_regions_1.pdf` | `fig:ees25_stability` |
