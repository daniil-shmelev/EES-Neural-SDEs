# ODE stability regions (paper §3, Fig. `fig:ees25_stability`)

Reproduces Figure 1 of the manuscript: stability domain for $\mathrm{EES}(2,5;1/10)$ compared to Kutta's RK3 and RK4 on the linear test ODE $\mathrm{d}y = \lambda y\,\mathrm{d}t$.

For an explicit Runge–Kutta method $(A, b)$, the stability function is
$$R(z) = \frac{\det(I - z A + z \mathbf 1 b^\top)}{\det(I - z A)}, \qquad z = \lambda h,$$
and the stability region is $\{z \in \mathbb C : |R(z)| < 1\}$. The script overlays the $|R(z)| = 1$ contour of all three schemes in a single panel.

The implementation mirrors `examples/ees_schemes/stability.ipynb` from the [kauri repo](https://github.com/daniil-shmelev/kauri), specialised to the figure-1 layout.

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
