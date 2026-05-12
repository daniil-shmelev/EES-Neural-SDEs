# Stability Figures

This directory contains the paper's standalone stability-plot scripts.

## ODE Stability Regions

Reproduces the ODE stability-region figure: the stability domain for
`EES(2,5;1/10)` and `EES(2,7)` compared with RK4, McCallum-Foster reversible
Euler, and Reversible Heun on the scalar linear ODE test problem.

```bash
uv pip install -e ".[stability-ode]"
python stability/stability_regions.py
```

The script writes `stability/results/stability_regions_1.pdf`.

## Mean-Square SDE Stability

Plots the mean-square stability figure: cross-sections of the stability
domains for RK3, RK4, and EES(2,5) on the linear SDE test equation
`dy = lambda y dt + mu y dW`.

```bash
uv pip install -e .
python stability/stability.py
```

The script writes `stability/results/stoch_stability.pdf`.
