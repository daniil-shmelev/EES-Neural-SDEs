# ODE Stability Regions

Reproduces the paper's ODE stability-region figure: the stability domain for
`EES(2,5;1/10)` compared with RK4, McCallum-Foster reversible Euler,
and Reversible Heun on the scalar linear ODE test problem.

## Setup

```bash
uv pip install -e ".[stability-ode]"
```

## Run

```bash
python experiments/stability_ode/stability_regions.py
```

The script writes `experiments/stability_ode/results/stability_regions_1.pdf`.
