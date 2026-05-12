# Mean-Square Stability

Plots the paper's mean-square stability figure: cross-sections of the stability
domains for RK3, RK4, and EES(2,5) on the linear SDE test equation
`dy = lambda y dt + mu y dW`.

## Setup

```bash
uv pip install -e .
```

## Run

```bash
python stability/sde/stability.py
```

The script writes `stability/sde/results/stoch_stability.pdf`.
