# Mean-Square Stability

Plots `fig:ees_stoch_stability`: cross-sections of the mean-square stability
domains for RK3, RK4, and EES(2,5) on the linear SDE test equation
`dy = lambda y dt + mu y dW`.

## Setup

```bash
uv pip install -e .
```

## Run

```bash
python experiments/stability/stability.py
```

The script writes `experiments/stability/results/stoch_stability.pdf`.
