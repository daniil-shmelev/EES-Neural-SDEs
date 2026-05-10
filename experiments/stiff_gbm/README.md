# Stiff GBM

TorchSDE experiment on high-volatility geometric Brownian motion, fitted to
European call prices. It supports `table:gbm`, `fig:gbm`, and `fig:gbm_grad`.

The integrator is selected via the `method` field in the in-script config
(`reversible_heun`, `ees25`, or `ees27`).

## Setup

```bash
uv pip install -e ".[stiff-gbm]"
```

This pulls the patched `torchsde` build with EES and MCF reversible solver
support.

## Run

```bash
python experiments/stiff_gbm/GBM.py
python experiments/stiff_gbm/plot_GBM.py
```

Rerun the scripts to regenerate `results/GBM_mse.pdf` and related outputs.
