# Stiff GBM

TorchSDE experiment on high-volatility geometric Brownian motion, fitted to
European call prices. It supports the paper's stiff-GBM metrics and loss
figure.

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

Rerun the scripts to regenerate `GBM_mse.pdf` and the method-specific outputs
under `plots/options_<method>/`.
