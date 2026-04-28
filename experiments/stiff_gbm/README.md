# Stiff GBM (paper §3.4, Tab. `table:gbm`, Figs `fig:gbm`/`fig:gbm_grad`)

Trains a neural SDE on high-volatility geometric Brownian motion dynamics, fitted to vanilla European call prices. Compares Reversible Heun against EES(2,5) and EES(2,7) at fixed step size.

The training loop is a torchsde-based `sdeint` over a Stratonovich neural SDE; the integrator is selected via the `method` field in the in-script `config` dict (`"reversible_heun"`, `"ees25"`, or `"ees27"`).

## Setup

```bash
uv pip install -e ".[stiff-gbm]"
```

This pulls Daniil's `torchsde` fork (`mcf` branch), which adds `EES25`, `EES27`, and the MCF reversible solvers to torchsde.

## Run

```bash
# Fit for each integrator (sets the `method` field inside the script then re-run, or pass --method via CLI)
python experiments/stiff_gbm/GBM.py

# Plot training MSE curves once all three runs have completed
python experiments/stiff_gbm/plot_GBM.py
```

## Results committed

None — rerun the scripts above to generate `results/GBM_mse.pdf` locally.
