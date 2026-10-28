# Stiff GBM (paper §3.4, Tab. `table:gbm`, Figs `fig:gbm`/`fig:gbm_grad`)

Trains a neural SDE on high-volatility geometric Brownian motion dynamics, fitted to vanilla European call prices. Compares Reversible Heun against EES(2,5) and EES(2,7) at fixed step size.

The training loop is a torchsde-based `sdeint` over a Stratonovich neural SDE; select the integrator with `--method reversible_heun`, `--method ees25` or `--method ees27`.

## Run

From the repository root:

```bash
./run stiff-gbm
./run stiff-gbm --method ees25
```

The default runs Reversible Heun, EES(2,5) and EES(2,7), then plots the comparison. `--method` selects one method. The launcher selects the uv dependencies, including Daniil's torchsde fork, and creates a fresh `results/runs/<timestamp>/` directory. Each method saves its effective `config.json` alongside the measurements.

Use `--epochs`, `--seed` and `--output-dir` to override those settings. `--help` identifies the training script containing the remaining experiment settings. Numerical and training defaults are unchanged.

## Results committed

None. New comparisons write `GBM_mse.pdf` inside the printed run directory.
