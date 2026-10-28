# OU latent SDE

Trains a Neural Latent SDE (LSDE) on high-volatility Ornstein–Uhlenbeck dynamics. Compares Reversible Heun against EES(2,5) on a `torchsde`-based pipeline. Adapted from the [Stable Neural SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs) tutorial.

Select an integrator with `--method`; the other default settings are in `OU.py`.

## Run

From the repository root:

```bash
./run ou
./run ou --method ees25
```

The default runs Reversible Heun and EES(2,5), then plots the comparison. `--method` selects one method. The launcher selects the uv dependencies, including Daniil's torchsde fork, and creates a fresh `results/runs/<timestamp>/` directory. Each method saves its effective `config.json` alongside the measurements.

Use `--epochs`, `--seed` and `--output-dir` to override those settings. `--help` identifies the training script containing the remaining experiment settings. Numerical and training defaults are unchanged.

## Results committed

New comparisons write `OU_mse.pdf` and `OU_mse_grads.pdf` inside the printed run directory.
