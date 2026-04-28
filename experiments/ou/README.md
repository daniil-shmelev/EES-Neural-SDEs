# OU latent SDE

Trains a Neural Latent SDE (LSDE) on high-volatility Ornstein–Uhlenbeck dynamics. Compares Reversible Heun against EES(2,5) on a `torchsde`-based pipeline. Adapted from the [Stable Neural SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs) tutorial.

The integrator is selected via the `method` field in the in-script `config` dict.

## Setup

```bash
uv pip install -e ".[ou]"
```

Pulls Daniil's `torchsde` fork (`mcf` branch) plus `torchcde`.

## Run

```bash
python experiments/ou/OU.py
python experiments/ou/plot_OU.py
```

## Results committed

| File | Backs |
|---|---|
| `results/OU_mse.pdf` | (companion plot) |
