# Stochastic Volatility

This experiment trains neural SDEs on seven stochastic-volatility datasets:
Black-Scholes, Heston, Rough Heston, Quadratic Rough Heston, Bergomi, Rough
Bergomi, and Classical Local Stochastic Volatility.

The implementation uses MLP drift/diffusion models on Euclidean state paths and
a fixed forward-NFE budget. The shipped solver set is:

- `ees25`
- `mcf_euler`
- `mcf_midpoint`
- `reversible_heun`

The loss is a truncated path-signature MMD objective on time-augmented paths.
Evaluation reports held-out loss and optional diagnostic predictions.

## Paper Results

| Paper item | Code/result |
|---|---|
| Rough Bergomi runtime/MSE | rough-Bergomi slice of the sweep |
| Seven-model stochastic-volatility sweep | full seven-model sweep |
| Stochastic-volatility model parameters | model/dataset parameter definitions |

## Setup

```bash
uv pip install -e ".[stochastic-volatility]"
```

This extra deliberately installs `diffrax-lowstorage`, pinned in
`pyproject.toml` to the Python 3.13-compatible revision used for the paper
experiments.

## Data

Training expects `.npz` datasets under `experiments/stochastic_volatility/data/`.
That directory is ignored by git and is not currently committed in this repo.
Expected filenames:

| File | Model |
|---|---|
| `black-scholes_data.npz` | Black-Scholes |
| `heston_data.npz` | Heston |
| `rough_heston_data.npz` | Rough Heston |
| `quadratic_rough_heston_data.npz` | Quadratic Rough Heston |
| `bergomi_data.npz` | Bergomi |
| `rough_bergomi_data.npz` | Rough Bergomi |
| `classical_local_stochastic_volatility_data.npz` | Classical Local Stochastic Volatility |

Each file must expose `driver` and `solution` arrays. The dataset loader uses a
70/15/15 train/validation/test split.

## Run

```bash
# Single config
python -m experiments.stochastic_volatility.experiment.train path/to/config.toml

# Full Cartesian sweep
python -m experiments.stochastic_volatility.experiment.train experiments/stochastic_volatility/configs/stoch_vol/sweep.toml

# One sweep entry
python -m experiments.stochastic_volatility.experiment.train experiments/stochastic_volatility/configs/stoch_vol/sweep.toml --index 0

# Count solver NFEs
python -m experiments.stochastic_volatility.experiment.count_nfe
```

Each run writes a timestamped directory under
`experiments/stochastic_volatility/results/<experiment>__<solver>__seed<N>__<timestamp>/`
containing `config.toml`, `nsde.eqx`, `history.json`, `metrics.json`, and
optional prediction/diagnostic artifacts.
