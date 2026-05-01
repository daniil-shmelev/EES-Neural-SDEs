# Stochastic volatility (paper §3.5, Tabs `tab:further_stoch_vol`, `tab:rough-volatility-parameters`)

Trains a neural SDE on seven stochastic volatility models — Black-Scholes, Heston, Rough Heston, Quadratic Rough Heston, Bergomi, Rough Bergomi, and Classical Local Stochastic Volatility — fitted by matching simulated path distributions. Drift and diffusion are MLPs on $\mathbb{R}^d$ with LipSwish activations and diagonal diffusion; integration uses a fixed NFE budget and compares `EES25`, `CFEES25`, `MCF_Euler`, `MCF_Midpoint`, `GL2`, and `ReversibleHeun` via `ReversibleAdjoint`.

Loss: truncated path-signature MMD (depth 6, biased $\widehat{\mathrm{MMD}}^2$ with dot-product kernel on time-augmented paths). Evaluation: two-sample KS statistic at $t = 55$.

This experiment backs:

- **Tab `tab:further_stoch_vol`** — solver comparison across all seven volatility models at fixed NFE budget
- **Tab `tab:rough-volatility-parameters`** — model parameters for the rough-volatility dataset family

## Setup

```bash
uv pip install -e ".[stoch_vol]"
```

Pre-computed training data for all seven models is committed under `data/`; no download step is required.

## Run

```bash
# Single training run (single config)
python -m experiments.stochastic_volatility.experiment.train path/to/config.toml

# Sweep run (all solver × model combinations in a sweep config)
python -m experiments.stochastic_volatility.experiment.train experiments/stochastic_volatility/configs/stoch_vol/sweep.toml

# Single entry from a sweep config
python -m experiments.stochastic_volatility.experiment.train path/to/sweep.toml --index 0

# Count actual NFE per solver step (validates the fixed-cost model)
python -m experiments.stochastic_volatility.experiment.count_nfe
```

Each run writes a timestamped output directory under `results/<experiment>__<solver>__seed<N>__<timestamp>/` containing `config.toml`, `nsde.eqx`, `history.json`, `metrics.json`, and optional `test_predictions.npz` and diagnostic plots.

## Results committed

None — rerun the scripts above to generate results locally.

## Data

| File | Model |
|---|---|
| `data/black-scholes_data.npz` | Black-Scholes |
| `data/heston_data.npz` | Heston |
| `data/rough_heston_data.npz` | Rough Heston |
| `data/quadratic_rough_heston_data.npz` | Quadratic Rough Heston |
| `data/bergomi_data.npz` | Bergomi |
| `data/rough_bergomi_data.npz` | Rough Bergomi |
| `data/classical_local_stochastic_volatility_data.npz` | Classical Local Stochastic Volatility |

Splits: 70 % train / 15 % val / 15 % test.
