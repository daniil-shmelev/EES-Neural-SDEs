# fBm/RDE Convergence

Numerical verification of global error rates for EES-class schemes on RDEs
driven by fractional Brownian motion at Hurst indices `H in {0.4, 0.5, 0.6}`.

- `convergence.py`: Euclidean Redmann-Riedel test problem for `EES_R(2,5)`
  and `EES_R(2,7)`.
- `scripts/so3_*.py`: the same test problem lifted to an SO(3) RDE, verifying
  forward and antisymmetric-order rates for `CF-EES(2,5)`.

## Paper Results

| Paper item | Result |
|---|---|
| EES fBm convergence summary | `results/ees_stochastic_convergence_H{40,50,60}.pdf` |
| CF-EES SO(3) convergence summary | `results/cfees_stochastic_convergence_H{40,50,60}.pdf`, `results/so3_reversibility_results.json` |

Older per-H plots remain committed in `results/` and are useful for regression
checks.

## Setup

```bash
# Euclidean only
uv pip install -e .

# SO(3) scripts
uv pip install -e ".[convergence-fbm]"
```

## Run

```bash
python experiments/convergence_fbm/convergence.py
python -m experiments.convergence_fbm.scripts.so3_local_order_verify
python -m experiments.convergence_fbm.scripts.so3_reversibility_verify
python -m experiments.convergence_fbm.scripts.so3_reversibility_plot
```
