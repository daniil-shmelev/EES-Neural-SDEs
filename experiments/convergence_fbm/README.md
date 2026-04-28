# fBm convergence (paper §3, Figs `fig:onvergence_4`–`6` + Figs `fig:cf_convergence_4`–`6`)

Numerical verification of the global error rates of EES-class schemes on RDEs driven by fractional Brownian motion at Hurst index $H \in \{0.4, 0.5, 0.6\}$. Two settings:

- **Euclidean fBm** (`convergence.py`) — reproduces the Redmann–Riedel (2020) example for $\mathrm{EES}_{\mathcal R}(2,5)$ and $\mathrm{EES}_{\mathcal R}(2,7)$. Reports both the discretization error $\mathcal E(h)$ and the backward (initial-condition recovery) error $\overleftarrow{\mathcal E}(h)$.
- **SO(3) RDE** (`scripts/so3_*.py`) — same Redmann–Riedel test bed, lifted to a Lie-group RDE driven by 2-dimensional fBm. Verifies the predicted $(p+1)\alpha - 1$ forward and antisymmetric-order rates of $\mathrm{CF\text{-}EES}(2,5)$.

## Run

```bash
# Euclidean (no extras needed)
python experiments/convergence_fbm/convergence.py

# SO(3) numerical convergence + reversibility sweep
python -m experiments.convergence_fbm.scripts.so3_local_order_verify
python -m experiments.convergence_fbm.scripts.so3_reversibility_verify
python -m experiments.convergence_fbm.scripts.so3_reversibility_plot       # writes the CF-EES SO(3) PDFs
```

## Setup

```bash
# Euclidean only (numpy + matplotlib + fbm)
uv pip install -e .

# SO(3) scripts need JAX + georax + diffrax fork
uv pip install -e ".[convergence-fbm]"
```

## Results committed

| File | Backs |
|---|---|
| `results/ees_stochastic_convergence_H{40,50,60}.pdf` | Figs `fig:onvergence_4/5/6` |
| `results/cfees_stochastic_convergence_H{40,50,60}.pdf` | Figs `fig:cf_convergence_4/5/6` |
| `results/so3_reversibility_results.json` | data backing the CF-EES SO(3) figures |
