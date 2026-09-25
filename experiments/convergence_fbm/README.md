# fBm convergence (paper §3, Figs `fig:onvergence_4`–`6` + Figs `fig:cf_convergence_4`–`6`)

Numerical verification of the global error rates of EES-class schemes on RDEs driven by fractional Brownian motion at Hurst index $H \in \{0.4, 0.5, 0.6\}$. Two settings:

- **Euclidean fBm** (`convergence.py`) — reproduces the Redmann–Riedel (2020) example for $\mathrm{EES}_{\mathcal R}(2,5)$ and $\mathrm{EES}_{\mathcal R}(2,7)$. Reports both the discretization error $\mathcal E(h)$ and the backward (initial-condition recovery) error $\overleftarrow{\mathcal E}(h)$.
- **SO(3) RDE** (`scripts/so3_*.py`) — a nonlinear matrix-polynomial equation driven by three independent fBm components. Measures recovery rates of $\mathrm{CF\text{-}EES}(2,5;1/10)$ using the pinned Georax solver. Transposing the state preserves the original left-action convention with the library's right-action SO(3) chart.

## Run

From the repository root:

```bash
./run convergence-fbm
./run convergence-fbm --so3
```

The first command runs the Euclidean experiment. The second runs the SO(3) recovery sweep and plots its saved JSON. Each command selects its uv dependencies and creates a fresh `results/runs/<timestamp>/` directory. `--help` shows setup locations and available settings. The Euclidean defaults are in `convergence.py`; the SO(3) defaults are in `scripts/so3_reversibility_verify.py`.

The deterministic local-order check and page compositor remain developer utilities in `scripts/`; neither is needed to launch the experiments.

The SO(3) step is checked against the original stage combinations with exact matrix exponentials. Existing result files were not regenerated during the API migration.

## Results committed

| File | Backs |
|---|---|
| `results/ees_stochastic_convergence_H{40,50,60}.pdf` | Figs `fig:onvergence_4/5/6` |
| `results/cfees_stochastic_convergence_H{40,50,60}.pdf` | Figs `fig:cf_convergence_4/5/6` |
| `results/convergence_fbm_24up.pdf` | one-page 24-panel Euclidean + SO(3) layout |
| `results/ees_stochastic_convergence_12up.pdf` | one-page Euclidean 12-panel layout |
| `results/cfees_stochastic_convergence_12up.pdf` | one-page SO(3) 12-panel layout |
| `results/so3_reversibility_results.json` | data backing the CF-EES SO(3) figures |
