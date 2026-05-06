# Explicit and Effectively Symmetric Schemes for Neural SDEs

Code for the NeurIPS 2026 paper *"Explicit and Effectively Symmetric Schemes for Neural SDEs"* by Daniil Shmelev, Luke Thompson, and Cristopher Salvi.

This umbrella repository reproduces every experiment, figure, and table in the paper. Each subdirectory under `experiments/` is self-contained: its own `README.md`, configs, data-fetch scripts, training/evaluation scripts, and result files that back the manuscript figures/tables.

## Repository layout

```
ees_core/                       Pure-numpy EES Butcher tableaux + RK driver (paper §3 numerics)
experiments/
  stability_ode/                ODE stability domains for EES(2,5;1/10) vs RK3/RK4 (Fig. fig:ees25_stability)
  stability/                    Mean-square stability cross-sections (Fig. fig:ees_stoch_stability)
  convergence_fbm/              Euclidean and SO(3) fBm convergence rates (Figs fig:onvergence_4-6 + fig:cf_convergence_4-6)
  order_verification/           Symbolic Lie–Butcher / MKW order-condition verification (Tab tab:alpha-cf-ees25-x)
  rna/                          RNA torus T^7 NSDE (paper §3.6, Tabs torus_*, Figs torus_*)
  sphere_latent_sde/            (git submodule) Latent SDE on S^15 with CF-EES + ReversibleAdjoint (paper §sec:sphere_latent_sde)
  stiff_gbm/                    Stiff GBM neural SDE (paper §3.4, Tab. table:gbm, Fig fig:gbm)
  ou/                           OU latent SDE (companion torchsde experiment)
docs/figure_table_index.md      Manuscript figure/table → script crossref
```

## Install

We recommend [`uv`](https://docs.astral.sh/uv/) and Python 3.13 (Python 3.10 is enough for the pure-numpy figures only).



Available extras:

| Extra | What it pulls in | Experiment(s) |
|---|---|---|
| (no extras) | numpy, matplotlib, scipy, fbm | `stability/`, `convergence_fbm/` |
| `algebra` | + kauri, sympy | `order_verification/`, `stability_ode/` |
| `stability-ode` | + kauri | `stability_ode/` (alias for `algebra`) |
| `jax-base` | + jax 0.10, equinox, optax, sammccallum/diffrax fork, georax, cyreal, diffrax-lowstorage, seali | shared base for all JAX experiments |
| `convergence-fbm` | jax-base | `convergence_fbm/scripts/` (SO(3) RDE convergence) |
| `order-verification` | algebra | `order_verification/` |
| `rna` | jax-base + biotite | `rna/` |
| `sphere` | torch + torchdiffeq + sklearn + tqdm | `sphere_latent_sde/` |
| `torchsde` | torch + torchsde fork (`mcf` branch) + torchcde | `stiff_gbm/`, `ou/` |
| `all` | all of the above | everything |
| `dev` | pytest, ruff, pyright | development |

## External libraries (pinned via `pyproject.toml`)

| Library | Source | Pinned at |
|---|---|---|
| `diffrax` | `github.com/sammccallum/diffrax` | `aeb1335b` |
| `diffrax-lowstorage` | `github.com/luke-a-thompson/diffrax-lowstorage` | `74fd4dba` |
| `georax` (the manuscript's anonymized "jax_geo_int") | `github.com/luke-a-thompson/georax` | `30ecbb9b` |
| `cyreal` | `github.com/luke-a-thompson/cyreal_dynamics` | `88f02657` |
| `Stochastax` | `github.com/luke-a-thompson/Stochastax` | `3775ec44` |
| `pySigLib` | `github.com/daniil-shmelev/pySigLib` | `v3.0.0` (`2beb2541`) |
| `kauri` | `github.com/daniil-shmelev/kauri` (`cf_methods` branch) | `0af3219a` |
| `torchsde` | `github.com/daniil-shmelev/torchsde` (`mcf` branch) | `a0b71269` |

## Reproducing the paper

Quick start commands per experiment:

```bash
# §3 ODE stability domains (Fig fig:ees25_stability)
python experiments/stability_ode/stability_regions.py

# §3 SDE mean-square stability cross-sections (Fig fig:ees_stoch_stability)
python experiments/stability/stability.py

# §3 fBm convergence (Figs fig:onvergence_4/5/6)
python experiments/convergence_fbm/convergence.py

# §3 CF-EES on SO(3) numerical convergence (Figs fig:cf_convergence_4/5/6)
python -m experiments.convergence_fbm.scripts.so3_local_order_verify
python -m experiments.convergence_fbm.scripts.so3_reversibility_verify
python -m experiments.convergence_fbm.scripts.so3_reversibility_plot

# Symbolic Lie–Butcher / MKW order-condition verification (Tab tab:alpha-cf-ees25-x)
python -m experiments.order_verification.scripts.scheme_a_algebraic_verify
python -m experiments.order_verification.scripts.scheme_a_tree_algebraic_verify
python -m experiments.order_verification.scripts.mkw_verify_cf

# §3.6 RNA torus T^7 (Tabs torus_*, Figs torus_*)
python -m experiments.rna.experiment.train_rna experiments/rna/configs/nsde.toml
python -m experiments.rna.scripts.run_rna_sweep
python -m experiments.rna.scripts.run_rna_hlo_sweep

# §sec:sphere_latent_sde Latent SDE on S^15 (Tab tab:sphere_parity, Fig fig:sphere_memory)
# (run from inside the submodule, since it's the unmodified Zeng et al. 2023 layout)
cd experiments/sphere_latent_sde
python activity_classification.py
python scripts/compute_parity_sweep.py
python scripts/memory_sweep_integrator.py
cd ../..

# §3.4 Stiff GBM neural SDE (Tab table:gbm, Fig fig:gbm)
python experiments/stiff_gbm/GBM.py        # set `method` in the script then re-run for each integrator
python experiments/stiff_gbm/plot_GBM.py

# OU latent SDE companion
python experiments/ou/OU.py
python experiments/ou/plot_OU.py
```

The full mapping from manuscript figure/table to script is in `docs/figure_table_index.md`.

## Datasets

Raw data is **not** shipped. Each experiment's `datasets/` (or `data/`) module fetches what it needs:

- **RNA torus** — BGSU representative-set RNA mmCIF files via `experiments/rna/datasets/download.py`
- **Sphere latent SDE** — UCI Human Activity Recognition; see `experiments/sphere_latent_sde/data/README.md`

## Citation

Anonymised

## License

Apache-2.0 — see [LICENSE](LICENSE).
