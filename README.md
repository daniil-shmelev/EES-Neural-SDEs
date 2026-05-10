# Explicit and Effectively Symmetric Schemes for Neural SDEs

Code for the paper *"Explicit and Effectively Symmetric Schemes for Neural SDEs"*.

This umbrella repository contains the implementations, scripts, committed
result files, and submodules used to reproduce the paper figures and tables.
Each experiment under `experiments/` has its own README with setup notes,
command lines, and result-file provenance.

## Repository Layout

```text
experiments/
  stability_ode/                ODE stability domains for EES(2,5) vs RK3/RK4
  stability/                    Mean-square SDE stability cross-sections
  convergence_fbm/              Euclidean and SO(3) fBm/RDE convergence checks
  order_verification/           Symbolic Lie-Butcher and MKW order verification
  ou/                           Ornstein-Uhlenbeck latent SDE companion experiment
  stiff_gbm/                    Stiff GBM neural SDE experiment
  stochastic_volatility/        Seven-model stochastic-volatility benchmark
  kuramoto/                     Stochastic Kuramoto neural SDE on T T^N
  torus/                        Random T^7 neural SDE memory benchmark
  sphere_latent_nsde/           JAX/georax HumanActivity sphere latent NSDE
  sphere_latent_sde/            Submodule: PyTorch HumanActivity sphere latent SDE
  ees_dynamical_fitting/        Submodule: molecular-dynamics fitting benchmark
  plotting.py                   Shared Matplotlib styling for paper figures
PAPER_RESULTS.md                Paper figure/table to code cross-reference
```

## Install

We recommend [`uv`](https://docs.astral.sh/uv/) and Python 3.13. Python 3.13
is required by the pinned JAX/Diffrax/georax solver stack.

```bash
uv pip install -e ".[dev]"
```

Available extras:

| Extra | What it pulls in | Experiment(s) |
|---|---|---|
| no extras | numpy, matplotlib, scipy, fbm | `stability/` |
| `algebra` | kauri, sympy | `order_verification/`, `stability_ode/` |
| `jax-base` | jax, equinox, optax, pinned diffrax build, georax, cyreal, seali | shared JAX base |
| `lowstorage` | pinned diffrax and diffrax-lowstorage builds | shared EES(2,5)/EES(2,7) low-storage solvers |
| `convergence-fbm` | `lowstorage` | Euclidean and SO(3) convergence scripts |
| `order-verification` | `algebra` | symbolic order scripts |
| `stability-ode` | `algebra` | ODE stability domains |
| `torus` | `jax-base` | Torus memory benchmark |
| `kuramoto` | `jax-base` | Kuramoto |
| `sphere-jax` | `jax-base` | JAX/georax sphere latent NSDE |
| `stochastic-volatility` | `jax-base`, `lowstorage` | stochastic-volatility benchmark |
| `sphere` | torch, torchdiffeq, sklearn, tqdm, pandas | PyTorch sphere submodule |
| `torchsde` | torch, patched torchsde build, torchcde | OU and stiff GBM base |
| `stiff-gbm` | `torchsde`, scipy | stiff GBM |
| `ou` | `torchsde`, scipy | OU |
| `all` | all experiment extras | everything |
| `dev` | pytest, ruff, pyright | development |

## External Libraries

Pinned direct references are declared in `pyproject.toml`.

| Library | Source | Pinned at |
|---|---|---|
| `diffrax` | `github.com/sammccallum/diffrax` | `aeb1335b` |
| `diffrax-lowstorage` | `github.com/luke-a-thompson/diffrax-lowstorage` | `6bca5807` |
| `georax` | `github.com/luke-a-thompson/georax` | `30ecbb9b` |
| `cyreal` | `github.com/luke-a-thompson/cyreal_dynamics` | `88f02657` |
| `pySigLib` | `github.com/daniil-shmelev/pySigLib` | `v3.0.0` (`2beb2541`) |
| `kauri` | `github.com/daniil-shmelev/kauri` (`cf_methods` branch) | `0af3219a` |
| `torchsde` | `github.com/daniil-shmelev/torchsde` (`mcf` branch) | `a0b71269` |

## Reproducing the Paper

Use `PAPER_RESULTS.md` as the canonical paper-to-script map. Common entrypoints:

```bash
# Stability and convergence
python experiments/stability_ode/stability_regions.py
python experiments/stability/stability.py
python experiments/convergence_fbm/convergence.py
python -m experiments.convergence_fbm.scripts.so3_reversibility_verify
python -m experiments.convergence_fbm.scripts.so3_reversibility_plot

# Symbolic order verification
python -m experiments.order_verification.scripts.scheme_a_algebraic_verify
python -m experiments.order_verification.scripts.scheme_a_tree_algebraic_verify
python -m experiments.order_verification.scripts.mkw_verify_cf

# Main neural SDE experiments
python -m experiments.torus.plots
python experiments/ou/OU.py
python experiments/stiff_gbm/GBM.py
python -m experiments.stochastic_volatility.experiment.train experiments/stochastic_volatility/configs/stoch_vol/sweep.toml --index 0
python -m experiments.kuramoto.experiment.train_kuramoto experiments/kuramoto/configs/kuramoto.toml
python -m experiments.sphere_latent_nsde.train_activity experiments/sphere_latent_nsde/configs/activity/smoke.toml

# Submodules
git submodule update --init --recursive
```

The PyTorch sphere baseline is run from `experiments/sphere_latent_sde/`. The molecular-dynamics fitting code is in `experiments/ees_dynamical_fitting/`.

## Datasets

Raw data is not shipped. Dataset directories and checkpoints are ignored by default.

- Sphere latent SDE/NSDE: UCI Human Activity Recognition data under `experiments/sphere_latent_sde/data_dir`.
- Stochastic volatility: expects generated `.npz` files under `experiments/stochastic_volatility/data`.
- Kuramoto: committed smoke data is present; full sweeps are generated by `experiments.kuramoto.datasets.pipeline`.

## Citation

Anonymised.

## License

Apache-2.0; see [LICENSE](LICENSE).
