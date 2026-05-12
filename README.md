# Explicit and Effectively Symmetric Schemes for Neural SDEs

Code for the paper *"Explicit and Effectively Symmetric Schemes for Neural SDEs on Lie Groups"*.

This umbrella repository contains the implementations, scripts, committed
result files, and submodules used to reproduce the paper figures and tables.
Each experiment under `experiments/` has its own README with setup notes.

## Repository Layout

```text
order_verification/             Symbolic MKW checks for CFEES order conditions
stability/
  ode/                          ODE stability domains for EES schemes and reversible solvers
  sde/                          Mean-square SDE stability cross-sections
experiments/
  convergence_fbm/              Euclidean and SO(3) fBm/RDE convergence checks
  ou/                           Ornstein-Uhlenbeck latent SDE companion experiment
  stiff_gbm/                    Stiff GBM neural SDE experiment
  stochastic_volatility/        Seven-model stochastic-volatility benchmark
  kuramoto/                     Stochastic Kuramoto neural SDE on T^N x R^N
  torus/                        Torus neural SDE memory benchmark
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

| Extra | What it pulls in | Area(s) |
|---|---|---|
| no extras | numpy, matplotlib, scipy, fbm | `stability/sde/` |
| `algebra` | kauri, sympy | `order_verification/`, `stability/ode/` |
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
python stability/ode/stability_regions.py
python stability/sde/stability.py
python experiments/convergence_fbm/convergence.py
python -m experiments.convergence_fbm.scripts.so3_reversibility_verify
python -m experiments.convergence_fbm.scripts.so3_reversibility_plot

# Symbolic order verification
python -m order_verification.verify

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

- Sphere latent SDE/NSDE: the Human Activity experiments use the UCI
  `ConfLongDemo_JSI.txt` file under
  `experiments/sphere_latent_sde/data_dir/PersonActivity/`. Download and
  preprocess it from the UCI Machine Learning Repository with:

  ```bash
  cd experiments/sphere_latent_sde
  python -c "from data.activity_provider import HumanActivityProvider; HumanActivityProvider('data_dir', download=True)"
  ```

  The JAX NSDE loader can also read the raw file directly from
  `experiments/sphere_latent_sde/data_dir/PersonActivity/raw/`; the raw URL is
  `https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt`.
- The PyTorch sphere baseline also has online download helpers for Rotating
  MNIST and PhysioNet 2012; see
  `experiments/sphere_latent_sde/README.md`.
- Stochastic volatility: expects generated `.npz` files under
  `experiments/stochastic_volatility/data`.
- Kuramoto, torus, OU, stability, convergence, and pendulum data are synthetic
  or generated locally. Kuramoto smoke data is committed; full Kuramoto sweeps
  are generated by `experiments.kuramoto.datasets.pipeline`.

## Citation

Anonymised.

## License

Apache-2.0; see [LICENSE](LICENSE).
