# Explicit and Effectively Symmetric Schemes for Neural SDEs on Lie Groups

Code and experiment records for [the paper](https://arxiv.org/abs/2509.20599). `main` is the target for the unified paper repository; `paper-unification` integrates the earlier experiment branches and local ODE work. See [PAPER_RESULTS.md](PAPER_RESULTS.md) for the figure/table index and [docs/unification.md](docs/unification.md) for the branch audit and reproduction gaps.

## Setup

Use Python 3.13. Initialise the pinned submodules when running sphere or molecular-dynamics experiments:

```sh
git submodule update --init --recursive
uv venv --python 3.13
uv pip install -e ".[convergence-ode,order-verification,dev]"
```

Install experiment extras as needed: `convergence-fbm`, `ou`, `stiff-gbm`, `option-pricing`, `kuramoto`, `torus`, `sphere-jax`, `sphere`, or `stochastic-volatility`. JAX extras install the CPU-capable stack; add `cuda` for Linux GPU runs. `all` installs all paper experiment extras. Solver revisions are pinned in [pyproject.toml](pyproject.toml). Historical measurements retain their original environment metadata.

## Experiments

| Directory | Purpose |
|---|---|
| `experiments/convergence_ode` | New deterministic SO(3) convergence, recovery and embedded diagnostics |
| `experiments/convergence_fbm` | Euclidean/SO(3) rough-driver convergence; 12/24-panel PDF compositor |
| `order_verification` | Kauri checks; `siam/` preserves the new exact ordered-forest verifier and records |
| `stability` | Original ODE/SDE figure recipes and separate revised Reversible-Heun overlay |
| `experiments/ou` | Fixed-NFE training, local relative-gradient diagnostics and single-step sweeps |
| `experiments/stiff_gbm` | High-dimensional stiff-drift GBM recovered from `gbm` |
| `experiments/option_pricing` | Previous `stiff_gbm` option-pricing example |
| `experiments/stochastic_volatility` | Seven-model sweep, fixed-step alternative and repeated inference |
| `experiments/kuramoto` | Training, memory/parity sweeps, adjoint ablation and Lyapunov diagnostics |
| `experiments/torus` | Torus memory benchmark |
| `experiments/sphere_latent_nsde` | JAX sphere learning and SRKMK-GeneralShARK baseline |
| `experiments/sphere_latent_sde` | Pinned PyTorch sphere submodule |
| `experiments/ees_dynamical_fitting` | Pinned molecular-dynamics submodule |
| `hpc` | Imperial RCS launchers and resumable runs |

Run module commands from the repository root:

```sh
python -m experiments.convergence_ode.convergence
python -m order_verification.verify
python -m experiments.kuramoto.scripts.run_runtime_parity --help
```

## Data and results

Compact existing measurements and archived manuscript records remain in Git. The new ODE experiment and compositor generate their outputs locally; their generated PDFs/CSV/JSON are not added by this integration. Large datasets and checkpoints remain outside the source history.

The sphere loader accepts UCI HumanActivity data under `experiments/sphere_latent_sde/data_dir`; see the [sphere README](experiments/sphere_latent_nsde/README.md). Stochastic-volatility datasets must be supplied separately; see the [volatility README](experiments/stochastic_volatility/README.md) for filenames and schema. Generate Kuramoto datasets with `python -m experiments.kuramoto.scripts.run_m1`. OU, GBM, torus, stability and convergence scripts generate synthetic inputs.

Published configurations and later calibration recipes are distinct. `kuramoto_runtime_parity.toml` keeps the earlier learning rate and width; HPC jobs explicitly select `kuramoto_hpc_calibrated.toml`. See [the audit](docs/unification.md) before treating new outputs as reproductions of archived numbers.

## Development and releases

Use short-lived branches and PRs targeting `main`. Record each figure/table command, configuration, seeds, source revision and result location in `PAPER_RESULTS.md`. Tag a paper snapshot after its figure/table checklist is verified. Retire old branches after integration is merged and their original tips are archived.

## Citation and licence

See [CITATION.cff](CITATION.cff) and [the paper](https://arxiv.org/abs/2509.20599). Licensed under Apache-2.0; see [LICENSE](LICENSE).
