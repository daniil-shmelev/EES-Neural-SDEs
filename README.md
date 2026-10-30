# Explicit and Effectively Symmetric Schemes for Neural SDEs

Experiments for the NeurIPS 2026 manuscript and SIAM resubmission by Daniil Shmelev, Luke Thompson and Cristopher Salvi. SIAM uses the geometric experiments and order verification; the Euclidean experiments support NeurIPS.

## Experiments

| Directory | Purpose | Manuscript |
|---|---|---|
| [kuramoto](experiments/kuramoto/README.md) | Geometric learning, adjoint diagnostics and memory | Both |
| [sphere_latent_nsde](experiments/sphere_latent_nsde/README.md) | JAX sphere-valued activity classification | Both |
| [convergence_fbm](experiments/convergence_fbm/README.md) | SO(3) rough-driver convergence | Both |
| [neurips/convergence_fbm](neurips/convergence_fbm/README.md) | Euclidean rough-driver convergence | NeurIPS |
| [convergence_ode](experiments/convergence_ode/README.md) | SO(3) ODE forward and recovery orders | SIAM |
| [order_verification](experiments/order_verification/siam/README.md) | Symbolic order conditions and saved SIAM measurements | Both |
| [stability_ode](neurips/stability_ode/README.md), [stability](neurips/stability/README.md) | Deterministic and stochastic stability regions | NeurIPS |
| [ou](neurips/ou/README.md) | High-volatility Ornstein–Uhlenbeck learning | NeurIPS |
| [stiff_gbm](neurips/stiff_gbm/README.md) | Stiff geometric Brownian motion | NeurIPS |
| [stochastic_volatility](neurips/stochastic_volatility/README.md) | Main-text rough Bergomi comparison | NeurIPS |
| [torus_memory](neurips/torus_memory/README.md) | Seven-torus memory benchmark in the introduction | NeurIPS |

NeurIPS-only experiments and historical measurements are under [`neurips/`](neurips/README.md). SIAM experiments remain under `experiments/`.

The [figure/table index](docs/figure_table_index.md) links manuscript results to scripts, saved records and external companion repositories.

## Run

Install `uv`, then run experiments from this repository root. `./run` selects the dependencies and default configuration. Every numerical run gets a fresh directory under its experiment's `results/runs/` (Kuramoto uses `results/unified/runs/`); the command prints its location. Output is shown in the terminal. Existing results are preserved.

| Command | Experiment / default setup |
|---|---|
| `./run kuramoto` | Fixed and adaptive learning sweep; [config.toml](experiments/kuramoto/config.toml) |
| `./run sphere` | Activity-classification [config.toml](experiments/sphere_latent_nsde/config.toml) |
| `./run rough-vol` | Rough Bergomi [sweep](neurips/stochastic_volatility/configs/stoch_vol/sweep.toml) |
| `./run rough-vol --table` | Print the table from saved predictions |
| `./run convergence-ode` | SO(3) forward and recovery convergence |
| `./run convergence-fbm` | Euclidean fBm convergence |
| `./run convergence-fbm --so3` | SO(3) recovery sweep and plot |
| `./run stability-ode` | ODE stability regions |
| `./run stability` | Stochastic stability cross-sections |
| `./run ou` | OU comparison: Reversible Heun and EES(2,5), then plot |
| `./run stiff-gbm` | GBM comparison: Reversible Heun, EES(2,5), EES(2,7), then plot |
| `./run torus-memory` | Seven-torus benchmark |
| `./run order-verification --manuscript /path/to/SIAM/ees_simods.tex` | Exact order and printed-coefficient checks (read-only) |

Use `./run <experiment> --help` for settings and setup locations, or `--dry-run` to inspect the commands without installing dependencies or running computations. Training sweeps accept `--config PATH`; `--output-dir PATH` selects a new output directory. Kuramoto and sphere each have one `config.toml` directly in their experiment directory: `[base]` holds shared settings, `[sweep]` selects seeds, and `[integrators.<name>]` lists adjoints, step counts and stepping modes. Edit that file to change run parameters; `--index N` selects one sweep entry. Other experiments forward additional options to their entrypoint. OU/GBM accept `--method`, `--epochs` and `--seed`; their remaining settings are in the training scripts linked by `--help`.

Every Kuramoto and sphere training run records `peak_gpu_memory_bytes` alongside runtime in `metrics.json`. This is the peak JAX GPU allocation through training and evaluation, including data, model and optimiser memory. CPU runs or backends without allocator statistics record `null`. Each configuration runs in a fresh process so peaks are independent.

No environment activation or separate installation command is needed. The launcher uses the shared `uv.lock`; the underlying modules remain available for development. Saved manuscript results retain their original provenance and are not replaced by new runs.

## Archives

Unused experiments, additional volatility models and earlier source versions are in [`/home/luke/nsde_archive`](../nsde_archive/README.md). Complete predecessor repositories are in [`/home/luke/archive/neural-sde-predecessors-20260924`](../archive/neural-sde-predecessors-20260924/README.md). The [import manifest](docs/consolidation_20260924.json) records the original result sources and checksums.

## Licence

Apache-2.0; see [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
