# Paper Results Cross-Reference

This index maps the arXiv v2 figures and tables to retained code and archived records. It is a provenance guide, not a claim that the unified environment has regenerated every number. Exact commands and unresolved differences are recorded below and in [the integration audit](docs/unification.md).

Result files committed under each script's `results/` directory back the numbers and visuals where listed. Some output PDFs are not committed by default; rerun the relevant script with `--output` to choose an export path.

## Figures

| Paper item | Code path | Repo result(s) |
|---|---|---|
| Intro memory-scaling curve | `python -m experiments.torus.plots` | `experiments/torus/results/fig_scaling_compact_nogrid.pdf`, `torus_memory_scaling.json` |
| EES(2,5) and EES(2,7) ODE stability domains | `python stability/ode.py` | `stability/results/stability_regions_1.pdf` |
| Mean-square SDE stability cross-sections | `python stability/sde.py` | `stability/results/stoch_stability.pdf` |
| OU/LSDE training loss | `python -m experiments.ou.train_paper --method ees25`, `python -m experiments.ou.plot_paper_loss` | rerun to regenerate `OU_mse.pdf` |
| Kuramoto phase-circle trajectory | `python -m experiments.kuramoto.datasets.visualize` or the data-generation pipeline | `experiments/kuramoto/data/kuramoto_N2_seed0.gif` |
| Kuramoto adjoint memory scaling | `python -m experiments.kuramoto.scripts.plot_memory_sweep` | `experiments/kuramoto/results/memory_sweep_N1000.json`; PDF generated to `experiments/kuramoto/results/fig_kuramoto_memory_scaling.pdf` by default |
| Sphere latent SDE memory scaling | `experiments/sphere_latent_sde/scripts/memory_sweep_integrator.py`, `plot_figures.py` | `experiments/sphere_latent_sde/figures/memory.pdf`, `results/memory_sweep_integrator.csv` |
| EES fBm convergence summary | `python experiments/convergence_fbm/convergence.py` | `experiments/convergence_fbm/results/ees_stochastic_convergence_H{40,50,60}.pdf` |
| CF-EES SO(3) convergence summary | `python -m experiments.convergence_fbm.scripts.so3_reversibility_verify`, `so3_reversibility_plot` | `experiments/convergence_fbm/results/cfees_stochastic_convergence_H{40,50,60}.pdf`, `so3_reversibility_results.json` |
| OU EES(2,5) vs EES(2,7) gradient error | `python -m experiments.ou.scripts.single_step_grad_error`, `python -m experiments.ou.scripts.plot_grad_error` | `experiments/ou/results/grad_error/single_step/OU_grad_error.pdf` |
| Stiff GBM loss and gradient MSE | `python -m experiments.stiff_gbm.GBM --method ees25`, `python -m experiments.stiff_gbm.plot_GBM`, `python -m experiments.stiff_gbm.plot_gradients` | rerun to regenerate |
| Molecular-dynamics water initial condition | `experiments/ees_dynamical_fitting/IR-fitting/` | submodule assets; figure exported separately |
| Molecular-dynamics training MSE | `experiments/ees_dynamical_fitting/IR-fitting/benchmark_ees.py` | generated from submodule benchmark |

## Tables

| Paper item | Code path | Repo result(s) |
|---|---|---|
| OU/LSDE solver comparison | `python -m experiments.ou.train_paper --method ees25` | rerun to regenerate metrics |
| Rough Bergomi runtime/MSE | `python -m experiments.stochastic_volatility.experiment.train ...` | generated under `experiments/stochastic_volatility/results/` |
| Kuramoto runtime/error comparison | `python -m experiments.kuramoto.experiment.train_kuramoto`, runtime-parity scripts | `experiments/kuramoto/results/runtime_parity_N1000_n50/**/metrics.json`, `runtime_parity_summary.json` |
| HumanActivity accuracy at compute parity | `experiments/sphere_latent_sde/scripts/compute_parity_sweep.py` | `experiments/sphere_latent_sde/results/compute_parity.csv` |
| Per-step compute/memory counts | derivation in the paper | no script |
| CFEES Hopf-algebra order conditions | `python -m order_verification.verify` | deterministic console output |
| Stiff GBM metrics | `python -m experiments.stiff_gbm.GBM --method ees25` | rerun to regenerate |
| Seven-model stochastic-volatility sweep | `python -m experiments.stochastic_volatility.experiment.train experiments/stochastic_volatility/configs/stoch_vol/sweep.toml` | generated under `experiments/stochastic_volatility/results/` |
| New/patched software packages | `pyproject.toml`, `.gitmodules` | source metadata |
| Stochastic-volatility model parameters | config/model definitions | `experiments/stochastic_volatility/experiment/config.py`, `dataset.py` |
| Kuramoto pilot diagnostics | `python -m experiments.kuramoto.scripts.pilot_adjoint_parity` | `experiments/kuramoto/results/pilot/**` |
| Kuramoto memory-sweep raw data | `python -m experiments.kuramoto.scripts.run_memory_sweep` | `experiments/kuramoto/results/memory_sweep_N1000.json`, `memory_sweep_cells_N1000/*.json` |
| Sphere memory raw data | submodule memory-sweep scripts | `experiments/sphere_latent_sde/results/memory_sweep*.csv` |
| Torus memory raw data | `python -m experiments.torus.scripts.run_memory_sweep` | `experiments/torus/results/torus_memory_scaling.json` |

## New ODE and later manuscript work

| Item | Command/configuration | Provenance and outputs |
|---|---|---|
| Deterministic SO(3) convergence | `python -m experiments.convergence_ode.convergence` | Default 2–128 steps, float64; PDF, CSV and versioned validation JSON generated in `experiments/convergence_ode/results/`, excluded from Git |
| 12/24-panel rough-driver figures | `python -m experiments.convergence_fbm.scripts.compose_12up_pages --kind all` | Composes existing per-Hurst PDFs; needs Ghostscript and pdflatex for vector output; generated summaries excluded from this PR |
| Exact ordered-forest/embedded verification | `python order_verification/siam/scripts/verify_cfees.py --manuscript /path/to/SIAM/ees_simods.tex` | Preserved SIAM verifier and archived `verification.txt`; requires the external manuscript and SymPy |
| SIAM learning-table records | `python order_verification/siam/scripts/summarize_records.py --manuscript /path/to/SIAM/ees_simods.tex` | Frozen `order_verification/siam/experiment_records`, including source checksums; reads/checks the manuscript unless `--update` is explicitly used |
| Kuramoto same-noise Lyapunov estimates | `python -m experiments.kuramoto.scripts.compute_same_noise_lyapunov --help` | Saved `experiments/kuramoto/results/same_noise_lyapunov.json` contains the grid, seeds and per-realisation estimates |
| Kuramoto solver/adjoint ablation | `python -m experiments.kuramoto.scripts.run_solver_adjoint_ablation --help` | Local compact records retained in `results/solver_adjoint_ablation_N1000`; uses the original parity TOML |
| Revised stability overlay | `python -m stability.sde_reversible_heun` | Separate generated `stoch_stability_reversible_heun.*`; preserves original `stability/sde.py` figure |
| Volatility fixed-step alternative | `python -m experiments.stochastic_volatility.experiment.train experiments/stochastic_volatility/configs/stoch_vol/rough_bergomi_fixed_stepsize.toml` | Different integration protocol from the original fixed-NFE table |
| Volatility repeated inference | `python -m experiments.stochastic_volatility.experiment.repeated_inference --help` | Requires the saved checkpoint/configuration pairs and external datasets |

## Configuration and release manifest

| Experiment | Retained configuration / seeds | Reproduction status |
|---|---|---|
| ODE convergence | CLI defaults, deterministic, float64 | Version metadata written on each run; no training data required |
| Kuramoto published parity | `experiments/kuramoto/configs/kuramoto_runtime_parity.toml`; seeds 0,1,2 in archived run configs | Retains width 128 and learning rate 1e-3 |
| Kuramoto later HPC calibration | `kuramoto_hpc_calibrated.toml`; seed arrays in `hpc/*.pbs` | Width 256 / learning rate 3e-4; distinct output directories |
| OU fixed-NFE | `train_paper.py` defaults; seed 42 | Original `gbm` recipe retained; data generation is now seeded before sampling |
| High-dimensional GBM | `GBM.py` defaults; seed 42 | Original drift range differs from the appendix; use documented overrides; corrected gradient diagnostic requires new measurements |
| Sphere | Each archived run's `config.toml`, PyTorch submodule commit, SIAM `sources.json` | Exact JAX action/environment and total-network NFE provenance need confirmation |
| Stochastic volatility | `configs/stoch_vol/sweep.toml`, saved per-run configs | Requires seven external datasets and training checkpoints |

Solver pins in `pyproject.toml` describe the unified executable environment. Submodule pins are Git links in the repository tree. Neither replaces historical per-run version metadata. Full training tables, runtime claims and missing external datasets must be verified before an arXiv reproduction release is tagged.
