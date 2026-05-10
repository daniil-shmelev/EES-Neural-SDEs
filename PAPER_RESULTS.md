# Paper Results Cross-Reference

This file maps the figures and tables in the published paper to the scripts and
result files in this repository. It uses PDF-facing descriptions rather than
source-only labels.

Result files committed under `experiments/<name>/results/` back the numbers
and visuals where listed. Some output PDFs are not committed by default; rerun
the relevant script with `--output` to choose an export path.

## Figures

| Paper item | Code path | Repo result(s) |
|---|---|---|
| Intro memory-scaling curve | `python -m experiments.rna.plots` | `experiments/rna/results/fig_scaling_compact_nogrid.pdf`, `rna_hlo_sweep.json`, `rna_benchmark_scaling*.json` |
| EES(2,5) ODE stability domain | `python experiments/stability_ode/stability_regions.py` | `experiments/stability_ode/results/stability_regions_1.pdf` |
| Mean-square SDE stability cross-sections | `python experiments/stability/stability.py` | `experiments/stability/results/stoch_stability.pdf` |
| OU/LSDE training loss | `python experiments/ou/OU.py`, `python experiments/ou/plot_OU.py` | rerun to regenerate `OU_mse.pdf` |
| Kuramoto phase-circle trajectory | `python -m experiments.kuramoto.datasets.visualize` or the data-generation pipeline | `experiments/kuramoto/data/kuramoto_N2_seed0.gif`; export a PDF copy for the paper |
| Kuramoto adjoint memory scaling | `python -m experiments.kuramoto.scripts.plot_memory_sweep` | `experiments/kuramoto/results/memory_sweep_N1000.json`; PDF generated to `experiments/kuramoto/results/fig_kuramoto_memory_scaling.pdf` by default |
| Sphere latent SDE memory scaling | `experiments/sphere_latent_sde/scripts/memory_sweep_integrator.py`, `plot_figures.py` | `experiments/sphere_latent_sde/figures/memory.pdf`, `results/memory_sweep_integrator.csv` |
| EES fBm convergence summary | `python experiments/convergence_fbm/convergence.py` | `experiments/convergence_fbm/results/ees_stochastic_convergence_H{40,50,60}.pdf` |
| CF-EES SO(3) convergence summary | `python -m experiments.convergence_fbm.scripts.so3_reversibility_verify`, `so3_reversibility_plot` | `experiments/convergence_fbm/results/cfees_stochastic_convergence_H{40,50,60}.pdf`, `so3_reversibility_results.json` |
| OU EES(2,5) vs EES(2,7) gradient error | `python experiments/ou/plot_OU.py` and OU grad-error scripts | `experiments/ou/results/grad_error/single_step/OU_grad_error.pdf` |
| Stiff GBM loss and gradient MSE | `python experiments/stiff_gbm/GBM.py`, `plot_GBM.py` | rerun to regenerate |
| Molecular-dynamics water initial condition | `experiments/ees_dynamical_fitting/IR-fitting/` | submodule assets; figure exported separately |
| Molecular-dynamics training MSE | `experiments/ees_dynamical_fitting/IR-fitting/benchmark_ees.py` | generated from submodule benchmark |

## Tables

| Paper item | Code path | Repo result(s) |
|---|---|---|
| OU/LSDE solver comparison | `python experiments/ou/OU.py` | rerun to regenerate metrics |
| Rough Bergomi runtime/MSE | `python -m experiments.stochastic_volatility.experiment.train ...` | generated under `experiments/stochastic_volatility/results/` |
| Kuramoto runtime/error comparison | `python -m experiments.kuramoto.experiment.train_kuramoto`, runtime-parity scripts | `experiments/kuramoto/results/runtime_parity_N1000_n50/**/metrics.json`, `runtime_parity_summary.json` |
| HumanActivity accuracy at compute parity | `experiments/sphere_latent_sde/scripts/compute_parity_sweep.py` | `experiments/sphere_latent_sde/results/compute_parity.csv` |
| Per-step compute/memory counts | derivation in the paper | no script |
| Symbolic alpha values on planar trees | `python -m experiments.order_verification.scripts.scheme_a_algebraic_verify`, `scheme_a_tree_algebraic_verify`, `mkw_verify_cf` | deterministic console output |
| Stiff GBM metrics | `python experiments/stiff_gbm/GBM.py` | rerun to regenerate |
| Seven-model stochastic-volatility sweep | `python -m experiments.stochastic_volatility.experiment.train experiments/stochastic_volatility/configs/stoch_vol/sweep.toml` | generated under `experiments/stochastic_volatility/results/` |
| New/patched software packages | `pyproject.toml`, `.gitmodules` | source metadata |
| Stochastic-volatility model parameters | config/model definitions | `experiments/stochastic_volatility/experiment/config.py`, `dataset.py` |
| Kuramoto pilot diagnostics | `python -m experiments.kuramoto.scripts.pilot_adjoint_parity` | `experiments/kuramoto/results/pilot/**` |
| Kuramoto memory-sweep raw data | `python -m experiments.kuramoto.scripts.memory_sweep` | `experiments/kuramoto/results/memory_sweep_N1000.json`, `memory_sweep_cells_N1000/*.json` |
| Sphere memory raw data | submodule memory-sweep scripts | `experiments/sphere_latent_sde/results/memory_sweep*.csv` |
| RNA torus memory raw data | `python -m experiments.rna.scripts.run_rna_hlo_sweep` | `experiments/rna/results/rna_hlo_sweep.json`, `rna_benchmark_scaling*.json` |

## Supplementary/Legacy Outputs

Some older RNA torus and per-H convergence outputs remain committed as
historical regression checks. They live under `experiments/rna/results/` and
`experiments/convergence_fbm/results/`.
