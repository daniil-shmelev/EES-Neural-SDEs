# Manuscript Figure/Table Cross-Reference

This is the repo-side map from the active manuscript labels in
the Overleaf project at `EES_Neural_SDEs_Overleaf/neurips2026/ees_neurips_manuscript.tex`
to the code and result files that support them.

Result files committed under `experiments/<name>/results/` back the numbers and visuals where listed. Some manuscript-ready PDFs live only in the Overleaf `figures/` directory after plotting; rerun the script with `--output` to write directly there.

## Figures

| Label | What it is | Code path | Repo result(s) |
|---|---|---|---|
| `fig:torus_scaling` | Intro memory scaling curve | `python -m experiments.rna.plots` | `experiments/rna/results/fig_scaling_compact_nogrid.pdf`, `rna_hlo_sweep.json`, `rna_benchmark_scaling*.json` |
| `fig:ees25_stability` | EES(2,5) ODE stability domain | `python experiments/stability_ode/stability_regions.py` | `experiments/stability_ode/results/stability_regions_1.pdf` |
| `fig:ees_stoch_stability` | Mean-square SDE stability cross-sections | `python experiments/stability/stability.py` | `experiments/stability/results/stoch_stability.pdf` |
| `fig:lsde` | OU/LSDE training loss | `python experiments/ou/OU.py`, `python experiments/ou/plot_OU.py` | rerun to regenerate `OU_mse.pdf` |
| `fig:kuramoto_trajectory` | Kuramoto phase-circle trajectory | `python -m experiments.kuramoto.datasets.visualize` or M1 pipeline | `experiments/kuramoto/data/kuramoto_N2_seed0.gif`; manuscript uses exported PDF |
| `fig:kuramoto_memory_scaling` | Kuramoto adjoint memory scaling | `python -m experiments.kuramoto.scripts.plot_memory_sweep` | `experiments/kuramoto/results/memory_sweep_N1000.json`; PDF generated to `experiments/kuramoto/results/fig_kuramoto_memory_scaling.pdf` by default |
| `fig:sphere_memory` | Sphere latent SDE memory scaling | `experiments/sphere_latent_sde/scripts/memory_sweep_integrator.py`, `plot_figures.py` | `experiments/sphere_latent_sde/figures/memory.pdf`, `results/memory_sweep_integrator.csv` |
| `fig:ees_convergence_12` | EES fBm convergence summary | `python experiments/convergence_fbm/convergence.py` | `experiments/convergence_fbm/results/ees_stochastic_convergence_H{40,50,60}.pdf` |
| `fig:cfees_convergence_12` | CF-EES SO(3) convergence summary | `python -m experiments.convergence_fbm.scripts.so3_reversibility_verify`, `so3_reversibility_plot` | `experiments/convergence_fbm/results/cfees_stochastic_convergence_H{40,50,60}.pdf`, `so3_reversibility_results.json` |
| `fig:ees25_vs_27_grad_error` | OU EES(2,5) vs EES(2,7) gradient error | `python experiments/ou/plot_OU.py` and OU grad-error scripts | `experiments/ou/results/grad_error/single_step/OU_grad_error.pdf` |
| `fig:gbm`, `fig:gbm_grad` | Stiff GBM loss and gradient MSE | `python experiments/stiff_gbm/GBM.py`, `plot_GBM.py` | rerun to regenerate |
| `fig:md_water_init` | Molecular-dynamics water initial condition | `experiments/ees_dynamical_fitting/IR-fitting/` | submodule assets; manuscript figure exported separately |
| `fig:md_mse` | Molecular-dynamics training MSE | `experiments/ees_dynamical_fitting/IR-fitting/benchmark_ees.py` | generated from submodule benchmark |

## Tables

| Label | What it is | Code path | Repo result(s) |
|---|---|---|---|
| `table:lsde` | OU/LSDE solver comparison | `python experiments/ou/OU.py` | rerun to regenerate metrics |
| `table:rough_bergomi` | Rough Bergomi runtime/MSE | `python -m experiments.stochastic_volatility.experiment.train ...` | generated under `experiments/stochastic_volatility/results/` |
| `tab:kuramoto_quality` | Kuramoto runtime/error comparison | `python -m experiments.kuramoto.experiment.train_kuramoto`, runtime-parity scripts | `experiments/kuramoto/results/runtime_parity_N1000_n50/**/metrics.json`, `runtime_parity_summary.json` |
| `tab:sphere_parity` | HumanActivity accuracy at compute parity | `experiments/sphere_latent_sde/scripts/compute_parity_sweep.py` | `experiments/sphere_latent_sde/results/compute_parity.csv` |
| `tab:exp_counts` | Per-step compute/memory counts | derivation in manuscript | no script |
| `tab:alpha-cf-ees25-x` | Symbolic alpha values on planar trees | `python -m experiments.order_verification.scripts.scheme_a_algebraic_verify`, `scheme_a_tree_algebraic_verify`, `mkw_verify_cf` | deterministic console output |
| `table:gbm` | Stiff GBM metrics | `python experiments/stiff_gbm/GBM.py` | rerun to regenerate |
| `tab:further_stoch_vol` | Seven-model stochastic-volatility sweep | `python -m experiments.stochastic_volatility.experiment.train experiments/stochastic_volatility/configs/stoch_vol/sweep.toml` | generated under `experiments/stochastic_volatility/results/` |
| `tab:new_packages` | New/patched software packages | `pyproject.toml`, `.gitmodules` | source metadata |
| `tab:rough-volatility-parameters` | Stochastic-volatility model parameters | config/model definitions | `experiments/stochastic_volatility/experiment/config.py`, `dataset.py` |
| `tab:kuramoto_pilot` | Kuramoto pilot diagnostics | `python -m experiments.kuramoto.scripts.pilot_adjoint_parity` | `experiments/kuramoto/results/pilot/**` |
| `tab:kuramoto_memory_data` | Kuramoto memory sweep raw data | `python -m experiments.kuramoto.scripts.memory_sweep` | `experiments/kuramoto/results/memory_sweep_N1000.json`, `memory_sweep_cells_N1000/*.json` |
| `tab:sphere_memory_data` | Sphere memory raw data | submodule memory sweep scripts | `experiments/sphere_latent_sde/results/memory_sweep*.csv` |
| `tab:torus_intro_memory_data` | RNA torus memory raw data | `python -m experiments.rna.scripts.run_rna_hlo_sweep` | `experiments/rna/results/rna_hlo_sweep.json`, `rna_benchmark_scaling*.json` |

## Legacy/Commented Labels

The manuscript still contains commented-out labels for the older RNA torus appendix tables (`tab:torus_scaling`, `tab:torus_runtime_error`, `tab:torus_adjoint_retrain`, `tab:torus_mae`) and the older per-H convergence figures (`fig:onvergence_4`, `fig:convergence_5`, `fig:convergence_6`, `fig:cf_convergence_4`, `fig:cf_convergence_5`, `fig:cf_convergence_6`). The supporting code/results are still in `experiments/rna/` and `experiments/convergence_fbm/`, but the active manuscript labels are the summary labels listed above.
