# Manuscript experiments and their sources

Scope: the active text of `neurips2026/ees_neurips_manuscript.tex`, `SIAM/ees_simods.tex` and the SIAM supplement in `/home/luke/68b70b0040ed79e6aef597a4`. Commented-out RNA prediction experiments and the SPD covariance prototype are outside this scope. Paths below are relative to the experiment repository unless absolute.

| Manuscript item | Manuscript | Code and records |
|---|---|---|
| `fig:ees25_stability`, `fig:ees_stoch_stability` | NeurIPS | `neurips/stability_ode/stability_regions.py`, `neurips/stability/stability.py`; saved plots in their `results/` directories. Older stability variants remain in the predecessor archive. Exact source-to-manuscript PDF equality is not established for the current plots. |
| `fig:lsde`, `table:lsde` | NeurIPS | `neurips/ou/OU.py`, `neurips/ou/plot_OU.py`, `neurips/ou/results/` and `plots/`. |
| `fig:gbm`, `fig:gbm_grad`, `table:gbm` | NeurIPS | `neurips/stiff_gbm/GBM.py`, `neurips/stiff_gbm/plot_GBM.py`. |
| `table:rough_bergomi`, `tab:further_stoch_vol` | NeurIPS | `neurips/stochastic_volatility/scripts/paper_table.py`; four rough Bergomi runs in `results/neurips_20260421/`; configuration `configs/stoch_vol/sweep.toml` (504 evaluations, seed 0). The 24 appendix runs and original full table are preserved in `/home/luke/nsde_archive/volatility_main_text_20260924/`. These records reproduce central errors and runtimes, not the printed uncertainty intervals. |
| `tab:rough-volatility-parameters` | NeurIPS | Dataset configuration description in the manuscript; rough Bergomi data in `neurips/stochastic_volatility/data/`; the other six datasets are in `/home/luke/nsde_archive/volatility_main_text_20260924/`. |
| `fig:torus_scaling`, `tab:torus_intro_memory_data` | NeurIPS | `neurips/torus_memory/benchmark.py`, `benchmark_single.py`, `plot.py`, `results/rna_benchmark_scaling.json` and `fig_scaling_compact_nogrid.pdf`. Extracted from the RNA implementation; unused prediction experiments and earlier benchmark variants are in `/home/luke/nsde_archive/experiments/rna/`. |
| `tab:alpha-cf-ees25-x`, forward and recovery order certificates | Both | `experiments/order_verification/scripts/`; SIAM-specific checks in `experiments/order_verification/siam/scripts/verify_cfees.py`. |
| `fig:ode-convergence`, `tab:ode-orders` | SIAM | `experiments/convergence_ode/convergence.py`, `results/ode_convergence.csv`, `ode_validation.json`, `cfees_ode_convergence.pdf`. |
| Rough-driver convergence figures, including `fig:cfees_convergence_12` | Both | Geometric manuscript PDFs in `experiments/convergence_fbm/results/`; original arrays and invocation metadata unavailable. Euclidean code and PDFs are in `neurips/convergence_fbm/`. `scripts/so3_reversibility_verify.py` and its JSON are a separate geometric recovery diagnostic, not the source data for the 12-panel figure. The shared compositor reads both result directories. |
| `fig:kuramoto_trajectory`, `tab:kuramoto_quality` | Both, with different learning tables | SIAM learning: `experiments/kuramoto/results/unified/reversible_second_order_verified_20260925/comparison.json` and the two completed run directories. NeurIPS learning: `neurips/kuramoto/results/kuramoto_runtime_parity.json` and `runtime_parity_N1000_n50/`. The trajectory PDF is retained in the manuscript repository; no byte-identical source PDF was found here. |
| `fig:kuramoto_memory_scaling`, `tab:kuramoto_memory_data`; SIAM `fig:learning-memory` | Both | Historical records in `experiments/kuramoto/results/memory_sweep_N1000.json` and the SIAM record copy. These used the old global MLP. `experiments/kuramoto/scripts/run_memory_sweep.py` now measures the same indexed mean-field model as training; its new results must not be mixed with those historical records. |
| `tab:sphere_parity` | Both | `experiments/sphere_latent_nsde/train_activity.py`, `configs/activity/sweep.toml`, `results/activity__*/`; SIAM snapshots in `experiments/order_verification/siam/experiment_records/sphere_training/`. |
| `fig:sphere_memory`, `tab:sphere_memory_data` | NeurIPS | Original implementation in `/home/luke/EES-LatentSDEonHS`; saved CSV in `neurips/sphere_memory/results/memory_sweep_integrator.csv`. This differs from the JAX classification implementation. |
| Molecular dynamics experiment, `fig:md_mse` | NeurIPS | Companion repository `/home/luke/ees-dynamical-fitting`. |
| `tab:exp_counts` | Both | Analytical operation counts; no measurement script. |

## Consolidation and provenance

The imports and their SHA-256 checksums are in [consolidation_20260924.json](consolidation_20260924.json). The prior index is preserved in `/home/luke/archive/neural-sde-predecessors-20260924/canonical_before_consolidation/docs/figure_table_index.md`. No training was rerun during consolidation, and no manuscript numbers were changed.

The consolidation check found one existing numerical discrepancy: rough Bergomi with MCF Midpoint gives a displayed terminal MSE of 7.82 from the saved predictions, while the NeurIPS table prints 7.81. All 28 rounded runtimes and the other 27 central errors match. The manuscript was not changed.

Unused experiments and diagnostic records were moved to `/home/luke/nsde_archive`; its `manifest.json` records original paths and checksums. Active geometric benchmarks now import Georax's public geometries. The torus helper module contains only angle wrapping. Pre-migration source files, including the removed custom geometry and adjoint implementations, are preserved in `/home/luke/nsde_archive/georax8cbae32_migration/`.

## Layout cleanup (2026-10-27)

NeurIPS-only experiments and historical measurements moved to `neurips/`. Superseded Kuramoto trial runs, the unused adjoint-ablation record and one unused sphere checkpoint-full run were deleted after explicit user approval. The current SIAM Kuramoto comparison, historical memory measurements, all eleven sphere runs contributing to the SIAM table, and their SIAM record snapshots were preserved. The [cleanup manifest](cleanup_20261027.json) records moves and hashes of deleted files. The learning-table checker now reads the verified second-order Kuramoto comparison rather than the old fixed-step records.
