# NeurIPS stochastic-volatility records

The original 28 runs (seven models, four solvers, seed 0, 504 generator evaluations per trajectory) were imported from `EES_Neural_SDE2/results` on 24 September 2026. The four rough Bergomi runs for the main-text table remain here. The 24 appendix runs and later 132-evaluation runs are now in `/home/luke/nsde_archive/volatility_main_text_20260924/experiments/stochastic_volatility/results/`, with their configurations, checkpoints, predictions, metrics and histories.

From the repository root, run `python -m experiments.stochastic_volatility.scripts.paper_table`. The original numerical conventions are retained: the displayed terminal MSE is 100 times the raw MSE, and runtime is training plus inference time. The script does not produce the manuscript's uncertainty intervals; these records each contain only one seed.

The consolidation check found one existing numerical discrepancy: rough Bergomi with MCF Midpoint gives a displayed terminal MSE of 7.82 from the saved predictions, while the NeurIPS table prints 7.81. All 28 rounded runtimes and the other 27 central errors match. The manuscript was not changed.
