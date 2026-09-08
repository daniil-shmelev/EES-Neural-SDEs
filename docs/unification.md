# Paper repository unification

The integration starts from `main` at `ca8f3398b8001d9339e785971601be5f46c8252c`. The audit compares source trees as well as ancestry because earlier PRs were squash-merged.

## Branch decisions

| Source | Audited tip | Decision |
|---|---|---|
| `main` | `ca8f339` | Baseline; already contains the CF-EES cleanup and paper layout. |
| `cfees` | `ca1569a` | Merge history and six ODE/compositor files. Its remaining tree is identical to `main`. |
| `add-ode-convergence` | `4a802f7` | Included through `cfees`; generated results remain excluded. |
| `hpc-launch-scripts` | `e1a72ff` | Merge launchers, resumable data/training and diagnostic drivers; separate calibrated TOML. |
| `kuramoto` | `97e56a8` | Tree exactly matches squash commit `dabb646`, already in `cfees`. Record ancestry without restoring the old layout. |
| `gbm` | `cdb009c` | Port high-dimensional GBM, fixed-NFE OU training and gradient plots. Preserve option pricing separately. Unrelated SDE-GAN code and Windows wrappers remain in original history. |
| Local checkout | `62d5e91` plus changes | Preserve shared plotting, ODE/SIAM verification, OU gradients, Kuramoto diagnostics, sphere SRKMK/checkpoint support, fixed-step volatility and repeated inference. |

The original working checkout is not reset or cleaned. Large local datasets/checkpoints, duplicate plots, `results_old/` and the scratch `pyproject_luke_new_georax.toml` are not copied into the source PR. Relevant dependency declarations are consolidated into `pyproject.toml`. Compact Kuramoto diagnostic measurements and SIAM manuscript records retain their original provenance.

## Scientific configurations

- Geometry adapters for sphere, Kuramoto and torus now implement the pinned georax chart API. Neural vector fields are passed through differentiable solver arguments; regression tests check finite, nonzero parameter gradients.
- Deterministic ODE work remains in `experiments/convergence_ode`. ODE and fBm share `experiments.convergence_plotting`; the compositor stays in `experiments/convergence_fbm/scripts`.
- The exact ordered-forest verifier and frozen learning records are under `order_verification/siam`. They describe later SIAM manuscript work, separately from the arXiv v2 figure index.
- `kuramoto_runtime_parity.toml` retains learning rate 1e-3 and width 128. `kuramoto_hpc_calibrated.toml` retains the later 3e-4/256 settings; HPC jobs explicitly select it.
- `experiments/ou/train_paper.py` preserves the fixed-NFE, time-dependent-drift OU experiment from `gbm`. `OU.py` retains the local constant-mean OU/relative-gradient diagnostics. These use distinct data recipes and output locations.
- High-dimensional GBM's original [-20,0] eigenvalues differ from the appendix formula. CLI overrides expose the stated paper range. The shared-noise and gradient-clearing fixes change new outputs; historical numbers are not silently relabelled.
- `stability/sde.py` retains RK3/RK4 comparisons. `stability/sde_reversible_heun.py` preserves the revised figure with different filenames. Its red overlay is an ODE stability-set intersection, not a stochastic stability boundary.
- Sphere memory figures are associated with the PyTorch submodule. Later SIAM training records contain JAX configurations, including SRKMK-GeneralShARK. Some archived sphere action/software metadata remains incomplete.
- Local sphere recipes are retained in `configs/local_archive` with portable data paths. SRKMK budgets there count drift calls (two per step); this is not automatically the paper's total network-evaluation accounting.
- Volatility defaults remain fixed-NFE. The fixed-step TOML and repeated-inference script preserve the alternative local protocol.

## Release conditions

Validation commands and outcomes are recorded in `docs/unification-validation.md`. A paper release additionally needs external volatility datasets and training hardware, confirmed sphere action/environment metadata, and reconciliation of the GBM drift and sphere NFE discrepancies. This integration does not claim that every published number has been regenerated. Do not create an `arxiv-v2` reproduction tag until those items are resolved.

After review and merge into `main`, archive original branch tips before deleting completed branches. Preserve the SDE-GAN history through an archive tag if `gbm` is retired. Future work should target `main` through short-lived PRs.
