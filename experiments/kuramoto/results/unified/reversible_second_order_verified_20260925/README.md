# Reversible Kuramoto training: second-order model

The completed full-data comparison keeps CF-EES on the ordinary `ReversibleAdjoint`, with no trajectory checkpoints or inverse-correction loop. The fix is a shared model constraint: exact phase kinematics `dtheta/dt = omega`, neural phase-dependent forcing, and learned linear damping `2*sigmoid(damping_logit)` in `(0, 2)`. The cached dataset has true damping 1. The same constraint is applied to CG2, and the original diffusion, architecture widths, energy-score loss, data and optimizer settings are retained. The existing network layout is preserved, with unused velocity input slots set to zero and only the acceleration output used.

| Method | Final test loss | Best validation loss | Complete invocation |
| --- | ---: | ---: | ---: |
| CG2, full checkpoints | 361.1944 | 363.8247 | 11.51 min |
| Adaptive CF-EES, reversible | 361.0294 | 363.6376 | 6.68 min |

CF-EES was 1.72 times faster, with a -0.046% difference in final test loss. These are measured single-seed results, not an average over independent repeats. Both runs completed 15 epochs at N=1000 with 1500/300/300 training/validation/test trajectories, batch size 32 and four forecast samples per example. Timing includes startup, compilation, evaluation and artifact generation, using an external monotonic clock. No GPU training runs overlapped.

## Reproduce

`./run kuramoto --adaptive --seed 0` selects the tested reversible settings: tolerance 0.03, maximum step 0.5 and minimum step 1e-4. The shared TOML selects the second-order drift for all methods. Exact commands and metrics are in [comparison.json](comparison.json); dependency and source hashes are in [provenance.json](provenance.json), with the existing Diffrax batched-replay patch in [diffrax_worktree.patch](diffrax_worktree.patch).

- [Reversible CF-EES history](cfees25_reversible/history.json), [metrics](cfees25_reversible/metrics.json), [best model](cfees25_reversible/model_best.eqx), [final model](cfees25_reversible/model_final.eqx), [log](cfees25_reversible.log).
- [CG2 history](cg2_full/history.json), [metrics](cg2_full/metrics.json), [best model](cg2_full/model_best.eqx), [final model](cg2_full/model_final.eqx), [log](cg2_full.log).
- [Numerical and artifact checks](validation.json), [machine-readable summary](summary.json), [comparison plot](comparison.png), [PDF plot](comparison.pdf).

## Diagnosis and short iterations

The original unrestricted drift could learn strongly contracting dynamics. Jacobian probes at states from the cached trajectory found negative drift eigenvalues around -3 to -3.5 per second; this is a local diagnostic rather than a measured global Lipschitz bound. Forward stability does not ensure accurate backward reconstruction over the five-second horizon. At identical saved failing weights and Brownian paths, the original reversible gradients had relative errors hundreds to thousands of times the checkpointed reference. The superseded failing-run artifacts were deleted during the approved cleanup; their paths and hashes are recorded in `../../../../../docs/cleanup_20261027.json`. The final comparison and its diagnostics remain here.

The initial small-width experiment did not reproduce the failure and was not treated as evidence of a fix. A short recovery test on the actual failing checkpoint did reproduce it: after 200 small-batch updates, CG2 reached validation loss 421.21 while uncorrected reversible EES reached 456.74. Five local inverse-residual corrections improved selected frozen gradients but still failed the training recovery test (loss 461.79 and large gradient errors), so that approach was removed from production.

The second-order model was tested with the full N=1000 network, 128 training examples, eight validation examples, two examples per update and two forecast samples, for 600 updates. Matched CG2 and conservative reversible EES achieved validation losses 416.80 and 416.70. The faster adaptive settings achieved 416.57 in 90.6 seconds against CG2's 132.9 seconds, with a same-weights relative gradient error of 0.00050. Those two-sample diagnostic losses are not interchangeable with the four-sample full-data losses above. Only after these short tests was the full-data pair run once.

The reusable short check is `python -m experiments.kuramoto.scripts.check_reversible_training --output-dir NEW_DIRECTORY --steps 600`, using the same patched Diffrax environment as training. `--modes ees` runs only the reversible candidate. The second-order restriction bounds direct velocity damping and removes velocity-dependent neural forcing; it is not a universal bound on all neural-field derivatives. The final saved-model gradient checks document the tested operating regime: relative errors were 0.0399–0.0423% on three fixed noise seeds, with gradient cosines above 0.9999999. Exploratory gradient and Jacobian measurements are retained in [diagnostics.json](diagnostics.json).

## Validation scope

The targeted suite passed 46 tests and 11 subtests, including phase kinematics, bounded damping, velocity-independent forcing, adaptive gradient comparisons, mesh replay, evaluation masks, checkpoint handling and launcher defaults. The final artifacts are checked for finite values, and reversible gradients at the saved best model are compared with checkpointed gradients at identical weights and noise. Reference gradients are used only for validation, never for EES training. New isolated memory sweeps and additional training seeds have not been run.
