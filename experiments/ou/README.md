# OU experiments

Install with `uv pip install -e ".[ou]"` and run module commands from the repository root.

`train_paper.py` preserves the fixed-NFE training recipe from the `gbm` branch: a time-dependent drift, terminal time 10, and method-specific step sizes. Outputs are under `experiments/ou/results/paper/<method>/`. Run `python -m experiments.ou.train_paper --method ees25`, repeating for `reversible_heun`, `mcf_euler`, `mcf_midpoint` and optionally `ees27`. `--epochs`, `--num-samples`, `--batch-size`, `--seed` and `--output-dir` support small checks. `python -m experiments.ou.plot_paper_loss` plots training loss; `python -m experiments.ou.plot_paper_gradients` plots the retained gradient-MSE diagnostic. Data generation is seeded before sampling in the unified script.

`OU.py` retains the local constant-mean OU recipe and relative-gradient diagnostics. Run `python -m experiments.ou.OU` and `python -m experiments.ou.plot_OU`; its method and training settings are in the script configuration, and it retains the original `plots/<method>/` outputs. Its `grad_error.pickle` stores relative-gradient errors and epoch indices, whereas `train_paper.py` writes `grad_err.pickle` containing gradient MSE. These recipes and diagnostics are not interchangeable.

The appendix single-step comparison remains in `scripts/single_step_grad_error.py` with saved records under `results/grad_error/single_step/`. See each script's `--help` for the fitting grid and output controls.
