# High-dimensional stiff GBM

The 25-dimensional drift-matrix experiment is recovered from `gbm:cdb009c` (`examples/GBM_2.py`). The former option-pricing code is preserved in `experiments/option_pricing`.

```sh
uv pip install -e ".[stiff-gbm]"
python -m experiments.stiff_gbm.GBM --method ees25
python -m experiments.stiff_gbm.plot_GBM
python -m experiments.stiff_gbm.plot_gradients
```

Supported methods are `reversible_heun`, `ees25`, `ees27`, `mcf_euler` and `mcf_midpoint`. Repeat training for each method before comparing curves. Outputs default to `experiments/stiff_gbm/results/<method>/`; `--output-dir` selects another directory. `--epochs`, `--num-samples`, `--batch-size` and `--seed` permit smaller validation runs.

The branch default has eigenvalues linearly spaced from -20 to 0. The arXiv v2 appendix instead specifies -20(1+i/25), for i=0,...,24. Select that stated drift with `--drift-min -20 --drift-max -39.2`. Neither configuration is presented as a verified reproduction of the archived table.

The unified script seeds data generation, uses the same Brownian path for direct/adjoint gradient comparisons, and clears reference gradients before the training backward pass. These fix defects in the old branch and change generated results. Original source remains at commit `cdb009c92a0e0504dd7ed5308419972f1bd64c46`.
