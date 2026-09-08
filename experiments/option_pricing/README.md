# Option pricing

The European-call-price GBM example formerly located in `experiments/stiff_gbm` is retained here. It is distinct from the high-dimensional stiff-drift system in the paper appendix.

```sh
uv pip install -e ".[option-pricing]"
python -m experiments.option_pricing.GBM
python -m experiments.option_pricing.plot_GBM
```

Select the method in the script configuration. Outputs retain the original `plots/options_<method>/` layout.
