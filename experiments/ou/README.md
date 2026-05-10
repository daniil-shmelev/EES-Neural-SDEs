# OU Latent SDE

Neural Latent SDE experiment on high-volatility Ornstein-Uhlenbeck dynamics.
This supports the manuscript OU/LSDE figure and table (`fig:lsde`,
`table:lsde`) and the appendix gradient-error comparison
(`fig:ees25_vs_27_grad_error`).

The training loop is based on `torchsde`; the integrator is selected by the
`method` field in the in-script config.

## Setup

```bash
uv pip install -e ".[ou]"
```

This pulls the patched `torchsde` build with EES/MCF solver support plus
`torchcde`.

## Run

```bash
python experiments/ou/OU.py
python experiments/ou/plot_OU.py
```

Committed gradient-error artifacts live in
`experiments/ou/results/grad_error/single_step/`. Rerun the scripts above to
regenerate training-loss figures such as `OU_mse.pdf`.
