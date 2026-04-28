# RNA torus T^7 (paper §3.6, Tables `tab:torus_*`, Figs `fig:torus_*`)

Trains a neural SDE on the flat 7-torus $\mathbb T^7 = \mathrm{SO}(2)^7$ for one-step-ahead RNA backbone-torsion forecasting (next residue's seven dihedrals from a 20-residue context window). Drift and diffusion are MLPs conditioned on a GRU encoding of $(\sin\theta, \cos\theta, \mathrm{onehot}(b))$ features; integration uses `CF-EES(2,5)` with `ReversibleAdjoint`.

Loss: wrapped energy score on the torus.

This experiment backs:

- **Tab `tab:torus_scaling`** — XLA scratch memory at $d=350$, batch $32$, $n_{\mathrm{steps}} \in [5, 10\,000]$
- **Tab `tab:torus_runtime_error`** — wall-clock + relative $\ell_2$ gradient error vs fine-$dt$ reference
- **Tab `tab:torus_adjoint_retrain`** — three retrains varying only the diffrax adjoint
- **Tab `tab:torus_mae`** — per-angle wrapped MAE on $17\,682$ non-canonical test residues
- **Fig `fig:torus_trajectory`** — stochastic trajectory on $\mathbb T^2 \subset \mathbb R^3$
- **Fig `fig:torus_bar`** — per-angle wrapped MAE bar chart
- **Fig `fig:torus_scaling`** — memory scaling curve

## Setup

```bash
uv pip install -e ".[rna]"

# fetch raw BGSU mmCIF files (not committed)
python -m experiments.rna.datasets.download
```

## Run

```bash
# Single training run
python -m experiments.rna.experiment.train_rna experiments/rna/configs/nsde.toml

# Memory/runtime sweep that produces tab:torus_scaling and tab:torus_runtime_error
python -m experiments.rna.scripts.run_rna_sweep
python -m experiments.rna.scripts.run_rna_hlo_sweep

# Adjoint retrain comparison (tab:torus_adjoint_retrain)
python -m experiments.rna.scripts.collect_adjoint_retrain

# Reference gradient for tab:torus_runtime_error
python -m experiments.rna.scripts.compute_reference_gradient

# Demo predictions for tab:torus_mae
python -m experiments.rna.scripts.train_and_save_predictions_demo
```

## Results committed

| File | Backs |
|---|---|
| `results/rna_benchmark_scaling{,_ext,_n10k,_rev,_rev15}.json` | `tab:torus_scaling` |
| `results/rna_hlo_{sweep,n_small,n_extreme,dim_wide}.json` | `tab:torus_scaling` |
| `results/rna_benchmark_ref_gradient.{json,npy}` | `tab:torus_runtime_error` reference |
| `results/rna_integrator_benchmark.json` | `tab:torus_runtime_error` |
| `results/rna_adjoint_retrain.json` | `tab:torus_adjoint_retrain` |
| `results/rna_predictions_demo.npz` (12 MB) | `tab:torus_mae`, `fig:torus_bar` |
| `results/fig_trajectory_compact.pdf` | `fig:torus_trajectory` |
| `results/fig_predictions_bar.pdf` | `fig:torus_bar` |
| `results/fig_scaling_compact_nogrid.pdf` | `fig:torus_scaling` |
