# RNA Torus `T^7`

Legacy torus-valued neural SDE experiment for one-step-ahead RNA backbone
torsion forecasting. This was the original `T^7` benchmark; the current main
paper torus experiment is the Kuramoto study in `experiments/kuramoto/`.
RNA results are still used for the intro memory-scaling figure and remain
useful as a biological torus benchmark.

The model predicts the next residue's seven dihedral angles from a 20-residue
context window. Drift and diffusion are MLPs conditioned on a GRU encoding of
`sin(theta)`, `cos(theta)`, and one-hot base features. Integration uses
`CF-EES(2,5)` with reversible adjoints where appropriate.

## Paper Results

| Paper item | Result |
|---|---|
| Intro memory-scaling curve | `results/fig_scaling_compact_nogrid.pdf` |
| RNA torus memory raw data | `results/rna_hlo_sweep.json`, `rna_benchmark_scaling*.json` |

Older RNA runtime, adjoint-retrain, and prediction-demo outputs remain
committed for regression checks.

## Setup

```bash
uv pip install -e ".[rna]"

# Fetch raw BGSU mmCIF files. They are not committed.
python -m experiments.rna.datasets.download
```

## Run

```bash
# Single training run
python -m experiments.rna.experiment.train_rna experiments/rna/configs/nsde.toml

# Memory/runtime sweeps
python -m experiments.rna.scripts.run_rna_sweep
python -m experiments.rna.scripts.run_rna_hlo_sweep

# Adjoint retrain comparison
python -m experiments.rna.scripts.collect_adjoint_retrain

# Reference gradient for runtime/error comparison
python -m experiments.rna.scripts.compute_reference_gradient

# Demo predictions and plotting
python -m experiments.rna.scripts.train_and_save_predictions_demo
python -m experiments.rna.plots
```

## Results Committed

| File | Purpose |
|---|---|
| `results/rna_benchmark_scaling{,_ext,_n10k}.json` | memory scaling |
| `results/rna_hlo_sweep.json` | XLA scratch-memory sweep |
| `results/rna_benchmark_ref_gradient.{json,npy}` | gradient reference |
| `results/rna_integrator_benchmark.json` | runtime/error comparison |
| `results/rna_adjoint_retrain.json` | adjoint retrain comparison |
| `results/rna_predictions_demo.npz` | saved prediction demo |
| `results/fig_trajectory_compact.pdf` | trajectory visualization |
| `results/fig_predictions_bar.pdf` | per-angle MAE chart |
| `results/fig_scaling_compact_nogrid.pdf` | memory scaling chart |
