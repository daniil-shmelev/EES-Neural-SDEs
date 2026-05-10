# Stochastic Kuramoto Neural SDE on `T T^N`

This experiment trains a neural SDE on the product state space
`T T^N = T^N x R^N` for stochastic second-order Kuramoto dynamics with
inertia. It is the current torus-valued experiment in the manuscript, replacing
the earlier RNA torsion benchmark as the main Lie-group memory-scaling study.

The simulator uses the standard power-grid form

```text
m theta_i'' = -theta_i' + Omega_i + (K/N) sum_j sin(theta_j - theta_i) + xi_i(t)
```

with bimodal natural frequencies `Omega_i in {+P, -P}`. The state is
`(theta, omega)`, with `theta` wrapped on the torus and `omega` Euclidean.

## What This Backs

| Manuscript item | Script/result |
|---|---|
| `fig:kuramoto_trajectory` | `experiments/kuramoto/data/kuramoto_N2_seed0.gif`, exported to the manuscript figure |
| `fig:kuramoto_memory_scaling` | `python -m experiments.kuramoto.scripts.plot_memory_sweep` |
| `tab:kuramoto_quality` | runtime-parity training outputs in `results/runtime_parity_N1000_n50/` |
| `tab:kuramoto_pilot` | pilot outputs in `results/pilot/` |
| `tab:kuramoto_memory_data` | `results/memory_sweep_N1000.json` and `results/memory_sweep_cells_N1000/` |

## Setup

```bash
uv pip install -e ".[kuramoto]"
```

## Data Generation

Small smoke data is committed for `N=2`; larger sweeps are generated locally or
on a GPU machine.

```bash
# Simulator smoke and verification
python -m experiments.kuramoto.scripts.run_m1 --smoke

# Full default data generation and verification
python -m experiments.kuramoto.scripts.run_m1
```

Defaults: `N in {2,4,8}`, `P=0.5`, `K=2.0`, `m=1`, `D=0.05`, horizon `T=5`,
`n_fine=16384`, `n_obs=200`, and 5000/1000/1000 train/val/test trajectories.

## Training And Memory Sweeps

```bash
# Train a single neural SDE config
python -m experiments.kuramoto.experiment.train_kuramoto experiments/kuramoto/configs/kuramoto.toml

# Runtime/gradient parity config
python -m experiments.kuramoto.experiment.train_kuramoto experiments/kuramoto/configs/kuramoto_runtime_parity.toml

# Plot the committed memory sweep
python -m experiments.kuramoto.scripts.plot_memory_sweep
```

The memory plot writes `experiments/kuramoto/results/fig_kuramoto_memory_scaling.pdf`
by default. Pass `--output <path>` when exporting directly to Overleaf.

## Result Files

| File | Purpose |
|---|---|
| `data/kuramoto_N2_seed0.{npz,json,gif}` | committed smoke dataset and trajectory animation |
| `data/kuramoto_N1000_seed0.json` | metadata for the large memory-sweep system |
| `results/simulator_verification.json` | deterministic phase-lock and stationarity checks |
| `results/memory_sweep_N1000.json` | raw memory-scaling table |
| `results/memory_sweep_cells_N1000/*.json` | per-cell memory measurements |
| `results/runtime_parity_N1000_n50/**` | runtime-parity configs, histories, and metrics |
