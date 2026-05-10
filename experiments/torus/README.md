# Torus `T^7` Memory Benchmark

This experiment reproduces the intro memory-scaling figure using a generic
neural SDE on the flat 7-torus.

The benchmark measures peak XLA scratch memory for one forward-and-backward
solve as the number of solver steps grows. It compares `CF-EES(2,5)` with a
reversible adjoint against `CG2` with full and recursive checkpoint adjoints.

## Paper Results

| Paper item | Result |
|---|---|
| Intro memory-scaling curve | `results/fig_scaling_compact_nogrid.pdf` |
| Torus memory raw data | `results/torus_memory_scaling.json` |

## Setup

```bash
uv pip install -e ".[torus]"
```

## Run

```bash
# Recreate the committed figure from the committed memory measurements.
python -m experiments.torus.plots

# Run a fresh sweep. This can be expensive for the largest step counts.
python -m experiments.torus.scripts.run_memory_sweep \
  --output experiments/torus/results/torus_memory_scaling_new.json

# Run a single small smoke cell.
python -m experiments.torus.scripts.memory_sweep_single \
  --n-steps 5 --num-angles 2 --batch-size 4 --n-reps 1
```
