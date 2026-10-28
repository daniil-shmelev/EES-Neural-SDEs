# Torus-memory benchmark

This directory retains the seven-torus benchmark used in the NeurIPS introduction (`fig:torus_scaling` and `tab:torus_intro_memory_data`). The unused RNA prediction experiment and SPD branches have moved to `/home/luke/nsde_archive/experiments/rna/`. The data-download and training infrastructure are no longer dependencies. The benchmark uses the public Georax `Torus`; `models/torus.py` contains only the angle-wrapping helper used by data and losses.

## Saved measurements

`results/rna_benchmark_scaling.json` contains the original measurements at dimension 7, batch size 1024, hidden width 128 and context width 64. `results/fig_scaling_compact_nogrid.pdf` is the original figure. These files were copied without alteration and retain their original solver provenance. New measurements use the pinned Georax revision and CF-EES(2,5;1/10); they do not reproduce the old library implementation exactly. The retained model includes the original four-way categorical encoding because it contributes to the measured computation.

## Commands

From the repository root, run an isolated measurement:

```bash
./run torus-memory --configs 7x50 --modes cfees25:reversible cg2:checkpoint_full cg2:checkpoint_recursive --batch-size 1024 --t1 1 --n-reps 3
```

The runner reports XLA scratch allocation and process RSS separately. Its retained scratch measurement uses the minimum of 15 compilations, as the original implementation did. Small CPU checks validate code execution only; manuscript memory measurements require the intended GPU environment.

The historical plotting routine is retained for provenance. It subtracts baselines and flattens the reversible series to its minimum, so its displayed curves differ from the raw absolute measurements. To reproduce that plotting convention without overwriting the saved figure:

```bash
python -m neurips.torus_memory.plot --output /tmp/torus_memory_historical.pdf
```

The launcher selects the uv dependencies and writes `benchmark.json` in a fresh `results/runs/<timestamp>/` directory. `--help` shows the default sweep and settings in `benchmark.py`.
