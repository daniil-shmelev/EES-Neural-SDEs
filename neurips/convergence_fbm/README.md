# Euclidean fBm convergence

This NeurIPS-only experiment measures forward and recovery errors for EES(2,5) and EES(2,7), using the Redmann–Riedel example and Hurst indices 0.4, 0.5 and 0.6. Run `./run convergence-fbm` from the repository root. Saved Euclidean PDFs and the combined Euclidean/SO(3) 24-panel page are in `results/`.

The shared geometric experiment remains in [`../../experiments/convergence_fbm/`](../../experiments/convergence_fbm/README.md). Its compositor reads Euclidean panels here and geometric panels there; use `python -m experiments.convergence_fbm.scripts.compose_12up_pages --kind all` to compose the separate and combined pages.
