# SO(3) rough-driver convergence

The geometric convergence code and saved figures used by SIAM and NeurIPS remain here. The SIAM figure is `results/cfees_stochastic_convergence_12up.pdf`, with per-Hurst source pages `cfees_stochastic_convergence_H{40,50,60}.pdf`. That figure is byte-identical to the manuscript copy. Original sampled arrays and invocation-specific metadata are unavailable; its described source is Georax's `docs/examples/stochastic_convergence.py`.

`results/so3_reversibility_results.json` is a separate CF-EES(2,5) recovery diagnostic, not the raw data behind the manuscript's 12-panel figure. The current `scripts/so3_reversibility_verify.py` tests a nonlinear matrix-polynomial problem with three driver components; it must not be presented as reproducing the manuscript's two-component rotation experiment.

Run `./run convergence-fbm --so3` from the repository root for the recovery diagnostic. It selects the required dependencies and creates a fresh directory under `results/runs/`. The local-order verifier and page compositor remain in `scripts/`.

Euclidean convergence code and saved pages are now in [`../../neurips/convergence_fbm/`](../../neurips/convergence_fbm/README.md); `./run convergence-fbm` selects that NeurIPS experiment. The compositor reads its two source directories separately, with `--results-dir` for SO(3) and `--euclidean-results-dir` for Euclidean panels and combined output.
