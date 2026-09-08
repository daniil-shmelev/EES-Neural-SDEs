# Unification validation

Validated on CPU using Python 3.13.14 and the pinned JAX/Diffrax/georax environment recorded in `unification-environment.json`. The Torch smoke runs used the existing Python 3.12.13 / Torch 2.12.0 installation with TorchSDE pinned to `a0b71269538197f15b11792400fb77e26737211d` and TorchCDE 0.2.5, installed in a separate temporary directory. All generated validation outputs stayed outside the committed source tree.

| Check | Result |
|---|---|
| `python -m pytest experiments/kuramoto/tests tests/test_unified_solvers.py -q` | 23 passed; no skipped tests. Covers tangent finite differences, model forwards, sphere solver/adjoint paths, and finite nonzero neural parameter gradients. Diffrax emits its existing CG2 stochastic-interpretation warning. |
| Full default ODE run: `python -m experiments.convergence_ode.convergence --output-dir /tmp/ode-validation` | Numerical assertions passed. Forward slopes 2.00008 / 2.00002; recovery slopes 4.99008 / 6.99796 for CF-EES(2,5) / CF-EES(2,7). PDF, CSV and package/source metadata JSON generated. |
| `python -m order_verification.verify` | Kauri verifies planar order 2 and antisymmetric orders 5/7, including both four-stage parameter branches and 24 table entries. |
| `python order_verification/siam/scripts/verify_cfees.py` | Exact ordered-forest checks and companion checks passed. |
| `python order_verification/siam/scripts/summarize_records.py` | Archived learning record counts, budgets and summary statistics passed. |
| `python -m experiments.stiff_gbm.GBM --method ees25 --epochs 1 --num-samples 25 --batch-size 20 --output-dir /tmp/gbm-smoke` | Passed; finite training loss and gradient-MSE output. |
| `python -m experiments.ou.train_paper --method ees25 --epochs 1 --num-samples 25 --batch-size 20 --output-dir /tmp/ou-smoke` | Passed; finite training loss and gradient-MSE output. |
| Compositor `--kind all --dpi 72` | Both 12-panel and combined 24-panel vector PDFs generated successfully with Ghostscript/pdflatex. |
| Training entry-point imports and small volatility signature loss | Passed for Kuramoto, sphere and volatility in the pinned environment. |
| Sphere/volatility TOML expansion | All 127 configurations load, including the local SRKMK sweep and fixed-step volatility grid. |
| Python/TOML parsing, every HPC shell script with `bash -n`, `git diff --check` | Passed. |
| Wheel build and content checks | Wheel includes the new ODE code, shared plot helper, exact verifier/records and stability modules. |
| `uv pip check` | All installed packages compatible. |
| Branch ancestry, record preservation and submodule availability | All six original remote tips retained in integration ancestry; SIAM measurement records copied byte-for-byte; both pinned submodule commits are accessible upstream. |

The current external manuscript's table labels differ from the verifier's archived layout, so optional manuscript-table comparisons fail their strict label checks. Neither manuscript nor frozen measurements were modified to conceal that mismatch. Algebraic and record-only validation now run independently; supplying `--manuscript` retains the additional strict checks.

These smoke checks do not regenerate full learning tables or benchmark GPU runtime/memory. External volatility datasets, full training, sphere action/environment provenance and the documented GBM drift/NFE discrepancies remain release-verification work. No arXiv reproduction tag is created by this integration. The CPU GitHub Actions workflow repeats solver and order checks on PRs targeting `main`.
