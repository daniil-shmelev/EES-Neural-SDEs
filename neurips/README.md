# NeurIPS-only experiments and records

Run commands from the repository root using `./run`; the command names are unchanged. These experiments and historical measurements are outside the current SIAM manuscript.

| Directory | Contents | Launcher |
|---|---|---|
| `ou/` | Ornstein–Uhlenbeck learning | `./run ou` |
| `stiff_gbm/` | Stiff geometric Brownian motion | `./run stiff-gbm` |
| `stability/`, `stability_ode/` | Stochastic and deterministic stability | `./run stability`, `./run stability-ode` |
| `stochastic_volatility/` | Rough Bergomi code, data and saved predictions | `./run rough-vol` |
| `torus_memory/` | Seven-torus memory benchmark | `./run torus-memory` |
| `convergence_fbm/` | Euclidean convergence code and plots; combined 24-panel plot | `./run convergence-fbm` |
| `kuramoto/results/` | Historical fixed-step runtime-parity records used by NeurIPS | Saved records; the current training launcher uses the newer SIAM model |
| `sphere_memory/results/` | Original sphere-memory CSV from the PyTorch companion | Saved records |

The geometric convergence code and plots shared with SIAM remain in `../experiments/convergence_fbm/`; `./run convergence-fbm --so3` selects that experiment. Angle wrapping shared by Kuramoto and the torus benchmark lives in `../ees_core/torus.py`.

Saved run configurations retain their original paths and implementation provenance. The [cleanup manifest](../docs/cleanup_20261027.json) maps relocated paths and inventories the deleted unused trial runs. The historical consolidation manifest and SIAM `sources.json` also retain their original paths; they describe the earlier imports rather than the present layout.
