"""Aggregate per-adjoint retrain metrics into a single JSON for the paper.

Reads ``metrics.json`` from each retrain output directory (produced by
``experiment.train_rna`` with ``--adjoint <name> --seed <s>``) and writes a
consolidated summary to ``results/rna_adjoint_retrain.json``.

Usage::

    python -m scripts.collect_adjoint_retrain \
        results/rna_retrain_adjoint_reversible \
        results/rna_retrain_adjoint_checkpoint_full \
        results/rna_retrain_adjoint_checkpoint_recursive \
        --output results/rna_adjoint_retrain.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+",
                        help="One or more directories produced by train_rna, "
                             "each containing a metrics.json file.")
    parser.add_argument("--output", type=Path,
                        default=Path("results/rna_adjoint_retrain.json"))
    args = parser.parse_args()

    runs: list[dict] = []
    for run_dir in args.run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"no metrics.json in {run_dir}")
        m = json.loads(metrics_path.read_text())
        runs.append({
            "run_dir": str(run_dir),
            **m,
        })

    # Extract the shared config for a header, keeping only fields we expect
    # to be identical across runs (everything except adjoint & walltime).
    first = runs[0]
    shared_keys = (
        "solver", "seed", "n_steps", "residues_per_state", "num_angles",
        "batch_size", "epochs", "n_train", "learning_rate", "dt", "hidden_dim",
    )
    shared = {k: first.get(k) for k in shared_keys if k in first}
    mismatches: dict[str, set] = {}
    for r in runs[1:]:
        for k in shared_keys:
            if k in r and r[k] != shared.get(k):
                mismatches.setdefault(k, set()).add((shared.get(k), r[k]))

    payload = {
        "config": shared,
        "config_mismatches": {k: list(v) for k, v in mismatches.items()},
        "runs": [
            {
                "adjoint": r.get("adjoint"),
                "test_wrapped_mae": r.get("test_wrapped_mae"),
                "test_loss": r.get("test_loss"),
                "train_walltime_s": r.get("train_walltime_s"),
                "run_dir": r["run_dir"],
            }
            for r in runs
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"wrote {args.output}")
    print(f"  config: {shared}")
    if mismatches:
        print(f"  WARNING: per-run config mismatches: {mismatches}")
    print(f"  {len(runs)} runs:")
    for r in payload["runs"]:
        mae = r.get("test_wrapped_mae")
        wt = r.get("train_walltime_s")
        mae_s = f"{mae:.4f}" if isinstance(mae, (int, float)) else "—"
        wt_s = f"{wt:.0f}s" if isinstance(wt, (int, float)) else "—"
        print(f"    {r.get('adjoint', '?'):>24s}  mae={mae_s}  walltime={wt_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
