"""Run the missing cells that separate solver and adjoint effects.

This complements ``run_runtime_parity.py``. Existing runtime-parity cells are
reused, while the two missing CF-EES + Full-adjoint cells are trained:

1. CF-EES Full at 50 steps versus existing CF-EES Reversible at 50 steps.
2. CF-EES Full at 75 steps versus existing CG2 Full at 75 steps.

The driver is resumable at cell granularity: a cell with a complete
``metrics.json`` is skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


MISSING_CELLS = (
    ("cfees25_full_n50", "cfees25", "checkpoint_full", 50),
    ("cfees25_full_n75", "cfees25", "checkpoint_full", 75),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/kuramoto/configs/kuramoto_runtime_parity.toml"
        ),
    )
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(
            "experiments/kuramoto/results/solver_adjoint_ablation_N1000"
        ),
    )
    return parser.parse_args()


def run_cell(
    args: argparse.Namespace,
    label: str,
    solver: str,
    adjoint: str,
    n_steps: int,
    seed: int,
) -> dict[str, object]:
    output_dir = args.output_base / f"{label}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[ablation] skip complete cell {output_dir}", flush=True)
        return {
            "label": label,
            "seed": seed,
            "solver": solver,
            "adjoint": adjoint,
            "n_steps": n_steps,
            "exit_code": 0,
            "skipped_existing": True,
            "out_dir": str(output_dir),
            **json.loads(metrics_path.read_text()),
        }

    command = [
        sys.executable,
        "-m",
        "experiments.kuramoto.experiment.train_kuramoto",
        str(args.config),
        "--output-dir",
        str(output_dir),
        "--N",
        str(args.N),
        "--n-steps",
        str(n_steps),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(seed),
        "--solver",
        solver,
        "--adjoint",
        adjoint,
        "--lr",
        str(args.lr),
    ]
    print(
        f"\n[ablation] {label} seed={seed} N={args.N} "
        f"steps={n_steps} adjoint={adjoint}",
        flush=True,
    )
    start = time.perf_counter()
    with (output_dir / "train.log").open("wb") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_time = time.perf_counter() - start
    result: dict[str, object] = {
        "label": label,
        "seed": seed,
        "solver": solver,
        "adjoint": adjoint,
        "n_steps": n_steps,
        "exit_code": process.returncode,
        "wall_clock_s": wall_time,
        "out_dir": str(output_dir),
    }
    if metrics_path.exists():
        result.update(json.loads(metrics_path.read_text()))
    print(
        f"[ablation] complete {label}/seed{seed}: "
        f"exit={process.returncode} wall={wall_time:.0f}s",
        flush=True,
    )
    return result


def main() -> int:
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_base / "ablation_summary.json"
    summary: dict[str, object] = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cells": [],
    }

    for label, solver, adjoint, n_steps in MISSING_CELLS:
        for seed in args.seeds:
            result = run_cell(
                args, label, solver, adjoint, n_steps, seed
            )
            summary["cells"].append(result)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            if result["exit_code"] != 0:
                print(f"[ablation] failed cell; see {result['out_dir']}", flush=True)
                return int(result["exit_code"])

    print(f"[ablation] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
