"""Run the RNA torsion memory-efficiency sweep over (solver, n_steps, k).

Each sweep cell is one subprocess so XLA's monotonic ``peak_bytes_in_use``
counter starts fresh per cell. After every cell finishes, we aggregate the
per-cell ``metrics.json`` into one ``summary.json`` for plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


N_STEPS_GRID = (5, 10, 20, 50, 100)
K_GRID = (1, 3, 5)
SOLVERS = ("cfees25", "cg2")


def _cell_dir(root: Path, solver: str, n_steps: int, k: int) -> Path:
    return root / f"{solver}_n{n_steps}_k{k}"


def _run_cell(
    config_path: Path,
    *,
    output_dir: Path,
    solver: str,
    n_steps: int,
    residues_per_state: int,
    epochs: int,
    batch_size: int,
    seed: int,
    max_chains: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "experiment.train_rna",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--solver",
        solver,
        "--n-steps",
        str(n_steps),
        "--residues-per-state",
        str(residues_per_state),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
    ]
    if max_chains is not None:
        cmd.extend(["--max-chains", str(max_chains)])

    log_path = output_dir / "stdout.log"
    print(f"  running {solver} n_steps={n_steps} k={residues_per_state}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(errors="ignore").splitlines()[-20:]
        oom = any("RESOURCE_EXHAUSTED" in line or "out of memory" in line.lower() for line in tail)
        return {
            "solver": solver,
            "n_steps": n_steps,
            "residues_per_state": residues_per_state,
            "oom": bool(oom),
            "returncode": result.returncode,
            "error_tail": tail,
        }
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return {
            "solver": solver,
            "n_steps": n_steps,
            "residues_per_state": residues_per_state,
            "oom": False,
            "returncode": result.returncode,
            "error_tail": ["metrics.json missing"],
        }
    metrics = json.loads(metrics_path.read_text())
    metrics["oom"] = False
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "rna" / "nsde.toml",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chains", type=int, default=100)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for sweep cells (default: results/rna_sweep_<timestamp>)",
    )
    parser.add_argument("--n-steps", type=int, nargs="+", default=list(N_STEPS_GRID))
    parser.add_argument("--k", type=int, nargs="+", default=list(K_GRID))
    parser.add_argument("--solvers", type=str, nargs="+", default=list(SOLVERS))
    parser.add_argument(
        "--n-steps-fixed",
        type=int,
        default=20,
        help="n_steps for the dimension sweep cells",
    )
    parser.add_argument(
        "--k-fixed",
        type=int,
        default=1,
        help="residues_per_state for the n_steps sweep cells",
    )
    args = parser.parse_args()

    output_root = args.output_root or (
        PROJECT_ROOT
        / "results"
        / f"rna_sweep_{datetime.now().strftime('%H%M%S_%Y%m%d')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[str, int, int]] = []
    for solver in args.solvers:
        for n_steps in args.n_steps:
            cells.append((solver, n_steps, args.k_fixed))
        for k in args.k:
            if k == args.k_fixed:
                continue
            cells.append((solver, args.n_steps_fixed, k))

    summary: dict = {
        "config_path": str(args.config),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_chains": args.max_chains,
        "cells": [],
    }

    print(f"running {len(cells)} sweep cells under {output_root}", flush=True)
    for solver, n_steps, k in cells:
        cell_dir = _cell_dir(output_root, solver, n_steps, k)
        if (cell_dir / "metrics.json").exists():
            print(f"  skipping {solver} n={n_steps} k={k} (already done)", flush=True)
            metrics = json.loads((cell_dir / "metrics.json").read_text())
            metrics["oom"] = metrics.get("oom", False)
        else:
            metrics = _run_cell(
                args.config,
                output_dir=cell_dir,
                solver=solver,
                n_steps=n_steps,
                residues_per_state=k,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                max_chains=args.max_chains,
            )
        summary["cells"].append(metrics)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
