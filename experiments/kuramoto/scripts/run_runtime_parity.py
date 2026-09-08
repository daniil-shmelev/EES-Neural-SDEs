"""Drive the runtime-parity training matrix.

Runs CF-EES(2,5) (Reversible), CG2 (Recursive), and CG2 (Full) at
per-method `n_steps` chosen so the wall-clock per training step is
roughly equal across methods (calibrated separately, see
RUNTIME_PARITY_PLAN.md).

Loop order is seeds outer, methods inner: for each seed we run all
three methods before moving on, so an interrupted run still leaves
complete cross-method comparisons for the seeds that finished.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# (label, solver, adjoint)
METHODS = [
    ("cfees25_reversible",  "cfees25", "reversible"),
    ("cg2_recursive",       "cg2",     "checkpoint_recursive"),
    ("cg2_full",            "cg2",     "checkpoint_full"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path,
        default=Path("experiments/kuramoto/configs/kuramoto_runtime_parity.toml"),
    )
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dtype", choices=("float32", "float64"), default=None)
    p.add_argument(
        "--n-steps-cfees", type=int, required=True,
        help="n_steps for CF-EES(2,5) + Reversible.",
    )
    p.add_argument(
        "--n-steps-cg2-recursive", type=int, default=None,
        help="n_steps for CG2 + Recursive. Default: same as --n-steps-cfees.",
    )
    p.add_argument(
        "--n-steps-cg2-full", type=int, default=None,
        help="n_steps for CG2 + Full. Default: same as --n-steps-cfees.",
    )
    p.add_argument(
        "--output-base", type=Path,
        default=Path("experiments/kuramoto/results/runtime_parity_N1000/"),
    )
    p.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help=f"Subset of method labels (default: all of {[m[0] for m in METHODS]}).",
    )
    p.add_argument(
        "--summary-path", type=Path, default=None,
        help="Summary JSON path. Default: <output-base>/runtime_parity_summary.json.",
    )
    return p.parse_args()


_METRIC_KEYS = (
    "best_val_loss", "final_train_loss", "final_val_loss", "final_test_loss",
    "final_test_theta_mae_mean", "final_test_omega_mae_mean", "train_walltime_s",
)


def _read_metrics(info: dict, metrics_path: Path) -> dict:
    try:
        metrics = json.loads(metrics_path.read_text())
        info.update({k: metrics.get(k) for k in _METRIC_KEYS})
    except Exception as exc:
        info["metrics_parse_error"] = str(exc)
    return info


def _run_cell(args, label: str, solver: str, adjoint: str, n_steps: int,
              seed: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    info = {
        "label": label, "seed": seed, "solver": solver, "adjoint": adjoint,
        "n_steps": n_steps, "out_dir": str(out_dir),
    }

    # Idempotent resume: a completed cell (metrics.json present) is skipped, so
    # resubmitting after a timeout never redoes finished work.
    if metrics_path.exists():
        info.update({"exit_code": 0, "wall_clock_s": 0.0, "skipped": True})
        _read_metrics(info, metrics_path)
        print(f"[runtime-parity] SKIP {label}/seed{seed} (already complete)", flush=True)
        return info

    cmd = [
        sys.executable, "-m",
        "experiments.kuramoto.experiment.train_kuramoto",
        str(args.config),
        "--output-dir", str(out_dir),
        "--N", str(args.N),
        "--n-steps", str(n_steps),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--seed", str(seed),
        "--solver", solver,
        "--adjoint", adjoint,
        "--lr", str(args.lr),
        "--auto-resume",  # continue in-place from a partial (timed-out) attempt
    ]
    if args.dtype is not None:
        cmd.extend(["--dtype", args.dtype])
    print(f"\n[runtime-parity] === {label} | seed={seed} | N={args.N} | "
          f"n_steps={n_steps} | epochs={args.epochs} ===", flush=True)
    print(f"[runtime-parity] log: {out_dir / 'train.log'}", flush=True)

    t0 = time.perf_counter()
    log_path = out_dir / "train.log"
    with open(log_path, "ab") as fh:  # append so a resumed attempt keeps its log
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    dt = time.perf_counter() - t0

    info.update({"exit_code": proc.returncode, "wall_clock_s": dt})
    if metrics_path.exists():
        _read_metrics(info, metrics_path)
    print(f"[runtime-parity] {label}/seed{seed} exit={proc.returncode} "
          f"t={dt:.0f}s test={info.get('final_test_loss', 'NA')}", flush=True)
    return info


def main() -> int:
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)

    n_steps_per_method = {
        "cfees25_reversible": args.n_steps_cfees,
        "cg2_recursive":      args.n_steps_cg2_recursive or args.n_steps_cfees,
        "cg2_full":           args.n_steps_cg2_full      or args.n_steps_cfees,
    }

    selected = METHODS
    if args.methods is not None:
        wanted = set(args.methods)
        selected = [m for m in METHODS if m[0] in wanted]
        if not selected:
            print(f"[runtime-parity] no methods match {args.methods}; "
                  f"available: {[m[0] for m in METHODS]}", flush=True)
            return 1

    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "n_steps_per_method": n_steps_per_method,
        "cells": [],
    }
    summary_path = args.summary_path or args.output_base / "runtime_parity_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for label, solver, adjoint in selected:
            cell_out = args.output_base / f"{label}_seed{seed}"
            n_steps = n_steps_per_method[label]
            cell_info = _run_cell(args, label, solver, adjoint, n_steps,
                                  seed, cell_out)
            summary["cells"].append(cell_info)
            summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[runtime-parity] wrote {summary_path} ({len(summary['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
