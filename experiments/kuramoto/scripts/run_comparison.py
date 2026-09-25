"""Compare CG2 Full, CG2 Recursive and CF-EES Reconstruction at 150 generator evaluations per trajectory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


from experiments.kuramoto.experiment.config import COMPARISON_METHODS, DEFAULT_CONFIG, load_config

METHODS = [(label, solver.value, adjoint) for label, solver, adjoint, _ in COMPARISON_METHODS]
DEFAULT_STEPS = {label: steps for label, _, _, steps in COMPARISON_METHODS}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path,
        default=DEFAULT_CONFIG,
    )
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--n-steps-cfees", type=int, default=DEFAULT_STEPS["cfees25_reversible"],
        help="n_steps for CF-EES(2,5) + Reversible.",
    )
    p.add_argument(
        "--n-steps-cg2-recursive", type=int, default=DEFAULT_STEPS["cg2_recursive"],
        help="n_steps for CG2 + Recursive. Default: 75 (150 generator evaluations).",
    )
    p.add_argument(
        "--n-steps-cg2-full", type=int, default=DEFAULT_STEPS["cg2_full"],
        help="n_steps for CG2 + Full. Default: 75 (150 generator evaluations).",
    )
    p.add_argument(
        "--output-base", type=Path,
        default=Path("experiments/kuramoto/results/unified/training/"),
    )
    p.add_argument(
        "--methods", type=str, nargs="+", default=None, choices=[m[0] for m in METHODS],
        help=f"Subset of method labels (default: all of {[m[0] for m in METHODS]}).",
    )
    args = p.parse_args()
    config = load_config(args.config)
    for name, field in (("N", "N"), ("batch_size", "batch_size"), ("epochs", "epochs"), ("lr", "learning_rate")):
        if getattr(args, name) is None:
            setattr(args, name, getattr(config, field))
    return args


def _run_cell(args, label: str, solver: str, adjoint: str, n_steps: int,
              seed: int, out_dir: Path) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m",
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
    ]
    print(f"\n[comparison] === {label} | seed={seed} | N={args.N} | "
          f"n_steps={n_steps} | epochs={args.epochs} ===", flush=True)
    print(f"[comparison] log: {out_dir.with_suffix('.log')}", flush=True)

    t0 = time.perf_counter()
    log_path = out_dir.with_suffix(".log")
    if log_path.exists():
        raise FileExistsError(f"Use a fresh log path: {log_path}")
    with log_path.open("w") as fh:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                fh.write(line)
                fh.flush()
            proc.wait()
    dt = time.perf_counter() - t0
    log_path.replace(out_dir / "train.log")

    info = {
        "label": label, "seed": seed, "solver": solver, "adjoint": adjoint,
        "n_steps": n_steps, "exit_code": proc.returncode,
        "wall_clock_s": dt, "out_dir": str(out_dir),
    }
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            info.update({k: metrics.get(k) for k in (
                "best_val_loss", "final_train_loss", "final_val_loss",
                "final_test_loss", "final_test_theta_mae_mean",
                "final_test_omega_mae_mean", "train_walltime_s",
            )})
        except Exception as exc:
            info["metrics_parse_error"] = str(exc)
    print(f"[comparison] {label}/seed{seed} exit={proc.returncode} "
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
            print(f"[comparison] no methods match {args.methods}; "
                  f"available: {[m[0] for m in METHODS]}", flush=True)
            return 1

    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "n_steps_per_method": n_steps_per_method,
        "cells": [],
    }
    summary_path = args.output_base / "comparison.json"

    for seed in args.seeds:
        for label, solver, adjoint in selected:
            cell_out = args.output_base / f"{label}_seed{seed}"
            n_steps = n_steps_per_method[label]
            cell_info = _run_cell(args, label, solver, adjoint, n_steps,
                                  seed, cell_out)
            summary["cells"].append(cell_info)
            summary_path.write_text(json.dumps(summary, indent=2))
            if cell_info["exit_code"]:
                return int(cell_info["exit_code"])

    print(f"\n[comparison] wrote {summary_path} ({len(summary['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
