"""Drive the M5 hero-training matrix at N=1000.

Each cell is a subprocess invocation of `train_kuramoto.main()` with a
specific (model, solver, adjoint, seed) config. Cells are run sequentially
to avoid GPU memory contention; results live in their own output dir per
cell so a crash doesn't lose previous cells.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# Stages-per-step for each solver. Used to compute n_steps under NFE parity:
# n_steps = nfe_budget / stages.
SOLVER_STAGES = {
    "cfees25": 3,
    "cfees27": 4,
    "cg2": 2,
    "cg4": 4,
    "rkmk": 1,
    # The Euclidean baseline is integrated by diffrax.Heun, which is 2 stages.
    "heun": 2,
}

# (model, solver, adjoint, stages_solver_key, label)
# `stages_solver_key` is the solver whose stage count drives n_steps for this
# variant — for the Euclidean baseline the integrator is Heun, not the
# config's `solver` field (which is just the dispatch hint).
DEFAULT_VARIANTS = [
    ("kuramoto_nsde",      "cfees25", "reversible",           "cfees25", "cfees25_reversible"),
    ("kuramoto_nsde",      "cg2",     "checkpoint_recursive", "cg2",     "cg2_treeverse"),
    ("euclidean_baseline", "cfees25", "direct",               "heun",    "euclidean_baseline"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Base TOML config to use as the starting point.")
    p.add_argument("--N", type=int, default=1000)
    p.add_argument(
        "--nfe-budget", type=int, default=1500,
        help="Target NFE per integration (drift+diffusion evals). "
             "Per variant n_steps is then nfe_budget / stages_per_step.",
    )
    p.add_argument(
        "--n-steps", type=int, default=None,
        help="Override per-variant n_steps. If set, --nfe-budget is ignored "
             "and ALL variants use the same n_steps (legacy behaviour).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--variants", type=str, nargs="+", default=None,
        help="Subset of variant labels (default: all of cfees25_reversible, "
             "cg2_treeverse, euclidean_baseline).",
    )
    p.add_argument("--include-coupled-ablation", action="store_true",
                   help="Also run cfees25+reversible with couple_theta_omega=True (one seed).")
    p.add_argument(
        "--output-base", type=Path,
        default=Path("experiments/kuramoto/results/hero_N1000/"),
    )
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def _run_cell(args, model: str, solver: str, adjoint: str,
              stages_key: str, label: str, seed: int, out_dir: Path,
              ablation_couple: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.n_steps is not None:
        n_steps = args.n_steps
    else:
        stages = SOLVER_STAGES.get(stages_key)
        if stages is None:
            raise ValueError(f"unknown stages_key {stages_key!r}; "
                             f"add to SOLVER_STAGES")
        n_steps = max(1, args.nfe_budget // stages)
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
        "--model", model,
        "--solver", solver,
        "--adjoint", adjoint,
        "--lr", str(args.lr),
    ]
    nfe_actual = n_steps * SOLVER_STAGES.get(stages_key, 0)
    print(f"\n[hero] === {label} | seed={seed} | N={args.N} | "
          f"n_steps={n_steps} | NFE={nfe_actual} | "
          f"batch={args.batch_size} | epochs={args.epochs} ===",
          flush=True)
    print(f"[hero] log: {out_dir / 'train.log'}", flush=True)

    t0 = time.perf_counter()
    log_path = out_dir / "train.log"
    with open(log_path, "wb") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    dt = time.perf_counter() - t0

    info = {
        "label": label, "seed": seed, "model": model, "solver": solver,
        "adjoint": adjoint, "stages_per_step": SOLVER_STAGES.get(stages_key),
        "n_steps": n_steps, "nfe_per_traj": nfe_actual,
        "exit_code": proc.returncode,
        "wall_clock_s": dt, "out_dir": str(out_dir),
        "ablation_couple_theta_omega": ablation_couple,
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
    print(f"[hero] {label}/seed{seed} exit={proc.returncode} t={dt:.0f}s "
          f"test={info.get('final_test_loss', 'NA')}", flush=True)
    return info


def main() -> int:
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)

    selected = list(DEFAULT_VARIANTS)
    if args.variants is not None:
        selected = [v for v in DEFAULT_VARIANTS if v[4] in set(args.variants)]
        if not selected:
            print(f"[hero] no variants match {args.variants}; available: "
                  f"{[v[4] for v in DEFAULT_VARIANTS]}", flush=True)
            return 1

    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cells": [],
    }
    summary_path = args.output_base / "hero_summary.json"

    for model, solver, adjoint, stages_key, label in selected:
        for seed in args.seeds:
            cell_out = args.output_base / f"{label}_seed{seed}"
            cell_info = _run_cell(args, model, solver, adjoint, stages_key,
                                  label, seed, cell_out)
            summary["cells"].append(cell_info)
            summary_path.write_text(json.dumps(summary, indent=2))

    if args.include_coupled_ablation:
        # Override couple_theta_omega via a secondary config on the fly.
        ablation_label = "cfees25_reversible_coupled"
        ablation_dir = args.output_base / f"{ablation_label}_seed{args.seeds[0]}"
        ablation_dir.mkdir(parents=True, exist_ok=True)
        # We don't have a CLI arg for couple_theta_omega; the cleanest hack is
        # a TOML override file. Skipping the implementation for now and
        # noting it as a manual follow-up if reviewers ask.
        print("[hero] coupled-diffusion ablation: not yet wired (no CLI flag); skipping.",
              flush=True)

    print(f"\n[hero] wrote {summary_path} ({len(summary['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
