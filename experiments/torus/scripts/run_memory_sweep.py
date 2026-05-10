"""Run the random torus memory sweep used by the intro figure."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODES = (
    ("cfees25", "reversible"),
    ("cg2", "checkpoint_full"),
    ("cg2", "checkpoint_recursive"),
    ("cg4", "checkpoint_full"),
    ("cg4", "checkpoint_recursive"),
)


def _run_cell(args: argparse.Namespace, solver: str, adjoint: str, n_steps: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "experiments.torus.scripts.memory_sweep_single",
        "--solver",
        solver,
        "--adjoint",
        adjoint,
        "--num-angles",
        str(args.num_angles),
        "--n-steps",
        str(n_steps),
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(args.context_length),
        "--hidden-dim",
        str(args.hidden_dim),
        "--ctx-dim",
        str(args.ctx_dim),
        "--dt",
        str(args.dt),
        "--n-reps",
        str(args.n_reps),
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "solver": solver,
            "adjoint": adjoint,
            "n_steps": n_steps,
            "num_angles": args.num_angles,
            "returncode": result.returncode,
            "error_tail": result.stderr.splitlines()[-20:],
        }
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--n-steps",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100, 200, 400, 800, 2000, 5000, 10000],
    )
    parser.add_argument("--num-angles", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--n-reps", type=int, default=3)
    args = parser.parse_args(argv)

    out_path = args.output or (
        PROJECT_ROOT
        / "experiments"
        / "torus"
        / "results"
        / f"torus_memory_scaling_{datetime.now().strftime('%H%M%S_%Y%m%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells = []
    for solver, adjoint in DEFAULT_MODES:
        for n_steps in args.n_steps:
            print(f"{solver}:{adjoint} n_steps={n_steps}", flush=True)
            cells.append(_run_cell(args, solver, adjoint, n_steps))

    payload = {
        "benchmark": "random_torus_nsde",
        "description": (
            "Peak XLA scratch-memory measurements for a generic random "
            "neural SDE on the 7-torus."
        ),
        "cells": cells,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
