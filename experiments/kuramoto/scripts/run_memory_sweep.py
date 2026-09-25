"""Drive the M4 memory sweep: subprocess-isolated cells across a grid of
(solver, adjoint, n_steps) at fixed N. Aggregates results into a single JSON.

Each cell runs in its own Python process (clean JAX state, isolated OOMs).
Cells that OOM are recorded with `oom: true` rather than crashing the sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

from experiments.kuramoto.experiment.config import COMPARISON_METHODS, DEFAULT_CONFIG, PROTOCOL, load_config, serialize_config


DEFAULT_NSTEPS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
DEFAULT_PAIRS = tuple((solver.value, adjoint) for _, solver, adjoint, _ in COMPARISON_METHODS)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-steps-grid", type=int, nargs="+",
                   default=list(DEFAULT_NSTEPS))
    p.add_argument(
        "--pairs", type=str, nargs="+",
        default=[f"{s}:{a}" for s, a in DEFAULT_PAIRS],
        choices=[f"{s}:{a}" for s, a in DEFAULT_PAIRS],
        help="(solver:adjoint) pairs",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--output", type=Path,
                   default=Path("experiments/kuramoto/results/unified/memory.json"))
    p.add_argument("--cell-timeout", type=int, default=900,
                   help="Per-cell timeout in seconds.")
    args = p.parse_args()
    config = load_config(args.config)
    for name in ("N", "batch_size", "seed"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(config, name))
    return args


def _run_cell(args, solver: str, adjoint: str, n_steps: int, cell_dir: Path) -> dict:
    out_path = cell_dir / f"cell_{solver}_{adjoint}_n{n_steps}.json"
    if out_path.exists():
        raise FileExistsError(f"Use a fresh output path: {out_path}")
    cmd = [
        sys.executable, "-m",
        "experiments.kuramoto.scripts.memory_sweep_single",
        "--config", str(args.config),
        "--N", str(args.N),
        "--solver", solver, "--adjoint", adjoint,
        "--n-steps", str(n_steps),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--n-warmup", str(args.n_warmup),
        "--n-reps", str(args.n_reps),
        "--output", str(out_path),
    ]
    print(f"\n[sweep] === {solver} | {adjoint} | n_steps={n_steps} ===", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.cell_timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"solver": solver, "adjoint": adjoint, "n_steps": n_steps,
                "oom": False, "error": f"TIMEOUT after {args.cell_timeout}s",
                "wall_clock_s_mean": None, "peak_bytes": None}
    dt = time.perf_counter() - t0

    if out_path.exists():
        try:
            cell = json.loads(out_path.read_text())
        except Exception:
            cell = {"solver": solver, "adjoint": adjoint, "n_steps": n_steps,
                    "oom": False, "error": "JSON parse failure",
                    "wall_clock_s_mean": None, "peak_bytes": None}
    else:
        cell = {"solver": solver, "adjoint": adjoint, "n_steps": n_steps,
                "oom": "out of memory" in (proc.stderr or "").lower(),
                "error": (proc.stderr or "")[-500:] or "no output file",
                "wall_clock_s_mean": None, "peak_bytes": None}
    cell["subprocess_wall_s"] = dt
    if cell.get("peak_bytes"):
        peak_mib = cell["peak_bytes"] / 2**20
        print(f"[sweep] peak={peak_mib:8.1f} MiB | wall={cell.get('wall_clock_s_mean'):.2f}s",
              flush=True)
    elif cell.get("oom"):
        print("[sweep] OOM", flush=True)
    elif cell.get("error"):
        print(f"[sweep] ERROR: {cell['error'][:200]}", flush=True)
    return cell


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output path: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cell_dir = args.output.parent / f"{args.output.stem}_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for p in args.pairs:
        s, a = p.split(":")
        pairs.append((s, a))

    cells = []
    for (solver, adjoint), n_steps in product(pairs, sorted(args.n_steps_grid)):
        cells.append(_run_cell(args, solver, adjoint, n_steps, cell_dir))
        # Persist after every cell so partial results survive a crash.
        args.output.write_text(json.dumps({
            "protocol": PROTOCOL,
            "experiment_config": json.loads(serialize_config(load_config(args.config))),
            "config": {
                "N": args.N, "batch_size": args.batch_size,
                "n_steps_grid": args.n_steps_grid, "pairs": args.pairs,
                "seed": args.seed,
            },
            "cells": cells,
        }, indent=2))

    print(f"\n[sweep] wrote {args.output} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
