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


DEFAULT_NSTEPS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
DEFAULT_PAIRS = (
    ("cfees25", "reversible"),
    ("cg2", "checkpoint_full"),
    ("cg2", "checkpoint_recursive"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--n-steps-grid", type=int, nargs="+",
                   default=list(DEFAULT_NSTEPS))
    p.add_argument(
        "--pairs", type=str, nargs="+",
        default=[f"{s}:{a}" for s, a in DEFAULT_PAIRS],
        help="(solver:adjoint) pairs",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--output", type=Path,
                   default=Path("experiments/kuramoto/results/memory_sweep_N1000.json"))
    p.add_argument("--cell-timeout", type=int, default=900,
                   help="Per-cell timeout in seconds.")
    return p.parse_args()


def _run_cell(args, solver: str, adjoint: str, n_steps: int, cell_dir: Path) -> dict:
    out_path = cell_dir / f"cell_{solver}_{adjoint}_n{n_steps}.json"
    cmd = [
        sys.executable, "-m",
        "experiments.kuramoto.scripts.memory_sweep_single",
        "--N", str(args.N),
        "--solver", solver, "--adjoint", adjoint,
        "--n-steps", str(n_steps),
        "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--T", str(args.T),
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cell_dir = args.output.parent / f"memory_sweep_cells_N{args.N}"
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
            "config": {
                "N": args.N, "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim, "T": args.T,
                "n_steps_grid": args.n_steps_grid, "pairs": args.pairs,
                "seed": args.seed,
            },
            "cells": cells,
        }, indent=2))

    print(f"\n[sweep] wrote {args.output} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
