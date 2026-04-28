"""Driver that runs each (solver, config) torus benchmark in a FRESH subprocess.

Isolating each measurement in its own Python process eliminates JIT-cache
contamination from prior solvers, so the RSS and HLO readings become the
actual peak working set for *that* solver on *that* configuration, not the
residual cache from the previous run. Mirrors
``../CFEES-Neural-SDEs/experiment/integrator_benchmark_isolated.py``.

Usage::

    python -m experiment.integrator_benchmark_isolated \\
        --configs 7x5,7x20,7x50,7x100,7x200,14x50,35x50,70x50,140x50 \\
        --solvers cg2 cfees25 \\
        --n-reps 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_subprocess(
    num_angles: int,
    n_steps: int,
    solver: str,
    n_reps: int,
    python_bin: str,
    *,
    adjoint: str,
    context_length: int,
    hidden_dim: int,
    ctx_dim: int,
    dt: float,
    batch_size: int,
    ref_grad: str | None = None,
) -> dict:
    cmd = [
        python_bin,
        "-m",
        "experiment.integrator_benchmark_single",
        "--num-angles", str(num_angles),
        "--n-steps", str(n_steps),
        "--solver", solver,
        "--adjoint", adjoint,
        "--n-reps", str(n_reps),
        "--context-length", str(context_length),
        "--hidden-dim", str(hidden_dim),
        "--ctx-dim", str(ctx_dim),
        "--dt", str(dt),
        "--batch-size", str(batch_size),
    ]
    if ref_grad is not None:
        cmd.extend(["--ref-grad", ref_grad])
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess failed with code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-2000:]}"
        )
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no JSON payload in stdout:\n{proc.stdout[-2000:]}")


def _parse_configs(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split("x")
        pairs.append((int(a), int(b)))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        type=str,
        default="7x5,7x10,7x20,7x50,7x100,7x200,14x50,35x50,70x50,140x50,210x50",
        help="Comma-separated DxN pairs where D=num_angles, N=n_steps.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "cfees25:reversible",
            "cg2:checkpoint_full",
            "cg2:checkpoint_recursive",
            "cg4:checkpoint_full",
            "cg4:checkpoint_recursive",
        ],
        help=(
            "Space-separated list of `solver:adjoint` combinations. Each is run "
            "as its own subprocess cell. Adjoint names match "
            "integrator_benchmark_single --adjoint: auto, reversible, direct, "
            "checkpoint_recursive (aka checkpoint_log), checkpoint_full."
        ),
    )
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Per-step dt (fixed across n_steps). Mutually "
                             "exclusive with --t1.")
    parser.add_argument("--t1", type=float, default=None,
                        help="If set, integrate to fixed final time t1 with "
                             "dt=t1/n_steps per cell. Required for the "
                             "gradient-vs-reference comparison to be "
                             "meaningful (all cells solve the same SDE on "
                             "[0, t1]).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "rna_benchmark_scaling.json",
    )
    parser.add_argument(
        "--ref-grad",
        type=str,
        default=None,
        help=(
            "Optional .npy path to a reference gradient. Pass through to each "
            "cell so it records rel_err_vs_ref / cos_sim_vs_ref. Use "
            "scripts/compute_reference_gradient.py to produce this file."
        ),
    )
    args = parser.parse_args()

    config_pairs = _parse_configs(args.configs)

    modes: list[tuple[str, str]] = []
    for m in args.modes:
        if ":" not in m:
            modes.append((m, "auto"))
        else:
            solver, adjoint = m.split(":", 1)
            modes.append((solver, adjoint))

    results: Dict[str, dict] = {}
    for num_angles, n_steps in config_pairs:
        cell_dt = (args.t1 / n_steps) if args.t1 is not None else args.dt
        for solver, adjoint in modes:
            key = f"{solver}-{adjoint}_d{num_angles}_n{n_steps}"
            print(f"[isolated] {key} dt={cell_dt:.3e} ...", end="", flush=True)
            try:
                r = run_subprocess(
                    num_angles,
                    n_steps,
                    solver,
                    args.n_reps,
                    args.python,
                    adjoint=adjoint,
                    context_length=args.context_length,
                    hidden_dim=args.hidden_dim,
                    ctx_dim=args.ctx_dim,
                    dt=cell_dt,
                    batch_size=args.batch_size,
                    ref_grad=args.ref_grad,
                )
                results[key] = r
                temp_mb = (r.get("temp_bytes") or 0) / (1 << 20)
                print(
                    f" mean={r['mean_s']:.4f}s"
                    f"  compile={r['compile_footprint_mb']:.1f}MB"
                    f"  rep+={r['rep_peak_incr_mb']:.2f}MB"
                    f"  hlo={temp_mb:.1f}MB",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                results[key] = {"error": str(e)}
                print(f" FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)

    mode_labels = [f"{s}:{a}" for s, a in modes]
    col_width = max(16, max(len(lbl) for lbl in mode_labels) + 2)
    col_fmt = f"  {{:>{col_width}s}}"
    header = (
        f"  {'d':>4s}  {'n_steps':>8s}"
        + "".join(col_fmt.format(lbl) for lbl in mode_labels)
    )
    print()
    print("=" * len(header))
    print(
        f"forward+backward mean wall clock (s) - {args.n_reps} reps after warmup, "
        "isolated subprocesses"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for num_angles, n_steps in config_pairs:
        row = f"  {num_angles:>4d}  {n_steps:>8d}"
        for solver, adjoint in modes:
            r = results.get(f"{solver}-{adjoint}_d{num_angles}_n{n_steps}", {})
            val = r.get("mean_s", float("nan"))
            row += f"  {val:>{col_width}.4f}"
        print(row)

    def _print_table(title: str, field: str, fmt: str = "{:>{w}.1f}", scale: float = 1.0) -> None:
        print()
        print(title)
        print(header)
        for num_angles, n_steps in config_pairs:
            row = f"  {num_angles:>4d}  {n_steps:>8d}"
            for solver, adjoint in modes:
                r = results.get(f"{solver}-{adjoint}_d{num_angles}_n{n_steps}", {})
                raw = r.get(field)
                if raw is None:
                    row += "  " + f"{'nan':>{col_width}}"
                else:
                    row += "  " + fmt.format(float(raw) * scale, w=col_width)
            print(row)

    _print_table(
        "compile footprint (MB) = rss(post-compile) - rss(post-model-built)",
        "compile_footprint_mb",
    )
    _print_table(
        "per-rep incremental RSS (MB) = max(rss during reps) - rss(post-compile)",
        "rep_peak_incr_mb",
        fmt="{:>{w}.2f}",
    )
    _print_table(
        "HLO compile-time scratch (MiB) from memory_analysis().temp_size_in_bytes",
        "temp_bytes",
        scale=1.0 / (1 << 20),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"\nsaved raw results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
