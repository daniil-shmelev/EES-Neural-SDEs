"""Hyperparameter grid search to calibrate the N=1000 CF-EES(2,5) Kuramoto run.

Scaling the model from N<=8 to N=1000 destabilised training: the gradient norm
explodes around epoch 3 (as the warmup LR reaches its peak) and the loss
diverges. This sweep runs CF-EES(2,5)+Reversible for a few epochs across a grid
of the most likely levers -- learning rate, gradient-clip norm, hidden width and
drift depth -- at N=1000, to find a stable, well-performing configuration before
committing to the full multi-seed run.

Each grid point is one config (one PBS array task via --index). Runs are
resumable and skip-if-already-complete. Use --summarize to rank finished points.

  # inspect the grid (its size sets the PBS array range):
  python -m experiments.kuramoto.scripts.run_calibration --list

  # one grid point (what each array task runs):
  python -m experiments.kuramoto.scripts.run_calibration --index 5 \
      --config experiments/kuramoto/configs/kuramoto_runtime_parity.toml \
      --epochs 8 --output-base experiments/kuramoto/results/calibration_N1000/

  # rank results once they finish:
  python -m experiments.kuramoto.scripts.run_calibration --summarize \
      --output-base experiments/kuramoto/results/calibration_N1000/
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path


# --- Grid. Array size = product of the list lengths (printed by --list). ---
# Edit freely; if you change the sizes, update `#PBS -J 0-<n-1>` in
# hpc/calibrate_kuramoto.pbs to match.
GRID: dict[str, list] = {
    "lr": [1e-3, 3e-4, 1e-4, 3e-5],
    "grad_clip_norm": [0.1, 1.0],
    "hidden_dim": [128, 256],
    "drift_depth": [3, 4],
}

# Held fixed for every grid point (matches the real EES cell at N=1000).
FIXED: dict[str, object] = {
    "N": 1000, "solver": "cfees25", "adjoint": "reversible",
    "n_steps": 50, "batch_size": 64, "seed": 0,
}

# config key -> train_kuramoto CLI flag
FLAG = {
    "N": "--N", "solver": "--solver", "adjoint": "--adjoint",
    "n_steps": "--n-steps", "batch_size": "--batch-size", "seed": "--seed",
    "lr": "--lr", "grad_clip_norm": "--grad-clip-norm",
    "hidden_dim": "--hidden-dim", "drift_depth": "--drift-depth",
    "diffusion_depth": "--diffusion-depth", "diffusion_scale": "--diffusion-scale",
}
SHORT = {
    "lr": "lr", "grad_clip_norm": "gc", "hidden_dim": "h", "drift_depth": "d",
    "diffusion_depth": "dd", "diffusion_scale": "ds", "batch_size": "bs",
    "n_steps": "ns",
}


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def grid_points() -> list[dict]:
    keys = list(GRID.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]


def slug(point: dict) -> str:
    return "_".join(f"{SHORT.get(k, k)}{_fmt(v)}" for k, v in point.items())


def _cmd(config: Path, out_dir: Path, epochs: int, point: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "experiments.kuramoto.experiment.train_kuramoto",
        str(config), "--output-dir", str(out_dir),
        "--epochs", str(epochs), "--auto-resume",
    ]
    for k, v in {**FIXED, **point}.items():
        cmd += [FLAG[k], str(v)]
    return cmd


def run_point(index: int, config: Path, epochs: int, out_base: Path) -> int:
    points = grid_points()
    if not (0 <= index < len(points)):
        print(f"[calib] index {index} out of range [0, {len(points)})", flush=True)
        return 1
    point = points[index]
    out_dir = out_base / slug(point)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics.json").exists():
        print(f"[calib] SKIP idx={index} {slug(point)} (already complete)", flush=True)
        return 0

    cmd = _cmd(config, out_dir, epochs, point)
    print(f"[calib] === idx={index} {slug(point)} | {point} ===", flush=True)
    t0 = time.perf_counter()
    with open(out_dir / "train.log", "ab") as fh:  # append: resumed attempts keep the log
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    dt = time.perf_counter() - t0
    print(f"[calib] idx={index} {slug(point)} exit={proc.returncode} t={dt:.0f}s", flush=True)
    return proc.returncode


def summarize(out_base: Path) -> int:
    rows = []
    for index, point in enumerate(grid_points()):
        d = out_base / slug(point)
        mp, hp = d / "metrics.json", d / "history.json"
        final_test = best_val = max_g = None
        status = "incomplete"
        if mp.exists():
            m = json.loads(mp.read_text())
            final_test = m.get("final_test_loss")
            best_val = m.get("best_val_loss")
            status = "stable"
            if hp.exists():
                gnorms = json.loads(hp.read_text()).get("epoch_grad_norm_max", []) or []
                finite = [g for g in gnorms if g is not None and math.isfinite(g)]
                max_g = max(finite) if finite else math.inf
                exploded = any(g is None or not math.isfinite(g) or g >= 1e6 for g in gnorms)
                bad_loss = final_test is None or not math.isfinite(final_test)
                if exploded or bad_loss:
                    status = "DIVERGED"
        rows.append((index, slug(point), final_test, best_val, max_g, status))

    def _key(r):
        ft = r[2]
        ok = isinstance(ft, (int, float)) and math.isfinite(ft)
        return (r[5] != "stable", ft if ok else math.inf)

    rows.sort(key=_key)
    print(f"\n{'idx':>3}  {'config':<30} {'test':>10} {'best_val':>10} {'max|g|':>11}  status")
    print("-" * 82)
    for index, name, ft, bv, mg, st in rows:
        ft_s = f"{ft:.2f}" if isinstance(ft, (int, float)) and math.isfinite(ft) else "--"
        bv_s = f"{bv:.2f}" if isinstance(bv, (int, float)) and math.isfinite(bv) else "--"
        mg_s = f"{mg:.2e}" if isinstance(mg, (int, float)) and math.isfinite(mg) else "--"
        print(f"{index:>3}  {name:<30} {ft_s:>10} {bv_s:>10} {mg_s:>11}  {st}")
    done = sum(1 for r in rows if r[5] != "incomplete")
    print(f"\n{done}/{len(rows)} configs complete. Pick the lowest-test 'stable' row.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, default=None, help="Grid index to run (PBS array task).")
    p.add_argument("--summarize", action="store_true", help="Rank completed grid points and exit.")
    p.add_argument("--list", action="store_true", help="Print the grid size + slugs and exit.")
    p.add_argument("--config", type=Path,
                   default=Path("experiments/kuramoto/configs/kuramoto_runtime_parity.toml"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--output-base", type=Path,
                   default=Path("experiments/kuramoto/results/calibration_N1000/"))
    args = p.parse_args()

    points = grid_points()
    if args.list or (args.index is None and not args.summarize):
        print(f"grid size = {len(points)}  ->  PBS array 0-{len(points) - 1}")
        for i, pt in enumerate(points):
            print(f"  {i:>3}  {slug(pt)}")
        if args.index is None and not args.summarize:
            print("\nRun one point with --index <i>, or --summarize to rank results.")
        return 0

    if args.summarize:
        return summarize(args.output_base)

    args.output_base.mkdir(parents=True, exist_ok=True)
    return run_point(args.index, args.config, args.epochs, args.output_base)


if __name__ == "__main__":
    sys.exit(main())
