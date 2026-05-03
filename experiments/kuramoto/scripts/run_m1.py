"""One-shot orchestrator for milestone M1: simulator verification + data generation.

Runs (in order):
  1. `run_verification` for $n = 2$ — energy-drift + Boltzmann-stationarity
     diagnostics. Output: `results/simulator_verification_n2.{json,npz}`.
  2. `generate_one_n` for each $n \\in $ ``--n-list``. Output:
     `data/pendulum_n{n}_seed{seed}.{npz,json}` plus a GIF for the smallest
     $n$ in the list.

This is the M1 entry point on the remote machine.

Usage (local smoke):
    python -m experiments.pendulum.scripts.run_m1 --smoke

Usage (remote production, default settings):
    python -m experiments.pendulum.scripts.run_m1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.pendulum.datasets.pipeline import generate_one_n
from experiments.pendulum.datasets.simulator import SimConfig
from experiments.pendulum.datasets.verification import run_verification


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-list", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--n-fine", type=int, default=16384)
    p.add_argument("--n-obs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-sim", type=int, default=128)
    p.add_argument("--data-dir", type=Path, default=Path("experiments/pendulum/data/"))
    p.add_argument(
        "--results-dir", type=Path, default=Path("experiments/pendulum/results/")
    )
    p.add_argument("--gif", action="store_true", default=True)
    p.add_argument("--no-gif", dest="gif", action="store_false")
    p.add_argument("--gif-fps", type=int, default=30)
    p.add_argument("--gif-trail", type=int, default=40)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument(
        "--verify-n-fine-grid",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
    )
    p.add_argument("--verify-n-traj-stationarity", type=int, default=128)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny config for local debugging (n=2, 4/2/2 traj, T=0.2).",
    )
    return p.parse_args()


def _apply_smoke(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.n_list = [2]
    args.n_train, args.n_val, args.n_test = 4, 2, 2
    args.T, args.n_fine, args.n_obs, args.batch_sim = 0.2, 128, 16, 4
    args.verify_n_fine_grid = [128, 256]
    args.verify_n_traj_stationarity = 4
    return args


def main() -> int:
    args = _apply_smoke(parse_args())
    smallest_n = min(args.n_list)

    if not args.skip_verify:
        run_verification(
            n=2, T=args.T, sigma=args.sigma, gamma=args.gamma,
            n_fine_grid=args.verify_n_fine_grid,
            n_traj_stationarity=args.verify_n_traj_stationarity,
            seed=args.seed, out_dir=args.results_dir,
        )
    else:
        print("[run-m1] --skip-verify set, skipping simulator verification.")

    cfg = SimConfig(
        T=args.T, n_fine=args.n_fine, n_obs=args.n_obs,
        sigma=args.sigma, gamma=args.gamma,
    )
    for n in args.n_list:
        generate_one_n(
            n=n, cfg=cfg,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            seed=args.seed, batch_sim=args.batch_sim, out_dir=args.data_dir,
            render_gif=args.gif and (n == smallest_n),
            gif_fps=args.gif_fps, gif_trail=args.gif_trail,
        )

    print("\n[run-m1] M1 complete.")
    print(f"  Verification: {args.results_dir.resolve()}")
    print(f"  Data + GIF:   {args.data_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
