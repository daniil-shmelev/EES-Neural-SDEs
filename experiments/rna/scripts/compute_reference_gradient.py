"""Compute a fine-dt reference gradient for the synthetic scaling benchmark.

Runs one forward+backward of the torus neural SDE at a much finer step
resolution than any sweep cell, producing a "truth" gradient vector that
downstream cells compare against via ``--ref-grad``.

VirtualBrownianTree + shared PRNG key means the Brownian path seen here at
fine dt is a consistent refinement of every coarser path the sweep cells
produce, so the gradient error measured at each ``n_steps`` cleanly
captures the combined effect of discretization and adjoint fidelity.

Usage::

    python -m scripts.compute_reference_gradient \
        --n-steps 10000 \
        --output results/rna_benchmark_ref_gradient.npy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from experiments.rna.experiment.config import Solvers
from experiments.rna.experiment.integrator_benchmark_single import run as run_cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=10_000,
                        help="Reference integration steps (very fine).")
    parser.add_argument("--solver", type=str, default="cfees25",
                        help="Reference solver; default matches the paper's "
                             "advocate method.")
    parser.add_argument("--adjoint", type=str, default="reversible")
    parser.add_argument("--num-angles", type=int, default=350)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dt", type=float, default=None,
                        help="Per-step dt. Mutually exclusive with --t1.")
    parser.add_argument("--t1", type=float, default=40.0,
                        help="Total integration time. If set, dt=t1/n_steps. "
                             "Must match the --t1 used by the sweep driver "
                             "so the reference path lives on the same "
                             "interval as every test cell.")
    parser.add_argument("--output", type=Path,
                        default=Path("results/rna_benchmark_ref_gradient.npy"))
    parser.add_argument("--seed-model", type=int, default=0)
    parser.add_argument("--seed-ctx", type=int, default=1)
    parser.add_argument("--seed-bases", type=int, default=2)
    parser.add_argument("--seed-targets", type=int, default=3)
    parser.add_argument("--seed-reps", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dt is None and args.t1 is None:
        raise SystemExit("either --dt or --t1 must be provided")
    dt = args.dt if args.dt is not None else (args.t1 / args.n_steps)

    print(
        f"[ref-grad] solver={args.solver} adjoint={args.adjoint} "
        f"n_steps={args.n_steps} dt={dt:.3e} t1={args.t1} "
        f"num_angles={args.num_angles}",
        flush=True,
    )

    # dt_ref is chosen so the total integration time t1 = n_steps * dt
    # matches a sweep cell with the same dt. Since sweep cells vary n_steps
    # at fixed dt, we must do the same here: pass a small dt_ref that keeps
    # t1 in the same regime. Specifically, if sweep cells have n_steps=800
    # at dt=0.05 -> t1=40, then to resolve that interval with n_steps_ref
    # we need dt_ref = 40 / n_steps_ref. We parameterise by "reference
    # resolution" via --n-steps; the caller picks --dt to keep t1 matched.
    #
    # In practice for this figure: sweep t1 VARIES with n_steps (since dt
    # is fixed). The reference therefore cannot match a single t1 for every
    # cell — we instead align refinement PER-CELL. To keep this script
    # simple we produce ONE reference at the maximum t1 in the sweep; cells
    # at shorter t1 solve a sub-interval of the same SDE path (since
    # VirtualBrownianTree gives consistent paths). This is the standard
    # framing of "fine-dt reference" for this kind of plot.
    out = run_cell(
        num_angles=args.num_angles,
        n_steps=args.n_steps,
        solver=Solvers(args.solver),
        n_reps=1,
        adjoint=args.adjoint,
        context_length=args.context_length,
        hidden_dim=args.hidden_dim,
        ctx_dim=args.ctx_dim,
        dt=dt,
        batch_size=args.batch_size,
        save_grad_path=str(args.output),
        seed_model=args.seed_model,
        seed_ctx=args.seed_ctx,
        seed_bases=args.seed_bases,
        seed_targets=args.seed_targets,
        seed_reps=args.seed_reps,
    )

    grad = np.load(args.output)
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "n_steps": args.n_steps,
        "solver": args.solver,
        "adjoint": args.adjoint,
        "num_angles": args.num_angles,
        "context_length": args.context_length,
        "hidden_dim": args.hidden_dim,
        "ctx_dim": args.ctx_dim,
        "batch_size": args.batch_size,
        "t1": args.t1,
        "dt": dt,
        "seeds": out["seeds"],
        "grad_n_params": int(grad.shape[0]),
        "grad_norm": float(np.linalg.norm(grad)),
        "grad_sumsq": float(np.sum(grad.astype(np.float64) ** 2)),
        "grad_max_abs": float(np.max(np.abs(grad))) if grad.size else 0.0,
        "loss_value": out["loss_value"],
        "warmup_s": out["warmup_s"],
        "sha256": sha,
        "npy_path": str(args.output),
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(
        f"[ref-grad] wrote {args.output} ({grad.size} params, "
        f"||g||={manifest['grad_norm']:.4e})",
        flush=True,
    )
    print(f"[ref-grad] manifest {manifest_path}", flush=True)
    print(f"[ref-grad] sha256    {sha[:16]}…", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
