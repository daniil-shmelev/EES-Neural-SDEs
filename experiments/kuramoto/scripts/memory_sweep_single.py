"""Single-cell memory benchmark for one Kuramoto memory-sweep cell.

Each cell runs in its own subprocess so JAX state is isolated:

  1. Builds a fresh Kuramoto NSDE at the requested config (random init).
  2. Runs one forward+backward of a synthetic energy-score loss on a random
     batch.
  3. Reports peak XLA scratch memory and wall-clock time.

Cells that OOM are caught and the JSON output records `oom: true` rather
than crashing the parent driver.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _set_xla_alloc_env():
    """Cap XLA's pre-allocation so multiple cells don't fight for the whole
    GPU and we still get usable `device.memory_stats()` output."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--solver", type=str, required=True,
                   choices=("cfees25", "cfees27", "cg2", "cg4", "rkmk"))
    p.add_argument("--adjoint", type=str, required=True,
                   choices=("reversible", "checkpoint_full",
                            "checkpoint_recursive"))
    p.add_argument("--n-steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    _set_xla_alloc_env()
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "N": args.N, "solver": args.solver, "adjoint": args.adjoint,
        "n_steps": args.n_steps, "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim, "T": args.T, "seed": args.seed,
        "oom": False, "error": None,
        "wall_clock_s_mean": None, "wall_clock_s_std": None,
        "peak_bytes": None, "n_params": None,
    }

    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from experiments.kuramoto.experiment.factories import build_adjoint, build_solver
        from experiments.kuramoto.experiment.config import Solvers
        from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE

        device = jax.devices()[0]
        dt = args.T / args.n_steps
        n_obs = 64

        model_key = jr.PRNGKey(args.seed)
        solver = build_solver(Solvers(args.solver))
        adjoint = build_adjoint(args.adjoint, args.n_steps)
        model = KuramotoNSDE(
            N=args.N, hidden_dim=args.hidden_dim,
            n_steps=args.n_steps, dt=dt, n_obs=n_obs,
            solver=solver, diffusion_scale=0.3, adjoint=adjoint,
            key=model_key,
        )
        n_params = sum(int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
        summary["n_params"] = n_params

        bk = jr.split(jr.PRNGKey(args.seed + 1), args.batch_size)
        theta0 = jr.uniform(jr.PRNGKey(2), (args.batch_size, args.N),
                             minval=-jnp.pi, maxval=jnp.pi)
        omega0 = 0.5 * jr.normal(jr.PRNGKey(3), (args.batch_size, args.N))

        @eqx.filter_jit
        def step(model, theta0, omega0, keys):
            def per_traj(model, th0, om0, k):
                th, om = model(th0, om0, k)
                return jnp.mean(jnp.sin(th) ** 2 + om ** 2)
            def loss(model):
                vals = jax.vmap(lambda th, om, k: per_traj(model, th, om, k))(
                    theta0, omega0, keys)
                return jnp.mean(vals)
            return eqx.filter_grad(loss)(model)

        # warmup (compile + first run)
        for _ in range(args.n_warmup):
            g = step(model, theta0, omega0, bk)
            jax.block_until_ready(g)

        # wall-clock
        times = []
        for _ in range(args.n_reps):
            t0 = time.perf_counter()
            g = step(model, theta0, omega0, bk)
            jax.block_until_ready(g)
            times.append(time.perf_counter() - t0)
        summary["wall_clock_s_mean"] = float(sum(times) / len(times))
        summary["wall_clock_s_std"] = float(
            (sum((t - summary["wall_clock_s_mean"]) ** 2 for t in times) / len(times)) ** 0.5
        )

        try:
            ms = device.memory_stats() or {}
            summary["peak_bytes"] = int(ms.get("peak_bytes_in_use", 0)) or None
            summary["bytes_in_use"] = int(ms.get("bytes_in_use", 0)) or None
            summary["bytes_limit"] = int(ms.get("bytes_limit", 0)) or None
        except Exception as exc:
            summary["error_memstats"] = str(exc)

    except Exception as exc:
        msg = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        summary["oom"] = "out of memory" in msg.lower() or "RESOURCE_EXHAUSTED" in msg
        summary["error"] = msg

    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("error",) or v is not None}))
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
