"""Measure one isolated forward/backward pass of the canonical training model and energy-score loss."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

from experiments.kuramoto.experiment.config import (
    DEFAULT_CONFIG,
    PROTOCOL,
    Solvers,
    load_config,
    serialize_config,
)


def _set_xla_alloc_env():
    """Cap XLA's pre-allocation so multiple cells don't fight for the whole
    GPU and we still get usable `device.memory_stats()` output."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--solver", type=str, required=True, choices=tuple(Solvers))
    p.add_argument(
        "--adjoint",
        type=str,
        required=True,
        choices=("reversible", "checkpoint_full", "checkpoint_recursive"),
    )
    p.add_argument("--n-steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    _set_xla_alloc_env()
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output path: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    overrides = dict(
        solver=Solvers(args.solver), adjoint=args.adjoint, n_steps=args.n_steps
    )
    for name in ("N", "batch_size", "seed"):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    config = replace(config, **overrides)

    summary = {
        "protocol": PROTOCOL,
        "config": json.loads(serialize_config(config)),
        "N": config.N,
        "solver": config.solver.value,
        "adjoint": config.adjoint,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "drift_kind": config.drift_kind,
        "hidden_dim": config.hidden_dim,
        "seed": config.seed,
        "oom": False,
        "error": None,
        "wall_clock_s_mean": None,
        "wall_clock_s_std": None,
        "peak_bytes": None,
        "n_params": None,
        "measurement": "isolated gradient of the training loss; excludes optimiser and resident full datasets",
    }

    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jax.random as jr

        from experiments.kuramoto.experiment.benchmark import prepare_benchmark

        device = jax.devices()[0]
        model, batch, loss, metadata = prepare_benchmark(config)
        summary["T"] = metadata["T"]
        summary["n_obs"] = metadata["n_obs"]
        summary["n_params"] = sum(
            int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))
        )
        mask = jnp.ones(config.batch_size)
        key = jr.key(config.seed + 1)

        @eqx.filter_jit
        def step(model, batch, key):
            return eqx.filter_grad(loss)(model, batch, mask, key)

        # warmup (compile + first run)
        for _ in range(args.n_warmup):
            g = step(model, batch, key)
            jax.block_until_ready(g)

        # wall-clock
        times = []
        for _ in range(args.n_reps):
            t0 = time.perf_counter()
            g = step(model, batch, key)
            jax.block_until_ready(g)
            times.append(time.perf_counter() - t0)
        summary["wall_clock_s_mean"] = float(sum(times) / len(times))
        summary["wall_clock_s_std"] = float(
            (sum((t - summary["wall_clock_s_mean"]) ** 2 for t in times) / len(times))
            ** 0.5
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
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("error",) or v is not None}
        )
    )
    gc.collect()
    return 1 if summary["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
