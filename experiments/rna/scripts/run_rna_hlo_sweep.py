"""Compile-time XLA scratch memory sweep for the RNA torus neural SDE.

For each ``(solver, n_steps, residues_per_state)`` cell we lower the
forward+backward graph of one training step and read
``compiled.compiled.memory_analysis().temp_size_in_bytes``. This is the
XLA scratch buffer that the compiled JIT region pre-allocates — a clean
measurement that excludes weights, optimiser state, and dataset buffers.
Each cell runs in its own subprocess so that nothing carries over
between cells.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _measure_one_cell_inproc(
    solver: str,
    n_steps: int,
    residues_per_state: int,
    *,
    batch_size: int,
    context_length: int,
    hidden_dim: int,
    ctx_dim: int,
    seed: int,
    dt: float,
) -> dict:
    """Build the model, lower one train step, return the HLO memory analysis."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from experiment.factories import build_solver, make_prediction_fn
    from experiment.rna_losses import make_wrapped_mse_loss
    from experiment.config import Solvers
    from models.torus_nsde import TorusNeuralSDE

    num_angles = 7 * residues_per_state
    key = jax.random.key(seed)
    model = TorusNeuralSDE(
        num_angles=num_angles,
        hidden_dim=hidden_dim,
        ctx_dim=ctx_dim,
        n_steps=n_steps,
        dt=dt,
        solver=build_solver(Solvers(solver)),
        diffusion_scale=0.1,
        key=key,
    )

    prediction_fn = make_prediction_fn()
    loss_fn = make_wrapped_mse_loss(
        input_key="context_angles",
        target_key="target_angles",
        prediction_fn=prediction_fn,
    )

    batch = {
        "context_angles": jnp.zeros((batch_size, context_length, num_angles), dtype=jnp.float32),
        "target_angles": jnp.zeros((batch_size, num_angles), dtype=jnp.float32),
    }
    mask = jnp.ones((batch_size,), dtype=jnp.bool_)
    step_key = jax.random.key(seed + 1)

    @eqx.filter_jit
    def grad_step(m, b, ms, k):
        return eqx.filter_value_and_grad(loss_fn)(m, b, ms, k)

    compiled = grad_step.lower(model, batch, mask, step_key).compile()
    inner = compiled.compiled
    ma = inner.memory_analysis()
    return {
        "solver": solver,
        "n_steps": n_steps,
        "residues_per_state": residues_per_state,
        "num_angles": num_angles,
        "batch_size": batch_size,
        "context_length": context_length,
        "hidden_dim": hidden_dim,
        "ctx_dim": ctx_dim,
        "temp_bytes": int(ma.temp_size_in_bytes),
        "argument_bytes": int(ma.argument_size_in_bytes),
        "output_bytes": int(ma.output_size_in_bytes),
    }


def _run_subprocess(
    *,
    solver: str,
    n_steps: int,
    residues_per_state: int,
    batch_size: int,
    context_length: int,
    hidden_dim: int,
    ctx_dim: int,
    seed: int,
    dt: float,
) -> dict:
    code = (
        "import json, sys\n"
        "from scripts.run_rna_hlo_sweep import _measure_one_cell_inproc\n"
        f"out = _measure_one_cell_inproc({solver!r}, {n_steps}, {residues_per_state},"
        f" batch_size={batch_size}, context_length={context_length},"
        f" hidden_dim={hidden_dim}, ctx_dim={ctx_dim}, seed={seed}, dt={dt})\n"
        "sys.stdout.write('METRIC ' + json.dumps(out) + '\\n')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        tail = result.stderr.splitlines()[-20:]
        oom = any(
            "RESOURCE_EXHAUSTED" in line or "out of memory" in line.lower()
            for line in tail
        )
        return {
            "solver": solver,
            "n_steps": n_steps,
            "residues_per_state": residues_per_state,
            "num_angles": 7 * residues_per_state,
            "oom": bool(oom),
            "returncode": result.returncode,
            "error_tail": tail,
        }
    record: dict | None = None
    for line in result.stdout.splitlines():
        if line.startswith("METRIC "):
            record = json.loads(line[len("METRIC ") :])
    if record is None:
        return {
            "solver": solver,
            "n_steps": n_steps,
            "residues_per_state": residues_per_state,
            "num_angles": 7 * residues_per_state,
            "oom": False,
            "returncode": result.returncode,
            "error_tail": result.stdout.splitlines()[-20:],
        }
    record["oom"] = False
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--solvers", nargs="+", default=["cfees25", "cg2"])
    parser.add_argument("--n-steps", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--n-steps-fixed", type=int, default=20)
    parser.add_argument("--k-fixed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = args.output or (
        PROJECT_ROOT
        / "results"
        / f"rna_hlo_sweep_{datetime.now().strftime('%H%M%S_%Y%m%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[str, int, int]] = []
    for solver in args.solvers:
        for n_steps in args.n_steps:
            cells.append((solver, n_steps, args.k_fixed))
        for k in args.k:
            if k == args.k_fixed:
                continue
            cells.append((solver, args.n_steps_fixed, k))

    print(f"running {len(cells)} HLO cells -> {out_path}", flush=True)
    records: list[dict] = []
    for solver, n_steps, k in cells:
        print(f"  {solver} n_steps={n_steps} k={k}", flush=True)
        record = _run_subprocess(
            solver=solver,
            n_steps=n_steps,
            residues_per_state=k,
            batch_size=args.batch_size,
            context_length=args.context_length,
            hidden_dim=args.hidden_dim,
            ctx_dim=args.ctx_dim,
            seed=args.seed,
            dt=args.dt,
        )
        records.append(record)
        if record.get("oom"):
            print(f"    OOM", flush=True)
        else:
            mb = record.get("temp_bytes", 0) / 2**20
            print(f"    temp_bytes={mb:.2f} MiB", flush=True)

    payload = {
        "args": {
            k: list(v) if isinstance(v, list) else v
            for k, v in vars(args).items()
            if k != "output"
        },
        "cells": records,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
