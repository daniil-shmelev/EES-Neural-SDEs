"""Run one random torus memory benchmark cell."""

from __future__ import annotations

import argparse
import json
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.torus.models import RandomTorusNeuralSDE
from experiments.torus.solvers import Solvers, build_adjoint, build_solver


def run(
    *,
    solver: Solvers,
    adjoint: str,
    num_angles: int,
    n_steps: int,
    n_reps: int,
    batch_size: int,
    context_length: int,
    hidden_dim: int,
    ctx_dim: int,
    dt: float,
    seed_model: int,
    seed_context: int,
    seed_labels: int,
    seed_target_labels: int,
    seed_reps: int,
) -> dict:
    model = RandomTorusNeuralSDE(
        num_angles=num_angles,
        hidden_dim=hidden_dim,
        ctx_dim=ctx_dim,
        n_steps=n_steps,
        dt=dt,
        solver=build_solver(solver),
        adjoint=build_adjoint(adjoint, n_steps),
        key=jax.random.key(seed_model),
    )
    context_angles = jax.random.uniform(
        jax.random.key(seed_context),
        (batch_size, context_length, num_angles),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )
    context_labels = jax.random.randint(
        jax.random.key(seed_labels),
        (batch_size, context_length, 1),
        minval=0,
        maxval=4,
    )
    target_labels = jax.random.randint(
        jax.random.key(seed_target_labels),
        (batch_size, 1),
        minval=0,
        maxval=4,
    )

    def loss_fn(m, ctx_angles, ctx_labels, tgt_labels, key):
        sample_keys = jax.random.split(key, ctx_angles.shape[0])
        preds = jax.vmap(lambda a, cl, tl, k: m(a, cl, tl, k))(
            ctx_angles,
            ctx_labels,
            tgt_labels,
            sample_keys,
        )
        return jnp.sum(preds**2)

    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
    rep_keys = jax.random.split(jax.random.key(seed_reps), n_reps + 1)
    lowered = grad_fn.lower(model, context_angles, context_labels, target_labels, rep_keys[0])
    compiled = lowered.compile()
    temp_bytes = int(compiled.compiled.memory_analysis().temp_size_in_bytes)

    start = time.time()
    loss, grads = grad_fn(model, context_angles, context_labels, target_labels, rep_keys[0])
    jax.block_until_ready((loss, grads))
    warmup_s = time.time() - start

    times = []
    for key in rep_keys[1:]:
        start = time.time()
        loss, grads = grad_fn(model, context_angles, context_labels, target_labels, key)
        jax.block_until_ready((loss, grads))
        times.append(time.time() - start)

    return {
        "solver": solver.value,
        "adjoint": adjoint,
        "num_angles": num_angles,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "context_length": context_length,
        "hidden_dim": hidden_dim,
        "ctx_dim": ctx_dim,
        "dt": dt,
        "temp_bytes": temp_bytes,
        "warmup_s": float(warmup_s),
        "mean_s": float(np.mean(times)) if times else None,
        "std_s": float(np.std(times)) if times else None,
        "n_reps": n_reps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=[s.value for s in Solvers], default="cfees25")
    parser.add_argument("--adjoint", default="reversible")
    parser.add_argument("--num-angles", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--n-reps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed-model", type=int, default=0)
    parser.add_argument("--seed-context", type=int, default=1)
    parser.add_argument("--seed-labels", type=int, default=2)
    parser.add_argument("--seed-target-labels", type=int, default=3)
    parser.add_argument("--seed-reps", type=int, default=42)
    args = parser.parse_args(argv)

    record = run(
        solver=Solvers(args.solver),
        adjoint=args.adjoint,
        num_angles=args.num_angles,
        n_steps=args.n_steps,
        n_reps=args.n_reps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        hidden_dim=args.hidden_dim,
        ctx_dim=args.ctx_dim,
        dt=args.dt,
        seed_model=args.seed_model,
        seed_context=args.seed_context,
        seed_labels=args.seed_labels,
        seed_target_labels=args.seed_target_labels,
        seed_reps=args.seed_reps,
    )
    sys.stdout.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
