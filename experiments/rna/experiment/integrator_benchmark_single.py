"""Run one (solver, config) benchmark of the torus neural SDE and dump JSON.

Intended to be launched as a fresh subprocess so peak-memory measurement is
not contaminated by JIT caches from other solver runs in the same process.
Mirrors ``../CFEES-Neural-SDEs/experiment/integrator_benchmark_single.py``
adapted to the torus model.

Usage::

    python -m experiment.integrator_benchmark_single \
        --num-angles 7 --n-steps 50 --solver cfees25 --n-reps 10
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import psutil

from experiments.rna.experiment.config import Solvers
from experiments.rna.experiment.factories import build_adjoint, build_solver
from experiments.rna.models.torus_nsde import TorusNeuralSDE


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def _hlo_temp_bytes(grad_fn, *args, n_compiles: int = 15) -> int | None:
    """Compile the gradient graph N times and return the minimum XLA scratch.

    XLA's ``memory_analysis().temp_size_in_bytes`` is not fully deterministic
    across repeated ``.compile()`` calls — the compiler's heuristic
    tile/fusion choices can land on different local optima on different
    invocations, which shows up as measurement noise of ~0.1-0.5 MiB at
    our model size. Taking the minimum across ``n_compiles`` calls gives
    the tightest bound on the true required scratch. ``.lower()`` is
    deterministic (produces the same HLO every call) so it runs once
    outside the loop; only ``.compile()`` is repeated.
    """
    try:
        lowered = grad_fn.lower(*args)
    except Exception:  # noqa: BLE001
        return None
    best: int | None = None
    for _ in range(max(1, n_compiles)):
        try:
            compiled = lowered.compile()
            t = int(compiled.compiled.memory_analysis().temp_size_in_bytes)
        except Exception:  # noqa: BLE001
            continue
        if best is None or t < best:
            best = t
    return best


def _flatten_grads(grads) -> jnp.ndarray:
    """Flatten the inexact leaves of a gradient pytree into a 1-D vector."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    if not leaves:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.concatenate([jnp.ravel(jnp.asarray(x)) for x in leaves])


def run(
    num_angles: int,
    n_steps: int,
    solver: Solvers,
    n_reps: int,
    *,
    adjoint: str = "auto",
    context_length: int = 20,
    hidden_dim: int = 128,
    ctx_dim: int = 64,
    dt: float = 0.05,
    batch_size: int = 32,
    ref_grad_path: str | None = None,
    save_grad_path: str | None = None,
    seed_model: int = 0,
    seed_ctx: int = 1,
    seed_bases: int = 2,
    seed_targets: int = 3,
    seed_reps: int = 42,
) -> dict:
    """Benchmark one ``(solver, num_angles, n_steps)`` forward+backward pass.

    Returns a dict containing:
    - ``compile_footprint_mb``: ``rss(post-compile) - rss(post-model-built)``
    - ``rep_peak_incr_mb``: peak additional RSS during timed reps
    - ``rep_peak_rss_mb``: absolute peak RSS during reps
    - ``rss_stage_{a,b,c,d}_mb``: RSS snapshots at successive stages
    - ``warmup_s`` and summary statistics of ``n_reps`` timed steps
    - ``temp_bytes``: XLA compile-time scratch (if available)
    """
    rss_stage_a = rss_mb()

    model = TorusNeuralSDE(
        num_angles=num_angles,
        hidden_dim=hidden_dim,
        ctx_dim=ctx_dim,
        n_steps=n_steps,
        dt=dt,
        solver=build_solver(solver),
        diffusion_scale=0.1,
        adjoint=build_adjoint(adjoint, n_steps),
        key=jax.random.key(seed_model),
    )
    context_angles = jax.random.uniform(
        jax.random.key(seed_ctx),
        (batch_size, context_length, num_angles),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )
    context_bases = jax.random.randint(
        jax.random.key(seed_bases),
        (batch_size, context_length, 1),
        minval=0,
        maxval=4,
    )
    target_bases = jax.random.randint(
        jax.random.key(seed_targets), (batch_size, 1), minval=0, maxval=4
    )
    # future_bases_window=0: future encoder is skipped, these are just stubs.
    future_bases = jnp.zeros((batch_size, 0, 1), dtype=jnp.int32)
    future_mask = jnp.zeros((batch_size, 0), dtype=jnp.int8)
    jax.block_until_ready(context_angles)

    gc.collect()
    rss_stage_b = rss_mb()  # model + data on device, pre-compile

    def loss_fn(
        m: eqx.Module,
        ctx_a: jax.Array,
        ctx_b: jax.Array,
        tgt_b: jax.Array,
        fut_b: jax.Array,
        fut_m: jax.Array,
        k: jax.Array,
    ) -> jax.Array:
        sample_keys = jax.random.split(k, ctx_a.shape[0])
        preds = jax.vmap(
            lambda a, cb, tb, fb, fm, sk: m(a, cb, tb, fb, fm, sk)
        )(ctx_a, ctx_b, tgt_b, fut_b, fut_m, sample_keys)
        return jnp.sum(preds**2)

    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
    rep_keys = jax.random.split(jax.random.key(seed_reps), n_reps + 1)

    temp_bytes = _hlo_temp_bytes(
        grad_fn,
        model,
        context_angles,
        context_bases,
        target_bases,
        future_bases,
        future_mask,
        rep_keys[0],
    )

    t0 = time.time()
    loss, grads = grad_fn(
        model,
        context_angles,
        context_bases,
        target_bases,
        future_bases,
        future_mask,
        rep_keys[0],
    )
    jax.block_until_ready((loss, grads))
    t_warmup = time.time() - t0

    # Gradient fingerprint for cross-adjoint accuracy comparison. These
    # scalars are deterministic for fixed (solver, adjoint, seeds, batch),
    # and (up to float32 rounding) should agree across adjoints that
    # differentiate the same discretized SDE.
    flat_grad = _flatten_grads(grads)
    flat_grad_np = np.asarray(jax.device_get(flat_grad))
    loss_value = float(loss)
    grad_norm = float(np.linalg.norm(flat_grad_np))
    grad_sumsq = float(np.sum(flat_grad_np.astype(np.float64) ** 2))
    grad_max_abs = float(np.max(np.abs(flat_grad_np))) if flat_grad_np.size else 0.0
    grad_n_params = int(flat_grad_np.shape[0])

    rel_err_vs_ref: float | None = None
    cos_sim_vs_ref: float | None = None
    ref_grad_sha: str | None = None
    if ref_grad_path is not None and os.path.exists(ref_grad_path):
        with open(ref_grad_path, "rb") as fh:
            ref_bytes = fh.read()
        ref_grad_sha = hashlib.sha256(ref_bytes).hexdigest()[:16]
        ref = np.load(io.BytesIO(ref_bytes)).astype(np.float64)
        if ref.shape != flat_grad_np.shape:
            raise ValueError(
                f"reference gradient shape {ref.shape} != current "
                f"gradient shape {flat_grad_np.shape}. The reference was "
                "computed with a different model config."
            )
        g64 = flat_grad_np.astype(np.float64)
        ref_norm = float(np.linalg.norm(ref))
        g_norm = float(np.linalg.norm(g64))
        rel_err_vs_ref = float(np.linalg.norm(g64 - ref) / ref_norm)
        cos_sim_vs_ref = float(np.dot(g64, ref) / (ref_norm * g_norm)) if g_norm > 0 else 0.0

    if save_grad_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_grad_path)) or ".", exist_ok=True)
        np.save(save_grad_path, flat_grad_np.astype(np.float32))

    gc.collect()
    rss_stage_c = rss_mb()  # post-compile, post-warmup

    times: list[float] = []
    rep_peak_rss = rss_stage_c
    for i in range(n_reps):
        t0 = time.time()
        loss, grads = grad_fn(
            model,
            context_angles,
            context_bases,
            target_bases,
            future_bases,
            future_mask,
            rep_keys[i + 1],
        )
        jax.block_until_ready((loss, grads))
        times.append(time.time() - t0)
        rep_peak_rss = max(rep_peak_rss, rss_mb())

    rss_stage_d = rss_mb()  # after all reps

    return {
        "num_angles": num_angles,
        "n_steps": n_steps,
        "solver": solver.value,
        "adjoint": adjoint,
        "context_length": context_length,
        "hidden_dim": hidden_dim,
        "ctx_dim": ctx_dim,
        "dt": dt,
        "batch_size": batch_size,
        "warmup_s": float(t_warmup),
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "n_reps": n_reps,
        "times_s": [float(t) for t in times],
        "compile_footprint_mb": float(rss_stage_c - rss_stage_b),
        "rep_peak_incr_mb": float(rep_peak_rss - rss_stage_c),
        "rep_peak_rss_mb": float(rep_peak_rss),
        "rss_stage_a_mb": float(rss_stage_a),
        "rss_stage_b_mb": float(rss_stage_b),
        "rss_stage_c_mb": float(rss_stage_c),
        "rss_stage_d_mb": float(rss_stage_d),
        "temp_bytes": temp_bytes,
        # --- accuracy fingerprint on the warmup forward+backward ---
        "loss_value": loss_value,
        "grad_norm": grad_norm,
        "grad_sumsq": grad_sumsq,
        "grad_max_abs": grad_max_abs,
        "grad_n_params": grad_n_params,
        "rel_err_vs_ref": rel_err_vs_ref,
        "cos_sim_vs_ref": cos_sim_vs_ref,
        "ref_grad_sha": ref_grad_sha,
        "ref_grad_path": ref_grad_path,
        "seeds": {
            "model": seed_model,
            "ctx": seed_ctx,
            "bases": seed_bases,
            "targets": seed_targets,
            "reps": seed_reps,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-angles", type=int, required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--solver", type=str, required=True)
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument(
        "--adjoint",
        type=str,
        default="auto",
        choices=(
            "auto",
            "reversible",
            "direct",
            "checkpoint_recursive",
            "checkpoint_log",  # deprecated alias for checkpoint_recursive
            "checkpoint_full",
        ),
        help=(
            "auto: ReversibleAdjoint for reversible solvers, DirectAdjoint otherwise. "
            "checkpoint_recursive: RecursiveCheckpointAdjoint() with diffrax's default "
            "Stumm-Walther online treeverse (O(sqrt(n)) checkpoints). "
            "checkpoint_full: RecursiveCheckpointAdjoint(checkpoints=max_steps), "
            "i.e. one snapshot per step -> O(n) activation tape. "
            "checkpoint_log is a deprecated alias for checkpoint_recursive."
        ),
    )
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--ref-grad",
        type=str,
        default=None,
        help=(
            "Optional .npy path to a reference gradient vector. If supplied, "
            "rel_err_vs_ref and cos_sim_vs_ref are computed against it. The "
            "reference must have been produced with the same model dims."
        ),
    )
    parser.add_argument(
        "--save-grad",
        type=str,
        default=None,
        help=(
            "Optional .npy path where this cell's flattened gradient vector "
            "is saved. Used by scripts/compute_reference_gradient.py."
        ),
    )
    parser.add_argument("--seed-model", type=int, default=0)
    parser.add_argument("--seed-ctx", type=int, default=1)
    parser.add_argument("--seed-bases", type=int, default=2)
    parser.add_argument("--seed-targets", type=int, default=3)
    parser.add_argument("--seed-reps", type=int, default=42)
    args = parser.parse_args()
    out = run(
        args.num_angles,
        args.n_steps,
        Solvers(args.solver),
        args.n_reps,
        adjoint=args.adjoint,
        context_length=args.context_length,
        hidden_dim=args.hidden_dim,
        ctx_dim=args.ctx_dim,
        dt=args.dt,
        batch_size=args.batch_size,
        ref_grad_path=args.ref_grad,
        save_grad_path=args.save_grad,
        seed_model=args.seed_model,
        seed_ctx=args.seed_ctx,
        seed_bases=args.seed_bases,
        seed_targets=args.seed_targets,
        seed_reps=args.seed_reps,
    )
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
