"""Gradient-error sweep for the stochastic-volatility neural SDE.

For each ``(experiment, nfe_budget, seed)`` cell we build the production
``SimpleNeuralSDE`` model used by ``train.py``, pull a fixed batch from the
chosen experiment's training set, and compute the truncated-signature loss
gradient w.r.t. the model parameters once with ``ReversibleAdjoint`` and
once with ``RecursiveCheckpointAdjoint``. Both calls share the same
``VirtualBrownianTree`` key, so any gradient discrepancy reflects only the
adjoint method, not noise realisation.

Output layout (under ``--output-dir``)::

    summary.json                    # aggregate stats per (experiment, nfe_budget)
    cells/<experiment>/nfe<nfe>_seed<seed>.npz  # full flat g_auto, g_rev, scalars
    eval_inputs.json                # snapshot of the cell config

The companion ``plot_grad_error.py`` consumes ``summary.json``.

Usage (NOT auto-run)::

    python -m experiments.stochastic_volatility.scripts.grad_error_sweep \
        --experiments rough_bergomi \
        --nfe-budgets 12,24,48,96,132 \
        --seeds 0,1,2 \
        --output-dir experiments/stochastic_volatility/results/grad_error/single_step
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from experiments.stochastic_volatility.experiment.config import (
    Devices,
    Experiments,
    Solvers,
    make_config,
)
from experiments.stochastic_volatility.experiment.factories import (
    make_loader,
    make_model,
    make_sample_fn,
)
from experiments.stochastic_volatility.experiment.losses import truncated_sig_loss
from experiments.stochastic_volatility.scripts._adjoint_helpers import (
    AdjointNeuralSDE,
    AdjointKind,
)


DEFAULT_NFE_BUDGETS = (12, 24, 48, 96, 132)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_EXPERIMENT = Experiments.ROUGH_BERGOMI
DEFAULT_SOLVER = Solvers.EES25


def _flatten_grads(grad_pytree) -> jax.Array:
    """Concatenate every Array leaf of an Equinox grad pytree into a 1-D vector."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(grad_pytree, eqx.is_array))
    if not leaves:
        return jnp.zeros(0)
    return jnp.concatenate([leaf.reshape(-1) for leaf in leaves])


def _build_loss_fn(state_dim: int, sample_fn):
    """Truncated signature MMD on (predictions, targets), as in train.py."""
    inner = truncated_sig_loss(depth=6, ambient_dim=state_dim)

    def loss_fn(model, batch, mask, key):
        predictions, targets = sample_fn(model, batch, mask, key)
        return inner(predictions, targets)

    return loss_fn


def _grad_for_adjoint(
    inner_model,
    sample_fn_factory,
    loss_fn,
    batch,
    mask,
    key: jax.Array,
    adjoint_kind: AdjointKind,
) -> tuple[float, jax.Array]:
    """Wrap ``inner_model`` in an :class:`AdjointNeuralSDE` of the requested
    adjoint and return ``(loss, flat-gradient)``.
    """
    wrapped = AdjointNeuralSDE(inner_model, adjoint_kind=adjoint_kind)
    sample_fn = sample_fn_factory(wrapped)

    def loss_with(model_):
        return loss_fn(model_, batch, mask, key)

    # We differentiate w.r.t. the AdjointNeuralSDE wrapper; its only
    # differentiable leaves live in ``wrapped.inner`` (the static
    # ``adjoint_kind`` is filtered out by eqx).
    loss, grads = eqx.filter_value_and_grad(loss_with)(wrapped)
    return float(loss), _flatten_grads(grads)


def _run_one_cell(
    *,
    experiment: Experiments,
    nfe_budget: int,
    seed: int,
    base_config_overrides: dict,
) -> dict:
    config = make_config(
        experiment=experiment,
        epochs=1,
        batch_size=base_config_overrides["batch_size"],
        learning_rate=1e-3,
        seed=seed,
        device=Devices.GPU,
        hidden_dim=base_config_overrides["hidden_dim"],
        nfe_budget=nfe_budget,
        total_time=base_config_overrides["total_time"],
        solver=DEFAULT_SOLVER,
        diffusion_scale=base_config_overrides["diffusion_scale"],
        skip_plots=True,
    )

    key = jax.random.key(seed)
    model_key, batch_key, fwd_key = jax.random.split(key, 3)

    inner_model = make_model(config, model_key)
    state_dim = inner_model.state_dim

    train_loader = make_loader(config, "train")
    train_state = train_loader.init_state(batch_key)
    batch, _, mask = train_loader.next(train_state)

    def sample_fn_factory(model):
        def sample_fn(m, b, msk, key):
            batch_size = b["solution"].shape[0]
            y0s = jnp.ones((batch_size, m.state_dim), dtype=jnp.float32)
            keys = jax.random.split(key, batch_size)
            predictions = jax.vmap(m)(y0s, keys)
            return predictions, b["solution"]
        return sample_fn

    loss_fn = _build_loss_fn(state_dim, sample_fn_factory(None))

    t0 = time.perf_counter()
    loss_auto, g_auto = _grad_for_adjoint(
        inner_model, sample_fn_factory, loss_fn, batch, mask, fwd_key, "autograd",
    )
    g_auto = jax.device_get(g_auto)
    wall_auto = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss_rev, g_rev = _grad_for_adjoint(
        inner_model, sample_fn_factory, loss_fn, batch, mask, fwd_key, "reversible",
    )
    g_rev = jax.device_get(g_rev)
    wall_rev = time.perf_counter() - t0

    g_auto_norm = float(np.linalg.norm(g_auto))
    g_rev_norm = float(np.linalg.norm(g_rev))
    diff_norm = float(np.linalg.norm(g_rev - g_auto))
    rel_err = diff_norm / max(g_auto_norm, 1e-30)
    denom = max(g_auto_norm * g_rev_norm, 1e-30)
    cos_sim = float(np.dot(g_rev, g_auto) / denom)

    return {
        "experiment": str(experiment),
        "nfe_budget": int(nfe_budget),
        "seed": int(seed),
        "n_steps": int(inner_model.solve_n_steps),
        "dt": float(inner_model.solve_dt),
        "rel_err": rel_err,
        "cos_sim": cos_sim,
        "g_auto_norm": g_auto_norm,
        "g_rev_norm": g_rev_norm,
        "loss_auto": loss_auto,
        "loss_rev": loss_rev,
        "wall_auto": wall_auto,
        "wall_rev": wall_rev,
        "g_auto": g_auto,
        "g_rev": g_rev,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/stochastic_volatility/results/grad_error/single_step"))
    p.add_argument("--experiments", type=str, default=str(DEFAULT_EXPERIMENT),
                   help="Comma-separated list of stoch-vol experiments.")
    p.add_argument("--nfe-budgets", type=str, default=",".join(str(x) for x in DEFAULT_NFE_BUDGETS),
                   help="Comma-separated list of NFE budgets to sweep.")
    p.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    p.add_argument("--batch-size", type=int, default=512,
                   help="Mini-batch size for the gradient comparison "
                        "(smaller than training batch_size=4096 to keep cell cost low).")
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--total-time", type=float, default=1.0)
    p.add_argument("--diffusion-scale", type=float, default=0.2)
    args = p.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    experiments = [Experiments(e.strip()) for e in args.experiments.split(",") if e.strip()]
    nfe_budgets = [int(x) for x in args.nfe_budgets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    base_overrides = {
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "total_time": args.total_time,
        "diffusion_scale": args.diffusion_scale,
    }

    print(f"[stochvol-grad] experiments={experiments}")
    print(f"[stochvol-grad] nfe_budgets={nfe_budgets}")
    print(f"[stochvol-grad] seeds={seeds}")
    print(f"[stochvol-grad] base_overrides={base_overrides}")

    rel_err_table = {str(e): {nfe: [] for nfe in nfe_budgets} for e in experiments}
    cos_sim_table = {str(e): {nfe: [] for nfe in nfe_budgets} for e in experiments}
    g_auto_norm_table = {str(e): {nfe: [] for nfe in nfe_budgets} for e in experiments}
    g_rev_norm_table = {str(e): {nfe: [] for nfe in nfe_budgets} for e in experiments}

    total_cells = len(experiments) * len(nfe_budgets) * len(seeds)
    cell_idx = 0
    overall_t0 = time.perf_counter()

    for experiment in experiments:
        exp_cells_dir = cells_dir / str(experiment)
        exp_cells_dir.mkdir(parents=True, exist_ok=True)
        for nfe in nfe_budgets:
            for seed in seeds:
                cell_idx += 1
                cell = _run_one_cell(
                    experiment=experiment, nfe_budget=nfe, seed=seed,
                    base_config_overrides=base_overrides,
                )
                cell_path = exp_cells_dir / f"nfe{nfe:04d}_seed{seed}.npz"
                np.savez_compressed(
                    cell_path,
                    g_auto=cell["g_auto"], g_rev=cell["g_rev"],
                    rel_err=cell["rel_err"], cos_sim=cell["cos_sim"],
                    g_auto_norm=cell["g_auto_norm"], g_rev_norm=cell["g_rev_norm"],
                    loss_auto=cell["loss_auto"], loss_rev=cell["loss_rev"],
                    wall_auto=cell["wall_auto"], wall_rev=cell["wall_rev"],
                    nfe_budget=nfe, n_steps=cell["n_steps"], dt=cell["dt"],
                    seed=seed, experiment=str(experiment),
                )

                rel_err_table[str(experiment)][nfe].append(cell["rel_err"])
                cos_sim_table[str(experiment)][nfe].append(cell["cos_sim"])
                g_auto_norm_table[str(experiment)][nfe].append(cell["g_auto_norm"])
                g_rev_norm_table[str(experiment)][nfe].append(cell["g_rev_norm"])

                print(
                    f"[stochvol-grad {cell_idx:>3}/{total_cells}] {experiment} "
                    f"nfe={nfe} seed={seed} rel_err={cell['rel_err']:.3e} "
                    f"cos_sim={cell['cos_sim']:.6f} dt={cell['dt']:.4e}",
                    flush=True,
                )

    def _stats(table) -> dict:
        out: dict = {}
        for e, by_nfe in table.items():
            block = {}
            for nfe, vals in by_nfe.items():
                arr = np.array(vals, dtype=float)
                if arr.size:
                    block[str(nfe)] = {
                        "mean": float(arr.mean()),
                        "stderr": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
                        "raw": arr.tolist(),
                    }
                else:
                    block[str(nfe)] = {"mean": None, "stderr": None, "raw": []}
            out[e] = block
        return out

    summary = {
        "experiments": [str(e) for e in experiments],
        "nfe_budgets": nfe_budgets,
        "seeds": seeds,
        "base_overrides": base_overrides,
        "rel_err": _stats(rel_err_table),
        "cos_sim": _stats(cos_sim_table),
        "g_auto_norm": _stats(g_auto_norm_table),
        "g_rev_norm": _stats(g_rev_norm_table),
        "elapsed_seconds": time.perf_counter() - overall_t0,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    manifest = {
        "summary_sha256": _sha256(summary_path),
        "n_cells": total_cells,
        "jax_version": jax.__version__,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[stochvol-grad] wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
