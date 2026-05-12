"""Compute a fine-dt reference gradient for the Kuramoto NSDE pilot.

Runs one forward+backward of the Kuramoto NSDE at a fine integration
resolution under CFEES25 + ReversibleAdjoint, producing a "truth"
gradient vector that the M3 pilot compares against to measure
discretization-plus-adjoint fidelity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.kuramoto.datasets.kuramoto_dataset import (
    KuramotoDataset,
    default_npz_path,
)
from experiments.kuramoto.experiment.factories import build_adjoint, build_solver
from experiments.kuramoto.experiment.config import Solvers
from experiments.kuramoto.experiment.losses import make_multi_horizon_energy_score
from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=Path("experiments/kuramoto/data/"))
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--n-steps-ref", type=int, default=10000,
                   help="Fine reference integration steps over the trajectory horizon.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed-model", type=int, default=0)
    p.add_argument("--seed-key", type=int, default=42)
    p.add_argument("--dtype", type=str, default="float32",
                   choices=("float32", "float64"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/kuramoto/results/pilot/reference_gradient.npy"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    dataset = KuramotoDataset(
        npz_path=default_npz_path(args.data_dir, args.N, args.data_seed),
        split="train",
    )
    T = float(dataset.t_grid[-1])
    n_obs = dataset.n_obs
    dt_ref = T / args.n_steps_ref

    model_key = jr.PRNGKey(args.seed_model)
    solver = build_solver(Solvers.CFEES25)
    adjoint = build_adjoint("reversible", args.n_steps_ref)

    model = KuramotoNSDE(
        N=args.N,
        hidden_dim=args.hidden_dim,
        n_steps=args.n_steps_ref,
        dt=dt_ref,
        n_obs=n_obs,
        solver=solver,
        diffusion_scale=0.3,
        adjoint=adjoint,
        key=model_key,
    )

    arrays = dataset.as_array_dict()
    n_take = args.batch_size
    batch = {k: jnp.asarray(v[:n_take]) for k, v in arrays.items()}

    loss_fn = make_multi_horizon_energy_score(n_samples=4)

    key = jr.PRNGKey(args.seed_key)
    print(
        f"[ref-grad] N={args.N} dt_ref={dt_ref:.3e} n_steps_ref={args.n_steps_ref} "
        f"batch={args.batch_size} dtype={args.dtype}",
        flush=True,
    )

    @eqx.filter_jit
    def grad_step(model, batch, key):
        return eqx.filter_grad(loss_fn)(model, batch, jnp.ones((n_take,)), key)

    grads = grad_step(model, batch, key)
    flat, _ = jax.tree.flatten(eqx.filter(grads, eqx.is_array))
    arr = np.concatenate([np.asarray(jax.device_get(x)).ravel() for x in flat])
    np.save(args.output, arr)
    print(f"[ref-grad] wrote {args.output} (n_params={arr.size})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
