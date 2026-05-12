"""M3 pilot: gradient fidelity + 3-adjoint training parity for the Kuramoto NSDE.

Three sub-stages, each gated by the previous:

  1. **Single-batch gradient fidelity** (cheap): for fixed (model, data,
     BM key), compute the gradient under each adjoint and report relative
     L2 vs the fine-dt reference produced by `compute_reference_gradient.py`.
     Pass criterion: float32 < 1e-3 across all adjoints; float64 < 1e-4.

  2. **Three-adjoint training parity** (expensive): same model, RNG seed,
     data; vary only the adjoint. Train 5 epochs at the production
     `n_steps`. Report final test loss for each adjoint x seed combo.
     Pass criterion: 95% CI on test loss overlaps between Reversible and
     full-checkpoint.

  3. **Long-horizon stress test**: re-run gradient fidelity at
     `n_steps in {1000, 5000}`. Pass criterion: rel-L2 stays below the
     M3.1 threshold.

Outputs `results/pilot/pilot_adjoint_parity.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from experiments.kuramoto.experiment.config import (
    ExperimentConfig,
    ModelKind,
    Solvers,
)
from experiments.kuramoto.experiment.factories import build_adjoint, build_solver, make_loader, make_model
from experiments.kuramoto.experiment.losses import make_multi_horizon_energy_score
from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE


ADJOINTS = ("reversible", "checkpoint_full", "checkpoint_recursive")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=Path("experiments/kuramoto/data/"))
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--reference-grad",
                   type=Path,
                   default=Path("experiments/kuramoto/results/pilot/reference_gradient.npy"))
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--n-steps-stress", type=int, nargs="+", default=[1000, 5000])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dtype", type=str, default="float32",
                   choices=("float32", "float64"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/kuramoto/results/pilot/"))
    p.add_argument("--skip-training-parity", action="store_true",
                   help="Run only the cheap gradient-fidelity sub-stages (M3.1 + M3.3).")
    return p.parse_args()


def _flatten_grads(grads) -> np.ndarray:
    flat, _ = jax.tree.flatten(eqx.filter(grads, eqx.is_array))
    return np.concatenate([np.asarray(jax.device_get(x)).ravel() for x in flat])


def gradient_fidelity_one(
    N: int,
    data_dir: Path,
    data_seed: int,
    n_steps: int,
    batch_size: int,
    hidden_dim: int,
    adjoint_name: str,
    seed_key: int,
    seed_model: int,
) -> tuple[np.ndarray, float]:
    """Compute one (model, batch, key) gradient under the requested adjoint."""
    dataset = KuramotoDataset(
        npz_path=default_npz_path(data_dir, N, data_seed), split="train",
    )
    T = float(dataset.t_grid[-1])
    n_obs = dataset.n_obs
    dt = T / n_steps

    model_key = jr.PRNGKey(seed_model)
    solver = build_solver(Solvers.CFEES25)
    adjoint = build_adjoint(adjoint_name, n_steps)
    model = KuramotoNSDE(
        N=N, hidden_dim=hidden_dim, n_steps=n_steps, dt=dt, n_obs=n_obs,
        solver=solver, diffusion_scale=0.3, adjoint=adjoint, key=model_key,
    )

    arrays = dataset.as_array_dict()
    n_take = batch_size
    batch = {k: jnp.asarray(v[:n_take]) for k, v in arrays.items()}
    loss_fn = make_multi_horizon_energy_score(n_samples=4)

    @eqx.filter_jit
    def grad_step(model, batch, key):
        return eqx.filter_grad(loss_fn)(model, batch, jnp.ones((n_take,)), key)

    t0 = time.perf_counter()
    grads = grad_step(model, batch, jr.PRNGKey(seed_key))
    arr = _flatten_grads(grads)
    return arr, float(time.perf_counter() - t0)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def stage_gradient_fidelity(args, ref_grad: np.ndarray, n_steps: int) -> dict:
    """M3.1 / M3.3 - single-batch gradient comparison vs the reference."""
    print(f"\n[pilot] === gradient fidelity, n_steps={n_steps}, dtype={args.dtype} ===")
    out = {"n_steps": n_steps, "dtype": args.dtype, "results": {}}
    for adj in ADJOINTS:
        grad, walltime = gradient_fidelity_one(
            N=args.N, data_dir=args.data_dir, data_seed=args.data_seed,
            n_steps=n_steps, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
            adjoint_name=adj, seed_key=args.seeds[0] + 1000, seed_model=args.seeds[0],
        )
        rel = relative_l2(grad, ref_grad)
        out["results"][adj] = {"rel_l2": rel, "wall_clock_s": walltime}
        print(f"[pilot] adj={adj} n_params={grad.size} rel_l2={rel:.3e} t={walltime:.2f}s",
              flush=True)
    return out


def stage_training_parity(args) -> dict:
    """M3.2 - short training run per (adjoint, seed), report final test loss."""
    print(f"\n[pilot] === 3-adjoint training parity (epochs={args.epochs}) ===")
    from experiments.kuramoto.experiment.train_kuramoto import fit
    from experiments.kuramoto.experiment.losses import make_eval_metrics

    base_cfg = ExperimentConfig(
        N=args.N,
        data_dir=str(args.data_dir),
        data_seed=args.data_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        n_steps=args.n_steps,
        solver=Solvers.CFEES25,
        model=ModelKind.KURAMOTO_NSDE,
        save_every_epoch=False,
    )

    out = {"n_steps": args.n_steps, "epochs": args.epochs, "results": {}}
    for adj in ADJOINTS:
        for seed in args.seeds:
            cfg = ExperimentConfig(**{**base_cfg.__dict__, "seed": seed, "adjoint": adj})
            run_dir = args.output_dir / f"trainparity_{adj}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            _, dataset = make_loader(cfg, "train")
            model_key = jr.PRNGKey(seed)
            model = make_model(cfg, dataset.metadata(), model_key)
            loss_fn = make_multi_horizon_energy_score(
                horizons=cfg.loss_horizons, n_samples=cfg.loss_n_samples,
            )
            metric_fn = make_eval_metrics(
                horizons=cfg.loss_horizons, n_samples=cfg.loss_n_samples,
            )
            t0 = time.perf_counter()
            _, history = fit(
                model, loss_fn=loss_fn, metric_fn=metric_fn,
                config=cfg, output_dir=run_dir,
            )
            wall = float(time.perf_counter() - t0)
            out["results"].setdefault(adj, []).append({
                "seed": int(seed),
                "final_test_loss": float(history["test_loss"][-1]),
                "final_val_loss": float(history["val_loss"][-1]),
                "final_train_loss": float(history["train_loss"][-1]),
                "wall_clock_s": wall,
            })
            print(f"[pilot] adj={adj} seed={seed} test={history['test_loss'][-1]:.4f} t={wall:.0f}s",
                  flush=True)
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    if not args.reference_grad.exists():
        raise SystemExit(
            f"Reference gradient not found at {args.reference_grad}. "
            f"Run `python -m experiments.kuramoto.scripts.compute_reference_gradient "
            f"--N {args.N} --dtype {args.dtype}` first."
        )
    ref_grad = np.load(args.reference_grad)
    print(f"[pilot] loaded reference gradient: n_params={ref_grad.size}", flush=True)

    summary = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    summary["m3_1_fidelity_at_train_n_steps"] = stage_gradient_fidelity(args, ref_grad, args.n_steps)
    summary["m3_3_fidelity_long_horizon"] = [
        stage_gradient_fidelity(args, ref_grad, n_s) for n_s in args.n_steps_stress
    ]
    if not args.skip_training_parity:
        summary["m3_2_training_parity"] = stage_training_parity(args)

    out_path = args.output_dir / "pilot_adjoint_parity.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[pilot] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
