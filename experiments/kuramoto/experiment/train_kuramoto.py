"""Training entrypoint for the Kuramoto NSDE.

Supports resuming training from a previous run via ``--resume-from <dir>``:
the model, optimiser state, and history are loaded from the specified
directory, and ``--epochs`` is interpreted as the *total* (resumed +
new) epoch count, so the warmup-cosine LR schedule lines up correctly.
The new output directory must differ from the resume source.

The fit loop persists everything we might want to plot later:

- per-epoch train and val loss (energy score)
- per-epoch test loss (re-evaluated each epoch on the test split)
- per-epoch wrapped-MAE on theta and MAE on omega, broken down by horizon
- per-epoch elapsed wall-clock
- learning-rate schedule values
- the model checkpoint at each epoch (`model_epoch_{i:03d}.eqx`)
- the best-by-val model (`model_best.eqx`)
- the final model (`model_final.eqx`)
- a JSON history dump (`history.json`) and a sidecar config (`config.json`)
- a sample of test predictions (`predictions_demo.npz`) for plotting
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.kuramoto.experiment.config import (
    ExperimentConfig,
    ModelKind,
    Solvers,
    load_config,
    serialize_config,
)
from experiments.kuramoto.experiment.factories import make_loader, make_model
from experiments.kuramoto.experiment.losses import (
    LossFn,
    PyTree,
    _kuramoto_predict_batch,
    make_eval_metrics,
    make_multi_horizon_energy_score,
)
from experiments.kuramoto.experiment.runtime import (
    make_output_dir,
    save_json,
    save_npz,
)


def _eval_split(
    model: eqx.Module,
    loader,
    loader_next,
    eval_step,
    metric_fn,
    state,
    key: jax.Array,
) -> tuple[float, dict[str, float], jax.Array, Any]:
    total_loss = 0.0
    metric_accum: dict[str, list[float]] = {}
    for _ in range(loader.steps_per_epoch):
        key, k1, k2 = jax.random.split(key, 3)
        batch, state, mask = loader_next(state)
        loss = float(eval_step(model, batch, mask, k1))
        total_loss += loss
        metrics = metric_fn(model, batch, k2)
        for name, value in metrics.items():
            metric_accum.setdefault(name, []).append(np.asarray(jax.device_get(value)))
    avg_loss = total_loss / loader.steps_per_epoch
    avg_metrics: dict[str, Any] = {}
    for name, vals in metric_accum.items():
        stacked = np.stack(vals, axis=0)
        avg_metrics[name] = stacked.mean(axis=0).tolist() if stacked.ndim > 1 else float(stacked.mean())
    return avg_loss, avg_metrics, key, state


def _latest_checkpoint(d: Path) -> tuple[Path, Path]:
    """Best-available (model, opt_state) checkpoint pair in ``d``.

    Prefers the end-of-run ``*_final.eqx``; otherwise falls back to the most
    recent per-epoch checkpoint (``model_epoch_NNN.eqx`` + ``opt_state_latest.eqx``),
    which is what a mid-training (timeout-killed) run leaves behind.
    """
    final_model, final_opt = d / "model_final.eqx", d / "opt_state_final.eqx"
    if final_model.exists() and final_opt.exists():
        return final_model, final_opt
    epoch_ckpts = sorted(d.glob("model_epoch_*.eqx"))
    latest_opt = d / "opt_state_latest.eqx"
    if epoch_ckpts and latest_opt.exists():
        return epoch_ckpts[-1], latest_opt
    raise FileNotFoundError(
        f"no resumable checkpoint in {d} (need model_final+opt_state_final "
        f"or model_epoch_*+opt_state_latest)"
    )


def fit(
    model: eqx.Module,
    *,
    loss_fn: LossFn,
    metric_fn,
    config: ExperimentConfig,
    output_dir: Path,
    resume_from: Path | None = None,
) -> tuple[eqx.Module, dict[str, list[float]]]:
    """Train ``model`` for ``config.epochs`` epochs total.

    If ``resume_from`` is given, load ``model_final.eqx``,
    ``opt_state_final.eqx``, and ``history.json`` from that directory and
    pick up where they left off. ``config.epochs`` is interpreted as the
    *total* (resumed + new) epoch count; the warmup-cosine LR schedule
    is built with ``decay_steps = total_epochs * steps_per_epoch`` so
    the loaded optimiser state's step counter continues to index into
    the same schedule curve.
    """
    train_loader, _ = make_loader(config, "train")
    val_loader, _ = make_loader(config, "val")
    test_loader, _ = make_loader(config, "test")
    train_next = jax.jit(train_loader.next)
    val_next = jax.jit(val_loader.next)
    test_next = jax.jit(test_loader.next)

    # Warmup-cosine LR schedule: linear ramp from 1e-5 -> peak across the
    # first 200 steps, then cosine decay to 1e-5 across the rest of
    # training. Mitigates early-step gradient blow-up that occasionally
    # produced ``grad_norm = inf`` in M5.
    schedule_steps = max(2, config.epochs * train_loader.steps_per_epoch)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=config.learning_rate,
        warmup_steps=min(200, max(1, schedule_steps // 2)),
        decay_steps=schedule_steps,
        end_value=1e-5,
    )
    optim = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(schedule),
    )

    if resume_from is not None:
        model_ckpt, opt_ckpt = _latest_checkpoint(resume_from)
        model = eqx.tree_deserialise_leaves(model_ckpt, model)
        opt_state_template = optim.init(eqx.filter(model, eqx.is_array))
        opt_state = eqx.tree_deserialise_leaves(opt_ckpt, opt_state_template)
        print(f"resume checkpoint: {model_ckpt.name} + {opt_ckpt.name}", flush=True)
    else:
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(current_model, opt_state, batch, mask, key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(current_model, batch, mask, key)
        # Global gradient L2 norm (pre-clip) for diagnostics.
        flat_grads = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        grad_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in flat_grads))
        update_ok = jnp.isfinite(grad_norm)
        updates, candidate_opt_state = optim.update(grads, opt_state, current_model)
        candidate_model = eqx.apply_updates(current_model, updates)
        # If the gradient is non-finite, leave model + opt_state unchanged
        # (skip the step). Operates over the array partitions so the
        # static fields of the equinox modules are passed through verbatim.
        cand_arr, model_static = eqx.partition(candidate_model, eqx.is_array)
        cur_arr, _ = eqx.partition(current_model, eqx.is_array)
        merged_arr = jax.tree.map(
            lambda new, old: jnp.where(update_ok, new, old), cand_arr, cur_arr,
        )
        new_model = eqx.combine(merged_arr, model_static)
        new_opt_state = jax.tree.map(
            lambda new, old: jnp.where(update_ok, new, old),
            candidate_opt_state, opt_state,
        )
        return new_model, new_opt_state, loss, grad_norm

    eval_step = eqx.filter_jit(loss_fn)

    key = jax.random.key(config.seed)
    key, train_key, val_key, test_key = jax.random.split(key, 4)
    train_state = train_loader.init_state(train_key)
    val_state = val_loader.init_state(val_key)
    test_state = test_loader.init_state(test_key)

    n_params = sum(int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))

    if resume_from is not None:
        import json as _json
        with (resume_from / "history.json").open() as f:
            history = _json.load(f)
        epoch_offset = len(history.get("train_loss", []))
        prior_val = history.get("val_loss", []) or [float("inf")]
        best_val_loss = float(min(prior_val))
        history["n_params"] = n_params
        history["n_train_steps_per_epoch"] = train_loader.steps_per_epoch
        history.setdefault("resumed_from", []).append(str(resume_from))
        print(
            f"resuming from {resume_from} (prior epochs: {epoch_offset}, "
            f"prior best val loss: {best_val_loss:.4f})",
            flush=True,
        )
    else:
        history = {
            # Per-epoch reductions
            "train_loss": [], "val_loss": [], "test_loss": [],
            "epoch_walltime_s": [],
            "val_theta_mae_mean": [], "val_omega_mae_mean": [],
            "test_theta_mae_mean": [], "test_omega_mae_mean": [],
            "val_theta_mae_per_horizon": [], "val_omega_mae_per_horizon": [],
            "test_theta_mae_per_horizon": [], "test_omega_mae_per_horizon": [],
            "epoch_train_loss_std": [], "epoch_grad_norm_mean": [],
            "epoch_grad_norm_max": [], "epoch_grad_norm_p95": [],
            "epoch_step_time_mean_s": [], "epoch_step_time_p95_s": [],
            "epoch_peak_bytes": [], "epoch_bytes_in_use": [],
            # Fine-grained per-step traces (one entry per training step, every epoch)
            "step_train_loss": [], "step_grad_norm": [], "step_time_s": [],
            "step_epoch_index": [],
            # Constants
            "n_params": n_params, "n_train_steps_per_epoch": train_loader.steps_per_epoch,
        }
        epoch_offset = 0
        best_val_loss = float("inf")

    best_model = model
    device = jax.devices()[0]

    def _peak_mem() -> tuple[int | None, int | None]:
        try:
            ms = device.memory_stats() or {}
            return (int(ms.get("peak_bytes_in_use", 0)) or None,
                    int(ms.get("bytes_in_use", 0)) or None)
        except Exception:
            return None, None

    if epoch_offset >= config.epochs:
        # Resumed run is already complete (e.g. killed between the final epoch
        # and metrics.json). Skip the empty training loop and finalize below.
        print(
            f"resume: {epoch_offset} epochs already done >= target "
            f"{config.epochs}; finalizing without further training.",
            flush=True,
        )
    for epoch in range(epoch_offset, config.epochs):
        epoch_t0 = time.time()
        per_step_loss: list[float] = []
        per_step_grad: list[float] = []
        per_step_time: list[float] = []

        for _ in range(train_loader.steps_per_epoch):
            key, step_key = jax.random.split(key)
            batch, train_state, mask = train_next(train_state)
            t_step = time.perf_counter()
            model, opt_state, loss, gnorm = train_step(model, opt_state, batch, mask, step_key)
            jax.block_until_ready(loss)
            dt_step = time.perf_counter() - t_step

            per_step_loss.append(float(loss))
            per_step_grad.append(float(gnorm))
            per_step_time.append(dt_step)

        train_loss = float(np.mean(per_step_loss))
        history["train_loss"].append(train_loss)
        history["epoch_train_loss_std"].append(float(np.std(per_step_loss)))
        history["epoch_grad_norm_mean"].append(float(np.mean(per_step_grad)))
        history["epoch_grad_norm_max"].append(float(np.max(per_step_grad)))
        history["epoch_grad_norm_p95"].append(float(np.percentile(per_step_grad, 95)))
        history["epoch_step_time_mean_s"].append(float(np.mean(per_step_time)))
        history["epoch_step_time_p95_s"].append(float(np.percentile(per_step_time, 95)))
        history["step_train_loss"].extend(per_step_loss)
        history["step_grad_norm"].extend(per_step_grad)
        history["step_time_s"].extend(per_step_time)
        history["step_epoch_index"].extend([epoch] * len(per_step_loss))

        val_loss, val_metrics, key, val_state = _eval_split(
            model, val_loader, val_next, eval_step, metric_fn, val_state, key,
        )
        test_loss, test_metrics, key, test_state = _eval_split(
            model, test_loader, test_next, eval_step, metric_fn, test_state, key,
        )

        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["val_theta_mae_mean"].append(val_metrics["theta_mae_mean"])
        history["val_omega_mae_mean"].append(val_metrics["omega_mae_mean"])
        history["val_theta_mae_per_horizon"].append(val_metrics["theta_mae_per_horizon"])
        history["val_omega_mae_per_horizon"].append(val_metrics["omega_mae_per_horizon"])
        history["test_theta_mae_mean"].append(test_metrics["theta_mae_mean"])
        history["test_omega_mae_mean"].append(test_metrics["omega_mae_mean"])
        history["test_theta_mae_per_horizon"].append(test_metrics["theta_mae_per_horizon"])
        history["test_omega_mae_per_horizon"].append(test_metrics["omega_mae_per_horizon"])

        peak, in_use = _peak_mem()
        history["epoch_peak_bytes"].append(peak)
        history["epoch_bytes_in_use"].append(in_use)

        epoch_walltime = float(time.time() - epoch_t0)
        history["epoch_walltime_s"].append(epoch_walltime)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            eqx.tree_serialise_leaves(output_dir / "model_best.eqx", best_model)
            eqx.tree_serialise_leaves(output_dir / "opt_state_best.eqx", opt_state)

        if config.save_every_epoch:
            eqx.tree_serialise_leaves(
                output_dir / f"model_epoch_{epoch:03d}.eqx", model
            )
            # Resumability: keep only the most-recent opt_state to save disk.
            eqx.tree_serialise_leaves(
                output_dir / "opt_state_latest.eqx", opt_state
            )

        save_json(output_dir / "history.json", history)

        peak_mib = (peak / 2**20) if peak else float("nan")
        print(
            f"epoch={epoch + 1}/{config.epochs} "
            f"train={train_loss:.4f}+/-{history['epoch_train_loss_std'][-1]:.3f} "
            f"val={val_loss:.4f} test={test_loss:.4f} "
            f"|g|={history['epoch_grad_norm_mean'][-1]:.3f} "
            f"theta_mae={val_metrics['theta_mae_mean']:.4f} "
            f"omega_mae={val_metrics['omega_mae_mean']:.4f} "
            f"peak={peak_mib:.0f}MiB time={epoch_walltime:.1f}s",
            flush=True,
        )

    eqx.tree_serialise_leaves(output_dir / "model_final.eqx", model)
    eqx.tree_serialise_leaves(output_dir / "opt_state_final.eqx", opt_state)
    return best_model, history


def predict_demo(
    model: eqx.Module,
    config: ExperimentConfig,
    n_samples_per_ic: int = 8,
    n_ic: int = 4,
    seed_offset: int = 100,
) -> dict[str, np.ndarray]:
    """Sample multiple stochastic forecasts from a few held-out initial conditions.

    Used by the plotting layer to draw sample-trajectory + ensemble-fan
    figures. Saves both the data ground-truth trajectories and the
    model-sampled ones for direct comparison.
    """
    test_loader, dataset = make_loader(config, "test")
    test_next = jax.jit(test_loader.next)

    key = jax.random.key(config.seed + seed_offset)
    key, loader_key = jax.random.split(key)
    state = test_loader.init_state(loader_key)
    batch, _, _ = test_next(state)

    n_ic = min(n_ic, batch["theta0"].shape[0])

    # Slice the first n_ic initial conditions.
    sliced = {k: v[:n_ic] for k, v in batch.items()}

    @eqx.filter_jit
    def one_pass(sk):
        return _kuramoto_predict_batch(model, sliced, sk)

    sample_keys = jax.random.split(key, n_samples_per_ic)
    sample_thetas, sample_omegas = jax.vmap(one_pass)(sample_keys)

    return {
        "theta0": np.asarray(sliced["theta0"]),
        "omega0": np.asarray(sliced["omega0"]),
        "target_theta": np.asarray(sliced["theta_traj"]),
        "target_omega": np.asarray(sliced["omega_traj"]),
        "sample_theta": np.asarray(jax.device_get(sample_thetas)),
        "sample_omega": np.asarray(jax.device_get(sample_omegas)),
        "t_grid": np.asarray(dataset.t_grid),
    }


def _override_from_args(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    overrides: dict[str, Any] = {}
    if args.solver is not None:
        overrides["solver"] = Solvers(args.solver)
    if args.adjoint is not None:
        overrides["adjoint"] = str(args.adjoint)
    if args.n_steps is not None:
        overrides["n_steps"] = int(args.n_steps)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.N is not None:
        overrides["N"] = int(args.N)
    if args.model is not None:
        overrides["model"] = ModelKind(args.model)
    if args.lr is not None:
        overrides["learning_rate"] = float(args.lr)
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = int(args.hidden_dim)
    if args.drift_depth is not None:
        overrides["drift_depth"] = int(args.drift_depth)
    if args.diffusion_depth is not None:
        overrides["diffusion_depth"] = int(args.diffusion_depth)
    if args.grad_clip_norm is not None:
        overrides["grad_clip_norm"] = float(args.grad_clip_norm)
    if args.diffusion_scale is not None:
        overrides["diffusion_scale"] = float(args.diffusion_scale)
    if not overrides:
        return config
    return dataclasses.replace(config, **overrides)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to experiment config TOML")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument(
        "--adjoint", type=str, default=None,
        choices=("auto", "reversible", "direct", "checkpoint_recursive",
                 "checkpoint_log", "checkpoint_full"),
    )
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument(
        "--model", type=str, default=None,
        choices=("kuramoto_nsde", "euclidean_baseline"),
    )
    parser.add_argument("--lr", type=float, default=None)
    # Architecture / regularisation overrides (used by the calibration sweep).
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--drift-depth", type=int, default=None)
    parser.add_argument("--diffusion-depth", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--diffusion-scale", type=float, default=None)
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Resume from a prior run dir's model_final.eqx + opt_state_final.eqx + "
             "history.json. Interpret --epochs as the *total* (resumed + new) count.",
    )
    parser.add_argument(
        "--auto-resume", action="store_true",
        help="If --output-dir already holds checkpoints, resume in-place from the "
             "latest one (for restarting after a timeout); no-op on a fresh dir.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = _override_from_args(config, args)

    _, train_dataset = make_loader(config, "train")
    metadata = train_dataset.metadata()

    model_key = jax.random.key(config.seed)
    model = make_model(config, metadata, model_key)

    loss_fn = make_multi_horizon_energy_score(
        horizons=config.loss_horizons, n_samples=config.loss_n_samples,
    )
    metric_fn = make_eval_metrics(
        horizons=config.loss_horizons, n_samples=config.loss_n_samples,
    )

    tag = f"N{config.N}_{config.model.value}_{config.solver.value}_{config.adjoint}_seed{config.seed}"
    output_dir = args.output_dir or make_output_dir(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(serialize_config(config))

    print(
        f"dataset: N={metadata['N']} train={metadata['split_size']} "
        f"T={metadata['T']:.2f}s n_obs={metadata['n_obs']}",
        flush=True,
    )
    print(
        f"model: {config.model.value} hidden={config.hidden_dim} "
        f"n_steps={config.n_steps} solver={config.solver.value} "
        f"adjoint={config.adjoint} dt={metadata['T']/config.n_steps:.4f}",
        flush=True,
    )
    print(f"output_dir: {output_dir}", flush=True)

    if args.resume_from is not None and args.resume_from.resolve() == output_dir.resolve():
        raise ValueError(
            "--resume-from must differ from --output-dir; use --auto-resume for "
            "in-place resume after a timeout."
        )

    resume_from = args.resume_from
    if args.auto_resume and resume_from is None:
        # In-place resume only when a usable checkpoint is present and the run was
        # not already finalized (metrics.json absent => a prior attempt was cut).
        has_ckpt = (output_dir / "history.json").exists() and (
            (output_dir / "model_final.eqx").exists()
            or any(output_dir.glob("model_epoch_*.eqx"))
        )
        if has_ckpt:
            resume_from = output_dir
            print(f"auto-resume: continuing in-place from {output_dir}", flush=True)

    train_t0 = time.time()
    best_model, history = fit(
        model,
        loss_fn=loss_fn, metric_fn=metric_fn,
        config=config, output_dir=output_dir,
        resume_from=resume_from,
    )
    train_walltime_s = float(time.time() - train_t0)

    demo = predict_demo(best_model, config)
    save_npz(output_dir / "predictions_demo.npz", **demo)

    save_json(
        output_dir / "metrics.json",
        {
            "train_walltime_s": train_walltime_s,
            "best_val_loss": float(min(history["val_loss"])) if history["val_loss"] else float("nan"),
            "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float("nan"),
            "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else float("nan"),
            "final_test_loss": float(history["test_loss"][-1]) if history["test_loss"] else float("nan"),
            "final_test_theta_mae_mean": float(history["test_theta_mae_mean"][-1]) if history["test_theta_mae_mean"] else float("nan"),
            "final_test_omega_mae_mean": float(history["test_omega_mae_mean"][-1]) if history["test_omega_mae_mean"] else float("nan"),
            "config": {
                "N": config.N, "model": config.model.value,
                "solver": config.solver.value, "adjoint": config.adjoint,
                "n_steps": config.n_steps, "epochs": config.epochs,
                "batch_size": config.batch_size, "seed": config.seed,
                "learning_rate": config.learning_rate,
            },
        },
    )
    print(f"saved artifacts to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
