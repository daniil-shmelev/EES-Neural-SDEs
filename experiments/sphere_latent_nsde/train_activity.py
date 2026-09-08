"""Train the JAX/georax HumanActivity latent SDE classifier."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from experiments.sphere_latent_nsde.config import (
    ActivityConfig,
    is_sweep_config,
    load_config,
    sweep_config_at,
    sweep_len,
    to_toml,
)
from experiments.sphere_latent_nsde.factories import make_dataset, make_loader, make_model
from experiments.sphere_latent_nsde.losses import activity_loss_value, activity_metrics_value

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class TrainResult:
    best_model: eqx.Module
    history: dict[str, list[dict[str, float]]]
    completed_epochs: int


def _count_parameters(model: eqx.Module) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return int(sum(x.size for x in leaves if hasattr(x, "size")))


def _limit_steps(steps_per_epoch: int, limit: int | None) -> int:
    return steps_per_epoch if limit is None else min(steps_per_epoch, int(limit))


def _tree_float_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _learning_rate(config: ActivityConfig, steps_per_lr_epoch: int):
    if config.lr_schedule == "constant":
        return config.learning_rate
    if config.lr_schedule == "cosine":
        steps_per_lr_epoch = max(int(steps_per_lr_epoch), 1)
        lr_min = config.learning_rate * config.lr_min_ratio
        restart = max(int(config.lr_restart), 1)

        def schedule(count):
            epoch = count // steps_per_lr_epoch
            cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * epoch / restart))
            return lr_min + (config.learning_rate - lr_min) * cosine

        return schedule
    raise ValueError(f"unknown lr_schedule {config.lr_schedule!r}")


def fit(model: eqx.Module, config: ActivityConfig, output_dir: Path, resume: bool = False) -> TrainResult:
    train_loader = make_loader(config, "train")
    val_loader = make_loader(config, "val")
    test_loader = make_loader(config, "test")
    train_next = jax.jit(train_loader.next)
    val_next = jax.jit(val_loader.next)
    test_next = jax.jit(test_loader.next)

    train_steps = _limit_steps(train_loader.steps_per_epoch, config.max_train_batches)
    eval_steps = {
        "val": _limit_steps(val_loader.steps_per_epoch, config.max_eval_batches),
        "test": _limit_steps(test_loader.steps_per_epoch, config.max_eval_batches),
    }

    optimizer = optax.adam(_learning_rate(config, train_loader.steps_per_epoch))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(current_model, batch, mask, key, aux_mul):
        return activity_loss_value(
            current_model,
            batch,
            mask,
            key,
            mc_samples=config.mc_train_samples,
            kl0_weight=config.kl0_weight,
            klp_weight=config.klp_weight,
            pxz_weight=config.pxz_weight,
            aux_weight=config.aux_weight,
            aux_weight_mul=aux_mul,
        )

    def metrics_fn(current_model, batch, mask, key, aux_mul, mc_samples):
        return activity_metrics_value(
            current_model,
            batch,
            mask,
            key,
            mc_samples=mc_samples,
            kl0_weight=config.kl0_weight,
            klp_weight=config.klp_weight,
            pxz_weight=config.pxz_weight,
            aux_weight=config.aux_weight,
            aux_weight_mul=aux_mul,
        )

    @eqx.filter_jit
    def train_step(current_model, current_opt_state, batch, mask, key, aux_mul):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            current_model, batch, mask, key, aux_mul
        )
        updates, new_opt_state = optimizer.update(
            grads, current_opt_state, eqx.filter(current_model, eqx.is_array)
        )
        new_model = eqx.apply_updates(current_model, updates)
        return new_model, new_opt_state, loss

    eval_step = eqx.filter_jit(metrics_fn)

    def eval_split(current_model, loader_next, state, steps, key, aux_mul):
        totals: dict[str, float] = {}
        for _ in range(steps):
            key, step_key = jax.random.split(key)
            batch, state, mask = loader_next(state)
            metrics = _tree_float_dict(
                eval_step(
                    current_model,
                    batch,
                    mask,
                    step_key,
                    aux_mul,
                    config.mc_eval_samples,
                )
            )
            for metric_name, metric_value in metrics.items():
                totals[metric_name] = totals.get(metric_name, 0.0) + metric_value
        averaged = {name: value / max(steps, 1) for name, value in totals.items()}
        return averaged, state, key

    key = jax.random.key(config.seed)
    key, train_key, val_key, test_key = jax.random.split(key, 4)
    train_state = train_loader.init_state(train_key)
    val_state = val_loader.init_state(val_key)
    test_state = test_loader.init_state(test_key)

    epoch_offset = 0
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": [], "test": []}
    best_model = model
    best_val_acc = -np.inf
    best_epoch = 0

    # Resume from the latest per-epoch checkpoint if present (timeout restart).
    # RNG is re-seeded from config.seed (as in the Kuramoto resume), so a resumed
    # run is not bit-identical to an uninterrupted one but converges fine.
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume and (output_dir / "model_latest.eqx").exists() and (output_dir / "history.json").exists():
        model = eqx.tree_deserialise_leaves(output_dir / "model_latest.eqx", model)
        opt_state = eqx.tree_deserialise_leaves(output_dir / "opt_state_latest.eqx", opt_state)
        best_model = eqx.tree_deserialise_leaves(output_dir / "model_best.eqx", best_model)
        history = json.loads((output_dir / "history.json").read_text())
        cstate = json.loads((output_dir / "ckpt_state.json").read_text())
        best_val_acc = float(cstate["best_val_acc"])
        best_epoch = int(cstate["best_epoch"])
        epoch_offset = len(history["train"])
        print(
            f"[resume] continuing from epoch {epoch_offset} "
            f"(best_val_acc={100.0 * best_val_acc:.2f}% @ epoch {best_epoch})",
            flush=True,
        )

    with tqdm(
        range(epoch_offset + 1, config.epochs + 1),
        desc=f"{config.experiment}/{config.method}",
        unit="epoch",
        dynamic_ncols=True,
    ) as epochs:
        for epoch in epochs:
            aux_mul_value = (epoch / 60.0) ** 2 if epoch < 60 else 1.0
            aux_mul = jnp.asarray(aux_mul_value, dtype=jnp.float32)

            train_loss = 0.0
            for _ in range(train_steps):
                key, step_key = jax.random.split(key)
                batch, train_state, mask = train_next(train_state)
                model, opt_state, loss = train_step(
                    model, opt_state, batch, mask, step_key, aux_mul
                )
                train_loss += float(loss)
            train_metrics = {"loss": train_loss / max(train_steps, 1)}

            val_metrics, val_state, key = eval_split(
                model, val_next, val_state, eval_steps["val"], key, aux_mul
            )
            if val_metrics["aux_acc"] > best_val_acc:
                best_val_acc = val_metrics["aux_acc"]
                best_epoch = epoch
                best_model = model
            val_metrics["aux_acc*"] = best_val_acc
            val_metrics["aux_acc_pct*"] = 100.0 * best_val_acc
            val_metrics["best_val_epoch"] = float(best_epoch)

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)
            epochs.set_postfix_str(
                "train={:.3e} val_acc*={:.2f}% best_epoch={}".format(
                    train_metrics["loss"],
                    val_metrics["aux_acc_pct*"],
                    best_epoch,
                )
            )

            # Per-epoch checkpoint so a timeout can resume from here.
            eqx.tree_serialise_leaves(output_dir / "model_latest.eqx", model)
            eqx.tree_serialise_leaves(output_dir / "opt_state_latest.eqx", opt_state)
            if best_epoch == epoch:
                eqx.tree_serialise_leaves(output_dir / "model_best.eqx", best_model)
            _save_json(output_dir / "history.json", history)
            _save_json(
                output_dir / "ckpt_state.json",
                {"best_val_acc": float(best_val_acc), "best_epoch": int(best_epoch)},
            )

    test_metrics, _, key = eval_split(
        best_model,
        test_next,
        test_state,
        eval_steps["test"],
        key,
        1.0,
    )
    test_metrics["aux_acc*"] = test_metrics["aux_acc"]
    test_metrics["aux_acc_pct*"] = test_metrics["aux_acc_pct"]
    test_metrics["best_val_epoch"] = float(best_epoch)
    history["test"].append(test_metrics)

    return TrainResult(best_model, history, config.epochs)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_output_dir(config: ActivityConfig, *, deterministic: bool = False) -> Path:
    slug = f"{config.experiment}__{config.method}__seed{config.seed}"
    if not deterministic:
        # Timestamp keeps independent runs separate. For --auto-resume we want a
        # stable, findable dir per (method, seed) so a resubmit resumes in place.
        slug += "__" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "results" / slug


def _configure_runtime(config: ActivityConfig) -> ActivityConfig:
    if config.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    return config


def _metrics_payload(
    *,
    config: ActivityConfig,
    parameter_count: int,
    train_size: int,
    val_size: int,
    test_size: int,
    train_result: TrainResult,
    train_time_s: float,
) -> dict[str, Any]:
    history = train_result.history
    final_test = history["test"][-1]
    final_val = history["val"][-1]
    return {
        "completed_epochs": int(train_result.completed_epochs),
        "parameters": int(parameter_count),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "train_time_s": round(train_time_s, 2),
        "learning_rate": float(config.learning_rate),
        "lr_schedule": config.lr_schedule,
        "lr_restart": int(config.lr_restart),
        "lr_min_ratio": float(config.lr_min_ratio),
        "best_epoch": int(final_test["best_val_epoch"]),
        "best_val_acc": float(final_val["aux_acc*"]),
        "best_val_acc_pct": float(final_val["aux_acc_pct*"]),
        "test_acc_at_best_val": float(final_test["aux_acc*"]),
        "test_acc_at_best_val_pct": float(final_test["aux_acc_pct*"]),
        "final_test_acc": float(final_test["aux_acc"]),
        "final_test_acc_pct": float(final_test["aux_acc_pct"]),
        "num_timepoints": int(config.num_timepoints),
        "nfe_budget": int(config.effective_nfe_budget),
        "nfe_per_step": int(config.nfe_per_step),
        "solve_n_steps": int(config.solve_n_steps),
        "solve_num_timepoints": int(config.solve_n_steps + 1),
        "actual_forward_nfe": int(config.solve_n_steps * config.nfe_per_step),
        "solver": config.solver,
        "adjoint": config.adjoint,
    }


def _save_artifacts(
    *,
    output_dir: Path,
    config: ActivityConfig,
    train_result: TrainResult,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.toml").write_text(to_toml(config), encoding="utf-8")
    eqx.tree_serialise_leaves(output_dir / "nsde.eqx", train_result.best_model)
    _save_json(output_dir / "history.json", train_result.history)
    _save_json(output_dir / "metrics.json", metrics)


def _run_single_config(config: ActivityConfig, *, auto_resume: bool = False) -> int:
    output_dir = _make_output_dir(config, deterministic=auto_resume)
    if auto_resume and (output_dir / "metrics.json").exists():
        print(f"[skip] {output_dir.name} already complete (metrics.json present)", flush=True)
        return 0
    train_dataset = make_dataset(config, "train")
    val_dataset = make_dataset(config, "val")
    test_dataset = make_dataset(config, "test")
    print(
        f"experiment={config.experiment}",
        f"method={config.method}",
        f"train={len(train_dataset)}",
        f"val={len(val_dataset)}",
        f"test={len(test_dataset)}",
        f"h_dim={config.h_dim}",
        f"z_dim={config.z_dim}",
        f"lr={config.learning_rate}",
        f"lr_schedule={config.lr_schedule}",
        f"lr_restart={config.lr_restart}",
        f"num_timepoints={config.num_timepoints}",
        f"nfe_budget={config.effective_nfe_budget}",
        f"nfe_per_step={config.nfe_per_step}",
        f"solve_n_steps={config.solve_n_steps}",
        flush=True,
    )

    key = jax.random.key(config.seed)
    model = make_model(config, key)
    parameter_count = _count_parameters(model)
    print(f"parameters={parameter_count}", flush=True)

    start = time.perf_counter()
    train_result = fit(model, config, output_dir, resume=auto_resume)
    train_time_s = time.perf_counter() - start

    metrics = _metrics_payload(
        config=config,
        parameter_count=parameter_count,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        train_result=train_result,
        train_time_s=train_time_s,
    )
    _save_artifacts(
        output_dir=output_dir,
        config=config,
        train_result=train_result,
        metrics=metrics,
    )
    print(f"saved artifacts to {output_dir}", flush=True)
    print(
        "best_epoch={best_epoch} val_acc={best_val_acc_pct:.2f}% "
        "test_acc={test_acc_at_best_val_pct:.2f}% "
        "nfe_budget={nfe_budget} solve_n_steps={solve_n_steps}".format(**metrics),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to TOML config or sweep")
    parser.add_argument("--index", type=int, default=None, help="Sweep index to run")
    parser.add_argument(
        "--auto-resume", action="store_true",
        help="Use a deterministic per-(method,seed) output dir and resume in-place "
             "from the latest epoch checkpoint after a timeout; skip if already complete.",
    )
    args = parser.parse_args()

    if args.index is not None:
        print(f"running sweep_index={args.index}", flush=True)
        return _run_single_config(
            _configure_runtime(sweep_config_at(args.config, args.index)),
            auto_resume=args.auto_resume,
        )

    if is_sweep_config(args.config):
        total = sweep_len(args.config)
        print(f"running sweep with {total} configurations", flush=True)
        for index in range(total):
            print(f"sweep_run={index + 1}/{total} sweep_index={index}", flush=True)
            status = _run_single_config(
                _configure_runtime(sweep_config_at(args.config, index)),
                auto_resume=args.auto_resume,
            )
            if status != 0:
                return status
        return 0

    return _run_single_config(
        _configure_runtime(load_config(args.config)),
        auto_resume=args.auto_resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
