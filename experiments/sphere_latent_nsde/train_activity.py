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


def fit(model: eqx.Module, config: ActivityConfig) -> TrainResult:
    train_loader = make_loader(config, "train")
    val_loader = make_loader(config, "val")
    test_loader = make_loader(config, "test")
    train_next = jax.jit(train_loader.next)
    val_next = jax.jit(val_loader.next)
    test_next = jax.jit(test_loader.next)

    optimizer = optax.adam(config.learning_rate)
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

    train_steps = _limit_steps(train_loader.steps_per_epoch, config.max_train_batches)
    eval_steps = {
        "val": _limit_steps(val_loader.steps_per_epoch, config.max_eval_batches),
        "test": _limit_steps(test_loader.steps_per_epoch, config.max_eval_batches),
    }

    history: dict[str, list[dict[str, float]]] = {"train": [], "val": [], "test": []}
    best_model = model
    best_val_acc = -np.inf
    best_epoch = 0

    with tqdm(
        range(1, config.epochs + 1),
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


def _make_output_dir(config: ActivityConfig) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    slug = f"{config.experiment}__{config.method}__seed{config.seed}__{timestamp}"
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
        "best_epoch": int(final_test["best_val_epoch"]),
        "best_val_acc": float(final_val["aux_acc*"]),
        "best_val_acc_pct": float(final_val["aux_acc_pct*"]),
        "test_acc_at_best_val": float(final_test["aux_acc*"]),
        "test_acc_at_best_val_pct": float(final_test["aux_acc_pct*"]),
        "final_test_acc": float(final_test["aux_acc"]),
        "final_test_acc_pct": float(final_test["aux_acc_pct"]),
        "num_timepoints": int(config.num_timepoints),
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


def _run_single_config(config: ActivityConfig) -> int:
    output_dir = _make_output_dir(config)
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
        f"num_timepoints={config.num_timepoints}",
        flush=True,
    )

    key = jax.random.key(config.seed)
    model = make_model(config, key)
    parameter_count = _count_parameters(model)
    print(f"parameters={parameter_count}", flush=True)

    start = time.perf_counter()
    train_result = fit(model, config)
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
        "test_acc={test_acc_at_best_val_pct:.2f}%".format(**metrics),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to TOML config or sweep")
    parser.add_argument("--index", type=int, default=None, help="Sweep index to run")
    args = parser.parse_args()

    if args.index is not None:
        print(f"running sweep_index={args.index}", flush=True)
        return _run_single_config(_configure_runtime(sweep_config_at(args.config, args.index)))

    if is_sweep_config(args.config):
        total = sweep_len(args.config)
        print(f"running sweep with {total} configurations", flush=True)
        for index in range(total):
            print(f"sweep_run={index + 1}/{total} sweep_index={index}", flush=True)
            status = _run_single_config(_configure_runtime(sweep_config_at(args.config, index)))
            if status != 0:
                return status
        return 0

    return _run_single_config(_configure_runtime(load_config(args.config)))


if __name__ == "__main__":
    raise SystemExit(main())
