"""Train the JAX/georax HumanActivity latent SDE classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import optax
from tqdm.auto import tqdm

from experiments.sphere_latent_nsde.config import ActivityConfig, load_config, replace_config, to_json
from experiments.sphere_latent_nsde.factories import make_dataset, make_loader, make_model
from experiments.sphere_latent_nsde.losses import activity_loss_value, activity_metrics_value


def _count_parameters(model: eqx.Module) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return int(sum(x.size for x in leaves if hasattr(x, "size")))


def _limit_steps(steps_per_epoch: int, limit: int | None) -> int:
    return steps_per_epoch if limit is None else min(steps_per_epoch, int(limit))


def _tree_float_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def fit(model: eqx.Module, config: ActivityConfig) -> tuple[eqx.Module, dict[str, list[dict[str, float]]]]:
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

    for epoch in tqdm(range(1, config.epochs + 1), desc="activity", unit="epoch"):
        aux_mul = (epoch / 60.0) ** 2 if epoch < 60 else 1.0

        train_loss = 0.0
        for _ in range(train_steps):
            key, step_key = jax.random.split(key)
            batch, train_state, mask = train_next(train_state)
            model, opt_state, loss = train_step(
                model, opt_state, batch, mask, step_key, aux_mul
            )
            train_loss += float(loss)
        train_loss /= max(train_steps, 1)
        train_metrics = {"loss": train_loss}

        split_metrics: dict[str, dict[str, float]] = {}
        for split, loader_next, state, steps in (
            ("val", val_next, val_state, eval_steps["val"]),
            ("test", test_next, test_state, eval_steps["test"]),
        ):
            totals: dict[str, float] = {}
            for _ in range(steps):
                key, step_key = jax.random.split(key)
                batch, state, mask = loader_next(state)
                metrics = _tree_float_dict(
                    eval_step(
                        model,
                        batch,
                        mask,
                        step_key,
                        aux_mul,
                        config.mc_eval_samples,
                    )
                )
                for metric_name, metric_value in metrics.items():
                    totals[metric_name] = totals.get(metric_name, 0.0) + metric_value
            split_metrics[split] = {
                metric_name: metric_value / max(steps, 1)
                for metric_name, metric_value in totals.items()
            }
            if split == "val":
                val_state = state
            else:
                test_state = state

        if split_metrics["val"]["aux_acc"] > best_val_acc:
            best_val_acc = split_metrics["val"]["aux_acc"]
            best_model = model
        split_metrics["val"]["aux_acc*"] = best_val_acc

        history["train"].append(train_metrics)
        history["val"].append(split_metrics["val"])
        history["test"].append(split_metrics["test"])

        tqdm.write(
            "epoch={:04d} train_loss={:.6f} val_acc={:.2f}% test_acc={:.2f}%".format(
                epoch,
                train_metrics["loss"],
                split_metrics["val"]["aux_acc_pct"],
                split_metrics["test"]["aux_acc_pct"],
            )
        )

    return best_model, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/sphere_latent_nsde/results"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--h-dim", type=int, default=None)
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--n-deg", type=int, default=None)
    parser.add_argument("--num-timepoints", type=int, default=None)
    parser.add_argument("--mc-train-samples", type=int, default=None)
    parser.add_argument("--mc-eval-samples", type=int, default=None)
    parser.add_argument("--solver", choices=("geometric_euler", "cfees25"), default=None)
    parser.add_argument("--adjoint", choices=("auto", "direct", "reversible"), default=None)
    parser.add_argument("--data-source", choices=("auto", "raw", "torch"), default=None)
    parser.add_argument("--split-strategy", choices=("auto", "numpy", "torch"), default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config) if args.config is not None else ActivityConfig()
    config = replace_config(
        config,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        n_deg=args.n_deg,
        num_timepoints=args.num_timepoints,
        solver=args.solver,
        adjoint=args.adjoint,
        data_source=args.data_source,
        split_strategy=args.split_strategy,
        mc_train_samples=args.mc_train_samples,
        mc_eval_samples=args.mc_eval_samples,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    if args.smoke:
        config = replace_config(
            config,
            epochs=1,
            batch_size=2,
            h_dim=4,
            z_dim=3,
            n_deg=2,
            num_timepoints=32,
            mc_train_samples=1,
            mc_eval_samples=1,
            max_train_batches=1,
            max_eval_batches=1,
        )

    train_dataset = make_dataset(config, "train")
    print(
        f"dataset sizes: train={len(train_dataset)} "
        f"val={len(make_dataset(config, 'val'))} test={len(make_dataset(config, 'test'))}",
        flush=True,
    )

    key = jax.random.key(config.seed)
    model = make_model(config, key)
    print(f"parameters={_count_parameters(model)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    best_model, history = fit(model, config)
    train_time_s = time.perf_counter() - start

    eqx.tree_serialise_leaves(args.output_dir / "activity_model.eqx", best_model)
    payload = {
        "config": json.loads(to_json(config)),
        "history": history,
        "final": {
            "train_time_s": train_time_s,
            "best_val_acc": max(row["aux_acc"] for row in history["val"]),
            "best_val_acc_pct": 100.0 * max(row["aux_acc"] for row in history["val"]),
            "final_test_acc": history["test"][-1]["aux_acc"],
            "final_test_acc_pct": history["test"][-1]["aux_acc_pct"],
        },
    }
    (args.output_dir / "activity_history.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
