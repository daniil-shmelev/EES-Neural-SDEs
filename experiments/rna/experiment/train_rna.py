"""Training entrypoint for the torus neural SDE on RNA torsion angles."""

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

from experiments.rna.datasets.dataset import RNATorsionDataset
from experiments.rna.experiment.config import ExperimentConfig, Experiments, Solvers, load_config
from experiments.rna.experiment.factories import make_loader, make_model
from experiments.rna.experiment.losses import LossFn, PyTree
from experiments.rna.experiment.rna_losses import (
    make_wrapped_mae_metric,
    make_wrapped_mse_loss,
    rna_predict_batch,
    wrapped_angle_diff,
)
from experiments.rna.experiment.runtime import make_output_dir, save_json
from experiments.rna.plots import plot_training_curves


def fit(
    model: eqx.Module,
    *,
    loss_fn: LossFn,
    config: ExperimentConfig,
    val_metric_fn: LossFn | None = None,
    val_metric_name: str = "val_metric",
) -> tuple[eqx.Module, dict[str, list[float]]]:
    train_loader = make_loader(config, "train")
    train_loader_next = jax.jit(train_loader.next)

    val_loader = make_loader(config, "val")
    val_loader_next = jax.jit(val_loader.next)

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        current_model: eqx.Module,
        opt_state,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(current_model, batch, mask, key)
        updates, new_opt_state = optimizer.update(grads, opt_state, current_model)
        new_model = eqx.apply_updates(current_model, updates)
        return new_model, new_opt_state, loss

    eval_step = eqx.filter_jit(loss_fn)
    metric_step = eqx.filter_jit(val_metric_fn) if val_metric_fn is not None else None

    key = jax.random.key(config.seed)
    key, train_key = jax.random.split(key)
    train_state = train_loader.init_state(train_key)

    key, val_key = jax.random.split(key)
    val_state = val_loader.init_state(val_key)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    if val_metric_fn is not None:
        history[val_metric_name] = []

    best_model = model
    best_score = float("inf")

    for epoch in range(config.epochs):
        train_loss = 0.0
        for _ in range(train_loader.steps_per_epoch):
            key, step_key = jax.random.split(key)
            batch, train_state, mask = train_loader_next(train_state)
            model, opt_state, loss = train_step(model, opt_state, batch, mask, step_key)
            train_loss += float(loss)
        train_loss /= train_loader.steps_per_epoch
        history["train_loss"].append(train_loss)

        log_line = f"epoch={epoch + 1}/{config.epochs} train_loss={train_loss:.3e}"

        val_loss = 0.0
        val_metric = 0.0
        for _ in range(val_loader.steps_per_epoch):
            key, step_key = jax.random.split(key)
            batch, val_state, mask = val_loader_next(val_state)
            loss = eval_step(model, batch, mask, step_key)
            val_loss += float(loss)
            if metric_step is not None:
                key, metric_key = jax.random.split(key)
                metric_value = metric_step(model, batch, mask, metric_key)
                val_metric += float(metric_value)
        val_loss /= val_loader.steps_per_epoch
        history["val_loss"].append(val_loss)
        log_line += f" val_loss={val_loss:.3e}"

        score = val_loss
        if metric_step is not None:
            val_metric /= val_loader.steps_per_epoch
            history[val_metric_name].append(val_metric)
            log_line += f" {val_metric_name}={val_metric:.6f}"
            score = val_metric

        if score < best_score:
            best_score = score
            best_model = model

        print(log_line, flush=True)

    return best_model, history


def evaluate(
    model: eqx.Module,
    *,
    loss_fn: LossFn,
    config: ExperimentConfig,
    seed_offset: int = 0,
) -> float:
    loader = make_loader(config, "test")
    loader_next = jax.jit(loader.next)
    eval_step = eqx.filter_jit(loss_fn)

    key = jax.random.key(config.seed + seed_offset)
    key, loader_key = jax.random.split(key)
    state = loader.init_state(loader_key)

    total_loss = 0.0
    for _ in range(loader.steps_per_epoch):
        key, step_key = jax.random.split(key)
        batch, state, mask = loader_next(state)
        total_loss += float(eval_step(model, batch, mask, step_key))

    return total_loss / loader.steps_per_epoch


def predict_dataset(
    model: eqx.Module,
    *,
    config: ExperimentConfig,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(config, "test")
    loader_next = jax.jit(loader.next)

    @eqx.filter_jit
    def predict_step(current_model, batch, key):
        return rna_predict_batch(current_model, batch, key)

    key = jax.random.key(config.seed + seed_offset)
    key, loader_key = jax.random.split(key)
    state = loader.init_state(loader_key)

    predicted_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    for _ in range(loader.steps_per_epoch):
        key, step_key = jax.random.split(key)
        batch, state, mask = loader_next(state)
        predictions = predict_step(model, batch, step_key)
        valid = np.asarray(jax.device_get(mask), dtype=bool)
        predicted_batches.append(np.asarray(jax.device_get(predictions))[valid])
        target_batches.append(
            np.asarray(jax.device_get(batch["target_angles"]))[valid]
        )

    return (
        np.concatenate(predicted_batches, axis=0),
        np.concatenate(target_batches, axis=0),
    )


def _override_from_args(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    overrides: dict[str, Any] = {}
    if args.solver is not None:
        overrides["solver"] = Solvers(args.solver)
    if args.n_steps is not None:
        overrides["n_steps"] = int(args.n_steps)
    if args.residues_per_state is not None:
        overrides["residues_per_state"] = int(args.residues_per_state)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    if args.max_chains is not None:
        overrides["max_chains"] = int(args.max_chains) if args.max_chains > 0 else None
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.adjoint is not None:
        overrides["adjoint"] = str(args.adjoint)
    if not overrides:
        return config
    return dataclasses.replace(config, **overrides)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to experiment config TOML")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--residues-per-state", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-chains", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--adjoint",
        type=str,
        default=None,
        choices=(
            "auto",
            "reversible",
            "direct",
            "checkpoint_recursive",
            "checkpoint_log",  # deprecated alias
            "checkpoint_full",
        ),
        help=(
            "Override ExperimentConfig.adjoint. 'auto' preserves historical "
            "behaviour; 'reversible', 'checkpoint_recursive', 'checkpoint_full' "
            "pin a specific diffrax adjoint so the retrain isolates its effect."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = _override_from_args(config, args)
    if config.experiment is not Experiments.RNA:
        raise ValueError(
            f"train_rna expects experiment='rna' but got {config.experiment.value}"
        )

    train_dataset = RNATorsionDataset(
        split="train",
        context_length=config.context_length,
        residues_per_state=config.residues_per_state,
        max_chains=config.max_chains,
    )
    metadata = train_dataset.metadata()

    model_key = jax.random.key(config.seed)
    model = make_model(config, metadata, model_key)

    loss_fn = make_wrapped_mse_loss()
    metric_fn = make_wrapped_mae_metric()

    output_dir = args.output_dir or make_output_dir(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"dataset sizes: train={len(train_dataset)}",
        f"num_angles={metadata['num_angles']}",
        f"chains={metadata['n_chains']}",
        flush=True,
    )
    print(
        "model config:",
        f"d={metadata['num_angles']}",
        f"ctx_len={config.context_length}",
        f"k={config.residues_per_state}",
        f"hidden={config.hidden_dim}",
        f"n_steps={config.n_steps}",
        f"solver={config.solver}",
        flush=True,
    )

    train_t0 = time.time()
    best_model, history = fit(
        model,
        loss_fn=loss_fn,
        config=config,
        val_metric_fn=metric_fn,
        val_metric_name="val_wrapped_mae",
    )
    train_walltime_s = float(time.time() - train_t0)

    test_loss = evaluate(best_model, loss_fn=loss_fn, config=config, seed_offset=1)
    predicted, actual = predict_dataset(
        best_model, config=config, seed_offset=2
    )
    test_mae = float(
        np.mean(np.abs(np.asarray(jax.device_get(
            wrapped_angle_diff(jnp.asarray(predicted), jnp.asarray(actual))
        ))))
    )

    eqx.tree_serialise_leaves(output_dir / "torus_nsde.eqx", best_model)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        predicted=predicted,
        actual=actual,
    )
    save_json(output_dir / "history.json", history)
    save_json(
        output_dir / "metrics.json",
        {
            "test_loss": float(test_loss),
            "test_wrapped_mae": float(test_mae),
            "solver": config.solver.value,
            "adjoint": str(config.adjoint),
            "seed": int(config.seed),
            "n_steps": int(config.n_steps),
            "residues_per_state": int(config.residues_per_state),
            "num_angles": int(metadata["num_angles"]),
            "batch_size": int(config.batch_size),
            "epochs": int(config.epochs),
            "n_train": int(len(train_dataset)),
            "learning_rate": float(config.learning_rate),
            "dt": float(config.dt),
            "hidden_dim": int(config.hidden_dim),
            "train_walltime_s": train_walltime_s,
        },
    )

    if not config.skip_plots:
        plot_training_curves(
            {f"{config.solver.value}_n{config.n_steps}_k{config.residues_per_state}": history},
            output_dir / "training_curves.png",
        )

    print(
        f"saved artifacts to {output_dir}",
        f"test_loss={test_loss:.3e}",
        f"test_mae={test_mae:.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
