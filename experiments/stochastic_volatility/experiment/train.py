"""Training entrypoint for the neural SDE experiment."""

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

from experiments.stochastic_volatility.experiment.config import (
    Devices,
    ExperimentConfig,
    is_sweep_config,
    load_config,
    sweep_config_at,
    sweep_len,
)
from experiments.stochastic_volatility.experiment.factories import make_loader, make_model, make_sample_fn
from experiments.stochastic_volatility.experiment.losses import LossFn, PyTree, SampleFn
from experiments.stochastic_volatility.experiment.plots import plot_sample_trajectories, plot_training_curves

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainResult:
    best_model: eqx.Module
    history: dict[str, list[float]]
    latest_epoch_batch: dict[str, np.ndarray] | None
    completed_epochs: int
    interrupted: bool


def _to_numpy_epoch_batch(
    predictions: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    *,
    epoch: int,
) -> dict[str, np.ndarray]:
    return {
        "predicted": np.asarray(jax.device_get(predictions)),
        "actual": np.asarray(jax.device_get(targets)),
        "mask": np.asarray(jax.device_get(mask), dtype=bool),
        "epoch": np.asarray(epoch, dtype=np.int32),
    }


def fit(
    model: eqx.Module,
    *,
    loss_fn: LossFn,
    config: ExperimentConfig,
    sample_fn: SampleFn | None = None,
    val_metric_fn: LossFn | None = None,
    val_metric_name: str = "val_metric",
) -> TrainResult:
    train_loader = make_loader(config, "train")
    train_loader_next = jax.jit(train_loader.next)

    val_loader = make_loader(config, "val")
    val_loader_next = jax.jit(val_loader.next)

    @eqx.filter_jit
    def train_step(
        current_model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> tuple[eqx.Module, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            current_model, batch, mask, key
        )
        updates = jax.tree_util.tree_map(
            lambda grad: None if grad is None else -config.learning_rate * grad,
            grads,
        )
        return eqx.apply_updates(current_model, updates), loss

    eval_step = eqx.filter_jit(loss_fn)
    sample_step = eqx.filter_jit(sample_fn) if sample_fn is not None else None
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
    best_score = jnp.inf
    latest_epoch_batch: dict[str, np.ndarray] | None = None
    completed_epochs = 0
    interrupted = False

    try:
        for epoch in range(config.epochs):
            train_loss = 0.0
            for _ in range(train_loader.steps_per_epoch):
                key, step_key = jax.random.split(key)
                batch, train_state, mask = train_loader_next(train_state)
                model, loss = train_step(model, batch, mask, step_key)
                train_loss += float(loss)
            train_loss /= train_loader.steps_per_epoch
            history["train_loss"].append(train_loss)

            log_line = f"epoch={epoch + 1}/{config.epochs} train_loss={train_loss:.3e}"

            val_loss = 0.0
            val_metric = 0.0
            last_batch = None
            last_mask = None
            for _ in range(val_loader.steps_per_epoch):
                key, step_key = jax.random.split(key)
                batch, val_state, mask = val_loader_next(val_state)
                val_loss += float(eval_step(model, batch, mask, step_key))
                last_batch = batch
                last_mask = mask
                if metric_step is not None:
                    key, metric_key = jax.random.split(key)
                    val_metric += float(metric_step(model, batch, mask, metric_key))
            val_loss /= val_loader.steps_per_epoch
            history["val_loss"].append(val_loss)
            log_line += f" val_loss={val_loss:.3e}"

            if (
                sample_step is not None
                and last_batch is not None
                and last_mask is not None
            ):
                key, sample_key = jax.random.split(key)
                predictions, targets = sample_step(
                    model, last_batch, last_mask, sample_key
                )
                latest_epoch_batch = _to_numpy_epoch_batch(
                    predictions,
                    targets,
                    last_mask,
                    epoch=epoch + 1,
                )

            score = val_loss
            if metric_step is not None:
                val_metric /= val_loader.steps_per_epoch
                history[val_metric_name].append(val_metric)
                log_line += f" {val_metric_name}={val_metric:.6f}"
                score = val_metric

            if score < float(best_score):
                best_score = jnp.asarray(score)
                best_model = model

            completed_epochs = epoch + 1
            print(log_line, flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print(
            "training interrupted by user; saving latest available artifacts",
            flush=True,
        )

    return TrainResult(
        best_model=best_model,
        history=history,
        latest_epoch_batch=latest_epoch_batch,
        completed_epochs=completed_epochs,
        interrupted=interrupted,
    )


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


def collect_samples(
    model: eqx.Module,
    *,
    sample_fn: SampleFn,
    config: ExperimentConfig,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(config, "test")
    loader_next = jax.jit(loader.next)
    sample_step = eqx.filter_jit(sample_fn)

    key = jax.random.key(config.seed + seed_offset)
    key, loader_key = jax.random.split(key)
    state = loader.init_state(loader_key)

    predicted_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    for _ in range(loader.steps_per_epoch):
        key, step_key = jax.random.split(key)
        batch, state, mask = loader_next(state)
        predictions, targets = sample_step(model, batch, mask, step_key)
        valid = np.asarray(jax.device_get(mask), dtype=bool)
        predicted_batches.append(np.asarray(jax.device_get(predictions))[valid])
        target_batches.append(np.asarray(jax.device_get(targets))[valid])

    return np.concatenate(predicted_batches), np.concatenate(target_batches)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _make_output_dir(config: ExperimentConfig) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    slug = f"{config.experiment}__{config.solver}__seed{config.seed}__{timestamp}"
    return PROJECT_ROOT / "results" / slug


def _save_config(path: Path, config: ExperimentConfig) -> None:
    lines = [
        f"experiment     = {str(config.experiment)!r}",
        f"solver         = {str(config.solver)!r}",
        f"seed           = {config.seed}",
        "",
        f"epochs         = {config.epochs}",
        f"batch_size     = {config.batch_size}",
        f"learning_rate  = {config.learning_rate}",
        "",
        f"hidden_dim     = {config.hidden_dim}",
        f"nfe_budget     = {config.nfe_budget}",
        f"total_time     = {config.total_time}",
        f"diffusion_scale = {config.diffusion_scale}",
        "",
        f"device         = {str(config.device)!r}",
        f"skip_plots     = {str(config.skip_plots).lower()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configure_runtime(config: ExperimentConfig) -> ExperimentConfig:
    if config.device == Devices.CPU:
        jax.config.update("jax_platform_name", "cpu")
    return config


def _effective_solver_metrics(
    model: eqx.Module, config: ExperimentConfig
) -> dict[str, Any]:
    solve_n_steps = getattr(model, "solve_n_steps", config.n_steps)
    solve_dt = getattr(model, "solve_dt", config.dt)
    nfe_per_step = getattr(model, "nfe_per_step", None)
    metrics = {
        "nfe_budget": int(config.nfe_budget),
        "total_time": float(config.total_time),
        "base_dt": float(config.dt),
        "solve_n_steps": int(solve_n_steps),
        "solve_dt": float(solve_dt),
    }
    if nfe_per_step is not None:
        metrics["nfe_per_step"] = int(nfe_per_step)
    return metrics


def _build_metrics_payload(
    *,
    solver_metrics: dict[str, Any],
    completed_epochs: int,
    interrupted: bool,
    train_time: float | None,
    inference_time: float | None,
    test_loss: float | None,
    extra_metrics: dict[str, Any],
    latest_epoch_batch: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        **solver_metrics,
        "completed_epochs": int(completed_epochs),
        "interrupted": bool(interrupted),
    }
    if train_time is not None:
        payload["train_time_s"] = round(train_time, 2)
    if inference_time is not None:
        payload["inference_time_s"] = round(inference_time, 2)
    if test_loss is not None:
        payload["test_loss"] = float(test_loss)
    if latest_epoch_batch is not None:
        payload["latest_epoch"] = int(latest_epoch_batch["epoch"].item())
    payload.update(extra_metrics)
    return payload


def _save_artifacts(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    model: eqx.Module,
    history: dict[str, list[float]],
    metrics: dict[str, Any],
    latest_epoch_batch: dict[str, np.ndarray] | None,
    test_predictions: dict[str, np.ndarray] | None,
) -> None:
    _save_config(output_dir / "config.toml", config)
    eqx.tree_serialise_leaves(output_dir / "nsde.eqx", model)
    _save_json(output_dir / "history.json", history)
    _save_json(output_dir / "metrics.json", metrics)

    if test_predictions is not None:
        np.savez_compressed(output_dir / "test_predictions.npz", **test_predictions)

    if latest_epoch_batch is not None:
        np.savez_compressed(output_dir / "latest_epoch_batch.npz", **latest_epoch_batch)
        plot_sample_trajectories(
            latest_epoch_batch["predicted"],
            latest_epoch_batch["actual"],
            output_dir / "latest_epoch_trajectories.png",
            mask=latest_epoch_batch["mask"],
        )

    if not config.skip_plots and any(history.get(name) for name in history):
        plot_training_curves({"nsde": history}, output_dir / "training_curves.png")


# ---------------------------------------------------------------------------
# Experiment-specific setup
# ---------------------------------------------------------------------------


def _setup_stoch_vol(config: ExperimentConfig, key: jax.Array) -> dict[str, Any]:
    from experiments.stochastic_volatility.experiment.dataset import StochVolatilityDataset
    from scipy.stats import ks_2samp

    from experiments.stochastic_volatility.experiment.losses import truncated_sig_loss

    train_dataset = StochVolatilityDataset(experiment=config.experiment, split="train")
    # state_dim after the [..0:1] slice in the dataset
    state_dim = int(train_dataset._solution.shape[-1])

    model = make_model(config, key)
    sample_fn = make_sample_fn(config)
    _inner_loss = truncated_sig_loss(depth=6, ambient_dim=state_dim)

    def loss_fn(model, batch, mask, key):
        predictions, targets = sample_fn(model, batch, mask, key)
        return _inner_loss(predictions, targets)

    def inference_metrics_fn(
        predicted: np.ndarray, actual: np.ndarray
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        T = predicted.shape[1]
        for t in [55]:
            if t >= T:
                continue
            stat, _ = ks_2samp(predicted[:, t, :].ravel(), actual[:, t, :].ravel())
            metrics[f"ks_t{t}"] = float(stat)
        return metrics

    return {
        "model": model,
        "sample_fn": sample_fn,
        "loss_fn": loss_fn,
        "val_metric_fn": None,
        "val_metric_name": "val_loss",
        "inference_metrics_fn": inference_metrics_fn,
        "n_examples": len(train_dataset),
        "state_dim": state_dim,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_single_config(config: ExperimentConfig) -> int:
    output_dir = _make_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_key = jax.random.key(config.seed)

    setup = _setup_stoch_vol(config, model_key)
    print(
        f"experiment={config.experiment}",
        f"train={setup['n_examples']}",
        f"state_dim={setup['state_dim']}",
        f"hidden_dim={config.hidden_dim}",
        f"nfe_budget={config.nfe_budget}",
        f"total_time={config.total_time:.8g}",
        f"solver={config.solver}",
        flush=True,
    )

    model = setup["model"]
    sample_fn = setup["sample_fn"]
    loss_fn = setup["loss_fn"]
    solver_metrics = _effective_solver_metrics(model, config)
    best_model = model
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    latest_epoch_batch: dict[str, np.ndarray] | None = None
    completed_epochs = 0
    interrupted = False
    train_time: float | None = None
    inference_time: float | None = None
    test_loss: float | None = None
    extra_metrics: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] | None = None

    try:
        t0 = time.perf_counter()
        train_result = fit(
            model,
            loss_fn=loss_fn,
            config=config,
            sample_fn=sample_fn,
            val_metric_fn=setup["val_metric_fn"],
            val_metric_name=setup["val_metric_name"],
        )
        train_time = time.perf_counter() - t0

        best_model = train_result.best_model
        history = train_result.history
        latest_epoch_batch = train_result.latest_epoch_batch
        completed_epochs = train_result.completed_epochs
        interrupted = train_result.interrupted

        if not interrupted:
            t0 = time.perf_counter()
            test_loss = evaluate(
                best_model, loss_fn=loss_fn, config=config, seed_offset=1
            )
            predicted, actual = collect_samples(
                best_model, sample_fn=sample_fn, config=config, seed_offset=2
            )

            test_predictions = {"predicted": predicted, "actual": actual}

            inference_metrics = setup["inference_metrics_fn"](predicted, actual)
            for k, v in inference_metrics.items():
                if k.startswith("_"):
                    test_predictions[k.lstrip("_")] = v
                else:
                    extra_metrics[k] = v

            inference_time = time.perf_counter() - t0
    except KeyboardInterrupt:
        interrupted = True
        print(
            "evaluation interrupted by user; saving latest available artifacts",
            flush=True,
        )

    metrics_payload = _build_metrics_payload(
        solver_metrics=solver_metrics,
        completed_epochs=completed_epochs,
        interrupted=interrupted,
        train_time=train_time,
        inference_time=inference_time,
        test_loss=test_loss,
        extra_metrics=extra_metrics,
        latest_epoch_batch=latest_epoch_batch,
    )
    _save_artifacts(
        output_dir=output_dir,
        config=config,
        model=best_model,
        history=history,
        metrics=metrics_payload,
        latest_epoch_batch=latest_epoch_batch,
        test_predictions=test_predictions,
    )

    print(f"saved artifacts to {output_dir}", flush=True)
    if test_loss is not None:
        summary = f"test_loss={test_loss:.3e}"
        for k, v in extra_metrics.items():
            summary += f" {k}={v:.6f}"
        summary += (
            f" nfe_budget={solver_metrics['nfe_budget']}"
            f" solve_n_steps={solver_metrics['solve_n_steps']}"
            f" solve_dt={solver_metrics['solve_dt']:.8g}"
        )
        print(summary, flush=True)
        return 0

    print(
        f"run interrupted={str(interrupted).lower()} completed_epochs={completed_epochs}",
        flush=True,
    )
    return 130 if interrupted else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Path to TOML (single config or sweep)"
    )
    parser.add_argument(
        "--index", type=int, default=None, help="Sweep index for a single sweep run"
    )
    args = parser.parse_args()

    if args.index is not None:
        config = _configure_runtime(sweep_config_at(args.config, args.index))
        print(f"running sweep_index={args.index}", flush=True)
        return _run_single_config(config)

    if is_sweep_config(args.config):
        total = sweep_len(args.config)
        print(f"running sweep with {total} configurations", flush=True)
        for index in range(total):
            print(f"sweep_run={index + 1}/{total} sweep_index={index}", flush=True)
            config = _configure_runtime(sweep_config_at(args.config, index))
            status = _run_single_config(config)
            if status != 0:
                return status
        return 0

    config = _configure_runtime(load_config(args.config))
    return _run_single_config(config)


if __name__ == "__main__":
    main()
