"""Evaluate saved stochastic-volatility checkpoints and summarise by seed."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from experiments.stochastic_volatility.experiment.config import (
    Devices,
    ExperimentConfig,
    Experiments,
    Solvers,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "repeated_inference_summary.json"


def _run_dir(path: Path) -> Path:
    return path.parent if path.is_file() else path


def _discover_runs(results_dir: Path) -> list[Path]:
    runs = [
        path
        for path in results_dir.iterdir()
        if path.is_dir()
        and (path / "config.toml").exists()
        and (path / "nsde.eqx").exists()
    ]
    return sorted(runs)


def _load_saved_model(run_dir: Path, config: ExperimentConfig) -> Any:
    import equinox as eqx
    import jax

    from experiments.stochastic_volatility.experiment.factories import make_model

    template = make_model(config, jax.random.key(config.seed))
    return eqx.tree_deserialise_leaves(run_dir / "nsde.eqx", template)


def _collect_metrics(
    model: Any,
    config: ExperimentConfig,
    *,
    eval_repeats: int,
    seed_offset: int,
    time_index: int,
    mse_scale: float,
) -> dict[str, float]:
    import equinox as eqx
    import jax

    from experiments.stochastic_volatility.experiment.factories import (
        make_loader,
        make_sample_fn,
    )

    loader = make_loader(config, "test")
    loader_next = jax.jit(loader.next)
    sample_step = eqx.filter_jit(make_sample_fn(config))

    ks_values: list[float] = []
    terminal_mse_values: list[float] = []
    inference_times: list[float] = []

    for repeat in range(eval_repeats):
        key = jax.random.key(seed_offset + repeat)
        key, loader_key = jax.random.split(key)
        state = loader.init_state(loader_key)

        predicted_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []

        t0 = time.perf_counter()
        for _ in range(loader.steps_per_epoch):
            key, step_key = jax.random.split(key)
            batch, state, mask = loader_next(state)
            predictions, targets = sample_step(model, batch, mask, step_key)
            valid = np.asarray(jax.device_get(mask), dtype=bool)
            predicted_batches.append(np.asarray(jax.device_get(predictions))[valid])
            target_batches.append(np.asarray(jax.device_get(targets))[valid])

        predicted = np.concatenate(predicted_batches)
        actual = np.concatenate(target_batches)
        if time_index >= predicted.shape[1]:
            raise ValueError(
                f"{config.experiment}/{config.solver} has only "
                f"{predicted.shape[1]} saved time points; cannot read t={time_index}."
            )

        stat, _ = ks_2samp(
            predicted[:, time_index, :].ravel(),
            actual[:, time_index, :].ravel(),
        )
        terminal_mse = np.mean((predicted[:, -1, :] - actual[:, -1, :]) ** 2)
        ks_values.append(float(stat))
        terminal_mse_values.append(float(mse_scale * terminal_mse))
        inference_times.append(time.perf_counter() - t0)

    return {
        f"ks_t{time_index}": float(np.mean(ks_values)),
        "terminal_mse": float(np.mean(terminal_mse_values)),
        "inference_time_s": float(np.mean(inference_times)),
    }


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summarise(values: list[float]) -> dict[str, float | str]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    two_sd = 2.0 * sd
    return {
        "mean": mean,
        "sd": sd,
        "two_sd": two_sd,
        "lower_2sd": mean - two_sd,
        "upper_2sd": mean + two_sd,
        "mean_pm_2sd": f"${mean:.2f} \\scriptstyle{{\\pm {two_sd:.2f}}}$",
    }


def _entry(
    run_dir: Path,
    config: ExperimentConfig,
    metrics: dict[str, float],
) -> dict[str, Any]:
    saved_metrics = _load_metrics(run_dir)
    train_time_s = saved_metrics.get("train_time_s")
    if train_time_s is not None:
        metrics["runtime_s"] = float(train_time_s) + metrics["inference_time_s"]

    return {
        "experiment": str(config.experiment),
        "solver": str(config.solver),
        "seed": int(config.seed),
        "run_dir": str(run_dir),
        "saved_metrics": saved_metrics,
        "metrics": metrics,
    }


def _sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    experiment_order = {str(name): i for i, name in enumerate(Experiments)}
    solver_order = {str(name): i for i, name in enumerate(Solvers)}
    return (
        experiment_order.get(entry["experiment"], len(experiment_order)),
        solver_order.get(entry["solver"], len(solver_order)),
        entry.get("run_dir", ""),
    )


def _group_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        key = (entry["experiment"], entry["solver"])
        groups.setdefault(key, []).append(entry)

    summaries: list[dict[str, Any]] = []
    for (experiment, solver), group in groups.items():
        metric_names = sorted({name for entry in group for name in entry["metrics"]})
        metrics = {}
        for name in metric_names:
            values = [
                float(entry["metrics"][name])
                for entry in group
                if name in entry["metrics"]
            ]
            metrics[name] = {"values": values, **_summarise(values)}
        summaries.append(
            {
                "experiment": experiment,
                "solver": solver,
                "seeds": sorted(int(entry["seed"]) for entry in group),
                "n_checkpoints": len(group),
                "metrics": metrics,
            }
        )
    return sorted(summaries, key=_sort_key)


def _write_table(path: Path, groups: list[dict[str, Any]]) -> None:
    lines = [
        "# Checkpoint Seed Summary",
        "",
        "| model | integrator | terminal MSE +/- 2sd | runtime s +/- 2sd | seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in sorted(groups, key=_sort_key):
        metrics = group["metrics"]
        runtime_metric = "runtime_s"
        if runtime_metric not in metrics:
            runtime_metric = "inference_time_s"
        lines.append(
            "| "
            f"{group['experiment']} | "
            f"{group['solver']} | "
            f"{metrics['terminal_mse']['mean_pm_2sd']} | "
            f"{metrics[runtime_metric]['mean_pm_2sd']} | "
            f"{', '.join(str(seed) for seed in group['seeds'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configure_runtime(config: ExperimentConfig, *, force_cpu: bool) -> None:
    if force_cpu or config.device == Devices.CPU:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"

        import jax

        jax.config.update("jax_platform_name", "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="Saved result directories or nsde.eqx files. Defaults to every run in --results-dir.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=4,
        help="Common inference seeds to average per checkpoint before seed summary.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=10_000,
        help="Base evaluation seed. Eval repeat r uses seed_offset + r.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=55,
        help="Saved time index used for the two-sample KS statistic.",
    )
    parser.add_argument(
        "--mse-scale",
        type=float,
        default=100.0,
        help="Scale applied to terminal MSE. The paper tables use 100.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--table-output",
        type=Path,
        default=None,
        help="Markdown table path. Defaults to <output>.md.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force JAX CPU backend.")
    args = parser.parse_args()

    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"
    if args.eval_repeats <= 0:
        raise ValueError("--eval-repeats must be positive.")

    run_dirs = (
        [_run_dir(path) for path in args.runs]
        if args.runs
        else _discover_runs(args.results_dir)
    )
    if not run_dirs:
        raise FileNotFoundError(f"No saved runs found in {args.results_dir}.")

    entries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config_path = run_dir / "config.toml"
        model_path = run_dir / "nsde.eqx"
        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Expected config.toml and nsde.eqx under {run_dir}."
            )

        config = load_config(config_path)
        _configure_runtime(config, force_cpu=args.cpu)
        print(
            f"running {config.experiment}/{config.solver} "
            f"train_seed={config.seed} eval_repeats={args.eval_repeats} "
            f"path={run_dir}",
            flush=True,
        )
        model = _load_saved_model(run_dir, config)
        metrics = _collect_metrics(
            model,
            config,
            eval_repeats=args.eval_repeats,
            seed_offset=args.seed_offset,
            time_index=args.time_index,
            mse_scale=args.mse_scale,
        )
        entries.append(_entry(run_dir, config, metrics))

    groups = _group_entries(entries)

    payload = {
        "eval_repeats": int(args.eval_repeats),
        "mse_scale": float(args.mse_scale),
        "seed_offset": int(args.seed_offset),
        "time_index": int(args.time_index),
        "checkpoints": sorted(entries, key=_sort_key),
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    table_output = args.table_output or args.output.with_suffix(".md")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    _write_table(table_output, groups)

    print(f"saved {args.output}", flush=True)
    print(f"saved {table_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
