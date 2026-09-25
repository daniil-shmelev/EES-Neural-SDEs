"""Reproduce central errors and runtimes from the saved NeurIPS volatility runs."""

import argparse
import json
import tomllib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "neurips_20260421"

MODEL_ORDER = [("rough_bergomi", "Rough Bergomi")]

METHOD_ORDER = [
    ("reversible_heun", "Reversible Heun"),
    ("mcf_euler", "MCF Euler"),
    ("mcf_midpoint", "MCF Midpoint"),
    ("ees25", r"$\mathrm{EES}_{\mathcal{R}}(2,5)$"),
]


@dataclass(frozen=True)
class RunMetrics:
    nfe_per_step: int
    solve_dt: float
    terminal_mse: float
    runtime_s: float


def _iter_result_dirs(results_dir: Path):
    for path in sorted(results_dir.iterdir()):
        if path.is_dir() and (path / "config.toml").exists():
            yield path


def _terminal_mse(test_predictions_path: Path) -> float:
    with np.load(test_predictions_path) as payload:
        predicted = payload["predicted"]
        actual = payload["actual"]
    return float(np.mean((predicted[:, -1, :] - actual[:, -1, :]) ** 2))


def _runtime_seconds(metrics: dict) -> float:
    train_time = float(metrics.get("train_time_s", 0.0))
    inference_time = float(metrics.get("inference_time_s", 0.0))
    return train_time + inference_time


def collect_latest_runs(results_dir: Path = RESULTS_DIR) -> dict[tuple[str, str], RunMetrics]:
    latest_dirs: dict[tuple[str, str], Path] = {}

    for run_dir in _iter_result_dirs(results_dir):
        config = tomllib.loads((run_dir / "config.toml").read_text(encoding="utf-8"))
        key = (config["experiment"], config["solver"])
        current = latest_dirs.get(key)
        if current is None or run_dir.name > current.name:
            latest_dirs[key] = run_dir

    rows: dict[tuple[str, str], RunMetrics] = {}
    for key, run_dir in latest_dirs.items():
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        rows[key] = RunMetrics(
            nfe_per_step=int(metrics["nfe_per_step"]),
            solve_dt=float(metrics["solve_dt"]),
            terminal_mse=_terminal_mse(run_dir / "test_predictions.npz"),
            runtime_s=_runtime_seconds(metrics),
        )

    return rows


def _format_step_size(value: float) -> str:
    frac = Fraction(value).limit_denominator(1000)
    if abs(float(frac) - value) < 1e-12:
        return f"${frac.numerator}/{frac.denominator}$"
    return f"{value:.8g}"


def _format_terminal_mse(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _format_runtime(value: float) -> str:
    return f"{value:.2f}"


def build_table(rows: dict[tuple[str, str], RunMetrics]) -> str:
    missing = [
        (model_key, method_key)
        for model_key, _ in MODEL_ORDER
        for method_key, _ in METHOD_ORDER
        if (model_key, method_key) not in rows
    ]
    if missing:
        missing_text = ", ".join(f"{model}/{method}" for model, method in missing)
        raise ValueError(f"Missing result rows: {missing_text}")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"{\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{c|c|c c c c}",
        r"Model & Method & \#Eval. / Step & Step Size & Terminal MSE ($\times 10^{-2}$) & Runtime (s) \\",
        r"\hline",
    ]

    for model_index, (model_key, model_label) in enumerate(MODEL_ORDER):
        for method_index, (method_key, method_label) in enumerate(METHOD_ORDER):
            metrics = rows[(model_key, method_key)]
            prefix = (
                rf"\multirow{{4}}{{*}}{{{model_label}}} &"
                if method_index == 0
                else "  &"
            )
            line = (
                f"{prefix} {method_label} & {metrics.nfe_per_step} "
                f"& {_format_step_size(metrics.solve_dt)} "
                f"& {_format_terminal_mse(metrics.terminal_mse)} "
                f"& {_format_runtime(metrics.runtime_s)} \\\\"
            )
            lines.append(line)
        if model_index < len(MODEL_ORDER) - 1:
            lines.append(r"\hline")

    lines.extend(
        [
            r"\end{tabular}}",
            r"\caption{Metrics for rough Bergomi dynamics.}",
            r"\label{table:rough_bergomi}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    rows = collect_latest_runs(args.results_dir)
    print(build_table(rows))


if __name__ == "__main__":
    main()
