"""Run the Kuramoto methods and seeds selected in config.toml."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.kuramoto.experiment.config import (
    DEFAULT_CONFIG,
    ExperimentConfig,
    load_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index", type=int, help="Run one entry from the TOML sweep.")
    parser.add_argument("--output-base", type=Path, required=True)
    return parser.parse_args()


def _run_cell(
    config_path: Path, index: int, config: ExperimentConfig, out_dir: Path
) -> dict:
    label, seed = config.method, config.seed
    solver, adjoint, n_steps = config.solver.value, config.adjoint, config.n_steps
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.kuramoto.experiment.train_kuramoto",
        str(config_path),
        "--index",
        str(index),
        "--output-dir",
        str(out_dir),
    ]
    print(
        f"\n[comparison] === {label} | seed={seed} | N={config.N} | "
        f"n_steps={n_steps} | epochs={config.epochs} ===",
        flush=True,
    )
    print(f"[comparison] log: {out_dir.with_suffix('.log')}", flush=True)

    t0 = time.perf_counter()
    log_path = out_dir.with_suffix(".log")
    if log_path.exists():
        raise FileExistsError(f"Use a fresh log path: {log_path}")
    with (
        log_path.open("w") as fh,
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as proc,
    ):
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
            fh.flush()
        proc.wait()
    dt = time.perf_counter() - t0
    log_path.replace(out_dir / "train.log")

    info = {
        "label": label,
        "seed": seed,
        "solver": solver,
        "adjoint": adjoint,
        "n_steps": n_steps,
        "exit_code": proc.returncode,
        "wall_clock_s": dt,
        "out_dir": str(out_dir),
    }
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            info.update(
                {
                    k: metrics.get(k)
                    for k in (
                        "best_val_loss",
                        "final_train_loss",
                        "final_val_loss",
                        "final_test_loss",
                        "final_test_theta_mae_mean",
                        "final_test_omega_mae_mean",
                        "train_walltime_s",
                    )
                }
            )
        except Exception as exc:
            info["metrics_parse_error"] = str(exc)
    print(
        f"[comparison] {label}/seed{seed} exit={proc.returncode} "
        f"t={dt:.0f}s test={info.get('final_test_loss', 'NA')}",
        flush=True,
    )
    return info


def main() -> int:
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)

    configs = load_runs(args.config)
    indices = range(len(configs)) if args.index is None else [args.index]
    if any(index < 0 or index >= len(configs) for index in indices):
        raise ValueError(f"Sweep index must be between 0 and {len(configs) - 1}.")
    summary = {"config": str(args.config), "cells": []}
    summary_path = args.output_base / "comparison.json"
    for index in indices:
        config = configs[index]
        cell_out = args.output_base / f"{index:03d}_{config.method}_seed{config.seed}"
        cell_info = _run_cell(args.config, index, config, cell_out)
        summary["cells"].append(cell_info)
        summary_path.write_text(json.dumps(summary, indent=2))
        if cell_info["exit_code"]:
            return int(cell_info["exit_code"])

    print(f"\n[comparison] wrote {summary_path} ({len(summary['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
