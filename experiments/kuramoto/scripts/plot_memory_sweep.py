"""Plot measured peak allocations for the unified Kuramoto experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.kuramoto.experiment.config import PROTOCOL, validate_method

LABELS = {
    "cg2:checkpoint_full": "CG2 (Full)",
    "cg2:checkpoint_recursive": "CG2 (Recursive)",
    "cfees25:reversible": "CF-EES(2,5) (Reconstruction)",
    "geometric_euler_heun:checkpoint_full": "Geometric Euler–Heun (Full)",
    "geometric_euler_heun:checkpoint_recursive": "Geometric Euler–Heun (Recursive)",
}


def read_cells(path):
    raw = json.loads(Path(path).read_text())
    if raw.get("protocol") != PROTOCOL:
        raise ValueError(
            "This is not a unified-model memory sweep. Historical plotting code is in /home/luke/nsde_archive/experiments/kuramoto/legacy/scripts/."
        )
    cells = raw["cells"]
    reference = None
    for cell in cells:
        validate_method(cell["solver"], cell["adjoint"])
        if cell.get("error") or cell.get("oom"):
            continue
        if cell.get("protocol") != PROTOCOL:
            raise ValueError("Memory cell lacks unified experiment provenance")
        config = dict(cell["config"])
        if (config["solver"], config["adjoint"]) != (cell["solver"], cell["adjoint"]):
            raise ValueError("Memory cell method disagrees with its configuration")
        if config.get("drift_kind") != "second_order_mean_field":
            raise ValueError("Memory cell uses a historical model architecture")
        for key in ("solver", "adjoint", "n_steps"):
            config.pop(key, None)
        if reference is None:
            reference = config
        elif config != reference:
            raise ValueError("Memory cells use different model, data or loss settings")
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/kuramoto/results/unified/memory.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/kuramoto/results/unified/memory.pdf"),
    )
    args = parser.parse_args()
    cells = read_cells(args.input)
    plt.rcParams.update(
        {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9}
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    plotted = 0
    for mode in sorted({f"{c['solver']}:{c['adjoint']}" for c in cells}):
        selected = sorted(
            (
                c
                for c in cells
                if f"{c['solver']}:{c['adjoint']}" == mode
                and c.get("peak_bytes")
                and not c.get("error")
            ),
            key=lambda c: c["n_steps"],
        )
        if selected:
            ax.plot(
                [c["n_steps"] for c in selected],
                [c["peak_bytes"] / 2**20 for c in selected],
                marker="o",
                label=LABELS.get(mode, mode),
            )
            plotted += 1
    if not plotted:
        raise ValueError("No successful memory measurements to plot")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Solver steps",
        ylabel="Peak GPU memory (MiB)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
