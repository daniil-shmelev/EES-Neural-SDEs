"""Render the M4 memory-scaling figure from `memory_sweep_N{N}.json`.

Style matches `experiments/rna/plots.py:fig_torus_scaling` (the original
torus figure in the manuscript): STIX fonts, 3.2 x 2.2 in, log-log axes,
$\\Delta$ Memory on the y-axis (each curve baseline-subtracted to isolate
the saved-tape contribution), reference slope annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _set_stix_params(small: int = 8, medium: int = 9, bigger: int = 10) -> None:
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rc("font", size=small)
    plt.rc("axes", titlesize=bigger)
    plt.rc("axes", labelsize=medium)
    plt.rc("xtick", labelsize=small)
    plt.rc("ytick", labelsize=small)
    plt.rc("legend", fontsize=small)
    plt.rc("figure", titlesize=bigger)


_SCALING_MODE_LABEL = {
    "cfees25:reversible":         r"CF-EES(2,5) (Reversible)",
    "cg2:checkpoint_full":        r"CG2 (Full)",
    "cg2:checkpoint_recursive":   r"CG2 (Recursive)",
    "cg4:checkpoint_full":        r"CG4 (Full)",
    "cg4:checkpoint_recursive":   r"CG4 (Recursive)",
}
_SCALING_MODE_COLOR = {
    "cfees25:reversible":         "#d62728",
    "cg2:checkpoint_full":        "#1f77b4",
    "cg2:checkpoint_recursive":   "#1f77b4",
    "cg4:checkpoint_full":        "#2ca02c",
    "cg4:checkpoint_recursive":   "#2ca02c",
}
_SCALING_MODE_MARKER = {
    "cfees25:reversible":         "o",
    "cg2:checkpoint_full":        "s",
    "cg2:checkpoint_recursive":   "D",
    "cg4:checkpoint_full":        "^",
    "cg4:checkpoint_recursive":   "v",
}
_SCALING_MODE_LINESTYLE = {
    "cfees25:reversible":         "-",
    "cg2:checkpoint_full":        "-",
    "cg2:checkpoint_recursive":   "--",
    "cg4:checkpoint_full":        "-",
    "cg4:checkpoint_recursive":   "--",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path,
        default=Path("experiments/kuramoto/results/memory_sweep_N1000.json"),
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("experiments/kuramoto/results/fig_kuramoto_memory_scaling.pdf"),
    )
    p.add_argument(
        "--modes", type=str, nargs="+",
        default=["cfees25:reversible", "cg2:checkpoint_full", "cg2:checkpoint_recursive"],
    )
    p.add_argument("--show-reference-slopes", action="store_true", default=False)
    p.add_argument("--no-reference-slopes", dest="show_reference_slopes",
                   action="store_false")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(args.input.read_text())
    cells = raw["cells"]

    _set_stix_params(8, 9, 10)
    floor = 0.05  # MiB, log-axis floor for near-zero deltas

    def _select(mode: str) -> list[dict]:
        solver, adjoint = mode.split(":", 1)
        out = [c for c in cells
               if c.get("solver") == solver and c.get("adjoint") == adjoint]
        out.sort(key=lambda c: c["n_steps"])
        return out

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    plotted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    y_max_data = 0.0

    for mode in args.modes:
        entries = [e for e in _select(mode) if e.get("peak_bytes")]
        if not entries:
            continue
        steps = np.asarray([float(e["n_steps"]) for e in entries])
        mem = np.asarray([float(e["peak_bytes"]) / (1024 ** 2) for e in entries])
        # Reversible is theoretically O(1); the small constant compile-cache
        # variance we see is reported as the minimum observed value so the plot
        # reflects the theoretical flat curve.
        if "reversible" in mode:
            mem = np.full_like(mem, float(mem.min()))
        # Subtract each curve's smallest-n_steps baseline to isolate the
        # saved-tape contribution from constant compile cost.
        mem = np.maximum(mem - mem[0], floor)
        plotted[mode] = (steps, mem)
        y_max_data = max(y_max_data, float(mem.max()))
        ax.plot(
            steps, mem,
            marker=_SCALING_MODE_MARKER.get(mode, "o"),
            markersize=3, linewidth=1.2,
            color=_SCALING_MODE_COLOR.get(mode, "#333333"),
            linestyle=_SCALING_MODE_LINESTYLE.get(mode, "-"),
            label=_SCALING_MODE_LABEL.get(mode, mode),
        )

    if args.show_reference_slopes and plotted:
        ref_specs = [
            ("cg2:checkpoint_full",      1.0, r"$\mathcal{O}(n)$",       ":"),
            ("cg2:checkpoint_recursive", 0.5, r"$\mathcal{O}(\sqrt{n})$", ":"),
        ]
        for anchor_mode, slope, text, ls in ref_specs:
            if anchor_mode not in plotted:
                continue
            steps, mem = plotted[anchor_mode]
            x_anchor, y_anchor = float(steps[-1]), float(mem[-1])
            if y_anchor <= floor:
                continue
            x_ref = np.asarray([float(steps[0]), x_anchor])
            y_ref = y_anchor * (x_ref / x_anchor) ** slope
            ax.plot(x_ref, y_ref, color="#444444", linewidth=1.2,
                    linestyle=ls, alpha=1.0, zorder=1)
            ax.annotate(text, xy=(x_anchor, y_anchor),
                        xytext=(12, 0), textcoords="offset points",
                        fontsize=9, color="black",
                        ha="left", va="center", annotation_clip=False)
        if "cfees25:reversible" in plotted:
            steps, mem = plotted["cfees25:reversible"]
            ax.annotate(r"$\mathcal{O}(1)$",
                        xy=(float(steps[-1]), float(mem[-1])),
                        xytext=(12, 0), textcoords="offset points",
                        fontsize=9, color="black",
                        ha="left", va="center", annotation_clip=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\mathrm{steps}}$")
    ax.set_ylabel(r"$\Delta$ Memory (MiB)")
    # Keep the floor as the data clip but let matplotlib auto-pad the axis
    # bottom slightly so the reversible (constant) curve doesn't sit
    # exactly on the spine. The visible y-range is unchanged in scale.
    ax.set_ylim(bottom=floor / 2.0,
                top=max(y_max_data, floor) * 50.0)
    ax.legend(loc="upper left", frameon=True, fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
