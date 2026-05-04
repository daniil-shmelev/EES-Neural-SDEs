"""Render the M4 memory-scaling figure from `memory_sweep_N{N}.json`.

Replaces the placeholder `figures/fig_kuramoto_memory_scaling.pdf` in the
manuscript with the real data from the sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "cfees25:reversible":         "#d62728",  # red, our method
    "cg2:checkpoint_full":        "#1f77b4",  # blue
    "cg2:checkpoint_recursive":   "#2ca02c",  # green
    "cg4:checkpoint_full":        "#9467bd",  # purple
    "cg4:checkpoint_recursive":   "#8c564b",  # brown
    "rkmk:reversible":            "#e377c2",
}
LABELS = {
    "cfees25:reversible":       r"$\mathrm{CF\text{-}EES}(2,5)$ + ReversibleAdjoint",
    "cg2:checkpoint_full":      r"CG2 + Checkpoint (full tape)",
    "cg2:checkpoint_recursive": r"CG2 + Checkpoint (treeverse)",
    "cg4:checkpoint_full":      r"CG4 + Checkpoint (full tape)",
    "cg4:checkpoint_recursive": r"CG4 + Checkpoint (treeverse)",
}
MARKERS = {
    "cfees25:reversible": "o",
    "cg2:checkpoint_full": "s",
    "cg2:checkpoint_recursive": "^",
    "cg4:checkpoint_full": "D",
    "cg4:checkpoint_recursive": "v",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("experiments/kuramoto/results/memory_sweep_N1000.json"))
    p.add_argument(
        "--output", type=Path,
        default=Path("/mnt/c/Users/Shmelev/source/overleaf/EES_Neural_SDEs_Overleaf/figures/fig_kuramoto_memory_scaling.pdf"),
    )
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--show-oom", action="store_true", default=True,
                   help="Mark OOM cells with an ✗ at the bottom of the column.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    cells = payload["cells"]
    cfg = payload["config"]
    N = cfg["N"]; batch = cfg["batch_size"]

    # Group by (solver, adjoint)
    series: dict[str, list[tuple[int, float | None, bool, float | None]]] = {}
    for c in cells:
        key = f"{c['solver']}:{c['adjoint']}"
        peak = c.get("peak_bytes")
        series.setdefault(key, []).append((
            c["n_steps"],
            peak / 2**20 if peak else None,
            bool(c.get("oom")),
            c.get("wall_clock_s_mean"),
        ))
    for k in series:
        series[k].sort(key=lambda x: x[0])

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=200)

    all_n_steps = sorted({c["n_steps"] for c in cells})

    for key, points in series.items():
        color = COLORS.get(key, "0.3")
        marker = MARKERS.get(key, "x")
        label = LABELS.get(key, key)

        ns = [p[0] for p in points if p[1] is not None]
        ms = [p[1] for p in points if p[1] is not None]
        if ns:
            ax.plot(ns, ms, color=color, marker=marker, lw=2.0, ms=6,
                    label=label, zorder=3)
        # OOM markers
        oom_ns = [p[0] for p in points if p[2] and p[1] is None]
        if oom_ns and args.show_oom:
            for nn in oom_ns:
                ax.scatter([nn], [1.0], color=color, marker='x', s=80,
                           linewidths=2.0, zorder=4)

    # Reference scaling lines
    if all_n_steps:
        n_arr = np.array(sorted(all_n_steps))
        # rough O(n) and O(sqrt(n)) reference, anchored at the smallest n_steps
        ax.plot(n_arr, n_arr / n_arr[0] * 50, color='0.7', ls=':', lw=1.0,
                label=r"$\mathcal{O}(n)$ ref", zorder=1)
        ax.plot(n_arr, np.sqrt(n_arr / n_arr[0]) * 50, color='0.7', ls='--',
                lw=1.0, label=r"$\mathcal{O}(\sqrt{n})$ ref", zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"integration steps  $n_{\mathrm{steps}}$")
    ax.set_ylabel(r"peak GPU memory (MiB)")
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(rf"$N={N}$ Kuramoto, batch ${batch}$ — one fwd+bwd",
                      fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.85)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
