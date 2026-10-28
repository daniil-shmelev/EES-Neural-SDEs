"""Historical plotting routine for the NeurIPS torus-memory measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _set_stix_params(small: int = 7, medium: int = 8, bigger: int = 9) -> None:
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
    "cfees25:reversible": r"CF-EES(2,5) + Reversible Adj.",
    "cg2:checkpoint_full": r"CG2 + Full Adj.",
    "cg2:checkpoint_recursive": r"CG2 + Recursive Adj.",
    "geometric_euler_heun:checkpoint_full": "Geometric Euler–Heun + Full Adj.",
    "geometric_euler_heun:checkpoint_recursive": "Geometric Euler–Heun + Recursive Adj.",
    "cg4:checkpoint_full": r"CG4 + Full Adj.",
    "cg4:checkpoint_recursive": r"CG4 + Recursive Adj.",
}
_SCALING_MODE_COLOR = {
    "cfees25:reversible": "#d62728",
    "cg2:checkpoint_full": "#1f77b4",
    "cg2:checkpoint_recursive": "#1f77b4",
    "geometric_euler_heun:checkpoint_full": "#9467bd",
    "geometric_euler_heun:checkpoint_recursive": "#9467bd",
    "cg4:checkpoint_full": "#2ca02c",
    "cg4:checkpoint_recursive": "#2ca02c",
}
_SCALING_MODE_MARKER = {
    "cfees25:reversible": "o",
    "cg2:checkpoint_full": "s",
    "cg2:checkpoint_recursive": "D",
    "geometric_euler_heun:checkpoint_full": "P",
    "geometric_euler_heun:checkpoint_recursive": "X",
    "cg4:checkpoint_full": "^",
    "cg4:checkpoint_recursive": "v",
}
_SCALING_MODE_LINESTYLE = {
    "cfees25:reversible": "-",
    "cg2:checkpoint_full": "-",
    "cg2:checkpoint_recursive": "--",
    "geometric_euler_heun:checkpoint_full": "-",
    "geometric_euler_heun:checkpoint_recursive": "--",
    "cg4:checkpoint_full": "-",
    "cg4:checkpoint_recursive": "--",
}


def fig_torus_scaling(
    json_path: Path,
    save_path: Path,
    modes: list[str] | None = None,
    show_reference_slopes: bool = False,
    subtract_baseline: bool = False,
) -> None:
    """Memory-scaling plot from ``rna_benchmark_scaling.json``.

    Plots absolute XLA scratch (MiB) on log-log axes for each
    (solver, adjoint) curve. If ``subtract_baseline`` is true, each
    curve is plotted as ``mem(n) - mem(min n)`` so only the part that
    grows with ``n_steps`` (the saved-tape contribution) is visible;
    this isolates the adjoint-tape memory from constant compile cost.
    Reference $\\mathcal{O}(n)$ / $\\mathcal{O}(\\sqrt{n})$ slopes are
    off by default (they only fit when the data spans several orders
    of magnitude).
    """
    if modes is None:
        modes = [
            "cfees25:reversible",
            "cg2:checkpoint_full",
            "cg2:checkpoint_recursive",
            "geometric_euler_heun:checkpoint_full",
            "geometric_euler_heun:checkpoint_recursive",
        ]
    _set_stix_params(8, 9, 10)

    with json_path.open() as f:
        raw = json.load(f)
    cells = list(raw.values()) if isinstance(raw, dict) else raw

    def _select(mode: str) -> list[dict]:
        solver, adjoint = mode.split(":", 1)
        out = [
            c
            for c in cells
            if c.get("solver") == solver and c.get("adjoint") == adjoint
        ]
        out.sort(key=lambda c: c["n_steps"])
        return out

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    floor = 0.05  # MiB, log-axis floor for near-zero deltas
    plotted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mode in modes:
        entries = _select(mode)
        if not entries:
            continue
        steps = np.asarray([float(e["n_steps"]) for e in entries])
        mem = np.asarray([(e.get("temp_bytes") or 0) / (1024**2) for e in entries])
        # Reversible is theoretically O(1); the ~0.1-0.5 MiB compile-heuristic
        # variance we see here is reported as the minimum observed value so the
        # plot reflects the theoretical flat curve.
        if "reversible" in mode:
            mem = np.full_like(mem, float(mem.min()))
        if subtract_baseline:
            mem = np.maximum(mem - mem[0], floor)
        plotted[mode] = (steps, mem)
        ax.plot(
            steps,
            mem,
            marker=_SCALING_MODE_MARKER.get(mode, "o"),
            markersize=3,
            linewidth=1.2,
            color=_SCALING_MODE_COLOR.get(mode, "#333333"),
            linestyle=_SCALING_MODE_LINESTYLE.get(mode, "-"),
            label=_SCALING_MODE_LABEL.get(mode, mode),
        )

    if show_reference_slopes and plotted:
        # Anchor each reference line to the rightmost point of the curve
        # whose theoretical scaling it represents. CFEES is the O(1) line
        # itself (already flat), so it gets a label but no separate reference.
        ref_specs = [
            ("cg2:checkpoint_full", 1.0, r"$\mathcal{O}(n)$", ":"),
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
            ax.plot(
                x_ref,
                y_ref,
                color="#444444",
                linewidth=1.2,
                linestyle=ls,
                alpha=1.0,
                zorder=1,
            )
            ax.annotate(
                text,
                xy=(x_anchor, y_anchor),
                xytext=(12, 0),
                textcoords="offset points",
                fontsize=9,
                color="black",
                ha="left",
                va="center",
                annotation_clip=False,
            )
        # O(1) label for the reversible curve (already flat).
        if "cfees25:reversible" in plotted:
            steps, mem = plotted["cfees25:reversible"]
            ax.annotate(
                r"$\mathcal{O}(1)$",
                xy=(float(steps[-1]), float(mem[-1])),
                xytext=(12, 0),
                textcoords="offset points",
                fontsize=9,
                color="black",
                ha="left",
                va="center",
                annotation_clip=False,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\mathrm{steps}}$")
    if subtract_baseline:
        ax.set_ylabel(r"$\Delta$ Memory (MiB)")
    else:
        ax.set_ylabel(r"XLA scratch (MiB)")
    ax.legend(
        loc="upper left" if subtract_baseline else "center left",
        frameon=True,
        fontsize=7,
    )
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "results/rna_benchmark_scaling.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig_torus_scaling(
        args.input, args.output, subtract_baseline=True, show_reference_slopes=True
    )
