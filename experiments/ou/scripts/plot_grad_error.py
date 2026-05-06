"""Plot the OU per-step gradient-error sweep into ``OU_grad_error.pdf``."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ees_core.plotting_params import set_plotting_params

PAPER_FIGURES_DIR = Path(
    "/mnt/c/Users/Shmelev/source/overleaf/EES_Neural_SDEs_Overleaf/figures"
)

STYLE = {
    "ees25": dict(color="red", marker="o", markersize=4, label="EES(2,5)"),
    "ees27": dict(color="green", marker="^", markersize=4, label="EES(2,7)"),
    "reversible_heun": dict(color="blue", marker="s", markersize=4, linestyle="--",
                            label="Reversible Heun"),
}


def _power_law_fit(
    h: np.ndarray, y: np.ndarray, *, y_min: float = 1e-6, y_max: float = 1.0,
) -> tuple[float, float, np.ndarray]:
    mask = (y > y_min) & (y < y_max) & np.isfinite(y) & np.isfinite(h)
    if mask.sum() < 2:
        return float("nan"), float("nan"), mask
    coeffs = np.polyfit(np.log(h[mask]), np.log(y[mask]), 1)
    return float(coeffs[0]), float(coeffs[1]), mask


def _publish_to_paper(local_pdf: Path, paper_name: str) -> None:
    if not PAPER_FIGURES_DIR.is_dir():
        print(f"[publish] {PAPER_FIGURES_DIR} not present — skipping.")
        return
    dst = PAPER_FIGURES_DIR / paper_name
    shutil.copy2(local_pdf, dst)
    print(f"[publish] copied {local_pdf} -> {dst}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path,
                   default=Path("experiments/ou/results/grad_error/single_step/summary.json"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/ou/results/grad_error/single_step/OU_grad_error.pdf"))
    p.add_argument("--paper-name", type=str, default="OU_grad_error.pdf")
    p.add_argument("--no-publish", action="store_true")
    p.add_argument("--no-fit", action="store_true")
    p.add_argument("--fit-y-min", type=float, default=1e-6)
    p.add_argument("--fit-y-max", type=float, default=1.0)
    p.add_argument("--ylim", type=str, default=None)
    p.add_argument("--xlim", type=str, default=None)
    p.add_argument("--h-min", type=float, default=None,
                   help="Drop h values below this threshold (the noise-floor regime).")
    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated subset of methods. Default: all in summary.json.")
    args = p.parse_args(argv)

    summary = json.loads(args.summary.read_text())
    h_values = np.array(summary["h"], dtype=float)
    methods = summary["methods"]
    if args.methods is not None:
        keep = [m.strip() for m in args.methods.split(",") if m.strip()]
        methods = [m for m in methods if m in keep]

    if args.h_min is not None:
        keep_idx = h_values >= args.h_min
        h_values = h_values[keep_idx]
    else:
        keep_idx = np.ones_like(h_values, dtype=bool)

    set_plotting_params(11, 12, 13)
    fig, ax = plt.subplots(figsize=(4, 3))

    for method in methods:
        if method not in STYLE:
            continue
        rel_err_mean = np.array(summary["rel_err"][method]["mean"], dtype=float)[keep_idx]
        rel_err_stderr = np.array(summary["rel_err"][method]["stderr"], dtype=float)[keep_idx]

        style = dict(STYLE[method])
        if not args.no_fit and method != "reversible_heun":
            slope, _, _ = _power_law_fit(
                h_values, rel_err_mean,
                y_min=args.fit_y_min, y_max=args.fit_y_max,
            )
            if np.isfinite(slope):
                style["label"] = f"{style['label']} (slope $\\approx {slope:.2f}$)"

        ax.errorbar(
            h_values, np.maximum(rel_err_mean, 1e-13),
            yerr=rel_err_stderr, capsize=2, linewidth=1.4,
            **style,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    if args.ylim:
        lo, hi = (float(x) for x in args.ylim.split(":"))
        ax.set_ylim(lo, hi)
    if args.xlim:
        lo, hi = (float(x) for x in args.xlim.split(":"))
        ax.set_xlim(lo, hi)
    ax.set_xlabel("Step size $h$")
    ax.set_ylabel("Relative gradient error")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[plot] wrote {args.output}")

    if not args.no_publish:
        _publish_to_paper(args.output, args.paper_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
