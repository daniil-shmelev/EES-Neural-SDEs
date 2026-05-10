"""Plot the stochastic-volatility gradient-error sweep.

Reads ``summary.json`` from ``grad_error_sweep.py`` and emits one PDF per
sweep + a cross-experiment overlay. Uses the project's standard plot style
(STIXGeneral, sizes 11/12/13, figsize (4, 3)).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import set_plotting_params


def _power_law_fit(
    h: np.ndarray, y: np.ndarray, *, y_min: float = 1e-6, y_max: float = 1.0,
) -> tuple[float, float, np.ndarray]:
    mask = (y > y_min) & (y < y_max) & np.isfinite(y) & np.isfinite(h)
    if mask.sum() < 2:
        return float("nan"), float("nan"), mask
    coeffs = np.polyfit(np.log(h[mask]), np.log(y[mask]), 1)
    return float(coeffs[0]), float(coeffs[1]), mask


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path,
                   default=Path("experiments/stochastic_volatility/results/grad_error/single_step/summary.json"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/stochastic_volatility/results/grad_error/single_step/StochVol_grad_error.pdf"))
    p.add_argument("--no-fit", action="store_true")
    p.add_argument("--fit-y-min", type=float, default=1e-6)
    p.add_argument("--fit-y-max", type=float, default=1.0)
    p.add_argument("--ylim", type=str, default=None)
    p.add_argument("--xlim", type=str, default=None)
    p.add_argument("--h-min", type=float, default=None,
                   help="Drop dt values below this threshold (the noise-floor regime).")
    args = p.parse_args(argv)

    summary = json.loads(args.summary.read_text())
    experiments = summary["experiments"]
    nfe_budgets = sorted(summary["nfe_budgets"])
    total_time = summary["base_overrides"]["total_time"]

    set_plotting_params(11, 12, 13)
    fig, ax = plt.subplots(figsize=(4, 3))

    palette = {
        "rough_bergomi": ("red", "o", "Rough Bergomi"),
        "bergomi": ("orange", "s", "Bergomi"),
        "heston": ("blue", "^", "Heston"),
        "rough_heston": ("purple", "v", "Rough Heston"),
        "quadratic_rough_heston": ("brown", "D", "Quadratic Rough Heston"),
        "black_scholes": ("grey", "P", "Black-Scholes"),
        "local_stoch_vol": ("teal", "X", "Local Stoch.\\ Vol"),
    }

    for exp in experiments:
        if exp not in palette:
            continue
        color, marker, label = palette[exp]
        rel_err_per_nfe = summary["rel_err"][exp]
        nfe_arr = np.array(nfe_budgets, dtype=float)
        dt_arr = total_time / nfe_arr  # one solver step per NFE budget unit
        means = np.array([rel_err_per_nfe[str(n)]["mean"] for n in nfe_budgets], dtype=float)
        stderrs = np.array([rel_err_per_nfe[str(n)]["stderr"] or 0.0 for n in nfe_budgets], dtype=float)

        keep = np.ones_like(dt_arr, dtype=bool)
        if args.h_min is not None:
            keep = dt_arr >= args.h_min

        full_label = label
        if not args.no_fit:
            slope, _, _ = _power_law_fit(
                dt_arr[keep], means[keep],
                y_min=args.fit_y_min, y_max=args.fit_y_max,
            )
            if np.isfinite(slope):
                full_label = f"{label} (slope $\\approx {slope:.2f}$)"

        ax.errorbar(
            dt_arr[keep], np.maximum(means[keep], 1e-13),
            yerr=stderrs[keep], capsize=2, linewidth=1.4,
            color=color, marker=marker, markersize=4, label=full_label,
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
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[plot] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
