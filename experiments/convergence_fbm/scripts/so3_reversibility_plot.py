"""Plot convergence curves from so3_reversibility_verify.py JSON output.

Produces a log-log plot: reversibility error ||Y_back - Y_0||_F vs h,
with reference slopes 6H - 1 (BCK order-5 prediction) and 4H - 1
(MKW order-3 prediction) drawn as dashed lines.

Usage:
  uv run python scripts/so3_reversibility_plot.py \
      so3_reversibility_results.json \
      --output so3_reversibility_convergence.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {0.4: "tab:blue", 0.5: "tab:orange", 0.6: "tab:green"}
MARKERS = {0.4: "o", 0.5: "s", 0.6: "^"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_in")
    ap.add_argument("--output", default="so3_reversibility_convergence.pdf")
    ap.add_argument("--title",
                    default="CF-EES(2,5;1/10) reversibility on SO(3)")
    args = ap.parse_args()

    with open(args.json_in) as f:
        records = json.load(f)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    for rec in records:
        H = rec["H"]
        hs = np.array([g["h"] for g in rec["per_grid"]])
        errs = np.array([g["mean_err"] for g in rec["per_grid"]])
        color = COLORS.get(H, "gray")
        marker = MARKERS.get(H, "o")

        ax.loglog(hs, errs, marker=marker, color=color,
                  linewidth=1.2, markersize=6,
                  label=f"H={H:.2f}  (slope={rec['fitted_slope']:.2f})")

        # Reference lines: 6H - 1 and 4H - 1
        h_ref = np.array([hs.min(), hs.max()])
        err_anchor = errs[-1]  # anchor at finest h
        h_anchor = hs[-1]

        # BCK order-5 line: slope 6H - 1
        c_bck = err_anchor / h_anchor ** (6 * H - 1)
        ax.loglog(h_ref, c_bck * h_ref ** (6 * H - 1),
                  color=color, linestyle="--", alpha=0.55, linewidth=1.0)

        # MKW order-3 line: slope 4H - 1 (drawn with a longer dash)
        c_mkw = err_anchor / h_anchor ** (4 * H - 1)
        ax.loglog(h_ref, c_mkw * h_ref ** (4 * H - 1),
                  color=color, linestyle=":", alpha=0.55, linewidth=1.0)

    ax.set_xlabel("step size  $h$")
    ax.set_ylabel(r"$\|\Phi_{-h}\!\circ\!\Phi_{+h}(Y_0) - Y_0\|_F$ (mean over trials)")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)

    # Build a single custom legend: the empirical curves + two reference
    # entries (dashed = BCK 6H-1, dotted = MKW 4H-1).
    from matplotlib.lines import Line2D
    empirical_handles, empirical_labels = ax.get_legend_handles_labels()
    ref_handles = [
        Line2D([0], [0], color="black", linestyle="--", alpha=0.65,
               label="reference slope $6H-1$ (BCK order 5)"),
        Line2D([0], [0], color="black", linestyle=":", alpha=0.65,
               label="reference slope $4H-1$ (MKW order 3)"),
    ]
    ax.legend(
        handles=empirical_handles + ref_handles,
        loc="lower right", fontsize=9, framealpha=0.9,
    )

    fig.tight_layout()
    out = Path(args.output)
    fig.savefig(out, dpi=200)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
