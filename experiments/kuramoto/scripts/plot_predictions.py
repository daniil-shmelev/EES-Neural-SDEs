"""Render sample-trajectory + ensemble-fan figures from a trained model's
`predictions_demo.npz`.

Produces a 2x2 panel:
  - top-left:  sample trajectory of theta_i(t) for one held-out IC
                 (data ground truth in solid black, NSDE samples in faded red)
  - top-right: sample trajectory of omega_i(t) (same format)
  - bottom-left: per-horizon test ES across epochs (training curve)
  - bottom-right: order-parameter r(t) for data vs NSDE samples

Uses the per-step + per-epoch logging from the extended fit() loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="A hero-training output dir containing "
                        "`predictions_demo.npz` and `history.json`.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ic", type=int, default=0,
                   help="Which initial-condition index to plot (default 0).")
    p.add_argument("--n-osc-show", type=int, default=4,
                   help="How many oscillator angles to draw (default 4).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    demo = np.load(args.run_dir / "predictions_demo.npz")
    hist = json.loads((args.run_dir / "history.json").read_text())

    target_theta = demo["target_theta"]   # (n_ic, T_obs, N)
    target_omega = demo["target_omega"]
    sample_theta = demo["sample_theta"]   # (n_samples, n_ic, T_obs, N)
    sample_omega = demo["sample_omega"]
    t_grid = demo["t_grid"]

    ic = args.ic
    n_osc = min(args.n_osc_show, target_theta.shape[-1])

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), dpi=200)
    fig.suptitle(f"Kuramoto NSDE forecast — IC #{ic}", fontsize=11)

    # Top left: theta_i(t) for first n_osc oscillators
    ax = axes[0, 0]
    cmap = plt.get_cmap('tab10')
    for i in range(n_osc):
        c = cmap(i)
        # Sample trajectories (faded)
        for s in range(sample_theta.shape[0]):
            ax.plot(t_grid, sample_theta[s, ic, :, i], color=c, alpha=0.18, lw=0.7)
        # Ground truth
        ax.plot(t_grid, target_theta[ic, :, i], color=c, lw=1.5,
                label=fr"$\theta_{{{i}}}$")
    ax.set_xlabel("t (s)"); ax.set_ylabel(r"$\theta_i$ (rad)")
    ax.set_title("Phases (truth solid, NSDE samples faint)")
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Top right: omega_i(t)
    ax = axes[0, 1]
    for i in range(n_osc):
        c = cmap(i)
        for s in range(sample_omega.shape[0]):
            ax.plot(t_grid, sample_omega[s, ic, :, i], color=c, alpha=0.18, lw=0.7)
        ax.plot(t_grid, target_omega[ic, :, i], color=c, lw=1.5,
                label=fr"$\omega_{{{i}}}$")
    ax.set_xlabel("t (s)"); ax.set_ylabel(r"$\omega_i$")
    ax.set_title("Angular velocities")
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bottom left: training curves (test ES + per-horizon theta MAE)
    ax = axes[1, 0]
    epochs = np.arange(1, len(hist["test_loss"]) + 1)
    ax.plot(epochs, hist["train_loss"], label="train ES", color='0.4', lw=1.2)
    ax.plot(epochs, hist["val_loss"], label="val ES", color='#1f77b4', lw=1.5)
    ax.plot(epochs, hist["test_loss"], label="test ES", color='#d62728', lw=1.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("energy score")
    ax.set_title("Training curves")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Bottom right: order parameter r(t) data vs samples
    ax = axes[1, 1]
    z_data = np.mean(np.exp(1j * target_theta[ic]), axis=-1)  # (T_obs,)
    r_data = np.abs(z_data)
    ax.plot(t_grid, r_data, color='black', lw=1.8, label="data")
    z_samp = np.mean(np.exp(1j * sample_theta[:, ic]), axis=-1)  # (n_samp, T_obs)
    r_samp = np.abs(z_samp)
    for s in range(r_samp.shape[0]):
        ax.plot(t_grid, r_samp[s], color='#d62728', alpha=0.3, lw=0.8)
    ax.plot(t_grid, r_samp.mean(axis=0), color='#d62728', lw=1.5,
            label="NSDE sample mean")
    ax.fill_between(t_grid, r_samp.min(axis=0), r_samp.max(axis=0),
                    color='#d62728', alpha=0.12, label="NSDE sample range")
    ax.set_xlabel("t (s)"); ax.set_ylabel(r"order param $r(t)$")
    ax.set_title("Coherence")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
