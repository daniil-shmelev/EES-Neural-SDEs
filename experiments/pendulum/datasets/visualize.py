"""Render a sampled n-link pendulum trajectory as an animated GIF.

Used by `scripts/generate_data.py` to produce a sanity-check / demo
animation of one of the trajectories that was just simulated. Pure
matplotlib, writes via the built-in `PillowWriter`, no extra deps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import jax  # noqa: E402
import matplotlib.animation as mpl_animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.pendulum.datasets.lagrangian import (  # noqa: E402
    PendulumParams,
    link_tip_positions,
)


def render_trajectory_gif(
    theta: np.ndarray,
    params: PendulumParams,
    t_grid: np.ndarray,
    out_path: Path | str,
    fps: int = 30,
    trail_length: int = 40,
    figsize_inches: float = 4.5,
    dpi: int = 100,
) -> Path:
    """Render an n-link pendulum trajectory to a GIF.

    Parameters
    ----------
    theta : (T, n) array of joint angles in radians.
    params : Physical parameters (used for link lengths).
    t_grid : (T,) array of timepoints in seconds (for the time-stamp overlay).
    out_path : Output GIF path.
    fps : Animation frame rate.
    trail_length : Number of past frames over which to fade in the tip trail.
    figsize_inches, dpi : Output sizing.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames, n_links = theta.shape
    if n_frames != t_grid.shape[0]:
        raise ValueError("theta and t_grid must have the same length.")

    tips = np.asarray(jax.vmap(lambda th: link_tip_positions(th, params))(theta))
    pivots = np.zeros((n_frames, 1, 2))
    chain = np.concatenate([pivots, tips], axis=1)

    chain_radius = float(np.sum(np.asarray(params.lengths)))
    pad = 0.1 * chain_radius

    fig, ax = plt.subplots(figsize=(figsize_inches, figsize_inches), dpi=dpi)
    ax.set_aspect("equal")
    ax.set_xlim(-chain_radius - pad, chain_radius + pad)
    ax.set_ylim(-chain_radius - pad, chain_radius + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.axhline(0.0, color="0.85", linewidth=0.6)
    ax.axvline(0.0, color="0.85", linewidth=0.6)

    (chain_line,) = ax.plot(
        [], [], color="#1f77b4", linewidth=2.0, marker="o", markersize=5.0,
        markerfacecolor="#1f77b4", markeredgecolor="none",
    )
    (pivot_dot,) = ax.plot([0.0], [0.0], marker="s", color="0.4", markersize=4.0)

    # Fading trail of the final tip.
    trail_lines = []
    for k in range(trail_length):
        alpha = (k + 1) / (trail_length + 1)
        (l,) = ax.plot([], [], color="#d62728", linewidth=1.0, alpha=alpha)
        trail_lines.append(l)

    time_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, va="top", ha="left",
        fontsize=9, color="0.3", family="monospace",
    )

    def init():
        chain_line.set_data([], [])
        for l in trail_lines:
            l.set_data([], [])
        time_text.set_text("")
        return [chain_line, *trail_lines, time_text, pivot_dot]

    def update(frame):
        xs = chain[frame, :, 0]
        ys = chain[frame, :, 1]
        chain_line.set_data(xs, ys)
        # Tip trail — last `trail_length` segments of the final mass.
        for k in range(trail_length):
            i = frame - trail_length + k
            if i < 0 or i + 1 >= n_frames:
                trail_lines[k].set_data([], [])
            else:
                trail_lines[k].set_data(
                    [chain[i, -1, 0], chain[i + 1, -1, 0]],
                    [chain[i, -1, 1], chain[i + 1, -1, 1]],
                )
        time_text.set_text(f"t = {t_grid[frame]:5.2f} s\nn = {n_links}")
        return [chain_line, *trail_lines, time_text, pivot_dot]

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=True,
    )

    writer = mpl_animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return out_path
