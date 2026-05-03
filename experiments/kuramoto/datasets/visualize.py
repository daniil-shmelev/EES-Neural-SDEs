r"""Render a Kuramoto trajectory as an animated GIF.

Standard visualisation: $N$ phase oscillators as dots on the unit circle
$e^{i\theta_j}$ together with the complex order-parameter arrow
$re^{i\Psi} = \frac{1}{N}\sum_j e^{i\theta_j}$. The arrow length $r \in
[0, 1]$ is the synchronisation coherence — $r \approx 0$ in the
incoherent regime, $r \approx 1$ at full synchrony.

Used by `scripts/run_m1.py` to produce a sanity-check animation of one
of the trajectories that was just simulated.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.animation as mpl_animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render_trajectory_gif(
    theta: np.ndarray,
    t_grid: np.ndarray,
    out_path: Path | str,
    fps: int = 30,
    trail_length: int = 40,
    figsize_inches: float = 4.5,
    dpi: int = 100,
) -> Path:
    """Render an N-oscillator Kuramoto trajectory to a GIF.

    Parameters
    ----------
    theta : (T, N) array of phases in radians (already wrapped).
    t_grid : (T,) timepoints in seconds.
    out_path : Output GIF path.
    fps : Animation frame rate.
    trail_length : Number of past frames over which to fade in the
        order-parameter trail.
    figsize_inches, dpi : Output sizing.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames, n_osc = theta.shape
    if n_frames != t_grid.shape[0]:
        raise ValueError("theta and t_grid must have the same length.")

    osc_x = np.cos(theta)
    osc_y = np.sin(theta)
    z = np.mean(np.exp(1j * theta), axis=1)
    r = np.abs(z)
    psi_x = z.real
    psi_y = z.imag

    fig, ax = plt.subplots(figsize=(figsize_inches, figsize_inches), dpi=dpi)
    ax.set_aspect("equal")
    pad = 0.15
    ax.set_xlim(-1 - pad, 1 + pad)
    ax.set_ylim(-1 - pad, 1 + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    circle_t = np.linspace(0.0, 2.0 * np.pi, 256)
    ax.plot(np.cos(circle_t), np.sin(circle_t), color="0.85", linewidth=1.0, zorder=0)
    ax.axhline(0.0, color="0.92", linewidth=0.5, zorder=0)
    ax.axvline(0.0, color="0.92", linewidth=0.5, zorder=0)

    cmap = plt.get_cmap("viridis")
    osc_colors = [cmap(i / max(n_osc - 1, 1)) for i in range(n_osc)]
    osc_scatter = ax.scatter(
        osc_x[0], osc_y[0], c=osc_colors, s=80, edgecolor="white", linewidth=1.0, zorder=3
    )

    arrow_artists: list = []  # repopulated each frame; fresh quiver per call

    trail_lines = []
    for k in range(trail_length):
        alpha = (k + 1) / (trail_length + 1)
        (l,) = ax.plot([], [], color="#d62728", linewidth=1.0, alpha=alpha, zorder=1)
        trail_lines.append(l)

    time_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, va="top", ha="left",
        fontsize=9, color="0.3", family="monospace",
    )

    def init():
        osc_scatter.set_offsets(np.column_stack([osc_x[0], osc_y[0]]))
        for l in trail_lines:
            l.set_data([], [])
        time_text.set_text("")
        return [osc_scatter, *trail_lines, time_text]

    def update(frame):
        for q in arrow_artists:
            q.remove()
        arrow_artists.clear()

        osc_scatter.set_offsets(np.column_stack([osc_x[frame], osc_y[frame]]))
        q = ax.quiver(
            0.0, 0.0, psi_x[frame], psi_y[frame],
            angles="xy", scale_units="xy", scale=1.0,
            color="#d62728", width=0.012, zorder=2,
        )
        arrow_artists.append(q)

        for k in range(trail_length):
            i = frame - trail_length + k
            if i < 0 or i + 1 >= n_frames:
                trail_lines[k].set_data([], [])
            else:
                trail_lines[k].set_data(
                    [psi_x[i], psi_x[i + 1]],
                    [psi_y[i], psi_y[i + 1]],
                )
        time_text.set_text(
            f"t = {t_grid[frame]:5.2f} s\nN = {n_osc}\nr = {r[frame]:.3f}"
        )
        return [osc_scatter, q, *trail_lines, time_text]

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
    )
    anim.save(out_path, writer=mpl_animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
