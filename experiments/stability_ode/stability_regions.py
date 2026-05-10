"""Reproduce the paper's ODE stability-region figure.

Stability domain for EES(2,5;1/10) and EES(2,7) compared to RK4, the
McCallum--Foster reversible variant of Euler, and Reversible Heun on the
linear test ODE dy = alpha y dt.

The McCallum--Foster reversible scheme of arXiv:2410.11648 couples two states
(y, z) via

    y_{n+1} = lambda y_n + (1 - lambda) z_n + Psi_h(z_n),
    z_{n+1} = z_n - Psi_{-h}(y_{n+1}),

with `Psi_h(y) = R(h alpha) y` for the linear test ODE; the linearised
iteration is a 2x2 matrix whose spectral radius `< 1` defines the A-stability
region. The closed-form eigenvalue expressions below are taken verbatim from

    https://github.com/sammccallum/reversible/blob/master/experiments/stability/plot.py

i.e. the same formulas used to generate Figure 1 of arXiv:2410.11648v1, with
the published `lambda = 0.8`.

Reversible Heun is mean-square stable only on the imaginary segment [-i, i]
(Kidger et al. 2021, Theorem D.19), drawn here as a bold coloured segment on
the imaginary axis since it has empty interior.

EES(2,5;1/10) and RK4 contours are computed from their classical Butcher
tableaux via `kauri`, mirroring `examples/ees_schemes/stability.ipynb` from
the kauri repository (https://github.com/daniil-shmelev/kauri).
"""

import os
from math import sqrt

import kauri as kr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import set_plotting_params

# Larger sizes than the kauri default; the figure is intended for compact
# paper-column rendering, so labels need to survive a substantial scale-down.
set_plotting_params(16, 18, 20)


def eig_reversible_euler(l, h, a):
    """Reversible-Euler eigenvalues, verbatim from McCallum's plot.py."""
    common = (
        a**2 * h**2 / 2
        - a * h * l / 2
        + a * h / 2
        + l / 2
    )
    disc = np.sqrt(
        a**4 * h**4
        - 2 * a**3 * h**3 * l
        + 2 * a**3 * h**3
        + a**2 * h**2 * l**2
        + 3 * a**2 * h**2
        - 2 * a * h * l**2
        + 2 * a * h
        + l**2
        - 2 * l
        + 1
    )
    return common - disc / 2 + 0.5, common + disc / 2 + 0.5


def rk_stability_magnitude(method: kr.RK, z: np.ndarray) -> np.ndarray:
    """`|R(z)|` for the given explicit RK method, broadcast over `z`.

    Computes `|det(I - z A + z 1 b^T) / det(I - z A)|` in one shot using
    numpy broadcasting (no per-pixel Python loop).
    """
    A = np.array(method.a, dtype=np.complex128)
    b = np.array(method.b, dtype=np.complex128).reshape(-1, 1)
    s = A.shape[0]
    eye = np.eye(s, dtype=np.complex128)
    one_bT = np.ones((s, 1), dtype=np.complex128) @ b.T  # (s, s)

    z = np.asarray(z, dtype=np.complex128)
    z_bcast = z[..., None, None]
    num = eye - z_bcast * A + z_bcast * one_bT
    den = eye - z_bcast * A
    return np.abs(np.linalg.det(num) / np.linalg.det(den))


def main(
    xlim=(-4.5, 0.5),
    ylim=(-3.5, 3.5),
    resolution=600,
    coupling_parameter=0.8,
    output_dir=None,
):
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results'
        )
    os.makedirs(output_dir, exist_ok=True)

    h = 1.0
    l = coupling_parameter

    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    a = X + 1j * Y

    e1, e2 = eig_reversible_euler(l, h, a)
    rho_mcf_euler = np.maximum(np.abs(e1), np.abs(e2))

    R_rk4 = rk_stability_magnitude(kr.rk4, a)
    R_ees25 = rk_stability_magnitude(kr.EES25(0.1), a)
    R_ees27 = rk_stability_magnitude(kr.EES27((5 - 3 * sqrt(2)) / 14), a)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    # Smallest region last (highest zorder) so it isn't overdrawn by the bigger
    # contours.  Legend matches the smallest-first visual order.
    contours = [
        # values, label, linestyle, colour, lw, zorder
        (rho_mcf_euler, 'MCF Euler',  '-', 'mediumblue',  2.5, 8),
        (R_rk4,         'RK4',        ':', 'black',       2.5, 6),
        (R_ees25,       'EES(2,5)',   '-', 'black',       2.5, 5),
        (R_ees27,       'EES(2,7)',   '-', 'forestgreen', 2.5, 4),
    ]

    # Reversible Heun: stable only on the imaginary segment [-i, i].
    rev_heun_colour = 'firebrick'
    rev_heun_lw = 5.0
    ax.plot([0.0, 0.0], [-1.0, 1.0], color=rev_heun_colour,
            linewidth=rev_heun_lw, solid_capstyle='round', zorder=10)
    legend_handles = [
        mlines.Line2D([], [], color=rev_heun_colour, linewidth=rev_heun_lw,
                      label='Reversible Heun')
    ]
    for values, label, ls, colour, lw, zorder in contours:
        ax.contour(X, Y, values, levels=[1.0],
                   colors=colour, linewidths=lw, linestyles=ls, zorder=zorder)
        legend_handles.append(
            mlines.Line2D([], [], color=colour, linewidth=lw, linestyle=ls,
                          label=label)
        )

    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([-4, -3, -2, -1, 0])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_title('Stability Regions')
    ax.legend(handles=legend_handles, loc='upper left', frameon=True,
              fontsize=12, handlelength=1.6, borderpad=0.4,
              labelspacing=0.3, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_regions_1.pdf'))
    plt.show()


if __name__ == '__main__':
    main()
