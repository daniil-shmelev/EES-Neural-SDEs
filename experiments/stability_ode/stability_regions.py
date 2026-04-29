"""Reproduce Figure 1 of the manuscript (`fig:ees25_stability`).

Stability domain for EES(2,5;1/10) compared to Kutta's RK3 and RK4 on the linear
test ODE dy = lambda y dt. The stability region for an explicit RK method (A, b)
is the set of complex z = lambda h such that

    R(z) := |det(I - z A + z 1 b^T) / det(I - z A)| < 1.

This script mirrors `examples/ees_schemes/stability.ipynb` from the kauri
repository (https://github.com/daniil-shmelev/kauri), specialised to a single
panel that overlays the three schemes' |R(z)| = 1 contours.
"""

import os

import kauri as kr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# Plotting style — matches the kauri stability notebook exactly.
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 12, 13, 14
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def stability_function(method: kr.RK):
    """|R(z)| = |det(I - z A + z 1 b^T) / det(I - z A)| for the given RK method."""
    A = np.array(method.a, dtype=np.complex128)
    b = np.array(method.b, dtype=np.complex128)[:, np.newaxis]
    eye = np.eye(len(A), dtype=np.complex128)
    one = np.ones(shape=b.shape, dtype=np.complex128)

    def _R(z):
        return np.abs(
            np.linalg.det(eye - z * A + z * one @ b.T)
            / np.linalg.det(eye - z * A)
        )

    return np.vectorize(_R)


def main(
    xlim=(-5.0, 2.5),
    ylim=(-3.6, 3.6),
    resolution=500,
    output_dir=None,
):
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results'
        )
    os.makedirs(output_dir, exist_ok=True)

    # Order matters — this is the legend order in Figure 1.
    methods = [
        (kr.kutta_rk3,  'RK3',       '--'),
        (kr.rk4,        'RK4',       ':'),
        (kr.EES25(0.1), 'EES(2,5)',  '-'),
    ]

    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    fig, ax = plt.subplots(figsize=(5, 5))

    legend_handles = []
    for method, label, linestyle in methods:
        R = stability_function(method)(Z)
        ax.contour(X, Y, R, levels=[1], colors='black',
                   linewidths=1.5, linestyles=linestyle)
        legend_handles.append(
            mlines.Line2D([], [], color='black', linewidth=1.5,
                          linestyle=linestyle, label=label)
        )

    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([-4, -2, 0, 2])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Stability Regions')
    ax.legend(handles=legend_handles, loc='upper left', frameon=True)

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'stability_regions_1.pdf'))
    plt.show()


if __name__ == '__main__':
    main()
