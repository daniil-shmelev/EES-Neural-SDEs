"""Convergence panels shared by the ODE and fractional-Brownian experiments."""

import matplotlib.markers as mkr
import numpy as np


def plot_error_curve(h, log_error, rate, ax, *, backward=False, reference_mask=None):
    """Plot log-errors and a prescribed slope with a fitted intercept."""
    x = np.log10(np.asarray(h))
    y = np.asarray(log_error)
    mask = np.ones(x.shape, dtype=bool) if reference_mask is None else reference_mask
    if not np.any(mask):
        raise ValueError("The slope guide needs at least one resolved error.")
    intercept = float(np.mean(y[mask]) - rate * np.mean(x[mask]))
    dx = np.array([x[0], x[-1]])
    err_label = (
        r"$\log_{10}(\overleftarrow{\mathcal{E}}(h))$"
        if backward else r"$\log_{10}(\mathcal{E}(h))$"
    )
    points = ax.scatter(
        x, y, marker=mkr.MarkerStyle("x", fillstyle="none"), color="crimson"
    )
    line, = ax.plot(dx, rate * dx + intercept, color="mediumblue")
    ax.legend([points, line], [err_label, f"{rate:.1f}" + r"$x + c$"])
    ax.set_xlabel(r"$\log_{10}(h)$")
    ax.set_ylabel(err_label)
    return intercept
