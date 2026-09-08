"""
This file plots cross-sections of the mean-square stability domains for RK4 and
EES(2,5) applied to SDEs, together with the published ODE stability set for
Reversible Heun. For EES(2,5), the solution is mean-square stable if and only if:

E[ ( 1 + x + (1/2)x^2 + (1/8)x^3 )^2 ] < 1

where x = lambda dt + mu dW is a normal random variable N(lambda dt, mu^2 dt). An explicit form
for the stability domain can be derived using the moments of a normal distribution. For simplicity,
we do this on-the-fly here.

Similarly, RK4 has the domain:

E[ ( 1 + x + (1/2)x^2 + (1/6)x^3 + (1/24)x^4 )^2 ] < 1.

Reversible Heun is not a one-step RK method, so it is not represented by the
standard Heun stability polynomial. Its published ODE stability set is the
imaginary segment [-i, i]; this is overlaid in red where it intersects these
cross-sections.

The resulting domains are 4-dimensional (2 complex dimensions). We plot 4 cross-sections
of these domains, given by:

(lambda dt, mu^2 dt) = (alpha, beta)                     <--- Cross-section along real axes
(lambda dt, mu^2 dt) = (alpha * i, beta * i)             <--- Cross-section along imaginary axes
(lambda dt, mu^2 dt) = (alpha * (1+i), beta * (1+i))     <--- Cross-section along diagonal
(lambda dt, mu^2 dt) = (alpha * (1+i), beta * (1-i))     <--- Cross-section along diagonal

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

###############################
# Plotting parameters
###############################

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 25, 30, 30

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

###############################
# Moments of N(x,y**2)
###############################

MOMENTS = [
    lambda x,y : 1.,
    lambda x,y : x,
    lambda x,y : x**2 + y**2,
    lambda x,y : x**3 + 3 * x * y**2,
    lambda x,y : x**4 + 6 * x**2 * y**2 + 3 * y**4,
    lambda x,y : x**5 + 10 * x**3*y**2 + 15*x*y**4,
    lambda x,y : x**6 + 15*x**4*y**2 + 45*x**2*y**4 + 15*y**6,
    lambda x,y : x**7 + 21*x**5*y**2 + 105*x**3*y**4 + 105*x*y**6,
    lambda x,y : x**8 + 28*x**6*y**2 + 210*x**4*y**4 + 420*x**2*y**6 + 105*y**8
]

def stability_function(x, y, x_scale, y_scale, coeffs):
    return abs(sum(c * m(x * x_scale, y * y_scale) for c, m in zip(coeffs, MOMENTS[:len(coeffs)])))

def ees25_(x, y, x_scale, y_scale):
    # coefficients of ( 1 + x + (1/2)x^2 + (1/8)x^3 )^2
    coeffs = [1., 2., 2., 5. / 4, 1. / 2, 1. / 8, 1. / 64]
    return stability_function(x, y, x_scale, y_scale, coeffs)

def rk4_(x, y, x_scale, y_scale):
    # coefficients of ( 1 + x + (1/2)x^2 + (1/6)x^3 + (1/24)x^4 )^2
    coeffs = [1., 2., 2., 4. / 3, 2. / 3, 1. / 4, 5. / 72, 1. / 72, 1. / 576]
    return stability_function(x, y, x_scale, y_scale, coeffs)

ees25 = np.vectorize(ees25_)
rk4 = np.vectorize(rk4_)

REV_HEUN_COLOUR = 'firebrick'
REV_HEUN_LW = 3.0


def plot_reversible_heun_intersection(ax, x_scale, xlim):
    """Draw the Reversible Heun ODE stability set in this cross-section."""
    # The published stability set is lambda dt in [-i, i] with no diffusion.
    # In these slices this gives a segment only for lambda dt = alpha i; the
    # other panels intersect it only at the origin.
    ax.plot([0.0], [0.0], marker='o', color=REV_HEUN_COLOUR,
            markersize=REV_HEUN_LW + 3, zorder=10)
    if x_scale != 1j:
        return

    left = max(xlim[0], -1.0)
    right = min(xlim[1], 1.0)
    if left <= right:
        ax.plot([left, right], [0.0, 0.0], color=REV_HEUN_COLOUR,
                linewidth=REV_HEUN_LW, solid_capstyle='round', zorder=10)


if __name__ == "__main__":

    resolution=500

    x_scale = [1, 1j, 1j+1, 1j+1]
    y_scale = [1, 1j, 1j+1, 1j-1]
    xlim = [(-3.5, 1), (-3.5, 3.5), (-2.5, 1), (-3.5, 1)]
    ylim = [(-1.5, 1.5), (-2.5, 2.5), (-1.5, 1.5), (-1, 1)]
    ax_titles = [
    r"$(\lambda dt, \mu^2 dt) = (\alpha, \beta)$",
    r"$(\lambda dt, \mu^2 dt) = (\alpha i, \beta i)$",
    r"$(\lambda dt, \mu^2 dt) = (\alpha (1 + i), \beta (1 + i))$",
    r"$(\lambda dt, \mu^2 dt) = (\alpha (1 + i), \beta (1 - i))$"
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22.5, 6.75))

    for ax, x_s, y_s, xlm, ylm, title_ in zip(axes, x_scale, y_scale, xlim, ylim, ax_titles):
        print("Plotting", title_)

        x = np.linspace(xlm[0], xlm[1], resolution)
        y = np.linspace(ylm[0], ylm[1], resolution)
        X, Y = np.meshgrid(x, y)

        plot_reversible_heun_intersection(ax, x_s, xlm)

        R = ees25(X, Y, x_s, y_s)
        ax.contour(X, Y, R, levels=[1], colors='black', linewidths=3)

        R = rk4(X, Y, x_s, y_s)
        ax.contour(X, Y, R, levels=[1], colors='black', linewidths=3, linestyles = ":")

        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$", rotation='horizontal')
        ax.set_title(title_, pad = 25)

    # proxy artists for legend
    line0 = mlines.Line2D([], [], color=REV_HEUN_COLOUR, linewidth=REV_HEUN_LW,
                          label=r"$\mathrm{Reversible\ Heun}$")
    line1 = mlines.Line2D([], [], color='black', linewidth=3, label=r"$\mathrm{EES}(2,5)$")
    line3 = mlines.Line2D([], [], color='black', linewidth=3, linestyle=":", label=r"$\mathrm{RK4}$")

    axes[1].legend(handles = [line0, line1, line3], loc='upper center',
                 bbox_to_anchor=(1.15, -0.2),fancybox=False, shadow=False, ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, wspace=0.25)
    import os as _os
    _results_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results")
    _os.makedirs(_results_dir, exist_ok=True)
    plt.savefig(_os.path.join(_results_dir, "stoch_stability_reversible_heun.pdf"))
    plt.savefig(_os.path.join(_results_dir, "stoch_stability_reversible_heun.eps"))
    plt.savefig(_os.path.join(_results_dir, "stoch_stability_reversible_heun.png"))
    plt.show()
