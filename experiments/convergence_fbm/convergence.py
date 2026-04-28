"""
This file contains code reproducing the results found in the paper
"Runge-Kutta methods for rough differential equations" (Redmann & Riedel, 2020).
The authors apply Heun's method to the RDE

dy(t) = cos(y(t)) dX^1(t) + sin(y(t))dX^2(t), y(0) = 1, t in [0,T].

where X is a two-dimensional fractional Brownian motion
with independent components and Hurst index H. The error

E(h) := max_{k = 0, ..., N} |y(t_k) - y_k^h|

is then evaluated for different choices of the step size h, and
the trends are compared against the expected order of convergence, 2H - 0.5.
For more details, see Redmann & Riedel, 2020.

We apply this example to test the convergence rates of EES(2,5). In addition
to the discretisation error, we evaluate the error in recovering the initial
condition, y_0.
"""

import os
from math import sqrt
from fbm import FBM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mkr

from ees_core.stoch_rk_methods import EES25, EES27
from ees_core import plotting_params

np.random.seed(0)

plotting_params.set_plotting_params(9, 10, 12)

def get_2d_fbm(n, H, length):
    f_ = FBM(n=n, hurst=H, length=length, method='daviesharte')
    X1 = f_.fbm()
    X2 = f_.fbm()
    return np.array([[a, b] for a, b in zip(X1, X2)])

def get_error(y_exact, y_vals, T):
    t = np.linspace(0, T, len(y_vals))
    t_exact = np.linspace(0, T, len(y_exact))
    true_vals = np.interp(t, t_exact, y_exact)
    return np.max(np.abs(true_vals - y_vals))

def plot(f, method, H, T, rate, ax, N = 10, backward = False):
    n = int(T * 2 ** 16)
    h = [2 ** (-i) for i in range(4, 14)]
    y = np.zeros(len(h))
    for i in range(N):
        X = get_2d_fbm(n, H=H, length=T)
        y_exact = method.run([1], f, X)
        y_exact = np.array(y_exact).flatten()

        error = []
        for h_ in h:
            step = int(n * h_)
            if not backward:
                y_vals = method.run([1], f, X[::step])
                y_vals = np.array(y_vals).flatten()
                error.append(get_error(y_exact, y_vals, T))
            else:
                y_vals = method.run([1], f, X[::step])
                y_vals = method.run(y_vals[-1], f, X[::step][::-1])
                error.append(abs(y_exact[0] - y_vals[-1][0]))

        y += np.log10(error)

    y /= N
    x = np.log10(h)
    dx = np.array([x[0], x[-1]])

    err_label = r'$\log_{10}(\mathcal{E}(h))$' if not backward else r'$\log_{10}(\overleftarrow{\mathcal{E}}(h))$'

    intercept = np.mean(y) - rate * np.mean(x)
    print("Error intercept: ", intercept)
    ax.scatter(x, y, marker=mkr.MarkerStyle('x', fillstyle='none'), color='crimson')
    ax.plot(dx, rate * dx + intercept, color='mediumblue')
    ax.legend([err_label, str(np.round(rate,1)) + r'$x + c$'])
    ax.set_xlabel(r'$\log_{10}(h)$')
    ax.set_ylabel(err_label)
    ax.set_title(r'$H = $' + str(np.round(H,1)))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_grid(f_, T_, H_, rates_, methods_, titles_):
    _, ax = plt.subplots(2, 2, figsize=(10 * (2/3), 10 * (2/3)))
    for i in range(2):
        for j in range(2):
            plot(f_, methods_[i], H_, T_, rates_[i*2 + j](H_), ax[i][j], backward = bool(j))
            ax[i][j].set_title(titles_[i * 2 + j])

    plt.tight_layout()
    out_stem = os.path.join(RESULTS_DIR, f"ees_stochastic_convergence_H{int(H_ * 100):02d}")
    plt.savefig(out_stem + ".png")
    plt.savefig(out_stem + ".pdf")


if __name__ == "__main__":
    EES27_param = (5 - 3 * sqrt(2)) / 14
    methods = [EES25, EES27]
    titles = [
        r'$\mathcal{E}(h)$ for $\mathrm{EES}_\mathcal{R}(2,5)$',
        r'$\overleftarrow{\mathcal{E}}(h)$ for $\mathrm{EES}_\mathcal{R}(2,5)$',
        r'$\mathcal{E}(h)$ for $\mathrm{EES}_\mathcal{R}(2,7)$',
        r'$\overleftarrow{\mathcal{E}}(h)$ for $\mathrm{EES}_\mathcal{R}(2,7)$'
    ]
    rates = [
        lambda x : 2 * x - 0.5,
        lambda x: 6 * x - 1,
        lambda x: 2 * x - 0.5,
        lambda x: 8 * x - 1
    ]

    def f(y):
        return [np.cos(y[0]), np.sin(y[0])]

    T = 1.

    plot_grid(f, T, 0.4, rates, methods, titles)
    plot_grid(f, T, 0.5, rates, methods, titles)
    plot_grid(f, T, 0.6, rates, methods, titles)
