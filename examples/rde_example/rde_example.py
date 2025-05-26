from fbm import FBM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mkr

import sys
sys.path.append('../../../NeuralSDEs')

from stoch_rk import StochRK
import plotting_params
plotting_params.set_plotting_params(9, 10, 12)

def get_2d_fbm(n, H = 0.75, length = 0.25):
    f = FBM(n=n, hurst=H, length=length, method='daviesharte')
    X1 = f.fbm()
    X2 = f.fbm()
    return np.array([[a, b] for a, b in zip(X1, X2)])

def get_error(y_exact, y_vals):
    T = 0.25
    t = np.linspace(0, 0.25, len(y_vals))
    t_exact = np.linspace(0, 0.25, len(y_exact))
    true_vals = np.interp(t, t_exact, y_exact)
    return np.max(np.abs(true_vals - y_vals))

def plot(f, method, H, ax):
    rate = 2 * H - 0.5
    T = 0.25
    n = int(T * 2 ** 16)
    X = get_2d_fbm(n, H=H)
    y_exact = method.run([1], f, X)
    y_exact = np.array(y_exact).flatten()

    h = [2 ** (-i) for i in range(4, 14)]
    error = []
    for h_ in h:
        step = int(n * h_)
        y_vals = method.run([1], f, X[::step])
        y_vals = np.array(y_vals).flatten()
        error.append(get_error(y_exact, y_vals))

    x = np.log10(h)
    y = np.log10(error)
    dx = np.array([x[0], x[-1]])

    intercept = np.mean(y) - rate * np.mean(x)
    ax.scatter(x, y, marker=mkr.MarkerStyle('o', fillstyle='none'), color='crimson')
    ax.plot(dx, rate * dx + intercept, color='mediumblue')
    ax.legend([r'$\log_{10}(\mathcal{E}(h))$', str(np.round(rate,1)) + r'$x + c$'])
    ax.set_xlabel(r'$\log_{10}(h)$')
    ax.set_ylabel(r'$\log_{10}(\mathcal{E}(h))$')
    ax.set_title(r'$H = $' + str(np.round(H,1)))

if __name__ == "__main__":
    H = [0.4, 0.5, 0.7]

    heun = StochRK([[0,0,0],
                    [1/3,0,0],
                    [0,2/3,0]],
                   [1/4,0,3/4], "Heun")

    def f(y):
        return [np.cos(y[0]), np.sin(y[0])]


    fig, ax = plt.subplots(3, 3, figsize = (10,10))
    for i in range(3):
        for j in range(3):
            plot(f, heun, H[i], ax[i][j])

    plt.tight_layout()
    plt.savefig("rde_example.png")
    plt.savefig("rde_example.pdf")