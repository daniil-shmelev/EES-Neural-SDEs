from math import sqrt
import kauri as kr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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

# moments of N(x,y**2)
moments = [
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

resolution=500

title = "Mean-Square Stability Regions"

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

fig, axes = plt.subplots(1, 4, figsize=(15 * 1.5, 4.5 * 1.5))

for ax, x_s, y_s, xlm, ylm, title_ in zip(axes, x_scale, y_scale, xlim, ylim, ax_titles):

    x = np.linspace(xlm[0], xlm[1], resolution)
    y = np.linspace(ylm[0], ylm[1], resolution)
    X, Y = np.meshgrid(x, y)

    def ees25(x, y):
        # treat (x, y) as two real variables, not as a single complex number
        coeffs = [1., 2., 2., 5./4, 1./2, 1./8, 1./64]
        return abs(sum(c * m(x * x_s, y * y_s) for c,m in zip(coeffs, moments[:len(coeffs)])))

    def rk3(x, y):
        # treat (x, y) as two real variables, not as a single complex number
        coeffs = [1., 2., 2., 4./3, 7./12, 1./6, 1./36]
        return abs(sum(c * m(x * x_s, y * y_s) for c,m in zip(coeffs, moments[:len(coeffs)])))

    def rk4(x, y):
        # treat (x, y) as two real variables, not as a single complex number
        coeffs = [1., 2., 2., 4./3, 2./3, 1./4, 5./72, 1./72, 1./576]
        return abs(sum(c * m(x * x_s, y * y_s) for c,m in zip(coeffs, moments[:len(coeffs)])))


    # evaluate on grid
    R = np.vectorize(ees25)(X, Y)
    ax.contour(X, Y, R, levels=[1], colors='black', linewidths=3, label = "EES(2,5)")

    R = np.vectorize(rk3)(X, Y)
    ax.contour(X, Y, R, levels=[1], colors='black', linewidths=3, linestyles = "--", label = "RK3")

    R = np.vectorize(rk4)(X, Y)
    ax.contour(X, Y, R, levels=[1], colors='black', linewidths=3, linestyles = ":", label = "RK4")

    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$", rotation='horizontal')
    ax.set_title(title_, pad = 25)

# proxy artists for legend
line1 = mlines.Line2D([], [], color='black', linewidth=3, label="EES(2,5)")
line2 = mlines.Line2D([], [], color='black', linewidth=3, linestyle="--", label="RK3")
line3 = mlines.Line2D([], [], color='black', linewidth=3, linestyle=":", label="RK4")

axes[1].legend(handles = [line1, line2, line3], loc='upper center',
             bbox_to_anchor=(1.15, -0.2),fancybox=False, shadow=False, ncol=3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25, wspace=0.25)
#plt.suptitle(title)
plt.savefig("stoch_stability.pdf")
plt.savefig("stoch_stability.eps")
plt.savefig("stoch_stability.png")
plt.show()
