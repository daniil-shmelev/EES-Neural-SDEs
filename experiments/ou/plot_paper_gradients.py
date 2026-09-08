"""
Plot the training loss after running OU.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from experiments.plotting import set_plotting_params

set_plotting_params(11, 12, 13)

if __name__ == "__main__":
    plt.figure(figsize=(4*1.1, 3*1.1))

    if os.path.isfile('experiments/ou/results/paper/reversible_heun/grad_err.pickle'):
        with open('experiments/ou/results/paper/reversible_heun/grad_err.pickle', 'rb') as handle:
            reversible_heun = pickle.load(handle)
        print(np.mean(reversible_heun))
        plt.plot(reversible_heun, label = "Reversible Heun", color="blue", linestyle = "--", zorder = 4)

    if os.path.isfile('experiments/ou/results/paper/mcf_euler/grad_err.pickle'):
        with open('experiments/ou/results/paper/mcf_euler/grad_err.pickle', 'rb') as handle:
            mcf_euler = pickle.load(handle)
        print(np.mean(mcf_euler))
        plt.plot(mcf_euler, label = "MCF Euler", color="magenta", linestyle = "--", zorder = 3)

    if os.path.isfile('experiments/ou/results/paper/mcf_midpoint/grad_err.pickle'):
        with open('experiments/ou/results/paper/mcf_midpoint/grad_err.pickle', 'rb') as handle:
            mcf_midpoint = pickle.load(handle)
        print(np.mean(mcf_midpoint))
        plt.plot(mcf_midpoint, label = "MCF Midpoint", color="purple", linestyle = "--", zorder = 2)

    if os.path.isfile('experiments/ou/results/paper/ees25/grad_err.pickle'):
        with open('experiments/ou/results/paper/ees25/grad_err.pickle', 'rb') as handle:
            ees25 = pickle.load(handle)
        print(np.mean(ees25))
        plt.plot(ees25, label="EES(2,5)", color="red", zorder = 1)

    # if os.path.isfile('experiments/ou/results/paper/ees27/grad_err.pickle'):
    #     with open('experiments/ou/results/paper/ees27/grad_err.pickle', 'rb') as handle:
    #         ees27 = pickle.load(handle)
    #     plt.plot(ees27, label="EES(2,7)", color="green", zorder = 0)

    plt.xlabel("Epoch")
    plt.ylabel("Gradient MSE")
    plt.legend(loc="lower left")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("OU_mse_grads.pdf")
    plt.savefig("OU_mse_grads.png", dpi=600)
    plt.show()
    plt.clf()
