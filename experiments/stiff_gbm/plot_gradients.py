"""
Plot the training loss after running GBM.py
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from experiments.plotting import set_plotting_params

set_plotting_params(11, 12, 13)

if __name__ == "__main__":
    plt.figure(figsize=(4*1.1, 3*1.1))

    if os.path.isfile('experiments/stiff_gbm/results/reversible_heun/grad_err.pickle'):
        with open('experiments/stiff_gbm/results/reversible_heun/grad_err.pickle', 'rb') as handle:
            reversible_heun = pickle.load(handle)
        print(np.mean(reversible_heun))
        plt.plot(reversible_heun, label = "Reversible Heun", color="blue", linestyle = "--", zorder = 4)

    if os.path.isfile('experiments/stiff_gbm/results/mcf_euler/grad_err.pickle'):
        with open('experiments/stiff_gbm/results/mcf_euler/grad_err.pickle', 'rb') as handle:
            mcf_euler = pickle.load(handle)
        print(np.mean(mcf_euler))
        plt.plot(mcf_euler, label = "MCF Euler", color="magenta", linestyle = "--", zorder = 3)

    if os.path.isfile('experiments/stiff_gbm/results/mcf_midpoint/grad_err.pickle'):
        with open('experiments/stiff_gbm/results/mcf_midpoint/grad_err.pickle', 'rb') as handle:
            mcf_midpoint = pickle.load(handle)
        print(np.mean(mcf_midpoint))
        plt.plot(mcf_midpoint, label = "MCF Midpoint", color="purple", linestyle = "--", zorder = 2)

    if os.path.isfile('experiments/stiff_gbm/results/ees25/grad_err.pickle'):
        with open('experiments/stiff_gbm/results/ees25/grad_err.pickle', 'rb') as handle:
            ees25 = pickle.load(handle)
        print(np.mean(ees25))
        plt.plot(ees25, label="EES(2,5)", color="red", zorder = 1)

    # if os.path.isfile('experiments/stiff_gbm/results/ees27/mse.pickle'):
    #     with open('experiments/stiff_gbm/results/ees27/mse.pickle', 'rb') as handle:
    #         ees27 = pickle.load(handle)
    #     print(np.mean(ees27))
    #     plt.plot(ees27, label="EES(2,7)", color="green", zorder = 0)

    ees25 = np.array(ees25)

    plt.xlabel("Epoch")
    plt.ylabel("Gradient MSE")
    plt.ylim(ees25.min() * 0.5, ees25.max() * 2)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("GBM_mse_2_grads.pdf")
    plt.show()
    plt.clf()
