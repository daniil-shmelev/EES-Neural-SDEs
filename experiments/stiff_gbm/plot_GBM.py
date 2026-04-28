"""
Plot the training loss after running GBM.py
"""

import os
import pickle
import matplotlib.pyplot as plt
from ees_core import set_plotting_params

set_plotting_params(11, 12, 13)

if __name__ == "__main__":
    plt.figure(figsize=(4, 3))

    if os.path.isfile('plots/options_reversible_heun/mse.pickle'):
        with open('plots/options_reversible_heun/mse.pickle', 'rb') as handle:
            reversible_heun = pickle.load(handle)
        plt.plot(reversible_heun, label = "Reversible Heun", color="blue", linestyle = "--", zorder = 2)

    if os.path.isfile('plots/options_ees25/mse.pickle'):
        with open('plots/options_ees25/mse.pickle', 'rb') as handle:
            ees25 = pickle.load(handle)
        plt.plot(ees25, label="EES(2,5)", color="red", zorder = 1)

    if os.path.isfile('plots/options_ees27/mse.pickle'):
        with open('plots/options_ees27/mse.pickle', 'rb') as handle:
            ees27 = pickle.load(handle)
        plt.plot(ees27, label="EES(2,7)", color="green", zorder = 0)

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("GBM_mse.pdf")
    plt.show()
    plt.clf()

