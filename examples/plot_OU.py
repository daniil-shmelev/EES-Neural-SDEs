"""
Plot the training loss after running OU.py
"""

import pickle
import matplotlib.pyplot as plt
from utils import set_plotting_params

set_plotting_params(11, 12, 13)

if __name__ == "__main__":

    with open('plots/ees25/mse.pickle', 'rb') as handle:
        ees25 = pickle.load(handle)

    with open('plots/reversible_heun/mse.pickle', 'rb') as handle:
        reversible_heun = pickle.load(handle)

    plt.figure(figsize=(4,3))
    plt.plot(ees25, label = "EES(2,5)", color = "red")
    plt.plot(reversible_heun, label = "Reversible Heun", color="blue", linestyle = "--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("OU_mse.pdf")
    plt.show()
    plt.clf()
