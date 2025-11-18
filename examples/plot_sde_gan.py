"""
Plot the training loss after running GBM.py
"""

import os
import pickle
import matplotlib.pyplot as plt
from utils import set_plotting_params

set_plotting_params(11, 12, 13)

if __name__ == "__main__":
    plt.figure(figsize=(4*1.1, 3*1.1))

    if os.path.isfile('./plots/sde_gan_reversible_heun/sde_gan_loss.pickle'):
        with open('./plots/sde_gan_reversible_heun/sde_gan_loss.pickle', 'rb') as handle:
            reversible_heun = pickle.load(handle)
        plt.plot(reversible_heun, label = "Reversible Heun", color="blue", linestyle = "--", zorder = 4)

    if os.path.isfile('./plots/sde_gan_mcf_euler/sde_gan_loss.pickle'):
        with open('./plots/sde_gan_mcf_euler/sde_gan_loss.pickle', 'rb') as handle:
            mcf_euler = pickle.load(handle)
        print(mcf_euler[-1])
        plt.plot(mcf_euler, label = "MCF Euler", color="magenta", linestyle = "--", zorder = 3)

    if os.path.isfile('./plots/sde_gan_mcf_midpoint/sde_gan_loss.pickle'):
        with open('./plots/sde_gan_mcf_midpoint/sde_gan_loss.pickle', 'rb') as handle:
            mcf_midpoint = pickle.load(handle)
        print(mcf_midpoint[-1])
        plt.plot(mcf_midpoint, label = "MCF Midpoint", color="purple", linestyle = "--", zorder = 2)

    if os.path.isfile('./plots/sde_gan_ees25/sde_gan_loss.pickle'):
        with open('./plots/sde_gan_ees25/sde_gan_loss.pickle', 'rb') as handle:
            ees25 = pickle.load(handle)
        plt.plot(ees25, label="EES(2,5)", color="red", zorder = 1)

    if os.path.isfile('./plots/sde_gan_ees27/sde_gan_loss.pickle'):
        with open('./plots/sde_gan_ees27/sde_gan_loss.pickle', 'rb') as handle:
            ees27 = pickle.load(handle)
        plt.plot(ees27, label="EES(2,7)", color="green", zorder = 0)

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    #plt.yscale("log")
    plt.tight_layout()
    plt.savefig("sde_gan_mse.pdf")
    plt.show()
    plt.clf()

