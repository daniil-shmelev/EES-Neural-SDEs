"""
Plot the training loss after running GBM.py
"""

import pickle
import matplotlib.pyplot as plt
from utils import set_plotting_params

set_plotting_params(11, 12, 13)

with open('plots/options_ees25/mse.pickle', 'rb') as handle:
    ees25 = pickle.load(handle)

with open('plots/options_reversible_heun/mse.pickle', 'rb') as handle:
    reversible_heun = pickle.load(handle)

plt.figure(figsize=(4,3))
plt.plot(ees25, label = "EES(2,5)", color = "red")
plt.plot(reversible_heun, label = "Reversible Heun", color="blue", linestyle = "--")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig("GBM_mse.pdf")
plt.show()
plt.clf()
