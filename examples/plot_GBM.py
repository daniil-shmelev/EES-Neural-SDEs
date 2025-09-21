import pickle
import matplotlib.pyplot as plt

def set_plotting_params(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
