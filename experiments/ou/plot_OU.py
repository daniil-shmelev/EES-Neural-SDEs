"""
Plot the training loss after running OU.py
"""

import os
import pickle
import matplotlib.pyplot as plt
from experiments.plotting import set_plotting_params

set_plotting_params(11, 12, 13)

METHODS = {
    "reversible_heun": {
        "label": "Reversible Heun",
        "color": "blue",
        "linestyle": "--",
        "zorder": 2,
    },
    "ees25": {
        "label": "EES(2,5)",
        "color": "red",
        "linestyle": "-",
        "zorder": 1,
    },
    "ees27": {
        "label": "EES(2,7)",
        "color": "green",
        "linestyle": "-",
        "zorder": 0,
    },
}

def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def load_results():
    results = {}
    for method in METHODS:
        mse_path = f'plots/{method}/mse.pickle'
        grad_path = f'plots/{method}/grad_error.pickle'
        results[method] = {}
        if os.path.isfile(mse_path):
            results[method]['mse'] = load_pickle(mse_path)
        if os.path.isfile(grad_path):
            grad_error = load_pickle(grad_path)
            if isinstance(grad_error, dict):
                results[method]['grad_epochs'] = grad_error['epochs']
                results[method]['grad_error'] = grad_error['values']
            else:
                results[method]['grad_epochs'] = list(range(1, len(grad_error) + 1))
                results[method]['grad_error'] = grad_error
    return results

def plot_mse(results):
    plt.figure(figsize=(4, 3))
    for method, style in METHODS.items():
        if 'mse' not in results[method]:
            continue
        plt.plot(
            results[method]['mse'],
            label=style['label'],
            color=style['color'],
            linestyle=style['linestyle'],
            zorder=style['zorder'],
        )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("OU_mse.pdf")
    plt.clf()

def plot_mse_and_grad_error(results):
    fig, ax_mse = plt.subplots(figsize=(4.4, 3.3))
    ax_grad = ax_mse.twinx()

    handles = []
    labels = []
    for method, style in METHODS.items():
        if 'mse' in results[method]:
            line, = ax_mse.plot(
                results[method]['mse'],
                label=style['label'],
                color=style['color'],
                linestyle=style['linestyle'],
                zorder=style['zorder'],
            )
            handles.append(line)
            labels.append(style['label'])
        if 'grad_error' in results[method]:
            line, = ax_grad.plot(
                results[method]['grad_epochs'],
                results[method]['grad_error'],
                label=style['label'] + " grad",
                color=style['color'],
                linestyle=":",
                zorder=style['zorder'],
            )
            handles.append(line)
            labels.append(style['label'] + " grad")

    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel("MSE")
    ax_grad.set_ylabel("Relative Gradient Error")
    ax_mse.set_yscale("log")
    ax_grad.set_yscale("log")
    ax_mse.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig("OU_mse_grads.pdf")
    plt.close(fig)

if __name__ == "__main__":
    results = load_results()
    plot_mse(results)
    plot_mse_and_grad_error(results)
