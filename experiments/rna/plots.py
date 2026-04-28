"""Paper-quality visualization for RNA torus training results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_training_curves(
    histories: dict[str, dict],
    save_path: Path,
) -> None:
    """Training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, hist in histories.items():
        train_values = hist.get("train_loss", hist.get("train"))
        val_values = hist.get("val_riemannian_dist", hist.get("val_loss", hist.get("val")))
        if train_values:
            axes[0].plot(np.asarray(train_values), label=name, linewidth=1.2)
        if val_values:
            axes[1].plot(np.asarray(val_values), label=name, linewidth=1.2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Metric")
    axes[1].set_title("Validation Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
