"""Paper-quality visualization for covariance forecasting results."""

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


def plot_riemannian_distance(
    distances: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Per-day Riemannian distance for all models."""
    fig, ax = plt.subplots()
    for name, d in distances.items():
        d = np.asarray(d)
        # Rolling average for readability
        window = min(20, len(d) // 5)
        if window > 1:
            smooth = np.convolve(d, np.ones(window) / window, mode="valid")
            ax.plot(smooth, label=name, linewidth=1.0)
        else:
            ax.plot(d, label=name, linewidth=1.0)

    ax.set_xlabel("Trading Day")
    ax.set_ylabel("AIRM Distance")
    ax.set_title("Riemannian Distance: Predicted vs True Covariance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)


def plot_eigenvalue_spectrum(
    predicted: np.ndarray,
    actual: np.ndarray,
    save_path: Path,
    model_name: str = "Model",
) -> None:
    """Compare eigenvalue distributions of predicted vs actual covariances."""
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    eigs_pred = np.linalg.eigvalsh(predicted).flatten()
    eigs_actual = np.linalg.eigvalsh(actual).flatten()

    fig, ax = plt.subplots()
    ax.hist(eigs_actual, bins=50, alpha=0.6, label="Actual", density=True)
    ax.hist(eigs_pred, bins=50, alpha=0.6, label=model_name, density=True)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_title("Eigenvalue Spectrum: Predicted vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)


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
