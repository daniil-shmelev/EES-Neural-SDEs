from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def plot_training_curves(
    histories: dict[str, dict],
    save_path: Path,
) -> None:
    """Training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, hist in histories.items():
        train_values = hist.get("train_loss", hist.get("train"))
        val_values = hist.get("val_loss", hist.get("val"))
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


def plot_sample_trajectories(
    predicted: np.ndarray,
    actual: np.ndarray,
    save_path: Path,
    *,
    mask: np.ndarray | None = None,
    max_examples: int = 8,
) -> None:
    """Overlay a batch of predicted and ground-truth trajectories."""
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)

    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        predicted = predicted[valid]
        actual = actual[valid]

    if predicted.size == 0 or actual.size == 0:
        return

    n_examples = min(int(predicted.shape[0]), max_examples)
    n_dims = int(predicted.shape[-1])
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), sharex=True)
    axes = np.atleast_1d(axes)
    time_axis = np.arange(predicted.shape[1])

    for dim, axis in enumerate(axes):
        for idx in range(n_examples):
            axis.plot(
                time_axis,
                actual[idx, :, dim],
                color="red",
                linewidth=1.2,
                alpha=0.8,
                label="ground truth" if idx == 0 else None,
            )
            axis.plot(
                time_axis,
                predicted[idx, :, dim],
                color="black",
                linewidth=1.0,
                alpha=0.8,
                label="predicted" if idx == 0 else None,
            )
        axis.set_ylabel(f"dim {dim}")
        axis.grid(True, alpha=0.3)

    axes[0].set_title("Latest Epoch Trajectories")
    axes[0].legend()
    axes[-1].set_xlabel("Time step")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
