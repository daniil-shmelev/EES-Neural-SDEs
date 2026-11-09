"""Visualise a saved activity classifier on one observed sequence.

Run from the repository root with ``python -m experiments.sphere_latent_nsde.visualize RUN_DIR``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from experiments.sphere_latent_nsde.config import load_config
from experiments.sphere_latent_nsde.dataset import TAG_IDS
from experiments.sphere_latent_nsde.factories import make_dataset, make_model

# These are the seven merged classes in dataset.LABEL_DICT, in index order.
CLASS_NAMES = (
    "Walking",
    "Falling",
    "Lying / lying down",
    "Sitting / sitting down",
    "Standing up",
    "On all fours",
    "Sitting on the ground",
)
CLASS_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#777777",
)
TAG_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _time_edges(times: np.ndarray) -> np.ndarray:
    """Observation-centred bins preserve the irregular temporal spacing."""
    midpoints = (times[1:] + times[:-1]) / 2
    return np.r_[times[0], midpoints, times[-1]]


def render_example(arrays: dict, metadata: dict, output_dir: Path) -> None:
    times = arrays["times"]
    observed_times = arrays["observed_times"]
    obs = arrays["obs"]
    mask = arrays["mask"].astype(bool)
    labels = arrays["labels"]
    predictions = arrays["predictions"]
    paths = arrays["paths"]
    reconstruction = arrays["reconstruction"].mean(axis=0)
    probabilities = arrays["probabilities"]
    # Fit one common PCA basis across samples and times in the observed window.
    visible = times <= observed_times[-1]
    flat = paths[:, visible].reshape(-1, paths.shape[-1])
    centre = flat.mean(axis=0)
    _, singular_values, basis = np.linalg.svd(flat - centre, full_matrices=False)
    projected = (paths - centre) @ basis[:3].T
    projected = np.pad(projected, ((0, 0), (0, 0), (0, 3 - projected.shape[-1])))
    variance = np.square(singular_values)
    explained = float(variance[:3].sum() / variance.sum()) if variance.sum() else 0.0
    metadata["pca_explained_variance_ratio"] = explained

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(15, 10), layout="constrained")
    grid = fig.add_gridspec(
        6, 2, width_ratios=(1.55, 1), height_ratios=(1, 1, 1, 0.45, 0.45, 1.6)
    )
    fig.suptitle(
        "Human activity from irregular sensor observations",
        fontsize=19,
        fontweight="bold",
    )
    sensor_axes = []
    for coordinate, name in enumerate(("x", "y", "z")):
        ax = fig.add_subplot(
            grid[coordinate, 0], sharex=sensor_axes[0] if sensor_axes else None
        )
        sensor_axes.append(ax)
        for tag, color in enumerate(TAG_COLORS):
            channel = 3 * tag + coordinate
            valid = mask[:, channel]
            ax.plot(times, reconstruction[:, channel], color=color, lw=1.5)
            ax.scatter(
                observed_times[valid],
                obs[valid, channel],
                color=color,
                s=12,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )
        ax.set_ylabel(f"Sensor {name}")
        ax.grid(alpha=0.15)
        ax.tick_params(labelbottom=False)
    sensor_axes[0].set_title(
        "Observed positions (dots) and reconstructed means (lines)",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    sensor_axes[0].legend(
        [Line2D([], [], color=c, marker="o", markersize=4) for c in TAG_COLORS],
        [f"Tag {i + 1}" for i in range(len(TAG_IDS))],
        ncol=4,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(8) - 0.5, cmap.N)
    edges = _time_edges(observed_times)
    for row, values, title in (
        (3, labels, "True activity"),
        (4, predictions, "Predicted"),
    ):
        ax = fig.add_subplot(grid[row, 0], sharex=sensor_axes[0])
        ax.pcolormesh(edges, [0, 1], values[None], cmap=cmap, norm=norm, shading="flat")
        ax.set_yticks([0.5], [title])
        ax.tick_params(axis="both", length=0, labelbottom=False)
        ax.spines[:].set_visible(False)

    prob_ax = fig.add_subplot(grid[5, 0], sharex=sensor_axes[0])
    for label, color in enumerate(CLASS_COLORS):
        prob_ax.plot(observed_times, probabilities[:, label], color=color, lw=1.7)
    prob_ax.set(
        ylim=(-0.03, 1.03),
        ylabel="Class probability",
        xlabel="Normalised time within sequence",
        xlim=(observed_times[0], observed_times[-1]),
    )
    prob_ax.grid(alpha=0.15)

    latent_ax = fig.add_subplot(grid[:3, 1], projection="3d")
    for path in projected:
        latent_ax.plot(*path[visible].T, color="#808C99", alpha=0.3, lw=0.9)
    first_path = projected[0, arrays["time_ids"]]
    latent_ax.plot(*projected[0, visible].T, color="#283746", lw=1.6)
    latent_ax.scatter(
        *first_path.T, c=labels, cmap=cmap, norm=norm, s=18, depthshade=False
    )
    latent_ax.scatter(*projected[0, 0], marker="*", color="black", s=90)
    latent_ax.set(xlabel="PC 1", ylabel="PC 2", zlabel="PC 3")
    latent_ax.set_title(
        f"Latent paths on $S^{{{paths.shape[-1] - 1}}}$, projected to 3D\n"
        f"PCA retains {explained:.1%} of this window's variance",
        fontsize=11,
        fontweight="bold",
    )
    latent_ax.view_init(elev=23, azim=35)

    info_ax = fig.add_subplot(grid[3:, 1])
    info_ax.axis("off")
    info_ax.legend(
        [Patch(color=c) for c in CLASS_COLORS],
        CLASS_NAMES,
        loc="upper left",
        frameon=False,
        fontsize=10,
    )
    info_ax.text(
        0.02,
        0.36,
        f"{metadata['split'].capitalize()} window {metadata['sample_index']}  |  "
        f"{metadata['mc_samples']} posterior samples\n"
        f"Accuracy on this window: {metadata['sample_accuracy']:.1%}\n"
        f"Maximum sphere norm error: {metadata['max_sphere_norm_error']:.1e}\n\n"
        "Grey curves: sampled latent paths; star: initial state.\n"
        "Coloured dots: true activities on one sampled path.\n"
        "PCA does not preserve spherical geometry.\n"
        "Inference uses the complete observed sequence.",
        transform=info_ax.transAxes,
        va="top",
        fontsize=9,
        linespacing=1.5,
    )
    fig.supxlabel(
        f"Checkpoint: {metadata['run_dir'].name}", fontsize=8, color="#555555"
    )
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"activity_example.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run with config.toml and nsde.eqx")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Index within the split; default: first activity transition",
    )
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mc_samples < 1:
        parser.error("--mc-samples must be positive")
    jax.config.update("jax_platform_name", args.device)
    config = load_config(args.run_dir / "config.toml")
    dataset = make_dataset(config, args.split)
    sample_index = args.sample_index
    if sample_index is None:
        labels = np.asarray(dataset.as_array_dict()["aux_obs"])
        transitions = np.flatnonzero(np.any(np.diff(labels, axis=1) != 0, axis=1))
        sample_index = int(transitions[0]) if len(transitions) else 0
    if not 0 <= sample_index < len(dataset):
        parser.error(f"--sample-index must be between 0 and {len(dataset) - 1}")
    print(
        f"Loading checkpoint; plotting {args.split} window {sample_index}", flush=True
    )
    model = eqx.tree_deserialise_leaves(
        args.run_dir / "nsde.eqx", make_model(config, jax.random.key(config.seed))
    )
    example = dataset[sample_index]
    batch = {name: value[None] for name, value in example.items()}

    @eqx.filter_jit
    def predict(model, batch, key):
        output = model(batch, key, mc_samples=args.mc_samples)
        return output.paths, output.recon_mu, output.aux_logits, output.solver_stats

    paths, reconstruction, logits, solver_stats = predict(
        model, batch, jax.random.key(args.seed)
    )
    time_ids = np.asarray(example["aux_tid"])
    # Match losses.auxiliary_accuracy: average logits, then classify at aux_tid.
    logits = np.asarray(logits).mean(axis=0)[0, time_ids]
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    paths = np.asarray(paths)[:, 0]
    arrays = {
        "times": np.asarray(model.desired_times()),
        "observed_times": np.asarray(example["inp_tps"]),
        "time_ids": time_ids,
        "obs": np.asarray(example["inp_obs"]),
        "mask": np.asarray(example["inp_msk"]),
        "labels": np.asarray(example["aux_obs"]),
        "predictions": probabilities.argmax(axis=-1),
        "probabilities": probabilities,
        "paths": paths,
        "reconstruction": np.asarray(reconstruction)[:, 0],
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite observations or model outputs; cannot plot.")
    metadata = {
        "run_dir": args.run_dir.resolve(),
        "split": args.split,
        "sample_index": sample_index,
        "selection": "first transition"
        if args.sample_index is None
        else "explicit index",
        "seed": args.seed,
        "mc_samples": args.mc_samples,
        "split_strategy_config": config.split_strategy,
        "sample_accuracy": float(np.mean(arrays["predictions"] == arrays["labels"])),
        "max_sphere_norm_error": float(
            np.max(np.abs(np.linalg.norm(paths, axis=-1) - 1))
        ),
        "class_names": CLASS_NAMES,
        "tag_ids": TAG_IDS,
        "solver_stats": {name: float(value) for name, value in solver_stats.items()},
        "note": "Illustrative inference on one sequence; not a test-set evaluation. "
        "An auto split uses Torch when installed and NumPy otherwise. "
        "Adaptive steps depend on the inference batch and MC sample count.",
    }
    output_dir = args.output_dir or args.run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    render_example(arrays, metadata, output_dir)
    np.savez_compressed(output_dir / "activity_example.npz", **arrays)
    (output_dir / "activity_example.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Saved PNG, PDF, arrays and metadata to {output_dir}")


if __name__ == "__main__":
    main()
