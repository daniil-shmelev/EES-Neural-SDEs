"""Prepare an isolated memory measurement of the training model and loss."""

from pathlib import Path

import jax
import jax.numpy as jnp

from experiments.kuramoto.datasets.kuramoto_dataset import (
    KuramotoDataset,
    default_npz_path,
)
from experiments.kuramoto.experiment.factories import make_model
from experiments.kuramoto.experiment.losses import make_multi_horizon_energy_score


def prepare_benchmark(config):
    dataset = KuramotoDataset(
        default_npz_path(Path(config.data_dir), config.N, config.data_seed), "train"
    )
    if len(dataset) < config.batch_size:
        raise ValueError("The dataset must contain a complete benchmark batch")
    # Keep the dataset on the host; transfer only the measured batch.
    theta = jnp.asarray(dataset.theta[: config.batch_size])
    omega = jnp.asarray(dataset.omega[: config.batch_size])
    batch = dict(
        theta0=theta[:, 0], omega0=omega[:, 0], theta_traj=theta, omega_traj=omega
    )
    metadata = dataset.metadata()
    model = make_model(config, metadata, jax.random.key(config.seed))
    loss = make_multi_horizon_energy_score(
        horizons=config.loss_horizons, n_samples=config.loss_n_samples
    )
    return model, batch, loss, metadata
