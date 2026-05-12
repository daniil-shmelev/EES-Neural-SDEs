"""Factories for activity data loaders and models."""

from __future__ import annotations

from typing import Literal

import jax
from cyreal.loader import DataLoader
from cyreal.transforms import BatchTransform

from experiments.sphere_latent_nsde.config import ActivityConfig
from experiments.sphere_latent_nsde.dataset import HumanActivityDataset
from experiments.sphere_latent_nsde.models import ActivityLatentSDE


def make_dataset(
    config: ActivityConfig,
    split: Literal["train", "val", "test"],
) -> HumanActivityDataset:
    return HumanActivityDataset(
        split=split,
        data_dir=config.data_dir,
        seed=config.split_seed,
        source=config.data_source,
        split_strategy=config.split_strategy,
        output_num_timepoints=config.num_timepoints,
    )


def make_loader(
    config: ActivityConfig,
    split: Literal["train", "val", "test"],
) -> DataLoader:
    dataset = make_dataset(config, split)
    return DataLoader(
        [
            dataset.make_array_source(),
            BatchTransform(
                batch_size=config.batch_size,
                drop_last=split == "train" and config.drop_last_train,
            ),
        ]
    )


def make_model(config: ActivityConfig, key: jax.Array) -> ActivityLatentSDE:
    return ActivityLatentSDE(
        h_dim=config.h_dim,
        z_dim=config.z_dim,
        n_deg=config.n_deg,
        num_timepoints=config.num_timepoints,
        nfe_budget=config.effective_nfe_budget,
        solver_name=config.solver,
        adjoint_name=config.adjoint,
        learnable_prior=config.learnable_prior,
        use_atanh=config.use_atanh,
        key=key,
    )
