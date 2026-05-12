"""Factory helpers for Kuramoto solver, adjoint, data, and model dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
from cyreal.loader import DataLoader
from cyreal.transforms import BatchTransform
from diffrax import (
    AbstractAdjoint,
    AbstractSolver,
    DirectAdjoint,
    RecursiveCheckpointAdjoint,
    ReversibleAdjoint,
)
from georax import CFEES25, CFEES27, CG2, CG4, RKMK

from experiments.kuramoto.datasets.kuramoto_dataset import (
    KuramotoDataset,
    default_npz_path,
)
from experiments.kuramoto.experiment.config import (
    ExperimentConfig,
    ModelKind,
    Solvers,
)

_GEORAX_SOLVERS: dict[Solvers, type[AbstractSolver]] = {
    Solvers.CFEES25: CFEES25,
    Solvers.CFEES27: CFEES27,
    Solvers.CG2: CG2,
    Solvers.CG4: CG4,
    Solvers.RKMK: RKMK,
}


def build_solver(name: Solvers) -> AbstractSolver:
    cls = _GEORAX_SOLVERS[name]
    return cls()


def build_adjoint(name: str | None, n_steps: int) -> AbstractAdjoint | None:
    """Map an adjoint-name string to a diffrax adjoint instance.

      - ``"auto"`` / ``None``: model picks ReversibleAdjoint for reversible
        solvers and DirectAdjoint otherwise.
      - ``"reversible"``: ``ReversibleAdjoint()``.
      - ``"direct"``: ``DirectAdjoint()``.
      - ``"checkpoint_recursive"``: diffrax default Stumm-Walther
        treeverse, $O(\\sqrt{n})$ checkpoints.
      - ``"checkpoint_full"``: $O(n)$ tape (one snapshot per step).
    """
    if name is None:
        return None
    name = name.lower()
    max_steps = n_steps + 8
    if name == "auto":
        return None
    if name == "reversible":
        return ReversibleAdjoint()
    if name == "direct":
        return DirectAdjoint()
    if name in ("checkpoint_recursive", "recursive", "checkpoint_log"):
        return RecursiveCheckpointAdjoint()
    if name in ("checkpoint_full", "full"):
        return RecursiveCheckpointAdjoint(checkpoints=max_steps)
    raise ValueError(f"unknown adjoint {name!r}")


def make_loader(
    config: ExperimentConfig,
    split: Literal["train", "val", "test"],
) -> tuple[DataLoader, KuramotoDataset]:
    """Build a cyreal DataLoader for a given split, plus the underlying dataset."""
    dataset = KuramotoDataset(
        npz_path=default_npz_path(Path(config.data_dir), config.N, config.data_seed),
        split=split,
    )
    loader = DataLoader(
        [
            dataset.make_array_source(),
            BatchTransform(
                batch_size=config.batch_size,
                drop_last=split == "train",
            ),
        ]
    )
    return loader, dataset


def make_model(
    config: ExperimentConfig,
    metadata: dict[str, int],
    key: jax.Array,
) -> eqx.Module:
    N = int(metadata["N"])
    n_obs = int(metadata["n_obs"])
    T = float(metadata.get("T", config.n_steps * 0.025))
    dt = T / config.n_steps
    adjoint = build_adjoint(config.adjoint, config.n_steps)

    if config.model is ModelKind.KURAMOTO_NSDE:
        from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE

        return KuramotoNSDE(
            N=N,
            hidden_dim=config.hidden_dim,
            n_steps=config.n_steps,
            dt=dt,
            n_obs=n_obs,
            solver=build_solver(config.solver),
            diffusion_scale=config.diffusion_scale,
            adjoint=adjoint,
            activation=config.activation,
            drift_depth=config.drift_depth,
            diffusion_depth=config.diffusion_depth,
            couple_theta_omega=config.couple_theta_omega,
            drift_kind=config.drift_kind,
            key=key,
        )
    if config.model is ModelKind.EUCLIDEAN_BASELINE:
        from experiments.kuramoto.models.euclidean_baseline import EuclideanKuramotoNSDE

        return EuclideanKuramotoNSDE(
            N=N,
            hidden_dim=config.hidden_dim,
            n_steps=config.n_steps,
            dt=dt,
            n_obs=n_obs,
            diffusion_scale=config.diffusion_scale,
            adjoint=adjoint,
            activation=config.activation,
            drift_depth=config.drift_depth,
            diffusion_depth=config.diffusion_depth,
            key=key,
        )
    raise ValueError(f"Unknown model kind: {config.model}")
