"""Build the Kuramoto model, data loader, solver and adjoint."""

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
    GeneralShARK,
    RecursiveCheckpointAdjoint,
    ReversibleAdjoint,
)
from georax import CFEES25, CG2, SRKMK, GeometricEulerHeun

from experiments.kuramoto.datasets.kuramoto_dataset import (
    KuramotoDataset,
    default_npz_path,
)
from experiments.kuramoto.experiment.config import (
    ExperimentConfig,
    ModelKind,
    Solvers,
    validate_method,
)

_GEORAX_SOLVERS: dict[Solvers, type[AbstractSolver]] = {
    Solvers.CFEES25: CFEES25,
    Solvers.CG2: CG2,
    Solvers.GEOMETRIC_EULER_HEUN: GeometricEulerHeun,
}


def build_solver(name: Solvers) -> AbstractSolver:
    if name == Solvers.SRKMK_GENERAL_SHARK:
        return SRKMK(GeneralShARK())
    cls = _GEORAX_SOLVERS[name]
    return cls()


def build_adjoint(
    solver: Solvers, name: str, n_steps: int, *, adaptive=False
) -> AbstractAdjoint:
    """Build the adjoint for a supported Kuramoto method."""
    validate_method(solver, name, adaptive=adaptive)
    if name == "reversible":
        return ReversibleAdjoint()
    if name == "checkpoint_recursive":
        return RecursiveCheckpointAdjoint()
    return RecursiveCheckpointAdjoint(checkpoints=n_steps + 8)


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
                pad_last_batch=split != "train",
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
    T = float(metadata["T"])
    dt = T / config.n_steps
    step_limit = config.adaptive_max_steps if config.adaptive else config.n_steps
    adjoint = build_adjoint(
        config.solver, config.adjoint, step_limit, adaptive=config.adaptive
    )

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
            adaptive=config.adaptive,
            adaptive_atol=config.adaptive_atol,
            adaptive_dtmin=config.adaptive_dtmin,
            adaptive_max_steps=config.adaptive_max_steps,
            key=key,
        )
    raise ValueError(f"Unknown model kind: {config.model}")
