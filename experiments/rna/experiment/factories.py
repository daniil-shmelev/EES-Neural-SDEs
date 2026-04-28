from typing import Callable, Literal

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

from experiments.rna.experiment.config import ExperimentConfig, Experiments, Solvers

_SOLVER_CLASSES: dict[Solvers, type[AbstractSolver]] = {
    Solvers.CFEES25: CFEES25,
    Solvers.CFEES27: CFEES27,
    Solvers.CG2: CG2,
    Solvers.CG4: CG4,
    Solvers.RKMK: RKMK,
}


def build_solver(name: Solvers) -> AbstractSolver:
    cls = _SOLVER_CLASSES[name]
    return cls()


def build_adjoint(name: str | None, n_steps: int) -> AbstractAdjoint | None:
    """Map an adjoint-name string to a diffrax adjoint instance.

    Mirrors ``experiment.integrator_benchmark_single.build_adjoint`` so that
    benchmark cells and training runs use the same logic.

      - ``"auto"`` / ``None``: return ``None`` so the model picks
        ``ReversibleAdjoint`` for reversible solvers and ``DirectAdjoint``
        otherwise.
      - ``"reversible"``: ``ReversibleAdjoint()``.
      - ``"direct"``: ``DirectAdjoint()``.
      - ``"checkpoint_recursive"`` (aka ``"checkpoint_log"``):
        ``RecursiveCheckpointAdjoint()`` with diffrax's default Stumm-Walther
        online binomial treeverse (O(sqrt(n_steps)) checkpoints).
      - ``"checkpoint_full"``: ``RecursiveCheckpointAdjoint(checkpoints=n_steps+8)``,
        i.e. one snapshot per step (O(n_steps) tape).
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
) -> DataLoader:
    match config.experiment:
        case Experiments.SPD:
            from datasets.spd.dataset import CovarianceDataset

            source = CovarianceDataset(split=split).make_array_source()
            return DataLoader(
                [
                    source,
                    BatchTransform(
                        batch_size=config.batch_size,
                        drop_last=split == "train",
                    ),
                ]
            )
        case Experiments.RNA:
            from datasets.rna.dataset import RNATorsionDataset

            source = RNATorsionDataset(
                split=split,
                context_length=config.context_length,
                residues_per_state=config.residues_per_state,
                future_bases_window=config.future_bases_window,
                filter_canonical=config.filter_canonical,
                canonical_threshold=config.canonical_threshold,
                max_chains=config.max_chains,
            ).make_array_source()
            return DataLoader(
                [
                    source,
                    BatchTransform(
                        batch_size=config.batch_size,
                        drop_last=split == "train",
                    ),
                ]
            )
        case _:
            raise ValueError(f"Unsupported experiment: {config.experiment}")


def make_model(
    config: ExperimentConfig,
    metadata: dict[str, int],
    key: jax.Array,
) -> eqx.Module:
    match config.experiment:
        case Experiments.SPD:
            from models.nsde import ManifoldNeuralSDE

            return ManifoldNeuralSDE(
                n_stocks=metadata["n_stocks"],
                hidden_dim=config.hidden_dim,
                ctx_dim=config.ctx_dim,
                n_steps=config.n_steps,
                dt=config.dt,
                solver=build_solver(config.solver),
                diffusion_scale=config.diffusion_scale,
                key=key,
            )
        case Experiments.RNA:
            from models.torus_nsde import TorusNeuralSDE

            return TorusNeuralSDE(
                num_angles=metadata["num_angles"],
                hidden_dim=config.hidden_dim,
                ctx_dim=config.ctx_dim,
                n_steps=config.n_steps,
                dt=config.dt,
                solver=build_solver(config.solver),
                diffusion_scale=config.diffusion_scale,
                residues_per_state=config.residues_per_state,
                future_bases_window=config.future_bases_window,
                future_ctx_dim=config.future_ctx_dim,
                activation=config.activation,
                drift_depth=config.drift_depth,
                diffusion_depth=config.diffusion_depth,
                adjoint=build_adjoint(config.adjoint, config.n_steps),
                key=key,
            )
        case _:
            raise ValueError(f"Unsupported experiment: {config.experiment}")


def make_prediction_fn() -> Callable[[eqx.Module, jax.Array, jax.Array], jax.Array]:
    def prediction_fn(
        model: eqx.Module,
        inputs: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        sample_keys = jax.random.split(key, inputs.shape[0])
        return jax.vmap(lambda context, sample_key: model(context, sample_key))(
            inputs, sample_keys
        )

    return prediction_fn
