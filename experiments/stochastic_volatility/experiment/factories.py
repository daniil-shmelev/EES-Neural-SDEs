from typing import Literal

import equinox as eqx
import jax
from cyreal.loader import DataLoader
from cyreal.transforms import BatchTransform

from experiments.stochastic_volatility.experiment.config import ExperimentConfig, Experiments
from experiments.stochastic_volatility.experiment.losses import SampleFn


def make_loader(
    config: ExperimentConfig,
    split: Literal["train", "val", "test"],
) -> DataLoader:
    match config.experiment:
        case (
            Experiments.BLACK_SCHOLES
            | Experiments.LOCAL_STOCH_VOL
            | Experiments.HESTON
            | Experiments.ROUGH_HESTON
            | Experiments.QUADRATIC_ROUGH_HESTON
            | Experiments.BERGOMI
            | Experiments.ROUGH_BERGOMI
        ):
            from experiments.stochastic_volatility.experiment.dataset import StochVolatilityDataset

            source = StochVolatilityDataset(
                experiment=config.experiment, split=split
            ).make_array_source()
            return DataLoader(
                [
                    source,
                    BatchTransform(
                        batch_size=config.batch_size,
                        drop_last=False,
                    ),
                ]
            )
        case _:
            raise ValueError(f"Unsupported experiment: {config.experiment}")


def _infer_solution_shape(config: ExperimentConfig) -> tuple[int, int]:
    """Returns (state_dim, n_time_steps) from a training batch."""
    loader = make_loader(config, "train")
    state = loader.init_state(jax.random.key(0))
    batch, _, _ = loader.next(state)
    sol = batch["solution"]
    return int(sol.shape[-1]), int(sol.shape[-2])


def make_model(
    config: ExperimentConfig,
    key: jax.Array,
) -> eqx.Module:
    from experiments.stochastic_volatility.models.nsde import SimpleNeuralSDE

    state_dim, n_save = _infer_solution_shape(config)
    return SimpleNeuralSDE(
        state_dim=state_dim,
        hidden_dim=config.hidden_dim,
        n_steps=config.n_steps,
        dt=config.dt,
        solver=config.solver.build(),
        diffusion_scale=config.diffusion_scale,
        n_save=n_save,
        save_path=True,
        key=key,
    )


def make_sample_fn(config: ExperimentConfig) -> SampleFn:
    def sample_fn(model, batch, mask, key):
        batch_size = batch["solution"].shape[0]
        y0s = jax.numpy.ones(
            (batch_size, model.state_dim), dtype=jax.numpy.float32
        )
        keys = jax.random.split(key, batch_size)
        predictions = jax.vmap(model)(y0s, keys)
        return predictions, batch["solution"]

    return sample_fn
