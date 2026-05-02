from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jax
import numpy as np
from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.time_utils import make_sequence_disk_source
from cyreal.datasets.utils import to_host_jax_array
from cyreal.sources import ArraySource, DiskSource

from experiments.stochastic_volatility.experiment.config import Experiments

_DATA_DIR = Path(__file__).parent.parent / "data"

STOCH_VOL_DATSET_PATHS = {
    Experiments.BLACK_SCHOLES: _DATA_DIR / "black-scholes_data.npz",
    Experiments.LOCAL_STOCH_VOL: _DATA_DIR / "classical_local_stochastic_volatility_data.npz",
    Experiments.HESTON: _DATA_DIR / "heston_data.npz",
    Experiments.ROUGH_HESTON: _DATA_DIR / "rough_heston_data.npz",
    Experiments.QUADRATIC_ROUGH_HESTON: _DATA_DIR / "quadratic_rough_heston_data.npz",
    Experiments.BERGOMI: _DATA_DIR / "bergomi_data.npz",
    Experiments.ROUGH_BERGOMI: _DATA_DIR / "rough_bergomi_data.npz",
}


@dataclass
class StochVolatilityDataset(DatasetProtocol):
    experiment: Experiments
    split: Literal["train", "val", "test"]
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    ordering: Literal["sequential", "shuffle"] = field(init=False)

    def __post_init__(self) -> None:
        self.ordering = "shuffle" if self.split == "train" else "sequential"

        data = np.load(STOCH_VOL_DATSET_PATHS[self.experiment])
        driver = np.asarray(data["driver"], dtype=np.float32)
        solution = np.asarray(data["solution"], dtype=np.float32)

        # For rough-volatility datasets the .npz contains multiple channels, but the
        # training pipeline is set up to learn a single target channel (e.g. price).
        # Keep only the first channel for both driver and solution to match the
        # behavior in `taming_the_ito_lyon.data.datasets.prepare_dataset`.
        driver = driver[..., 0:1]
        solution = solution[..., 0:1]

        if driver.shape != solution.shape:
            raise ValueError(
                f"driver and solution must have the same shape, got {driver.shape} and {solution.shape}"
            )
        if driver.ndim != 3:
            raise ValueError(
                f"Expected arrays shaped (num_examples, T, C), got driver shape {driver.shape}"
            )

        driver_split = _select_example_split(
            driver,
            split=self.split,
            train_fraction=self.train_fraction,
            val_fraction=self.val_fraction,
        )
        solution_split = _select_example_split(
            solution,
            split=self.split,
            train_fraction=self.train_fraction,
            val_fraction=self.val_fraction,
        )

        self._driver = to_host_jax_array(driver_split)
        self._solution = to_host_jax_array(solution_split)

    def __len__(self) -> int:
        return int(self._driver.shape[0])

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        return {
            "driver": self._driver[index],
            "solution": self._solution[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        """Expose the full dataset as a PyTree of JAX arrays."""
        return {"driver": self._driver, "solution": self._solution}

    def make_array_source(self) -> ArraySource:
        return ArraySource(self.as_array_dict(), ordering=self.ordering)

    def make_disk_source(self) -> DiskSource:
        return make_sequence_disk_source(
            contexts=np.asarray(self._driver),
            targets=np.asarray(self._solution),
            ordering=self.ordering,
            prefetch_size=128,
        )


def _select_example_split(
    array: np.ndarray,
    *,
    split: Literal["train", "val", "test"],
    train_fraction: float,
    val_fraction: float = 0.0,
) -> np.ndarray:
    """Split independent examples along axis 0 (no overlap)."""
    n = int(len(array))
    if n <= 0:
        raise ValueError("Array must be non-empty.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    train_end = min(max(int(n * train_fraction), 1), n)
    if val_fraction > 0.0:
        val_end = min(max(int(n * (train_fraction + val_fraction)), train_end + 1), n)
    else:
        val_end = train_end

    if split == "train":
        return array[:train_end]
    if split == "val":
        if val_fraction == 0.0:
            raise ValueError("val_fraction must be > 0 when split='val'.")
        return array[train_end:val_end]
    return array[val_end:]


if __name__ == "__main__":
    dataset = StochVolatilityDataset(
        Experiments.ROUGH_BERGOMI,
        "train",
    )
    print(dataset._driver.shape)
    print(dataset._solution.shape)
