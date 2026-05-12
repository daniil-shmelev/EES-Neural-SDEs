"""Cached-NPZ dataset wrapper for the Kuramoto experiment.

Loads `data/kuramoto_N{N}_seed{seed}.npz` (produced by M1) and exposes a
cyreal-compatible array source for the training loop. Each sample is a
single trajectory: (theta_0, omega_0) initial condition + the full
sub-sampled trajectory (theta, omega) at all observation timepoints +
the time grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from cyreal.sources import ArraySource


class KuramotoDataset:
    """Wrap one $N$-oscillator NPZ split into a cyreal-compatible array source."""

    def __init__(
        self,
        npz_path: Path,
        split: Literal["train", "val", "test"],
    ):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Dataset NPZ not found: {npz_path}. Run "
                f"`python -m experiments.kuramoto.scripts.run_m1` to produce it."
            )
        meta_path = npz_path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata sidecar not found: {meta_path}.")
        with np.load(npz_path) as npz:
            self.theta = np.asarray(npz[f"theta_{split}"])  # (B, T_obs, N)
            self.omega = np.asarray(npz[f"omega_{split}"])  # (B, T_obs, N)
            self.t_grid = np.asarray(npz["t_grid"])  # (T_obs,)
        self.meta = json.loads(meta_path.read_text())
        if self.theta.shape[0] != self.omega.shape[0]:
            raise ValueError(f"theta / omega split-length mismatch: "
                             f"{self.theta.shape[0]} vs {self.omega.shape[0]}")
        self.split = split
        self.N = int(self.meta["N"])
        self.n_obs = int(self.theta.shape[1])
        self.ordering = "shuffle" if split == "train" else "sequential"

    def __len__(self) -> int:
        return self.theta.shape[0]

    def metadata(self) -> dict:
        return {
            "N": self.N,
            "n_obs": self.n_obs,
            "T": float(self.meta.get("T", float(self.t_grid[-1]))),
            "split_size": len(self),
            **{k: v for k, v in self.meta.items() if k not in ("library_versions",)},
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        """Return the per-trajectory arrays keyed by name."""
        return {
            "theta0": jnp.asarray(self.theta[:, 0]),
            "omega0": jnp.asarray(self.omega[:, 0]),
            "theta_traj": jnp.asarray(self.theta),
            "omega_traj": jnp.asarray(self.omega),
        }

    def make_array_source(self) -> ArraySource:
        """Return a cyreal `ArraySource` keyed for the training loop."""
        return ArraySource(self.as_array_dict(), ordering=self.ordering)


def default_npz_path(data_dir: Path, N: int, seed: int = 0) -> Path:
    return Path(data_dir) / f"kuramoto_N{N}_seed{seed}.npz"
