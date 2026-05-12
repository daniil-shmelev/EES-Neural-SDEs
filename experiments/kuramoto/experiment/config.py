"""Frozen-dataclass + TOML config for the Kuramoto NSDE experiment."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path


class Devices(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class Solvers(StrEnum):
    CFEES25 = "cfees25"
    CFEES27 = "cfees27"
    CG2 = "cg2"
    CG4 = "cg4"
    RKMK = "rkmk"


class ModelKind(StrEnum):
    KURAMOTO_NSDE = "kuramoto_nsde"
    EUCLIDEAN_BASELINE = "euclidean_baseline"


@dataclass(frozen=True)
class ExperimentConfig:
    # Core training
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 0
    device: Devices = Devices.GPU
    dtype: str = "float32"

    # Model architecture
    model: ModelKind = ModelKind.KURAMOTO_NSDE
    hidden_dim: int = 128
    drift_depth: int = 3
    diffusion_depth: int = 2
    activation: str = "silu"
    diffusion_scale: float = 0.3
    couple_theta_omega: bool = False
    drift_kind: str = "mean_field"
    # Options:
    #   "mean_field"          - moments + node MLP, default; O(N), perm-equivariant.
    #   "indexed_mean_field"  - mean_field + per-osc embedding; matches the
    #                           dataset's index-pinned Omega assignment but
    #                           breaks pure equivariance.
    #   "equivariant"         - full DeepSets phi(z_ij) -> mean-pool; O(N^2),
    #                           viable only at N <= ~64 on consumer GPUs.
    #   "mlp"                 - legacy 3N->2N MLP; retained for old checkpoints.

    # SDE integrator
    solver: Solvers = Solvers.CFEES25
    n_steps: int = 200
    adjoint: str = "auto"

    # Data
    N: int = 2
    data_seed: int = 0
    data_dir: str = "experiments/kuramoto/data/"

    # Loss / training
    loss_horizons: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0)
    loss_n_samples: int = 4
    grad_clip_norm: float = 1.0

    # Outputs
    save_every_epoch: bool = True
    skip_plots: bool = False


def make_config(**overrides) -> ExperimentConfig:
    return ExperimentConfig(**overrides)


def load_config(path: Path) -> ExperimentConfig:
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    if "device" in data:
        data["device"] = Devices(data["device"])
    if "solver" in data:
        data["solver"] = Solvers(data["solver"])
    if "model" in data:
        data["model"] = ModelKind(data["model"])
    if "loss_horizons" in data:
        data["loss_horizons"] = tuple(float(x) for x in data["loss_horizons"])
    return ExperimentConfig(**data)


def serialize_config(config: ExperimentConfig) -> str:
    """JSON-serialise the config (StrEnums become strings, tuples become lists)."""
    payload = asdict(config)
    for k, v in payload.items():
        if isinstance(v, StrEnum):
            payload[k] = v.value
        elif isinstance(v, tuple):
            payload[k] = list(v)
    return json.dumps(payload, indent=2, sort_keys=True)
