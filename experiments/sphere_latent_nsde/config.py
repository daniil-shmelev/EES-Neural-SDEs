"""Configuration for the sphere latent NSDE activity experiment."""

from __future__ import annotations

import dataclasses
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from experiments.sphere_latent_nsde.dataset import NUM_TIMEPOINTS


@dataclass(frozen=True)
class ActivityConfig:
    data_dir: Path = Path("/home/luke/EES-LatentSDEonHS/data_dir")
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 0
    h_dim: int = 32
    z_dim: int = 16
    n_deg: int = 4
    num_timepoints: int = NUM_TIMEPOINTS
    solver: Literal["geometric_euler", "cfees25"] = "geometric_euler"
    adjoint: Literal["auto", "direct", "reversible"] = "auto"
    learnable_prior: bool = False
    use_atanh: bool = False
    kl0_weight: float = 1e-4
    klp_weight: float = 1e-4
    pxz_weight: float = 1.0
    aux_weight: float = 10.0
    aux_hidden_dim: int = 32
    mc_train_samples: int = 1
    mc_eval_samples: int = 1
    split_seed: int = 42
    split_strategy: Literal["auto", "numpy", "torch"] = "auto"
    data_source: Literal["auto", "raw", "torch"] = "auto"
    drop_last_train: bool = True
    max_train_batches: int | None = None
    max_eval_batches: int | None = None


def load_config(path: Path) -> ActivityConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    if "data_dir" in data:
        data["data_dir"] = Path(data["data_dir"])
    return ActivityConfig(**data)


def replace_config(config: ActivityConfig, **updates) -> ActivityConfig:
    clean = {key: value for key, value in updates.items() if value is not None}
    if "data_dir" in clean:
        clean["data_dir"] = Path(clean["data_dir"])
    return dataclasses.replace(config, **clean)


def to_json(config: ActivityConfig) -> str:
    data = asdict(config)
    data["data_dir"] = str(data["data_dir"])
    return json.dumps(data, indent=2, sort_keys=True)
