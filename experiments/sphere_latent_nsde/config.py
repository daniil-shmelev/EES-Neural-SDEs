"""Configuration for the sphere latent NSDE activity experiment."""

from __future__ import annotations

import itertools
import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

from experiments.sphere_latent_nsde.dataset import NUM_TIMEPOINTS

METHODS = {
    "geometric_euler_direct": ("geometric_euler", "direct"),
    "geometric_euler_recursive_checkpoint": (
        "geometric_euler",
        "recursive_checkpoint",
    ),
    "cfees25_reversible": ("cfees25", "reversible"),
    "cg2_direct": ("cg2", "direct"),
}

SOLVER_NFE_PER_STEP = {
    "geometric_euler": 1,
    "cg2": 2,
    "cfees25": 3,
}
COMMON_NFE_GRANULARITY = math.lcm(*SOLVER_NFE_PER_STEP.values())


@dataclass(frozen=True)
class ActivityConfig:
    experiment: str = "activity"
    data_dir: Path = Path("experiments/sphere_latent_sde/data_dir")
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    lr_schedule: Literal["constant", "cosine"] = "constant"
    lr_restart: int = 30
    lr_min_ratio: float = 0.0
    seed: int = 0
    device: Literal["cpu", "gpu"] = "gpu"
    h_dim: int = 32
    z_dim: int = 16
    n_deg: int = 4
    num_timepoints: int = NUM_TIMEPOINTS
    nfe_budget: int | None = None
    solver: Literal["geometric_euler", "cg2", "cfees25"] = "geometric_euler"
    adjoint: Literal["auto", "direct", "recursive_checkpoint", "reversible"] = "auto"
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

    @property
    def method(self) -> str:
        return f"{self.solver}_{self.adjoint}"

    @property
    def nfe_per_step(self) -> int:
        return SOLVER_NFE_PER_STEP[self.solver]

    @property
    def effective_nfe_budget(self) -> int:
        if self.nfe_budget is not None:
            return int(self.nfe_budget)
        output_intervals = int(self.num_timepoints) - 1
        budget = output_intervals - (output_intervals % COMMON_NFE_GRANULARITY)
        if budget <= 0:
            raise ValueError(
                "num_timepoints must provide at least one common NFE interval; "
                "set nfe_budget explicitly for very small grids."
            )
        return budget

    @property
    def solve_n_steps(self) -> int:
        budget = self.effective_nfe_budget
        nfe_per_step = self.nfe_per_step
        if budget % nfe_per_step != 0:
            raise ValueError(
                f"nfe_budget={budget} must be divisible by {nfe_per_step} "
                f"for solver={self.solver!r}."
            )
        return budget // nfe_per_step


def load_config(path: Path) -> ActivityConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return _build_config(data)


def is_sweep_config(path: Path) -> bool:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return "base" in data and "sweep" in data


def load_sweep(path: Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data["base"], data["sweep"]


def sweep_len(path: Path) -> int:
    _, sweep_axes = load_sweep(path)
    return sum(1 for _ in itertools.product(*sweep_axes.values()))


def sweep_config_at(path: Path, index: int) -> ActivityConfig:
    base, sweep_axes = load_sweep(path)
    keys = list(sweep_axes.keys())
    combos = list(itertools.product(*sweep_axes.values()))
    if index < 0 or index >= len(combos):
        raise IndexError(f"sweep index {index} out of range [0, {len(combos)})")
    return _build_config({**base, **dict(zip(keys, combos[index]))})


def to_toml(config: ActivityConfig) -> str:
    data = asdict(config)
    lines: list[str] = []
    for name in (
        "experiment",
        "data_dir",
        "epochs",
        "batch_size",
        "learning_rate",
        "lr_schedule",
        "lr_restart",
        "lr_min_ratio",
        "seed",
        "device",
        "h_dim",
        "z_dim",
        "n_deg",
        "num_timepoints",
        "nfe_budget",
        "solver",
        "adjoint",
        "learnable_prior",
        "use_atanh",
        "kl0_weight",
        "klp_weight",
        "pxz_weight",
        "aux_weight",
        "aux_hidden_dim",
        "mc_train_samples",
        "mc_eval_samples",
        "split_seed",
        "split_strategy",
        "data_source",
        "drop_last_train",
        "max_train_batches",
        "max_eval_batches",
    ):
        value = data[name]
        if value is None:
            continue
        lines.append(f"{name} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _build_config(data: dict[str, Any]) -> ActivityConfig:
    data = dict(data)
    method = data.pop("method", None)
    if method is not None:
        if method not in METHODS:
            raise ValueError(f"Unknown activity method: {method!r}")
        data["solver"], data["adjoint"] = METHODS[method]
    if "data_dir" in data:
        data["data_dir"] = Path(data["data_dir"])

    valid = {field.name for field in fields(ActivityConfig)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise KeyError(f"Unknown activity config keys: {unknown}")
    return ActivityConfig(**data)


def _toml_value(value: Any) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
