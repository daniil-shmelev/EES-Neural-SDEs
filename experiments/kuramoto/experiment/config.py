"""TOML configuration for the Kuramoto training and memory benchmarks."""

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
    CG2 = "cg2"
    GEOMETRIC_EULER_HEUN = "geometric_euler_heun"


class ModelKind(StrEnum):
    KURAMOTO_NSDE = "kuramoto_nsde"


COMPARISON_METHODS = (
    ("cg2_full", Solvers.CG2, "checkpoint_full", 75),
    ("cg2_recursive", Solvers.CG2, "checkpoint_recursive", 75),
    ("cfees25_reversible", Solvers.CFEES25, "reversible", 50),
    ("geometric_euler_heun_full", Solvers.GEOMETRIC_EULER_HEUN, "checkpoint_full", 150),
    (
        "geometric_euler_heun_recursive",
        Solvers.GEOMETRIC_EULER_HEUN,
        "checkpoint_recursive",
        150,
    ),
)


def validate_method(solver, adjoint, *, adaptive=False):
    """Keep fixed-step comparisons restricted; allow checkpointed adaptive training."""
    if adaptive and solver == Solvers.CFEES25 and adjoint == "checkpoint_recursive":
        return
    if (solver, adjoint) not in {(s, a) for _, s, a, _ in COMPARISON_METHODS}:
        raise ValueError(
            "Kuramoto supports only CG2 or Geometric Euler–Heun with checkpoint_full "
            "or checkpoint_recursive, "
            "and CF-EES(2,5) with reversible."
        )


@dataclass(frozen=True)
class ExperimentConfig:
    # Core training
    epochs: int = 15
    batch_size: int = 32
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
    diffusion_scale: float = 0.1
    couple_theta_omega: bool = False
    drift_kind: str = "second_order_mean_field"

    # SDE integrator
    solver: Solvers = Solvers.CFEES25
    n_steps: int = 50
    adjoint: str = "reversible"
    adaptive: bool = False
    adaptive_atol: float = 1e-3
    adaptive_dtmin: float = 1e-4
    adaptive_max_steps: int = 4096

    # Data
    N: int = 1000
    data_seed: int = 0
    data_dir: str = "experiments/kuramoto/data/"

    # Loss / training
    loss_horizons: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0)
    loss_n_samples: int = 4
    grad_clip_norm: float = 1.0

    # Outputs
    save_every_epoch: bool = True
    skip_plots: bool = False

    def __post_init__(self):
        if self.adaptive:
            if self.solver != Solvers.CFEES25:
                raise ValueError("Adaptive stepping requires a CF-EES embedded pair.")
            if self.adaptive_atol <= 0 or self.adaptive_dtmin <= 0:
                raise ValueError(
                    "Adaptive tolerance and minimum step must be positive."
                )
            if self.adaptive_max_steps < self.n_steps:
                raise ValueError("adaptive_max_steps must be at least n_steps.")
        validate_method(self.solver, self.adjoint, adaptive=self.adaptive)
        if self.model != ModelKind.KURAMOTO_NSDE or self.drift_kind not in (
            "indexed_mean_field",
            "second_order_mean_field",
        ):
            raise ValueError("Use a supported mean-field Kuramoto drift.")


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "kuramoto.toml"
PROTOCOL = "kuramoto_second_order_mean_field_georaxbf5c355_v4"


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
