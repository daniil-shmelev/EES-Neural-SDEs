"""TOML configuration for the Kuramoto training benchmark."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from experiments.config import cosine_stage_count, expand_integrators, method_label


class Devices(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class Solvers(StrEnum):
    CF_COSINE_EES = "cf_cosine_ees"
    CG2 = "cg2"
    GEOMETRIC_EULER_HEUN = "geometric_euler_heun"
    SRKMK_GENERAL_SHARK = "srkmk_general_shark"


class ModelKind(StrEnum):
    KURAMOTO_NSDE = "kuramoto_nsde"


COMPARISON_METHODS = (
    ("cg2_full", Solvers.CG2, "checkpoint_full"),
    ("cg2_recursive", Solvers.CG2, "checkpoint_recursive"),
    ("srkmk_general_shark_full", Solvers.SRKMK_GENERAL_SHARK, "checkpoint_full"),
    (
        "srkmk_general_shark_recursive",
        Solvers.SRKMK_GENERAL_SHARK,
        "checkpoint_recursive",
    ),
    ("cf_cosine_ees_p5_reversible", Solvers.CF_COSINE_EES, "reversible"),
    ("geometric_euler_heun_full", Solvers.GEOMETRIC_EULER_HEUN, "checkpoint_full"),
    (
        "geometric_euler_heun_recursive",
        Solvers.GEOMETRIC_EULER_HEUN,
        "checkpoint_recursive",
    ),
)


def validate_method(solver, adjoint, *, adaptive=False):
    """Keep fixed-step comparisons restricted; allow checkpointed adaptive training."""
    if (
        adaptive
        and solver == Solvers.CF_COSINE_EES
        and adjoint == "checkpoint_recursive"
    ):
        return
    if (solver, adjoint) not in {(s, a) for _, s, a in COMPARISON_METHODS}:
        raise ValueError(
            "Kuramoto supports only CG2, Geometric Euler–Heun or SRKMK–GeneralShARK with checkpoint_full "
            "or checkpoint_recursive, "
            "and cosine EES with reversible."
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
    solver: Solvers = Solvers.CF_COSINE_EES
    reversible_order: int = 5
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
        cosine_stage_count(self.reversible_order)
        if type(self.n_steps) is not int or self.n_steps <= 0:
            raise ValueError("n_steps must be a positive integer.")
        if self.adaptive:
            if self.solver != Solvers.CF_COSINE_EES:
                raise ValueError(
                    "Adaptive stepping requires a cosine EES embedded pair."
                )
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

    @property
    def method(self):
        return method_label(
            self.solver,
            self.adjoint,
            self.adaptive,
            reversible_order=self.reversible_order,
        )

    @property
    def embedded(self):
        return self.solver == Solvers.CF_COSINE_EES and self.adaptive


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"
PROTOCOL = "kuramoto_second_order_mean_field_cosine_georax20b401c_v5"


def make_config(**overrides) -> ExperimentConfig:
    return ExperimentConfig(**overrides)


def load_config(path: Path) -> ExperimentConfig:
    """Read shared settings for diagnostics."""
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return _build_config(data.get("base", data))


def load_runs(path: Path) -> list[ExperimentConfig]:
    """Expand the integrators and adjoints selected in TOML."""
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return [_build_config(settings) for settings in expand_integrators(data)]


def _build_config(data) -> ExperimentConfig:
    data = dict(data)
    embedded = data.pop("embedded", None)
    method = data.pop("method", None)
    if method is not None:
        methods = {
            label: (solver, adjoint) for label, solver, adjoint in COMPARISON_METHODS
        }
        methods["cf_cosine_ees_p5_adaptive_reversible"] = (
            Solvers.CF_COSINE_EES,
            "reversible",
        )
        if method not in methods:
            raise ValueError(f"Unknown Kuramoto method: {method!r}")
        data["solver"], data["adjoint"] = methods[method]
        data["adaptive"] = method == "cf_cosine_ees_p5_adaptive_reversible"
    if "device" in data:
        data["device"] = Devices(data["device"])
    if "solver" in data:
        data["solver"] = Solvers(data["solver"])
    if "model" in data:
        data["model"] = ModelKind(data["model"])
    if "loss_horizons" in data:
        data["loss_horizons"] = tuple(float(x) for x in data["loss_horizons"])
    config = ExperimentConfig(**data)
    if embedded is not None and (
        type(embedded) is not bool or embedded != config.embedded
    ):
        raise ValueError(
            "embedded must match whether cosine EES uses adaptive stepping."
        )
    return config


def serialize_config(config: ExperimentConfig) -> str:
    """JSON-serialise the config (StrEnums become strings, tuples become lists)."""
    payload = asdict(config)
    payload["embedded"] = config.embedded
    for k, v in payload.items():
        if isinstance(v, StrEnum):
            payload[k] = v.value
        elif isinstance(v, tuple):
            payload[k] = list(v)
    return json.dumps(payload, indent=2, sort_keys=True)
