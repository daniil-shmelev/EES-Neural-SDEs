import itertools
import json
import tomllib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import seali


class Experiments(StrEnum):
    BLACK_SCHOLES = "black_scholes"
    LOCAL_STOCH_VOL = "local_stoch_vol"
    HESTON = "heston"
    ROUGH_HESTON = "rough_heston"
    QUADRATIC_ROUGH_HESTON = "quadratic_rough_heston"
    BERGOMI = "bergomi"
    ROUGH_BERGOMI = "rough_bergomi"


class Devices(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class Solvers(StrEnum):
    EES25 = "ees25"
    MCF_EULER = "mcf_euler"
    MCF_MIDPOINT = "mcf_midpoint"
    REVERSIBLE_HEUN = "reversible_heun"

    GL2 = "gl2"
    CFEES25 = "cfees25"

    def build(self):
        from diffrax import Euler, Midpoint, ReversibleHeun
        from diffrax_lowstorage import EES25
        from georax import CFEES25, CG2

        return {
            "ees25": EES25,
            "mcf_euler": Euler,
            "mcf_midpoint": Midpoint,
            "reversible_heun": ReversibleHeun,
            "gl2": CG2,
            "cfees25": CFEES25,
        }[self]()


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: Experiments
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    device: Devices
    # model hyperparams
    hidden_dim: int
    nfe_budget: int
    total_time: float
    solver: Solvers
    diffusion_scale: float
    # output
    skip_plots: bool

    @property
    def n_steps(self) -> int:
        return self.nfe_budget

    @property
    def dt(self) -> float:
        return self.total_time / self.nfe_budget


def make_config(
    *,
    experiment: Experiments,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: Devices,
    hidden_dim: int,
    nfe_budget: int,
    total_time: float,
    solver: Solvers,
    diffusion_scale: float,
    skip_plots: bool,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=experiment,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        nfe_budget=nfe_budget,
        total_time=total_time,
        solver=solver,
        diffusion_scale=diffusion_scale,
        skip_plots=skip_plots,
    )


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return _build_config(data)


def is_sweep_config(path: Path) -> bool:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return "base" in data and "sweep" in data


def _build_config(data: dict[str, Any]) -> ExperimentConfig:
    nfe_budget = data.get("nfe_budget", data.get("n_steps"))
    if nfe_budget is None:
        raise KeyError("Config must define `nfe_budget` (or legacy `n_steps`).")

    total_time = data.get("total_time")
    if total_time is None:
        dt = data.get("dt")
        if dt is None:
            raise KeyError("Config must define `total_time` (or legacy `dt`).")
        total_time = nfe_budget * dt

    return ExperimentConfig(
        experiment=Experiments(data["experiment"]),
        device=Devices(data["device"]),
        solver=Solvers(data["solver"]),
        epochs=data["epochs"],
        batch_size=data["batch_size"],
        learning_rate=data["learning_rate"],
        seed=data["seed"],
        hidden_dim=data["hidden_dim"],
        nfe_budget=nfe_budget,
        total_time=total_time,
        diffusion_scale=data["diffusion_scale"],
        skip_plots=data["skip_plots"],
    )


def load_sweep(path: Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data["base"], data["sweep"]


def sweep_len(path: Path) -> int:
    _, sweep_axes = load_sweep(path)
    return sum(1 for _ in itertools.product(*sweep_axes.values()))


def sweep_config_at(path: Path, index: int) -> ExperimentConfig:
    base, sweep_axes = load_sweep(path)
    keys = list(sweep_axes.keys())
    combos = list(itertools.product(*sweep_axes.values()))
    if index < 0 or index >= len(combos):
        raise IndexError(f"sweep index {index} out of range [0, {len(combos)})")
    overrides = dict(zip(keys, combos[index]))
    return _build_config({**base, **overrides})


def _serialize_config(config: ExperimentConfig) -> str:
    return json.dumps(asdict(config), indent=2, sort_keys=True)


HELP = seali.Help(
    help="""
    Build an experiment config and print it as JSON.

    $USAGE

    $OPTIONS_AND_FLAGS
    """,
    style=seali.Style(heading=seali.BOLD),
    arguments={
        "experiment": "Experiment preset to encode into the config.",
        "epochs": "Number of training epochs.",
        "batch_size": "Mini-batch size.",
        "learning_rate": "Optimizer learning rate.",
        "seed": "Random seed.",
        "device": "Preferred runtime device.",
        "hidden_dim": "Hidden dimension of the neural SDE.",
        "nfe_budget": "Fixed forward NFE budget.",
        "total_time": "Total SDE integration time horizon.",
        "solver": "SDE solver.",
        "diffusion_scale": "Scale of the diffusion coefficient.",
        "skip_plots": "Skip saving diagnostic plots.",
        "output": "Optional path to write the JSON config to.",
    },
    option_prompts={
        "experiment": "experiment",
        "epochs": "int",
        "batch_size": "int",
        "learning_rate": "float",
        "seed": "int",
        "device": "device",
        "hidden_dim": "int",
        "nfe_budget": "int",
        "total_time": "float",
        "solver": "solver",
        "diffusion_scale": "float",
        "skip_plots": "flag",
        "output": "path",
    },
)


@seali.command(help=HELP)
def main(
    *,
    experiment: Experiments = Experiments.BLACK_SCHOLES,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: Devices = Devices.GPU,
    hidden_dim: int = 128,
    nfe_budget: int = 5,
    total_time: float = 1.0,
    solver: Solvers = Solvers.CFEES25,
    diffusion_scale: float = 1.0,
    skip_plots: bool = False,
    output: Path | None = None,
):
    """Build an experiment config and print it as JSON."""
    config = make_config(
        experiment=experiment,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        nfe_budget=nfe_budget,
        total_time=total_time,
        solver=solver,
        diffusion_scale=diffusion_scale,
        skip_plots=skip_plots,
    )
    payload = _serialize_config(config)

    if output is None:
        print(payload)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
