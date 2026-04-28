import json
import tomllib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import seali


class Experiments(StrEnum):
    SPD = "spd"
    RNA = "rna"


class Devices(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class Solvers(StrEnum):
    CFEES25 = "cfees25"
    CFEES27 = "cfees27"
    CG2 = "cg2"
    CG4 = "cg4"
    RKMK = "rkmk"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: Experiments
    epochs: int
    batch_size: int
    learning_rate: float
    loss_beta: float
    seed: int
    device: Devices
    hidden_dim: int
    ctx_dim: int
    n_steps: int
    dt: float
    solver: Solvers
    diffusion_scale: float
    min_eigenvalue: float
    context_length: int = 20
    residues_per_state: int = 1
    future_bases_window: int = 0
    future_ctx_dim: int = 32
    activation: str = "silu"
    drift_depth: int = 3
    diffusion_depth: int = 2
    filter_canonical: bool = False
    canonical_threshold: float = 1.0
    max_chains: int | None = None
    skip_plots: bool = False
    adjoint: str = "auto"
    """Diffrax adjoint: auto | reversible | direct | checkpoint_recursive
    (aka checkpoint_log) | checkpoint_full.  ``auto`` preserves the historical
    behaviour of picking ReversibleAdjoint for reversible solvers and
    DirectAdjoint otherwise."""


def make_config(
    *,
    experiment: Experiments = Experiments.SPD,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    loss_beta: float = 0.0,
    seed: int = 0,
    device: Devices = Devices.GPU,
    hidden_dim: int = 128,
    ctx_dim: int = 64,
    n_steps: int = 5,
    dt: float = 0.2,
    solver: Solvers = Solvers.CFEES25,
    diffusion_scale: float = 1.0,
    min_eigenvalue: float = 1e-6,
    context_length: int = 20,
    residues_per_state: int = 1,
    future_bases_window: int = 0,
    future_ctx_dim: int = 32,
    activation: str = "silu",
    drift_depth: int = 3,
    diffusion_depth: int = 2,
    filter_canonical: bool = False,
    canonical_threshold: float = 1.0,
    max_chains: int | None = None,
    skip_plots: bool = False,
    adjoint: str = "auto",
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=experiment,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_beta=loss_beta,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        ctx_dim=ctx_dim,
        n_steps=n_steps,
        dt=dt,
        solver=solver,
        diffusion_scale=diffusion_scale,
        min_eigenvalue=min_eigenvalue,
        context_length=context_length,
        residues_per_state=residues_per_state,
        future_bases_window=future_bases_window,
        future_ctx_dim=future_ctx_dim,
        activation=activation,
        drift_depth=drift_depth,
        diffusion_depth=diffusion_depth,
        filter_canonical=filter_canonical,
        canonical_threshold=canonical_threshold,
        max_chains=max_chains,
        skip_plots=skip_plots,
        adjoint=adjoint,
    )


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    raw_max_chains = data.get("max_chains")
    return ExperimentConfig(
        experiment=Experiments(data["experiment"]),
        device=Devices(data["device"]),
        solver=Solvers(data["solver"]),
        epochs=data["epochs"],
        batch_size=data["batch_size"],
        learning_rate=data["learning_rate"],
        loss_beta=data["loss_beta"],
        seed=data["seed"],
        hidden_dim=data["hidden_dim"],
        ctx_dim=data["ctx_dim"],
        n_steps=data["n_steps"],
        dt=data["dt"],
        diffusion_scale=data["diffusion_scale"],
        min_eigenvalue=data["min_eigenvalue"],
        context_length=data.get("context_length", 20),
        residues_per_state=data.get("residues_per_state", 1),
        future_bases_window=data.get("future_bases_window", 0),
        future_ctx_dim=data.get("future_ctx_dim", 32),
        activation=data.get("activation", "silu"),
        drift_depth=data.get("drift_depth", 3),
        diffusion_depth=data.get("diffusion_depth", 2),
        filter_canonical=data.get("filter_canonical", False),
        canonical_threshold=data.get("canonical_threshold", 1.0),
        max_chains=None if raw_max_chains in (None, 0, -1) else int(raw_max_chains),
        skip_plots=data.get("skip_plots", False),
        adjoint=data.get("adjoint", "auto"),
    )


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
        "loss_beta": "Weight on the auxiliary loss term.",
        "seed": "Random seed.",
        "device": "Preferred runtime device.",
        "hidden_dim": "Hidden dimension of the neural SDE.",
        "ctx_dim": "Context dimension.",
        "n_steps": "Number of SDE integration steps.",
        "dt": "SDE time step size.",
        "solver": "SDE solver.",
        "diffusion_scale": "Scale of the diffusion coefficient.",
        "min_eigenvalue": "Minimum eigenvalue clamp for SPD projections.",
        "skip_plots": "Skip saving diagnostic plots.",
        "output": "Optional path to write the JSON config to.",
    },
    option_prompts={
        "experiment": "experiment",
        "epochs": "int",
        "batch_size": "int",
        "learning_rate": "float",
        "loss_beta": "float",
        "seed": "int",
        "device": "device",
        "hidden_dim": "int",
        "ctx_dim": "int",
        "n_steps": "int",
        "dt": "float",
        "solver": "solver",
        "diffusion_scale": "float",
        "min_eigenvalue": "float",
        "skip_plots": "flag",
        "output": "path",
    },
)


@seali.command(help=HELP)
def main(
    *,
    experiment: Experiments = Experiments.SPD,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    loss_beta: float = 0.0,
    seed: int = 0,
    device: Devices = Devices.GPU,
    hidden_dim: int = 128,
    ctx_dim: int = 64,
    n_steps: int = 5,
    dt: float = 0.2,
    solver: Solvers = Solvers.CFEES25,
    diffusion_scale: float = 1.0,
    min_eigenvalue: float = 1e-6,
    skip_plots: bool = False,
    output: Path | None = None,
):
    """Build an experiment config and print it as JSON."""
    config = make_config(
        experiment=experiment,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_beta=loss_beta,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        ctx_dim=ctx_dim,
        n_steps=n_steps,
        dt=dt,
        solver=solver,
        diffusion_scale=diffusion_scale,
        min_eigenvalue=min_eigenvalue,
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
