"""Check training method support and prevent legacy result mixing."""

import json

import pytest

from experiments.kuramoto.experiment.config import (
    COMPARISON_METHODS,
    DEFAULT_CONFIG,
    PROTOCOL,
    ExperimentConfig,
    load_config,
    serialize_config,
)
from experiments.kuramoto.scripts.plot_memory_sweep import read_cells


def test_plot_rejects_legacy_and_mixed_experiments(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cells": []}))
    with pytest.raises(ValueError, match="unified-model"):
        read_cells(path)
    config = json.loads(serialize_config(load_config(DEFAULT_CONFIG)))
    cell = dict(
        protocol=PROTOCOL,
        config=config,
        solver="cfees25",
        adjoint="reversible",
        n_steps=50,
        peak_bytes=100,
    )
    other = {**cell, "config": {**config, "loss_n_samples": 1}}
    path.write_text(json.dumps(dict(protocol=PROTOCOL, cells=[cell, other])))
    with pytest.raises(ValueError, match="different model, data or loss"):
        read_cells(path)


@pytest.mark.parametrize("label,solver,adjoint", COMPARISON_METHODS)
def test_supported_training_methods(label, solver, adjoint):
    config = ExperimentConfig(solver=solver, adjoint=adjoint)
    assert config.solver == solver and config.adjoint == adjoint


@pytest.mark.parametrize(
    "solver,adjoint",
    [
        ("cfees25", "checkpoint_full"),
        ("cfees25", "checkpoint_recursive"),
        ("cfees25", "reversible_checkpointed"),
        ("cfees25", "direct"),
        ("cfees25", "auto"),
        ("cg2", "reversible"),
        ("geometric_euler_heun", "reversible"),
        ("geometric_euler", "checkpoint_full"),
        ("cfees27", "reversible"),
        ("cg4", "checkpoint_full"),
    ],
)
def test_reject_other_training_methods(solver, adjoint):
    with pytest.raises(ValueError, match="supports only"):
        ExperimentConfig(solver=solver, adjoint=adjoint)


def test_plot_rejects_excluded_method(tmp_path):
    config = json.loads(serialize_config(load_config(DEFAULT_CONFIG)))
    config["adjoint"] = "checkpoint_full"
    cell = dict(
        protocol=PROTOCOL,
        config=config,
        solver="cfees25",
        adjoint="checkpoint_full",
        n_steps=50,
        peak_bytes=100,
    )
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(dict(protocol=PROTOCOL, cells=[cell])))
    with pytest.raises(ValueError, match="supports only"):
        read_cells(path)
