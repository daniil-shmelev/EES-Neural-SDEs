"""Check shared training/benchmark semantics and prevent legacy data mixing."""
import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.kuramoto.experiment.config import COMPARISON_METHODS, DEFAULT_CONFIG, PROTOCOL, ExperimentConfig, load_config, serialize_config
from experiments.kuramoto.experiment.benchmark import prepare_benchmark
from experiments.kuramoto.experiment.factories import make_model
from experiments.kuramoto.experiment.losses import make_multi_horizon_energy_score
from experiments.kuramoto.scripts.plot_memory_sweep import read_cells


def test_shared_model_and_loss(tmp_path):
    # Small real-format cached dataset, so this test exercises the complete
    # benchmark preparation without running a training experiment.
    config = dataclasses.replace(load_config(DEFAULT_CONFIG), N=3, batch_size=2, hidden_dim=8, n_steps=4, data_dir=str(tmp_path))
    theta = np.linspace(-0.2, 0.3, 2*6*3, dtype=np.float32).reshape(2, 6, 3)
    omega = theta / 2
    path = tmp_path/'kuramoto_N3_seed0.npz'
    np.savez(path, theta_train=theta, omega_train=omega, t_grid=np.linspace(0, .4, 6, dtype=np.float32))
    path.with_suffix('.json').write_text(json.dumps({'N':3, 'T':.4, 'n_obs':6}))
    model, batch, measured_loss, metadata = prepare_benchmark(config)
    training_model = make_model(config, metadata, jax.random.key(config.seed))
    for a,b in zip(jax.tree.leaves(eqx.filter(model,eqx.is_array)),jax.tree.leaves(eqx.filter(training_model,eqx.is_array)),strict=True):
        np.testing.assert_array_equal(a,b)
    training_loss = make_multi_horizon_energy_score(horizons=config.loss_horizons,n_samples=config.loss_n_samples)
    key = jax.random.key(3)
    np.testing.assert_array_equal(measured_loss(model,batch,jnp.ones(2),key),training_loss(training_model,batch,jnp.ones(2),key))
    assert metadata['n_obs'] == 6


def test_plot_rejects_legacy_and_mixed_experiments(tmp_path):
    path=tmp_path/'memory.json'
    path.write_text(json.dumps({'cells':[]}))
    with pytest.raises(ValueError,match='unified-model'):
        read_cells(path)
    config=json.loads(serialize_config(load_config(DEFAULT_CONFIG)))
    cell=dict(protocol=PROTOCOL,config=config,solver='cfees25',adjoint='reversible',n_steps=50,peak_bytes=100)
    other={**cell,'config':{**config,'loss_n_samples':1}}
    path.write_text(json.dumps(dict(protocol=PROTOCOL,cells=[cell,other])))
    with pytest.raises(ValueError,match='different model, data or loss'):
        read_cells(path)


@pytest.mark.parametrize("label,solver,adjoint,steps", COMPARISON_METHODS)
def test_only_the_three_training_methods(label, solver, adjoint, steps):
    config = ExperimentConfig(solver=solver, adjoint=adjoint, n_steps=steps)
    assert config.solver == solver and config.adjoint == adjoint


@pytest.mark.parametrize("solver,adjoint", [
    ("cfees25", "checkpoint_full"), ("cfees25", "checkpoint_recursive"),
    ("cfees25", "reversible_checkpointed"), ("cfees25", "direct"),
    ("cfees25", "auto"), ("cg2", "reversible"),
    ("cfees27", "reversible"), ("cg4", "checkpoint_full"),
])
def test_reject_other_training_methods(solver, adjoint):
    with pytest.raises(ValueError, match="supports only"):
        ExperimentConfig(solver=solver, adjoint=adjoint)


def test_plot_rejects_excluded_method(tmp_path):
    config = json.loads(serialize_config(load_config(DEFAULT_CONFIG)))
    config["adjoint"] = "checkpoint_full"
    cell = dict(protocol=PROTOCOL, config=config, solver="cfees25", adjoint="checkpoint_full", n_steps=50, peak_bytes=100)
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(dict(protocol=PROTOCOL, cells=[cell])))
    with pytest.raises(ValueError, match="supports only"):
        read_cells(path)
