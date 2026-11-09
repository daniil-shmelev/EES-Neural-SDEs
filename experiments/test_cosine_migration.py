"""Cosine order, embedding and provenance across the active training factories."""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from georax import CFCosineEES

from experiments.kuramoto.experiment.config import (
    ExperimentConfig,
    Solvers,
    _build_config,
    serialize_config,
)
from experiments.kuramoto.experiment.factories import build_solver as kuramoto_solver
from experiments.sphere_latent_nsde.config import ActivityConfig, load_config, to_toml
from experiments.sphere_latent_nsde.factories import make_model
from experiments.sphere_latent_nsde.models import PathEncoder, build_adjoint


@pytest.mark.parametrize("order", [3, 5, 7])
@pytest.mark.parametrize("adaptive", [False, True])
def test_order_embedding_and_config_roundtrip(order, adaptive, tmp_path):
    sphere = ActivityConfig(
        solver="cf_cosine_ees",
        adjoint="reversible",
        reversible_order=order,
        n_steps=4,
        adaptive=adaptive,
        h_dim=4,
        z_dim=4,
        n_deg=2,
    )
    path = tmp_path / "sphere.toml"
    path.write_text(to_toml(sphere))
    assert load_config(path) == sphere
    assert f"embedded = {str(adaptive).lower()}" in path.read_text()
    assert sphere.nfe_per_step == (order + 1) // 2
    assert sphere.effective_nfe_budget == 4 * sphere.nfe_per_step
    assert sphere.solve_n_steps == 4
    assert f"_p{order}_" in sphere.method
    model = make_model(sphere, jax.random.key(0))
    assert isinstance(model.solver, CFCosineEES)
    assert model.solver.reversible_order == order
    assert model.solver.embedded == adaptive
    assert model.embedded_nfe_per_step == int(adaptive)

    kuramoto = ExperimentConfig(reversible_order=order, adaptive=adaptive)
    saved = json.loads(serialize_config(kuramoto))
    assert saved["embedded"] == adaptive
    assert _build_config(saved) == kuramoto
    solver = kuramoto_solver(
        Solvers.CF_COSINE_EES, reversible_order=order, adaptive=adaptive
    )
    assert solver.recurrence.num_stages == sphere.nfe_per_step
    assert solver.embedded == adaptive
    assert kuramoto.method == sphere.method


@pytest.mark.parametrize("order", [True, 1, 4, 5.0])
def test_invalid_cosine_orders(order):
    with pytest.raises(ValueError, match="odd integer"):
        ActivityConfig(solver="cf_cosine_ees", reversible_order=order)
    with pytest.raises(ValueError, match="odd integer"):
        ExperimentConfig(reversible_order=order)


def test_old_solver_names_are_not_reinterpreted():
    with pytest.raises(ValueError, match="Unsupported solver"):
        ActivityConfig(solver="cfees25")
    with pytest.raises(ValueError, match="supports only"):
        ExperimentConfig(solver="cfees25")


def test_adaptive_sphere_requires_enabled_companion():
    encoder = PathEncoder(h_dim=4, z_dim=4, n_deg=2, key=jax.random.key(0))
    solver = CFCosineEES(reversible_order=5)
    with pytest.raises(ValueError, match="embedded=True"):
        encoder.solve_paths(
            jnp.zeros((1, 4)),
            jnp.array([[[1.0, 0.0, 0.0, 0.0]]]),
            jax.random.key(1),
            times=jnp.linspace(0.0, 0.99, 4),
            solver=solver,
            adjoint=build_adjoint(solver),
            adaptive=True,
        )


def test_adaptive_work_includes_companion():
    config = ActivityConfig(
        solver="cf_cosine_ees",
        adjoint="reversible",
        reversible_order=7,
        adaptive=True,
        n_steps=3,
        num_timepoints=8,
        h_dim=4,
        z_dim=4,
        n_deg=2,
    )
    model = make_model(config, jax.random.key(0))
    batch = {
        "inp_obs": jnp.zeros((1, 4, 12)),
        "inp_msk": jnp.ones((1, 4, 12)),
        "inp_tps": jnp.linspace(0.0, 0.9, 4)[None],
    }

    @eqx.filter_jit
    def run(model):
        output = model(batch, jax.random.key(1))
        return output.paths, output.solver_stats

    paths, work = run(model)
    np.testing.assert_allclose(jnp.linalg.norm(paths, axis=-1), 1.0, atol=2e-6)
    assert work["forward_drift_evaluations"] == work["num_steps"] * 5
    assert work["forward_diffusion_evaluations"] == work["num_steps"] * 5
    assert work["embedded_generator_evaluations"] == work["num_steps"]


def test_saved_embedding_cannot_disable_adaptive_pair(tmp_path):
    config = ActivityConfig(solver="cf_cosine_ees", adaptive=True, n_steps=4)
    path = tmp_path / "bad.toml"
    path.write_text(to_toml(config).replace("embedded = true", "embedded = false"))
    with pytest.raises(ValueError, match="embedded must match"):
        load_config(path)
    saved = json.loads(serialize_config(ExperimentConfig(adaptive=True)))
    saved["embedded"] = False
    with pytest.raises(ValueError, match="embedded must match"):
        _build_config(saved)
