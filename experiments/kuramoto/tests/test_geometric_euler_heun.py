"""Check Stratonovich Euler–Heun updates and training gradients through Georax."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import VirtualBrownianTree
from georax import GeometricEulerHeun

from ees_core.torus import wrap_to_pi
from experiments.kuramoto.experiment.config import ExperimentConfig, Solvers
from experiments.kuramoto.experiment.factories import make_model
from experiments.kuramoto.experiment.losses import make_multi_horizon_energy_score


def _config(**overrides):
    return ExperimentConfig(
        solver=Solvers.GEOMETRIC_EULER_HEUN,
        N=3,
        hidden_dim=8,
        drift_depth=1,
        diffusion_depth=1,
        **overrides,
    )


def test_geometric_euler_heun_matches_wrapped_euler_heun_step():
    config = _config(adjoint="checkpoint_full", n_steps=1, diffusion_scale=1.0)
    model = make_model(config, {"N": 3, "n_obs": 2, "T": 0.1}, jax.random.key(0))
    assert type(model.solver) is GeometricEulerHeun
    model = eqx.tree_at(
        lambda m: m.diffusion_field.mlp.layers[-1].bias, model, jnp.zeros(3)
    )
    theta0 = jnp.array([3.13, -3.13, 0.4])
    omega0 = jnp.array([0.4, -0.4, 0.1])
    y0 = jnp.concatenate([theta0, omega0])
    key = jax.random.key(1)
    brownian = VirtualBrownianTree(0.0, 0.1, tol=0.025, shape=(3,), key=key)
    dw = brownian.evaluate(0.0, 0.1)
    g0 = model.diffusion_field(0.0, y0, None).mv(dw)
    predictor = y0 + g0
    predictor = predictor.at[:3].set(wrap_to_pi(predictor[:3]))
    g1 = model.diffusion_field(0.0, predictor, None).mv(dw)
    # Ensure the predictor makes this test distinguish Euler–Heun from Euler.
    assert jnp.max(jnp.abs(g1 - g0)) > 1e-5
    expected = y0 + 0.1 * model.drift_field(0.0, y0, None) + 0.5 * (g0 + g1)
    expected = expected.at[:3].set(wrap_to_pi(expected[:3]))
    theta, omega = eqx.filter_jit(lambda m: m(theta0, omega0, key))(model)
    np.testing.assert_allclose(theta[-1], expected[:3], atol=1e-6)
    np.testing.assert_allclose(omega[-1], expected[3:], atol=1e-6)
    assert theta[-1, 0] < 0 and theta[-1, 1] > 0


def test_geometric_euler_heun_energy_score_checkpoint_gradients_agree():
    config = _config(adjoint="checkpoint_full", n_steps=150)
    metadata = {"N": 3, "n_obs": 5, "T": 0.4}
    full = make_model(config, metadata, jax.random.key(2))
    recursive = make_model(
        replace(config, adjoint="checkpoint_recursive"), metadata, jax.random.key(2)
    )
    theta = jnp.linspace(-0.3, 0.4, 2 * 5 * 3).reshape(2, 5, 3)
    omega = theta / 2
    batch = {
        "theta0": theta[:, 0],
        "omega0": omega[:, 0],
        "theta_traj": theta,
        "omega_traj": omega,
    }
    loss = make_multi_horizon_energy_score(horizons=(0.5, 1.0), n_samples=2)
    value_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    value, gradient = value_grad(full, batch, jnp.ones(2), jax.random.key(3))
    other_value, other_gradient = value_grad(
        recursive, batch, jnp.ones(2), jax.random.key(3)
    )
    assert np.isfinite(value)
    np.testing.assert_array_equal(value, other_value)
    for actual, expected in zip(
        jax.tree.leaves(other_gradient), jax.tree.leaves(gradient), strict=True
    ):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)
    for field in (gradient.drift_field, gradient.diffusion_field):
        assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(field))
