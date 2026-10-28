"""Adaptive sphere meshes, geometric error control and reconstruction gradients."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax import RecursiveCheckpointAdjoint, ReversibleAdjoint, SaveAt, StepTo
from georax import CFEES25

from experiments.sphere_latent_nsde.config import (
    ActivityConfig,
    load_config,
    sweep_config_at,
    sweep_len,
    to_toml,
)
from experiments.sphere_latent_nsde.factories import make_model
from experiments.sphere_latent_nsde.models import PathEncoder, sphere_error_norm


@pytest.mark.parametrize(
    "override,match",
    [
        ({"solver": "geometric_euler_heun"}, "embedded pair"),
        ({"solver": "cg2"}, "embedded pair"),
        ({"adaptive_atol": 0.0}, "adaptive_atol"),
        ({"adaptive_atol": float("nan")}, "adaptive_atol"),
        ({"adaptive_dtmin": 0.0}, "adaptive_dtmin"),
        ({"adaptive_dtmin": 1.0}, "adaptive_dtmin"),
        ({"adaptive_max_steps": 1}, "adaptive_max_steps"),
    ],
)
def test_invalid_adaptive_settings(override, match):
    settings = {"solver": "cfees25", "adaptive": True, "nfe_budget": 12, **override}
    with pytest.raises(ValueError, match=match):
        ActivityConfig(**settings)


def test_sweep_and_serialisation_keep_adaptive_distinct(tmp_path):
    path = Path(__file__).parent / "configs" / "activity" / "sweep.toml"
    configs = [sweep_config_at(path, i) for i in range(sweep_len(path))]
    adaptive = [c for c in configs if c.adaptive]
    assert len(adaptive) == 1
    config = adaptive[0]
    assert config.method == "cfees25_adaptive_reversible"
    assert any(c.method == "cfees25_reversible" and not c.adaptive for c in configs)
    output = tmp_path / "config.toml"
    output.write_text(to_toml(config))
    assert load_config(output) == config
    model = make_model(config, jax.random.key(0))
    assert model.adaptive and type(model.adjoint) is ReversibleAdjoint
    assert model.adaptive_atol == config.adaptive_atol
    assert model.solve_num_timepoints == config.solve_n_steps + 1


def test_sphere_error_norm_does_not_dilute_one_large_error():
    error = jnp.zeros((2, 3, 4)).at[1, 2, :2].set(jnp.array([0.03, 0.04]))
    np.testing.assert_allclose(sphere_error_norm(error), 0.05)
    np.testing.assert_allclose(sphere_error_norm(jnp.tile(error, (2, 1, 1))), 0.05)


def test_adaptive_rejections_replay_and_reversible_gradients():
    model = PathEncoder(h_dim=4, z_dim=4, n_deg=3, key=jax.random.key(0))
    model = eqx.tree_at(lambda m: m.sigma, model, jnp.array(0.4))
    h = jnp.array([[0.4, -0.2, 0.3, 0.1], [-0.1, 0.3, 0.2, 0.4]])
    z0 = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0]), (2, 2, 4))
    times = jnp.linspace(0.0, 0.99, 4)
    key = jax.random.key(1)

    def solve(m, adjoint, **kwargs):
        return m.solve_paths(
            h,
            z0,
            key,
            times=times,
            solver=CFEES25(),
            adjoint=adjoint,
            adaptive=True,
            adaptive_atol=1e-5,
            adaptive_dtmin=1e-4,
            adaptive_max_steps=256,
            **kwargs,
        )

    reference_adjoint = RecursiveCheckpointAdjoint()
    mesh = eqx.filter_jit(
        lambda m: solve(m, reference_adjoint, saveat=SaveAt(t0=True, steps=True))
    )(model)
    assert int(mesh.stats["num_rejected_steps"]) > 0
    accepted = mesh.ts[np.isfinite(np.asarray(mesh.ts))]
    assert np.ptp(np.diff(accepted)) > 1e-4
    np.testing.assert_allclose(accepted[-1], times[-1])
    assert np.min(np.abs(np.asarray(accepted) - float(times[1]))) < 1e-7

    reference = eqx.filter_jit(lambda m: solve(m, reference_adjoint))(model)
    replay = eqx.filter_jit(
        lambda m: solve(m, reference_adjoint, stepsize_controller=StepTo(ts=accepted))
    )(model)
    np.testing.assert_allclose(replay.ys, reference.ys, atol=2e-6, rtol=0)
    # Hundreds of float32 chart actions accumulate small radial roundoff.
    np.testing.assert_allclose(jnp.linalg.norm(reference.ys, axis=-1), 1.0, atol=2e-5)

    @eqx.filter_jit
    def value_grad(m, adjoint):
        def loss(m):
            sol = solve(m, adjoint)
            paths = jnp.moveaxis(sol.ys, 0, 2)
            return jnp.sum(paths[..., -1, 1]) + 0.01 * jnp.sum(
                m.path_kl(paths, h, times)
            )

        return eqx.filter_value_and_grad(loss)(m)

    value, gradient = value_grad(model, ReversibleAdjoint())
    expected_value, expected_gradient = value_grad(model, reference_adjoint)
    np.testing.assert_array_equal(value, expected_value)
    actual = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(gradient)])
    expected = np.concatenate(
        [np.asarray(x).ravel() for x in jax.tree.leaves(expected_gradient)]
    )
    assert np.isfinite(actual).all() and np.linalg.norm(expected) > 0
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-3
