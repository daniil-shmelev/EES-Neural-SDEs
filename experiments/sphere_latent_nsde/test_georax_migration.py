"""Exercise the library solver path for every reported sphere method."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.sphere_latent_nsde.config import ActivityConfig, sweep_config_at
from experiments.sphere_latent_nsde.models import (
    PathEncoder,
    build_adjoint,
    build_solver,
)


@pytest.mark.parametrize(
    "method,adjoint_name",
    [
        ("geometric_euler_heun", "direct"),
        ("geometric_euler_heun", "checkpoint_full"),
        ("geometric_euler_heun", "recursive_checkpoint"),
        ("cg2", "checkpoint_full"),
        ("cfees25", "reversible"),
        ("srkmk_general_shark", "checkpoint_full"),
    ],
)
def test_batched_paths_and_gradients(method, adjoint_name):
    model = PathEncoder(h_dim=4, z_dim=4, n_deg=2, key=jax.random.key(0))
    h = jnp.ones((2, 4)) * 0.1
    z0 = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0]), (2, 2, 4))
    times = jnp.linspace(0.0, 0.04, 3)
    solver = build_solver(method)
    adjoint = build_adjoint(solver, adjoint_name, max_steps=11)

    @eqx.filter_jit
    def evaluate(m):
        def loss(m):
            paths = m.integrate_paths(
                h, z0, jax.random.key(1), times=times, solver=solver, adjoint=adjoint
            )
            return jnp.sum(paths[..., -1, 1]) + jnp.sum(
                m.path_kl(paths, h, times)
            ), paths

        return eqx.filter_value_and_grad(loss, has_aux=True)(m)

    (value, paths), grads = evaluate(model)
    assert paths.shape == (2, 2, 3, 4)
    assert np.isfinite(value)
    np.testing.assert_allclose(jnp.linalg.norm(paths, axis=-1), 1.0, atol=2e-6)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    assert any(np.any(np.asarray(x) != 0) for x in leaves)


def test_euler_heun_sweep_uses_drift_budget_and_new_solver_name():
    from pathlib import Path

    from georax import GeometricEulerHeun

    config_path = Path(__file__).parent / "config.toml"
    for index in (0, 1):
        config = sweep_config_at(config_path, index)
        assert config.solver == "geometric_euler_heun"
        assert type(build_solver(config.solver)) is GeometricEulerHeun
        assert config.solve_n_steps == config.effective_nfe_budget
        assert config.diffusion_nfe_per_step == 2
    assert ActivityConfig().solver == "geometric_euler_heun"
    with pytest.raises(ValueError, match="unknown solver"):
        build_solver("geometric_euler")
