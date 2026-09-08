"""Exercise the solver paths introduced by paper unification on small systems."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax_lowstorage import EES25

from experiments.sphere_latent_nsde.models import (
    PathEncoder,
    build_adjoint,
    build_solver,
)
from experiments.stochastic_volatility.models.nsde import SimpleNeuralSDE


@pytest.mark.parametrize(
    "name,adjoint_name",
    [
        ("geometric_euler", "checkpoint_full"),
        ("cg2", "checkpoint_recursive"),
        ("cfees25", "checkpoint_full"),
        ("cfees25", "reversible"),
        ("srkmk_general_shark", "checkpoint_full"),
    ],
)
def test_sphere_paths_and_gradients(name, adjoint_name):
    encoder = PathEncoder(
        h_dim=4, z_dim=3, n_deg=2, learnable_prior=False,
        time_min=0.0, time_max=1.0, key=jax.random.key(1),
    )
    solver = build_solver(name)
    adjoint = build_adjoint(solver, adjoint_name, max_steps=16)
    times = jnp.linspace(0.0, 0.1, 4)
    z0 = jnp.array([[[1.0, 0.0, 0.0]]])

    def solve(h):
        return encoder.integrate_paths(
            h, z0, jax.random.key(2), times=times,
            solver=solver, adjoint=adjoint,
        )

    h = jnp.ones((1, 4)) * 0.1
    paths = solve(h)
    np.testing.assert_allclose(jnp.linalg.norm(paths, axis=-1), 1.0, atol=2e-5)
    grad = jax.grad(lambda x: jnp.sum(solve(x)[..., 1]))(h)
    assert np.isfinite(np.asarray(grad)).all()
    assert np.linalg.norm(np.asarray(grad)) > 0


@pytest.mark.parametrize("step_size", [None, 0.07])
def test_volatility_short_solve_and_reverse_gradient(step_size):
    model = SimpleNeuralSDE(
        state_dim=1, hidden_dim=4, n_steps=6, dt=0.05,
        solver=EES25(), diffusion_scale=0.1, n_save=4, save_path=True,
        step_size=step_size, key=jax.random.key(3),
    )
    paths = model(jnp.ones(1), jax.random.key(4))
    assert paths.shape == (4, 1)
    assert np.isfinite(np.asarray(paths)).all()
    grads = eqx.filter_grad(lambda m: jnp.sum(m(jnp.ones(1), jax.random.key(4))))(model)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))
    assert model.solve_n_steps == (2 if step_size is None else 5)


def test_full_checkpoint_adjoint_requires_step_bound():
    with pytest.raises(ValueError, match="max_steps"):
        build_adjoint(build_solver("cg2"), "checkpoint_full")
    adjoint = build_adjoint(build_solver("cg2"), "checkpoint_full", max_steps=8)
    assert isinstance(adjoint, diffrax.RecursiveCheckpointAdjoint)


def test_torus_model_has_nonzero_parameter_gradient():
    from experiments.torus.models import TorusNeuralSDE

    model = TorusNeuralSDE(
        num_angles=2, hidden_dim=4, ctx_dim=8, n_steps=2, dt=0.05,
        key=jax.random.key(5),
    )
    angles = jnp.array([[0.1, -0.2], [0.2, -0.1]])
    labels = jnp.zeros((2, 1), dtype=jnp.int32)
    target_labels = jnp.zeros((1,), dtype=jnp.int32)
    loss = lambda m: jnp.sum(m(angles, labels, target_labels, jax.random.key(6)))
    value, grad = eqx.filter_value_and_grad(loss)(model)
    assert np.isfinite(float(value))
    leaves = jax.tree.leaves(grad.vector_field)
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    assert sum(float(jnp.sum(x**2)) for x in leaves) > 0


def test_kuramoto_has_nonzero_parameter_gradient():
    from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE

    model = KuramotoNSDE(
        N=2, hidden_dim=4, n_steps=2, dt=0.05, n_obs=3,
        diffusion_scale=0.1, key=jax.random.key(7),
    )
    loss = lambda m: jnp.sum(m(jnp.array([0.1, -0.2]), jnp.zeros(2), jax.random.key(8))[1])
    value, grad = eqx.filter_value_and_grad(loss)(model)
    assert np.isfinite(float(value))
    leaves = jax.tree.leaves(grad.drift_field)
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    assert sum(float(jnp.sum(x**2)) for x in leaves) > 0
