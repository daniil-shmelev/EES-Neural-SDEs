"""Smoke test for the indexed mean-field Kuramoto NSDE.

Constructs each model at small N, runs a single forward pass, and checks
output shapes and torus-wrap conventions. Skipped if the JAX environment
isn't fully wired (e.g. `georax` missing on the dev box).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

try:
    from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE

    HAVE_GEORAX = True
except Exception:
    HAVE_GEORAX = False


pytestmark = pytest.mark.skipif(not HAVE_GEORAX, reason="JAX/georax stack unavailable.")


def test_kuramoto_nsde_forward_shape():
    model = KuramotoNSDE(
        N=2,
        hidden_dim=16,
        n_steps=8,
        dt=0.1,
        n_obs=4,
        diffusion_scale=0.1,
        key=jr.PRNGKey(0),
    )
    theta0 = jnp.array([0.3, -0.2])
    omega0 = jnp.array([0.1, 0.0])
    theta, omega = model(theta0, omega0, jr.PRNGKey(1))
    assert theta.shape == (4, 2)
    assert omega.shape == (4, 2)
    assert jnp.all(jnp.abs(theta) <= jnp.pi + 1e-6)


def test_diffusion_operator_matches_dense_product_and_gradient():
    """The matrix-free path preserves the original diffusion and its VJP."""
    from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE, state_features

    model = KuramotoNSDE(N=3, hidden_dim=8, key=jax.random.key(21))
    y = jnp.linspace(-0.3, 0.5, 6)
    dw = jnp.array([0.2, -0.1, 0.3])

    def dense(field):
        scales = (
            jax.nn.softplus(field.mlp(state_features(y, 3))) * field.diffusion_scale
        )
        matrix = jnp.concatenate([jnp.zeros((3, 3)), jnp.diag(scales)])
        return matrix @ dw

    def sparse(field):
        return field(0.0, y, None).mv(dw)

    np.testing.assert_allclose(
        sparse(model.diffusion_field), dense(model.diffusion_field), rtol=1e-6
    )
    grad_dense = eqx.filter_grad(lambda f: jnp.sum(dense(f) ** 2))(
        model.diffusion_field
    )
    grad_sparse = eqx.filter_grad(lambda f: jnp.sum(sparse(f) ** 2))(
        model.diffusion_field
    )
    for actual, expected in zip(
        jax.tree.leaves(grad_sparse), jax.tree.leaves(grad_dense), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("adaptive", [False, True])
def test_saving_loss_horizons_preserves_forecasts_and_gradients(adaptive):
    model = KuramotoNSDE(
        N=3,
        hidden_dim=8,
        n_steps=8,
        dt=0.05,
        n_obs=20,
        adaptive=adaptive,
        key=jax.random.key(2),
    )
    theta, omega = jnp.array([0.1, 0.2, 0.3]), jnp.array([0.2, -0.1, 0.1])
    idx = jnp.array([2, 4, 9, 19])
    key = jax.random.key(3)

    def loss(m, selected):
        th, om = m(theta, omega, key, save_indices=idx if selected else None)
        if not selected:
            th, om = th[idx], om[idx]
        return jnp.sum(jnp.cos(th) + om**2)

    run = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    expected, expected_grad = run(model, False)
    actual, actual_grad = run(model, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    for a, e in zip(
        jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True
    ):
        np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-7)
