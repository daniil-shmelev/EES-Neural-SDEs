"""Smoke tests for the Kuramoto NSDE and Euclidean baseline.

Constructs each model at small N, runs a single forward pass, and checks
output shapes and torus-wrap conventions. Skipped if the JAX environment
isn't fully wired (e.g. `georax` missing on the dev box).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

try:
    from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE
    from experiments.kuramoto.models.euclidean_baseline import EuclideanKuramotoNSDE
    HAVE_GEORAX = True
except Exception:
    HAVE_GEORAX = False


pytestmark = pytest.mark.skipif(not HAVE_GEORAX, reason="JAX/georax stack unavailable.")


def test_kuramoto_nsde_forward_shape():
    model = KuramotoNSDE(
        N=2, hidden_dim=16, n_steps=8, dt=0.1, n_obs=4,
        diffusion_scale=0.1, key=jr.PRNGKey(0),
    )
    theta0 = jnp.array([0.3, -0.2])
    omega0 = jnp.array([0.1, 0.0])
    theta, omega = model(theta0, omega0, jr.PRNGKey(1))
    assert theta.shape == (4, 2)
    assert omega.shape == (4, 2)
    assert jnp.all(jnp.abs(theta) <= jnp.pi + 1e-6)


def test_euclidean_baseline_forward_shape():
    model = EuclideanKuramotoNSDE(
        N=2, hidden_dim=16, n_steps=8, dt=0.1, n_obs=4,
        diffusion_scale=0.1, key=jr.PRNGKey(0),
    )
    theta0 = jnp.array([0.3, -0.2])
    omega0 = jnp.array([0.1, 0.0])
    theta, omega = model(theta0, omega0, jr.PRNGKey(1))
    assert theta.shape == (4, 2)
    assert omega.shape == (4, 2)
    assert jnp.all(jnp.abs(theta) <= jnp.pi + 1e-6)
