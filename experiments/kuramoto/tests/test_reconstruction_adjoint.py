"""Check reversible reconstruction against a reference gradient on a small model."""
from __future__ import annotations

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import RecursiveCheckpointAdjoint, ReversibleAdjoint

from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE


def test_reconstruction_gradient_matches_reference():
    model = KuramotoNSDE(
        N=3, hidden_dim=8, n_steps=50, dt=0.1, n_obs=100,
        drift_depth=1, diffusion_depth=1, diffusion_scale=0.1,
        adjoint=RecursiveCheckpointAdjoint(checkpoints=58),
        key=jax.random.key(0),
    )
    last = model.drift_field.rho_node.layers[-1]
    model = eqx.tree_at(
        lambda m: (m.drift_field.rho_node.layers[-1].weight,
                   m.drift_field.rho_node.layers[-1].bias),
        model,
        (0.02 * jax.random.normal(jax.random.key(1), last.weight.shape),
         jnp.array([0.3, 0.7])),
    )

    @eqx.filter_jit
    def value_and_gradient(m):
        def loss(m):
            theta, omega = m(jnp.array([0.2, -0.3, 0.4]),
                             jnp.array([0.1, -0.2, 0.3]), jax.random.key(2))
            return jnp.mean(jnp.cos(theta) + omega**2)
        return eqx.filter_value_and_grad(loss)(m)

    reference_value, reference_gradient = value_and_gradient(model)
    reconstructed = copy.copy(model)
    object.__setattr__(reconstructed, "adjoint", ReversibleAdjoint())
    value, gradient = value_and_gradient(reconstructed)
    np.testing.assert_array_equal(value, reference_value)
    flatten = lambda tree: np.concatenate([
        np.asarray(x).ravel() for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))
    ])
    reference = flatten(reference_gradient)
    relative_error = np.linalg.norm(flatten(gradient) - reference) / np.linalg.norm(reference)
    assert relative_error < 5e-5
