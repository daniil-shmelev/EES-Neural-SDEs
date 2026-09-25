"""Second-order structure and checkpoint-free gradients on learned forcing."""

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import RecursiveCheckpointAdjoint, ReversibleAdjoint
from georax import CFEES25

from experiments.kuramoto.experiment.config import ExperimentConfig
from experiments.kuramoto.experiment.factories import make_model
from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE


def _model():
    model = KuramotoNSDE(
        N=3, hidden_dim=8, drift_depth=1, diffusion_depth=1,
        drift_kind='second_order_mean_field', adaptive=True,
        n_steps=10, dt=.5, n_obs=100, adaptive_atol=.01,
        adaptive_dtmin=1e-4, adaptive_max_steps=256,
        adjoint=ReversibleAdjoint(), key=jax.random.key(0),
    )
    last = model.drift_field.rho_node.layers[-1]
    return eqx.tree_at(
        lambda m: (m.drift_field.rho_node.layers[-1].weight,
                   m.drift_field.rho_node.layers[-1].bias),
        model, (.1 * jax.random.normal(jax.random.key(1), last.weight.shape),
                jnp.array([0., .7])),
    )


def test_phase_kinematics_and_linear_bounded_damping():
    field = _model().drift_field
    theta = jnp.array([3.1, -.2, .4])
    velocity = jnp.array([.3, -.4, .2])
    shift = jnp.array([.4, .1, -.3])
    first = field(0., jnp.concatenate([theta, velocity]), None)
    second = field(0., jnp.concatenate([theta, velocity + shift]), None)
    np.testing.assert_array_equal(first[:3], velocity)
    np.testing.assert_allclose(second[3:] - first[3:], -shift, atol=2e-7)
    shifted_phase = field(0., jnp.concatenate([theta + 2*jnp.pi, velocity]), None)
    np.testing.assert_allclose(shifted_phase, first, atol=2e-6)
    for raw in [-100., 100.]:
        bounded = eqx.tree_at(lambda f: f.damping_logit, field, jnp.array(raw))
        damping = 2 * jax.nn.sigmoid(bounded.damping_logit)
        assert 0 <= damping <= 2
        a = bounded(0., jnp.concatenate([theta, velocity]), None)
        b = bounded(0., jnp.concatenate([theta, velocity + shift]), None)
        np.testing.assert_allclose(b[3:] - a[3:], -damping * shift, atol=2e-7)


def test_structured_adaptive_reconstruction_gradient():
    model = _model()
    reference = copy.copy(model)
    object.__setattr__(reference, 'adjoint', RecursiveCheckpointAdjoint())

    @eqx.filter_jit
    def value_gradient(m):
        def loss(m):
            theta, omega = m(jnp.array([3.1, -.3, .4]),
                             jnp.array([.3, -.2, .1]), jax.random.key(2),
                             save_indices=jnp.array([12, 24, 49, 99]))
            return jnp.mean(jnp.cos(theta) + omega**2)
        return eqx.filter_value_and_grad(loss)(m)

    value, gradient = value_gradient(model)
    reference_value, reference_gradient = value_gradient(reference)
    np.testing.assert_array_equal(value, reference_value)
    flat = lambda tree: np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])
    expected = flat(reference_gradient)
    error = np.linalg.norm(flat(gradient) - expected) / np.linalg.norm(expected)
    assert error < 2e-3


def test_active_model_keeps_ordinary_reversible_adjoint():
    config = ExperimentConfig(adaptive=True, N=3, hidden_dim=8)
    model = make_model(config, {'N': 3, 'n_obs': 100, 'T': 5.}, jax.random.key(0))
    assert type(model.solver) is CFEES25
    assert type(model.adjoint) is ReversibleAdjoint
    assert hasattr(model.drift_field, 'damping_logit')
