"""Adaptive CF-EES on the tangent torus and reconstruction on its accepted mesh."""
import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax import RecursiveCheckpointAdjoint, ReversibleAdjoint, SaveAt, StepTo

from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE, tangent_torus_error_norm
from experiments.kuramoto.experiment.config import ExperimentConfig, Solvers


def test_periodic_error_norm():
    atol = .01
    error = jnp.array([2*jnp.pi-.002, -2*jnp.pi+.003, .004, -.005])
    expected = np.sqrt(np.mean(np.square(np.array([-.002,.003,.004,-.005])/atol)))
    np.testing.assert_allclose(tangent_torus_error_norm(error, atol=atol), expected, rtol=3e-5)


def test_adaptive_configuration_rejects_solvers_without_embedded_pair():
    with pytest.raises(ValueError, match='embedded pair'):
        ExperimentConfig(adaptive=True, solver=Solvers.CG2)


def test_rejection_replay_and_reconstruction_gradients():
    model = KuramotoNSDE(
        N=3, hidden_dim=8, n_steps=10, dt=.2, n_obs=31,
        drift_depth=1, diffusion_depth=1, adaptive=True,
        adaptive_atol=2e-4, adaptive_dtmin=1e-5, adaptive_max_steps=512,
        adjoint=RecursiveCheckpointAdjoint(checkpoints=512),
        key=jax.random.key(0),
    )
    last = model.drift_field.rho_node.layers[-1]
    model = eqx.tree_at(
        lambda m: (m.drift_field.rho_node.layers[-1].weight,
                   m.drift_field.rho_node.layers[-1].bias), model,
        (.2 * jax.random.normal(jax.random.key(1), last.weight.shape), jnp.array([1.5, .7])),
    )
    theta0 = jnp.array([3.13, -.3, .4])
    omega0 = jnp.array([.1, -.2, .3])
    key = jax.random.key(2)

    mesh_solution = eqx.filter_jit(lambda m: m.solve(
        theta0, omega0, key, saveat=SaveAt(t0=True, steps=True)))(model)
    assert int(mesh_solution.stats['num_rejected_steps']) > 0
    times = mesh_solution.ts[np.isfinite(np.asarray(mesh_solution.ts))]
    steps = np.diff(times)
    assert np.ptp(steps) > .001
    assert steps.max() <= model.dt + 1e-6
    # Replay the accepted partition against the same Brownian tree. Rejections
    # must not replace the noise sampled for that trajectory.
    adaptive_solution = eqx.filter_jit(lambda m: m.solve(theta0, omega0, key))(model)
    replay = eqx.filter_jit(lambda m: m.solve(
        theta0, omega0, key, stepsize_controller=StepTo(ts=times)))(model)
    np.testing.assert_allclose(adaptive_solution.ys, replay.ys, atol=2e-6, rtol=0)

    @eqx.filter_jit
    def value_and_gradient(m):
        def loss(m):
            theta, omega = m(theta0, omega0, key)
            return jnp.mean(jnp.cos(theta) + omega**2)
        return eqx.filter_value_and_grad(loss)(m)

    value_full, grad_full = value_and_gradient(model)
    reconstructed = copy.copy(model)
    object.__setattr__(reconstructed, 'adjoint', ReversibleAdjoint())
    value_reverse, grad_reverse = value_and_gradient(reconstructed)
    np.testing.assert_array_equal(value_reverse, value_full)
    flat = lambda g: np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(g)])
    reference = flat(grad_full)
    error = np.linalg.norm(flat(grad_reverse)-reference)/np.linalg.norm(reference)
    assert error < 2e-4


def test_batched_adaptive_reconstruction_gradients():
    model = KuramotoNSDE(
        N=3, hidden_dim=8, n_steps=10, dt=.2, n_obs=31,
        drift_depth=1, diffusion_depth=1, adaptive=True,
        adaptive_atol=2e-4, adaptive_dtmin=1e-5, adaptive_max_steps=512,
        adjoint=ReversibleAdjoint(), key=jax.random.key(0),
    )
    last = model.drift_field.rho_node.layers[-1]
    model = eqx.tree_at(
        lambda m: (m.drift_field.rho_node.layers[-1].weight,
                   m.drift_field.rho_node.layers[-1].bias), model,
        (.2 * jax.random.normal(jax.random.key(1), last.weight.shape), jnp.array([1.5, .7])),
    )
    theta = jnp.array([[3.13, -.3, .4], [-.5, .2, .9]])
    omega = jnp.array([[.1, -.2, .3], [2., -1., .4]])
    keys = jax.random.split(jax.random.key(2), 2)

    @eqx.filter_jit
    def accepted_steps(m):
        return jax.vmap(lambda th, om, key: m.solve(th, om, key).stats["num_accepted_steps"])(theta, omega, keys)

    counts = accepted_steps(model)
    assert counts[0] != counts[1]

    @eqx.filter_jit
    def value_and_gradient(m):
        def loss(m):
            th, om = jax.vmap(m)(theta, omega, keys)
            return jnp.mean(jnp.cos(th) + om**2)
        return eqx.filter_value_and_grad(loss)(m)

    reference = copy.copy(model)
    object.__setattr__(reference, "adjoint", RecursiveCheckpointAdjoint(checkpoints=512))
    value_full, grad_full = value_and_gradient(reference)
    value_reverse, grad_reverse = value_and_gradient(model)
    np.testing.assert_array_equal(value_reverse, value_full)
    flat = lambda g: np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(g)])
    reference_grad = flat(grad_full)
    error = np.linalg.norm(flat(grad_reverse) - reference_grad) / np.linalg.norm(reference_grad)
    assert np.isfinite(error) and error < 2e-4


def test_adaptive_training_can_use_checkpointed_gradients():
    from experiments.kuramoto.experiment.factories import make_model

    config = ExperimentConfig(adaptive=True, adjoint='checkpoint_recursive', N=3, hidden_dim=8)
    model = make_model(config, {'N': 3, 'T': 1., 'n_obs': 8}, jax.random.key(0))
    assert isinstance(model.adjoint, RecursiveCheckpointAdjoint)
    with pytest.raises(ValueError, match='supports only'):
        ExperimentConfig(adaptive=False, adjoint='checkpoint_recursive')


def test_short_accepted_final_step_is_successful_under_jit_and_vmap():
    from diffrax import Heun, ODETerm, RESULTS, diffeqsolve
    from experiments.kuramoto.models.kuramoto_nsde import EndpointPIDController

    def solve(dt):
        controller = EndpointPIDController(
            rtol=0., atol=.01, dtmin=.001, dtmax=dt,
            factormax=1., force_dtmin=False, end_time=1.,
        )
        return diffeqsolve(ODETerm(lambda t, y, args: jnp.ones_like(y)), Heun(),
                          t0=0., t1=1., dt0=dt, y0=jnp.array(0.),
                          stepsize_controller=controller, throw=False)

    solutions = jax.jit(jax.vmap(solve))(jnp.array([.33332, .24999]))
    assert bool(jnp.all(solutions.result == RESULTS.successful))
    np.testing.assert_allclose(solutions.ts, 1.)
    np.testing.assert_allclose(solutions.ys, 1., atol=1e-6)


@pytest.mark.parametrize('start,end', [(0., .0001), (.9999, 1.)])
def test_minimum_step_failure_is_preserved_for_rejected_steps(start, end):
    from diffrax import RESULTS
    from experiments.kuramoto.models.kuramoto_nsde import EndpointPIDController

    controller = EndpointPIDController(rtol=0., atol=.01, dtmin=.001,
                                       force_dtmin=False, end_time=1.)
    keep, _, _, _, _, result = controller.adapt_step_size(
        start, end, jnp.array(0.), jnp.array(1.), None,
        jnp.array(100.), 2., (jnp.array(1.), jnp.array(1.), False),
    )
    assert not bool(keep)
    assert result == RESULTS.dt_min_reached


def test_short_accepted_interior_step_still_reports_minimum_failure():
    from diffrax import RESULTS
    from experiments.kuramoto.models.kuramoto_nsde import EndpointPIDController

    controller = EndpointPIDController(rtol=0., atol=.01, dtmin=.001,
                                       force_dtmin=False, end_time=1.)
    keep, _, _, _, _, result = controller.adapt_step_size(
        0., .00001, jnp.array(0.), jnp.array(.00001), None,
        jnp.array(0.), 2., (jnp.array(1.), jnp.array(1.), False),
    )
    assert bool(keep)
    assert result == RESULTS.dt_min_reached
