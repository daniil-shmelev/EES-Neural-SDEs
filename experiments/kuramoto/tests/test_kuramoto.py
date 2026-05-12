"""Sanity tests for the stochastic 2nd-order Kuramoto module.

Verifies the analytic two-oscillator phase-lock prediction and that the
simulator runs end-to-end and respects shape / wrap conventions.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.kuramoto.datasets.kuramoto import (
    KuramotoParams,
    critical_coupling,
    kuramoto_drift,
    order_parameter,
)
from experiments.kuramoto.datasets.simulator import (
    SimConfig,
    sample_initial_conditions,
    simulate_one,
)


def test_bimodal_balanced():
    """For even N, sum of natural frequencies must be zero."""
    p = KuramotoParams.bimodal(N=4, P=0.7)
    np.testing.assert_allclose(float(jnp.sum(p.natural_freqs)), 0.0, atol=1e-7)


def test_drift_zero_at_synced_no_freq():
    """All-equal phases + zero natural frequencies + zero velocity -> zero drift on omega."""
    p = KuramotoParams(
        N=3, m=1.0, K=1.0, natural_freqs=jnp.zeros(3),
    )
    theta = jnp.array([0.4, 0.4, 0.4])
    omega = jnp.zeros(3)
    dtheta, domega = kuramoto_drift(theta, omega, p)
    np.testing.assert_allclose(np.asarray(dtheta), np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(np.asarray(domega), np.zeros(3), atol=1e-6)


def test_order_parameter_synced_unity():
    """All identical phases give r = 1."""
    theta = jnp.array([0.7, 0.7, 0.7, 0.7])
    r, _ = order_parameter(theta)
    np.testing.assert_allclose(float(r), 1.0, atol=1e-6)


def test_order_parameter_uniform_zero():
    """Equally-spaced phases on the circle give r approximately 0."""
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 6, endpoint=False)
    r, _ = order_parameter(theta)
    assert float(r) < 1e-6


def test_two_oscillator_phase_lock():
    """For N=2, K > 2P, deterministic dynamics phase-lock at arcsin(2P/K)."""
    P, K = 0.5, 2.0
    target = float(jnp.arcsin(2.0 * P / K))
    params = KuramotoParams.bimodal(N=2, P=P, K=K)
    cfg = SimConfig(T=20.0, n_fine=4096, n_obs=64, D=0.0)
    theta_traj, _ = simulate_one(
        params,
        cfg,
        theta0=jnp.array([0.3, -0.3]),
        omega0=jnp.zeros(2),
        key=jr.PRNGKey(0),
    )
    delta = float(theta_traj[-1, 0] - theta_traj[-1, 1])
    delta_wrapped = float(jnp.mod(delta + jnp.pi, 2.0 * jnp.pi) - jnp.pi)
    np.testing.assert_allclose(abs(delta_wrapped), target, atol=2e-2)


def test_simulate_one_smoke():
    """End-to-end shape + wrap check on the SDE simulator."""
    params = KuramotoParams.bimodal(N=4, P=0.5, K=1.0)
    cfg = SimConfig(T=0.2, n_fine=128, n_obs=8, D=0.05)
    theta0, omega0 = sample_initial_conditions(params, n_traj=1, omega_scale=0.5, key=jr.PRNGKey(0))
    th, om = simulate_one(params, cfg, theta0[0], omega0[0], jr.PRNGKey(1))
    assert th.shape == (8, 4)
    assert om.shape == (8, 4)
    assert jnp.all(jnp.abs(th) <= jnp.pi + 1e-6)


def test_critical_coupling_scales_with_P():
    assert float(critical_coupling(1.0)) > 2.0
    assert float(critical_coupling(2.0)) > float(critical_coupling(1.0))
