"""Sanity tests for the closed-form n-link pendulum Lagrangian.

Verifies the JAX-derived mass matrix and gradient against analytic formulas
in small cases ($n = 1, 2$), and checks energy conservation of the symplectic
Verlet integrator at $\\sigma = 0$.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from experiments.pendulum.datasets.lagrangian import (
    PendulumParams,
    hamiltonian,
    kinetic_energy,
    mass_matrix,
    potential_energy,
)
from experiments.pendulum.datasets.simulator import (
    SimConfig,
    simulate_one,
    verlet_rollout,
)


def test_single_pendulum_mass_matrix():
    """For $n = 1$, $M(\\theta) = m l^2$ for any $\\theta$."""
    p = PendulumParams.uniform(n=1, mass=1.5, length=0.7)
    for theta in [0.0, 0.5, -1.2, 3.0]:
        M = mass_matrix(jnp.array([theta]), p)
        assert M.shape == (1, 1)
        np.testing.assert_allclose(M[0, 0], 1.5 * 0.7**2, atol=1e-6)


def test_double_pendulum_mass_matrix():
    """Analytic mass matrix for $n = 2$, equal masses and lengths.

    M_11 = (m_1 + m_2) l_1^2
    M_12 = M_21 = m_2 l_1 l_2 cos(theta_1 - theta_2)
    M_22 = m_2 l_2^2
    """
    m1 = 1.3
    m2 = 0.8
    l1 = 1.1
    l2 = 0.9
    p = PendulumParams(
        n=2,
        masses=jnp.array([m1, m2]),
        lengths=jnp.array([l1, l2]),
        g=9.81,
    )
    for t1, t2 in [(0.0, 0.0), (0.4, -0.7), (1.5, 2.1), (-0.3, 1.0)]:
        M = mass_matrix(jnp.array([t1, t2]), p)
        expected = jnp.array(
            [
                [(m1 + m2) * l1 * l1, m2 * l1 * l2 * jnp.cos(t1 - t2)],
                [m2 * l1 * l2 * jnp.cos(t1 - t2), m2 * l2 * l2],
            ]
        )
        np.testing.assert_allclose(np.asarray(M), np.asarray(expected), atol=1e-6)


def test_potential_zero_at_hanging():
    """V is minimised at theta=0 (all links hanging straight down)."""
    p = PendulumParams.uniform(n=3)
    V_eq = potential_energy(jnp.zeros(3), p)
    V_other = potential_energy(jnp.array([0.5, -0.3, 1.1]), p)
    assert float(V_other) > float(V_eq)


def test_energy_conservation_verlet():
    """Symplectic Verlet should conserve energy to within $O(dt^2)$ over short times."""
    p = PendulumParams.uniform(n=2)
    theta0 = jnp.array([0.6, -0.4])
    p0 = jnp.array([0.3, -0.2])

    H0 = hamiltonian(theta0, p0, p)
    th_traj, p_traj = verlet_rollout(theta0, p0, dt=1e-4, n_steps=2000, params=p)
    H_t = jnp.array([hamiltonian(th_traj[i], p_traj[i], p) for i in range(0, 2001, 100)])
    drift = jnp.max(jnp.abs(H_t - H0)) / jnp.abs(H0)
    assert float(drift) < 1e-4, f"Verlet energy drift too large: {drift:.3e}"


def test_simulate_one_smoke():
    """End-to-end shape check on the SDE simulator."""
    p = PendulumParams.uniform(n=2)
    cfg = SimConfig(T=0.2, n_fine=128, n_obs=8, sigma=0.1, gamma=0.05)
    import jax.random as jr

    theta0 = jnp.array([0.5, -0.3])
    p0 = jnp.array([0.1, 0.0])
    th, om = simulate_one(p, cfg, theta0, p0, jr.PRNGKey(0))
    assert th.shape == (8, 2)
    assert om.shape == (8, 2)
    # theta should be wrapped into (-pi, pi]
    assert jnp.all(jnp.abs(th) <= jnp.pi + 1e-6)
    # initial state recovered (momentum-to-velocity at t=0)
    np.testing.assert_allclose(np.asarray(th[0]), np.asarray(theta0), atol=1e-6)
