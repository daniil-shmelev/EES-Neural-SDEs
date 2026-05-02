"""Closed-form Lagrangian and Hamiltonian for the n-link planar pendulum.

The dynamics are derived analytically from the Lagrangian
$L(\\theta, \\dot\\theta) = T(\\theta, \\dot\\theta) - V(\\theta)$
with point masses $m_k$ at link tips, link lengths $l_k$, gravity $g$.

The mass matrix $M(\\theta)$ and gradient $\\nabla V(\\theta)$ are obtained
via JAX autodiff on the closed-form $T$ and $V$ — no symbolic derivation,
scales to any $n$.

Convention: link $k$ swings from joint $k-1$, the angle $\\theta_k$ is
measured counter-clockwise from the downward vertical, so the equilibrium
$\\theta = 0$ has all links hanging straight down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class PendulumParams:
    """Physical parameters of an n-link planar pendulum.

    All length-n vectors. Gravity is a single scalar.
    """

    n: int
    masses: Float[Array, "n"]
    lengths: Float[Array, "n"]
    g: float

    @staticmethod
    def uniform(n: int, mass: float = 1.0, length: float = 1.0, g: float = 9.81) -> "PendulumParams":
        return PendulumParams(
            n=n,
            masses=jnp.full((n,), mass),
            lengths=jnp.full((n,), length),
            g=g,
        )


def link_tip_positions(theta: Float[Array, "n"], params: PendulumParams) -> Float[Array, "n 2"]:
    """Cumulative tip positions in the plane.

    Returns r_k = sum_{j<=k} l_j (sin theta_j, -cos theta_j) for k=1..n.
    """
    sx = params.lengths * jnp.sin(theta)
    sy = -params.lengths * jnp.cos(theta)
    return jnp.stack([jnp.cumsum(sx), jnp.cumsum(sy)], axis=-1)


def link_tip_velocities(
    theta: Float[Array, "n"],
    dtheta: Float[Array, "n"],
    params: PendulumParams,
) -> Float[Array, "n 2"]:
    """Velocities of the link tips in the plane.

    dr_k/dt = sum_{j<=k} l_j dtheta_j (cos theta_j, sin theta_j).
    """
    vx = params.lengths * dtheta * jnp.cos(theta)
    vy = params.lengths * dtheta * jnp.sin(theta)
    return jnp.stack([jnp.cumsum(vx), jnp.cumsum(vy)], axis=-1)


def kinetic_energy(
    theta: Float[Array, "n"],
    dtheta: Float[Array, "n"],
    params: PendulumParams,
) -> Float[Array, ""]:
    """T = 1/2 sum_k m_k ||dr_k/dt||^2."""
    v = link_tip_velocities(theta, dtheta, params)
    return 0.5 * jnp.sum(params.masses * jnp.sum(v * v, axis=-1))


def potential_energy(
    theta: Float[Array, "n"],
    params: PendulumParams,
) -> Float[Array, ""]:
    """V = -g sum_k m_k cumsum(l_j cos theta_j)."""
    cy = jnp.cumsum(params.lengths * jnp.cos(theta))
    return -params.g * jnp.sum(params.masses * cy)


def mass_matrix(theta: Float[Array, "n"], params: PendulumParams) -> Float[Array, "n n"]:
    """M(theta) = d^2 T / d(dtheta)^2.

    Since T is quadratic in dtheta, the Hessian is independent of dtheta and
    equals the configuration-dependent mass matrix.
    """
    zeros = jnp.zeros_like(theta)
    return jax.hessian(lambda dt: kinetic_energy(theta, dt, params))(zeros)


def hamiltonian(
    theta: Float[Array, "n"],
    p: Float[Array, "n"],
    params: PendulumParams,
) -> Float[Array, ""]:
    """H(theta, p) = 1/2 p^T M(theta)^{-1} p + V(theta)."""
    M = mass_matrix(theta, params)
    Minv_p = jnp.linalg.solve(M, p)
    return 0.5 * jnp.dot(p, Minv_p) + potential_energy(theta, params)


def hamiltonian_drift(
    theta: Float[Array, "n"],
    p: Float[Array, "n"],
    params: PendulumParams,
    gamma: float = 0.0,
) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
    """Drift of the underdamped Langevin SDE on $T^*\\mathbb{T}^n$.

    $d\\theta = M(\\theta)^{-1} p\\,dt$
    $dp = (-\\nabla_\\theta H(\\theta, p) - \\gamma p)\\,dt + (\\text{noise})$

    Returns the deterministic part as a tuple $(\\dot\\theta, \\dot p)$. Noise
    is added by the caller (this keeps the function pure for use with diffrax
    `ODETerm`s and reusable for both the simulator and other tooling).
    """
    M = mass_matrix(theta, params)
    Minv_p = jnp.linalg.solve(M, p)
    dtheta = Minv_p
    grad_H_theta = jax.grad(hamiltonian, argnums=0)(theta, p, params)
    dp = -grad_H_theta - gamma * p
    return dtheta, dp


def momentum_to_velocity(
    theta: Float[Array, "n"],
    p: Float[Array, "n"],
    params: PendulumParams,
) -> Float[Array, "n"]:
    """Convert canonical momentum to angular velocity: omega = M(theta)^{-1} p."""
    return jnp.linalg.solve(mass_matrix(theta, params), p)


def velocity_to_momentum(
    theta: Float[Array, "n"],
    omega: Float[Array, "n"],
    params: PendulumParams,
) -> Float[Array, "n"]:
    """Convert angular velocity to canonical momentum: p = M(theta) omega."""
    return mass_matrix(theta, params) @ omega


def total_energy_velocity(
    theta: Float[Array, "n"],
    omega: Float[Array, "n"],
    params: PendulumParams,
) -> Float[Array, ""]:
    """Convenience: E = T(theta, omega) + V(theta) for diagnostics on (theta, omega) data."""
    return kinetic_energy(theta, omega, params) + potential_energy(theta, params)


def make_state_drift(params: PendulumParams, gamma: float) -> Callable:
    """Build an `ODETerm`-compatible drift function $f(t, (theta, p), args) \\to (dtheta, dp)$.

    The returned function takes a stacked state of shape `(2n,)` (theta first,
    then p) and returns the deterministic part of $d(theta, p)/dt$ in the
    same shape. This is what `diffrax.ODETerm` expects.
    """

    n = params.n

    def drift(t, y, args):
        del t, args
        theta = y[:n]
        p = y[n:]
        dtheta, dp = hamiltonian_drift(theta, p, params, gamma=gamma)
        return jnp.concatenate([dtheta, dp])

    return drift


def make_state_diffusion(params: PendulumParams, sigma: float) -> Callable:
    """Build a `ControlTerm`-compatible diffusion function.

    The diffusion is decoupled additive noise on momentum:
    $dp = ... + \\sigma\\,dW$, where $W \\in \\mathbb{R}^n$.
    Returns a function $g(t, y, args) \\to G \\in \\mathbb{R}^{2n \\times n}$
    such that the increment is $G\\,dW$.
    """
    n = params.n

    def diffusion(t, y, args):
        del t, y, args
        zero_block = jnp.zeros((n, n))
        sigma_block = sigma * jnp.eye(n)
        return jnp.concatenate([zero_block, sigma_block], axis=0)

    return diffusion
