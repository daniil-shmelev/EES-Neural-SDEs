"""Reference simulator for the underdamped Langevin SDE on $T^*\\mathbb{T}^n$.

Integrates the Hamiltonian dynamics of an n-link planar pendulum with
linear friction and additive Brownian forcing on momentum:

  $d\\theta = M(\\theta)^{-1} p\\,dt$
  $dp = -\\nabla_\\theta H(\\theta, p)\\,dt - \\gamma p\\,dt + \\sigma\\,dW$

The diffrax solver of choice is `Heun` (Stratonovich, strong order 0.5,
weak order 1). Because the diffusion is constant additive noise on $p$,
Stratonovich and It\\^o coincide, so the choice of `Heun` over an It\\^o
solver is immaterial.

We use `VirtualBrownianTree` for refinement-consistent Brownian paths,
matching the pattern used elsewhere in the repository
(`experiments/rna/models/torus_nsde.py`).

A symplectic-Verlet sub-routine is provided separately for sanity checks
of the Hamiltonian flow at $\\sigma = 0$, $\\gamma = 0$.
"""

from __future__ import annotations

from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from experiments.pendulum.datasets.lagrangian import (
    PendulumParams,
    hamiltonian,
    make_state_diffusion,
    make_state_drift,
    mass_matrix,
    momentum_to_velocity,
    total_energy_velocity,
)
from experiments.rna.models.torus import wrap_to_pi


class SimConfig(NamedTuple):
    T: float
    n_fine: int  # integration steps over [0, T]; dt_fine = T / n_fine
    n_obs: int  # number of save-at points (uniform in [0, T])
    sigma: float  # noise scale on momentum
    gamma: float  # friction coefficient

    @property
    def dt_fine(self) -> float:
        return self.T / self.n_fine


def simulate_one(
    params: PendulumParams,
    cfg: SimConfig,
    theta0: Float[Array, "n"],
    p0: Float[Array, "n"],
    key: PRNGKeyArray,
) -> tuple[Float[Array, "T n"], Float[Array, "T n"]]:
    """Simulate a single trajectory.

    Returns (theta, omega) sampled at n_obs uniform points in [0, T].
    Theta is wrapped to (-pi, pi].
    """
    drift = make_state_drift(params, gamma=cfg.gamma)
    diffusion = make_state_diffusion(params, sigma=cfg.sigma)
    n = params.n

    bm = diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=cfg.T,
        tol=cfg.dt_fine / 4.0,
        shape=(n,),
        key=key,
    )
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, bm),
    )
    y0 = jnp.concatenate([theta0, p0])
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, cfg.T, cfg.n_obs))
    sol = diffrax.diffeqsolve(
        terms,
        diffrax.Heun(),
        t0=0.0,
        t1=cfg.T,
        dt0=cfg.dt_fine,
        y0=y0,
        saveat=saveat,
        max_steps=cfg.n_fine + 16,
        adjoint=diffrax.DirectAdjoint(),
    )
    ys = sol.ys
    theta_traj = wrap_to_pi(ys[:, :n])
    p_traj = ys[:, n:]
    omega_traj = jax.vmap(lambda th, pp: momentum_to_velocity(th, pp, params))(theta_traj, p_traj)
    return theta_traj, omega_traj


def chaos_onset_energy_scale(params: PendulumParams) -> Float[Array, ""]:
    """Approximate total-energy budget at which the n-link pendulum becomes chaotic.

    For the equal-mass equal-length double pendulum the chaotic onset is
    around the energy needed to flip the upper link, $E \\approx 2 m g l$.
    For larger $n$ we use the same gravitational scale, which is enough to
    seed an interesting trajectory; the chaotic regime is insensitive to
    the exact value.
    """
    return 2.0 * jnp.sum(params.masses) * params.g * jnp.mean(params.lengths)


def sample_initial_conditions(
    params: PendulumParams,
    n_traj: int,
    energy_low: float,
    energy_high: float,
    key: PRNGKeyArray,
) -> tuple[Float[Array, "B n"], Float[Array, "B n"]]:
    """Draw (theta0, p0) with theta uniform on (-pi, pi]^n and total energy in [low, high].

    For each sample we draw theta uniformly, draw a unit Gaussian direction in
    p-space, and rescale p so the kinetic energy makes the total energy match
    a uniform sample from [energy_low, energy_high]. If the chosen total
    energy is below V(theta), the kinetic budget is floored at a small
    positive value (the system still gets a small kick to start moving).
    """
    n = params.n
    k_theta, k_dir, k_e = jr.split(key, 3)
    theta = jr.uniform(k_theta, (n_traj, n), minval=-jnp.pi, maxval=jnp.pi)
    e_target = jr.uniform(k_e, (n_traj,), minval=energy_low, maxval=energy_high)
    dirs = jr.normal(k_dir, (n_traj, n))

    def make_p(theta_i, e_i, u):
        V_i = total_energy_velocity(theta_i, jnp.zeros(n), params)
        kinetic_target = jnp.maximum(e_i - V_i, 1e-3)
        # E_kin = (1/2) p^T M^{-1} p; with p = s*u, s^2 = 2 K / (u^T M^{-1} u).
        Minv_u = jnp.linalg.solve(mass_matrix(theta_i, params), u)
        u_norm = jnp.dot(u, Minv_u)
        s = jnp.sqrt(2.0 * kinetic_target / jnp.maximum(u_norm, 1e-12))
        return s * u

    p = jax.vmap(make_p)(theta, e_target, dirs)
    return theta, p


# Symplectic Verlet — sanity-check reference for the sigma=0 Hamiltonian flow.


def verlet_step(
    theta: Float[Array, "n"],
    p: Float[Array, "n"],
    dt: float,
    params: PendulumParams,
) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
    """One step of velocity-Verlet for the Hamiltonian flow (sigma=0, gamma=0).

    Strang split: half kick — drift — half kick. The drift step uses an
    implicit-midpoint approximation since $M(\\theta)$ depends on $\\theta$;
    one fixed-point pass is enough at the small step sizes used for sanity
    checks. Symplectic to second order.
    """
    grad_H_theta = jax.grad(hamiltonian, argnums=0)(theta, p, params)
    p_half = p - 0.5 * dt * grad_H_theta

    def drift_theta(th):
        return jnp.linalg.solve(mass_matrix(th, params), p_half)

    theta_mid = theta + 0.5 * dt * drift_theta(theta)
    theta_new = theta + dt * drift_theta(theta_mid)
    grad_H_theta_new = jax.grad(hamiltonian, argnums=0)(theta_new, p_half, params)
    p_new = p_half - 0.5 * dt * grad_H_theta_new
    return theta_new, p_new


def verlet_rollout(
    theta0: Float[Array, "n"],
    p0: Float[Array, "n"],
    dt: float,
    n_steps: int,
    params: PendulumParams,
) -> tuple[Float[Array, "T n"], Float[Array, "T n"]]:
    """Roll out a Verlet trajectory; return (theta, p) at every step."""

    def step(carry, _):
        th, pp = carry
        th_new, pp_new = verlet_step(th, pp, dt, params)
        return (th_new, pp_new), (th_new, pp_new)

    _, (th_traj, p_traj) = jax.lax.scan(step, (theta0, p0), None, length=n_steps)
    th_traj = jnp.concatenate([theta0[None], th_traj], axis=0)
    p_traj = jnp.concatenate([p0[None], p_traj], axis=0)
    return th_traj, p_traj
