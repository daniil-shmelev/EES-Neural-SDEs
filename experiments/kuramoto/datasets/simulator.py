r"""Simulator for the stochastic 2nd-order Kuramoto SDE on $T\mathbb{T}^N$.

Integrates the canonical equation (Olmi & Torcini 2024 eq. (1) with
$K_2 = 0$; equivalently Filatrella, Nielsen & Pedersen 2008 +
Schmietendorf et al. 2014 noise term):

  $\dot\theta_i = \omega_i$
  $m\,\dot\omega_i = -\omega_i + \Omega_i + (K/N)\sum_j \sin(\theta_j - \theta_i)
                     + \xi_i(t)$
  $\langle\xi_i(t)\xi_j(s)\rangle = 2D\delta_{ij}\delta(t-s)$

The diffrax solver of choice is `Heun` (Stratonovich, strong order 0.5,
weak order 1). Because the diffusion is constant additive noise on
$\omega$, Stratonovich and Ito coincide.

We use `VirtualBrownianTree` for refinement-consistent Brownian paths.
"""

from __future__ import annotations

from typing import NamedTuple

import diffrax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from experiments.kuramoto.datasets.kuramoto import (
    KuramotoParams,
    make_state_diffusion,
    make_state_drift,
)
from experiments.kuramoto.geometry import wrap_to_pi


class SimConfig(NamedTuple):
    T: float
    n_fine: int
    n_obs: int
    D: float

    @property
    def dt_fine(self) -> float:
        return self.T / self.n_fine


def simulate_one(
    params: KuramotoParams,
    cfg: SimConfig,
    theta0: Float[Array, " N"],
    omega0: Float[Array, " N"],
    key: PRNGKeyArray,
) -> tuple[Float[Array, "T N"], Float[Array, "T N"]]:
    """Simulate a single trajectory.

    Returns (theta, omega) sampled at n_obs uniform points in [0, T].
    Theta is wrapped to [-pi, pi).
    """
    drift = make_state_drift(params)
    diffusion = make_state_diffusion(params, D=cfg.D)
    N = params.N

    bm = diffrax.VirtualBrownianTree(
        t0=0.0, t1=cfg.T, tol=cfg.dt_fine / 4.0, shape=(N,), key=key,
    )
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, bm),
    )
    y0 = jnp.concatenate([theta0, omega0])
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
    theta_traj = wrap_to_pi(ys[:, :N])
    omega_traj = ys[:, N:]
    return theta_traj, omega_traj


def sample_initial_conditions(
    params: KuramotoParams,
    n_traj: int,
    omega_scale: float,
    key: PRNGKeyArray,
) -> tuple[Float[Array, "B N"], Float[Array, "B N"]]:
    r"""Draw $(\theta_0, \omega_0)$ for a batch of trajectories.

    $\theta_0 \sim \mathcal{U}(-\pi, \pi]^N$ (uniform random initial phases).
    $\omega_0 \sim \mathcal{N}(0, \mathrm{omega\_scale}^2 \cdot I_N)$
    (Gaussian initial velocities centred on the synchronous frame).
    """
    N = params.N
    k_theta, k_omega = jr.split(key)
    theta = jr.uniform(k_theta, (n_traj, N), minval=-jnp.pi, maxval=jnp.pi)
    omega = omega_scale * jr.normal(k_omega, (n_traj, N))
    return theta, omega
