r"""Euclidean-embedding NSDE baseline for the Kuramoto experiment.

State $(\sin\theta, \cos\theta, \omega) \in \mathbb{R}^{3N}$ (embedded
on a non-Lie-group flat manifold). Drift and diffusion are MLPs over
the same input features as `KuramotoNSDE`. Integration is via standard
diffrax `Heun` + `DirectAdjoint` (or any specified adjoint) — no
Lie-group machinery.

Predictions are projected back to the torus via
$\theta_i = \mathrm{atan2}(\sin\theta_i, \cos\theta_i)$ at evaluation
time. The intent is to expose constraint drift in the embedded
representation over long horizons, providing a sanity baseline that the
$T\mathbb{T}^N$-valued NSDE has to beat.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    AbstractAdjoint,
    ControlTerm,
    DirectAdjoint,
    Heun,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)

from experiments.kuramoto.models.kuramoto_nsde import resolve_activation


def _state_to_features(y: jax.Array, N: int) -> jax.Array:
    """y has shape (3N,) = (sin theta, cos theta, omega). Pass through."""
    del N
    return y


def _features_to_theta(y: jax.Array, N: int) -> jax.Array:
    """Project features back to the torus: theta = atan2(sin, cos)."""
    sin_t = y[..., :N]
    cos_t = y[..., N : 2 * N]
    return jnp.arctan2(sin_t, cos_t)


class EuclideanDriftField(eqx.Module):
    mlp: eqx.nn.MLP
    N: int = eqx.field(static=True)

    def __init__(self, N, hidden_dim, *, depth: int = 3, activation=jax.nn.silu, key):
        mlp = eqx.nn.MLP(
            in_size=3 * N, out_size=3 * N, width_size=hidden_dim,
            depth=depth, activation=activation, key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))
        last = eqx.tree_at(lambda l: l.weight, last, jnp.zeros_like(last.weight))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.N = N

    def __call__(self, t, y, args):
        del t, args
        return self.mlp(y)


class EuclideanDiffusionField(eqx.Module):
    mlp: eqx.nn.MLP
    N: int = eqx.field(static=True)
    diffusion_scale: float = eqx.field(static=True)

    def __init__(self, N, hidden_dim, diffusion_scale=0.3, *, depth=2, activation=jax.nn.silu, key):
        mlp = eqx.nn.MLP(
            in_size=3 * N, out_size=N, width_size=hidden_dim,
            depth=depth, activation=activation, key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.full_like(last.bias, -2.0))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.N = N
        self.diffusion_scale = float(diffusion_scale)

    def __call__(self, t, y, args):
        del t, args
        N = self.N
        scales = jax.nn.softplus(self.mlp(y)) * self.diffusion_scale
        # Noise on the omega component only (rows 2N : 3N), N Brownian dims.
        zero_block = jnp.zeros((2 * N, N), dtype=scales.dtype)
        omega_block = jnp.diag(scales)
        return jnp.concatenate([zero_block, omega_block], axis=0)


class EuclideanKuramotoNSDE(eqx.Module):
    """Euclidean (sin, cos, omega) baseline NSDE for the Kuramoto task."""

    drift_field: EuclideanDriftField
    diffusion_field: EuclideanDiffusionField
    name: str = eqx.field(static=True)
    N: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_obs: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    adjoint: AbstractAdjoint | None = eqx.field(static=True)

    def __init__(
        self,
        N: int,
        hidden_dim: int = 128,
        n_steps: int = 200,
        dt: float = 0.025,
        n_obs: int = 200,
        diffusion_scale: float = 0.3,
        adjoint: AbstractAdjoint | None = None,
        activation: str = "silu",
        drift_depth: int = 3,
        diffusion_depth: int = 2,
        *,
        key,
    ):
        k_drift, k_diff = jax.random.split(key, 2)
        act_fn = resolve_activation(activation)

        self.name = "euclidean_kuramoto_nsde"
        self.N = N
        self.n_steps = n_steps
        self.n_obs = n_obs
        self.dt = dt
        self.adjoint = adjoint

        self.drift_field = EuclideanDriftField(
            N=N, hidden_dim=hidden_dim, depth=drift_depth, activation=act_fn, key=k_drift,
        )
        self.diffusion_field = EuclideanDiffusionField(
            N=N, hidden_dim=hidden_dim, diffusion_scale=diffusion_scale,
            depth=diffusion_depth, activation=act_fn, key=k_diff,
        )

    def __call__(
        self,
        theta0: jax.Array,
        omega0: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Forecast a trajectory from initial state $(\\theta_0, \\omega_0)$.

        Returns (theta, omega) at the saved timepoints, with theta projected
        back to $(-\\pi, \\pi]$ via atan2.
        """
        N = self.N
        t1 = float(self.n_steps * self.dt)
        t_save = jnp.linspace(0.0, t1, self.n_obs)
        # Embed initial condition.
        y0 = jnp.concatenate([jnp.sin(theta0), jnp.cos(theta0), omega0])

        bm = VirtualBrownianTree(
            t0=0.0, t1=t1, tol=self.dt / 4.0, shape=(N,), key=key,
        )
        term = MultiTerm(
            ODETerm(self.drift_field),
            ControlTerm(self.diffusion_field, bm),
        )
        adjoint = self.adjoint if self.adjoint is not None else DirectAdjoint()

        sol = diffeqsolve(
            term, Heun(), t0=0.0, t1=t1, dt0=self.dt, y0=y0,
            saveat=SaveAt(ts=t_save), adjoint=adjoint,
            max_steps=self.n_steps + 16,
        )
        ys = sol.ys  # (n_obs, 3N)
        theta_traj = _features_to_theta(ys, N)
        omega_traj = ys[:, 2 * N :]
        return theta_traj, omega_traj
