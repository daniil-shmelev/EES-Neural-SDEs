r"""Neural SDE on $T\mathbb{T}^N = \mathbb{T}^N \times \mathbb{R}^N$ for
forecasting trajectories of the stochastic 2nd-order Kuramoto network.

Drift and diffusion fields are MLPs over the feature representation
$(\sin\theta, \cos\theta, \omega) \in \mathbb{R}^{3N}$. Integration is via
diffrax `MultiTerm(ODETerm, ControlTerm)` wrapped in a georax
`GeometricTerm` so that CFEES (or any other georax solver) advances the
state on the product Lie group with one exponential per stage.

Mirrors the structure of `experiments/rna/models/torus_nsde.py`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    AbstractAdjoint,
    AbstractReversibleSolver,
    AbstractSolver,
    ControlTerm,
    DirectAdjoint,
    MultiTerm,
    ODETerm,
    ReversibleAdjoint,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
from georax import CFEES25, GeometricTerm

from experiments.kuramoto.models.product_torus import ProductTorusEuclidean

_ACTIVATIONS = {
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
    "mish": jax.nn.mish,
}


def resolve_activation(name: str):
    try:
        return _ACTIVATIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown activation {name!r}; must be one of {sorted(_ACTIVATIONS)}"
        ) from exc


def state_features(y: jax.Array, N: int) -> jax.Array:
    r"""Map state $(\theta, \omega) \in \mathbb{R}^{2N}$ to
    $(\sin\theta, \cos\theta, \omega) \in \mathbb{R}^{3N}$."""
    theta = y[..., :N]
    omega = y[..., N:]
    return jnp.concatenate([jnp.sin(theta), jnp.cos(theta), omega], axis=-1)


class KuramotoDriftField(eqx.Module):
    r"""MLP drift on $T\mathbb{T}^N$.

    Input: state features $(\sin\theta, \cos\theta, \omega) \in \mathbb{R}^{3N}$.
    Output: Lie-algebra increment $(\dot\theta, \dot\omega) \in \mathbb{R}^{2N}$.
    """

    mlp: eqx.nn.MLP
    geometry: ProductTorusEuclidean

    def __init__(self, geometry, hidden_dim, *, depth: int = 3, activation=jax.nn.silu, key):
        N = geometry.N
        mlp = eqx.nn.MLP(
            in_size=3 * N,
            out_size=2 * N,
            width_size=hidden_dim,
            depth=depth,
            activation=activation,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))
        last = eqx.tree_at(lambda l: l.weight, last, jnp.zeros_like(last.weight))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.geometry = geometry

    def __call__(self, t, y, args):
        del t, args
        return self.mlp(state_features(y, self.geometry.N))


class KuramotoDiffusionField(eqx.Module):
    r"""MLP diffusion on $T\mathbb{T}^N$.

    Input: state features $(\sin\theta, \cos\theta, \omega)$.
    Output: a $2N \times N$ matrix $G(t, y)$ such that the SDE increment
    is $G\,dW$ with $W \in \mathbb{R}^N$ Brownian. By default the noise
    only acts on $\omega$ (lower $N \times N$ block); the upper block is
    zero — matching the structure of the data-generating SDE
    (Filatrella et al. 2008 / Olmi & Torcini 2024).
    """

    mlp: eqx.nn.MLP
    geometry: ProductTorusEuclidean
    diffusion_scale: float = eqx.field(static=True)
    couple_theta_omega: bool = eqx.field(static=True)

    def __init__(
        self,
        geometry,
        hidden_dim,
        diffusion_scale: float = 0.3,
        *,
        depth: int = 2,
        activation=jax.nn.silu,
        couple_theta_omega: bool = False,
        key,
    ):
        N = geometry.N
        out_dim = N if not couple_theta_omega else 2 * N
        mlp = eqx.nn.MLP(
            in_size=3 * N,
            out_size=out_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=activation,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.full_like(last.bias, -2.0))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.geometry = geometry
        self.diffusion_scale = float(diffusion_scale)
        self.couple_theta_omega = bool(couple_theta_omega)

    def __call__(self, t, y, args):
        del t, args
        N = self.geometry.N
        scales = jax.nn.softplus(self.mlp(state_features(y, N))) * self.diffusion_scale
        if self.couple_theta_omega:
            full = scales  # (2N,)
            return jnp.diag(full)[:, : N]  # (2N, N)
        # Noise only on omega: top zero block, bottom diag(scales).
        zero_block = jnp.zeros((N, N), dtype=scales.dtype)
        omega_block = jnp.diag(scales)
        return jnp.concatenate([zero_block, omega_block], axis=0)


class KuramotoNSDE(eqx.Module):
    r"""Neural SDE on $T\mathbb{T}^N$ for trajectory forecasting."""

    drift_field: KuramotoDriftField
    diffusion_field: KuramotoDiffusionField
    name: str = eqx.field(static=True)

    N: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_obs: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    solver: AbstractSolver = eqx.field(static=True)
    adjoint: AbstractAdjoint | None = eqx.field(static=True)

    def __init__(
        self,
        N: int,
        hidden_dim: int = 128,
        n_steps: int = 200,
        dt: float = 0.025,
        n_obs: int = 200,
        solver: AbstractSolver | None = None,
        diffusion_scale: float = 0.3,
        adjoint: AbstractAdjoint | None = None,
        activation: str = "silu",
        drift_depth: int = 3,
        diffusion_depth: int = 2,
        couple_theta_omega: bool = False,
        *,
        key,
    ):
        k_drift, k_diff = jax.random.split(key, 2)

        geometry = ProductTorusEuclidean(N)
        act_fn = resolve_activation(activation)
        if solver is None:
            solver = CFEES25()

        self.name = "kuramoto_nsde"
        self.N = N
        self.n_steps = n_steps
        self.n_obs = n_obs
        self.dt = dt
        self.solver = solver
        self.adjoint = adjoint

        self.drift_field = KuramotoDriftField(
            geometry=geometry,
            hidden_dim=hidden_dim,
            depth=drift_depth,
            activation=act_fn,
            key=k_drift,
        )
        self.diffusion_field = KuramotoDiffusionField(
            geometry=geometry,
            hidden_dim=hidden_dim,
            diffusion_scale=diffusion_scale,
            depth=diffusion_depth,
            activation=act_fn,
            couple_theta_omega=couple_theta_omega,
            key=k_diff,
        )

    def __call__(
        self,
        theta0: jax.Array,
        omega0: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Forecast a single trajectory from initial state $(\\theta_0, \\omega_0)$.

        Returns (theta, omega), each of shape (n_obs, N), at the saved
        timepoints.
        """
        N = self.N
        t1 = float(self.n_steps * self.dt)
        t_save = jnp.linspace(0.0, t1, self.n_obs)

        y0 = jnp.concatenate([theta0, omega0])
        brownian_path = VirtualBrownianTree(
            t0=0.0, t1=t1, tol=self.dt / 4.0, shape=(N,), key=key,
        )
        term = GeometricTerm(
            inner=MultiTerm(
                ODETerm(self.drift_field),
                ControlTerm(self.diffusion_field, brownian_path),
            ),
            geometry=self.drift_field.geometry,
        )

        if self.adjoint is not None:
            adjoint = self.adjoint
        elif isinstance(self.solver, AbstractReversibleSolver):
            adjoint = ReversibleAdjoint()
        else:
            adjoint = DirectAdjoint()

        sol = diffeqsolve(
            term, self.solver, t0=0.0, t1=t1, dt0=self.dt, y0=y0,
            saveat=SaveAt(ts=t_save), adjoint=adjoint,
            max_steps=self.n_steps + 16,
        )
        ys = sol.ys  # (n_obs, 2N)
        return ys[:, :N], ys[:, N:]
