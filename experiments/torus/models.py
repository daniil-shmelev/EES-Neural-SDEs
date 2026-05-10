"""Generic neural SDE on the flat torus for memory benchmarks."""

from __future__ import annotations

from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    AbstractAdjoint,
    AbstractReversibleSolver,
    AbstractSolver,
    DirectAdjoint,
    ODETerm,
    ReversibleAdjoint,
    SaveAt,
    diffeqsolve,
)
from georax import CFEES25, GeometricTerm, LieGroup
from jaxtyping import Array

NUM_LABELS = 4


def wrap_to_pi(x: Array) -> Array:
    """Wrap angles into the half-open interval (-pi, pi]."""
    return jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def angle_features(theta: jax.Array) -> jax.Array:
    return jnp.concatenate([jnp.sin(theta), jnp.cos(theta)])


def encode_labels(labels: jax.Array) -> jax.Array:
    one_hot = jax.nn.one_hot(labels, NUM_LABELS, dtype=jnp.float32)
    return one_hot.reshape(*labels.shape[:-1], labels.shape[-1] * NUM_LABELS)


class Torus(LieGroup):
    """Flat n-torus T^d as a Lie group."""

    d: int = eqx.field(static=True)

    def __init__(self, d: int):
        d = int(d)
        if d < 1:
            raise ValueError("Torus(d) requires d >= 1.")
        object.__setattr__(self, "d", d)

    @property
    def dimension(self) -> int:
        return self.d

    @override
    def frame(self, x: Array) -> Array:
        return jnp.eye(self.d, dtype=x.dtype)

    @override
    def to_frame(self, x: Array, v: Array) -> Array:
        del x
        return v

    @override
    def from_frame(self, x: Array, a: Array) -> Array:
        del x
        return a

    @override
    def retraction(self, x: Array, v: Array) -> Array:
        return wrap_to_pi(x + v)

    @override
    def chart_differential_inv(self, a: Array, b: Array) -> Array:
        del a
        return b


class GRUEncoder(eqx.Module):
    cell: eqx.nn.GRUCell
    proj: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, *, key):
        k1, k2 = jax.random.split(key)
        self.cell = eqx.nn.GRUCell(input_dim, hidden_dim, key=k1)
        self.proj = eqx.nn.Linear(hidden_dim, out_dim, key=k2)
        self.hidden_dim = hidden_dim

    def __call__(self, x_seq: jax.Array) -> jax.Array:
        def step(h, x):
            h = self.cell(x, h)
            return h, None

        h0 = jnp.zeros((self.hidden_dim,), dtype=x_seq.dtype)
        h_final, _ = jax.lax.scan(step, h0, x_seq)
        return self.proj(h_final)


class TorusDriftField(eqx.Module):
    mlp: eqx.nn.MLP
    geometry: Torus

    def __init__(self, geometry: Torus, ctx_dim: int, hidden_dim: int, *, key):
        d = geometry.dimension
        mlp = eqx.nn.MLP(
            in_size=2 * d + ctx_dim,
            out_size=d,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda layer: layer.bias, last, jnp.zeros_like(last.bias))
        self.mlp = eqx.tree_at(lambda model: model.layers[-1], mlp, last)
        self.geometry = geometry

    def __call__(self, t, theta, ctx):
        del t
        inp = jnp.concatenate([angle_features(theta), ctx])
        return self.geometry.from_frame(theta, self.mlp(inp))


class TorusDiffusionField(eqx.Module):
    mlp: eqx.nn.MLP
    geometry: Torus
    diffusion_scale: float

    def __init__(
        self,
        geometry: Torus,
        ctx_dim: int,
        hidden_dim: int,
        diffusion_scale: float,
        *,
        key,
    ):
        d = geometry.dimension
        mlp = eqx.nn.MLP(
            in_size=2 * d + ctx_dim,
            out_size=d,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.silu,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda layer: layer.weight, last, last.weight * 0.1)
        last = eqx.tree_at(lambda layer: layer.bias, last, jnp.full_like(last.bias, -2.0))
        self.mlp = eqx.tree_at(lambda model: model.layers[-1], mlp, last)
        self.geometry = geometry
        self.diffusion_scale = diffusion_scale

    def __call__(self, t, theta, ctx):
        del t
        inp = jnp.concatenate([angle_features(theta), ctx])
        scales = jax.nn.softplus(self.mlp(inp)) * self.diffusion_scale
        return self.geometry.frame(theta) * scales[None, :]


class RandomTorusSDEField(eqx.Module):
    """Drift plus a fixed random noise path for one sampled SDE trajectory."""

    drift: TorusDriftField
    diffusion: TorusDiffusionField
    dt: float = eqx.field(static=True)

    @property
    def geometry(self) -> Torus:
        return self.drift.geometry

    def __call__(self, t, theta, args):
        ctx, noise = args
        idx = jnp.floor(t / self.dt).astype(jnp.int32)
        idx = jnp.clip(idx, 0, noise.shape[0] - 1)
        drift = self.drift(t, theta, ctx)
        diffusion = self.diffusion(t, theta, ctx)
        return drift + diffusion @ noise[idx]


class RandomTorusNeuralSDE(eqx.Module):
    """Randomly initialised neural SDE on T^d."""

    encoder: GRUEncoder
    vector_field: RandomTorusSDEField
    name: str = eqx.field(static=True)

    d: int = eqx.field(static=True)
    labels_per_state: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    solver: AbstractSolver = eqx.field(static=True)
    adjoint: AbstractAdjoint | None = eqx.field(static=True)

    def __init__(
        self,
        num_angles: int,
        hidden_dim: int = 128,
        ctx_dim: int = 64,
        n_steps: int = 5,
        dt: float = 0.05,
        solver: AbstractSolver = CFEES25(),
        diffusion_scale: float = 0.1,
        labels_per_state: int = 1,
        adjoint: AbstractAdjoint | None = None,
        *,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        geometry = Torus(num_angles)
        label_dim = labels_per_state * NUM_LABELS
        self.name = "random_torus_nsde"
        self.d = geometry.dimension
        self.labels_per_state = labels_per_state
        self.n_steps = n_steps
        self.dt = dt
        self.solver = solver
        self.adjoint = adjoint
        self.encoder = GRUEncoder(2 * self.d + label_dim, hidden_dim, ctx_dim - label_dim, key=k1)
        drift_field = TorusDriftField(geometry, ctx_dim, hidden_dim, key=k2)
        diffusion_field = TorusDiffusionField(
            geometry,
            ctx_dim,
            hidden_dim,
            diffusion_scale,
            key=k3,
        )
        self.vector_field = RandomTorusSDEField(drift_field, diffusion_field, dt)

    def __call__(
        self,
        context_angles: jax.Array,
        context_labels: jax.Array,
        target_labels: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        y0 = wrap_to_pi(context_angles[-1])
        angle_seq = jax.vmap(angle_features)(context_angles)
        label_seq = encode_labels(context_labels)
        encoder_input = jnp.concatenate([angle_seq, label_seq], axis=-1)
        ctx_from_encoder = self.encoder(encoder_input)
        target_label_feat = encode_labels(target_labels)
        ctx = jnp.concatenate([ctx_from_encoder, target_label_feat])

        t1 = self.n_steps * self.dt
        noise = jax.random.normal(
            key,
            (self.n_steps + 1, self.d),
            dtype=context_angles.dtype,
        ) / jnp.sqrt(jnp.asarray(self.dt, dtype=context_angles.dtype))
        term = GeometricTerm(
            inner=ODETerm(self.vector_field),
            geometry=self.vector_field.geometry,
        )
        if self.adjoint is not None:
            adjoint = self.adjoint
        elif isinstance(self.solver, AbstractReversibleSolver):
            adjoint = ReversibleAdjoint()
        else:
            adjoint = DirectAdjoint()

        sol = diffeqsolve(
            term,
            self.solver,
            t0=0.0,
            t1=t1,
            dt0=self.dt,
            y0=y0,
            args=(ctx, noise),
            saveat=SaveAt(t1=True),
            adjoint=adjoint,
            max_steps=self.n_steps + 1,
        )
        return wrap_to_pi(sol.ys[0])
