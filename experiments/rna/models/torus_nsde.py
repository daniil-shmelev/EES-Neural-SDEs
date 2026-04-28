"""Neural SDE on the flat n-torus T^d = SO(2)^d for RNA torsion forecasting."""

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

from experiments.rna.datasets.preprocessing import NUM_BASES
from experiments.rna.models.torus import Torus, wrap_to_pi

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


def angle_features(theta: jax.Array) -> jax.Array:
    """Standard sin/cos encoding for inputs that live on a torus."""
    return jnp.concatenate([jnp.sin(theta), jnp.cos(theta)])


def encode_bases(bases: jax.Array) -> jax.Array:
    """One-hot encode an int array of base ids and flatten the trailing axis."""
    one_hot = jax.nn.one_hot(bases, NUM_BASES, dtype=jnp.float32)
    return one_hot.reshape(*bases.shape[:-1], bases.shape[-1] * NUM_BASES)


class TorusDriftField(eqx.Module):
    mlp: eqx.nn.MLP
    geometry: Torus

    def __init__(
        self,
        geometry,
        ctx_dim,
        hidden_dim,
        *,
        depth: int = 3,
        activation=jax.nn.silu,
        key,
    ):
        d = geometry.dimension
        mlp = eqx.nn.MLP(
            in_size=2 * d + ctx_dim,
            out_size=d,
            width_size=hidden_dim,
            depth=depth,
            activation=activation,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.geometry = geometry

    def __call__(self, t, theta, ctx):
        del t
        inp = jnp.concatenate([angle_features(theta), ctx])
        coeffs = self.mlp(inp)
        return self.geometry.from_frame(theta, coeffs)


class TorusDiffusionField(eqx.Module):
    mlp: eqx.nn.MLP
    geometry: Torus
    diffusion_scale: float

    def __init__(
        self,
        geometry,
        ctx_dim,
        hidden_dim,
        diffusion_scale,
        *,
        depth: int = 2,
        activation=jax.nn.silu,
        key,
    ):
        d = geometry.dimension
        mlp = eqx.nn.MLP(
            in_size=2 * d + ctx_dim,
            out_size=d,
            width_size=hidden_dim,
            depth=depth,
            activation=activation,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.weight, last, last.weight * 0.1)
        last = eqx.tree_at(lambda l: l.bias, last, jnp.full_like(last.bias, -2.0))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.geometry = geometry
        self.diffusion_scale = diffusion_scale

    def __call__(self, t, theta, ctx):
        del t
        inp = jnp.concatenate([angle_features(theta), ctx])
        scales = jax.nn.softplus(self.mlp(inp)) * self.diffusion_scale
        basis = self.geometry.frame(theta)  # (d, d) identity
        return basis * scales[None, :]


class GRUEncoder(eqx.Module):
    """GRU encoder over a sequence of per-residue feature vectors."""

    cell: eqx.nn.GRUCell
    proj: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)

    def __init__(self, input_dim, hidden_dim, ctx_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.cell = eqx.nn.GRUCell(input_dim, hidden_dim, key=k1)
        self.proj = eqx.nn.Linear(hidden_dim, ctx_dim, key=k2)
        self.hidden_dim = hidden_dim

    def __call__(self, x_seq):
        def step(h, x):
            h = self.cell(x, h)
            return h, None

        h0 = jnp.zeros((self.hidden_dim,), dtype=x_seq.dtype)
        h_final, _ = jax.lax.scan(step, h0, x_seq)
        return self.proj(h_final)


class FutureBaseEncoder(eqx.Module):
    """Small MLP over the flattened (one-hot future bases, mask) vector."""

    mlp: eqx.nn.MLP

    def __init__(
        self, window: int, bases_per_state: int, out_dim: int, *, activation=jax.nn.silu, key
    ):
        in_dim = window * (bases_per_state * NUM_BASES + 1)
        self.mlp = eqx.nn.MLP(
            in_size=in_dim,
            out_size=out_dim,
            width_size=max(out_dim * 2, 32),
            depth=2,
            activation=activation,
            key=key,
        )

    def __call__(self, future_bases: jax.Array, future_mask: jax.Array) -> jax.Array:
        """future_bases: (F, bases_per_state) int; future_mask: (F,) int."""
        encoded = encode_bases(future_bases)  # (F, bases_per_state * NUM_BASES)
        mask = future_mask.astype(jnp.float32)[:, None]
        gated = encoded * mask
        inp = jnp.concatenate([gated.reshape(-1), mask.reshape(-1)])
        return self.mlp(inp)


class TorusNeuralSDE(eqx.Module):
    """Neural SDE on T^d for one-step torsion-angle forecasting."""

    encoder: GRUEncoder
    future_encoder: FutureBaseEncoder | None
    drift_field: TorusDriftField
    diffusion_field: TorusDiffusionField
    name: str = eqx.field(static=True)

    d: int = eqx.field(static=True)
    bases_per_state: int = eqx.field(static=True)
    ctx_dim: int = eqx.field(static=True)
    future_bases_window: int = eqx.field(static=True)
    future_ctx_dim: int = eqx.field(static=True)
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
        diffusion_scale: float = 0.3,
        adjoint: AbstractAdjoint | None = None,
        residues_per_state: int = 1,
        future_bases_window: int = 0,
        future_ctx_dim: int = 32,
        activation: str = "silu",
        drift_depth: int = 3,
        diffusion_depth: int = 2,
        *,
        key,
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        geometry = Torus(num_angles)
        d = geometry.dimension
        bases_per_state = residues_per_state
        base_feature_dim = bases_per_state * NUM_BASES
        act_fn = resolve_activation(activation)

        self.name = "torus_nsde"
        self.d = d
        self.bases_per_state = bases_per_state
        self.ctx_dim = ctx_dim
        self.future_bases_window = future_bases_window
        # Future encoder contributes extra dims only when enabled.
        effective_future_ctx = future_ctx_dim if future_bases_window > 0 else 0
        self.future_ctx_dim = effective_future_ctx
        self.n_steps = n_steps
        self.dt = dt
        self.solver = solver
        self.adjoint = adjoint

        # Encoder consumes per-step (sin/cos angle features, one-hot bases).
        encoder_input_dim = 2 * d + base_feature_dim
        self.encoder = GRUEncoder(
            encoder_input_dim, hidden_dim, ctx_dim - base_feature_dim, key=k1
        )
        if future_bases_window > 0:
            self.future_encoder = FutureBaseEncoder(
                window=future_bases_window,
                bases_per_state=bases_per_state,
                out_dim=future_ctx_dim,
                activation=act_fn,
                key=k4,
            )
        else:
            self.future_encoder = None
        # Drift/diffusion see (sin/cos angle features, ctx) where ctx carries
        # the encoder output, the target residue's base one-hot, and
        # (optionally) the future-base encoder's output.
        total_ctx_dim = ctx_dim + effective_future_ctx
        self.drift_field = TorusDriftField(
            geometry=geometry,
            ctx_dim=total_ctx_dim,
            hidden_dim=hidden_dim,
            depth=drift_depth,
            activation=act_fn,
            key=k2,
        )
        self.diffusion_field = TorusDiffusionField(
            geometry=geometry,
            ctx_dim=total_ctx_dim,
            hidden_dim=hidden_dim,
            diffusion_scale=diffusion_scale,
            depth=diffusion_depth,
            activation=act_fn,
            key=k3,
        )

    def __call__(
        self,
        context_angles: jax.Array,
        context_bases: jax.Array,
        target_bases: jax.Array,
        future_bases: jax.Array,
        future_mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Forecast the next state's angles.

        context_angles: (T, d)
        context_bases:  (T, bases_per_state) int
        target_bases:   (bases_per_state,) int
        future_bases:   (future_bases_window, bases_per_state) int
        future_mask:    (future_bases_window,) int
        key:            PRNG key
        """
        y0 = wrap_to_pi(context_angles[-1])

        angle_seq = jax.vmap(angle_features)(context_angles)  # (T, 2d)
        base_seq = encode_bases(context_bases)  # (T, bases_per_state * 4)
        encoder_input = jnp.concatenate([angle_seq, base_seq], axis=-1)
        ctx_from_encoder = self.encoder(encoder_input)  # (ctx_dim - base_feat,)
        target_base_feat = encode_bases(target_bases)  # (bases_per_state * 4,)
        ctx = jnp.concatenate([ctx_from_encoder, target_base_feat])
        if self.future_encoder is not None:
            future_feat = self.future_encoder(future_bases, future_mask)
            ctx = jnp.concatenate([ctx, future_feat])

        t1 = self.n_steps * self.dt
        brownian_path = VirtualBrownianTree(
            t0=0.0,
            t1=t1,
            tol=self.dt / 4.0,
            shape=(self.d,),
            key=key,
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
            term,
            self.solver,
            t0=0.0,
            t1=t1,
            dt0=self.dt,
            y0=y0,
            args=ctx,
            saveat=SaveAt(t1=True),
            adjoint=adjoint,
            max_steps=self.n_steps + 1,
        )

        return wrap_to_pi(sol.ys[0])
