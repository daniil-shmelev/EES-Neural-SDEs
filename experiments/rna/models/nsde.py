import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
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
from georax import CFEES25, CG2, SPD, GeometricTerm


def sym(x):
    return 0.5 * (x + x.T)


class DriftField(eqx.Module):
    """SPD drift field packaged as an explicit PyTree term."""

    mlp: eqx.nn.MLP
    geometry: SPD

    def __init__(self, geometry, state_dim, ctx_dim, hidden_dim, *, key):
        mlp = eqx.nn.MLP(
            in_size=state_dim + ctx_dim,
            out_size=geometry.dimension,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.weight, last, last.weight * 0.001)
        last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))

        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.geometry = geometry

    def _state_features(self, sigma):
        return jnp.reshape(sym(sigma), (-1,))

    def __call__(self, t, sigma, ctx):
        del t
        state = self._state_features(sigma)
        inp = jnp.concatenate([state, ctx])
        coeffs = self.mlp(inp)
        return self.geometry.from_frame(sigma, coeffs)


class DiffusionField(eqx.Module):
    """SPD diffusion field packaged as an explicit PyTree term."""

    mlp: eqx.nn.MLP
    geometry: SPD
    diffusion_scale: float

    def __init__(
        self,
        geometry,
        state_dim,
        ctx_dim,
        hidden_dim,
        diffusion_scale,
        *,
        key,
    ):
        mlp = eqx.nn.MLP(
            in_size=state_dim + ctx_dim,
            out_size=geometry.dimension,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.silu,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.weight, last, last.weight * 0.001)
        last = eqx.tree_at(lambda l: l.bias, last, jnp.full_like(last.bias, -3.0))

        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.geometry = geometry
        self.diffusion_scale = diffusion_scale

    def _state_features(self, sigma):
        return jnp.reshape(sym(sigma), (-1,))

    def __call__(self, t, sigma, ctx):
        del t
        state = self._state_features(sigma)
        inp = jnp.concatenate([state, ctx])
        scales = jax.nn.softplus(self.mlp(inp)) * self.diffusion_scale
        basis = self.geometry.frame(sigma)
        return basis * scales[None, None, :]


class GRUEncoder(eqx.Module):
    """GRU encoder over a sequence of flattened SPD matrices."""

    cell: eqx.nn.GRUCell
    proj: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)

    def __init__(self, input_dim, hidden_dim, ctx_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.cell = eqx.nn.GRUCell(input_dim, hidden_dim, key=k1)
        self.proj = eqx.nn.Linear(hidden_dim, ctx_dim, key=k2)
        self.hidden_dim = hidden_dim

    def __call__(self, x_seq):
        """x_seq: (seq_len, input_dim) -> ctx: (ctx_dim,)"""

        def step(h, x):
            h = self.cell(x, h)
            return h, None

        h0 = jnp.zeros((self.hidden_dim,), dtype=x_seq.dtype)
        h_final, _ = jax.lax.scan(step, h0, x_seq)
        return self.proj(h_final)


class ManifoldNeuralSDE(eqx.Module):
    """Pure georax neural SDE on SPD(n)."""

    encoder: GRUEncoder
    drift_field: DriftField
    diffusion_field: DiffusionField
    name: str = eqx.field(static=True)

    n: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    ctx_dim: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    solver: AbstractSolver = eqx.field(static=True)

    def __init__(
        self,
        n_stocks=5,
        hidden_dim=128,
        ctx_dim=64,
        n_steps=5,
        dt=0.2,
        solver: AbstractSolver = CFEES25(),
        diffusion_scale=1.0,
        *,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)

        geometry = SPD(n_stocks)
        d = geometry.dimension
        state_dim = n_stocks * n_stocks

        self.name = "neural_sde"
        self.n = n_stocks
        self.d = d
        self.state_dim = state_dim
        self.ctx_dim = ctx_dim
        self.n_steps = n_steps
        self.dt = dt
        self.solver = solver

        self.encoder = GRUEncoder(state_dim, hidden_dim, ctx_dim, key=k1)

        self.drift_field = DriftField(
            geometry=geometry,
            state_dim=state_dim,
            ctx_dim=ctx_dim,
            hidden_dim=hidden_dim,
            key=k2,
        )
        self.diffusion_field = DiffusionField(
            geometry=geometry,
            state_dim=state_dim,
            ctx_dim=ctx_dim,
            hidden_dim=hidden_dim,
            diffusion_scale=diffusion_scale,
            key=k3,
        )

    def _state_features(self, sigma):
        """Pure state encoding: symmetric matrix flattened to R^(n*n)."""
        return jnp.reshape(sym(sigma), (-1,))

    def __call__(self, context_spd, key):
        """Args:
        context_spd: (window, n, n) past SPD matrices
        key: PRNG key

        Returns:
        (n, n) predicted SPD matrix
        """
        y0 = sym(context_spd[-1])

        context_features = jax.vmap(self._state_features)(context_spd)
        ctx = self.encoder(context_features)

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

        adjoint = (
            ReversibleAdjoint()
            if isinstance(self.solver, AbstractReversibleSolver)
            else DirectAdjoint()
        )

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
            max_steps=self.n_steps + 8,
        )

        return sym(sol.ys[0])
