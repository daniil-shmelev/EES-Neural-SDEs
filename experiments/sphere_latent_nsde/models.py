"""Equinox models for HumanActivity latent SDE classification on a sphere."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from diffrax import (
    AbstractAdjoint,
    AbstractReversibleSolver,
    AbstractSolver,
    ControlTerm,
    DirectAdjoint,
    GeneralShARK,
    MultiTerm,
    RecursiveCheckpointAdjoint,
    ReversibleAdjoint,
    SaveAt,
    SpaceTimeLevyArea,
    VirtualBrownianTree,
    diffeqsolve,
)
from georax import CFEES25, CG2, GeometricEuler, GeometricTerm, SRKMK
from jaxtyping import Array

from experiments.sphere_latent_nsde.custom_vbt import (
    cfees25_integrate_reversible,
    step_noise,
)
from experiments.sphere_latent_nsde.config import SOLVER_NFE_PER_STEP
from experiments.sphere_latent_nsde.dataset import INPUT_DIM, NUM_CLASSES, NUM_TIMEPOINTS
from experiments.sphere_latent_nsde.distributions import PowerSpherical
from experiments.sphere_latent_nsde.geometry import (
    Sphere,
    normalize,
    vec_to_matrix,
)


def apply_linear(linear: eqx.nn.Linear, x: Array) -> Array:
    y = jnp.einsum("...i,oi->...o", x, linear.weight)
    if linear.bias is not None:
        y = y + linear.bias
    return y


_BBC_SQRT177 = math.sqrt(177.0)
_BBC_X1 = 1.0
_BBC_X2 = 0.25
_BBC_X3 = (-1.0 + _BBC_SQRT177) / 2.0
_BBC_X5 = 11.0 / 630.0
_BBC_X7 = 1.0 / 2520.0
_BBC_X4 = 1.0 / 6.0 - _BBC_X5 * _BBC_X3
_BBC_X6 = (10.0 - _BBC_X3) / 2520.0
_BBC_Y2 = 0.5 - _BBC_X3 * _BBC_X4


def _bbc_expm_8(a: Array) -> Array:
    """BBC/Sastre matrix-exponential polynomial used by the Torch baseline."""

    ident = jnp.eye(a.shape[-1], dtype=a.dtype)
    a2 = a @ a
    a4 = a2 @ (_BBC_X1 * a + _BBC_X2 * a2)
    a8 = (_BBC_X3 * a2 + a4) @ (
        _BBC_X4 * ident + _BBC_X5 * a + _BBC_X6 * a2 + _BBC_X7 * a4
    )
    return ident + a + _BBC_Y2 * a2 + a8


def _apply_bbc_increment(z: Array, coeffs: Array, basis: Array) -> Array:
    omega = vec_to_matrix(coeffs, basis)
    q = _bbc_expm_8(omega)
    return jnp.einsum("...ij,...j->...i", q, z)


class _SRKMKWithScanKind(SRKMK):
    scan_kind: Literal["lax", "checkpointed", "bounded"] | None

    def __init__(
        self,
        solver: AbstractSolver,
        *,
        additive_after_pullback: bool = False,
        scan_kind: Literal["lax", "checkpointed", "bounded"] | None = None,
    ):
        super().__init__(solver, additive_after_pullback=additive_after_pullback)
        object.__setattr__(self, "scan_kind", scan_kind)


def build_solver(
    name: Literal["geometric_euler", "cg2", "cfees25", "srkmk_general_shark"],
) -> AbstractSolver:
    if name == "geometric_euler":
        return GeometricEuler()
    if name == "cg2":
        return CG2()
    if name == "cfees25":
        return CFEES25()
    if name == "srkmk_general_shark":
        return _SRKMKWithScanKind(GeneralShARK())
    raise ValueError(f"unknown solver {name!r}")


def build_adjoint(
    solver: AbstractSolver,
    name: Literal[
        "auto",
        "direct",
        "recursive_checkpoint",
        "checkpoint_recursive",
        "checkpoint_full",
        "full",
        "reversible",
    ] = "auto",
    *,
    max_steps: int | None = None,
) -> AbstractAdjoint:
    if name == "direct":
        return DirectAdjoint()
    if name in ("recursive_checkpoint", "checkpoint_recursive"):
        return RecursiveCheckpointAdjoint()
    if name in ("checkpoint_full", "full"):
        if max_steps is None:
            raise ValueError("checkpoint_full adjoint requires max_steps.")
        return RecursiveCheckpointAdjoint(checkpoints=max_steps)
    if name == "reversible":
        return ReversibleAdjoint()
    if isinstance(solver, AbstractReversibleSolver):
        return ReversibleAdjoint()
    return DirectAdjoint()


def _uses_diffrax_checkpoint_adjoint(adjoint: AbstractAdjoint) -> bool:
    return isinstance(adjoint, RecursiveCheckpointAdjoint)


def _interpolate_paths_to_grid(
    paths: Array,
    source_times: Array,
    target_times: Array,
    geometry: Sphere,
) -> Array:
    """Linearly interpolate ambient coordinates, then project to the sphere."""

    paths_by_time = jnp.moveaxis(paths, 2, 0)
    flat_paths = paths_by_time.reshape((paths_by_time.shape[0], -1))
    flat_interp = jax.vmap(
        lambda values: jnp.interp(target_times, source_times, values),
        in_axes=1,
        out_axes=1,
    )(flat_paths)
    out = flat_interp.reshape((target_times.shape[0],) + paths_by_time.shape[1:])
    out = jnp.moveaxis(out, 0, 2)
    return geometry.project(out)


def _sphere_drift_coeffs(t, y, args):
    time_fn, h, _sigma = args
    time_steps = jnp.asarray([t], dtype=h.dtype)

    if h.ndim == 1:
        coeffs = time_fn(h, time_steps)[0]
    else:
        flat_h = h.reshape((-1, h.shape[-1]))
        coeffs = jax.vmap(lambda one_h: time_fn(one_h, time_steps)[0])(flat_h)
        coeffs = coeffs.reshape(h.shape[:-1] + (time_fn.out_dim,))

    return jnp.broadcast_to(coeffs, y.shape[:-1] + (time_fn.out_dim,))


def _sphere_diffusion_coeffs(t, y, args):
    del t
    time_fn, _h, sigma = args
    diagonal = sigma.astype(y.dtype) * jnp.ones(
        y.shape[:-1] + (time_fn.out_dim,), dtype=y.dtype
    )
    return lx.DiagonalLinearOperator(diagonal)


class MultiTimeAttention(eqx.Module):
    q_linear: eqx.nn.Linear
    k_linear: eqx.nn.Linear
    out_linear: eqx.nn.Linear
    embed_time: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    embed_time_k: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        embed_time: int = 16,
        num_heads: int = 1,
        *,
        key: Array,
    ) -> None:
        if embed_time % num_heads != 0:
            raise ValueError("embed_time must be divisible by num_heads.")
        k1, k2, k3 = jax.random.split(key, 3)
        self.q_linear = eqx.nn.Linear(embed_time, embed_time, key=k1)
        self.k_linear = eqx.nn.Linear(embed_time, embed_time, key=k2)
        self.out_linear = eqx.nn.Linear(input_dim * num_heads, hidden_dim, key=k3)
        self.embed_time = embed_time
        self.num_heads = num_heads
        self.embed_time_k = embed_time // num_heads

    def __call__(
        self,
        query: Array,
        key_times: Array,
        value: Array,
        mask: Array | None = None,
    ) -> Array:
        q = apply_linear(self.q_linear, query)
        k = apply_linear(self.k_linear, key_times)
        q = q.reshape(q.shape[0], self.num_heads, self.embed_time_k).transpose(1, 0, 2)
        k = k.reshape(k.shape[0], self.num_heads, self.embed_time_k).transpose(1, 0, 2)

        scores = jnp.einsum("hqe,hte->hqt", q, k) / jnp.sqrt(
            jnp.asarray(self.embed_time_k, dtype=value.dtype)
        )
        scores = jnp.broadcast_to(scores[..., None], scores.shape + value.shape[-1:])
        if mask is not None:
            scores = jnp.where(mask[None, None, :, :] == 0, -1e9, scores)
        weights = jax.nn.softmax(scores, axis=-2)
        attended = jnp.sum(weights * value[None, None, :, :], axis=-2)
        attended = attended.transpose(1, 0, 2).reshape(query.shape[0], -1)
        return apply_linear(self.out_linear, attended)


class TorchGRUCell(eqx.Module):
    """GRU cell with PyTorch's two-bias parameterization and gate equations."""

    weight_ih: Array
    weight_hh: Array
    bias_ih: Array
    bias_hh: Array
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, input_size: int, hidden_size: int, *, key: Array) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        limit = math.sqrt(1.0 / hidden_size)
        self.weight_ih = jax.random.uniform(
            k1,
            (3 * hidden_size, input_size),
            minval=-limit,
            maxval=limit,
        )
        self.weight_hh = jax.random.uniform(
            k2,
            (3 * hidden_size, hidden_size),
            minval=-limit,
            maxval=limit,
        )
        self.bias_ih = jax.random.uniform(
            k3,
            (3 * hidden_size,),
            minval=-limit,
            maxval=limit,
        )
        self.bias_hh = jax.random.uniform(
            k4,
            (3 * hidden_size,),
            minval=-limit,
            maxval=limit,
        )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

    def __call__(self, x: Array, h: Array) -> Array:
        i_r, i_z, i_n = jnp.split(self.weight_ih @ x + self.bias_ih, 3)
        h_r, h_z, h_n = jnp.split(self.weight_hh @ h + self.bias_hh, 3)
        reset = jax.nn.sigmoid(i_r + h_r)
        update = jax.nn.sigmoid(i_z + h_z)
        new = jnp.tanh(i_n + reset * h_n)
        return new + update * (h - new)


class MTANEncoder(eqx.Module):
    attention: MultiTimeAttention
    gru_cell: TorchGRUCell
    linear_time: eqx.nn.Linear
    periodic_time: eqx.nn.Linear
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    embed_time: int = eqx.field(static=True)
    query_size: int = eqx.field(static=True)
    learn_emb: bool = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        embed_time: int = 128,
        query_size: int = 128,
        num_heads: int = 1,
        learn_emb: bool = True,
        *,
        key: Array,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.attention = MultiTimeAttention(
            2 * input_dim,
            hidden_dim=hidden_dim,
            embed_time=embed_time,
            num_heads=num_heads,
            key=k1,
        )
        self.gru_cell = TorchGRUCell(hidden_dim, hidden_dim, key=k2)
        self.linear_time = eqx.nn.Linear(1, 1, key=k3)
        self.periodic_time = eqx.nn.Linear(1, embed_time - 1, key=k4)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_time = embed_time
        self.query_size = query_size
        self.learn_emb = learn_emb

    def _learn_time_embedding(self, tt: Array) -> Array:
        tt = tt[..., None]
        linear = apply_linear(self.linear_time, tt)
        periodic = jnp.sin(apply_linear(self.periodic_time, tt))
        return jnp.concatenate([linear, periodic], axis=-1)

    def _fixed_time_embedding(self, tt: Array) -> Array:
        position = 48.0 * tt[..., None]
        div_term = jnp.exp(
            jnp.arange(0, self.embed_time, 2, dtype=tt.dtype)
            * -(jnp.log(10.0) / self.embed_time)
        )
        pe = jnp.zeros(tt.shape + (self.embed_time,), dtype=tt.dtype)
        pe = pe.at[..., 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[..., 1::2].set(jnp.cos(position * div_term))
        return pe

    def _time_embedding(self, tt: Array) -> Array:
        if self.learn_emb:
            return self._learn_time_embedding(tt)
        return self._fixed_time_embedding(tt)

    def __call__(self, x: Array, time_steps: Array) -> Array:
        mask = x[:, self.input_dim :]
        mask = jnp.concatenate([mask, mask], axis=-1)
        key_emb = self._time_embedding(time_steps)
        query = jnp.linspace(0.0, 1.0, self.query_size, dtype=x.dtype)
        query_emb = self._time_embedding(query)
        attended = self.attention(query_emb, key_emb, x, mask)

        def step(h, x_t):
            h_next = self.gru_cell(x_t, h)
            return h_next, None

        h0 = jnp.zeros((self.hidden_dim,), dtype=x.dtype)
        h_final, _ = jax.lax.scan(step, h0, attended)
        return h_final


class ActivityRecogNetwork(eqx.Module):
    mtan: MTANEncoder
    use_atanh: bool = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 32,
        use_atanh: bool = False,
        *,
        key: Array,
    ) -> None:
        self.mtan = MTANEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embed_time=128,
            query_size=128,
            num_heads=1,
            learn_emb=True,
            key=key,
        )
        self.use_atanh = use_atanh

    def __call__(self, observed_data: Array, observed_mask: Array, observed_tp: Array) -> Array:
        h = self.mtan(
            jnp.concatenate([observed_data, observed_mask.astype(observed_data.dtype)], axis=-1),
            observed_tp,
        )
        if self.use_atanh:
            eps = jnp.asarray(1e-5, dtype=h.dtype)
            h = jnp.arctanh(h - jnp.sign(h) * eps)
        return h


class Chebyshev(eqx.Module):
    linear: eqx.nn.Linear
    degree: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    time_min: float = eqx.field(static=True)
    time_max: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        degree: int,
        out_dim: int,
        *,
        time_min: float = 0.0,
        time_max: float = 1.0,
        key: Array,
    ) -> None:
        self.linear = eqx.nn.Linear(input_dim, degree * out_dim, key=key)
        self.degree = int(degree)
        self.out_dim = int(out_dim)
        self.time_min = float(time_min)
        self.time_max = float(time_max)

    def interval_transform(self, t: Array) -> Array:
        return (t - self.time_min) / (self.time_max - self.time_min)

    def __call__(self, h: Array, time_steps: Array) -> Array:
        time_steps = self.interval_transform(time_steps)
        degrees = jnp.arange(self.degree, dtype=time_steps.dtype)
        monomials = jnp.cos(jnp.outer(jnp.arccos(time_steps), degrees))
        coeffs = apply_linear(self.linear, h).reshape(self.out_dim, self.degree)
        return (coeffs @ monomials.T).T


class PathEncoder(eqx.Module):
    loc_map: eqx.nn.Linear
    scale_map: eqx.nn.Linear
    time_fn: Chebyshev
    geometry: Sphere = eqx.field(static=True)
    sigma: Array
    learnable_prior: bool = eqx.field(static=True)
    prior_h: Array | None

    def __init__(
        self,
        *,
        h_dim: int,
        z_dim: int,
        n_deg: int,
        learnable_prior: bool = False,
        time_min: float = 0.0,
        time_max: float = 1.98,
        key: Array,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        geometry = Sphere(z_dim)
        group_dim = geometry.lie_algebra_dimension
        self.loc_map = eqx.nn.Linear(h_dim, z_dim, key=k1)
        self.scale_map = eqx.nn.Linear(h_dim, 1, key=k2)
        self.time_fn = Chebyshev(
            h_dim,
            n_deg,
            group_dim,
            time_min=time_min,
            time_max=time_max,
            key=k3,
        )
        self.geometry = geometry
        self.sigma = jnp.asarray(0.1, dtype=jnp.float32)
        self.learnable_prior = learnable_prior
        self.prior_h = (
            jax.random.normal(k4, (h_dim,), dtype=jnp.float32)
            if learnable_prior
            else None
        )

    def posterior_initial(self, h: Array) -> PowerSpherical:
        loc = normalize(apply_linear(self.loc_map, h))
        scale = jnp.square(apply_linear(self.scale_map, h)[..., 0]) * 100.0
        scale = jnp.minimum(jnp.maximum(scale, 1e-6), 50000.0)
        return PowerSpherical(loc=loc, scale=scale)

    def posterior_drift(self, h: Array, times: Array) -> Array:
        return self.time_fn(h, times)

    def prior_drift(self, times: Array) -> Array:
        if self.learnable_prior:
            assert self.prior_h is not None
            return self.time_fn(self.prior_h, times)
        return jnp.zeros((times.shape[0], self.geometry.dimension), dtype=times.dtype)

    def _posterior_drift_batched(self, h: Array, times: Array) -> Array:
        if h.ndim == 1:
            return self.posterior_drift(h, times)
        flat_h = h.reshape((-1, h.shape[-1]))
        drift = jax.vmap(lambda one_h: self.posterior_drift(one_h, times))(flat_h)
        return drift.reshape(h.shape[:-1] + drift.shape[1:])

    def integrate_paths(
        self,
        h: Array,
        z0: Array,
        key: Array,
        *,
        times: Array,
        solver: AbstractSolver,
        adjoint: AbstractAdjoint,
    ) -> Array:
        """Integrate all Monte Carlo samples and batch elements in one solve.

        ``z0`` is shaped ``(mc, batch, z_dim)`` and the returned path is shaped
        ``(mc, batch, time, z_dim)``.
        """

        # Checkpoint adjoints must go through Diffrax so its tape policy is used.
        if isinstance(solver, GeometricEuler) and not _uses_diffrax_checkpoint_adjoint(
            adjoint
        ):
            return self._integrate_paths_geometric_euler(h, z0, key, times=times)
        if isinstance(solver, CFEES25) and isinstance(adjoint, ReversibleAdjoint):
            return self._integrate_paths_cfees25(h, z0, key, times=times)

        dt = times[1] - times[0]
        t0 = times[0]
        t1 = times[-1]
        max_steps = int(times.shape[0]) + 8
        driver_dim = self.geometry.dimension
        z0 = self.geometry.project(z0)
        solve_args = (self.time_fn, h, self.sigma.astype(z0.dtype))

        brownian_shape = z0.shape[:-1] + (driver_dim,)
        brownian = (
            VirtualBrownianTree(
                t0=t0,
                t1=t1,
                tol=dt / 4.0,
                shape=brownian_shape,
                key=key,
                levy_area=SpaceTimeLevyArea,
            )
            if isinstance(solver, SRKMK)
            else VirtualBrownianTree(
                t0=t0,
                t1=t1,
                tol=dt / 4.0,
                shape=brownian_shape,
                key=key,
            )
        )
        term = MultiTerm(
            GeometricTerm(_sphere_drift_coeffs, self.geometry),
            ControlTerm(_sphere_diffusion_coeffs, brownian),
        )
        sol = diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=z0,
            args=solve_args,
            saveat=SaveAt(ts=times),
            adjoint=adjoint,
            max_steps=max_steps,
        )
        return jnp.moveaxis(sol.ys, 0, 2)

    def _integrate_paths_geometric_euler(
        self,
        h: Array,
        z0: Array,
        key: Array,
        *,
        times: Array,
    ) -> Array:
        """Fixed-grid geometric Euler with replayable independent increments."""

        dt = jnp.diff(times)
        z0 = self.geometry.project(z0)
        driver_dim = self.geometry.dimension
        drift = self._posterior_drift_batched(h, times[:-1])
        drift = jnp.broadcast_to(drift, z0.shape[:-1] + drift.shape[-2:])
        drift = jnp.moveaxis(drift, -2, 0)
        sigma = self.sigma.astype(z0.dtype)
        noise_shape = z0.shape[:-1] + (driver_dim,)

        def step(z, inputs):
            step_index, dt_i, drift_i = inputs
            noise = step_noise(key, step_index, noise_shape, z.dtype)
            increment = drift_i * dt_i + sigma * jnp.sqrt(dt_i) * noise
            z_next = _apply_bbc_increment(z, increment, self.geometry.basis)
            return z_next, z_next

        step_indices = jnp.arange(times.shape[0] - 1)
        _, zs = jax.lax.scan(step, z0, (step_indices, dt, drift))
        zs = jnp.concatenate([z0[None], zs], axis=0)
        return jnp.moveaxis(zs, 0, 2)

    def _integrate_paths_cfees25(
        self,
        h: Array,
        z0: Array,
        key: Array,
        *,
        times: Array,
    ) -> Array:
        """Fixed-grid CFEES25 with replayable Brownian increments."""

        return cfees25_integrate_reversible(
            (self.time_fn, h, z0, self.sigma.astype(z0.dtype)),
            times,
            self.geometry.basis,
            key,
        )

    def path_kl(self, paths: Array, h: Array, times: Array) -> Array:
        grid = times[:-1]
        dt = jnp.diff(times)
        prior = self.prior_drift(grid)
        if h.ndim == 1:
            delta = self.posterior_drift(h, grid) - prior
            drift_mats = vec_to_matrix(delta, self.geometry.basis)
            projection = jnp.einsum("tij,stj->sti", drift_mats, paths[:, :-1, :])
            norm_squared = jnp.sum(projection**2, axis=-1)
            per_sample = jnp.sum(norm_squared * dt[None, :], axis=-1) / jnp.square(
                self.sigma
            )
            return jnp.mean(per_sample)

        posterior = self._posterior_drift_batched(h, grid)
        delta = posterior - prior[None, :, :]
        drift_mats = vec_to_matrix(delta, self.geometry.basis)
        projection = jnp.einsum("btij,sbtj->sbti", drift_mats, paths[:, :, :-1, :])
        norm_squared = jnp.sum(projection**2, axis=-1)
        per_sample = jnp.sum(norm_squared * dt[None, None, :], axis=-1) / jnp.square(
            self.sigma
        )
        return jnp.mean(per_sample, axis=0)


@dataclass(frozen=True)
class LatentSDEOutput:
    paths: Array
    recon_mu: Array
    aux_logits: Array
    kl0: Array
    klp: Array


class ActivityLatentSDE(eqx.Module):
    recog_net: ActivityRecogNetwork
    path_encoder: PathEncoder
    recon_net: eqx.nn.Linear
    aux_net: eqx.nn.Linear
    solver: AbstractSolver = eqx.field(static=True)
    adjoint: AbstractAdjoint = eqx.field(static=True)
    z_dim: int = eqx.field(static=True)
    num_timepoints: int = eqx.field(static=True)
    nfe_budget: int = eqx.field(static=True)
    nfe_per_step: int = eqx.field(static=True)
    solve_n_steps: int = eqx.field(static=True)
    solve_num_timepoints: int = eqx.field(static=True)
    time_end: float = eqx.field(static=True)
    recon_sigma: Array

    def __init__(
        self,
        *,
        input_dim: int = INPUT_DIM,
        num_classes: int = NUM_CLASSES,
        num_timepoints: int = NUM_TIMEPOINTS,
        h_dim: int = 32,
        z_dim: int = 16,
        n_deg: int = 4,
        nfe_budget: int | None = None,
        solver_name: Literal[
            "geometric_euler",
            "cg2",
            "cfees25",
            "srkmk_general_shark",
        ] = "geometric_euler",
        adjoint_name: Literal[
            "auto",
            "direct",
            "recursive_checkpoint",
            "checkpoint_recursive",
            "checkpoint_full",
            "full",
            "reversible",
        ] = "auto",
        learnable_prior: bool = False,
        use_atanh: bool = False,
        key: Array,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        solver = build_solver(solver_name)
        if nfe_budget is None:
            nfe_budget = int(num_timepoints) - 1
        nfe_per_step = SOLVER_NFE_PER_STEP[solver_name]
        if nfe_budget <= 0:
            raise ValueError("nfe_budget must be positive.")
        if nfe_budget % nfe_per_step != 0:
            raise ValueError(
                f"nfe_budget={nfe_budget} must be divisible by {nfe_per_step} "
                f"for solver={solver_name!r}."
            )
        self.recog_net = ActivityRecogNetwork(
            input_dim=input_dim,
            hidden_dim=h_dim,
            use_atanh=use_atanh,
            key=k1,
        )
        self.path_encoder = PathEncoder(
            h_dim=h_dim,
            z_dim=z_dim,
            n_deg=n_deg,
            learnable_prior=learnable_prior,
            time_min=0.0,
            time_max=1.98,
            key=k2,
        )
        self.recon_net = eqx.nn.Linear(z_dim, input_dim, key=k3)
        self.aux_net = eqx.nn.Linear(z_dim, num_classes, key=k4)
        solve_n_steps = int(nfe_budget) // int(nfe_per_step)
        max_steps = solve_n_steps + 1 + 8
        self.solver = solver
        self.adjoint = build_adjoint(solver, adjoint_name, max_steps=max_steps)
        self.z_dim = z_dim
        self.num_timepoints = int(num_timepoints)
        self.nfe_budget = int(nfe_budget)
        self.nfe_per_step = int(nfe_per_step)
        self.solve_n_steps = solve_n_steps
        self.solve_num_timepoints = self.solve_n_steps + 1
        self.time_end = 0.99
        self.recon_sigma = jnp.asarray(1.0, dtype=jnp.float32)

    def desired_times(self, dtype=jnp.float32) -> Array:
        return jnp.linspace(0.0, self.time_end, self.num_timepoints, dtype=dtype)

    def solve_times(self, dtype=jnp.float32) -> Array:
        return jnp.linspace(0.0, self.time_end, self.solve_num_timepoints, dtype=dtype)

    def __call__(self, batch: dict[str, Array], key: Array, *, mc_samples: int = 1) -> LatentSDEOutput:
        h = jax.vmap(self.recog_net)(
            batch["inp_obs"], batch["inp_msk"], batch["inp_tps"]
        )
        posterior = self.path_encoder.posterior_initial(h)
        k_z0, k_path = jax.random.split(key)
        z0 = posterior.rsample(k_z0, (mc_samples,))
        output_times = self.desired_times(dtype=batch["inp_obs"].dtype)
        solve_times = self.solve_times(dtype=batch["inp_obs"].dtype)
        solve_paths = self.path_encoder.integrate_paths(
            h,
            z0,
            k_path,
            times=solve_times,
            solver=self.solver,
            adjoint=self.adjoint,
        )
        kl0 = posterior.kl_to_uniform()
        klp = self.path_encoder.path_kl(solve_paths, h, solve_times)
        if self.solve_num_timepoints == self.num_timepoints:
            paths = solve_paths
        else:
            paths = _interpolate_paths_to_grid(
                solve_paths,
                solve_times,
                output_times,
                self.path_encoder.geometry,
            )
        recon_mu = apply_linear(self.recon_net, paths)
        aux_logits = apply_linear(self.aux_net, paths)
        return LatentSDEOutput(
            paths=paths,
            recon_mu=recon_mu,
            aux_logits=aux_logits,
            kl0=kl0,
            klp=klp,
        )
