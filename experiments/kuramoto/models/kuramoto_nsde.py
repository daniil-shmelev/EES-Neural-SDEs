r"""Neural SDE on $T\mathbb{T}^N = \mathbb{T}^N \times \mathbb{R}^N$ for
forecasting the stochastic second-order Kuramoto network.

Georax's TangentTorus stores angles followed by velocities. The drift and
diffusion return frame coordinates for the GeometricTerm/MultiTerm API.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from diffrax import (
    AbstractAdjoint,
    AbstractReversibleSolver,
    AbstractSolver,
    ClipStepSizeController,
    ConstantStepSize,
    ControlTerm,
    MultiTerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    ReversibleAdjoint,
    SaveAt,
    StepTo,
    VirtualBrownianTree,
    diffeqsolve,
)
from georax import CFEES25, GeometricTerm, TangentTorus

from ees_core.torus import wrap_to_pi

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


def tangent_torus_error_norm(error, *, atol):
    """Absolute RMS error with shortest phase differences on the torus.

    PIDController supplies the unscaled error (rtol=0, atol=1); scaling here
    ensures that phase differences are wrapped before applying the tolerance.
    """
    n = error.shape[-1] // 2
    difference = jnp.concatenate([wrap_to_pi(error[:n]), error[n:]])
    return jnp.sqrt(jnp.mean(jnp.square(difference / atol)))


class IndexedMeanFieldDriftField(eqx.Module):
    r"""Shared node drift with global trigonometric/frequency moments and
    learned oscillator embeddings. The embeddings represent the dataset's
    index-dependent natural frequencies; no alternative drift is selected.
    """

    rho_node: eqx.nn.MLP
    embedding: jax.Array
    geometry: TangentTorus
    N: int = eqx.field(static=True)
    n_moments: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    drift_bound: float = eqx.field(static=True)

    def __init__(
        self,
        geometry,
        hidden_dim,
        *,
        depth: int = 3,
        activation=jax.nn.silu,
        embed_dim: int = 16,
        drift_bound: float = 50.0,
        key,
    ):
        N = int(geometry.state_shape[0] // 2)
        n_moments = 6
        ke, km = jax.random.split(key, 2)
        embedding = 0.1 * jax.random.normal(ke, (N, embed_dim))
        rho_node = eqx.nn.MLP(
            in_size=3 + n_moments + embed_dim,
            out_size=2,
            width_size=hidden_dim,
            depth=depth,
            activation=activation,
            key=km,
        )
        last = rho_node.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))
        last = eqx.tree_at(lambda l: l.weight, last, jnp.zeros_like(last.weight))
        self.rho_node = eqx.tree_at(lambda m: m.layers[-1], rho_node, last)
        self.embedding = embedding
        self.geometry = geometry
        self.N = N
        self.n_moments = n_moments
        self.embed_dim = embed_dim
        self.drift_bound = float(drift_bound)

    def __call__(self, t, y, args):
        del t, args
        N = self.N
        theta = y[:N]
        omega = y[N:]
        s = jnp.sin(theta)
        c = jnp.cos(theta)
        s2 = jnp.sin(2.0 * theta)
        c2 = jnp.cos(2.0 * theta)
        moments = jnp.stack(
            [
                jnp.mean(s),
                jnp.mean(c),
                jnp.mean(s2),
                jnp.mean(c2),
                jnp.mean(omega),
                jnp.mean(omega * omega),
            ]
        )  # (n_moments,)
        moments_b = jnp.broadcast_to(moments[None, :], (N, self.n_moments))
        node_in = jnp.concatenate(
            [
                s[:, None],
                c[:, None],
                omega[:, None],
                moments_b,
                self.embedding,
            ],
            axis=-1,
        )  # (N, 3 + n_moments + embed_dim)
        out = jax.vmap(self.rho_node)(node_in)  # (N, 2)
        b = self.drift_bound
        out = b * jnp.tanh(out / b)
        dtheta = out[:, 0]
        domega = out[:, 1]
        return jnp.concatenate([dtheta, domega])


class SecondOrderMeanFieldDriftField(IndexedMeanFieldDriftField):
    """Learn phase-dependent forcing and bounded damping, with dtheta = omega.

    The Kuramoto data have second-order kinematics and linear velocity damping.
    These relations remove learned phase drift and bound direct velocity
    damping, which made float32 reconstruction unreliable in the unconstrained
    model. The damping is learned in (0, 2),
    initialized at 1. The forcing and diffusion remain learned neural fields.

    Retain the original network layout and initialization for matched numerical
    comparisons. Its velocity input slots are zero and only its force output is
    used; the first output and corresponding unused input weights have zero
    gradients.
    """

    damping_logit: jax.Array

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.damping_logit = jnp.array(0.0, dtype=self.embedding.dtype)

    def __call__(self, t, y, args):
        n = self.N
        phase_state = jnp.concatenate([y[:n], jnp.zeros_like(y[n:])])
        force = super().__call__(t, phase_state, args)[n:]
        damping = 2 * jax.nn.sigmoid(self.damping_logit)
        return jnp.concatenate([y[n:], force - damping * y[n:]])


class KuramotoDiffusionField(eqx.Module):
    r"""MLP diffusion on $T\mathbb{T}^N$.

    Input: state features $(\sin\theta, \cos\theta, \omega)$.
    Output: a linear operator $G(t, y)$ such that the SDE increment
    is $G\,dW$ with $W \in \mathbb{R}^N$ Brownian. By default the noise
    only acts on $\omega$ (lower $N \times N$ block); the upper block is
    zero — matching the structure of the data-generating SDE
    (Filatrella et al. 2008 / Olmi & Torcini 2024).
    """

    mlp: eqx.nn.MLP
    geometry: TangentTorus
    diffusion_scale: float = eqx.field(static=True)
    couple_theta_omega: bool = eqx.field(static=True)

    def __init__(
        self,
        geometry,
        hidden_dim,
        diffusion_scale: float = 0.1,
        *,
        depth: int = 2,
        activation=jax.nn.silu,
        couple_theta_omega: bool = False,
        key,
    ):
        N = geometry.state_shape[0] // 2
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
        last = eqx.tree_at(lambda l: l.bias, last, jnp.full_like(last.bias, -4.0))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.geometry = geometry
        self.diffusion_scale = float(diffusion_scale)
        self.couple_theta_omega = bool(couple_theta_omega)

    def __call__(self, t, y, args):
        del t, args
        N = self.geometry.state_shape[0] // 2
        scales = jax.nn.softplus(self.mlp(state_features(y, N))) * self.diffusion_scale

        # Apply the rectangular diagonal matrix directly: materialising it costs
        # 2*N*N entries per trajectory at every solver stage.
        def product(dw):
            noise = scales[:N] * dw
            zero = jnp.zeros_like(noise)
            if self.couple_theta_omega:
                return jnp.concatenate([noise, zero])
            return jnp.concatenate([zero, noise])

        return lx.FunctionLinearOperator(
            product, jax.ShapeDtypeStruct((N,), scales.dtype)
        )


class KuramotoNSDE(eqx.Module):
    r"""Neural SDE on $T\mathbb{T}^N$ for trajectory forecasting."""

    drift_field: eqx.Module
    diffusion_field: KuramotoDiffusionField
    name: str = eqx.field(static=True)

    N: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_obs: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    solver: AbstractSolver = eqx.field(static=True)
    adjoint: AbstractAdjoint | None = eqx.field(static=True)
    adaptive: bool = eqx.field(static=True)
    adaptive_atol: float = eqx.field(static=True)
    adaptive_dtmin: float = eqx.field(static=True)
    adaptive_max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        N: int,
        hidden_dim: int = 128,
        n_steps: int = 50,
        dt: float = 0.1,
        n_obs: int = 100,
        solver: AbstractSolver | None = None,
        diffusion_scale: float = 0.1,
        adjoint: AbstractAdjoint | None = None,
        activation: str = "silu",
        drift_depth: int = 3,
        diffusion_depth: int = 2,
        couple_theta_omega: bool = False,
        drift_kind: str = "indexed_mean_field",
        adaptive: bool = False,
        adaptive_atol: float = 1e-3,
        adaptive_dtmin: float = 1e-4,
        adaptive_max_steps: int = 4096,
        *,
        key,
    ):
        k_drift, k_diff = jax.random.split(key, 2)

        geometry = TangentTorus(N)
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
        if adaptive:
            if not isinstance(solver, CFEES25):
                raise ValueError("Adaptive stepping requires a CF-EES embedded pair.")
            if adaptive_atol <= 0 or not 0 < adaptive_dtmin <= dt:
                raise ValueError("Require atol > 0 and 0 < adaptive_dtmin <= dt.")
            if adaptive_max_steps < n_steps:
                raise ValueError("adaptive_max_steps must be at least n_steps.")
        self.adaptive = adaptive
        self.adaptive_atol = adaptive_atol
        self.adaptive_dtmin = adaptive_dtmin
        self.adaptive_max_steps = adaptive_max_steps

        drift_classes = {
            "indexed_mean_field": IndexedMeanFieldDriftField,
            "second_order_mean_field": SecondOrderMeanFieldDriftField,
        }
        if drift_kind not in drift_classes:
            raise ValueError(f"Unknown Kuramoto drift: {drift_kind}")
        self.drift_field = drift_classes[drift_kind](
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

    def __call__(self, theta0, omega0, key, *, save_indices=None):
        """Return phases and velocities at the observation times."""
        saveat = None
        if save_indices is not None:
            times = jnp.linspace(0.0, self.n_steps * self.dt, self.n_obs)
            saveat = SaveAt(ts=times[save_indices])
        ys = self.solve(theta0, omega0, key, saveat=saveat).ys
        return ys[:, : self.N], ys[:, self.N :]

    def solve(
        self,
        theta0,
        omega0,
        key,
        *,
        saveat=None,
        adjoint=None,
        stepsize_controller=None,
    ):
        """Return the Diffrax solution, including accepted/rejected step counts.

        The optional Diffrax arguments support diagnostics on the accepted mesh.
        Reconstruction uses ReversibleAdjoint's stored forward step times.
        """
        N = self.N
        t1 = float(self.n_steps * self.dt)
        t_save = jnp.linspace(0.0, t1, self.n_obs)

        y0 = jnp.concatenate([theta0, omega0])
        brownian_path = VirtualBrownianTree(
            t0=0.0,
            t1=t1,
            tol=self.adaptive_dtmin / 10.0 if self.adaptive else self.dt / 4.0,
            shape=(N,),
            key=key,
        )
        term = MultiTerm(
            GeometricTerm(self.drift_field, self.drift_field.geometry),
            ControlTerm(self.diffusion_field, brownian_path),
        )

        if adjoint is None:
            if self.adjoint is not None:
                adjoint = self.adjoint
            elif isinstance(self.solver, AbstractReversibleSolver):
                adjoint = ReversibleAdjoint()
            else:
                adjoint = RecursiveCheckpointAdjoint()

        if stepsize_controller is None:
            if self.adaptive:
                stepsize_controller = PIDController(
                    rtol=0.0,
                    atol=1.0,
                    norm=eqx.Partial(tangent_torus_error_norm, atol=self.adaptive_atol),
                    pcoeff=0.1,
                    icoeff=0.3,
                    dtmin=self.adaptive_dtmin,
                    dtmax=self.dt,
                    force_dtmin=True,
                )
                # Full-state-dependent diffusion need not commute: retain revealed endpoints.
                stepsize_controller = ClipStepSizeController(
                    stepsize_controller, store_rejected_steps=100
                )
            else:
                stepsize_controller = ConstantStepSize()

        max_steps = self.adaptive_max_steps if self.adaptive else self.n_steps + 16

        return diffeqsolve(
            term,
            self.solver,
            t0=0.0,
            t1=t1,
            dt0=None if isinstance(stepsize_controller, StepTo) else self.dt,
            y0=y0,
            saveat=SaveAt(ts=t_save) if saveat is None else saveat,
            adjoint=adjoint,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
