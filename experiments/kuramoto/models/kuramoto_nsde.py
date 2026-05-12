r"""Neural SDE on $T\mathbb{T}^N = \mathbb{T}^N \times \mathbb{R}^N$ for
forecasting trajectories of the stochastic 2nd-order Kuramoto network.

Drift and diffusion fields are MLPs over the feature representation
$(\sin\theta, \cos\theta, \omega) \in \mathbb{R}^{3N}$. Integration is via
diffrax `MultiTerm(ODETerm, ControlTerm)` wrapped in a georax
`GeometricTerm` so that CFEES (or any other georax solver) advances the
state on the product Lie group with one exponential per stage.

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

    Retained for backward-compatibility with existing checkpoints; the
    default drift is now :class:`MeanFieldDriftField` (selected via
    ``drift_kind="mean_field"`` in the experiment config). The
    :class:`EquivariantDriftField` general-DeepSets variant is also
    available as ``drift_kind="equivariant"``.
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


class MeanFieldDriftField(eqx.Module):
    r"""Permutation-equivariant mean-field drift on $T\mathbb{T}^N$.

    Computes a fixed set of global moments (means of trigonometric and
    polynomial features of the per-oscillator state) and feeds the
    concatenation of (per-node features, broadcast moments) into a
    per-node MLP $\rho$ whose 2-dim output is the per-oscillator
    Lie-algebra increment $(\dot\theta_i, \dot\omega_i)$.

    Concretely, the moments are
    $(\overline{\sin\theta},\overline{\cos\theta},\overline{\sin 2\theta},
    \overline{\cos 2\theta},\overline{\omega},\overline{\omega^2})$,
    sufficient to express Kuramoto's natural drift exactly (the
    $(K/N)\sum_j \sin(\theta_j-\theta_i)$ coupling decomposes as
    $\overline{\sin\theta}\cos\theta_i - \overline{\cos\theta}\sin\theta_i$)
    plus second-harmonic and second-omega-moment terms for headroom.

    Permutation-equivariance: parameter count is $O(1)$ in $N$, and
    permuting the oscillator indices in the input permutes the output
    identically. Memory and compute are $O(N)$, viable at $N=1000$ on a
    consumer GPU.

    The last layer of $\rho$ is zero-initialised so the drift starts at
    zero, matching :class:`KuramotoDriftField`'s init pattern.
    """

    rho_node: eqx.nn.MLP
    geometry: ProductTorusEuclidean
    N: int = eqx.field(static=True)
    n_moments: int = eqx.field(static=True)
    drift_bound: float = eqx.field(static=True)

    def __init__(
        self,
        geometry,
        hidden_dim,
        *,
        depth: int = 3,
        activation=jax.nn.silu,
        drift_bound: float = 50.0,
        key,
    ):
        n_moments = 6
        rho_node = eqx.nn.MLP(
            in_size=3 + n_moments,
            out_size=2,
            width_size=hidden_dim,
            depth=depth,
            activation=activation,
            key=key,
        )
        last = rho_node.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))
        last = eqx.tree_at(lambda l: l.weight, last, jnp.zeros_like(last.weight))
        self.rho_node = eqx.tree_at(lambda m: m.layers[-1], rho_node, last)
        self.geometry = geometry
        self.N = int(geometry.N)
        self.n_moments = n_moments
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
        # Global moments (broadcast to every oscillator).
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
            [s[:, None], c[:, None], omega[:, None], moments_b], axis=-1
        )  # (N, 3 + n_moments)
        out = jax.vmap(self.rho_node)(node_in)  # (N, 2)
        # Soft-bound the drift output. ``drift_bound`` is chosen large
        # enough that natural Kuramoto magnitudes (O(1)-O(10)) sit in
        # tanh's near-linear regime, but unbounded MLP excursions during
        # training cannot send the integrator to NaN. Without this
        # bound the smoke test trains for ~270 steps and then locks into
        # an unrecoverable NaN-grad state.
        b = self.drift_bound
        out = b * jnp.tanh(out / b)
        dtheta = out[:, 0]
        domega = out[:, 1]
        return jnp.concatenate([dtheta, domega])


class IndexedMeanFieldDriftField(eqx.Module):
    r"""Mean-field drift with a learnable per-oscillator index embedding.

    Identical to :class:`MeanFieldDriftField` except that a learnable
    feature vector :math:`e_i \in \mathbb{R}^{e}` is concatenated to
    each oscillator's per-node input. This breaks pure
    permutation-equivariance, which is the *correct* symmetry-class for
    the cached dataset: the simulator pins the natural frequency
    $\Omega_i$ as a deterministic function of the oscillator index $i$
    (generators $i < N/2$ have $\Omega_i = +P$, consumers
    $i \ge N/2$ have $\Omega_i = -P$; see
    `experiments/kuramoto/datasets/kuramoto.py:bimodal`). The
    collective $\sin(\theta_j-\theta_i)$ coupling is still captured via
    the $O(N)$ moments aggregator; only the per-oscillator $\Omega_i$
    bias term needs the index.

    Param count is $N \cdot e + \rho$-MLP, scaling linearly in $N$ but
    with a much smaller constant than the legacy 3N->2N MLP (which
    scales as $\sim N \cdot h$).
    """

    rho_node: eqx.nn.MLP
    embedding: jax.Array
    geometry: ProductTorusEuclidean
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
        N = int(geometry.N)
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


class EquivariantDriftField(eqx.Module):
    r"""Permutation-equivariant DeepSets drift on $T\mathbb{T}^N$.

    Per-pair edge MLP $\varphi$ on $(\sin(\theta_j - \theta_i),
    \cos(\theta_j - \theta_i), \omega_j - \omega_i) \in \mathbb{R}^3$,
    mean-pooled over $j$ and fed (with the per-node features
    $\sin\theta_i, \cos\theta_i, \omega_i$) into a per-node MLP $\rho$
    whose 2-dim output is the per-oscillator Lie-algebra increment
    $(\dot\theta_i, \dot\omega_i)$.

    Equivariance: parameter count is independent of $N$, and permuting the
    oscillator indices in the input permutes the output identically.

    The last layer of $\rho$ is zero-initialised so the drift starts at
    zero, matching :class:`KuramotoDriftField`'s init pattern.
    """

    phi_pair: eqx.nn.MLP
    rho_node: eqx.nn.MLP
    geometry: ProductTorusEuclidean
    N: int = eqx.field(static=True)

    def __init__(
        self,
        geometry,
        hidden_dim,
        *,
        depth_pair: int = 2,
        depth_node: int = 2,
        activation=jax.nn.silu,
        key,
    ):
        kp, kn = jax.random.split(key, 2)
        phi_pair = eqx.nn.MLP(
            in_size=3,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=depth_pair,
            activation=activation,
            key=kp,
        )
        rho_node = eqx.nn.MLP(
            in_size=3 + hidden_dim,
            out_size=2,
            width_size=hidden_dim,
            depth=depth_node,
            activation=activation,
            key=kn,
        )
        last = rho_node.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))
        last = eqx.tree_at(lambda l: l.weight, last, jnp.zeros_like(last.weight))
        self.phi_pair = phi_pair
        self.rho_node = eqx.tree_at(lambda m: m.layers[-1], rho_node, last)
        self.geometry = geometry
        self.N = int(geometry.N)

    def __call__(self, t, y, args):
        del t, args
        N = self.N
        theta = y[:N]
        omega = y[N:]

        # Mean-pool $\varphi(z_{ij})$ over $j$. The unrolled
        # ``vmap(vmap(phi))`` form materialises a $(N, N, h)$ tensor
        # which OOMs at $N{=}1000$. We instead scan over $j$, which
        # gives $O(N\,h)$ working memory in forward. The scan body is
        # wrapped in ``jax.checkpoint`` so reverse-mode AD does not have
        # to store all $N$ carries (a stored carry per iter is also
        # $O(N\,h)$, so the unmitigated scan would need $O(N^2 h)$ for
        # backward, same OOM cliff in disguise). Cost: about 2x drift FLOPs.
        phi = self.phi_pair
        hidden_dim = phi.layers[-1].out_features

        @jax.checkpoint
        def _accum(carry, j):
            pair_in_j = jnp.stack(
                [
                    jnp.sin(theta[j] - theta),
                    jnp.cos(theta[j] - theta),
                    omega[j] - omega,
                ],
                axis=-1,
            )  # (N, 3)
            feats_j = jax.vmap(phi)(pair_in_j)  # (N, hidden)
            return carry + feats_j, None

        init = jnp.zeros((N, hidden_dim), dtype=theta.dtype)
        pooled_sum, _ = jax.lax.scan(_accum, init, jnp.arange(N))
        pooled = pooled_sum / jnp.asarray(N, dtype=pooled_sum.dtype)

        node_in = jnp.concatenate(
            [
                jnp.sin(theta)[:, None],
                jnp.cos(theta)[:, None],
                omega[:, None],
                pooled,
            ],
            axis=-1,
        )  # (N, 3 + hidden)
        out = jax.vmap(self.rho_node)(node_in)  # (N, 2)
        dtheta = out[:, 0]
        domega = out[:, 1]
        return jnp.concatenate([dtheta, domega])


class KuramotoDiffusionField(eqx.Module):
    r"""MLP diffusion on $T\mathbb{T}^N$.

    Input: state features $(\sin\theta, \cos\theta, \omega)$.
    Output: a $2N \times N$ matrix $G(t, y)$ such that the SDE increment
    is $G\,dW$ with $W \in \mathbb{R}^N$ Brownian. By default the noise
    only acts on $\omega$ (lower $N \times N$ block); the upper block is
    zero, matching the structure of the data-generating SDE
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
        last = eqx.tree_at(lambda l: l.bias, last, jnp.full_like(last.bias, -4.0))
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

    drift_field: eqx.Module
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
        drift_kind: str = "mean_field",
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

        if drift_kind == "mean_field":
            self.drift_field = MeanFieldDriftField(
                geometry=geometry,
                hidden_dim=hidden_dim,
                depth=drift_depth,
                activation=act_fn,
                key=k_drift,
            )
        elif drift_kind == "indexed_mean_field":
            self.drift_field = IndexedMeanFieldDriftField(
                geometry=geometry,
                hidden_dim=hidden_dim,
                depth=drift_depth,
                activation=act_fn,
                key=k_drift,
            )
        elif drift_kind == "equivariant":
            self.drift_field = EquivariantDriftField(
                geometry=geometry,
                hidden_dim=hidden_dim,
                depth_pair=drift_depth,
                depth_node=drift_depth,
                activation=act_fn,
                key=k_drift,
            )
        elif drift_kind == "mlp":
            self.drift_field = KuramotoDriftField(
                geometry=geometry,
                hidden_dim=hidden_dim,
                depth=drift_depth,
                activation=act_fn,
                key=k_drift,
            )
        else:
            raise ValueError(
                f"unknown drift_kind {drift_kind!r}; expected one of "
                "'mean_field', 'indexed_mean_field', 'equivariant', 'mlp'"
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
