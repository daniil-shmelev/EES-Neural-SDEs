r"""Stochastic second-order Kuramoto model on the cotangent bundle $T\mathbb{T}^N$.

The dynamical equation, matching Olmi & Torcini (2024) eq. (1) with
their higher-order coupling switched off ($K_2 = 0$), and equivalent to
the deterministic power-grid model of Filatrella, Nielsen & Pedersen (2008)
extended with the standard Gaussian momentum forcing of
Schmietendorf et al. (2014) / Schäfer et al. (2018):

.. math::
    m\,\ddot{\theta}_i = -\dot{\theta}_i + \Omega_i
        + \frac{K}{N}\sum_j \sin(\theta_j - \theta_i) + \xi_i(t),
    \qquad \langle\xi_i(t)\xi_j(s)\rangle = 2D\,\delta_{ij}\delta(t-s).

We work in the standard normalised form with damping coefficient $1$ and
inertia $m$ (configurable; default $m = 1$). The natural frequencies
$\Omega_i$ follow the bimodal generator/consumer convention of Filatrella
et al. (2008): half the nodes are generators with $\Omega_i = +P$, the
other half consumers with $\Omega_i = -P$, so the network is balanced
($\sum_i \Omega_i = 0$ when $N$ is even).

State for SDE integration is the stacked vector
$(\theta_1, \ldots, \theta_N, \omega_1, \ldots, \omega_N) \in \mathbb{R}^{2N}$,
with $\omega_i = \dot\theta_i$ the angular velocity. The diffusion is
additive on $\omega$, so Stratonovich and Itô coincide and the diffrax
`Heun` solver gives weak-order-one convergence.

References
----------
- Filatrella, Nielsen & Pedersen (2008). "Analysis of a power grid using
  a Kuramoto-like model." *European Physical Journal B*, 61, 485-491.
- Schmietendorf, Peinke, Friedrich & Kamps (2014). "Self-organized
  synchronization and voltage stability in networks of synchronous
  machines." *Eur. Phys. J. Special Topics*, 223, 2577.
- Schäfer, Witthaut, Timme & Latora (2018). "Dynamically induced cascading
  failures in power grids." *Nature Communications*, 9, 1975.
- Olmi & Torcini (2024). "Stochastic Kuramoto oscillators with inertia
  and higher-order interactions." arXiv:2407.14874.
- Acebrón et al. (2005). "The Kuramoto model: A simple paradigm for
  synchronization phenomena." *Rev. Mod. Phys.*, 77, 137.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class KuramotoParams(eqx.Module):
    """Parameters of the stochastic 2nd-order Kuramoto network.

    `eqx.Module` (a registered JAX pytree) so that `eqx.filter_jit` can
    partition array fields (dynamic) from non-array fields (static).
    """

    N: int = eqx.field(static=True)
    m: float = eqx.field(static=True)
    K: float = eqx.field(static=True)
    natural_freqs: Float[Array, " N"]

    def __init__(
        self,
        N: int,
        m: float,
        K: float,
        natural_freqs: Float[Array, " N"],
    ):
        self.N = int(N)
        self.m = float(m)
        self.K = float(K)
        self.natural_freqs = jnp.asarray(natural_freqs)

    @staticmethod
    def bimodal(N: int, P: float = 0.5, K: float = 1.0, m: float = 1.0) -> "KuramotoParams":
        """Filatrella-style bimodal natural frequencies: half +P, half -P.

        Requires even $N$ for exact balance ($\\sum_i \\Omega_i = 0$). For
        odd $N$ we put one extra generator and the imbalance is absorbed
        as a small bias in the synchronous-frame frequency.
        """
        n_gen = (N + 1) // 2
        n_con = N - n_gen
        natural = jnp.concatenate([jnp.full((n_gen,), P), jnp.full((n_con,), -P)])
        return KuramotoParams(N=N, m=m, K=K, natural_freqs=natural)


def kuramoto_drift(
    theta: Float[Array, " N"],
    omega: Float[Array, " N"],
    params: KuramotoParams,
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Deterministic drift of the stochastic 2nd-order Kuramoto SDE.

    Returns $(\dot\theta, \dot\omega)$ such that

    .. math::
        \dot\theta_i = \omega_i,
        \quad
        \dot\omega_i = \frac{1}{m}\!\left[ -\omega_i + \Omega_i
            + \frac{K}{N} \sum_j \sin(\theta_j - \theta_i)\right].
    """
    sin_diff = jnp.sin(theta[None, :] - theta[:, None])
    coupling = (params.K / params.N) * jnp.sum(sin_diff, axis=1)
    omega_dot = (-omega + params.natural_freqs + coupling) / params.m
    return omega, omega_dot


def order_parameter(theta: Float[Array, " N"]) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Kuramoto complex order parameter $r e^{i\Psi} = \frac{1}{N}\sum_j e^{i\theta_j}$.

    Returns the magnitude $r \in [0, 1]$ (coherence) and mean phase $\Psi$.
    """
    z = jnp.mean(jnp.exp(1j * theta))
    return jnp.abs(z), jnp.angle(z)


def make_state_drift(params: KuramotoParams) -> Callable:
    """Build an `ODETerm`-compatible drift function on the stacked state.

    The returned function takes a stacked state of shape `(2N,)` (theta
    first, then omega) and returns the deterministic part of $d(\\theta,
    \\omega)/dt$ in the same shape. Compatible with `diffrax.ODETerm`.
    """
    N = params.N

    def drift(t, y, args):
        del t, args
        theta = y[:N]
        omega = y[N:]
        dtheta, domega = kuramoto_drift(theta, omega, params)
        return jnp.concatenate([dtheta, domega])

    return drift


def make_state_diffusion(params: KuramotoParams, D: float) -> Callable:
    r"""Build a `ControlTerm`-compatible diffusion function.

    The diffusion is decoupled additive noise on $\omega$ with
    coefficient $\sqrt{2D}/m$ — the standard Gaussian white-noise
    representation $\xi_i(t)$ with $\langle\xi_i(t)\xi_j(s)\rangle = 2D
    \delta_{ij}\delta(t-s)$ scaled by the inertia.

    Returns a function $g(t, y, args) \to G \in \mathbb{R}^{2N \times N}$
    such that the increment is $G\,dW$ with $W \in \mathbb{R}^N$ standard
    Brownian motion.
    """
    N = params.N
    sigma_eff = jnp.sqrt(2.0 * D) / params.m

    def diffusion(t, y, args):
        del t, y, args
        zero_block = jnp.zeros((N, N))
        sigma_block = sigma_eff * jnp.eye(N)
        return jnp.concatenate([zero_block, sigma_block], axis=0)

    return diffusion


def critical_coupling(P: float) -> float:
    r"""Approximate synchronization threshold for the bimodal noise-free model.

    For the deterministic 2nd-order Kuramoto with bimodal $\pm P$ natural
    frequencies, the partial-synchronisation transition occurs near
    $K_c \approx 2P / \sin(\pi / 3) \approx 2.31 P$ (Acebrón et al. 2005,
    Section IV.B; Filatrella et al. 2008, Section 4). Used as an order-
    of-magnitude scale for picking the operating regime.
    """
    return 2.0 * P / jnp.sin(jnp.pi / 3.0)
