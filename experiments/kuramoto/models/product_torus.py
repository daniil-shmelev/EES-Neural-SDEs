r"""$\mathbb{T}^N \times \mathbb{R}^N$ as a product Lie group for the Kuramoto NSDE.

State $(\theta, \omega) \in \mathbb{R}^{2N}$ (stacked, theta first).
The Lie algebra is $\mathbb{R}^{2N}$ with trivial bracket. Retraction is
elementwise wrap on the first $N$ components and identity-add on the
last $N$ — matching the structure of the underlying SDE on the cotangent
bundle of the $N$-torus.

Mirrors the local `Torus` class in `experiments/rna/models/torus.py`,
but lifted to the product space so a single neural-SDE solve advances
both phases and angular velocities through the same `GeometricTerm`.
"""

from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike
from georax._geometry.base import LieGroup, LocalFlow, flow_order
from jaxtyping import Array

from experiments.rna.models.torus import wrap_to_pi


def wrap_first_half(x: Array, N: int) -> Array:
    theta = wrap_to_pi(x[..., :N])
    omega = x[..., N:]
    return jnp.concatenate([theta, omega], axis=-1)


class ProductTorusEuclideanFlow(LocalFlow):
    r"""Exact local flow for $\mathbb{T}^N \times \mathbb{R}^N$.

    $\phi((\theta, \omega), (a, b)) = (\mathrm{wrap}(\theta + a),
    \omega + b)$.
    """

    order: flow_order = "exact"
    inverse_order: flow_order = "exact"

    def forward(self, x: Array, a: Array, geometry: "ProductTorusEuclidean") -> Array:
        N = geometry.N
        return wrap_first_half(x + a, N)

    def d_inverse(self, x: Array, y: Array, geometry: "ProductTorusEuclidean") -> Array:
        del x, geometry
        return y


class ProductTorusEuclidean(LieGroup):
    r"""$\mathbb{T}^N \times \mathbb{R}^N$, encoded as $\mathbb{R}^{2N}$.

    State convention: $y = (\theta_1, \ldots, \theta_N, \omega_1, \ldots,
    \omega_N)$. Lie algebra is the same $\mathbb{R}^{2N}$ with trivial
    bracket; the frame is $I_{2N}$.
    """

    N: int = eqx.field(static=True)

    def __init__(self, N: int, *, flow: LocalFlow | None = None):
        N = int(N)
        if N < 1:
            raise ValueError("ProductTorusEuclidean(N) requires N >= 1.")
        object.__setattr__(self, "N", N)
        object.__setattr__(
            self,
            "flow",
            ProductTorusEuclideanFlow() if flow is None else flow,
        )

    @property
    def dimension(self) -> int:
        return 2 * self.N

    @override
    def frame(self, x: Array) -> Array:
        return jnp.eye(self.dimension, dtype=x.dtype)

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
        return wrap_first_half(x + v, self.N)

    @override
    def chart_differential_inv(self, a: Array, b: Array) -> Array:
        del a
        return b

    @override
    def select_flow_method(self, required_order: RealScalarLike) -> LocalFlow:
        del required_order
        flow: LocalFlow = ProductTorusEuclideanFlow()
        object.__setattr__(self, "flow", flow)
        return flow
