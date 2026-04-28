"""Flat n-torus T^d = SO(2)^d as a georax LieGroup.

Ambient state is x in R^d interpreted as angles wrapped to (-pi, pi]. The Lie
algebra is R^d with trivial bracket, so all chart machinery degenerates to
identity and the retraction is componentwise wrap.
"""

from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from diffrax._custom_types import RealScalarLike
from georax._geometry.base import LieGroup, LocalFlow, flow_order
from jaxtyping import Array


def wrap_to_pi(x: Array) -> Array:
    """Wrap angles into the half-open interval (-pi, pi]."""
    return jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi


class TorusFlow(LocalFlow):
    """Exact local flow for the flat torus: phi(x, a) = wrap(x + a)."""

    order: flow_order = "exact"
    inverse_order: flow_order = "exact"

    def forward(self, x: Array, a: Array, geometry: "Torus") -> Array:
        del geometry
        return wrap_to_pi(x + a)

    def d_inverse(self, x: Array, y: Array, geometry: "Torus") -> Array:
        del x, geometry
        return y


class Torus(LieGroup):
    """Flat n-torus T^d as a Lie group.

    The Lie algebra is R^d with the standard basis. The frame at every point
    is the identity map, the chart differential is the identity, and the
    retraction is componentwise wrap.
    """

    d: int = eqx.field(static=True)

    def __init__(self, d: int, *, flow: LocalFlow | None = None):
        d = int(d)
        if d < 1:
            raise ValueError("Torus(d) requires d >= 1.")
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "flow", TorusFlow() if flow is None else flow)

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

    @override
    def select_flow_method(self, required_order: RealScalarLike) -> LocalFlow:
        del required_order
        flow: LocalFlow = TorusFlow()
        object.__setattr__(self, "flow", flow)
        return flow
