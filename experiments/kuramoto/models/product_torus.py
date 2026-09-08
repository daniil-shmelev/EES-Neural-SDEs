"""Exact flat geometry on T^N x R^N for the Kuramoto model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from georax import LocalChart, Manifold
from jaxtyping import Array

from experiments.kuramoto.geometry import wrap_to_pi


def wrap_first_half(x: Array, N: int) -> Array:
    return jnp.concatenate([wrap_to_pi(x[..., :N]), x[..., N:]], axis=-1)


class ProductTorusEuclideanChart(LocalChart):
    """Exact additive chart, with periodic phase coordinates."""

    def apply(self, x, a, geometry):
        return wrap_first_half(x + a, geometry.N)

    def inverse_differential(self, x, a, b, geometry):
        return b


class ProductTorusEuclidean(Manifold):
    N: int = eqx.field(static=True)
    _chart_class = ProductTorusEuclideanChart

    def __init__(self, N: int):
        if int(N) < 1:
            raise ValueError("ProductTorusEuclidean(N) requires N >= 1.")
        self.N = int(N)

    @property
    def dimension(self):
        return 2 * self.N

    @property
    def state_shape(self):
        return (self.dimension,)

    @property
    def coordinate_shape(self):
        return (self.dimension,)

    def trivialise(self, x, v):
        return v

    def detrivialise(self, x, a):
        return a

    def frame_bracket(self, x, a, b):
        return jnp.zeros_like(a)
