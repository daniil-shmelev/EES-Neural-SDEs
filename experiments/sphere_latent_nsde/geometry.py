"""Sphere geometry for georax solvers.

The original Latent-SDE-on-homogeneous-spaces code evolves points on
``S^{n-1}`` by applying exponentials from ``so(n)`` to ambient vectors. This
module exposes that same homogeneous-space action as a georax ``Manifold``.
The coordinate convention intentionally matches the PyTorch baseline:
coordinates enumerate the strictly lower triangular entries of a skew matrix.
"""

from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from diffrax._custom_types import RealScalarLike
from georax._geometry.base import LocalChart, Manifold
from jaxtyping import Array


_INVERSE_DEXP_COEFFS = {
    1: -0.5,
    2: 1.0 / 12.0,
    4: -1.0 / 720.0,
    6: 1.0 / 30240.0,
    8: -1.0 / 1209600.0,
    10: 1.0 / 47900160.0,
}


def normalize(x: Array, eps: float = 1e-7) -> Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def _lie_bracket(a: Array, b: Array) -> Array:
    return a @ b - b @ a


def _inverse_dexp(
    a: Array,
    b: Array,
    geometry: "Sphere",
    order: RealScalarLike,
) -> Array:
    omega = geometry.coords_to_alg(a, dtype=b.dtype)
    ad_power = geometry.coords_to_alg(b, dtype=b.dtype)
    corrected = ad_power

    for degree in range(1, int(order)):
        ad_power = _lie_bracket(omega, ad_power)
        coeff = _INVERSE_DEXP_COEFFS.get(degree)
        if coeff is not None:
            corrected = corrected + jnp.asarray(coeff, dtype=b.dtype) * ad_power

    return geometry.alg_to_coords(corrected)


class SphereExpChart(LocalChart):
    """Exact ``SO(n)`` action chart on the sphere."""

    order: RealScalarLike = 12
    inverse_order: RealScalarLike = 12

    @override
    def apply(self, x: Array, a: Array, geometry: "Sphere") -> Array:
        omega = geometry.coords_to_alg(a, dtype=x.dtype)
        q = jsp_linalg.expm(omega)
        return jnp.einsum("...ij,...j->...i", q, x)

    @override
    def inverse_differential(
        self, x: Array, a: Array, b: Array, geometry: "Sphere"
    ) -> Array:
        del x
        return _inverse_dexp(a, b, geometry, self.inverse_order)


class SphereTaylorChart(LocalChart):
    """Taylor action chart followed by projection back to the sphere."""

    order: RealScalarLike
    inverse_order: RealScalarLike
    degree: int = eqx.field(static=True)

    def __init__(self, order: int):
        order = int(order)
        degree = max(2, order)
        if degree % 2:
            degree += 1
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "inverse_order", order)
        object.__setattr__(self, "degree", degree)

    @override
    def apply(self, x: Array, a: Array, geometry: "Sphere") -> Array:
        omega = geometry.coords_to_alg(a, dtype=x.dtype)
        term = x
        y = x
        for k in range(1, self.degree + 1):
            term = jnp.einsum("...ij,...j->...i", omega, term) / jnp.asarray(
                k, dtype=x.dtype
            )
            y = y + term
        return normalize(y)

    @override
    def inverse_differential(
        self, x: Array, a: Array, b: Array, geometry: "Sphere"
    ) -> Array:
        del x
        return _inverse_dexp(a, b, geometry, self.inverse_order)


class Sphere(Manifold):
    """Unit sphere ``S^{n-1}`` with frame coordinates in ``so(n)``.

    Georax integrators operate in frame coordinates. For this homogeneous
    sphere representation those coordinates are the coefficients of a skew
    matrix, ordered as PyTorch's ``torch.tril_indices(offset=-1)`` did in the
    reference implementation.
    """

    n: int = eqx.field(static=True)
    _lower_i: tuple[int, ...] = eqx.field(static=True)
    _lower_j: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, n: int, *, chart: LocalChart | None = None):
        n = int(n)
        if n < 2:
            raise ValueError("Sphere(n) requires n >= 2.")

        lower_i, lower_j = np.tril_indices(n, k=-1)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "chart", chart)
        object.__setattr__(self, "_lower_i", tuple(int(i) for i in lower_i))
        object.__setattr__(self, "_lower_j", tuple(int(j) for j in lower_j))

    @property
    def ambient_dimension(self) -> int:
        return self.n

    @property
    def lie_algebra_dimension(self) -> int:
        return len(self._lower_i)

    @property
    def dimension(self) -> int:
        """Frame-coordinate dimension, used as the Brownian driver dimension."""

        return self.lie_algebra_dimension

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.n,)

    @property
    def coordinate_shape(self) -> tuple[int, ...]:
        return (self.lie_algebra_dimension,)

    def check_state_shape(self, x: Array) -> None:
        if x.shape[-1:] != self.state_shape:
            raise ValueError(f"Expected sphere states ending in {self.state_shape}, got {x.shape}.")

    def check_coordinate_shape(self, a: Array) -> None:
        if a.shape[-1:] != self.coordinate_shape:
            raise ValueError(f"Expected frame coordinates ending in {self.coordinate_shape}, got {a.shape}.")

    def zero_coordinates(self, x: Array) -> Array:
        self.check_state_shape(x)
        return jnp.zeros(x.shape[:-1] + self.coordinate_shape, dtype=x.dtype)

    def trivialise(self, x: Array, v: Array) -> Array:
        # The skew lift v x^T - x v^T sends a unit x to its tangent v.
        omega = v[..., :, None] * x[..., None, :] - x[..., :, None] * v[..., None, :]
        return self.alg_to_coords(omega)

    def detrivialise(self, x: Array, a: Array) -> Array:
        return jnp.einsum("...ij,...j->...i", self.coords_to_alg(a), x)

    def frame_bracket(self, x: Array, a: Array, b: Array) -> Array:
        del x
        # Constant generators act on the left: [X_a, X_b] = X_{ba-ab}.
        return -self.alg_to_coords(_lie_bracket(self.coords_to_alg(a), self.coords_to_alg(b)))

    @property
    def basis(self) -> Array:
        d = self.lie_algebra_dimension
        lower_i = jnp.asarray(self._lower_i)
        lower_j = jnp.asarray(self._lower_j)
        basis = jnp.zeros((d, self.n, self.n), dtype=jnp.float32)
        k = jnp.arange(d)
        basis = basis.at[k, lower_i, lower_j].set(1.0)
        basis = basis.at[k, lower_j, lower_i].set(-1.0)
        return basis

    def coords_to_alg(self, a: Array, *, dtype=None) -> Array:
        coeffs = jnp.asarray(a, dtype=dtype)
        return jnp.einsum("...d,dij->...ij", coeffs, self.basis.astype(coeffs.dtype))

    def alg_to_coords(self, omega: Array) -> Array:
        omega = 0.5 * (omega - jnp.swapaxes(omega, -1, -2))
        return omega[..., jnp.asarray(self._lower_i), jnp.asarray(self._lower_j)]

    def project(self, x: Array) -> Array:
        return normalize(x)

    @override
    def select_chart(self, required_order: RealScalarLike) -> LocalChart:
        chart: LocalChart = SphereTaylorChart(int(required_order))
        object.__setattr__(self, "chart", chart)
        return chart


def vec_to_matrix(vec: Array, basis: Array) -> Array:
    """Convert lower-triangular ``so(n)`` coordinates to skew matrices."""

    return jnp.einsum("...d,dij->...ij", vec, basis)
