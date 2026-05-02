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
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from diffrax._custom_types import RealScalarLike
from georax._geometry.base import LocalChart, Manifold
from jaxtyping import Array


def normalize(x: Array, eps: float = 1e-7) -> Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


class SphereExpChart(LocalChart):
    """Exact ``SO(n)`` action chart on the sphere."""

    order: RealScalarLike = 12
    inverse_order: RealScalarLike = 12

    @override
    def apply(self, x: Array, a: Array, geometry: "Sphere") -> Array:
        omega = geometry.coords_to_alg(a, dtype=x.dtype)
        q = jsp_linalg.expm(omega)
        return jnp.einsum("...ij,...j->...i", q, x)


class SphereCayleyChart(LocalChart):
    """Second-order Cayley chart for the ``SO(n)`` action on the sphere."""

    order: RealScalarLike = 2
    inverse_order: RealScalarLike = 2

    @override
    def apply(self, x: Array, a: Array, geometry: "Sphere") -> Array:
        omega = geometry.coords_to_alg(a, dtype=x.dtype)
        ident = jnp.eye(geometry.n, dtype=x.dtype)
        rhs = jnp.einsum("...ij,...j->...i", ident + 0.5 * omega, x)
        y = jnp.linalg.solve(ident - 0.5 * omega, rhs[..., None])[..., 0]
        return normalize(y)


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
        if required_order <= 2:
            chart: LocalChart = SphereCayleyChart()
        else:
            chart = SphereTaylorChart(int(required_order))
        object.__setattr__(self, "chart", chart)
        return chart


def vec_to_matrix(vec: Array, basis: Array) -> Array:
    """Convert lower-triangular ``so(n)`` coordinates to skew matrices."""

    return jnp.einsum("...d,dij->...ij", vec, basis)


def apply_so_action(x: Array, coeffs: Array, geometry: Sphere) -> Array:
    """Apply ``exp(sum_i coeffs_i E_i)`` to one sphere point."""

    return geometry.apply_increment(x, coeffs)


def geometric_euler_given_increments(
    z0: Array,
    drift: Array,
    noise: Array,
    dt: Array,
    geometry: Sphere,
) -> Array:
    """Reference-style geometric Euler using pre-sampled standard normals.

    This helper exists for parity checks against the PyTorch baseline. Training
    uses georax/diffrax solvers directly.
    """

    omegas = drift * dt[:, None] + noise * jnp.sqrt(dt)[:, None]

    def step(z, omega):
        z_next = geometry.apply_increment(z, omega)
        return z_next, z_next

    _, zs = jax.lax.scan(step, z0, omegas)
    return jnp.concatenate([z0[None], zs], axis=0)
