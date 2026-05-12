"""JAX PowerSpherical pieces used by the sphere latent SDE."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
from jaxtyping import Array

from experiments.sphere_latent_nsde.geometry import normalize

_EPS = 1e-7


def hyperspherical_uniform_sample(
    key: Array,
    *,
    shape: tuple[int, ...],
    dim: int,
    dtype=jnp.float32,
) -> Array:
    samples = jax.random.normal(key, shape + (dim,), dtype=dtype)
    return normalize(samples, eps=_EPS)


def hyperspherical_uniform_entropy(dim: int, *, dtype=jnp.float32) -> Array:
    dim_arr = jnp.asarray(dim, dtype=dtype)
    return (
        jnp.log(jnp.asarray(2.0, dtype=dtype))
        + 0.5 * dim_arr * jnp.log(jnp.asarray(math.pi, dtype=dtype))
        - jsp_special.gammaln(0.5 * dim_arr)
    )


@dataclass(frozen=True)
class PowerSpherical:
    loc: Array
    scale: Array

    @property
    def dim(self) -> int:
        return int(self.loc.shape[-1])

    def rsample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        """Reparameterized sample matching ``power_spherical``'s construction."""

        dim = self.dim
        dtype = self.loc.dtype
        k_t, k_s = jax.random.split(key)
        alpha = (dim - 1) / 2 + self.scale
        beta = jnp.full_like(self.scale, (dim - 1) / 2)
        t = 2.0 * jax.random.beta(
            k_t,
            alpha,
            beta,
            shape=sample_shape + self.scale.shape,
        ) - 1.0
        s = hyperspherical_uniform_sample(
            k_s,
            shape=sample_shape + self.scale.shape,
            dim=dim - 1,
            dtype=dtype,
        )
        x = jnp.concatenate(
            [t[..., None], s * jnp.sqrt(jnp.maximum(1.0 - t[..., None] ** 2, _EPS))],
            axis=-1,
        )

        e1 = jnp.zeros_like(self.loc)
        e1 = e1.at[..., 0].set(1.0)
        u = e1 - self.loc
        u = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + _EPS)
        while u.ndim < x.ndim:
            u = u[None]
        return x - 2.0 * jnp.sum(x * u, axis=-1, keepdims=True) * u

    def log_normalizer(self) -> Array:
        dim = self.dim
        alpha = (dim - 1) / 2 + self.scale
        beta = jnp.full_like(self.scale, (dim - 1) / 2)
        return -(
            (alpha + beta) * jnp.log(2.0)
            + jsp_special.gammaln(alpha)
            - jsp_special.gammaln(alpha + beta)
            + beta * jnp.log(math.pi)
        )

    def entropy(self) -> Array:
        dim = self.dim
        alpha = (dim - 1) / 2 + self.scale
        beta = jnp.full_like(self.scale, (dim - 1) / 2)
        return -(
            self.log_normalizer()
            + self.scale
            * (jnp.log(2.0) + jsp_special.digamma(alpha) - jsp_special.digamma(alpha + beta))
        )

    def kl_to_uniform(self) -> Array:
        return -self.entropy() + hyperspherical_uniform_entropy(
            self.dim, dtype=self.loc.dtype
        )

