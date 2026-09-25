"""Normalisation used by the spherical probability distributions."""

import jax.numpy as jnp
from jaxtyping import Array


def normalize(x: Array, eps: float = 1e-7) -> Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)
