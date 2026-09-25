"""Angle wrapping for torus data and losses."""

import jax.numpy as jnp
from jaxtyping import Array


def wrap_to_pi(x: Array) -> Array:
    """Wrap angles into the half-open interval [-pi, pi)."""
    return jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi
