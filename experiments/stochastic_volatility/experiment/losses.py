from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

PyTree = Any
LossFn = Callable[[eqx.Module, PyTree, jax.Array, jax.Array], jax.Array]
SampleFn = Callable[
    [eqx.Module, PyTree, jax.Array, jax.Array], tuple[jax.Array, jax.Array]
]


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(bool)
    safe_values = jnp.where(mask, values, jnp.zeros_like(values))
    return jnp.sum(safe_values) / jnp.maximum(jnp.sum(mask.astype(values.dtype)), 1.0)


def make_supervised_mse_loss(*, sample_fn: SampleFn) -> LossFn:
    def loss_fn(model, batch, mask, key):
        predictions, targets = sample_fn(model, batch, mask, key)
        axes = tuple(range(1, predictions.ndim))
        per_example_loss = jnp.mean((predictions - targets) ** 2, axis=axes)
        return masked_mean(per_example_loss, mask)

    return loss_fn


def truncated_sig_loss(
    depth: int = 6,
    ambient_dim: int = 1,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    from stochastax.control_lifts import compute_path_signature
    from stochastax.hopf_algebras import ShuffleHopfAlgebra

    augmented_dim = int(ambient_dim) + 1
    hopf = ShuffleHopfAlgebra.build(ambient_dim=augmented_dim, depth=int(depth))

    def loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        assert pred.shape[0] == target.shape[0], (
            f"pred and target must have the same batch size, got {pred.shape[0]} and {target.shape[0]}"
        )
        assert int(pred.shape[-1]) == int(ambient_dim), (
            f"Expected path feature dimension {int(ambient_dim)}, got {int(pred.shape[-1])}. "
            "Pass the correct `ambient_dim` when constructing the loss."
        )

        def _time_augment(path: jax.Array) -> jax.Array:
            times = jnp.linspace(0.0, 1.0, path.shape[0], dtype=path.dtype)[:, None]
            return jnp.concatenate([times, path], axis=-1)

        def _phi(path: jax.Array) -> jax.Array:
            augmented_path = _time_augment(path)
            sig = compute_path_signature(
                path=augmented_path, depth=depth, hopf=hopf, mode="full"
            )
            return sig.flatten()

        phi_pred = jax.vmap(_phi)(pred)
        phi_target = jax.vmap(_phi)(target)

        # Biased MMD^2 with dot-product kernel: E[k(X,X')] + E[k(Y,Y')] - 2 E[k(X,Y)]
        k_pp = jnp.mean(phi_pred @ phi_pred.T)
        k_tt = jnp.mean(phi_target @ phi_target.T)
        k_pt = jnp.mean(phi_pred @ phi_target.T)
        return k_pp + k_tt - 2.0 * k_pt

    return loss
