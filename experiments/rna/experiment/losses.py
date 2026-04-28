"""Loss functions and Riemannian geometry helpers for SPD manifold training."""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from georax import SPD

PyTree = Any
LossFn = Callable[[eqx.Module, PyTree, jax.Array, jax.Array], jax.Array]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(bool)
    safe_values = jnp.where(mask, values, jnp.zeros_like(values))
    return jnp.sum(safe_values) / jnp.maximum(jnp.sum(mask.astype(values.dtype)), 1.0)


def replace_masked_spd_examples(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(bool)
    n = values.shape[-1]
    replacement = jnp.broadcast_to(jnp.eye(n, dtype=values.dtype), values.shape)
    expanded_mask = jnp.reshape(mask, mask.shape + (1,) * (values.ndim - 1))
    return jnp.where(expanded_mask, values, replacement)


# ---------------------------------------------------------------------------
# SPD matrix helpers
# ---------------------------------------------------------------------------


def project_to_spd(matrix: jax.Array, eps: float = 0.0) -> jax.Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return (eigenvectors * jnp.clip(eigenvalues, a_min=eps)) @ jnp.swapaxes(eigenvectors, -1, -2)


def matrix_sqrt(matrix: jax.Array) -> jax.Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return (eigenvectors * (eigenvalues**0.5)) @ jnp.swapaxes(eigenvectors, -1, -2)


def matrix_inv_sqrt(matrix: jax.Array) -> jax.Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return (eigenvectors * (eigenvalues**-0.5)) @ jnp.swapaxes(eigenvectors, -1, -2)


def inverse_congruence_coords(
    geometry: SPD,
    base: jax.Array,
    point: jax.Array,
) -> jax.Array:
    sqrt_base = matrix_sqrt(base)
    inv_sqrt_base = matrix_inv_sqrt(base)
    middle = sqrt_base @ point @ sqrt_base
    sqrt_middle = matrix_sqrt(middle)
    group_element = inv_sqrt_base @ sqrt_middle @ inv_sqrt_base
    eigenvalues, eigenvectors = jnp.linalg.eigh(group_element)
    lift = (eigenvectors * jnp.log(eigenvalues)) @ jnp.swapaxes(eigenvectors, -1, -2)
    return geometry._sym_to_coords(lift)


def affine_invariant_distance(
    predicted: jax.Array,
    target: jax.Array,
) -> jax.Array:
    inv_sqrt_target = matrix_inv_sqrt(target)
    aligned = inv_sqrt_target @ predicted @ inv_sqrt_target
    eigenvalues, _ = jnp.linalg.eigh(aligned)
    return jnp.linalg.norm(jnp.log(eigenvalues))


# ---------------------------------------------------------------------------
# Loss / metric factories
# ---------------------------------------------------------------------------


def make_supervised_mse_loss(
    *,
    input_key: str,
    target_key: str,
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array]
    | None = None,
) -> LossFn:
    def default_prediction_fn(
        model: eqx.Module,
        inputs: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        del key
        return model(inputs)

    predict = prediction_fn or default_prediction_fn

    def loss_fn(
        model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        inputs = replace_masked_spd_examples(batch[input_key], mask)
        targets = replace_masked_spd_examples(batch[target_key], mask)
        predictions = predict(model, inputs, key)
        axes = tuple(range(1, predictions.ndim))
        per_example_loss = jnp.mean((predictions - targets) ** 2, axis=axes)
        return masked_mean(per_example_loss, mask)

    return loss_fn


def make_georax_chart_loss(
    *,
    input_key: str,
    target_key: str,
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
    geometry: SPD,
    base_matrix: jax.Array,
    beta: float = 0.0,
) -> LossFn:
    def loss_fn(
        model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        inputs = replace_masked_spd_examples(batch[input_key], mask)
        targets = replace_masked_spd_examples(batch[target_key], mask)
        predictions = prediction_fn(model, inputs, key)
        coord_targets = jax.vmap(
            lambda target: inverse_congruence_coords(geometry, base_matrix, target)
        )(targets)
        coord_predictions = jax.vmap(
            lambda predicted: inverse_congruence_coords(geometry, base_matrix, predicted)
        )(predictions)
        diff = coord_predictions - coord_targets
        per_example_loss = jnp.mean(diff**2, axis=-1)
        if beta > 0.0:
            distances = jax.vmap(affine_invariant_distance)(predictions, targets)
            per_example_loss = per_example_loss + beta * (distances**2)
        return masked_mean(per_example_loss, mask)

    return loss_fn


def make_riemannian_distance_metric(
    *,
    input_key: str,
    target_key: str,
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
) -> LossFn:
    def metric_fn(
        model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        inputs = replace_masked_spd_examples(batch[input_key], mask)
        targets = replace_masked_spd_examples(batch[target_key], mask)
        predictions = prediction_fn(model, inputs, key)
        distances = jax.vmap(affine_invariant_distance)(predictions, targets)
        return masked_mean(distances, mask)

    return metric_fn
