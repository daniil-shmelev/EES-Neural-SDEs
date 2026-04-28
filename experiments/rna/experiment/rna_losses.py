"""Wrapped-angle losses and metrics for torus-valued RNA targets."""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from experiments.rna.experiment.losses import masked_mean
from experiments.rna.models.torus import wrap_to_pi

PyTree = Any
LossFn = Callable[[eqx.Module, PyTree, jax.Array, jax.Array], jax.Array]


def wrapped_angle_diff(a: jax.Array, b: jax.Array) -> jax.Array:
    return wrap_to_pi(a - b)


def rna_predict_batch(
    model: eqx.Module, batch: PyTree, key: jax.Array
) -> jax.Array:
    """vmap the TorusNeuralSDE over the batch dimension.

    Expects ``batch`` to contain ``context_angles``, ``context_bases``,
    ``target_bases``, ``future_bases``, and ``future_mask``. Returns
    predicted angles of shape ``(batch, num_angles)``.
    """
    context_angles = batch["context_angles"]
    context_bases = batch["context_bases"]
    target_bases = batch["target_bases"]
    future_bases = batch["future_bases"]
    future_mask = batch["future_mask"]
    sample_keys = jax.random.split(key, context_angles.shape[0])
    return jax.vmap(
        lambda ca, cb, tb, fb, fm, k: model(ca, cb, tb, fb, fm, k)
    )(context_angles, context_bases, target_bases, future_bases, future_mask, sample_keys)


def make_wrapped_mse_loss(*, target_key: str = "target_angles") -> LossFn:
    """Wrapped MSE between the single-sample SDE prediction and the target."""

    def loss_fn(model, batch, mask, key):
        predictions = rna_predict_batch(model, batch, key)
        targets = batch[target_key]
        per_example = jnp.mean(wrapped_angle_diff(predictions, targets) ** 2, axis=-1)
        return masked_mean(per_example, mask)

    return loss_fn


def make_wrapped_mae_metric(*, target_key: str = "target_angles") -> LossFn:
    """Wrapped mean absolute error; used for model selection and reporting."""

    def metric_fn(model, batch, mask, key):
        predictions = rna_predict_batch(model, batch, key)
        targets = batch[target_key]
        per_example = jnp.mean(
            jnp.abs(wrapped_angle_diff(predictions, targets)), axis=-1
        )
        return masked_mean(per_example, mask)

    return metric_fn


def make_wrapped_energy_score_loss(
    *, target_key: str = "target_angles", n_samples: int = 8
) -> LossFn:
    """Strictly proper energy score for the SDE's predictive distribution.

    For each example, draws ``n_samples`` stochastic rollouts of the SDE and
    evaluates the wrapped-L1 energy score

        ES(F, y) = E_{X~F} ||X - y||_1 - 0.5 * E_{X, X'~F} ||X - X'||_1

    with the per-component metric ``||a - b||_1 =
    sum_i |arctan2(sin(a_i - b_i), cos(a_i - b_i))|``. The first term drives
    the distribution's centre toward the target; the second term rewards
    diversity across samples and prevents mode collapse.
    """

    def loss_fn(model, batch, mask, key):
        targets = batch[target_key]
        sample_keys = jax.random.split(key, n_samples)
        samples = jax.vmap(lambda k: rna_predict_batch(model, batch, k))(sample_keys)
        # samples: (S, B, D), targets: (B, D)
        divergence = jnp.mean(
            jnp.sum(jnp.abs(wrapped_angle_diff(samples, targets[None])), axis=-1),
            axis=0,
        )  # (B,)
        pairwise = jnp.sum(
            jnp.abs(wrapped_angle_diff(samples[:, None], samples[None, :])), axis=-1
        )  # (S, S, B)
        diversity = 0.5 * jnp.mean(pairwise, axis=(0, 1))  # (B,)
        per_example = divergence - diversity
        return masked_mean(per_example, mask)

    return loss_fn
