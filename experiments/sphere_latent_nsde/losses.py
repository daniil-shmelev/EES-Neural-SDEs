"""Losses and metrics for the activity latent SDE."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from experiments.sphere_latent_nsde.models import LatentSDEOutput

PyTree = Any


def masked_mean(values: Array, mask: Array) -> Array:
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def gather_time(values: Array, tids: Array) -> Array:
    """Gather ``values[..., time, channels]`` at per-example integer tids.

    ``values`` is expected to have shape ``(mc, batch, T, channels)`` and
    ``tids`` shape ``(batch, L)``.
    """

    mc, batch, _, channels = values.shape
    idx = jnp.broadcast_to(tids[None, :, :, None], (mc, batch, tids.shape[1], channels))
    return jnp.take_along_axis(values, idx, axis=2)


def gaussian_nll(x: Array, loc: Array, scale: Array) -> Array:
    log_scale = jnp.log(scale)
    return 0.5 * jnp.square((x - loc) / scale) + log_scale + 0.5 * jnp.log(2.0 * jnp.pi)


def _reconstruction_terms(
    output: LatentSDEOutput,
    batch: PyTree,
    *,
    decoder_scale: Array,
) -> tuple[Array, Array]:
    mu = gather_time(output.recon_mu, batch["evd_tid"])
    obs = batch["evd_obs"][None]
    mask = batch["evd_msk"][None]
    nll = gaussian_nll(obs, mu, decoder_scale)
    nll = jnp.where(mask == 1, nll, 0.0)
    log_pxz_per_example = jnp.sum(jnp.mean(nll, axis=0), axis=(1, 2))
    numel = jnp.sum(batch["evd_msk"], axis=(1, 2))
    return log_pxz_per_example, numel


def _cross_entropy_per_time(logits: Array, targets: Array) -> Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]


def auxiliary_loss(output: LatentSDEOutput, batch: PyTree) -> Array:
    logits_at_t = gather_time(output.aux_logits, batch["aux_tid"])
    logits_mean = jnp.mean(logits_at_t, axis=0)
    ce = _cross_entropy_per_time(logits_mean, batch["aux_obs"])
    return jnp.mean(ce, axis=1)


def auxiliary_accuracy(output: LatentSDEOutput, batch: PyTree) -> Array:
    logits_at_t = gather_time(output.aux_logits, batch["aux_tid"])
    logits_mean = jnp.mean(logits_at_t, axis=0)
    pred = jnp.argmax(logits_mean, axis=-1)
    return jnp.mean((pred == batch["aux_obs"]).astype(jnp.float32), axis=1)


def activity_loss_value(
    model,
    batch,
    mask,
    key,
    *,
    mc_samples: int,
    kl0_weight: float = 1e-4,
    klp_weight: float = 1e-4,
    pxz_weight: float = 1.0,
    aux_weight: float = 10.0,
    aux_weight_mul: Array | float = 1.0,
) -> Array:
    output = model(batch, key, mc_samples=mc_samples)
    decoder_scale = jnp.maximum(jnp.square(model.recon_sigma), 1e-6)
    log_pxz, numel = _reconstruction_terms(output, batch, decoder_scale=decoder_scale)
    elbo_per_example = (
        kl0_weight * output.kl0
        + klp_weight * output.klp
        + pxz_weight * log_pxz
    ) / jnp.maximum(numel, 1.0)
    aux_per_example = auxiliary_loss(output, batch)
    total = elbo_per_example + aux_weight * aux_weight_mul * aux_per_example
    return masked_mean(total, mask)


def activity_metrics_value(
    model,
    batch,
    mask,
    key,
    *,
    mc_samples: int,
    kl0_weight: float = 1e-4,
    klp_weight: float = 1e-4,
    pxz_weight: float = 1.0,
    aux_weight: float = 10.0,
    aux_weight_mul: Array | float = 1.0,
) -> dict[str, Array]:
    output = model(batch, key, mc_samples=mc_samples)
    decoder_scale = jnp.maximum(jnp.square(model.recon_sigma), 1e-6)
    log_pxz, numel = _reconstruction_terms(output, batch, decoder_scale=decoder_scale)
    elbo_per_example = (
        kl0_weight * output.kl0
        + klp_weight * output.klp
        + pxz_weight * log_pxz
    ) / jnp.maximum(numel, 1.0)
    aux_per_example = auxiliary_loss(output, batch)
    loss_per_example = elbo_per_example + aux_weight * aux_weight_mul * aux_per_example
    aux_acc = masked_mean(auxiliary_accuracy(output, batch), mask)
    return {
        "loss": masked_mean(loss_per_example, mask),
        "aux_val": masked_mean(aux_per_example, mask),
        "aux_acc": aux_acc,
        "aux_acc_pct": 100.0 * aux_acc,
        "elbo_val": masked_mean(elbo_per_example, mask),
        "elbo_kl0": masked_mean(output.kl0, mask),
        "elbo_klp": masked_mean(output.klp, mask),
        "elbo_log_pxz": masked_mean(log_pxz, mask),
    }
