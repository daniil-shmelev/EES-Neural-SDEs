r"""Multi-horizon energy-score loss for the Kuramoto NSDE.

For each example, draws ``n_samples`` stochastic SDE rollouts and
evaluates the energy score at each of ``horizons`` (fractions of the
trajectory length, e.g. ``(0.125, 0.25, 0.5, 1.0)``). Per-horizon score:

.. math::
    \mathrm{ES}_h(F, y_h) = \mathbb{E}_{X \sim F} \, d(X, y_h) -
        \tfrac{1}{2}\,\mathbb{E}_{X, X' \sim F} \, d(X, X')

with the wrapped-on-$\theta$, plain-on-$\omega$ distance

.. math::
    d((\theta_a, \omega_a), (\theta_b, \omega_b)) =
        \sum_i |\mathrm{wrap}(\theta_a^i - \theta_b^i)| +
        \sum_i |\omega_a^i - \omega_b^i|.

The total loss is a uniform mean across horizons.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ees_core.torus import wrap_to_pi

PyTree = Any
LossFn = Callable[[eqx.Module, PyTree, jax.Array, jax.Array], jax.Array]


def _wrapped_diff(a: jax.Array, b: jax.Array) -> jax.Array:
    return wrap_to_pi(a - b)


def _state_distance(
    theta_a: jax.Array,
    omega_a: jax.Array,
    theta_b: jax.Array,
    omega_b: jax.Array,
) -> jax.Array:
    r"""$d = \sum_i |\mathrm{wrap}(\theta_a^i - \theta_b^i)| + \sum_i |\omega_a^i - \omega_b^i|$.

    Reduces over the trailing oscillator axis only; broadcasts across any
    leading batch / sample axes the caller wants.
    """
    theta_term = jnp.sum(jnp.abs(_wrapped_diff(theta_a, theta_b)), axis=-1)
    omega_term = jnp.sum(jnp.abs(omega_a - omega_b), axis=-1)
    return theta_term + omega_term


def _kuramoto_predict_batch(
    model: eqx.Module,
    batch: PyTree,
    key: jax.Array,
    *,
    save_indices=None,
) -> tuple[jax.Array, jax.Array]:
    """vmap the NSDE over the batch dimension. Returns (theta, omega) trajectories."""
    theta0 = batch["theta0"]
    omega0 = batch["omega0"]
    sample_keys = jax.random.split(key, theta0.shape[0])
    if save_indices is None:
        return jax.vmap(model)(theta0, omega0, sample_keys)
    return jax.vmap(lambda th, om, sk: model(th, om, sk, save_indices=save_indices))(
        theta0,
        omega0,
        sample_keys,
    )


def _order_parameter(theta: jax.Array) -> jax.Array:
    r"""Kuramoto order parameter $r(t) = |\tfrac{1}{N}\sum_j e^{i\theta_j}|$.

    Reduces over the trailing oscillator axis. Broadcasts across any
    leading batch / time / sample axes.
    """
    return jnp.abs(jnp.mean(jnp.exp(1j * theta), axis=-1))


def _sample_horizons(model, batch, key, horizons, n_samples):
    target_theta = batch["theta_traj"]
    target_omega = batch["omega_traj"]
    n_obs = target_theta.shape[1]
    idx = jnp.clip((horizons * (n_obs - 1)).astype(jnp.int32), 1, n_obs - 1)
    sample_keys = jax.random.split(key, n_samples)
    sample_theta, sample_omega = jax.vmap(
        lambda sk: _kuramoto_predict_batch(model, batch, sk, save_indices=idx)
    )(sample_keys)
    return (
        sample_theta,
        sample_omega,
        target_theta[:, idx, :],
        target_omega[:, idx, :],
    )


def _batch_mean(values, mask):
    """Average axis zero over real examples, excluding padded batch entries."""
    if mask is None:
        return jnp.mean(values, axis=0)
    mask = jnp.asarray(mask, dtype=bool)
    expanded = mask.reshape((mask.shape[0],) + (1,) * (values.ndim - 1))
    return jnp.sum(jnp.where(expanded, values, 0), axis=0) / jnp.sum(mask)


def _energy_score(samples, mask, aux_r_weight):
    theta_s, omega_s, theta_t, omega_t = samples
    divergence = jnp.mean(
        _state_distance(theta_s, omega_s, theta_t[None], omega_t[None]),
        axis=0,
    )  # (B, H)
    diversity = 0.5 * jnp.mean(
        _state_distance(
            theta_s[:, None],
            omega_s[:, None],
            theta_s[None, :],
            omega_s[None, :],
        ),
        axis=(0, 1),
    )
    main = jnp.mean(_batch_mean(divergence - diversity, mask))
    if aux_r_weight == 0.0:
        return main
    r_pred_mean = _batch_mean(jnp.mean(_order_parameter(theta_s), axis=0), mask)
    r_data_mean = _batch_mean(_order_parameter(theta_t), mask)
    return main + aux_r_weight * jnp.sum((r_pred_mean - r_data_mean) ** 2)


def _forecast_metrics(samples, mask):
    theta_s, omega_s, theta_t, omega_t = samples
    theta_h = jnp.mean(theta_s, axis=0)
    omega_h = jnp.mean(omega_s, axis=0)
    wrapped_mae = _batch_mean(
        jnp.mean(jnp.abs(_wrapped_diff(theta_h, theta_t)), axis=-1), mask
    )
    omega_mae = _batch_mean(jnp.mean(jnp.abs(omega_h - omega_t), axis=-1), mask)
    return {
        "theta_mae_per_horizon": wrapped_mae,
        "omega_mae_per_horizon": omega_mae,
        "theta_mae_mean": jnp.mean(wrapped_mae),
        "omega_mae_mean": jnp.mean(omega_mae),
    }


def make_multi_horizon_energy_score(
    horizons: Sequence[float] = (0.125, 0.25, 0.5, 1.0),
    n_samples: int = 4,
    aux_r_weight: float = 0.1,
) -> LossFn:
    """Mean energy score with optional order-parameter moment matching."""
    horizons_t = jnp.asarray(horizons, dtype=jnp.float32)

    def loss_fn(model, batch, mask, key):
        samples = _sample_horizons(model, batch, key, horizons_t, n_samples)
        return _energy_score(samples, mask, aux_r_weight)

    return loss_fn


def make_eval_metrics(
    horizons: Sequence[float] = (0.125, 0.25, 0.5, 1.0),
    n_samples: int = 4,
) -> Callable:
    """Per-horizon wrapped phase MAE and velocity MAE for monitoring."""
    horizons_t = jnp.asarray(horizons, dtype=jnp.float32)

    @eqx.filter_jit
    def metric_fn(model, batch, key, mask=None):
        samples = _sample_horizons(model, batch, key, horizons_t, n_samples)
        return _forecast_metrics(samples, mask)

    return metric_fn


def make_eval_step(
    horizons: Sequence[float] = (0.125, 0.25, 0.5, 1.0),
    n_samples: int = 4,
    aux_r_weight: float = 0.1,
) -> Callable:
    """Compute loss and metrics from one shared ensemble of SDE forecasts."""
    horizons_t = jnp.asarray(horizons, dtype=jnp.float32)

    @eqx.filter_jit
    def eval_step(model, batch, mask, key):
        samples = _sample_horizons(model, batch, key, horizons_t, n_samples)
        return _energy_score(samples, mask, aux_r_weight), _forecast_metrics(
            samples, mask
        )

    return eval_step
