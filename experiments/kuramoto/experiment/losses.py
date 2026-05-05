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

The total loss is a uniform mean across horizons. This generalises the
single-horizon energy score in
`experiments/rna/experiment/rna_losses.py:68-100`.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from experiments.rna.models.torus import wrap_to_pi

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
    model: eqx.Module, batch: PyTree, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """vmap the NSDE over the batch dimension. Returns (theta, omega) trajectories."""
    theta0 = batch["theta0"]
    omega0 = batch["omega0"]
    sample_keys = jax.random.split(key, theta0.shape[0])
    return jax.vmap(model)(theta0, omega0, sample_keys)


def _order_parameter(theta: jax.Array) -> jax.Array:
    r"""Kuramoto order parameter $r(t) = |\tfrac{1}{N}\sum_j e^{i\theta_j}|$.

    Reduces over the trailing oscillator axis. Broadcasts across any
    leading batch / time / sample axes.
    """
    return jnp.abs(jnp.mean(jnp.exp(1j * theta), axis=-1))


def make_multi_horizon_energy_score(
    horizons: Sequence[float] = (0.125, 0.25, 0.5, 1.0),
    n_samples: int = 4,
    aux_r_weight: float = 0.1,
) -> LossFn:
    """Average energy score across multiple horizons of the trajectory.

    If ``aux_r_weight > 0`` an auxiliary moment-matching term is added
    that drives the sample-mean Kuramoto order parameter $r(t)$ toward
    the data-mean $r(t)$ at each horizon. The strictly-proper energy
    score is moment-blind, so without this auxiliary term the slow
    synchronisation onset is invisible to the gradient signal.
    """
    horizons_t = jnp.asarray(horizons, dtype=jnp.float32)

    def loss_fn(model, batch, mask, key):
        del mask  # batches are dense for the Kuramoto dataset
        target_theta = batch["theta_traj"]  # (B, T_obs, N)
        target_omega = batch["omega_traj"]
        T_obs = target_theta.shape[1]

        # Indices into the saved-time axis for the requested horizons.
        idx = jnp.clip((horizons_t * (T_obs - 1)).astype(jnp.int32), 1, T_obs - 1)

        sample_keys = jax.random.split(key, n_samples)

        def one_sample(sk):
            return _kuramoto_predict_batch(model, batch, sk)

        sample_theta, sample_omega = jax.vmap(one_sample)(sample_keys)
        # sample_theta: (S, B, T_obs, N), target_theta: (B, T_obs, N)

        per_horizon_theta_s = sample_theta[:, :, idx, :]  # (S, B, H, N)
        per_horizon_omega_s = sample_omega[:, :, idx, :]
        per_horizon_theta_t = target_theta[:, idx, :]  # (B, H, N)
        per_horizon_omega_t = target_omega[:, idx, :]

        # Divergence: E_{X~F} d(X, y_h) for each (b, h)
        divergence = jnp.mean(
            _state_distance(
                per_horizon_theta_s,
                per_horizon_omega_s,
                per_horizon_theta_t[None],
                per_horizon_omega_t[None],
            ),
            axis=0,
        )  # (B, H)

        # Diversity: 0.5 * E_{X, X'~F} d(X, X') for each (b, h)
        pairwise = _state_distance(
            per_horizon_theta_s[:, None],
            per_horizon_omega_s[:, None],
            per_horizon_theta_s[None, :],
            per_horizon_omega_s[None, :],
        )  # (S, S, B, H)
        diversity = 0.5 * jnp.mean(pairwise, axis=(0, 1))  # (B, H)

        per_example_per_horizon = divergence - diversity  # (B, H)
        main = jnp.mean(per_example_per_horizon)

        if aux_r_weight == 0.0:
            return main

        # Auxiliary moment-matching: per-horizon mean order parameter.
        r_sample = _order_parameter(per_horizon_theta_s)  # (S, B, H)
        r_target = _order_parameter(per_horizon_theta_t)  # (B, H)
        r_pred_mean = jnp.mean(r_sample, axis=(0, 1))  # (H,)
        r_data_mean = jnp.mean(r_target, axis=0)  # (H,)
        aux = jnp.sum((r_pred_mean - r_data_mean) ** 2)
        return main + aux_r_weight * aux

    return loss_fn


def make_eval_metrics(
    horizons: Sequence[float] = (0.125, 0.25, 0.5, 1.0),
    n_samples: int = 4,
) -> Callable:
    """Per-horizon mean wrapped-MAE on theta + MAE on omega for monitoring.

    Returns a callable that produces a dict mapping a metric key to a scalar.
    Used for richer per-epoch logging beyond the energy score.
    """
    horizons_t = jnp.asarray(horizons, dtype=jnp.float32)

    @eqx.filter_jit
    def metric_fn(model, batch, key):
        target_theta = batch["theta_traj"]
        target_omega = batch["omega_traj"]
        T_obs = target_theta.shape[1]
        idx = jnp.clip((horizons_t * (T_obs - 1)).astype(jnp.int32), 1, T_obs - 1)
        sample_keys = jax.random.split(key, n_samples)

        def one_sample(sk):
            return _kuramoto_predict_batch(model, batch, sk)

        sample_theta, sample_omega = jax.vmap(one_sample)(sample_keys)
        theta_h = jnp.mean(sample_theta[:, :, idx, :], axis=0)  # (B, H, N)
        omega_h = jnp.mean(sample_omega[:, :, idx, :], axis=0)
        theta_t = target_theta[:, idx, :]
        omega_t = target_omega[:, idx, :]
        wrapped_mae = jnp.mean(jnp.abs(_wrapped_diff(theta_h, theta_t)), axis=(0, 2))  # (H,)
        omega_mae = jnp.mean(jnp.abs(omega_h - omega_t), axis=(0, 2))
        return {
            "theta_mae_per_horizon": wrapped_mae,
            "omega_mae_per_horizon": omega_mae,
            "theta_mae_mean": jnp.mean(wrapped_mae),
            "omega_mae_mean": jnp.mean(omega_mae),
        }

    return metric_fn
