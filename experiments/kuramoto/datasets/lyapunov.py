r"""Same-noise Lyapunov estimator for the stochastic Kuramoto benchmark.

The model has additive noise, so its tangent equation contains no explicit
stochastic increment.  The base trajectory is nevertheless random, and the
tangent coefficients are evaluated along that random trajectory.

This module uses the one-vector Benettin method.  At every renormalisation it
projects out the common phase and common frequency components.  This restricts
the tangent dynamics to phase/frequency differences and prevents the exact
global-phase neutral mode from masking the largest physically nontrivial
exponent.

The implementation is NumPy-only so the diagnostic can run independently of
the JAX/diffrax training environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LyapunovConfig:
    """Parameters for one ensemble Lyapunov estimate."""

    N: int
    P: float = 0.5
    K: float = 2.0
    m: float = 1.0
    D: float = 0.05
    dt: float = 0.01
    burn_in: float = 50.0
    horizon: float = 200.0
    renorm_interval: float = 0.5
    n_realizations: int = 16
    omega_scale: float = 0.5
    seed: int = 0


def _bimodal_frequencies(N: int, P: float) -> FloatArray:
    n_generators = (N + 1) // 2
    return np.concatenate(
        (
            np.full(n_generators, P, dtype=np.float64),
            np.full(N - n_generators, -P, dtype=np.float64),
        )
    )


def _drift(
    theta: FloatArray,
    omega: FloatArray,
    natural_frequencies: FloatArray,
    K: float,
    m: float,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the Kuramoto drift in O(batch * N) work."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    mean_sin = np.mean(sin_theta, axis=1, keepdims=True)
    mean_cos = np.mean(cos_theta, axis=1, keepdims=True)
    coupling = K * (mean_sin * cos_theta - mean_cos * sin_theta)
    return omega, (-omega + natural_frequencies[None, :] + coupling) / m


def _tangent_drift(
    theta: FloatArray,
    delta_theta: FloatArray,
    delta_omega: FloatArray,
    K: float,
    m: float,
) -> tuple[FloatArray, FloatArray]:
    r"""Apply the drift Jacobian without constructing its dense matrix.

    The coupling Jacobian-vector product is

    .. math::
        K\left[
          \frac1N\sum_j\cos(\theta_j-\theta_i)\delta\theta_j
          -\delta\theta_i\frac1N\sum_j\cos(\theta_j-\theta_i)
        \right].
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    mean_cos = np.mean(cos_theta, axis=1, keepdims=True)
    mean_sin = np.mean(sin_theta, axis=1, keepdims=True)
    mean_cos_delta = np.mean(
        cos_theta * delta_theta, axis=1, keepdims=True
    )
    mean_sin_delta = np.mean(
        sin_theta * delta_theta, axis=1, keepdims=True
    )

    weighted_delta = (
        cos_theta * mean_cos_delta + sin_theta * mean_sin_delta
    )
    row_mean = cos_theta * mean_cos + sin_theta * mean_sin
    coupling_jvp = K * (weighted_delta - row_mean * delta_theta)
    return delta_omega, (-delta_omega + coupling_jvp) / m


def _heun_step(
    theta: FloatArray,
    omega: FloatArray,
    delta_theta: FloatArray,
    delta_omega: FloatArray,
    noise_increment: FloatArray,
    natural_frequencies: FloatArray,
    K: float,
    m: float,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Advance the base state and derivative of the same stochastic step."""
    dtheta_0, domega_0 = _drift(
        theta, omega, natural_frequencies, K, m
    )
    ddelta_theta_0, ddelta_omega_0 = _tangent_drift(
        theta, delta_theta, delta_omega, K, m
    )

    theta_predictor = theta + dt * dtheta_0
    omega_predictor = omega + dt * domega_0 + noise_increment
    delta_theta_predictor = delta_theta + dt * ddelta_theta_0
    delta_omega_predictor = delta_omega + dt * ddelta_omega_0

    dtheta_1, domega_1 = _drift(
        theta_predictor, omega_predictor, natural_frequencies, K, m
    )
    ddelta_theta_1, ddelta_omega_1 = _tangent_drift(
        theta_predictor,
        delta_theta_predictor,
        delta_omega_predictor,
        K,
        m,
    )

    theta_next = theta + 0.5 * dt * (dtheta_0 + dtheta_1)
    omega_next = (
        omega + 0.5 * dt * (domega_0 + domega_1) + noise_increment
    )
    delta_theta_next = delta_theta + 0.5 * dt * (
        ddelta_theta_0 + ddelta_theta_1
    )
    delta_omega_next = delta_omega + 0.5 * dt * (
        ddelta_omega_0 + ddelta_omega_1
    )
    theta_next = (theta_next + np.pi) % (2.0 * np.pi) - np.pi
    return (
        theta_next,
        omega_next,
        delta_theta_next,
        delta_omega_next,
    )


def _project_and_normalize(
    delta_theta: FloatArray,
    delta_omega: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Project to phase/frequency differences, then normalize each vector."""
    delta_theta = delta_theta - np.mean(
        delta_theta, axis=1, keepdims=True
    )
    delta_omega = delta_omega - np.mean(
        delta_omega, axis=1, keepdims=True
    )
    norms = np.sqrt(
        np.sum(delta_theta**2 + delta_omega**2, axis=1)
    )
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(float).tiny):
        raise FloatingPointError("Tangent norm vanished or became non-finite.")
    return (
        delta_theta / norms[:, None],
        delta_omega / norms[:, None],
        norms,
    )


def _integer_steps(duration: float, dt: float, name: str) -> int:
    steps = round(duration / dt)
    if steps <= 0 or not np.isclose(steps * dt, duration, atol=1e-12):
        raise ValueError(f"{name}={duration} must be a positive multiple of dt={dt}.")
    return steps


def _summarize(values: FloatArray) -> dict[str, Any]:
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    standard_error = std / np.sqrt(n) if n > 1 else 0.0
    half_width = 1.96 * standard_error
    return {
        "mean": mean,
        "std_across_realizations": std,
        "standard_error": standard_error,
        "normal_95_ci": [mean - half_width, mean + half_width],
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "per_realization": values.tolist(),
    }


def estimate_largest_nontrivial_exponent(
    cfg: LyapunovConfig,
) -> dict[str, Any]:
    """Estimate the projected largest same-noise Lyapunov exponent.

    The returned checkpoint estimates use the first 25%, 50%, 75%, and 100%
    of the requested accumulation horizon.  Uncertainty is the standard error
    across independent initial conditions and Brownian paths.
    """
    if cfg.N < 2:
        raise ValueError("N must be at least 2 after quotienting the phase mode.")
    if cfg.m <= 0.0 or cfg.D < 0.0 or cfg.n_realizations < 1:
        raise ValueError("Require m > 0, D >= 0, and n_realizations >= 1.")

    burn_steps = _integer_steps(cfg.burn_in, cfg.dt, "burn_in")
    accumulation_steps = _integer_steps(cfg.horizon, cfg.dt, "horizon")
    renorm_steps = _integer_steps(
        cfg.renorm_interval, cfg.dt, "renorm_interval"
    )
    if burn_steps % renorm_steps or accumulation_steps % renorm_steps:
        raise ValueError(
            "burn_in and horizon must be multiples of renorm_interval."
        )

    rng = np.random.default_rng(cfg.seed)
    shape = (cfg.n_realizations, cfg.N)
    theta = rng.uniform(-np.pi, np.pi, size=shape)
    omega = cfg.omega_scale * rng.standard_normal(size=shape)
    delta_theta = rng.standard_normal(size=shape)
    delta_omega = rng.standard_normal(size=shape)
    delta_theta, delta_omega, _ = _project_and_normalize(
        delta_theta, delta_omega
    )

    natural_frequencies = _bimodal_frequencies(cfg.N, cfg.P)
    noise_scale = np.sqrt(2.0 * cfg.D * cfg.dt) / cfg.m

    def advance_one_step() -> None:
        nonlocal theta, omega, delta_theta, delta_omega
        noise_increment = noise_scale * rng.standard_normal(size=shape)
        theta, omega, delta_theta, delta_omega = _heun_step(
            theta,
            omega,
            delta_theta,
            delta_omega,
            noise_increment,
            natural_frequencies,
            cfg.K,
            cfg.m,
            cfg.dt,
        )

    for step in range(1, burn_steps + 1):
        advance_one_step()
        if step % renorm_steps == 0:
            delta_theta, delta_omega, _ = _project_and_normalize(
                delta_theta, delta_omega
            )

    n_intervals = accumulation_steps // renorm_steps
    log_growth = np.empty(
        (cfg.n_realizations, n_intervals), dtype=np.float64
    )
    for step in range(1, accumulation_steps + 1):
        advance_one_step()
        if step % renorm_steps == 0:
            delta_theta, delta_omega, norms = _project_and_normalize(
                delta_theta, delta_omega
            )
            log_growth[:, step // renorm_steps - 1] = np.log(norms)

    checkpoint_indices = sorted(
        {max(1, round(fraction * n_intervals)) for fraction in (0.25, 0.5, 0.75, 1.0)}
    )
    cumulative_growth = np.cumsum(log_growth, axis=1)
    checkpoints: dict[str, Any] = {}
    for interval_count in checkpoint_indices:
        elapsed = interval_count * cfg.renorm_interval
        exponents = cumulative_growth[:, interval_count - 1] / elapsed
        checkpoints[f"{elapsed:g}"] = _summarize(exponents)

    final_summary = checkpoints[f"{cfg.horizon:g}"]
    ci_low, ci_high = final_summary["normal_95_ci"]
    if ci_low > 0.0:
        classification = "positive"
    elif ci_high < 0.0:
        classification = "negative"
    else:
        classification = "unresolved"

    return {
        "method": "one-vector Benettin on the stochastic Heun map",
        "mode": "same-noise, global phase/frequency components projected out",
        "parameters": {
            "N": cfg.N,
            "P": cfg.P,
            "K": cfg.K,
            "m": cfg.m,
            "D": cfg.D,
            "dt": cfg.dt,
            "burn_in": cfg.burn_in,
            "horizon": cfg.horizon,
            "renorm_interval": cfg.renorm_interval,
            "n_realizations": cfg.n_realizations,
            "omega_scale": cfg.omega_scale,
            "seed": cfg.seed,
        },
        "checkpoints_seconds": checkpoints,
        "final_classification": classification,
    }
