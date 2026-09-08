"""Checks for the same-noise Kuramoto tangent integrator."""

from __future__ import annotations

import numpy as np

from experiments.kuramoto.datasets.lyapunov import (
    LyapunovConfig,
    _bimodal_frequencies,
    _heun_step,
    _project_and_normalize,
    estimate_largest_nontrivial_exponent,
)


def test_heun_tangent_matches_finite_difference():
    rng = np.random.default_rng(11)
    batch, N = 2, 5
    theta = rng.uniform(-1.0, 1.0, size=(batch, N))
    omega = rng.normal(size=(batch, N))
    delta_theta = rng.normal(size=(batch, N))
    delta_omega = rng.normal(size=(batch, N))
    noise = 0.03 * rng.normal(size=(batch, N))
    natural = _bimodal_frequencies(N, 0.5)
    epsilon = 1e-7

    base = _heun_step(
        theta,
        omega,
        delta_theta,
        delta_omega,
        noise,
        natural,
        K=2.0,
        m=1.0,
        dt=0.01,
    )
    perturbed = _heun_step(
        theta + epsilon * delta_theta,
        omega + epsilon * delta_omega,
        delta_theta,
        delta_omega,
        noise,
        natural,
        K=2.0,
        m=1.0,
        dt=0.01,
    )

    finite_difference_theta = (perturbed[0] - base[0]) / epsilon
    finite_difference_omega = (perturbed[1] - base[1]) / epsilon
    np.testing.assert_allclose(
        finite_difference_theta, base[2], rtol=2e-7, atol=5e-9
    )
    np.testing.assert_allclose(
        finite_difference_omega, base[3], rtol=2e-7, atol=5e-9
    )


def test_global_phase_mode_is_exactly_neutral():
    rng = np.random.default_rng(12)
    N = 6
    theta = rng.uniform(-np.pi, np.pi, size=(1, N))
    omega = rng.normal(size=(1, N))
    delta_theta = np.ones((1, N))
    delta_omega = np.zeros((1, N))
    result = _heun_step(
        theta,
        omega,
        delta_theta,
        delta_omega,
        0.02 * rng.normal(size=(1, N)),
        _bimodal_frequencies(N, 0.5),
        K=2.0,
        m=1.0,
        dt=0.01,
    )
    np.testing.assert_allclose(result[2], delta_theta, atol=1e-15)
    np.testing.assert_allclose(result[3], delta_omega, atol=1e-15)


def test_projection_removes_common_components_and_normalizes():
    rng = np.random.default_rng(13)
    delta_theta = rng.normal(size=(3, 8))
    delta_omega = rng.normal(size=(3, 8))
    projected_theta, projected_omega, _ = _project_and_normalize(
        delta_theta, delta_omega
    )
    np.testing.assert_allclose(np.mean(projected_theta, axis=1), 0.0, atol=1e-16)
    np.testing.assert_allclose(np.mean(projected_omega, axis=1), 0.0, atol=1e-16)
    norms = np.sqrt(
        np.sum(projected_theta**2 + projected_omega**2, axis=1)
    )
    np.testing.assert_allclose(norms, 1.0, atol=1e-15)


def test_small_estimate_is_finite():
    estimate = estimate_largest_nontrivial_exponent(
        LyapunovConfig(
            N=2,
            dt=0.02,
            burn_in=0.2,
            horizon=0.4,
            renorm_interval=0.1,
            n_realizations=2,
            seed=14,
        )
    )
    final = estimate["checkpoints_seconds"]["0.4"]
    assert np.isfinite(final["mean"])
