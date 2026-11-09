"""Preserve the benchmark's left action when using Georax's right-action SO."""

import numpy as np
from scipy.linalg import expm

from experiments.convergence_fbm.scripts.so3_reversibility_verify import (
    cosine_ees_step,
    integrate,
    make_so3_vfs,
)


def test_cosine_step_matches_independent_left_action():
    fields = make_so3_vfs(3, seed=42)
    start = expm(np.array([[0.0, -0.2, 0.1], [0.2, 0.0, -0.3], [-0.1, 0.3, 0.0]]))
    increment = np.array([0.02, -0.01, 0.03])
    # Three-stage cosine combinations, evaluated independently with exact exponentials.
    weights = ((0.5,), (-0.5, 1.0), (0.25, -0.5, 0.25))
    reference = start.copy()
    generators = []
    for row in weights:
        generators.append(sum(vf(reference) * dx for vf, dx in zip(fields, increment)))
        reference = expm(sum(w * g for w, g in zip(row, generators))) @ reference
    actual = cosine_ees_step(start, fields, increment)
    np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=2e-14, rtol=0)


def test_forward_reverse_scan_uses_same_driver_increments():
    fields = make_so3_vfs(3, seed=7)
    path = np.random.default_rng(8).normal(scale=0.01, size=(5, 3)).cumsum(axis=0)
    forward = integrate(np.eye(3), fields, path)
    reverse = integrate(forward, fields, path, reverse=True)
    expected = forward
    for increment in -np.diff(path, axis=0)[::-1]:
        expected = cosine_ees_step(expected, fields, increment)
    np.testing.assert_allclose(reverse, expected, atol=2e-14, rtol=0)
