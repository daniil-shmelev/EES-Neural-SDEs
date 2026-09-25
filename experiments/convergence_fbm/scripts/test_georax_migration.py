"""Preserve the benchmark's left action when using Georax's right-action SO."""
import numpy as np
from scipy.linalg import expm

from experiments.convergence_fbm.scripts.so3_reversibility_verify import (
    cfees25_step, integrate, make_so3_vfs,
)


def test_library_step_matches_original_left_action():
    fields = make_so3_vfs(3, seed=42)
    start = expm(np.array([[0., -.2, .1], [.2, 0., -.3], [-.1, .3, 0.]]))
    increment = np.array([.02, -.01, .03])
    # Published stage combinations, evaluated independently with exact exponentials.
    weights = ((1/3,), (-7/16, 15/16), (49/240, -7/16, 2/5))
    reference = start.copy()
    generators = []
    for row in weights:
        generators.append(sum(vf(reference) * dx for vf, dx in zip(fields, increment)))
        reference = expm(sum(w * g for w, g in zip(row, generators))) @ reference
    actual = cfees25_step(start, fields, increment)
    np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=2e-14, rtol=0)


def test_forward_reverse_scan_uses_same_driver_increments():
    fields = make_so3_vfs(3, seed=7)
    path = np.random.default_rng(8).normal(scale=.01, size=(5, 3)).cumsum(axis=0)
    forward = integrate(np.eye(3), fields, path)
    reverse = integrate(forward, fields, path, reverse=True)
    expected = forward
    for increment in -np.diff(path, axis=0)[::-1]:
        expected = cfees25_step(expected, fields, increment)
    np.testing.assert_allclose(reverse, expected, atol=2e-14, rtol=0)
