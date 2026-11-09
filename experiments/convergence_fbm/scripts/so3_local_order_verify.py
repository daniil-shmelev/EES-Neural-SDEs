"""Measure cosine EES local recovery on a nonlinear SO(3) ODE.

For reversible order 5, compare the observed one-step recovery rate with h**6, excluding errors at the floating-point floor.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from experiments.convergence_fbm.scripts.so3_reversibility_verify import (
    cosine_ees_step as driver_step,
)


def antisym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M - M.T)


def make_xi(seed: int, scale: float = 0.4):
    rng = np.random.default_rng(seed)
    A = [scale * rng.standard_normal((3, 3)) for _ in range(5)]

    def xi(Y: np.ndarray) -> np.ndarray:
        Y2 = Y @ Y
        Y3 = Y2 @ Y
        Y4 = Y3 @ Y
        M = A[0] + A[1] @ Y + A[2] @ Y2 + A[3] @ Y3 + A[4] @ Y4
        return antisym(M)

    return xi


def cosine_ees_step(Y: np.ndarray, xi, h: float) -> np.ndarray:
    """One step (ODE case, increment h along the one 'driver')."""
    return driver_step(Y, [xi], np.asarray([h]))


def main():
    print("=" * 76)
    print("  Deterministic local-order verification: cosine EES (p=5) on SO(3)")
    print("=" * 76)
    print()
    print("  ODE: dY/dt = xi(Y) Y, xi : R^{3x3} -> so(3), poly-4 in Y.")
    print("  Measure E(h) = || Phi_{-h}(Phi_{+h}(Y_0)) - Y_0 ||_F.")
    print("  Expect slope 6 (ambient BCK order 5 => local err O(h^6)).")
    print()

    for seed in range(4):
        xi = make_xi(seed)
        rng_Y = np.random.default_rng(100 + seed)
        Y0 = expm(antisym(0.3 * rng_Y.standard_normal((3, 3))))
        # Check Y0 is in SO(3).
        assert abs(np.linalg.det(Y0) - 1.0) < 1e-10
        assert np.allclose(Y0 @ Y0.T, np.eye(3), atol=1e-10)

        hs = [2 ** (-k) for k in range(3, 12)]
        errs = []
        for h in hs:
            Y1 = cosine_ees_step(Y0, xi, h)
            Y_back = cosine_ees_step(Y1, xi, -h)
            errs.append(float(np.linalg.norm(Y_back - Y0, ord="fro")))

        print(f"--- seed={seed} ---")
        print(f"  {'h':>11} {'err':>13} {'err/h^6':>14}")
        for h, e in zip(hs, errs):
            print(f"  {h:>11.5g} {e:>13.3e} {e / h**6:>14.3e}")

        # Fit slope on the asymptotic tail (drop first two h and any
        # floating-point noise floor).
        tail = [(h, e) for h, e in zip(hs, errs) if e > 1e-14]
        tail = tail[2:-1]  # drop 2 coarsest and the very last in case
        # round-off is starting
        if len(tail) >= 3:
            hh, ee = zip(*tail)
            slope, intercept = np.polyfit(np.log(hh), np.log(ee), 1)
            print(f"  fitted slope (asymptotic tail) = {slope:.4f}  (expect 6)")
        print()


if __name__ == "__main__":
    main()
