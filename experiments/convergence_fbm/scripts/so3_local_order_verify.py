"""Deterministic local-order verification for CF-EES(2,5;1/10) on SO(3).

Tests the ONE-STEP local reversibility error
    E(h) = || Phi_{-h}(Phi_{+h}(Y_0)) - Y_0 ||_F
for a deterministic ODE on SO(3) with a nonlinear vector field xi:
R^{3x3} -> so(3).

If the scheme has ambient BCK antisymmetric order 5, then E(h) = O(h^6),
i.e. log-log slope = 6.

If the intrinsic MKW post-Lie structure with non-zero torsion governed
the rate (as it would on a non-abelian Lie group with a torsion-bearing
connection), we would see slope 4 instead (MKW order 3 -> O(h^4)).

This is a clean test because there is no fBm rough-path averaging --
the one-step error is deterministic and should lie on a smooth power
curve.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


BETA = np.array([
    [1/3,     0,       0],
    [-7/16,   15/16,   0],
    [49/240, -7/16,   2/5],
], dtype=np.float64)


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


def cfees25_step(Y: np.ndarray, xi, h: float) -> np.ndarray:
    """One step (ODE case, increment h along the one 'driver')."""
    Ks = []
    Y_prev = Y
    for l in range(3):
        K_l = xi(Y_prev)
        Ks.append(K_l)
        V_l = h * sum(BETA[l, i] * Ks[i] for i in range(l + 1))
        Y_prev = expm(V_l) @ Y_prev
    return Y_prev


def main():
    print("=" * 76)
    print("  Deterministic local-order verification: CF-EES(2,5;1/10) on SO(3)")
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
            Y1 = cfees25_step(Y0, xi, h)
            Y_back = cfees25_step(Y1, xi, -h)
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
