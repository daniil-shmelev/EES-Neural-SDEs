"""Directly verify the antisymmetric order of Scheme A (2N-register CF-EES(2,5;1/10))
on a deterministic ODE on a linear Lie group.

This script re-implements CF-EES(2,5;1/10) in its TRUE 2N-register form
(Scheme A, the one used by the empirical sweep and by georax's CFEES25):

    for l = 1, 2, 3:
        K_l = xi(Y_tilde_{l-1})              # evaluated at PREVIOUS cumulative state
        V_l = h * sum_{i<=l} beta[l][i] * K_i  # K_1 reused across l, NOT re-evaluated
        Y_tilde_l = exp(V_l) @ Y_tilde_{l-1}

This is different from the "cascade of classical sub-schemes" (Scheme B) whose
character was computed by mkw_verify_cf.py.

We measure ONE-STEP LOCAL reversibility error:
    err(h) = || Phi_{-h}(Phi_h(Y_0)) - Y_0 ||

If Scheme A has MKW antisymmetric order q, then err(h) = O(h^{q+1}).
For q = 5 (the expected "EES(2,5)" behavior on a Lie group), slope should be 6.

Setup:
    - Linear Lie group G = GL(3, R) acting on itself by left translation Λ(g, Y) = g Y.
    - Generic nonlinear xi(Y) = A_0 + A_1 Y + A_2 Y^T + A_3 (Y @ Y^T).
      (Note: xi maps G -> gl(3) = Lie algebra of gl(3); we don't need it to take
       values in a proper Lie subalgebra for this test — the question is about
       the local reversibility defect in the ambient matrix norm.)

Usage:
    python scheme_a_order_verify.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


# ============================================================================
# CF-EES(2,5; 1/10) weight vectors (from the Proposition we added to the main ms)
# ============================================================================

BETA = np.array([
    [1/3,     0,       0],
    [-7/16,   15/16,   0],
    [49/240, -7/16,   2/5],
])


# ============================================================================
# Generic vector field on GL(3, R)  (xi : G -> gl(3))
# ============================================================================

def make_generic_xi(seed: int = 0):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((3, 3))
    A1 = rng.standard_normal((3, 3))
    A2 = rng.standard_normal((3, 3))
    A3 = rng.standard_normal((3, 3))

    def xi(Y: np.ndarray) -> np.ndarray:
        # Nonlinear in Y to exercise as many elementary differentials as possible.
        return A0 + A1 @ Y + Y @ A2 + A3 @ (Y @ Y.T)

    return xi


# ============================================================================
# Scheme A: one step of CF-EES(2,5; 1/10) via the 2N-register recurrence.
# Action Λ(g, Y) = g @ Y (left translation on GL(3, R)).
# ============================================================================

def cfees25_scheme_A_step(Y: np.ndarray, xi, h: float) -> np.ndarray:
    """One step of CF-EES(2,5; 1/10) from Y, step size h, using Scheme A
    (2N-register recurrence with K reuse).

    K_l = xi(Y_tilde_{l-1})  -- evaluated at cumulative previous state.
    V_l = h * sum_{i<=l} beta[l,i] * K_i  -- K_i's are REUSED across l.
    Y_tilde_l = exp(V_l) @ Y_tilde_{l-1}.
    """
    Y_prev = Y
    Ks = []
    for l in range(3):
        K_l = xi(Y_prev)
        Ks.append(K_l)
        V_l = h * sum(BETA[l, i] * Ks[i] for i in range(l + 1))
        Y_prev = expm(V_l) @ Y_prev
    return Y_prev


# ============================================================================
# Local reversibility test
# ============================================================================

def local_reversibility_error(Y0: np.ndarray, xi, h: float) -> float:
    """One step forward with step h, then one step backward with step -h.
    Returns || Phi_{-h}(Phi_h(Y0)) - Y0 ||_F."""
    Y1 = cfees25_scheme_A_step(Y0, xi, h)
    Y_back = cfees25_scheme_A_step(Y1, xi, -h)
    return float(np.linalg.norm(Y_back - Y0, ord="fro"))


def fit_slope(hs, errs):
    log_h = np.log(np.asarray(hs))
    log_e = np.log(np.asarray(errs))
    slope, intercept = np.polyfit(log_h, log_e, 1)
    return slope, intercept


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 72)
    print("  Direct verification of Scheme A (2N-CF-EES(2,5;1/10)) antisymmetric")
    print("  order via ONE-STEP LOCAL forward-backward reversibility error.")
    print("=" * 72)
    print()
    print("  Scheme: G = GL(3, R), Λ(g,Y) = gY, generic nonlinear xi.")
    print("  Expected: if antisym order q = 5, local err(h) ~ h^6.")
    print()

    for seed in range(4):
        xi = make_generic_xi(seed)
        Y0 = np.eye(3) + 0.1 * np.random.default_rng(100 + seed).standard_normal((3, 3))

        hs = [2 ** (-k) for k in range(3, 10)]
        errs = []
        for h in hs:
            err = local_reversibility_error(Y0, xi, h)
            errs.append(err)

        print(f"  --- seed = {seed} ---")
        print(f"  {'h':>12} {'err':>15} {'err/h^6':>15}")
        for h, err in zip(hs, errs):
            ratio = err / (h ** 6)
            print(f"  {h:>12.6g} {err:>15.3e} {ratio:>15.3e}")

        # Fit only on the asymptotic tail (small h, above roundoff).
        # Skip the largest h (pre-asymptotic) and the smallest (roundoff contaminated).
        # For seed-0 the smallest h is fine; we trim from both ends to be safe.
        fit_pairs = [(h, e) for h, e in zip(hs, errs) if 1e-13 < e]
        fit_pairs = fit_pairs[2:]  # drop two largest h (pre-asymptotic)
        if len(fit_pairs) >= 2:
            fit_hs, fit_errs = zip(*fit_pairs)
            slope, _ = fit_slope(fit_hs, fit_errs)
            print(f"  fitted slope (asymptotic tail) = {slope:.4f}  (expected 6)")
        else:
            print(f"  not enough clean points to fit")
        print()


if __name__ == "__main__":
    main()
