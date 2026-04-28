"""Algebraic (exact rational) verification of Scheme A's antisymmetric
order on CF-EES(2,5; 1/10).

Setup: G = GL(3, Q) with left translation Lambda(g, Y) = g Y. A generic
vector field xi(Y) = A0 + A1*Y + A2*Y^2 is built with random rational
entries. We take y = I_3 + perturbation and expand forward-backward
residual to h^6 using sympy exact rational arithmetic.

If Scheme A has MKW antisymmetric order q = 5, the residual must vanish
EXACTLY at h^1, h^2, ..., h^5 and be nonzero at h^6.

Because A0, A1, A2 and y are chosen with generic random rational entries,
accidental cancellation at any h^k is of measure zero. Passing this
algebraic test for multiple seeds constitutes exact algebraic verification
that the order of the forward-backward LB character residual on every
elementary differential (for every planar tree compatible with the
selected polynomial degree of xi) is at least 6.

xi degree 2 -> detects all planar trees whose internal nodes have at most
2 children. To extend to 3- and 4-child nodes (reaching weight-5 trees
with high branching), set XI_DEGREE = 3 or 4.
"""

from __future__ import annotations

import sympy as sp
from sympy import Matrix, Rational, Symbol, eye, zeros, factorial, simplify, Poly
import random


N = 3               # dimension of GL(N)
XI_DEGREE = 4       # polynomial degree of xi(Y) — covers xi^(0)..xi^(4),
                    #   i.e. every planar tree up to weight 5 (whose root has
                    #   at most 4 children).
MAX_H = 6           # expand up to h^MAX_H
h = Symbol("h")


BETA = [
    [Rational(1, 3), Rational(0), Rational(0)],
    [Rational(-7, 16), Rational(15, 16), Rational(0)],
    [Rational(49, 240), Rational(-7, 16), Rational(2, 5)],
]


def random_rational_matrix(rng: random.Random, n: int, denom_max: int = 7) -> Matrix:
    return Matrix(
        n, n,
        lambda i, j: Rational(rng.randint(-5, 5), rng.randint(1, denom_max)),
    )


def build_xi_coeffs(rng: random.Random, degree: int) -> list:
    return [random_rational_matrix(rng, N) for _ in range(degree + 1)]


def xi(Y: Matrix, coeffs: list) -> Matrix:
    """xi(Y) = sum_k coeffs[k] * Y^k"""
    result = coeffs[0].copy()
    Y_power = eye(N)
    for k in range(1, len(coeffs)):
        Y_power = Y_power * Y
        result = result + coeffs[k] * Y_power
    return result


def truncate_matrix_in_h(M: Matrix, max_h: int) -> Matrix:
    """For a matrix whose entries are polynomials in h, drop terms of
    degree > max_h."""
    out = zeros(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            expr = sp.expand(M[i, j])
            poly = sp.Poly(expr, h)
            new = sp.Integer(0)
            for (deg,), coef in poly.terms():
                if deg <= max_h:
                    new = new + coef * h**deg
            out[i, j] = new
    return out


def expm_series(V: Matrix, max_h: int) -> Matrix:
    """exp(V) = I + V + V^2/2 + ... + V^k/k!, truncated at h^max_h.

    Assumes V starts at h^1 (so V^{max_h + 1} = 0 after truncation)."""
    result = eye(N)
    term = eye(N)
    for k in range(1, max_h + 1):
        term = truncate_matrix_in_h(term * V, max_h)
        if term == zeros(N, N):
            break
        result = result + term / factorial(k)
    return truncate_matrix_in_h(result, max_h)


def exact_flow_taylor(Y0: Matrix, xi_coeffs: list, max_h: int) -> Matrix:
    """Taylor expansion of Y(h) solving dY/dh = xi(Y) Y with Y(0) = Y0,
    up to h^max_h, via the explicit recurrence
        (k+1) y_{k+1} = [h^k coefficient of F(Y_{<=k}(h))],   F(Y) = xi(Y) Y.
    Uses exact rational arithmetic."""
    Y = Matrix(Y0)  # h^0 term
    for k in range(max_h):
        F = xi(Y, xi_coeffs) * Y
        F = truncate_matrix_in_h(F, k)
        y_next = zeros(N, N)
        for i in range(N):
            for j in range(N):
                expr = sp.expand(F[i, j])
                coef_k = expr.coeff(h, k)
                y_next[i, j] = coef_k / (k + 1)
        Y = Y + h ** (k + 1) * y_next
        Y = truncate_matrix_in_h(Y, max_h)
    return Y


def scheme_a_step(Y0: Matrix, h_sign: int, xi_coeffs: list, max_h: int) -> Matrix:
    """One step of CF-EES(2,5; 1/10) Scheme A from state Y0 with step sign h_sign."""
    Y = Y0
    K_list = []
    for l in range(3):
        K_l = xi(Y, xi_coeffs)
        K_l = truncate_matrix_in_h(K_l, max_h)
        K_list.append(K_l)

        V_l = zeros(N, N)
        for i in range(l + 1):
            coef = Rational(h_sign) * BETA[l][i]
            if coef != 0:
                V_l = V_l + coef * K_list[i]
        V_l = h * V_l
        V_l = truncate_matrix_in_h(V_l, max_h)

        expV = expm_series(V_l, max_h)
        Y = truncate_matrix_in_h(expV * Y, max_h)
    return Y


def main():
    print("=" * 74)
    print(f"  Algebraic (exact-rational) verification of Scheme A antisym order")
    print(f"  Setup:  G = GL({N}, Q),   xi(Y) = sum_{{k=0..{XI_DEGREE}}} A_k Y^k")
    print(f"  Expanding forward-backward residual up to h^{MAX_H}.")
    print("=" * 74)
    print()

    def h_slice_zero(residual: Matrix, k: int) -> bool:
        for i in range(N):
            for j in range(N):
                if sp.expand(residual[i, j]).coeff(h, k) != 0:
                    return False
        return True

    def first_nonzero_h_degree(residual: Matrix, max_deg: int):
        for k in range(max_deg + 1):
            if not h_slice_zero(residual, k):
                return k
        return None

    for seed in range(3):
        print(f"--- seed = {seed} -------------------------------------------------")
        rng = random.Random(seed)
        xi_coeffs = build_xi_coeffs(rng, XI_DEGREE)

        y0 = eye(N) + Rational(1, 3) * random_rational_matrix(rng, N)

        print(f"  computing Scheme A forward step Phi_{{+h}}(y0) ...", flush=True)
        Y_fwd = scheme_a_step(y0, +1, xi_coeffs, MAX_H)

        # --- Classical forward-order check ---
        print(f"  computing exact flow Y(h) Taylor series up to h^{MAX_H} ...",
              flush=True)
        Y_exact = exact_flow_taylor(y0, xi_coeffs, MAX_H)

        fwd_err = truncate_matrix_in_h(Y_fwd - Y_exact, MAX_H)
        fwd_first = first_nonzero_h_degree(fwd_err, MAX_H)
        p = (fwd_first - 1) if fwd_first is not None else MAX_H
        print(f"  forward error Phi_{{+h}}(y0) - Y(h):")
        for k in range(MAX_H + 1):
            zero = h_slice_zero(fwd_err, k)
            verdict = "ZERO" if zero else "NONZERO"
            print(f"    h^{k}: {verdict}")
        print(f"  ==>  classical order p = {p}  (expected 2 for EES(2,5))")
        print()

        # --- Antisymmetric-order check ---
        print(f"  computing Scheme A backward step Phi_{{-h}}(Phi_{{+h}}(y0)) ...",
              flush=True)
        Y_round = scheme_a_step(Y_fwd, -1, xi_coeffs, MAX_H)

        rev_err = truncate_matrix_in_h(Y_round - y0, MAX_H)
        rev_first = first_nonzero_h_degree(rev_err, MAX_H)
        q = (rev_first - 1) if rev_first is not None else MAX_H
        print(f"  reversibility error Phi_{{-h}}(Phi_{{+h}}(y0)) - y0:")
        for k in range(MAX_H + 1):
            zero = h_slice_zero(rev_err, k)
            verdict = "ZERO (antisymmetric)" if zero else "NONZERO"
            print(f"    h^{k}: {verdict}")
        print(f"  ==>  antisymmetric order q = {q}  (expected 5 for EES(2,5))")
        print()


if __name__ == "__main__":
    main()
