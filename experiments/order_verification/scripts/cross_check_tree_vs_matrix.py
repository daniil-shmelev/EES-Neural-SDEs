"""Cross-check: does the tree-algebra defect at weight 4 actually
produce a nonzero matrix residual at h^4?

If tree-level D([•,[•]]) = -1/8 and D([[•],•]) = -1/8, and the
elementary differentials evaluate to nonzero matrices at y_0, then
the matrix residual at h^4 should be (-1/4) * ED, nonzero.

We compute both sides explicitly on a concrete random ξ to see which
is right.
"""
import sympy as sp
from sympy import Matrix, Rational, Symbol, eye, zeros, factorial
import random

from scheme_a_algebraic_verify import (
    scheme_a_step, xi, random_rational_matrix, BETA, N as _ignore,
    truncate_matrix_in_h, h, expm_series
)

# Override N to 3 for speed
N = 3
MAX_H = 6


def xi_local(Y, coeffs):
    result = coeffs[0].copy()
    Yp = eye(Y.rows)
    for k in range(1, len(coeffs)):
        Yp = Yp * Y
        result = result + coeffs[k] * Yp
    return result


def fundamental_F(Y, coeffs):
    """F(Y) = xi(Y) * Y (fundamental VF for left translation)."""
    return xi_local(Y, coeffs) * Y


def elem_diff_bracket_bracket(y0, coeffs):
    """Compute EDiff([•,[•]])(y0) = F''(y)[F(y), F'(y)[F(y)]]
    for F(Y) = xi(Y) Y with xi a polynomial.

    Since xi is a polynomial in Y, F is a polynomial; we Taylor-expand
    F(y + e1*W1 + e2*W2) in e1, e2 and read off the e1*e2 coefficient."""
    e1, e2 = sp.symbols("e1 e2")
    W1 = fundamental_F(y0, coeffs)
    # W2 = EDiff([•])(y0) = F'(y)[W1]
    # Compute F'(y)[W1] by taking directional derivative of F in direction W1
    # F(y + e*W1) = poly(e); extract e^1 coefficient.
    Y_pert = y0 + e1 * W1
    F_pert = fundamental_F(Y_pert, coeffs)
    # Extract e1 coefficient
    W2 = zeros(N, N)
    for i in range(N):
        for j in range(N):
            coef = sp.expand(F_pert[i, j]).coeff(e1, 1)
            W2[i, j] = coef

    # Now compute F''(y)[W1, W2] = mixed e1*e2 coefficient of F(y + e1*W1 + e2*W2)
    Y_pert2 = y0 + e1 * W1 + e2 * W2
    F_pert2 = fundamental_F(Y_pert2, coeffs)
    out = zeros(N, N)
    for i in range(N):
        for j in range(N):
            coef = sp.expand(F_pert2[i, j]).coeff(e1, 1).coeff(e2, 1)
            out[i, j] = coef
    return out


def main():
    rng = random.Random(0)
    xi_coeffs = [random_rational_matrix(rng, N) for _ in range(5)]
    y0 = eye(N) + Rational(1, 3) * random_rational_matrix(rng, N)

    # Compute EDiff([•,[•]]) = F''(y)[F(y), F'(y)[F(y)]]
    ED_asym = elem_diff_bracket_bracket(y0, xi_coeffs)
    print("EDiff([•,[•]])(y0) = ")
    for i in range(N):
        print(f"  {[ED_asym[i, j] for j in range(N)]}")

    # By symmetry of F'', EDiff([[•],•])(y0) should equal EDiff([•,[•]])(y0)
    print("\nIf scheme A's MKW defect is -1/8 at both trees, matrix contribution at h^4:")
    tree_contrib = Rational(-1, 8) * ED_asym + Rational(-1, 8) * ED_asym
    print("  -1/4 * EDiff =")
    for i in range(N):
        print(f"  {[tree_contrib[i, j] for j in range(N)]}")

    # Now compute the ACTUAL matrix residual at h^4 from the real forward-backward run
    print("\nComputing actual matrix residual at h^4 from scheme_a_step ...", flush=True)
    Yf = scheme_a_step(y0, +1, xi_coeffs, MAX_H)
    Yb = scheme_a_step(Yf, -1, xi_coeffs, MAX_H)
    residual = truncate_matrix_in_h(Yb - y0, MAX_H)
    print("Matrix residual at h^4:")
    for i in range(N):
        row = []
        for j in range(N):
            coef_h4 = sp.expand(residual[i, j]).coeff(h, 4)
            row.append(coef_h4)
        print(f"  {row}")


if __name__ == "__main__":
    main()
