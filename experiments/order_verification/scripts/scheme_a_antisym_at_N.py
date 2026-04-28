"""Quick diagnostic: check Scheme A's matrix-level forward-backward residual
at various matrix-Lie-group dimensions N.

Goal: see whether the 'order 5' seen at N=3 persists at N=5 (no hidden
linear dependence in the tangent space), or collapses to a lower order
(= the tree-algebraic MKW antisym order).
"""
from __future__ import annotations
import sys
import sympy as sp
from sympy import Matrix, Rational, Symbol, eye, zeros, factorial
import random

BETA = [
    [Rational(1, 3), Rational(0), Rational(0)],
    [Rational(-7, 16), Rational(15, 16), Rational(0)],
    [Rational(49, 240), Rational(-7, 16), Rational(2, 5)],
]

XI_DEGREE = 4
MAX_H = 6
h = Symbol("h")


def random_rational_matrix(rng, n, denom_max=7):
    return Matrix(n, n, lambda i, j: Rational(rng.randint(-5, 5), rng.randint(1, denom_max)))


def xi(Y, coeffs):
    result = coeffs[0].copy()
    Yp = eye(Y.rows)
    for k in range(1, len(coeffs)):
        Yp = Yp * Y
        result = result + coeffs[k] * Yp
    return result


def truncate_h_mat(M, max_h):
    N = M.rows
    out = zeros(N, N)
    for i in range(N):
        for j in range(N):
            expr = sp.expand(M[i, j])
            poly = sp.Poly(expr, h)
            new = sp.Integer(0)
            for (deg,), coef in poly.terms():
                if deg <= max_h:
                    new = new + coef * h ** deg
            out[i, j] = new
    return out


def expm_series(V, max_h):
    N = V.rows
    result = eye(N)
    term = eye(N)
    for k in range(1, max_h + 1):
        term = truncate_h_mat(term * V, max_h)
        if term == zeros(N, N):
            break
        result = result + term / factorial(k)
    return truncate_h_mat(result, max_h)


def scheme_a_step(Y0, h_sign, xi_coeffs, max_h):
    N = Y0.rows
    Y = Y0
    K_list = []
    for l in range(3):
        K_l = truncate_h_mat(xi(Y, xi_coeffs), max_h)
        K_list.append(K_l)
        V_l = zeros(N, N)
        for i in range(l + 1):
            coef = Rational(h_sign) * BETA[l][i]
            if coef != 0:
                V_l = V_l + coef * K_list[i]
        V_l = h * V_l
        V_l = truncate_h_mat(V_l, max_h)
        expV = expm_series(V_l, max_h)
        Y = truncate_h_mat(expV * Y, max_h)
    return Y


def h_slice_zero(residual, N, k):
    for i in range(N):
        for j in range(N):
            if sp.expand(residual[i, j]).coeff(h, k) != 0:
                return False
    return True


def main():
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"=== Scheme A antisym on GL({N}, Q), xi degree {XI_DEGREE} ===", flush=True)
    seed = 0
    rng = random.Random(seed)
    xi_coeffs = [random_rational_matrix(rng, N) for _ in range(XI_DEGREE + 1)]
    y0 = eye(N) + Rational(1, 3) * random_rational_matrix(rng, N)

    print(f"forward step ...", flush=True)
    Yf = scheme_a_step(y0, +1, xi_coeffs, MAX_H)
    print(f"backward step ...", flush=True)
    Yb = scheme_a_step(Yf, -1, xi_coeffs, MAX_H)
    residual = truncate_h_mat(Yb - y0, MAX_H)

    first_nonzero = None
    for k in range(MAX_H + 1):
        is_zero = h_slice_zero(residual, N, k)
        print(f"  h^{k}: {'ZERO' if is_zero else 'NONZERO'}", flush=True)
        if not is_zero and first_nonzero is None:
            first_nonzero = k

    q = (first_nonzero - 1) if first_nonzero is not None else MAX_H
    print(f"==> Scheme A antisym order (N={N}): q = {q}", flush=True)


if __name__ == "__main__":
    main()
