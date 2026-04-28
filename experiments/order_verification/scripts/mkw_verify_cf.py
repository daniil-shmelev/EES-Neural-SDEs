"""Independent MKW antisymmetric-order check for the CF-SCHEME LB character of
EES(2,5;x) (not the classical-RK character).

The CF scheme is
    Y_{n+1} = exp(V_3) exp(V_2) exp(V_1) . Y_n
with V_l = h * sum_i beta_{l,i} K_i and K_i = f(Y_{i-1}). Its LB character on
planar rooted forests is the NCK-convolution (concatenation-Hopf-algebra
convolution) of the three individual exponential characters phi_l, where
phi_l is the classical RK character of the tableau (A, beta_l).

Here:
    beta_1 = (B_1, 0, 0)
    beta_2 = (B_2 A_2, B_2, 0)
    beta_3 = (B_3 A_3 A_2, B_3 A_3, B_3)

The 2N-Williamson coefficients (A_i, B_i) are derived from the classical
EES(2,5;x) tableau via Bazavov eqs (25)-(26).

We then check the MKW antisymmetric order of this CF character by convolving
it with its grading-twist via the LAC coproduct (not NCK).
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Iterator, Optional

import sympy as sp

from mkw_verify import (  # reuse helpers
    LEAF, PTree, PForest, EMPTY_FOREST, Nil,
    planar_trees_of_weight,
    lac_coproduct_terms,
    convolve,
    eval_forest, eval_tree_or_nil,
)


# ============================================================================
# NCK (admissible-cut, not LAC) coproduct on planar trees.
# Foissy H_PR: admissible cut = no level-ordering constraint.
# ============================================================================

def nck_recursive_cuts(tree: PTree) -> Iterator[tuple[PForest, PTree]]:
    """Admissible-cut coproduct (planar): arbitrary SUBSET of children at
    each node can be cut, not just a left prefix."""
    m = len(tree.children)
    for subset in itertools.product([0, 1], repeat=m):
        cut_children: list[PTree] = []
        remaining_children: list[PTree] = []
        for j, bit in enumerate(subset):
            if bit == 1:
                cut_children.append(tree.children[j])
            else:
                remaining_children.append(tree.children[j])
        subchoices = [list(nck_recursive_cuts(c)) for c in remaining_children]
        for combo in itertools.product(*subchoices) if subchoices else [()]:
            P_all = list(cut_children)
            Rs: list[PTree] = []
            for (Pc, Rc) in combo:
                P_all.extend(Pc.trees)
                Rs.append(Rc)
            yield PForest(tuple(P_all)), PTree(tuple(Rs))


def nck_coproduct_terms(tree: PTree) -> list[tuple[PForest, object]]:
    terms: list[tuple[PForest, object]] = [(PForest((tree,)), Nil)]
    for P, R in nck_recursive_cuts(tree):
        terms.append((P, R))
    return terms


def convolve_nck(phi, psi, tree: PTree) -> sp.Expr:
    total = sp.Integer(0)
    for P, R in nck_coproduct_terms(tree):
        total += eval_forest(phi, P) * eval_tree_or_nil(psi, R)
    return total


# ============================================================================
# Classical RK character (same as mkw_verify.py)
# ============================================================================

def classical_rk_character(A, b):
    s = len(b)

    @lru_cache(None)
    def eval_stage(tree: PTree, i: int) -> sp.Expr:
        prod = sp.Integer(1)
        for c in tree.children:
            sub = sp.Integer(0)
            for j in range(s):
                sub += A[i][j] * eval_stage(c, j)
            prod *= sub
        return prod

    @lru_cache(None)
    def phi(tree: PTree) -> sp.Expr:
        total = sp.Integer(0)
        for i in range(s):
            prod = sp.Integer(1)
            for c in tree.children:
                sub = sp.Integer(0)
                for j in range(s):
                    sub += A[i][j] * eval_stage(c, j)
                prod *= sub
            total += b[i] * prod
        return total

    return phi


# ============================================================================
# Derive 2N Williamson (A_i, B_i) from classical EES(2,5;x) tableau
# ============================================================================

def williamson_2N_coeffs(x):
    """Explicit 2N Williamson coefficients for EES(2,5;x).

    Per main manuscript Proposition C.1 (the 2N tableau), for EES(2,5;x):
        A_2 = (coefficient)
        A_3 = (coefficient)
        B_1, B_2, B_3 = (coefficients)
    For x = 1/10, these are A_2=-7/15, A_3=-35/32, B_1=1/3, B_2=15/16,
    B_3=2/5 (from the manuscript).

    Derivation: given classical a21, a31, a32, b1, b2, b3, solve the
    recurrence
        a21 = B_1
        a31 = A_2 a_1,2 + B_1; but a_1,2 = 0, so a31 = B_1? No - need
        correct formula.
    Actually from the ODE-case unrolling:
        Y_1 = Y_0 + B_1 h K_1            => a_{21} = B_1
        Y_2 = Y_1 + B_2 h (A_2 K_1 + K_2) = Y_0 + (B_1 + B_2 A_2) h K_1 + B_2 h K_2
            => a_{31} = B_1 + B_2 A_2, a_{32} = B_2
        y_{n+1} = Y_2 + B_3 h (A_3 A_2 K_1 + A_3 K_2 + K_3)
            => b_1 = B_1 + B_2 A_2 + B_3 A_3 A_2
            => b_2 = B_2 + B_3 A_3
            => b_3 = B_3
    Given (a21, a31, a32, b1, b2, b3), solve:
        B_1 = a_{21}
        B_2 = a_{32}
        B_3 = b_3
        A_2 = (a_{31} - B_1) / B_2
        A_3 = (b_2 - B_2) / B_3
    Check consistency: b_1 should equal a_{31} + B_3 A_3 A_2.
    """
    a21 = (1 + 2 * x) / (4 * (1 - x))
    a31 = (4 * x - 1) ** 2 / (4 * (x - 1) * (1 - 4 * x ** 2))
    a32 = (1 - x) / (1 - 4 * x ** 2)
    b1 = x
    b2 = sp.Rational(1, 2)
    b3 = sp.Rational(1, 2) - x

    B1 = a21
    B2 = a32
    B3 = b3
    A2 = (a31 - B1) / B2
    A3 = (b2 - B2) / B3

    # Consistency check
    check_b1 = a31 + B3 * A3 * A2
    diff = sp.simplify(check_b1 - b1)
    assert diff == 0, f"Williamson 2N consistency failed: {diff}"

    return A2, A3, B1, B2, B3


def cf_lift_beta_vectors(x):
    """Return [beta_1, beta_2, beta_3] for the Bazavov 2N-CF lift of EES(2,5;x)."""
    A2, A3, B1, B2, B3 = williamson_2N_coeffs(x)
    beta_1 = [B1, sp.Integer(0), sp.Integer(0)]
    beta_2 = [B2 * A2, B2, sp.Integer(0)]
    beta_3 = [B3 * A3 * A2, B3 * A3, B3]
    return [beta_1, beta_2, beta_3]


# ============================================================================
# Build CF-scheme LB character via NCK convolution of individual exponential
# characters. This is what kauri.CFMethod.lb_character does.
# ============================================================================

def cf_scheme_character(A, betas):
    """Return a callable phi(tree) for the CF scheme with STAGE matrix A
    (shared across exponentials) and weight vectors betas. Uses NCK
    convolution of individual exponential characters.

    phi_l is the classical RK character of the tableau (A, beta_l); the CF
    scheme's character is phi_3 *_NCK phi_2 *_NCK phi_1 (applied right-to-left
    since the innermost exponential is applied first)."""
    phis = [classical_rk_character(A, b) for b in betas]

    @lru_cache(None)
    def phi_cf(tree: PTree) -> sp.Expr:
        # Start with phi_1 (innermost exponential)
        # result_prev = phi_1; iteratively convolve result_prev with phi_{l+1}
        # via NCK coproduct. The ordering convention: the CF scheme is
        # y_{n+1} = exp(V_J)...exp(V_1) y_n, so the LB character is
        #   phi_J *_NCK phi_{J-1} *_NCK ... *_NCK phi_1
        # In kauri's cf.py, `result = exp_maps[0]; for l>=1: result = nck_map_product(exp_maps[l], result)`
        # so it's nck(phi_l, prev) iteratively, which is equivalent to
        # phi_J *_NCK phi_{J-1} *_NCK ... *_NCK phi_1 by associativity.
        curr = phis[0]
        for l in range(1, len(phis)):
            prev_curr = curr
            next_phi = phis[l]
            # Define new curr = nck(next_phi, prev_curr)
            # But can't just redefine in loop because of closure capture.
            curr = _nck_conv_map(next_phi, prev_curr)
        return curr(tree)

    return phi_cf


def _nck_conv_map(phi, psi):
    """Return a callable that, given a tree, returns (phi *_NCK psi)(tree)."""
    @lru_cache(None)
    def out(tree: PTree) -> sp.Expr:
        total = sp.Integer(0)
        for P, R in nck_coproduct_terms(tree):
            total += eval_forest(phi, P) * eval_tree_or_nil(psi, R)
        return total
    return out


def _lac_conv_map(phi, psi):
    """Return a callable that, given a tree, returns (phi *_LAC psi)(tree).
    This is the MKW/LAC convolution; by LMK-2011 Thm 4.3.4 / Lundervold
    thesis Thm 2.15, this is the one that computes composition of LB
    series on a homogeneous space."""
    @lru_cache(None)
    def out(tree: PTree) -> sp.Expr:
        total = sp.Integer(0)
        for coeff, P, R in lac_coproduct_terms(tree):
            total += sp.Integer(coeff) * eval_forest(phi, P) * eval_tree_or_nil(psi, R)
        return total
    return out


def cf_scheme_character_lac(A, betas):
    """Same as cf_scheme_character but composes via LAC (MKW) coproduct.
    This is what Theorem 2.1 of the scaffold / Lundervold-Munthe-Kaas 4.3.4
    actually prescribes for LB-series composition."""
    phis = [classical_rk_character(A, b) for b in betas]

    @lru_cache(None)
    def phi_cf(tree: PTree) -> sp.Expr:
        curr = phis[0]
        for l in range(1, len(phis)):
            curr = _lac_conv_map(phis[l], curr)
        return curr(tree)

    return phi_cf


# ============================================================================
# Main
# ============================================================================

def main():
    x = sp.Symbol("x", real=True)

    # Classical RK tableau (stage matrix A shared across exponentials)
    a21 = (1 + 2 * x) / (4 * (1 - x))
    a31 = (4 * x - 1) ** 2 / (4 * (x - 1) * (1 - 4 * x ** 2))
    a32 = (1 - x) / (1 - 4 * x ** 2)
    A_stage = [
        [sp.Integer(0)] * 3,
        [a21, 0, 0],
        [a31, a32, 0],
    ]

    betas = cf_lift_beta_vectors(x)
    print("CF scheme betas for EES(2,5;x):")
    for l, b in enumerate(betas):
        print(f"  beta_{l+1} =", [sp.simplify(v) for v in b])

    phi_cf_nck = cf_scheme_character(A_stage, betas)
    phi_cf_lac = cf_scheme_character_lac(A_stage, betas)

    def run_antisym_check(phi_cf, label):
        def phi_cf_bar(t: PTree) -> sp.Expr:
            return (-1) ** t.weight * phi_cf(t)

        print()
        print("=" * 78)
        print(f"  CF character constructed via {label}")
        print("=" * 78)

        # MKW (LAC) antisym order of this character
        print()
        print(f"--- MKW antisym defect of {label}-composed character ---")
        print(f"{'w':>2} {'tree':<22} {'phi_CF(tau)':<30} {'MKW defect':<30}")
        print("-" * 90)
        mkw_order: Optional[int] = None
        for w in range(1, 6):
            all_zero_at_w = True
            for t in planar_trees_of_weight(w):
                phi_t = sp.simplify(phi_cf(t))
                D = sp.simplify(convolve(phi_cf_bar, phi_cf, t))
                flag = " <-- NONZERO" if D != 0 else ""
                print(
                    f"{w:>2} {repr(t):<22} {str(phi_t)[:28]:<30} {str(D)[:28]:<30}{flag}"
                )
                if D != 0:
                    all_zero_at_w = False
            if not all_zero_at_w and mkw_order is None:
                mkw_order = w - 1
            print()
        if mkw_order is None:
            mkw_order = 5
        print(f"==> MKW antisym order (character built via {label}): {mkw_order}")

        # NCK antisym order of this character
        print()
        print(f"--- NCK antisym defect of {label}-composed character ---")
        print(f"{'w':>2} {'tree':<22} {'NCK defect':<30}")
        print("-" * 60)
        nck_order: Optional[int] = None
        for w in range(1, 6):
            all_zero_at_w = True
            for t in planar_trees_of_weight(w):
                D = sp.simplify(convolve_nck(phi_cf_bar, phi_cf, t))
                flag = " <-- NONZERO" if D != 0 else ""
                print(f"{w:>2} {repr(t):<22} {str(D)[:28]:<30}{flag}")
                if D != 0:
                    all_zero_at_w = False
            if not all_zero_at_w and nck_order is None:
                nck_order = w - 1
            print()
        if nck_order is None:
            nck_order = 5
        print(f"==> NCK antisym order (character built via {label}): {nck_order}")
        return mkw_order, nck_order

    # LAC (MKW) composition is what Theorem 2.1 / LMK11 prescribes.
    # NCK composition is what kauri's CFMethod.lb_character does.
    mkw_lac, nck_lac = run_antisym_check(phi_cf_lac, "LAC (Thm 2.1)")
    mkw_nck, nck_nck = run_antisym_check(phi_cf_nck, "NCK (kauri)")

    print()
    print("=" * 78)
    print("  Summary")
    print("=" * 78)
    print(f"  Empirical reversibility rate (6H-1) suggests antisym order 5.")
    print(f"  Theorem-2.1 prescribes LAC composition; result:")
    print(f"    - MKW antisym of LAC-composed char: {mkw_lac}")
    print(f"    - NCK antisym of LAC-composed char: {nck_lac}")
    print(f"  kauri uses NCK composition; result:")
    print(f"    - MKW antisym of NCK-composed char: {mkw_nck}")
    print(f"    - NCK antisym of NCK-composed char: {nck_nck}")


if __name__ == "__main__":
    main()
