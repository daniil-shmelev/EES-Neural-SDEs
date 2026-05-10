"""Tree-algebraic verification of Scheme A's MKW antisymmetric order.

We work directly in the planar-forest Hopf algebra H_N = H_MKW, bypassing
any ambient-matrix representation. The LB character phi_A of Scheme A
(2N-register CF-EES(2,5; 1/10)) is extracted tree-by-tree by tracking
its 2N recurrence as a linear combination of planar forests with
rational coefficients. The MKW antisymmetric identity

    (bar phi_A *_N phi_A)(tau) = eps(tau)

is then verified symbolically for every planar tree tau of weight <= 5
via the LAC-coproduct convolution (the same *_N used in the scaffold's
Theorem 2.1).

This is purely an algebra-over-trees computation: no matrix groups, no
ambient vector fields, no numerical floats. Only planar trees, rationals,
and three operations (xi-evaluation, linear combination, exp(V)-action)
implemented on the free planar-forest algebra.

The three operations:

  1. xi(state):  Lie-algebra series whose coefficient on the tree
     B_+(omega) (root with omega as ordered children) equals the state's
     coefficient on the forest omega.  (Grafting rule for elementary
     differentials.)

  2. h * lin_combo of Lie-algebra series:  standard linear combination,
     with an h-degree bump.

  3. Lambda(exp(V), state) on a linear Lie group via left translation
     Lambda(g, y) = g y.  In the planar algebra this is the iterated
     left-concatenation:
         V^k = sum_{tau_1,...,tau_k} v_{tau_1} ... v_{tau_k} * (tau_1 tau_2 ... tau_k)
     and exp(V) . state = sum_k V^k / k! . state, where V^k . state is
     computed by prepending forests left-to-right.

At the end, we extract phi_A(tau) for every planar tree tau of weight
up to 6, and check MKW antisymmetry via the LAC coproduct of mkw_verify.py.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Dict, Tuple

try:
    from .mkw_verify import (  # type: ignore[import-not-found]
        LEAF, PTree, PForest, EMPTY_FOREST, Nil,
        planar_trees_of_weight,
        lac_coproduct_terms,
        eval_forest, eval_tree_or_nil,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from mkw_verify import (
        LEAF, PTree, PForest, EMPTY_FOREST, Nil,
        planar_trees_of_weight,
        lac_coproduct_terms,
        eval_forest, eval_tree_or_nil,
    )


# ============================================================================
# Types
#
# State character: dict { (h_degree, forest) : rational_coefficient }
#   -- represents a formal point on the manifold, expanded in h.
#      Implicit rule: the coefficient of (0, EMPTY_FOREST) is "the y-term".
#
# Lie character:   dict { (h_degree, tree)   : rational_coefficient }
#   -- represents a Lie-algebra series (a vector field in LB language).
# ============================================================================

StateChar = Dict[Tuple[int, PForest], Fraction]
LieChar   = Dict[Tuple[int, PTree],   Fraction]


def state_zero() -> StateChar:
    return {(0, EMPTY_FOREST): Fraction(1)}  # "Y_0 = y", the initial state


def prune_zero(d):
    return {k: v for k, v in d.items() if v != 0}


# ============================================================================
# Operation 1:  xi applied to a state
# ============================================================================

def xi_of_state(state: StateChar) -> LieChar:
    """The LB character of xi(state).  For each (h, omega) term in the
    state with coefficient c, the Lie-algebra series contains the tree
    B_+(omega) (root with children = omega.trees) with the same h-degree
    and coefficient c."""
    out: LieChar = {}
    for (h_deg, forest), c in state.items():
        tree = PTree(forest.trees)          # B_+ : forest -> tree
        key = (h_deg, tree)
        out[key] = out.get(key, Fraction(0)) + c
    return prune_zero(out)


# ============================================================================
# Operation 2:  scale by h and combine
# ============================================================================

def lin_combo_lie(scalars, lie_chars) -> LieChar:
    out: LieChar = {}
    for s, lc in zip(scalars, lie_chars):
        if s == 0:
            continue
        for k, c in lc.items():
            out[k] = out.get(k, Fraction(0)) + Fraction(s) * c
    return prune_zero(out)


def scale_h_lie(lc: LieChar, power: int) -> LieChar:
    return {(h + power, t): c for (h, t), c in lc.items()}


# ============================================================================
# Operation 3:  Lambda(exp(V), state) for linear Lie group left translation
#
# V^k acts on a state by left-concatenating k trees (coming from V) in
# front of the state's forest, then sum over k and divide by k!.
# ============================================================================

def action_V_on_state(V: LieChar, state: StateChar, max_h: int) -> StateChar:
    """Single left-action: V . state.
    For each (h_v, tau) in V and (h_s, omega) in state, contribute
    (h_v + h_s, (tau,) + omega.trees) with coefficient product."""
    out: StateChar = {}
    for (h_v, tree_v), c_v in V.items():
        for (h_s, forest_s), c_s in state.items():
            new_h = h_v + h_s
            if new_h > max_h:
                continue
            new_forest = PForest((tree_v,) + forest_s.trees)
            key = (new_h, new_forest)
            out[key] = out.get(key, Fraction(0)) + c_v * c_s
    return prune_zero(out)


def exp_action_on_state(V: LieChar, state: StateChar, max_h: int) -> StateChar:
    """exp(V) . state  = sum_{k>=0} V^k . state / k!, truncated at h^max_h."""
    out: StateChar = dict(state)         # k = 0 term
    Wk: StateChar = dict(state)          # W_0 = state; W_k = V^k . state
    for k in range(1, max_h + 1):
        Wk = action_V_on_state(V, Wk, max_h)
        if not Wk:
            break
        inv_k_fact = Fraction(1, factorial(k))
        for key, c in Wk.items():
            out[key] = out.get(key, Fraction(0)) + c * inv_k_fact
    return prune_zero(out)


# ============================================================================
# CF-EES(2,5; 1/10) beta weights used in the paper
# ============================================================================

BETA = [
    [Fraction(1, 3),    Fraction(0),      Fraction(0)],
    [Fraction(-7, 16),  Fraction(15, 16), Fraction(0)],
    [Fraction(49, 240), Fraction(-7, 16), Fraction(2, 5)],
]


# ============================================================================
# Scheme A recurrence as a composition of tree-algebra operations
# ============================================================================

def scheme_a_character(max_h: int, h_sign: int = 1) -> StateChar:
    """Run Scheme A's 2N recurrence in the planar-forest Hopf algebra and
    return the resulting state character  phi_A  (as a dict of (h_degree,
    planar-forest) -> rational).

    h_sign = +1 for forward step; h_sign = -1 for backward step.
    """
    state = state_zero()
    K_list = []  # list of Lie characters K_1, K_2, K_3
    for l in range(3):
        K_l = xi_of_state(state)
        K_list.append(K_l)
        # V_l = h * sum_i BETA[l][i] * K_i  (with sign)
        V_l: LieChar = lin_combo_lie(
            [Fraction(h_sign) * BETA[l][i] for i in range(l + 1)],
            K_list[: l + 1],
        )
        V_l = scale_h_lie(V_l, 1)
        # Truncate V_l in h
        V_l = {k: v for k, v in V_l.items() if k[0] <= max_h}
        # Apply exp(V_l) to state
        state = exp_action_on_state(V_l, state, max_h)
    return state


# ============================================================================
# Extract phi_A(tau) for planar trees, verify MKW antisymmetry
# ============================================================================

def extract_phi_tree(state: StateChar, w_max: int) -> Dict[PTree, Fraction]:
    """Extract phi on single-tree forests (for inspection / classical-order check)."""
    phi: Dict[PTree, Fraction] = {}
    for w in range(1, w_max + 1):
        for tau in planar_trees_of_weight(w):
            key = (w, PForest((tau,)))
            phi[tau] = state.get(key, Fraction(0))
    return phi


def phi_on_forest(state: StateChar, forest: PForest) -> Fraction:
    """Read off the character's value on a planar forest directly from the
    state character, at h-degree equal to the forest's weight.

    This respects the non-multiplicative (on concatenation) nature of the
    character on H_N, since the character is only a morphism for the
    shuffle product -- on a concatenated forest tau_1 tau_2 the value is
    a genuine forest coefficient that is *not* phi(tau_1) * phi(tau_2)."""
    return state.get((forest.weight, forest), Fraction(0))


def phi_on_tree_or_nil(state: StateChar, R) -> Fraction:
    """Character on a single tree (or Nil = empty tree / counit image)."""
    if R is Nil:
        return Fraction(1)
    key = (R.weight, PForest((R,)))
    return state.get(key, Fraction(0))


def phi_on_forest_treeprod(state: StateChar, forest: PForest) -> Fraction:
    """Tree-product extension (shuffle-multiplicative):
        phi(tau_1 tau_2 ... tau_k) := prod_i phi(tau_i).
    This is the convention used by mkw_verify.py's eval_forest.
    """
    v = Fraction(1)
    for t in forest.trees:
        key = (t.weight, PForest((t,)))
        v *= state.get(key, Fraction(0))
    return v


def convolve_lac(state: StateChar, tree: PTree, convention: str = "forest") -> Fraction:
    """(bar phi *_N phi)(tree) with the LAC coproduct.

    convention = "forest":   read phi(omega) directly off the state at
                             (|omega|, omega) -- true concatenation-basis value.
    convention = "tree_prod": use phi(tau_1 tau_2 ...) = prod_i phi(tau_i) --
                             the shuffle-multiplicative extension used by
                             mkw_verify.py.
    """
    if convention == "forest":
        def phi_on(P): return phi_on_forest(state, P)
    elif convention == "tree_prod":
        def phi_on(P): return phi_on_forest_treeprod(state, P)
    else:
        raise ValueError(f"unknown convention: {convention}")

    total = Fraction(0)
    for coeff, P, R in lac_coproduct_terms(tree):
        barphi_P = Fraction((-1) ** P.weight) * phi_on(P)
        phi_R = phi_on_tree_or_nil(state, R)
        total += Fraction(coeff) * barphi_P * phi_R
    return total


# ============================================================================
# Main
# ============================================================================

def main():
    MAX_H = 6

    print("=" * 76)
    print("  Tree-algebraic verification of Scheme A MKW antisymmetric order")
    print("  (planar-forest Hopf algebra, exact rational arithmetic, no matrices)")
    print("=" * 76)
    print()

    print(f"  Building phi_A = LB character of Scheme A on H_N (up to h^{MAX_H}) ...",
          flush=True)
    state = scheme_a_character(MAX_H, h_sign=+1)
    phi_tree = extract_phi_tree(state, MAX_H)
    print(f"  Extracted phi_A on {len(phi_tree)} planar trees (weight 1..{MAX_H}).")
    print()

    # ------------------------------------------------------------------
    # Shuffle-consistency check: for two trees tau1 != tau2,
    #   phi(tau1 tau2) + phi(tau2 tau1) ?= phi(tau1) * phi(tau2)
    # ------------------------------------------------------------------
    print("  Shuffle consistency check (phi is shuffle-multiplicative):")
    from itertools import product as iprod
    test_pairs = [
        (LEAF, LEAF),
        (LEAF, PTree((LEAF,))),
        (PTree((LEAF,)), PTree((LEAF,))),
        (LEAF, PTree((LEAF, LEAF))),
        (PTree((LEAF,)), PTree((LEAF, LEAF))),
    ]
    for t1, t2 in test_pairs:
        w = t1.weight + t2.weight
        if w > MAX_H: continue
        f12 = PForest((t1, t2))
        f21 = PForest((t2, t1))
        v12 = state.get((w, f12), Fraction(0))
        v21 = state.get((w, f21), Fraction(0))
        sum_shuffle = v12 + v21
        product_tree = phi_tree.get(t1, Fraction(0)) * phi_tree.get(t2, Fraction(0))
        # For tau1==tau2, shuffle gives 2*concat so single value = phi(tau)^2 / 2
        if t1 == t2:
            expected = product_tree  # phi(t) * phi(t) = phi(t shuffle t) = 2 * phi(tt) so phi(tt) = prod/2
            actual = v12  # single concatenation
            check = "OK" if actual == product_tree / 2 else f"MISMATCH (should be {product_tree}/2 = {product_tree/2})"
            print(f"    tau={t1!r}: phi(tau tau) = {actual}, phi(tau)^2/2 = {product_tree/2}  [{check}]")
        else:
            check = "OK" if sum_shuffle == product_tree else f"MISMATCH (expected {product_tree})"
            print(f"    tau1={t1!r}, tau2={t2!r}: phi(t1 t2) + phi(t2 t1) = {v12} + {v21} = {sum_shuffle}, phi(t1)*phi(t2) = {product_tree}  [{check}]")
    print()

    # Show phi_A values on low-weight trees
    print("  Values of phi_A on selected planar trees and forests:")
    selected_trees = [
        (LEAF, "leaf"),
        (PTree((LEAF,)), "[leaf]"),
        (PTree((LEAF, LEAF)), "[leaf,leaf]"),
        (PTree((PTree((LEAF,)),)), "[[leaf]]"),
    ]
    for tau, sym in selected_trees:
        print(f"    phi_A({sym:>8}) = {phi_tree.get(tau, Fraction(0))}")
    selected_forests = [
        (PForest((LEAF, LEAF)), "leaf leaf   (forest of two leaves)"),
        (PForest((LEAF, PTree((LEAF,)))), "leaf [leaf]   (forest)"),
    ]
    for omega, sym in selected_forests:
        print(f"    phi_A({sym:<30}) = {phi_on_forest(state, omega)}")
    print()

    # Classical-order check against exact-flow character on planar trees:
    #   phi_exp(leaf) = 1, phi_exp([leaf]) = 1/2.
    print("  Classical order check (phi_A vs exact-flow character):")
    expected = {LEAF: Fraction(1), PTree((LEAF,)): Fraction(1, 2)}
    for tau, sym in [(LEAF, "leaf"), (PTree((LEAF,)), "[leaf]")]:
        got = phi_tree.get(tau, Fraction(0))
        exp_ = expected[tau]
        match = "OK" if got == exp_ else f"MISMATCH (expected {exp_})"
        print(f"    tree {sym:>6}: phi_A = {got}, exact = {exp_}  [{match}]")
    print()

    # Run the MKW antisym check under BOTH conventions for comparison.
    for conv in ("forest", "tree_prod"):
        print(f"  --- Convention: {conv} ---")
        if conv == "forest":
            print("  phi(tau_1 tau_2 ...) = state's coefficient on the concatenated forest (true value)")
        else:
            print("  phi(tau_1 tau_2 ...) = prod_i phi(tau_i) (shuffle-multiplicative; mkw_verify convention)")
        print(f"  {'w':>2}  {'tree':<22} {'(bar phi *_N phi)(tau)':<30} {'verdict':<30}")
        print("-" * 90)
        mkw_order = None
        for w in range(1, MAX_H + 1):
            all_zero = True
            for tau in planar_trees_of_weight(w):
                defect = convolve_lac(state, tau, convention=conv)
                verdict = "ZERO (antisymmetric)" if defect == 0 else "NONZERO"
                if w <= 5:  # show detail for low weights
                    print(f"  {w:>2}  {repr(tau):<22} {str(defect)[:28]:<30} {verdict:<30}")
                if defect != 0:
                    all_zero = False
            if not all_zero and mkw_order is None:
                mkw_order = w - 1
            if w <= 5:
                print()
        if mkw_order is None:
            mkw_order = MAX_H
        print(f"  ==>  Scheme A MKW antisym order (tree-algebraic, {conv}): {mkw_order}")
        print()


if __name__ == "__main__":
    main()
