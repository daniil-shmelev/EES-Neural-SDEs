"""Independent verification of MKW left-admissible-cut coproduct and the
antisymmetric-order defect for EES(2,5;x).

Self-contained: uses only sympy + stdlib (no kauri, no georax).

We check whether the MKW antisymmetric defect (phi_bar * phi)(tau) vanishes
on all planar trees of weight <= q. If it vanishes for all w in {1, ..., 5}
and first appears at w=6, then the MKW antisymmetric order of phi is >= 5.

If it appears already at w=3 (cherry), the order is 2.

We compare with the BCK (non-planar) antisymmetric defect too, for
sanity-check against Shmelev 2025's q=5 result on the cherry and larger
non-planar trees.

Usage:
    python mkw_verify.py
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterator, Optional

import sympy as sp


# ============================================================================
# Planar rooted tree data structure
# ============================================================================

@dataclass(frozen=True)
class PTree:
    """Planar rooted tree: root with ORDERED tuple of child subtrees.

    A leaf is PTree() with empty `children`. We DO NOT use this same object
    to represent the "empty tree" (unit of the coalgebra); that is a
    separate sentinel, Nil."""
    children: tuple["PTree", ...] = ()

    @property
    def weight(self) -> int:
        return 1 + sum(c.weight for c in self.children)

    def __repr__(self) -> str:
        if not self.children:
            return "•"
        return "[" + ",".join(repr(c) for c in self.children) + "]"


LEAF = PTree(())


@dataclass(frozen=True)
class PForest:
    """Ordered forest; may be empty."""
    trees: tuple[PTree, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.trees

    @property
    def weight(self) -> int:
        return sum(t.weight for t in self.trees)

    def __repr__(self) -> str:
        if self.is_empty:
            return "1"
        return " ".join(repr(t) for t in self.trees)


EMPTY_FOREST = PForest(())


# Sentinel for "the empty tree" (the counit-image under cut-the-whole-tree)
Nil = object()  # passes as the "empty tree" when evaluating phi


# ============================================================================
# Planar tree enumeration
# ============================================================================

@lru_cache(None)
def ordered_compositions(n: int) -> tuple[tuple[int, ...], ...]:
    if n == 0:
        return ((),)
    out: list[tuple[int, ...]] = []
    for k in range(1, n + 1):
        for tail in ordered_compositions(n - k):
            out.append((k,) + tail)
    return tuple(out)


@lru_cache(None)
def planar_trees_of_weight(w: int) -> tuple[PTree, ...]:
    if w <= 0:
        return ()
    if w == 1:
        return (LEAF,)
    out: list[PTree] = []
    for child_weights in ordered_compositions(w - 1):
        pools = [planar_trees_of_weight(cw) for cw in child_weights]
        for combo in itertools.product(*pools):
            out.append(PTree(tuple(combo)))
    return tuple(out)


# ============================================================================
# Left-admissible-cut (LAC) coproduct on a planar tree.
#
# Definition (CEFM20 §7.1 / LMK-2010 Table 1):
#   A cut of a planar tree τ is a subset of its edges. It is "admissible"
#   if any root-to-leaf path contains at most one cut edge. It is
#   "left-admissible" if additionally at every internal node, the cut
#   edges form a left PREFIX of the ordered children.
#
# The coproduct:
#   Δ(τ) = τ ⊗ 1  +  1 ⊗ τ  +  Σ_{non-trivial c} P^c(τ) ⊗ R^c(τ)
# where P^c is the forest of pruned subtrees (in planar order) and R^c is
# the remaining tree with the cut subtrees removed.
# ============================================================================

def shuffle_forests(a: PForest, b: PForest) -> list[tuple[int, PForest]]:
    """Shuffle product of two planar forests. Returns [(coeff, forest), ...]
    accumulated over all distinct resulting forests.

    Recursive definition:
        ∅ ⊔ ω = {ω}
        ω ⊔ ∅ = {ω}
        (a₁ ω₁) ⊔ (b₁ ω₂) = a₁ · (ω₁ ⊔ (b₁ ω₂)) + b₁ · ((a₁ ω₁) ⊔ ω₂)
    """
    if not a.trees:
        return [(1, b)]
    if not b.trees:
        return [(1, a)]
    accum: dict[PForest, int] = {}
    # First element from a: prepend a[0] to shuffle(a[1:], b)
    for c, rest in shuffle_forests(PForest(a.trees[1:]), b):
        new_forest = PForest((a.trees[0],) + rest.trees)
        accum[new_forest] = accum.get(new_forest, 0) + c
    # First element from b: prepend b[0] to shuffle(a, b[1:])
    for c, rest in shuffle_forests(a, PForest(b.trees[1:])):
        new_forest = PForest((b.trees[0],) + rest.trees)
        accum[new_forest] = accum.get(new_forest, 0) + c
    return [(c, f) for f, c in accum.items() if c != 0]


def shuffle_many(forests: list[PForest]) -> list[tuple[int, PForest]]:
    """Iterated shuffle of a list of planar forests.
    Returns [(coeff, forest), ...] for the shuffle product ⊔_i forests[i]."""
    if not forests:
        return [(1, EMPTY_FOREST)]
    current = [(1, forests[0])]
    for f in forests[1:]:
        new = {}
        for c1, acc in current:
            for c2, s in shuffle_forests(acc, f):
                new[s] = new.get(s, 0) + c1 * c2
        current = [(c, s) for s, c in new.items() if c != 0]
    return current


def lac_recursive_cuts(tree: PTree) -> Iterator[tuple[int, PForest, PTree]]:
    """Yield (coeff, P, R) triples for all LAC cuts of `tree` -- including the
    empty cut (coeff=1, P=∅, R=τ) but NOT the "full cut" τ ⊗ 1 (added at
    the top level by lac_coproduct_terms).

    At the root with m ordered children, pick a prefix cut count k ∈ {0, ..., m}.
    The first k children are entirely removed into the prefix part of P.
    Each of the remaining m-k children independently undergoes its own LAC
    cut, contributing its own (coeff_i, P_i, R_i). The total pruned forest P
    is the SHUFFLE of the prefix (as a forest) with all the P_i's (since
    pruned parts from different vertices shuffle together -- Lundervold
    thesis Thm 2.14, MKW 2008 Prop. 2). The total coeff is the product of
    child coeffs times the shuffle coefficient of the particular P forest.
    """
    m = len(tree.children)
    for k in range(m + 1):
        prefix_forest = PForest(tuple(tree.children[:k]))
        remaining = tree.children[k:]
        subchoices = [list(lac_recursive_cuts(c)) for c in remaining]
        for combo in itertools.product(*subchoices) if subchoices else [()]:
            combo_coeff = 1
            P_child_forests: list[PForest] = []
            R_trees: list[PTree] = []
            for (ci, Pi, Ri) in combo:
                combo_coeff *= ci
                P_child_forests.append(Pi)
                R_trees.append(Ri)
            # Shuffle prefix_forest with all child P forests
            all_forests = [prefix_forest] + P_child_forests
            shuffled = shuffle_many(all_forests)
            R_total = PTree(tuple(R_trees))
            for sc, sf in shuffled:
                yield (combo_coeff * sc, sf, R_total)


def lac_coproduct_terms(tree: PTree) -> list[tuple[int, PForest, object]]:
    """Return Δ(tree) as a list of (coeff, P, R) triples where R is either
    a PTree or Nil (for the "full cut" term τ ⊗ 1)."""
    terms: list[tuple[int, PForest, object]] = []
    # Full cut: τ ⊗ 1  (R = Nil = empty tree)
    terms.append((1, PForest((tree,)), Nil))
    # All LAC cuts including the empty cut (P=∅, R=τ)
    for coeff, P, R in lac_recursive_cuts(tree):
        terms.append((coeff, P, R))
    return terms


# ============================================================================
# Convolution of two characters against Δ
# ============================================================================

def eval_forest(phi, forest: PForest) -> sp.Expr:
    if forest.is_empty:
        return sp.Integer(1)
    v = sp.Integer(1)
    for t in forest.trees:
        v = v * phi(t)
    return v


def eval_tree_or_nil(phi, R) -> sp.Expr:
    """Evaluate phi on R: if R is Nil (empty tree), return 1 (counit image);
    otherwise call phi(R)."""
    if R is Nil:
        return sp.Integer(1)
    return phi(R)


def convolve(phi, psi, tree: PTree) -> sp.Expr:
    total = sp.Integer(0)
    for coeff, P, R in lac_coproduct_terms(tree):
        total += sp.Integer(coeff) * eval_forest(phi, P) * eval_tree_or_nil(psi, R)
    return total


# ============================================================================
# BCK (non-planar) coproduct for cross-check.
# On planar trees we use the SAME recursion but without any ordering
# restriction: at each node with m children, any SUBSET of children can be
# cut (not just a left prefix). This produces the standard BCK coproduct
# when we view planar trees as tree shapes with arbitrary child order.
# ============================================================================

def bck_recursive_cuts(tree: PTree) -> Iterator[tuple[PForest, PTree]]:
    m = len(tree.children)
    for subset in itertools.product([0, 1], repeat=m):
        P_trees: list[PTree] = []
        remaining_positions: list[int] = []
        for j, bit in enumerate(subset):
            if bit == 1:
                P_trees.append(tree.children[j])
            else:
                remaining_positions.append(j)
        # Recursively cut remaining children
        remaining = [tree.children[j] for j in remaining_positions]
        subchoices = [list(bck_recursive_cuts(c)) for c in remaining]
        for combo in itertools.product(*subchoices) if subchoices else [()]:
            P_all = list(P_trees)
            Rs: list[PTree] = []
            for (P_child, R_child) in combo:
                P_all.extend(P_child.trees)
                Rs.append(R_child)
            yield PForest(tuple(P_all)), PTree(tuple(Rs))


def bck_coproduct_terms(tree: PTree) -> list[tuple[PForest, object]]:
    terms: list[tuple[PForest, object]] = [(PForest((tree,)), Nil)]
    for P, R in bck_recursive_cuts(tree):
        terms.append((P, R))
    return terms


def convolve_bck(phi, psi, tree: PTree) -> sp.Expr:
    total = sp.Integer(0)
    for P, R in bck_coproduct_terms(tree):
        total += eval_forest(phi, P) * eval_tree_or_nil(psi, R)
    return total


# ============================================================================
# Classical RK elementary weights character on planar trees
# ============================================================================

def classical_rk_character(A: list[list[sp.Expr]], b: list[sp.Expr]):
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
# Main check
# ============================================================================

def main():
    x = sp.Symbol("x", real=True)

    # EES(2,5;x) tableau from Shmelev 2025
    a21 = (1 + 2 * x) / (4 * (1 - x))
    a31 = (4 * x - 1) ** 2 / (4 * (x - 1) * (1 - 4 * x ** 2))
    a32 = (1 - x) / (1 - 4 * x ** 2)
    A = [[sp.Integer(0)] * 3, [a21, 0, 0], [a31, a32, 0]]
    b = [x, sp.Rational(1, 2), sp.Rational(1, 2) - x]

    phi = classical_rk_character(A, b)

    def phi_bar(t: PTree) -> sp.Expr:
        return (-1) ** t.weight * phi(t)

    # Sanity: Δ(cherry)
    cherry = PTree((LEAF, LEAF))
    print(f"Cherry = {cherry!r}")
    print("LAC coproduct terms for cherry:")
    seen = {}
    for coeff, P, R in lac_coproduct_terms(cherry):
        key = (repr(P), "Nil" if R is Nil else repr(R))
        seen[key] = seen.get(key, 0) + coeff
    for k, c in seen.items():
        print(f"  coeff={c}  {k[0]}  ⊗  {k[1]}")
    print()

    # Check antisymmetric defect by weight
    print("=== MKW antisymmetric defect (phi_bar * phi)(tau) ===")
    print(f"{'w':>2} {'tree':<20} {'phi(tau)':<30} {'defect':<30}")
    print("-" * 90)
    mkw_order: Optional[int] = None
    for w in range(1, 7):
        all_zero_at_w = True
        for t in planar_trees_of_weight(w):
            phi_t = sp.simplify(phi(t))
            D = sp.simplify(convolve(phi_bar, phi, t))
            marker = " <-- NONZERO" if D != 0 else ""
            print(f"{w:>2} {repr(t):<20} {str(phi_t)[:28]:<30} {str(D)[:28]:<30}{marker}")
            if D != 0:
                all_zero_at_w = False
        if not all_zero_at_w and mkw_order is None:
            mkw_order = w - 1
        print()
    if mkw_order is None:
        mkw_order = 6
    print(f"==> MKW antisymmetric order of EES(2,5;x) is {mkw_order} (first non-zero at weight {mkw_order + 1 if mkw_order < 6 else '>6'})")

    # Cross-check: BCK antisymmetric defect on planar trees
    print()
    print("=== BCK antisymmetric defect (phi_bar *_BCK phi)(tau) ===")
    print("(should vanish up to weight 5 per Shmelev 2025)")
    print(f"{'w':>2} {'tree':<20} {'defect':<30}")
    print("-" * 65)
    bck_order: Optional[int] = None
    for w in range(1, 7):
        all_zero_at_w = True
        for t in planar_trees_of_weight(w):
            D = sp.simplify(convolve_bck(phi_bar, phi, t))
            marker = " <-- NONZERO" if D != 0 else ""
            print(f"{w:>2} {repr(t):<20} {str(D)[:28]:<30}{marker}")
            if D != 0:
                all_zero_at_w = False
        if not all_zero_at_w and bck_order is None:
            bck_order = w - 1
        print()
    if bck_order is None:
        bck_order = 6
    print(f"==> BCK antisymmetric order of EES(2,5;x) is {bck_order}")


if __name__ == "__main__":
    main()
