"""Exact Lie-Butcher verification of the cosine CF-EES schemes with kauri.

For the s-stage cosine scheme, s = 3, 4, 5, this checks in exact arithmetic:

* forward order two: the Lie-Butcher character agrees with the exact flow
  on every ordered forest of degree at most two;
* antisymmetric order exactly 2s-1: the MKW symmetry defect equals the
  counit on every ordered forest of degree at most 2s-1, and not at 2s;
* for s = 3, the planar-tree coefficients printed in the manuscript table.

`kauri.ReusedStageCFMethod` (https://github.com/daniil-shmelev/kauri)
implements Owren's pseudo-stage recursion and the MKW composition. SymPy
only supplies the exact number fields Q, Q(sqrt 2) and Q(sqrt 5).

    python verify_cfees.py [--stages 3 4 5]
"""

from __future__ import annotations

import argparse
from functools import cache
import time

import kauri
import sympy as sp

# Square root adjoined to Q for the s-stage cosine coefficients.
FIELD_ROOT = {3: None, 4: 2, 5: 5}

# Character of the three-stage scheme on planar trees of order 0..5, in the
# order of `kauri.planar_trees_up_to_order` (the order of the printed table).
TREE_TABLE = (
    "1 1 1/2 3/16 1/8 5/96 3/32 1/32 1/32 0 "
    "3/256 7/192 1/48 3/128 0 1/192 1/32 1/128 1/192 0 0 0 0 0"
).split()


def cosine_scheme(s):
    """Field and Williamson coefficients (A_2..A_s), (B_1..B_s)."""
    root = FIELD_ROOT[s]
    field = sp.QQ if root is None else sp.QQ.algebraic_field(sp.sqrt(root))
    cos = [
        field.from_sympy(sp.radsimp(sp.cos(sp.pi * j / s).rewrite(sp.sqrt)))
        for j in range(s + 1)
    ]
    d = [cos[j] - cos[j + 1] for j in range(s)]
    a = [-d[i - 1] / d[i] for i in range(1, s)]
    b = d[:-1] + [d[-1] / 2]
    return field, a, b


@cache
def ordered_forests(degree):
    """All ordered forests of planar trees with `degree` vertices."""
    if degree == 0:
        return (kauri.OrderedForest(()),)
    return tuple(
        kauri.OrderedForest(forest.tree_list + (tree,))
        for last in range(1, degree + 1)
        for forest in ordered_forests(degree - last)
        for tree in kauri.planar_trees_of_order(last)
    )


def verify(s):
    start = time.monotonic()
    field, a, b = cosine_scheme(s)
    method = kauri.ReusedStageCFMethod(a, b, name=f"CF-EES(2,{2 * s - 1})")
    character = method.lb_character()
    defect = method.symmetry_defect_map()
    print(f"{method.name}: {s}-stage cosine scheme over {field}")

    leaf, chain = kauri.PlanarTree([]), kauri.PlanarTree([[]])
    half = field.one / 2
    for trees, exact in [((leaf,), field.one), ((chain,), half), ((leaf, leaf), half)]:
        assert character(kauri.OrderedForest(trees)) == exact, trees
    print("  forward order two: 3 forests, exact-flow coefficients matched")

    order = 2 * s - 1
    for degree in range(1, order + 1):
        forests = ordered_forests(degree)
        residuals = sum(defect(forest) != field.zero for forest in forests)
        print(f"  degree {degree}: {len(forests)} forests, {residuals} nonzero residuals")
        assert not residuals
    assert any(defect(forest) != field.zero for forest in ordered_forests(order + 1))
    print(f"  antisymmetric order exactly {order} ({time.monotonic() - start:.1f}s)")

    if s == 3:
        trees = list(kauri.planar_trees_up_to_order(5))
        assert len(trees) == len(TREE_TABLE)
        for tree, printed in zip(trees, TREE_TABLE, strict=True):
            assert field.to_sympy(character(tree)) == sp.Rational(printed), tree
        print(f"  all {len(trees)} printed tree coefficients matched")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", type=int, nargs="+", default=sorted(FIELD_ROOT),
                        choices=sorted(FIELD_ROOT))
    for s in parser.parse_args().stages:
        verify(s)
    print("All CF-EES order checks passed.")


if __name__ == "__main__":
    main()
