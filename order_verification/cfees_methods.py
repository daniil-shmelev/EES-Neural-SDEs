# Copyright 2026 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Symbolic CFEES method data used by the Hopf-algebra verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import kauri
import sympy as sp


X = sp.Symbol("x")


@dataclass(frozen=True)
class CFEESMethodSpec:
    """A symbolic reused-stage CFEES method specification."""

    name: str
    a: tuple[sp.Expr, ...]
    b: tuple[sp.Expr, ...]
    order: int
    antisymmetric_order: int
    domain: str


def cfees25(x: sp.Expr = X) -> CFEESMethodSpec:
    """Symbolic Williamson 2N form of CFEES(2,5;x)."""

    a = (
        (4 * x**2 - 2 * x + 1) / (2 * (x - 1)),
        -(4 * x**2 - 2 * x + 1) / ((2 * x - 1) ** 2 * (2 * x + 1)),
    )
    b = (
        (2 * x + 1) / (4 * (1 - x)),
        (1 - x) / (1 - 4 * x**2),
        (1 - 2 * x) / 2,
    )
    return CFEESMethodSpec(
        name="CFEES(2,5;x)",
        a=tuple(sp.cancel(v) for v in a),
        b=tuple(sp.cancel(v) for v in b),
        order=2,
        antisymmetric_order=5,
        domain="x not in {1, -1/2, 1/2}",
    )


def ees27_tableau(x: sp.Expr = X, *, branch: int = +1) -> tuple[list[list[sp.Expr]], list[sp.Expr]]:
    """Symbolic EES(2,7;x) Butcher tableau for one sqrt(2) branch."""

    if branch not in (+1, -1):
        raise ValueError("branch must be +1 or -1")

    sqrt2 = branch * sp.sqrt(2)

    b1 = x
    b2 = sp.Rational(1, 2) * (2 - sqrt2) - (1 - sqrt2) * x
    b3 = (1 - sqrt2) * (x - 1)
    b4 = sp.Rational(1, 2) * (2 - sqrt2) - x

    alpha = (-2 * x + sqrt2 + 1) * (2 * x + sqrt2) / (
        (2 * x - 1) * (4 * x**2 - 4 * x - 1)
    )
    beta = (1 + sqrt2 - 2 * x) * (2 + sqrt2 - 2 * x) / (
        (2 * x - 1)
        * (2 * x**2 - 4 * x + 1)
        * (4 * x**2 - 4 * x - 1)
    )

    a21 = (-2 + sqrt2 * (1 - 2 * x)) / (4 * (x - 1))
    a31 = (2 * x + sqrt2 - 2) * (4 * x + sqrt2 - 2) * alpha / (
        4 * sqrt2 * (x - 1)
    )
    a32 = sp.Rational(1, 2) * (-1 + sqrt2) * alpha
    a41 = beta * (2 * x - sqrt2) * (
        -40 * x**4
        + (80 - 40 * sqrt2) * x**3
        - (88 - 60 * sqrt2) * x**2
        + (48 - 34 * sqrt2) * x
        + 7 * sqrt2
        - 10
    ) / (8 * (x - 1) * (2 * x**2 - 1))
    a42 = (
        sp.Rational(1, 2)
        * (2 - sqrt2)
        * x
        * (x - 1)
        * (4 * x + sqrt2 - 2)
        * beta
    )
    a43 = (2 - sqrt2) * (2 * x - sqrt2) * (2 + sqrt2 - 2 * x) * (
        x - 1
    ) * (2 * x - 1) / (
        4 * (2 * x**2 - 1) * (2 * x**2 - 4 * x + 1)
    )

    a = [
        [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0)],
        [a21, sp.Integer(0), sp.Integer(0), sp.Integer(0)],
        [a31, a32, sp.Integer(0), sp.Integer(0)],
        [a41, a42, a43, sp.Integer(0)],
    ]
    b = [b1, b2, b3, b4]
    return a, b


def cfees27(x: sp.Expr = X, *, branch: int = +1) -> CFEESMethodSpec:
    """Symbolic Williamson 2N form of CFEES(2,7;x)."""

    a_rk, b_rk = ees27_tableau(x, branch=branch)
    a21 = a_rk[1][0]
    a31, a32 = a_rk[2][0], a_rk[2][1]
    a42, a43 = a_rk[3][1], a_rk[3][2]
    b3, b4 = b_rk[2], b_rk[3]

    a = (
        (a31 - a21) / a32,
        (a42 - a32) / a43,
        (b3 - a43) / b4,
    )
    b = (a21, a32, a43, b4)

    branch_name = "+sqrt(2)" if branch == +1 else "-sqrt(2)"
    return CFEESMethodSpec(
        name=f"CFEES(2,7;x,{branch_name})",
        a=tuple(sp.cancel(v) for v in a),
        b=tuple(sp.cancel(v) for v in b),
        order=2,
        antisymmetric_order=7,
        domain=(
            "admissible EES(2,7;x) values: denominators of the "
            "symbolic tableau and Williamson coefficients are non-zero"
        ),
    )


def symbolic_method_specs() -> tuple[CFEESMethodSpec, ...]:
    """The symbolic families checked by the verifier."""

    return (cfees25(), cfees27(branch=+1), cfees27(branch=-1))


def graft(*children: kauri.PlanarTree) -> kauri.PlanarTree:
    return kauri.PlanarTree([child.list_repr for child in children])


LEAF = kauri.PlanarTree([])
C2 = graft(LEAF)
C3 = graft(C2)
C4 = graft(C3)
C5 = graft(C4)


def cfees25_table_values(x: sp.Expr = X) -> dict[kauri.PlanarTree, sp.Expr]:
    """Published CFEES(2,5;x) character values through tree order five."""

    two_leaves = graft(LEAF, LEAF)
    three_leaves = graft(LEAF, LEAF, LEAF)
    four_leaves = graft(LEAF, LEAF, LEAF, LEAF)

    return {
        kauri.EMPTY_PLANAR_TREE: sp.Integer(1),
        LEAF: sp.Integer(1),
        C2: sp.Rational(1, 2),
        two_leaves: (2 * x - 5) / (32 * (x - 1)),
        C3: sp.Rational(1, 8),
        three_leaves: -(2 * x + 7) / (192 * (x - 1)),
        graft(LEAF, C2): -(x + 2) / (32 * (x - 1)),
        graft(C2, LEAF): sp.Rational(1, 32),
        graft(two_leaves): -(2 * x + 1) / (64 * (x - 1)),
        C4: sp.Integer(0),
        four_leaves: (8 * x**3 + 24 * x**2 + 36 * x - 41)
        / (6144 * (x - 1) ** 3),
        graft(LEAF, LEAF, C2): (4 * x**2 + 10 * x + 13)
        / (768 * (x - 1) ** 2),
        graft(LEAF, C2, LEAF): -(4 * x + 5) / (384 * (x - 1)),
        graft(LEAF, two_leaves): ((2 * x + 1) * (x + 2))
        / (256 * (x - 1) ** 2),
        graft(LEAF, C3): sp.Integer(0),
        graft(C2, LEAF, LEAF): sp.Rational(1, 192),
        graft(C2, C2): -sp.Rational(1, 64) / (2 * x - 1),
        graft(two_leaves, LEAF): -(2 * x + 1) / (256 * (x - 1)),
        graft(three_leaves): (2 * x + 1) ** 2 / (768 * (x - 1) ** 2),
        graft(graft(LEAF, C2)): sp.Integer(0),
        graft(C3, LEAF): sp.Integer(0),
        graft(graft(C2, LEAF)): sp.Integer(0),
        graft(graft(two_leaves)): sp.Integer(0),
        C5: sp.Integer(0),
    }


def trees_by_order(order: int) -> Iterable[kauri.PlanarTree]:
    return kauri.planar_trees_of_order(order)
