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

"""Reusable exact symbolic MKW checks for the CFEES methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import kauri
import sympy as sp
from sympy.polys.domains import QQ
from sympy.polys.fields import field

from .cfees_methods import (
    CFEES25_TABLE_TREES,
    CFEESMethodSpec,
    cfees25,
)
from .exact_method import ExactReusedStageCFMethod


CheckKind = Literal["character", "defect"]


@dataclass(frozen=True)
class OrderCheckResult:
    method_name: str
    planar_order: int
    antisymmetric_order: int


_RATIONAL_FUNCTION_FIELD, _ = field("x", QQ.algebraic_field(sp.sqrt(2)))


def build_exact_method(spec: CFEESMethodSpec) -> kauri.ReusedStageCFMethod:
    """Build the method over the exact field QQ(sqrt(2))(x)."""

    a = [_RATIONAL_FUNCTION_FIELD.from_expr(coeff) for coeff in spec.a]
    b = [_RATIONAL_FUNCTION_FIELD.from_expr(coeff) for coeff in spec.b]
    return ExactReusedStageCFMethod(a, b, spec.name)


def canonical(expr: object) -> sp.Expr:
    """Canonicalise a symbolic expression enough for rational identity checks."""

    if isinstance(expr, sp.Basic):
        sympy_expr = expr
    elif hasattr(expr, "as_expr"):
        sympy_expr = expr.as_expr()
    else:
        sympy_expr = sp.sympify(expr)
    return sp.cancel(sp.radsimp(sympy_expr))


def _evaluate_order(
    target: kauri.Map,
    order: int,
    kind: CheckKind,
) -> tuple[sp.Expr, ...]:
    defects = []
    for tree in kauri.planar_trees_of_order(order):
        defect = target(tree)
        if kind == "character":
            defect -= sp.Rational(1, tree.factorial())
        elif kind != "defect":
            raise ValueError(f"unknown check kind {kind!r}")
        defects.append(sp.Integer(0) if defect == 0 else defect)
    return tuple(defects)


def verify_planar_order(
    spec: CFEESMethodSpec,
    *,
    method: kauri.ReusedStageCFMethod | None = None,
    max_order: int | None = None,
) -> int:
    character = (build_exact_method(spec) if method is None else method).lb_character()
    check_order = spec.order if max_order is None else min(spec.order, max_order)
    for order in range(1, check_order + 1):
        values = _evaluate_order(character, order, "character")
        bad = [value for value in values if value != 0]
        if bad:
            raise AssertionError(
                f"{spec.name}: planar order check failed at order "
                f"{order}: {canonical(bad[0])}"
            )

    return check_order


def verify_antisymmetric_order(
    spec: CFEESMethodSpec,
    *,
    method: kauri.ReusedStageCFMethod | None = None,
    max_order: int | None = None,
) -> int:
    defect_map = (
        build_exact_method(spec) if method is None else method
    ).symmetry_defect_map()
    check_order = (
        spec.antisymmetric_order
        if max_order is None
        else min(spec.antisymmetric_order, max_order)
    )
    for order in range(1, check_order + 1):
        values = _evaluate_order(defect_map, order, "defect")
        bad = [value for value in values if value != 0]
        if bad:
            raise AssertionError(
                f"{spec.name}: antisymmetric check failed at order "
                f"{order}: {canonical(bad[0])}"
            )

    return check_order


def derive_cfees25_table_values(
    *,
    method: kauri.ReusedStageCFMethod | None = None,
) -> dict[kauri.PlanarTree, sp.Expr]:
    spec = cfees25()
    character = (build_exact_method(spec) if method is None else method).lb_character()
    return {
        tree: canonical(character(tree))
        for tree in CFEES25_TABLE_TREES
    }


def verify_method(
    spec: CFEESMethodSpec,
    *,
    method: kauri.ReusedStageCFMethod | None = None,
    max_antisymmetric_order: int | None = None,
) -> OrderCheckResult:
    method = build_exact_method(spec) if method is None else method
    planar_order = verify_planar_order(spec, method=method)
    antisymmetric_order = verify_antisymmetric_order(
        spec,
        method=method,
        max_order=max_antisymmetric_order,
    )
    return OrderCheckResult(
        method_name=spec.name,
        planar_order=planar_order,
        antisymmetric_order=antisymmetric_order,
    )
