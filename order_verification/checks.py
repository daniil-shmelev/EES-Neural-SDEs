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
    CFEESMethodSpec,
    cfees25,
    cfees25_table_values,
    trees_by_order,
)
from .exact_method import ExactReusedStageCFMethod


CheckKind = Literal["character", "defect"]


@dataclass(frozen=True)
class OrderCheckResult:
    method_name: str
    planar_order: int
    antisymmetric_order: int


_RATIONAL_FUNCTION_FIELD, _ = field("x", QQ.algebraic_field(sp.sqrt(2)))


def _to_field(expr):
    return _RATIONAL_FUNCTION_FIELD.from_expr(expr)


def _to_sympy_expr(expr) -> sp.Expr:
    if isinstance(expr, sp.Basic):
        return expr
    if hasattr(expr, "as_expr"):
        return expr.as_expr()
    return sp.sympify(expr)


def build_exact_method(spec: CFEESMethodSpec) -> kauri.ReusedStageCFMethod:
    """Build the method over the exact field QQ(sqrt(2))(x)."""

    a = [_to_field(coeff) for coeff in spec.a]
    b = [_to_field(coeff) for coeff in spec.b]
    return ExactReusedStageCFMethod(a, b, spec.name)


def canonical(expr: sp.Expr) -> sp.Expr:
    """Canonicalise a symbolic expression enough for rational identity checks."""

    return sp.cancel(sp.radsimp(_to_sympy_expr(expr)))


def exact_character_value(tree: kauri.PlanarTree) -> sp.Expr:
    return sp.Rational(1, tree.factorial())


def _defect_value(kind: CheckKind, tree: kauri.PlanarTree, value: sp.Expr) -> sp.Expr:
    if kind == "character":
        defect = value - exact_character_value(tree)
        return sp.Integer(0) if defect == 0 else defect
    if kind == "defect":
        return sp.Integer(0) if value == 0 else value
    raise ValueError(f"unknown check kind {kind!r}")


def _evaluate_order(
    target: kauri.Map,
    order: int,
    kind: CheckKind,
) -> tuple[sp.Expr, ...]:
    trees = tuple(trees_by_order(order))
    values = tuple(target(tree) for tree in trees)
    return tuple(
        _defect_value(kind, tree, value)
        for tree, value in zip(trees, values, strict=True)
    )


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


def verify_cfees25_table(
    *,
    method: kauri.ReusedStageCFMethod | None = None,
) -> None:
    spec = cfees25()
    character = (build_exact_method(spec) if method is None else method).lb_character()
    for tree, expected in cfees25_table_values().items():
        diff = character(tree) - _to_field(expected)
        if diff != 0:
            raise AssertionError(
                f"CFEES(2,5;x) table mismatch at "
                f"{tree.list_repr}: {canonical(diff)}"
            )


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
