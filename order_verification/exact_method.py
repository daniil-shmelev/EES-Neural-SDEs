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

"""Exact reused-stage CF method used by the symbolic verifier."""

from __future__ import annotations

from functools import lru_cache
from math import factorial

import kauri
from kauri._protocols import ForestLike
from kauri.mkw.mkw import _as_basis_aware_map
from kauri.trees import EMPTY_PLANAR_TREE


def _b_plus_repr(tree_reprs: tuple) -> tuple:
    return tree_reprs + (0,)


class ExactReusedStageCFMethod(kauri.ReusedStageCFMethod):
    """Reused-stage CF method with tuple-based cached recursion."""

    def _build_lb_character(self, a_coeffs, b_coeffs):
        rows, zero, one = self._owren_rows(a_coeffs, b_coeffs)
        final_row = self.s

        @lru_cache(maxsize=None)
        def g(row_index: int, exp_count: int, tree_repr):
            if tree_repr is None:
                return one

            children = tree_repr[:-1]
            if not children:
                return one
            if exp_count == 0:
                return zero

            total = zero
            for split in range(len(children) + 1):
                left = _b_plus_repr(children[:split])
                right = _b_plus_repr(children[split:])
                total = total + (
                    g(row_index, exp_count - 1, left)
                    * exp_character(row_index, exp_count, right)
                )
            return total

        @lru_cache(maxsize=None)
        def exp_character(row_index: int, exp_count: int, tree_repr):
            if tree_repr is None:
                return one

            children = tree_repr[:-1]
            if not children:
                return one

            total = one
            for child in children:
                total = total * vector_field(row_index, exp_count, child)
            return total / factorial(len(children))

        @lru_cache(maxsize=None)
        def vector_field(row_index: int, exp_count: int, tree_repr):
            coeffs = rows[row_index][exp_count - 1]
            total = zero
            for stage_index, coeff in enumerate(coeffs):
                total = total + coeff * g(
                    stage_index,
                    len(rows[stage_index]),
                    tree_repr,
                )
            return total

        def _char(x):
            if isinstance(x, ForestLike):
                tree_reprs = tuple(
                    t.list_repr for t in x.tree_list if t.list_repr is not None
                )
                if not tree_reprs:
                    return one
                return g(final_row, len(rows[final_row]), _b_plus_repr(tree_reprs))

            if x == EMPTY_PLANAR_TREE:
                return one
            return g(final_row, len(rows[final_row]), _b_plus_repr((x.list_repr,)))

        return _as_basis_aware_map(_char)
