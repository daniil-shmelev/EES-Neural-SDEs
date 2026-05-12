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

"""Command-line entrypoint for the symbolic CFEES order checks."""

from __future__ import annotations

import argparse

from .checks import build_exact_method, derive_cfees25_table_values, verify_method
from .cfees_methods import SYMBOLIC_METHOD_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the symbolic MKW Hopf-algebra character conditions for "
            "CFEES(2,5;x) and CFEES(2,7;x)."
        )
    )
    parser.add_argument(
        "--skip-table",
        action="store_true",
        help="Skip deriving the CFEES(2,5;x) appendix table values.",
    )
    parser.add_argument(
        "--max-antisymmetric-order",
        type=int,
        default=None,
        help=(
            "Cap the antisymmetric order checked. This is useful for "
            "short benchmarks before running the full order-seven check."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for spec in SYMBOLIC_METHOD_SPECS:
        method = build_exact_method(spec)
        if not args.skip_table and spec.name == "CFEES(2,5;x)":
            table_values = derive_cfees25_table_values(method=method)
            print("CFEES(2,5;x) derived character table values:")
            for tree, value in table_values.items():
                tree_repr = (
                    "EMPTY_PLANAR_TREE"
                    if tree.list_repr is None
                    else str(tree.list_repr)
                )
                print(f"  {tree_repr}: {value}")
            print(f"CFEES(2,5;x) table entries: {len(table_values)}")

        result = verify_method(
            spec,
            method=method,
            max_antisymmetric_order=args.max_antisymmetric_order,
        )
        if args.max_antisymmetric_order is None:
            print(
                f"{result.method_name}: planar order {result.planar_order}, "
                f"antisymmetric order {result.antisymmetric_order}."
            )
        else:
            print(
                f"{result.method_name}: planar order {result.planar_order}, "
                "antisymmetric conditions checked through order "
                f"{result.antisymmetric_order}."
            )

    print("All symbolic CFEES order checks passed.")


if __name__ == "__main__":
    main()
