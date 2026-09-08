"""Exact ordered-forest verification for the SIAM manuscript.

Requires SymPy. Run from any directory:
    python3 verify_cfees.py --manuscript /path/to/SIAM/ees_simods.tex

The default checks the entire three-stage family over Q(x) and the
selected four-stage method over Q(sqrt(2)). Both proposed embedded
companions and the existing georax three-stage companion are checked.

A tree is a tuple of its ordered children; a forest is a tuple of trees.
The empty forest is the multiplicative identity. Products concatenate
forests. Grafting raises the degree by one. These are the frozen-flow
Taylor recurrences in the manuscript, not a non-planar RK verifier.
"""

from __future__ import annotations

import argparse
from functools import cache
from math import comb, factorial
from pathlib import Path
import re
import time

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


@cache
def tree_degree(tree):
    return 1 + sum(tree_degree(child) for child in tree)


@cache
def forest_degree(forest):
    return sum(tree_degree(tree) for tree in forest)


class ForestAlgebra:
    def __init__(self, domain, degree):
        self.domain = domain
        self.degree = degree
        self.zero = domain.zero
        self.one = domain.one

    def scale(self, series, coefficient):
        if not coefficient:
            return {}
        return {forest: coefficient * value for forest, value in series.items()}

    def add(self, left, right):
        result = left.copy()
        for forest, coefficient in right.items():
            result[forest] = result.get(forest, self.zero) + coefficient
            if not result[forest]:
                del result[forest]
        return result

    def multiply(self, left, right):
        result = {}
        for u, a in left.items():
            for v, b in right.items():
                if forest_degree(u) + forest_degree(v) <= self.degree:
                    forest = u + v
                    result[forest] = result.get(forest, self.zero) + a * b
        return {forest: value for forest, value in result.items() if value}

    def exponential(self, series):
        power = result = {(): self.one}
        for k in range(1, self.degree + 1):
            power = self.multiply(power, series)
            if not power:
                break
            result = self.add(result, self.scale(power, self.one / factorial(k)))
        return result

    def graft(self, series):
        return {
            (forest,): coefficient
            for forest, coefficient in series.items()
            if forest_degree(forest) < self.degree
        }

    def step(self, series, a, b, sign=1):
        increment = {}
        for ai, bi in zip(a, b, strict=True):
            increment = self.add(
                self.scale(increment, ai), self.scale(self.graft(series), sign)
            )
            series = self.multiply(series, self.exponential(self.scale(increment, bi)))
        return series


def family_five():
    symbol = sp.Symbol("x")
    domain = sp.QQ.frac_field(symbol)
    x = domain.from_sympy(symbol)
    a = [domain.zero, (4*x*x-2*x+1)/(2*(x-1)),
         -(4*x*x-2*x+1)/((2*x-1)**2*(2*x+1))]
    b = [(2*x+1)/(4*(1-x)), (1-x)/(1-4*x*x), (1-2*x)/2]
    return domain, a, b


def family_seven():
    root = sp.sqrt(2)
    domain = sp.QQ.algebraic_field(root)
    r = domain.from_sympy(root)
    a = [domain.zero, (-7+4*r)/3, -(4+5*r)/12, 3*(-31+8*r)/49]
    b = [(2-r)/3, (4+r)/8, 3*(3-r)/7, (9-4*r)/14]
    return domain, a, b


def tableau(domain, a, b):
    size = len(a)
    increment = [domain.zero] * size
    state = [domain.zero] * size
    rows = []
    for i, (ai, bi) in enumerate(zip(a, b, strict=True)):
        rows.append(state.copy())
        increment = [ai*coefficient for coefficient in increment]
        increment[i] += domain.one
        state = [u+bi*v for u, v in zip(state, increment, strict=True)]
    return rows, state, increment


def verify_published_tree_table(domain, forward, source):
    """Compare the printed LaTeX entries with the exact stage expansion."""
    leaf = ()
    chain2 = (leaf,)
    chain3 = (chain2,)
    two_leaves = (leaf, leaf)
    trees = {
        "TLeaf": leaf,
        "TChainTwo": chain2,
        "TTwoLeaves": two_leaves,
        "TChainThree": chain3,
        "TThreeLeaves": (leaf, leaf, leaf),
        "TLeafChainTwo": (leaf, chain2),
        "TChainTwoLeaf": (chain2, leaf),
        "TNestedTwoLeaves": (two_leaves,),
        "TChainFour": (chain3,),
        "TFourLeaves": (leaf, leaf, leaf, leaf),
        "TLeafLeafChainTwo": (leaf, leaf, chain2),
        "TLeafChainTwoLeaf": (leaf, chain2, leaf),
        "TLeafNestedTwoLeaves": (leaf, two_leaves),
        "TLeafChainThree": (leaf, chain3),
        "TChainTwoLeafLeaf": (chain2, leaf, leaf),
        "TChainTwoChainTwo": (chain2, chain2),
        "TNestedTwoLeavesLeaf": (two_leaves, leaf),
        "TNestedThreeLeaves": ((leaf, leaf, leaf),),
        "TNestedLeafChainTwo": ((leaf, chain2),),
        "TChainThreeLeaf": (chain3, leaf),
        "TNestedChainTwoLeaf": ((chain2, leaf),),
        "TDoubleNestedTwoLeaves": ((two_leaves,),),
        "TChainFive": ((chain3,),),
    }
    assert len(set(trees.values())) == 23
    assert [sum(tree_degree(t) == n for t in trees.values()) for n in range(1, 6)] == [1, 1, 2, 5, 14]

    def fraction_expression(latex):
        # The table uses integer constants and one fraction per nonconstant entry.
        match = re.fullmatch(r"(-?)\\tfrac\{([^{}]+)\}\{([^{}]+)\}", latex)
        expression = (f"{match[1]}(({match[2]})/({match[3]}))" if match else latex)
        assert re.fullmatch(r"[0-9x()+*/^ -]+", expression), latex
        return parse_expr(expression.replace("^", "**"),
                          local_dict={"x": sp.Symbol("x")},
                          transformations=standard_transformations + (implicit_multiplication_application,))

    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", source.read_text(), re.S)
    printed = [table for table in tables if r"\label{tab:alpha-cf-ees25-x}" in table]
    assert len(printed) == 1
    entries = re.findall(r"\\(T\w+)\s*&\s*\$([^$]+)\$", printed[0])
    assert len(entries) == 24
    assert {name for name, _ in entries} == set(trees) | {"TEmpty"}
    for name, latex in entries:
        forest = () if name == "TEmpty" else (trees[name],)
        printed = domain.from_sympy(fraction_expression(latex))
        assert printed == forward.get(forest, domain.zero), (name, latex)
    print("  All 24 printed tree-table entries verified in Q(x), including zeros.", flush=True)


def verify(name, degree, coefficients, manuscript):
    domain, a, b = coefficients
    algebra = ForestAlgebra(domain, degree)
    identity = {(): domain.one}
    start = time.monotonic()
    forward = algebra.step(identity, a, b)
    leaf, chain = (), ((),)
    expected = {(): domain.one, (leaf,): domain.one,
                (chain,): domain.one/2, (leaf, leaf): domain.one/2}
    assert {w: c for w, c in forward.items() if forest_degree(w) <= 2} == expected
    if degree == 5 and manuscript is not None:
        verify_published_tree_table(domain, forward, manuscript)
    print(f"{name}: forward order two verified ({time.monotonic()-start:.2f}s)", flush=True)
    reverse = algebra.step(forward, a, b, sign=-1)
    for n in range(1, degree+1):
        residual = {w: c for w, c in reverse.items() if forest_degree(w) == n}
        print(f"  degree {n}: {comb(2*n, n)//(n+1)} forests, "
              f"{len(residual)} nonzero residuals", flush=True)
        assert not residual, residual
    assert reverse == identity
    rows, weights, increment = tableau(domain, a, b)
    nodes = [sum(row, domain.zero) for row in rows]
    assert sum(weights, domain.zero) == domain.one
    assert sum((bi*ci for bi, ci in zip(weights, nodes)), domain.zero) == domain.one/2
    sigma = sum(increment, domain.zero)
    mu = sum((di*ci for di, ci in zip(increment, nodes)), domain.zero)/sigma
    print(f"  final-increment normalization: {sp.factor(domain.to_sympy(sigma))}")
    print(f"  embedded second-order coefficient: {sp.factor(domain.to_sympy(mu))}")
    assert mu != domain.one/2
    print(f"{name}: all checks passed ({time.monotonic()-start:.2f}s)", flush=True)


def verify_selected_companions():
    q = sp.QQ
    five = q, [q.zero, q(-7, 15), q(-35, 32)], [q(1, 3), q(15, 16), q(2, 5)]
    for (domain, a, b), expected_sigma, expected_mu in [
        (five, sp.Rational(5, 12), sp.Rational(9, 8)),
        (family_seven(), (10-sp.sqrt(2))/21, 3*sp.sqrt(2)/4),
    ]:
        rows, _, increment = tableau(domain, a, b)
        sigma = sum(increment, domain.zero)
        nodes = [sum(row, domain.zero) for row in rows]
        mu = sum((di*ci for di, ci in zip(increment, nodes)), domain.zero)/sigma
        assert sigma == domain.from_sympy(expected_sigma)
        assert mu == domain.from_sympy(expected_mu)

    algebra = ForestAlgebra(q, 2)
    _, a, b = five
    state, increment, stages = {(): q.one}, {}, []
    for ai, bi in zip(a[:2], b[:2], strict=True):
        stages.append(algebra.graft(state))
        increment = algebra.add(algebra.scale(increment, ai), stages[-1])
        state = algebra.multiply(state, algebra.exponential(algebra.scale(increment, bi)))
    extra = algebra.add(algebra.scale(stages[0], q(5, 48)),
                        algebra.scale(stages[1], q(1, 16)))
    companion = algebra.multiply(state, algebra.exponential(extra))
    leaf, chain = (), ((),)
    assert companion == {(): q.one, (leaf,): q.one,
                         (chain,): q(1, 3), (leaf, leaf): q(1, 2)}
    print("Both final-increment companions and the georax penultimate companion verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, help="Also verify the printed tree table in this manuscript.")
    args = parser.parse_args()
    verify("CF-EES(2,5;x)", 5, family_five(), args.manuscript)
    verify("CF-EES(2,7), selected parameter", 7, family_seven(), args.manuscript)
    verify_selected_companions()


if __name__ == "__main__":
    main()
