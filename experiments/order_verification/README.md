# Lie-Butcher / MKW Order-Condition Verification

Symbolic verification of the order conditions of
`CF-EES(2,5;x)` on the Munthe-Kaas-Wright Hopf algebra of planar rooted
forests. These scripts support the paper's order-condition theorem and the
symbolic alpha-value table.

1. **Planar order 2**: `alpha(tau) = 1 / tau!` for every planar tree of order
   at most 2.
2. **Antisymmetric order 5**:
   `D := (sign * alpha) star_MKW alpha` satisfies `D(tau) = epsilon(tau)` for
   every planar tree of order at most 5.

All scripts are pure `sympy`, exact-rational checks. They do not require JAX.

## Run

```bash
# MKW Hopf-algebra independent verification
python -m experiments.order_verification.scripts.mkw_verify
python -m experiments.order_verification.scripts.mkw_verify_cf

# Scheme-alpha character: planar order, antisymmetric order, tree/algebraic checks
python -m experiments.order_verification.scripts.scheme_a_order_verify
python -m experiments.order_verification.scripts.scheme_a_antisym_at_N
python -m experiments.order_verification.scripts.scheme_a_algebraic_verify
python -m experiments.order_verification.scripts.scheme_a_tree_algebraic_verify

# Tree-vs-matrix consistency cross-check
python -m experiments.order_verification.scripts.cross_check_tree_vs_matrix
```

## Setup

```bash
uv pip install -e ".[order-verification]"
```

This extra installs `kauri` and `sympy`. The verification scripts themselves
are self-contained `sympy` programs; `kauri` provides the symbolic Runge-Kutta
tooling used by related experiment scripts.
