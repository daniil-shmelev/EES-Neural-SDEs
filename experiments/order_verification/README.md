# Lie–Butcher / MKW order-condition verification (paper App. `sec:cfees-order-conditions`, Tab. `tab:alpha-cf-ees25-x`)

Symbolic verification of the order conditions of $\mathrm{CF\text{-}EES}(2,5;x)$ on the Munthe-Kaas–Wright Hopf algebra of planar rooted forests. Backs Theorem `thm:cf-ees25-orders`:

1. **Planar order $2$**: $\alpha(\tau) = 1/\tau!$ for every planar tree of order $\leq 2$.
2. **Antisymmetric order $5$**: $D := (\mathrm{sign} \cdot \alpha) \star_{\mathrm{MKW}} \alpha$ satisfies $D(\tau) = \varepsilon(\tau)$ for every planar tree of order $\leq 5$.

All scripts are pure-`sympy`, exact-rational, no JAX. They produce the symbolic table of $\alpha(\tau)$ values listed in the manuscript.

## Run

```bash
# MKW Hopf-algebra independent verification (no kauri)
python -m experiments.order_verification.scripts.mkw_verify
python -m experiments.order_verification.scripts.mkw_verify_cf

# Scheme α character: planar order, antisymmetric order, tree/algebraic checks
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

(pulls `kauri` and `sympy`; the verification scripts themselves are self-contained sympy, but `kauri` is the package that produced the symbolic table in the manuscript.)
