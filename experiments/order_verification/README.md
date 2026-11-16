# Lie–Butcher / MKW order-condition verification (paper App. `sec:cfees-order-conditions`, Tab. `tab:alpha-cf-ees25-x`)

Symbolic verification of the order conditions of $\mathrm{CF\text{-}EES}(2,5;x)$ on the Munthe-Kaas–Wright Hopf algebra of planar rooted forests. Backs Theorem `thm:cf-ees25-orders`:

1. **Planar order $2$**: $\alpha(\tau) = 1/\tau!$ for every planar tree of order $\leq 2$.
2. **Antisymmetric order $5$**: $D := (\mathrm{sign} \cdot \alpha) \star_{\mathrm{MKW}} \alpha$ satisfies $D(\tau) = \varepsilon(\tau)$ for every planar tree of order $\leq 5$.

All scripts are pure-`sympy`, exact-rational, no JAX. They produce the symbolic table of $\alpha(\tau)$ values listed in the manuscript.

## Run

From the repository root:

```bash
./run order-verification
```

The launcher selects the uv dependencies and runs the exact verifier in `siam/scripts/verify_cfees.py`. It uses [kauri](https://github.com/daniil-shmelev/kauri) to check the cosine schemes with s = 3, 4, 5 on all ordered forests (forward order two; antisymmetric order exactly 5, 7 and 9) and the 24 printed tree coefficients of the three-stage scheme; SymPy only supplies the number fields. It reports to the terminal and does not read the manuscript. Use `--help` for options. The earlier independent algebraic cross-checks remain in `scripts/` for development.
