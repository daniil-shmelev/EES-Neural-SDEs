# Hopf-Algebra Order Verification

This package verifies the symbolic Lie-Butcher order conditions for the
CFEES methods on the Munthe-Kaas-Wright Hopf algebra of planar rooted
forests.

The verifier constructs the reused-stage CFEES character with
`kauri.ReusedStageCFMethod`, which implements the Owren pseudo-stage recursion
used in the published appendix. Coefficients are evaluated over the exact
rational-function field `QQ(sqrt(2))(x)`. It then checks the order conditions and derives the
appendix-table values:

1. `CFEES(2,5;x)` has planar order `2` and antisymmetric order `5` for all
   admissible `x`.
2. `CFEES(2,7;x)` has planar order `2` and antisymmetric order `7` for all
   admissible `x`, for both `+sqrt(2)` and `-sqrt(2)` branches.
3. The symbolic character values of `CFEES(2,5;x)` through tree order `5`
   are derived from the constructed character in the appendix-table order.

## Run

```bash
python -m order_verification.verify
```

## Setup

```bash
uv pip install -e ".[order-verification]"
```

The `order-verification` extra installs `kauri` and `sympy`.
