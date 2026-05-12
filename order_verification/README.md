# Hopf-Algebra Order Verification

This package verifies the symbolic Lie-Butcher order conditions for the
CFEES methods on the Munthe-Kaas-Wright Hopf algebra of planar rooted
forests. It is a repository-level verification package rather than a numerical
experiment.

The verifier constructs the reused-stage CFEES character with
`kauri.ReusedStageCFMethod`, which implements the Owren pseudo-stage recursion
used in the published appendix. Coefficients are evaluated over the exact
rational-function field `QQ(sqrt(2))(x)`, so zero checks remain symbolic
identities in `x`. It then checks:

1. `CFEES(2,5;x)` has planar order `2` and antisymmetric order `5` for all
   admissible `x`.
2. `CFEES(2,7;x)` has planar order `2` and antisymmetric order `7` for all
   admissible `x`, for both `+sqrt(2)` and `-sqrt(2)` branches.
3. The symbolic character values of `CFEES(2,5;x)` through tree order `5`
   match the table printed in the appendix.

All checks are symbolic identities in `x`, with the usual admissibility
condition that the method denominators are non-zero. The verifier checks the
vanishing order conditions up to the stated planar and antisymmetric orders;
it does not test the first order above those conditions.

## Run

```bash
python -m order_verification.verify
```

The verifier runs serially and reuses each constructed Kauri method instance
across the table and order checks.

For short timing runs, cap the antisymmetric order:

```bash
python -m order_verification.verify --skip-table --max-antisymmetric-order 3
```

## Setup

```bash
uv pip install -e ".[order-verification]"
```

The `order-verification` extra installs `kauri` and `sympy`.
