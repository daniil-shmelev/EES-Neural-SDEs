# Explicit and Effectively Symmetric Schemes for Neural SDEs

This repository contains code supporting the paper "Explicit and Effectively Symmetric Schemes for Neural SDEs".
We implement and test Explicit and Effectively Symmetric (EES) Runge-Kutta schemes for SDEs.

## Implementations of EES Schemes

The following repositories contain implementations of EES schemes.
<span style="color:red">To run the code in the `examples` directory, please
pip install `diffrax` and/or `torchsde` directly from the forks below.</span>

### Diffrax

A fork of the `diffrax` repository supporting EES schemes is available at:

```python
https://github.com/daniil-shmelev/diffrax
```

### torchsde

A fork of the `torchsde` repository implementing EES schemes as Stratonovich SDE solvers
is available at:

```python
https://github.com/daniil-shmelev/torchsde
```

## Experiments

This repository contains code reproducing the following experiments:

- `stability.py` plots cross-sections of the mean-square stability domains for RK3, RK4 and EES(2,5)
applied to SDEs.
- `convergence.py` reproduces an example from the paper "Runge-Kutta methods for rough differential equations" (Redmann & Riedel, 2020)
concerning convergence rates of RDE schemes. We apply this example to test the convergence rates of EES(2,5). In addition
to the discretisation error, we evaluate the error in recovering the initial condition, y_0.
