"""Shared helpers for the stochastic-volatility adjoint-comparison sweeps.

The existing ``SimpleNeuralSDE.__call__`` (in ``models.nsde``) hard-codes
``adjoint = ReversibleAdjoint()``. Rather than mutate that production model,
we wrap it in :class:`AdjointNeuralSDE` (an ``eqx.Module`` composition) that
selects the adjoint at construction time — so the Adam updates still flow
through the inner ``SimpleNeuralSDE`` parameters, but the
``diffrax.diffeqsolve`` call uses the adjoint we ask for.

Two adjoints are exposed:

- ``"reversible"`` — ``diffrax.ReversibleAdjoint()`` (O(1) memory; the
  paper's headline path).
- ``"autograd"`` — ``diffrax.RecursiveCheckpointAdjoint()`` (the universal
  discretise-then-optimise adjoint; serves as the autograd reference for
  the gradient-error comparison).
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    AbstractAdjoint,
    AbstractReversibleSolver,
    ControlTerm,
    MultiTerm,
    ODETerm,
    RecursiveCheckpointAdjoint,
    ReversibleAdjoint,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)

from experiments.stochastic_volatility.models.nsde import (
    SimpleNeuralSDE,
    _PatchedUReversible,
)

AdjointKind = Literal["autograd", "reversible"]


def build_adjoint(kind: AdjointKind, n_steps: int) -> AbstractAdjoint:
    """Map a string to a diffrax adjoint instance.

    ``n_steps`` is the actual number of solver steps (after dividing the NFE
    budget by ``nfe_per_step``); we use it to size the recursive-checkpoint
    tape generously so that small-budget cells don't run out of room.
    """
    kind = kind.lower()
    if kind == "reversible":
        return ReversibleAdjoint()
    if kind == "autograd":
        return RecursiveCheckpointAdjoint(checkpoints=n_steps + 8)
    raise ValueError(f"unknown adjoint {kind!r}")


def _solve_with_adjoint(
    model: SimpleNeuralSDE,
    y0: jax.Array,
    key: jax.Array,
    *,
    adjoint: AbstractAdjoint,
    autograd_path: bool,
) -> jax.Array:
    """Re-runs the same diffeqsolve as ``SimpleNeuralSDE.__call__`` but with
    the supplied adjoint. The reversible-adjoint path requires an
    ``AbstractReversibleSolver``; the autograd path uses the original solver
    unwrapped (no need to fake reversibility).
    """
    t1 = model.n_steps * model.dt
    brownian_path = VirtualBrownianTree(
        t0=0.0,
        t1=t1,
        tol=model.solve_dt / 4.0,
        shape=(model.state_dim,),
        key=key,
    )
    term = MultiTerm(
        ODETerm(model.drift_field),
        ControlTerm(model.diffusion_field, brownian_path),
    )

    if autograd_path:
        solver = model.solver
    else:
        solver = (
            model.solver
            if isinstance(model.solver, AbstractReversibleSolver)
            else _PatchedUReversible(model.solver)
        )

    if model.save_path:
        ts = jnp.linspace(0.0, t1, model.n_save, dtype=jnp.float32)
        saveat = SaveAt(ts=ts)
    else:
        saveat = SaveAt(t1=True)

    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t1,
        dt0=model.solve_dt,
        y0=y0,
        args=None,
        saveat=saveat,
        adjoint=adjoint,
        max_steps=model.solve_n_steps + 8,
    )
    assert sol.ys is not None
    return sol.ys if model.save_path else sol.ys[0]


class AdjointNeuralSDE(eqx.Module):
    """Composition of ``SimpleNeuralSDE`` plus a fixed adjoint choice.

    Forwarding ``__call__`` lets this work as a drop-in replacement wherever
    ``SimpleNeuralSDE`` is used (e.g.\ ``make_sample_fn``'s ``jax.vmap(model)``).
    """

    inner: SimpleNeuralSDE
    adjoint_kind: str = eqx.field(static=True)

    def __init__(self, inner: SimpleNeuralSDE, adjoint_kind: AdjointKind = "reversible"):
        self.inner = inner
        if adjoint_kind not in ("autograd", "reversible"):
            raise ValueError(
                f"adjoint_kind must be 'autograd' or 'reversible', got {adjoint_kind!r}"
            )
        self.adjoint_kind = adjoint_kind

    @property
    def state_dim(self) -> int:
        return self.inner.state_dim

    def __call__(self, y0: jax.Array, key: jax.Array) -> jax.Array:
        adjoint = build_adjoint(self.adjoint_kind, self.inner.solve_n_steps)
        return _solve_with_adjoint(
            self.inner,
            y0,
            key,
            adjoint=adjoint,
            autograd_path=(self.adjoint_kind == "autograd"),
        )
