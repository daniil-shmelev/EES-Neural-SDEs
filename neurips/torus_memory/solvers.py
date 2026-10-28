"""Solver and adjoint choices for the retained torus-memory benchmark."""

from enum import StrEnum

from diffrax import (
    AbstractAdjoint,
    AbstractSolver,
    DirectAdjoint,
    RecursiveCheckpointAdjoint,
    ReversibleAdjoint,
)
from georax import CFEES25, CFEES27, CG2, CG4


class Solvers(StrEnum):
    CFEES25 = "cfees25"
    CFEES27 = "cfees27"
    CG2 = "cg2"
    CG4 = "cg4"


_SOLVER_CLASSES: dict[Solvers, type[AbstractSolver]] = {
    Solvers.CFEES25: CFEES25,
    Solvers.CFEES27: CFEES27,
    Solvers.CG2: CG2,
    Solvers.CG4: CG4,
}


def build_solver(name: Solvers) -> AbstractSolver:
    cls = _SOLVER_CLASSES[name]
    return cls()


def build_adjoint(name: str | None, n_steps: int) -> AbstractAdjoint | None:
    """Map an adjoint-name string to a diffrax adjoint instance.

    - ``"auto"`` / ``None``: return ``None`` so the model picks
      ``ReversibleAdjoint`` for reversible solvers and ``DirectAdjoint``
      otherwise.
    - ``"reversible"``: ``ReversibleAdjoint()``.
    - ``"direct"``: ``DirectAdjoint()``.
    - ``"checkpoint_recursive"`` (aka ``"checkpoint_log"``):
      ``RecursiveCheckpointAdjoint()`` with diffrax's default Stumm-Walther
      online binomial treeverse (O(sqrt(n_steps)) checkpoints).
    - ``"checkpoint_full"``: ``RecursiveCheckpointAdjoint(checkpoints=n_steps+8)``,
      i.e. one snapshot per step (O(n_steps) tape).
    """
    if name is None:
        return None
    name = name.lower()
    max_steps = n_steps + 8
    if name == "auto":
        return None
    if name == "reversible":
        return ReversibleAdjoint()
    if name == "direct":
        return DirectAdjoint()
    if name in ("checkpoint_recursive", "recursive", "checkpoint_log"):
        return RecursiveCheckpointAdjoint()
    if name in ("checkpoint_full", "full"):
        return RecursiveCheckpointAdjoint(checkpoints=max_steps)
    raise ValueError(f"unknown adjoint {name!r}")
