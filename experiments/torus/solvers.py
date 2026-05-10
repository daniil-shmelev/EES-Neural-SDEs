"""Solver and adjoint factories for the generic torus benchmark."""

from __future__ import annotations

from enum import StrEnum

from diffrax import (
    AbstractAdjoint,
    AbstractSolver,
    DirectAdjoint,
    RecursiveCheckpointAdjoint,
    ReversibleAdjoint,
)
from georax import CFEES25 as _GeoraxCFEES25
from georax import CG2, CG4, RKMK


class CFEES25(_GeoraxCFEES25):
    """CFEES25 with a backward-step shim across Diffrax ABI versions."""

    def backward_step(self, terms, t0, t1, y1, args, *state_and_jump):
        if len(state_and_jump) == 2:
            solver_state, made_jump = state_and_jump
        elif len(state_and_jump) == 3:
            _, solver_state, made_jump = state_and_jump
        else:
            raise TypeError(
                "CFEES25.backward_step expected solver state and made_jump, "
                f"got {len(state_and_jump)} trailing arguments."
            )
        y0, _, dense_info, solver_state, result = self.step(
            terms, t1, t0, y1, args, solver_state, made_jump
        )
        return y0, dense_info, solver_state, result


class Solvers(StrEnum):
    CFEES25 = "cfees25"
    CG2 = "cg2"
    CG4 = "cg4"
    RKMK = "rkmk"


_GEORAX_SOLVERS: dict[Solvers, type[AbstractSolver]] = {
    Solvers.CFEES25: CFEES25,
    Solvers.CG2: CG2,
    Solvers.CG4: CG4,
    Solvers.RKMK: RKMK,
}


def build_solver(name: Solvers) -> AbstractSolver:
    return _GEORAX_SOLVERS[name]()


def build_adjoint(name: str | None, n_steps: int) -> AbstractAdjoint | None:
    if name is None:
        return None
    max_steps = n_steps + 8
    match name.lower():
        case "auto":
            return None
        case "reversible":
            return ReversibleAdjoint()
        case "direct":
            return DirectAdjoint()
        case "checkpoint_recursive" | "recursive" | "checkpoint_log":
            return RecursiveCheckpointAdjoint()
        case "checkpoint_full" | "full":
            return RecursiveCheckpointAdjoint(checkpoints=max_steps)
        case _:
            raise ValueError(f"unknown adjoint {name!r}")
