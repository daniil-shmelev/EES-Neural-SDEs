from collections.abc import Callable
from typing import ClassVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    AbstractERK,
    AbstractReversibleSolver,
    ControlTerm,
    LocalLinearInterpolation,
    MultiTerm,
    ODETerm,
    ReversibleAdjoint,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
from diffrax._solution import update_result
from equinox.internal import ω

ω = cast(Callable, ω)


SOLVER_NFE_PER_STEP = {
    "Euler": 2,
    "Midpoint": 4,
    "EES25": 3,
    "CFEES25": 3,
    "CG2": 2,
    "ReversibleHeun": 1,
}


class _PatchedUReversible(AbstractReversibleSolver):
    """Local replacement for diffrax.UReversible.

    The installed Diffrax build in this workspace does not initialize the wrapped
    `solver` field, so we keep the wrapper local to this repository.
    """

    solver: AbstractERK
    coupling_parameter: float
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    @property
    def root_finder(self):
        return self.solver.root_finder

    @property
    def root_find_max_steps(self):
        return self.solver.root_find_max_steps

    def __init__(self, solver: AbstractERK, coupling_parameter: float = 0.999):
        if hasattr(solver, "disable_fsal"):
            solver = eqx.tree_at(lambda s: s.disable_fsal, solver, True)
        self.solver = solver
        self.coupling_parameter = coupling_parameter

    def order(self, terms):
        return self.solver.order(terms)

    def strong_order(self, terms):
        return self.solver.strong_order(terms)

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, args
        return y0

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del made_jump
        z0 = solver_state

        step_z0, _, _, _, result1 = self.solver.step(
            terms, t0, t1, z0, args, None, True
        )
        y1 = (self.coupling_parameter * (ω(y0) - ω(z0)) + ω(step_z0)).ω

        step_y1, y_error, _, _, result2 = self.solver.step(
            terms, t1, t0, y1, args, None, True
        )
        z1 = (ω(y1) + ω(z0) - ω(step_y1)).ω

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, z1, update_result(result1, result2)

    def backward_step(self, terms, t0, t1, y1, args, ts_state, solver_state, made_jump):
        del made_jump, ts_state
        z1 = solver_state

        step_y1, _, _, _, result1 = self.solver.step(
            terms, t1, t0, y1, args, None, True
        )
        z0 = (ω(z1) - ω(y1) + ω(step_y1)).ω

        step_z0, _, _, _, result2 = self.solver.step(
            terms, t0, t1, z0, args, None, True
        )
        y0 = ((1 / self.coupling_parameter) * (ω(y1) - ω(step_z0)) + ω(z0)).ω

        dense_info = dict(y0=y0, y1=y1)
        return y0, dense_info, z0, update_result(result1, result2)

    def func(self, terms, t0, y0, args):
        return self.solver.func(terms, t0, y0, args)


def _solver_nfe_per_step(solver) -> int:
    solver_name = type(solver).__name__
    try:
        return SOLVER_NFE_PER_STEP[solver_name]
    except KeyError as exc:
        raise ValueError(
            f"No fixed-NFE cost registered for solver {solver_name!r}."
        ) from exc


def lipswish(x):
    return 0.909 * jax.nn.silu(x)


class DriftField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, state_dim: int, hidden_dim: int, *, key: jax.Array):
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + 1,  # +1 for time
            out_size=state_dim,
            width_size=hidden_dim,
            depth=3,
            activation=lipswish,
            key=key,
        )

    def __call__(self, t, y, args):
        del args
        t_vec = jnp.atleast_1d(t)
        return self.mlp(jnp.concatenate([t_vec, y]))


class DiffusionField(eqx.Module):
    mlp: eqx.nn.MLP
    diffusion_scale: float

    def __init__(
        self, state_dim: int, hidden_dim: int, diffusion_scale: float, *, key: jax.Array
    ):
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + 1,  # +1 for time
            out_size=state_dim,
            width_size=hidden_dim,
            depth=2,
            activation=lipswish,
            key=key,
        )
        self.diffusion_scale = diffusion_scale

    def __call__(self, t, y, args):
        del args
        t_vec = jnp.atleast_1d(t)
        scales = (
            jax.nn.softplus(self.mlp(jnp.concatenate([t_vec, y])))
            * self.diffusion_scale
        )
        # Diagonal diffusion: shape (state_dim, state_dim)
        return jnp.diag(scales)


class SimpleNeuralSDE(eqx.Module):
    """Euclidean neural SDE on R^d.

    Here `n_steps` is treated as a forward NFE budget. The actual number of
    solver steps depends on the chosen integrator.

    When save_path=True saves `n_save` states on a common grid and returns
    shape (n_save, state_dim). Use for full-path generation tasks.
    """

    drift_field: DriftField
    diffusion_field: DiffusionField
    name: str = eqx.field(static=True)

    state_dim: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)  # Forward NFE budget
    dt: float = eqx.field(static=True)  # Baseline dt for a 1-NFE-per-step solver
    solve_n_steps: int = eqx.field(static=True)
    solve_dt: float = eqx.field(static=True)
    nfe_per_step: int = eqx.field(static=True)
    n_save: int = eqx.field(
        static=True
    )  # Number of time points to save in the output path
    solver: AbstractERK | AbstractReversibleSolver = eqx.field(static=True)
    save_path: bool = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        n_steps: int,
        dt: float,
        solver: AbstractERK | AbstractReversibleSolver,
        diffusion_scale: float,
        *,
        n_save: int | None = None,
        save_path: bool = False,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)

        self.name = "simple_neural_sde"
        self.state_dim = state_dim
        self.n_steps = n_steps
        self.dt = dt
        self.solver = solver
        self.save_path = save_path

        self.nfe_per_step = _solver_nfe_per_step(solver)
        if n_steps % self.nfe_per_step != 0:
            raise ValueError(
                f"Fixed-NFE mode requires n_steps={n_steps} to be divisible by "
                f"{self.nfe_per_step} for solver {type(solver).__name__}."
            )
        self.solve_n_steps = n_steps // self.nfe_per_step
        self.solve_dt = dt * self.nfe_per_step
        self.n_save = n_save if n_save is not None else self.solve_n_steps

        print(
            "[SimpleNeuralSDE] "
            f"solver={type(solver).__name__} "
            f"nfe_budget={self.n_steps} "
            f"nfe_per_step={self.nfe_per_step} "
            f"solve_n_steps={self.solve_n_steps} "
            f"required_dt={self.solve_dt:.8g}"
        )

        self.drift_field = DriftField(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            key=k1,
        )
        self.diffusion_field = DiffusionField(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            diffusion_scale=diffusion_scale,
            key=k2,
        )

    def __call__(self, y0, key):
        """Args:
        y0: (state_dim,) initial state
        key: PRNG key

        Returns:
        save_path=False: (state_dim,) final state
        save_path=True:  (n_save, state_dim) full path
        """
        t1 = self.n_steps * self.dt
        brownian_path = VirtualBrownianTree(
            t0=0.0,
            t1=t1,
            tol=self.solve_dt / 4.0,
            shape=(self.state_dim,),
            key=key,
        )

        term = MultiTerm(
            ODETerm(self.drift_field),
            ControlTerm(self.diffusion_field, brownian_path),
        )

        adjoint = ReversibleAdjoint()
        solver = (
            self.solver
            if isinstance(self.solver, AbstractReversibleSolver)
            else _PatchedUReversible(self.solver)
        )

        if self.save_path:
            ts = jnp.linspace(0.0, t1, self.n_save, dtype=jnp.float32)
            saveat = SaveAt(ts=ts)
        else:
            saveat = SaveAt(t1=True)

        sol = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t1,
            dt0=self.solve_dt,
            y0=y0,
            args=None,
            saveat=saveat,
            adjoint=adjoint,
            max_steps=self.solve_n_steps + 8,
        )
        assert sol.ys is not None

        if self.save_path:
            return sol.ys  # (solve_n_steps, state_dim)
        return sol.ys[0]  # (state_dim,)
