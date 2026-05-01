"""Count drift/diffusion calls for the fixed-NFE SDE solvers."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import warnings
from dataclasses import dataclass
from typing import Literal

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp

from experiments.stochastic_volatility.experiment.config import Solvers
from experiments.stochastic_volatility.models.nsde import SimpleNeuralSDE


Mode = Literal["forward", "grad"]


@dataclass
class CallCounter:
    f: int = 0
    g: int = 0

    def inc_f(self) -> None:
        self.f += 1

    def inc_g(self) -> None:
        self.g += 1


class CountingDrift(eqx.Module):
    field: eqx.Module
    counter: CallCounter = eqx.field(static=True)

    def __call__(self, t, y, args):
        jax.debug.callback(self.counter.inc_f, ordered=True)
        return self.field(t, y, args)


class CountingDiffusion(eqx.Module):
    field: eqx.Module
    counter: CallCounter = eqx.field(static=True)

    def __call__(self, t, y, args):
        jax.debug.callback(self.counter.inc_g, ordered=True)
        return self.field(t, y, args)


def _with_counted_fields(
    model: SimpleNeuralSDE,
    counter: CallCounter,
) -> SimpleNeuralSDE:
    model = eqx.tree_at(
        lambda m: m.drift_field,
        model,
        CountingDrift(model.drift_field, counter),
    )
    return eqx.tree_at(
        lambda m: m.diffusion_field,
        model,
        CountingDiffusion(model.diffusion_field, counter),
    )


def _run_once(model: SimpleNeuralSDE, mode: Mode) -> jax.Array:
    y0 = jnp.ones((model.state_dim,), dtype=jnp.float32)
    key = jax.random.key(123)

    if mode == "forward":
        return model(y0, key)

    def loss_fn(current_model):
        ys = current_model(y0, key)
        return jnp.sum(ys**2)

    value, _ = eqx.filter_value_and_grad(loss_fn)(model)
    return value


def _count_solver(solver: Solvers, *, nfe_budget: int, mode: Mode) -> dict[str, int]:
    counter = CallCounter()
    with contextlib.redirect_stdout(io.StringIO()):
        model = SimpleNeuralSDE(
            state_dim=1,
            hidden_dim=4,
            n_steps=nfe_budget,
            dt=1.0 / nfe_budget,
            solver=solver.build(),
            diffusion_scale=0.2,
            n_save=8,
            save_path=True,
            key=jax.random.key(0),
        )
    model = _with_counted_fields(model, counter)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*is not marked as converging to either the Itô or the Stratonovich solution\.",
        )
        out = _run_once(model, mode)
    jax.block_until_ready(out)

    expected = model.nfe_per_step * model.solve_n_steps
    return {
        "nfe_per_step": model.nfe_per_step,
        "solve_n_steps": model.solve_n_steps,
        "expected_forward": expected,
        "f": counter.f,
        "g": counter.g,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfe-budget", type=int, default=504)
    parser.add_argument(
        "--mode",
        choices=["forward", "grad"],
        default="forward",
        help="Count one forward solve, or one scalar loss+gradient evaluation.",
    )
    args = parser.parse_args()

    solvers = [
        Solvers.REVERSIBLE_HEUN,
        Solvers.MCF_EULER,
        Solvers.MCF_MIDPOINT,
        Solvers.EES25,
    ]

    print(
        "solver,nfe_per_step,solve_n_steps,expected_forward,f_calls,g_calls",
        flush=True,
    )
    for solver in solvers:
        counts = _count_solver(solver, nfe_budget=args.nfe_budget, mode=args.mode)
        print(
            f"{solver},"
            f"{counts['nfe_per_step']},"
            f"{counts['solve_n_steps']},"
            f"{counts['expected_forward']},"
            f"{counts['f']},"
            f"{counts['g']}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
