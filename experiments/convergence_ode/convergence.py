"""CF-EES ODE convergence on SO(3), using Diffrax/georax and shared plots.

Run from the repository root:
    python -m experiments.convergence_ode.convergence
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from georax import AbstractCommutatorFreeSolver, CFEES25, CFEES27, GeometricTerm, SO
from georax._solver.commutator_free import CommutatorFreeTableau

from .plotting import plot_error_curve
from experiments.plotting import set_plotting_params


def hat(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


A = hat((0.4, -0.2, 0.7))
B = hat((-0.3, 0.8, 0.1))


def frame_coeffs(t, rotation, args):
    del t, args
    # SO(3) uses right multiplication and upper-triangular skew coordinates.
    generator = rotation.T @ jnp.asarray(A) @ rotation + jnp.asarray(B)
    return generator[jnp.triu_indices(3, k=1)]


def exact_solution(ts):
    return np.stack([expm(float(t) * A) @ expm(float(t) * B) for t in ts])


def embedded_solver(principal):
    """Express the final-increment companion with georax's CF tableau engine.

    The main trajectories use CFEES25/CFEES27 directly. This tableau is used
    only for the single-step embedded diagnostic and shares their generators.
    """
    recurrence = principal.recurrence
    stages = recurrence.num_stages
    a = np.r_[0.0, recurrence.A]
    b = np.asarray(recurrence.B)
    beta = np.zeros((stages, stages))
    delta = np.zeros(stages)
    for i in range(stages):
        delta *= a[i]
        delta[i] += 1.0
        beta[i] = b[i] * delta
    tableau = CommutatorFreeTableau(
        c=tuple(recurrence.C),
        stage_exps=tuple(
            tuple(beta[j, :i].copy() for j in range(i)) for i in range(stages)
        ),
        final_exps=tuple(row.copy() for row in beta),
        embedded_final_exps=(delta / delta.sum(),),
    )

    class FinalIncrementPair(AbstractCommutatorFreeSolver):
        def order(self, terms):
            return 2

        def error_order(self, terms):
            return 2

        def antisymmetric_order(self, terms):
            return principal.antisymmetric_order(terms)

    FinalIncrementPair.tableau = tableau
    return FinalIncrementPair()


def make_runner(solver, steps, terminal_time):
    geometry = SO(3)
    term = GeometricTerm(frame_coeffs, geometry=geometry)
    h = terminal_time / steps
    ts = jnp.linspace(0.0, terminal_time, steps + 1)

    @jax.jit
    def run(y0):
        forward = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=terminal_time, dt0=h, y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.ConstantStepSize(),
            adjoint=diffrax.ReversibleAdjoint(), max_steps=steps + 4,
        )
        reverse = diffrax.diffeqsolve(
            term, solver, t0=terminal_time, t1=0.0, dt0=-h,
            y0=forward.ys[-1], saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.ConstantStepSize(),
            adjoint=diffrax.ReversibleAdjoint(), max_steps=steps + 4,
        )
        return forward.ys, reverse.ys[0]

    return run


def single_step_diagnostics(solver, h, y0):
    term = GeometricTerm(frame_coeffs, geometry=SO(3))
    state = solver.init(term, 0.0, h, y0, None)
    y1, _, _, state, result = solver.step(term, 0.0, h, y0, None, state, False)
    assert result == diffrax.RESULTS.successful
    recovered, _, _, _, result = solver.step(term, h, 0.0, y1, None, state, False)
    assert result == diffrax.RESULTS.successful
    pair = embedded_solver(solver)
    pair.init(term, 0.0, h, y0, None)
    paired, discrepancy, _, _, result = pair.step(term, 0.0, h, y0, None, None, False)
    assert result == diffrax.RESULTS.successful
    # The generic CF tableau and the low-storage solver must agree.
    np.testing.assert_allclose(paired, y1, rtol=0, atol=5e-14)
    return float(jnp.linalg.norm(recovered - y0)), float(jnp.linalg.norm(discrepancy))


def resolved_fit(rows, key):
    floor = 100 * np.finfo(np.float64).eps
    resolved = [row for row in rows if row[key] > floor]
    # Use the finest three resolved values, and report the chosen grid.
    selected = resolved[-3:]
    if len(selected) < 2:
        return {"slope": None, "steps": [row["steps"] for row in selected]}
    slope = np.polyfit(
        np.log10([row["h"] for row in selected]),
        np.log10([row[key] for row in selected]), 1,
    )[0]
    return {"slope": float(slope), "steps": [row["steps"] for row in selected]}


def plot_results(rows, output_dir):
    """Render saved measurements with the shared convergence plotting function."""
    set_plotting_params(9, 10, 12)
    fig, axes = plt.subplots(2, 2, figsize=(10 * (2 / 3), 4))
    for index, (stages, order) in enumerate([(3, 5), (4, 7)]):
        method_rows = [row for row in rows if row["stages"] == stages]
        for col, (key, rate) in enumerate([("forward", 2), ("recovery", order)]):
            errors = np.array([row[key] for row in method_rows])
            hs = np.array([row["h"] for row in method_rows])
            plot_error_curve(
                hs, np.log10(np.maximum(errors, np.finfo(float).tiny)), rate,
                axes[index, col], backward=bool(col),
                reference_mask=errors > 100 * np.finfo(float).eps,
            )
            symbol = r"\overleftarrow{\mathcal{E}}(h)" if col else r"\mathcal{E}(h)"
            axes[index, col].set_title(rf"${symbol}$ for $\mathrm{{CF\text{{-}}EES}}(2,{order})$")
    fig.tight_layout()
    fig.savefig(output_dir / "cfees_ode_convergence.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-power", type=int, default=1)
    parser.add_argument("--max-power", type=int, default=7)
    parser.add_argument("--terminal-time", type=float, default=1.0)
    parser.add_argument("--plot-only", action="store_true", help="Redraw the saved CSV without solving.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.min_power < 0 or args.max_power < args.min_power or args.terminal_time <= 0:
        parser.error("Use 0 <= min-power <= max-power and a positive terminal time.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with (args.output_dir / "ode_convergence.csv").open(newline="") as stream:
            rows = [{key: (int(value) if key in {"stages", "steps"} else float(value))
                     for key, value in row.items()} for row in csv.DictReader(stream)]
        plot_results(rows, args.output_dir)
        return
    y0 = jnp.eye(3, dtype=jnp.float64)
    # Check the action convention against the exact differential equation.
    term = GeometricTerm(frame_coeffs, geometry=SO(3))
    rotation = jnp.asarray(exact_solution([0.37])[0])
    np.testing.assert_allclose(
        term.geometry.detrivialise(rotation, frame_coeffs(0.37, rotation, None)),
        A @ rotation + rotation @ B, rtol=0, atol=2e-15,
    )
    rows, methods = [], {}
    for solver in [CFEES25(), CFEES27()]:
        order = solver.antisymmetric_order(None)
        method_rows = []
        for power in range(args.min_power, args.max_power + 1):
            steps = 2**power
            ts = np.linspace(0.0, args.terminal_time, steps + 1)
            trajectory, recovered = make_runner(solver, steps, args.terminal_time)(y0)
            trajectory, recovered = np.asarray(trajectory), np.asarray(recovered)
            h = args.terminal_time / steps
            local, embedded = single_step_diagnostics(solver, h, y0)
            row = {
                "stages": solver.recurrence.num_stages, "steps": steps, "h": h,
                "forward": float(np.linalg.norm(trajectory - exact_solution(ts), axis=(1, 2)).max()),
                "recovery": float(np.linalg.norm(recovered - np.eye(3))),
                "local_recovery": local, "embedded": embedded,
                "orthogonality": float(np.linalg.norm(
                    trajectory.swapaxes(1, 2) @ trajectory - np.eye(3), axis=(1, 2)
                ).max()),
            }
            assert all(np.isfinite(row[key]) for key in row)
            assert row["orthogonality"] < 1e-11
            method_rows.append(row)
            rows.append(row)
        fits = {key: resolved_fit(method_rows, key) for key in
                ["forward", "recovery", "local_recovery", "embedded"]}
        methods[str(order)] = fits
        print(f"CF-EES(2,{order}): {json.dumps(fits)}", flush=True)
    with (args.output_dir / "ode_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "terminal_time": args.terminal_time,
        "powers": [args.min_power, args.max_power],
        "arithmetic": "float64", "platform": jax.default_backend(),
        "python": platform.python_version(), "jax": jax.__version__,
        "diffrax": diffrax.__version__, "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "action": "georax SO(3), degree-eight Taylor exponential plus QR",
        "resolved_error_threshold": float(100 * np.finfo(float).eps),
        "methods": methods,
    }
    (args.output_dir / "ode_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(rows, args.output_dir)


if __name__ == "__main__":
    main()
