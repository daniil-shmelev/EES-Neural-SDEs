"""Quick wall-clock parity check across hero variants at M5 config.

For each variant (solver, adjoint, model), runs `n_warmup` warmup +
`n_reps` measurement passes of one forward+backward of the multi-horizon
energy-score loss on a synthetic batch, and reports the mean wall-clock
plus the ratio versus the slowest variant.

If any variant is more than `--max-ratio` slower than the slowest, the
script suggests an n_steps adjustment to equalize wall-clock. (We only
SUGGEST — adjusting also changes accuracy, so the user picks the
trade-off.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# (label, model_kind, solver, adjoint, stages_for_nfe)
VARIANTS = [
    ("cfees25_reversible", "kuramoto_nsde",      "cfees25", "reversible",           3),
    ("cg2_treeverse",      "kuramoto_nsde",      "cg2",     "checkpoint_recursive", 2),
    ("euclidean_baseline", "euclidean_baseline", None,      "direct",               2),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--nfe-budget", type=int, default=1500)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--max-ratio", type=float, default=1.30,
                   help="Tolerable wall-clock ratio between fastest and slowest "
                        "variant before suggesting an n_steps adjustment.")
    p.add_argument("--n-samples", type=int, default=4,
                   help="MC samples per energy-score eval (matches loss config).")
    p.add_argument("--output", type=Path,
                   default=Path("experiments/kuramoto/results/parity_check_N1000.json"))
    return p.parse_args()


def build_model(kind: str, N: int, hidden_dim: int, n_steps: int, dt: float,
                solver_name: str | None, adjoint_name: str, key):
    from experiments.kuramoto.experiment.factories import build_adjoint, build_solver
    from experiments.kuramoto.experiment.config import Solvers
    if kind == "kuramoto_nsde":
        from experiments.kuramoto.models.kuramoto_nsde import KuramotoNSDE
        solver = build_solver(Solvers(solver_name))
        return KuramotoNSDE(
            N=N, hidden_dim=hidden_dim, n_steps=n_steps, dt=dt, n_obs=64,
            solver=solver, diffusion_scale=0.3,
            adjoint=build_adjoint(adjoint_name, n_steps),
            key=key,
        )
    if kind == "euclidean_baseline":
        from experiments.kuramoto.models.euclidean_baseline import EuclideanKuramotoNSDE
        return EuclideanKuramotoNSDE(
            N=N, hidden_dim=hidden_dim, n_steps=n_steps, dt=dt, n_obs=64,
            diffusion_scale=0.3,
            adjoint=build_adjoint(adjoint_name, n_steps),
            key=key,
        )
    raise ValueError(f"unknown model kind {kind!r}")


def benchmark_one(N: int, batch_size: int, hidden_dim: int, T: float,
                  n_steps: int, n_samples: int, n_warmup: int, n_reps: int,
                  model_kind: str, solver_name: str | None,
                  adjoint_name: str, seed: int = 0) -> dict:
    dt = T / n_steps
    model = build_model(
        model_kind, N, hidden_dim, n_steps, dt, solver_name, adjoint_name,
        key=jr.PRNGKey(seed),
    )
    n_params = sum(int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))

    theta0 = jr.uniform(jr.PRNGKey(seed + 1), (batch_size, N),
                         minval=-jnp.pi, maxval=jnp.pi)
    omega0 = 0.5 * jr.normal(jr.PRNGKey(seed + 2), (batch_size, N))
    sample_keys = jr.split(jr.PRNGKey(seed + 3), n_samples * batch_size).reshape(
        n_samples, batch_size, -1
    )

    @eqx.filter_jit
    def loss_grad(model):
        def per_traj(model, th0, om0, k):
            th, om = model(th0, om0, k)
            return jnp.mean(th ** 2 + om ** 2)
        def loss(model):
            def one_sample(sk):
                return jax.vmap(lambda th, om, k: per_traj(model, th, om, k))(
                    theta0, omega0, sk)
            sample_losses = jax.vmap(one_sample)(sample_keys)
            return jnp.mean(sample_losses)
        return eqx.filter_grad(loss)(model)

    for _ in range(n_warmup):
        g = loss_grad(model)
        jax.block_until_ready(g)

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        g = loss_grad(model)
        jax.block_until_ready(g)
        times.append(time.perf_counter() - t0)

    return {
        "n_steps": n_steps, "n_params": n_params,
        "wall_s_mean": float(sum(times) / len(times)),
        "wall_s_min": float(min(times)),
        "wall_s_max": float(max(times)),
        "wall_s_std": float(
            (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5
        ),
    }


def main() -> int:
    args = parse_args()
    print(f"\n[parity] N={args.N}, batch={args.batch_size}, hidden={args.hidden_dim}, "
          f"T={args.T}, nfe_budget={args.nfe_budget}, n_samples={args.n_samples}",
          flush=True)
    print(f"[parity] {args.n_warmup} warmup + {args.n_reps} timed reps per cell\n",
          flush=True)

    rows = []
    for label, kind, solver, adj, stages in VARIANTS:
        n_steps = max(1, args.nfe_budget // stages)
        print(f"[parity] benchmarking {label}  (n_steps={n_steps}, NFE={n_steps * stages})",
              flush=True)
        try:
            res = benchmark_one(
                N=args.N, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
                T=args.T, n_steps=n_steps, n_samples=args.n_samples,
                n_warmup=args.n_warmup, n_reps=args.n_reps,
                model_kind=kind, solver_name=solver, adjoint_name=adj,
            )
            rows.append({"label": label, "stages": stages, **res})
            print(f"[parity]   wall = {res['wall_s_mean']:.3f} ± {res['wall_s_std']:.3f}s "
                  f"(min {res['wall_s_min']:.3f}, max {res['wall_s_max']:.3f})\n",
                  flush=True)
        except Exception as exc:
            rows.append({"label": label, "stages": stages, "error": str(exc)[-300:]})
            print(f"[parity]   ERROR: {exc}\n", flush=True)

    ok = [r for r in rows if "wall_s_mean" in r]
    if not ok:
        print("[parity] no successful cells; skipping ratio analysis")
    else:
        slowest = max(ok, key=lambda r: r["wall_s_mean"])
        print("\n[parity] === wall-clock summary ===")
        print(f"[parity] slowest variant: {slowest['label']} "
              f"({slowest['wall_s_mean']:.3f}s, n_steps={slowest['n_steps']})")
        print(f"[parity] {'label':<22} {'n_steps':>7} {'wall_s':>9} {'ratio':>7} "
              f"{'sugg_n_steps_for_parity':>28}")
        for r in ok:
            ratio = r["wall_s_mean"] / slowest["wall_s_mean"]
            # If we want this variant to take the same wall as the slowest,
            # multiply n_steps by 1/ratio (assumes wall_clock is linear in n_steps).
            sugg = int(round(r["n_steps"] / ratio)) if ratio > 0 else r["n_steps"]
            r["wall_ratio_vs_slowest"] = ratio
            r["suggested_n_steps_for_wall_parity"] = sugg
            r["suggested_nfe_for_wall_parity"] = sugg * r["stages"]
            print(f"[parity] {r['label']:<22} {r['n_steps']:>7} "
                  f"{r['wall_s_mean']:>9.3f} {ratio:>7.3f} {sugg:>28}")

        gap = max(r["wall_ratio_vs_slowest"] for r in ok) - min(
            r["wall_ratio_vs_slowest"] for r in ok
        )
        if (1.0 / min(r["wall_ratio_vs_slowest"] for r in ok)) > args.max_ratio:
            print(f"\n[parity] WARNING: wall-clock spread exceeds {args.max_ratio:.2f}× "
                  f"(slowest/fastest = {1.0/min(r['wall_ratio_vs_slowest'] for r in ok):.3f}). "
                  "Consider passing --n-steps overrides to run_hero_training.py "
                  "to equalize wall-clock instead of NFE.")
        else:
            print(f"\n[parity] OK: wall-clock spread within {args.max_ratio:.2f}× tolerance.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": vars(args) | {"output": str(args.output)},
        "rows": rows,
    }, indent=2, default=str))
    print(f"\n[parity] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
