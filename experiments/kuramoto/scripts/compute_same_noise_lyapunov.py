"""Compute the largest nontrivial same-noise Kuramoto Lyapunov exponent.

Example:
    uv run --no-project --with numpy python -m \
        experiments.kuramoto.scripts.compute_same_noise_lyapunov
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.kuramoto.datasets.lyapunov import (
    LyapunovConfig,
    estimate_largest_nontrivial_exponent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N-list", type=int, nargs="+", default=[2, 1000])
    parser.add_argument("--dt-list", type=float, nargs="+", default=[0.02, 0.01])
    parser.add_argument("--P", type=float, default=0.5)
    parser.add_argument("--K", type=float, default=2.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.05)
    parser.add_argument("--burn-in", type=float, default=50.0)
    parser.add_argument("--horizon", type=float, default=200.0)
    parser.add_argument("--renorm-interval", type=float, default=0.5)
    parser.add_argument("--n-realizations", type=int, default=16)
    parser.add_argument("--omega-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/kuramoto/results/same_noise_lyapunov.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: dict[str, object] = {
        "quantity": "largest nontrivial same-noise Lyapunov exponent",
        "units": "s^-1",
        "symmetry_handling": (
            "subtract mean delta-theta and mean delta-omega at every "
            "Benettin renormalization"
        ),
        "runs": {},
    }

    for N in args.N_list:
        n_results: dict[str, object] = {}
        for dt_index, dt in enumerate(args.dt_list):
            # Give every (N, dt) cell a deterministic independent random stream.
            cell_seed = args.seed + 1_000_003 * N + 10_007 * dt_index
            cfg = LyapunovConfig(
                N=N,
                P=args.P,
                K=args.K,
                m=args.m,
                D=args.D,
                dt=dt,
                burn_in=args.burn_in,
                horizon=args.horizon,
                renorm_interval=args.renorm_interval,
                n_realizations=args.n_realizations,
                omega_scale=args.omega_scale,
                seed=cell_seed,
            )
            start = time.perf_counter()
            estimate = estimate_largest_nontrivial_exponent(cfg)
            elapsed = time.perf_counter() - start
            estimate["wall_time_seconds"] = elapsed
            n_results[f"{dt:g}"] = estimate

            final = estimate["checkpoints_seconds"][f"{args.horizon:g}"]
            ci_low, ci_high = final["normal_95_ci"]
            print(
                f"N={N:4d} dt={dt:g}: lambda={final['mean']:+.6f} "
                f"+/- {1.96 * final['standard_error']:.6f} s^-1 "
                f"(95% CI [{ci_low:+.6f}, {ci_high:+.6f}]) "
                f"[{elapsed:.1f}s]",
                flush=True,
            )
        results["runs"][str(N)] = n_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
