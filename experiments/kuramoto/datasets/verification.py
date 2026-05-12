r"""Quality diagnostics for the stochastic 2nd-order Kuramoto simulator.

Two checks, both used by the M1 orchestrator before any production data
generation runs:

1. **Analytic two-oscillator phase-locked steady state** (deterministic
   limit, $D = 0$). For the bimodal model with $N = 2$, $\Omega_1 = +P$,
   $\Omega_2 = -P$, $K > 2P$, the deterministic dynamics admit a unique
   stable phase-locked equilibrium with phase difference

   .. math:: \Delta\theta_\infty = \arcsin(2P / K).

   We integrate at several `n_fine` values and report the relative error
   of the long-time phase difference vs the analytic value. This isolates
   integrator discretisation error from any noise effects.

2. **Order-parameter stationarity** (full SDE, $D > 0$). For a moderate
   $N$ ensemble the Kuramoto order parameter $r(t) = |N^{-1}\sum_j
   e^{i\theta_j}|$ should fluctuate around a stationary mean determined
   by the $K / D$ ratio. Long-running mean and standard deviation should
   not blow up.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.kuramoto.datasets.kuramoto import KuramotoParams, order_parameter
from experiments.kuramoto.datasets.simulator import (
    SimConfig,
    sample_initial_conditions,
    simulate_one,
)


def _order_parameter_trace(theta_traj: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda th: order_parameter(th)[0])(theta_traj)


def two_oscillator_phase_lock_study(
    P: float,
    K: float,
    m: float,
    T: float,
    n_fine_list: list[int],
    key: jr.PRNGKeyArray,
) -> tuple[dict[int, dict], float]:
    r"""Deterministic ($D = 0$) study of the 2-oscillator phase lock.

    For $\Omega_{1,2} = \pm P$ and $K > 2P$, the stable phase-locked
    equilibrium has $\Delta\theta = \arcsin(2P/K)$. Returns per-`n_fine`
    metrics and the analytic target.
    """
    if K <= 2.0 * P:
        raise ValueError(
            f"two_oscillator_phase_lock requires K > 2P (got K={K}, P={P})."
        )
    target = float(jnp.arcsin(2.0 * P / K))
    params = KuramotoParams.bimodal(N=2, P=P, K=K, m=m)
    theta0 = jr.uniform(key, (2,), minval=-jnp.pi, maxval=jnp.pi)
    omega0 = jnp.zeros(2)

    results: dict[int, dict] = {}
    for n_fine in n_fine_list:
        cfg = SimConfig(T=T, n_fine=n_fine, n_obs=min(n_fine + 1, 256), D=0.0)
        theta_traj, _ = simulate_one(params, cfg, theta0, omega0, jr.PRNGKey(123))
        delta = float(theta_traj[-1, 0] - theta_traj[-1, 1])
        delta_wrapped = float(jnp.mod(delta + jnp.pi, 2.0 * jnp.pi) - jnp.pi)
        rel_err = abs(abs(delta_wrapped) - target) / max(target, 1e-12)
        results[n_fine] = {
            "dt": T / n_fine,
            "delta_theta_final": delta_wrapped,
            "delta_theta_target": target,
            "relative_error": rel_err,
        }
        print(
            f"[verify] n_fine={n_fine:>6d} dt={T/n_fine:.3e} "
            f"|delta_theta|={abs(delta_wrapped):.5f} target={target:.5f} "
            f"rel_err={rel_err:.3e}",
            flush=True,
        )
    return results, target


def stationarity_study(
    params: KuramotoParams,
    cfg: SimConfig,
    n_traj: int,
    omega_scale: float,
    key: jr.PRNGKeyArray,
) -> dict:
    r"""Ensemble run at $D > 0$: check that the order parameter $r(t)$ stays bounded.

    Reports per-time mean / std of $r$ across the trajectory ensemble, plus
    the time-averaged stationary value at the tail.
    """
    k_ic, k_sim = jr.split(key)
    theta0, omega0 = sample_initial_conditions(params, n_traj, omega_scale, k_ic)
    sim_keys = jr.split(k_sim, n_traj)

    th_traj, _ = jax.vmap(simulate_one, in_axes=(None, None, 0, 0, 0))(
        params, cfg, theta0, omega0, sim_keys
    )
    r_per_t = jax.vmap(_order_parameter_trace)(th_traj)  # (B, T)
    return {
        "r_mean_per_t": np.asarray(jnp.mean(r_per_t, axis=0)).tolist(),
        "r_std_per_t": np.asarray(jnp.std(r_per_t, axis=0)).tolist(),
        "r_stationary_mean": float(jnp.mean(r_per_t[:, r_per_t.shape[1] // 2:])),
        "r_stationary_std": float(jnp.std(r_per_t[:, r_per_t.shape[1] // 2:])),
    }


def run_verification(
    N: int,
    P: float,
    K: float,
    m: float,
    D: float,
    T: float,
    n_fine_grid: list[int],
    n_traj_stationarity: int,
    omega_scale: float,
    seed: int,
    out_dir: Path,
) -> Path:
    """Top-level driver. Produces `simulator_verification.{json,npz}`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key = jr.PRNGKey(seed)
    k_lock, k_stat = jr.split(key)

    print(
        f"\n[verify] === Two-oscillator phase-lock study (P={P}, K={K}, T={T}) ==="
    )
    t0 = time.perf_counter()
    if K > 2.0 * P:
        lock_results, target = two_oscillator_phase_lock_study(
            P, K, m, T, n_fine_grid, k_lock
        )
    else:
        print(
            f"[verify] K={K} <= 2P={2*P}: phase lock not predicted, skipping study."
        )
        lock_results, target = {}, float("nan")
    print(f"[verify] phase-lock study took {time.perf_counter() - t0:.1f}s")

    print(
        f"[verify] === Order-parameter stationarity (N={N}, D={D}, "
        f"n_traj={n_traj_stationarity}) ==="
    )
    cfg_stat = SimConfig(T=T, n_fine=max(n_fine_grid), n_obs=64, D=D)
    params = KuramotoParams.bimodal(N=N, P=P, K=K, m=m)
    t0 = time.perf_counter()
    stat = stationarity_study(params, cfg_stat, n_traj_stationarity, omega_scale, k_stat)
    print(
        f"[verify] stationarity study took {time.perf_counter() - t0:.1f}s; "
        f"r_stat={stat['r_stationary_mean']:.3f} +/- {stat['r_stationary_std']:.3f}"
    )

    summary = {
        "N": N, "P": P, "K": K, "m": m, "D": D, "T": T,
        "phase_lock_target": target,
        "phase_lock": {str(k): v for k, v in lock_results.items()},
        "stationarity": {
            "r_stationary_mean": stat["r_stationary_mean"],
            "r_stationary_std": stat["r_stationary_std"],
        },
    }
    json_path = out_dir / "simulator_verification.json"
    json_path.write_text(json.dumps(summary, indent=2))
    npz_path = out_dir / "simulator_verification.npz"
    np.savez(
        npz_path,
        n_fine_grid=np.array(list(lock_results.keys()) if lock_results else []),
        delta_theta_target=target,
        delta_theta_finals=np.array(
            [lock_results[k]["delta_theta_final"] for k in lock_results]
        ),
        r_mean_per_t=np.array(stat["r_mean_per_t"]),
        r_std_per_t=np.array(stat["r_std_per_t"]),
    )
    print(f"[verify] wrote {json_path}")
    print(f"[verify] wrote {npz_path}")
    return json_path
