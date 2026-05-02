"""Quality diagnostics for the pendulum SDE simulator.

Two functions, both used by the M1 orchestrator before any production data
generation runs:

- `hamiltonian_drift_study` — at $\\sigma = 0$, $\\gamma = 0$, compares
  diffrax `Heun` against a symplectic-Verlet reference at several step
  sizes. Reveals how fine $dt$ must be for energy drift to be acceptable.
- `stationarity_study` — at $\\sigma > 0$, $\\gamma > 0$, runs an ensemble
  of trajectories and reports per-time energy mean/std. The mean should
  saturate to the stationary value rather than blowing up.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.pendulum.datasets.lagrangian import (
    PendulumParams,
    hamiltonian,
    total_energy_velocity,
    velocity_to_momentum,
)
from experiments.pendulum.datasets.simulator import (
    SimConfig,
    chaos_onset_energy_scale,
    sample_initial_conditions,
    simulate_one,
    verlet_rollout,
)


def _energy_trace_from_velocity(
    theta_traj: jnp.ndarray, omega_traj: jnp.ndarray, params: PendulumParams
) -> jnp.ndarray:
    p_traj = jax.vmap(lambda th, om: velocity_to_momentum(th, om, params))(
        theta_traj, omega_traj
    )
    return jax.vmap(lambda th, p_: hamiltonian(th, p_, params))(theta_traj, p_traj)


def _energy_trace_from_momentum(
    theta_traj: jnp.ndarray, p_traj: jnp.ndarray, params: PendulumParams
) -> jnp.ndarray:
    return jax.vmap(lambda th, p_: hamiltonian(th, p_, params))(theta_traj, p_traj)


def hamiltonian_drift_study(
    params: PendulumParams,
    T: float,
    n_fine_list: list[int],
    key: jr.PRNGKeyArray,
) -> tuple[dict[int, dict], float]:
    """Run a single deterministic trajectory at each `n_fine`, return drift metrics."""
    n = params.n
    theta0 = jr.uniform(key, (n,), minval=-jnp.pi, maxval=jnp.pi)
    p0 = jr.normal(jr.fold_in(key, 1), (n,)) * 0.5
    H0 = float(hamiltonian(theta0, p0, params))

    results: dict[int, dict] = {}
    for n_fine in n_fine_list:
        dt = T / n_fine
        cfg = SimConfig(T=T, n_fine=n_fine, n_obs=min(n_fine + 1, 256), sigma=0.0, gamma=0.0)

        th_heun, om_heun = simulate_one(params, cfg, theta0, p0, jr.PRNGKey(123))
        H_heun = np.asarray(_energy_trace_from_velocity(th_heun, om_heun, params))
        max_drift_heun = float(np.max(np.abs(H_heun - H0)) / max(abs(H0), 1e-12))

        th_v, p_v = verlet_rollout(theta0, p0, dt=dt, n_steps=n_fine, params=params)
        idx = np.linspace(0, n_fine, th_heun.shape[0]).astype(int)
        H_v = np.asarray(_energy_trace_from_momentum(th_v[idx], p_v[idx], params))
        max_drift_verlet = float(np.max(np.abs(H_v - H0)) / max(abs(H0), 1e-12))

        results[n_fine] = {
            "dt": dt,
            "max_drift_heun_rel": max_drift_heun,
            "max_drift_verlet_rel": max_drift_verlet,
            "H_heun": H_heun.tolist(),
            "H_verlet": H_v.tolist(),
        }
        print(
            f"[verify] n_fine={n_fine:>6d} dt={dt:.3e} "
            f"|H-H0|/|H0|: Heun={max_drift_heun:.3e} Verlet={max_drift_verlet:.3e}",
            flush=True,
        )
    return results, H0


def stationarity_study(
    params: PendulumParams,
    cfg: SimConfig,
    n_traj: int,
    key: jr.PRNGKeyArray,
) -> dict:
    """Ensemble run at $\\sigma > 0, \\gamma > 0$ — report per-time energy stats."""
    energy_scale = float(chaos_onset_energy_scale(params))
    k_ic, k_sim = jr.split(key)
    theta0, p0 = sample_initial_conditions(
        params, n_traj, 0.5 * energy_scale, 1.5 * energy_scale, k_ic
    )
    sim_keys = jr.split(k_sim, n_traj)

    th_traj, om_traj = jax.vmap(simulate_one, in_axes=(None, None, 0, 0, 0))(
        params, cfg, theta0, p0, sim_keys
    )
    energy_per_t = jax.vmap(
        lambda th_b, om_b: jax.vmap(lambda th, om: total_energy_velocity(th, om, params))(
            th_b, om_b
        )
    )(th_traj, om_traj)
    return {
        "energy_mean_per_t": np.asarray(jnp.mean(energy_per_t, axis=0)).tolist(),
        "energy_std_per_t": np.asarray(jnp.std(energy_per_t, axis=0)).tolist(),
        "energy_final_distribution": np.asarray(energy_per_t[:, -1]).tolist(),
        "energy_scale": energy_scale,
    }


def run_verification(
    n: int,
    T: float,
    sigma: float,
    gamma: float,
    n_fine_grid: list[int],
    n_traj_stationarity: int,
    seed: int,
    out_dir: Path,
) -> Path:
    """Top-level driver. Produces `simulator_verification_n{n}.{json,npz}`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    params = PendulumParams.uniform(n=n)
    key = jr.PRNGKey(seed)
    k_ham, k_stat = jr.split(key)

    print(f"\n[verify] === Hamiltonian drift study (n={n}, T={T}) ===")
    t0 = time.perf_counter()
    drift_results, H0 = hamiltonian_drift_study(params, T, n_fine_grid, k_ham)
    print(f"[verify] drift study took {time.perf_counter() - t0:.1f}s")

    print(
        f"[verify] === Stationarity (sigma={sigma}, gamma={gamma}, "
        f"n_traj={n_traj_stationarity}) ==="
    )
    cfg = SimConfig(
        T=T, n_fine=max(n_fine_grid), n_obs=64, sigma=sigma, gamma=gamma
    )
    t0 = time.perf_counter()
    stat = stationarity_study(params, cfg, n_traj_stationarity, k_stat)
    print(f"[verify] stationarity study took {time.perf_counter() - t0:.1f}s")

    summary = {
        "n": n, "T": T, "H0": H0,
        "drift": {
            str(k): {kk: vv for kk, vv in v.items() if kk not in ("H_heun", "H_verlet")}
            for k, v in drift_results.items()
        },
        "stationarity": {k: v for k, v in stat.items() if k != "energy_final_distribution"},
    }
    json_path = out_dir / f"simulator_verification_n{n}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    npz_path = out_dir / f"simulator_verification_n{n}.npz"
    np.savez(
        npz_path,
        n_fine_grid=np.array(n_fine_grid),
        H0=H0,
        H_heun_traces=np.array([drift_results[k]["H_heun"] for k in n_fine_grid], dtype=object),
        H_verlet_traces=np.array([drift_results[k]["H_verlet"] for k in n_fine_grid], dtype=object),
        energy_mean_per_t=np.array(stat["energy_mean_per_t"]),
        energy_std_per_t=np.array(stat["energy_std_per_t"]),
        energy_final_distribution=np.array(stat["energy_final_distribution"]),
    )
    print(f"[verify] wrote {json_path}")
    print(f"[verify] wrote {npz_path}")
    return json_path
