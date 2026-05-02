"""Reusable simulation + persistence pipeline for the pendulum dataset.

Wraps the lower-level `simulator.py` calls in a JIT-compiled batched form
and a save-to-disk helper, so the CLI orchestrator stays thin.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.pendulum.datasets.lagrangian import PendulumParams
from experiments.pendulum.datasets.simulator import (
    SimConfig,
    chaos_onset_energy_scale,
    sample_initial_conditions,
    simulate_one,
)


@eqx.filter_jit
def simulate_batch_jit(
    params: PendulumParams,
    cfg: SimConfig,
    theta0: jnp.ndarray,
    p0: jnp.ndarray,
    keys: jr.PRNGKeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled batched simulator.

    Compilation key is `(params, cfg, batch_axis_size)` — recompiles when any
    of these change. The Python-level chunked loop in `simulate_split` keeps
    the batch axis size constant except for a possible smaller final chunk.
    """
    return jax.vmap(simulate_one, in_axes=(None, None, 0, 0, 0))(
        params, cfg, theta0, p0, keys
    )


def simulate_split(
    params: PendulumParams,
    cfg: SimConfig,
    n_traj: int,
    energy_low: float,
    energy_high: float,
    key: jr.PRNGKeyArray,
    batch_sim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample initial conditions and simulate `n_traj` trajectories in chunks of `batch_sim`."""
    k_ic, k_sim = jr.split(key)
    theta0, p0 = sample_initial_conditions(params, n_traj, energy_low, energy_high, k_ic)
    sim_keys = jr.split(k_sim, n_traj)

    theta_chunks: list[np.ndarray] = []
    omega_chunks: list[np.ndarray] = []
    for start in range(0, n_traj, batch_sim):
        end = min(start + batch_sim, n_traj)
        th_b, om_b = simulate_batch_jit(
            params, cfg, theta0[start:end], p0[start:end], sim_keys[start:end]
        )
        theta_chunks.append(np.asarray(th_b))
        omega_chunks.append(np.asarray(om_b))
    return np.concatenate(theta_chunks, axis=0), np.concatenate(omega_chunks, axis=0)


def library_versions() -> dict[str, str]:
    """Snapshot of the modules whose updates would invalidate cached data."""
    versions: dict[str, str] = {}
    for name in ("jax", "jaxlib", "diffrax", "equinox"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "missing"
    return versions


def save_split(
    out_path: Path,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    t_grid: np.ndarray,
    meta: dict,
) -> None:
    """Persist the (theta, omega) splits as `.npz` plus a `.json` meta sidecar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        theta_train=splits["train"][0],
        omega_train=splits["train"][1],
        theta_val=splits["val"][0],
        omega_val=splits["val"][1],
        theta_test=splits["test"][0],
        omega_test=splits["test"][1],
        t_grid=t_grid,
    )
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def generate_one_n(
    n: int,
    cfg: SimConfig,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    batch_sim: int,
    out_dir: Path,
    render_gif: bool,
    gif_fps: int,
    gif_trail: int,
) -> Path:
    """Generate a single $n$-link dataset and persist it (with optional GIF)."""
    params = PendulumParams.uniform(n=n)
    energy_scale = float(chaos_onset_energy_scale(params))
    energy_low, energy_high = 1.0 * energy_scale, 1.5 * energy_scale

    print(
        f"\n[pipeline] === generate n={n} dt_fine={cfg.dt_fine:.3e} "
        f"E_scale={energy_scale:.2f} ==="
    )
    master = jr.PRNGKey(seed + n)
    k_train, k_val, k_test = jr.split(master, 3)

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, n_split, key in [
        ("train", n_train, k_train),
        ("val", n_val, k_val),
        ("test", n_test, k_test),
    ]:
        t0 = time.perf_counter()
        theta, omega = simulate_split(
            params, cfg, n_split, energy_low, energy_high, key, batch_sim
        )
        print(
            f"[pipeline] split={name} n={n_split} shape={theta.shape} "
            f"time={time.perf_counter() - t0:.1f}s"
        )
        splits[name] = (theta, omega)

    t_grid = np.linspace(0.0, cfg.T, cfg.n_obs)
    out_path = out_dir / f"pendulum_n{n}_seed{seed}.npz"
    meta = {
        "n": n, "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "sigma": cfg.sigma, "gamma": cfg.gamma, "T": cfg.T,
        "n_fine": cfg.n_fine, "n_obs": cfg.n_obs, "seed": seed,
        "energy_low": energy_low, "energy_high": energy_high,
        "energy_scale": energy_scale,
        "library_versions": library_versions(),
    }
    save_split(out_path, splits, t_grid, meta)
    print(f"[pipeline] wrote {out_path} ({out_path.stat().st_size / 2**20:.1f} MiB)")

    if render_gif:
        from experiments.pendulum.datasets.visualize import render_trajectory_gif

        gif_path = out_dir / f"pendulum_n{n}_seed{seed}.gif"
        render_trajectory_gif(
            theta=splits["test"][0][0],
            params=params,
            t_grid=t_grid,
            out_path=gif_path,
            fps=gif_fps,
            trail_length=gif_trail,
        )
        print(f"[pipeline] wrote {gif_path} ({gif_path.stat().st_size / 2**20:.1f} MiB)")
    return out_path
