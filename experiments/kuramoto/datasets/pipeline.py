"""Reusable simulation + persistence pipeline for the Kuramoto dataset.

Wraps the lower-level `simulator.py` calls in a JIT-compiled batched form
and a save-to-disk helper, so the CLI orchestrator stays thin.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.kuramoto.datasets.kuramoto import KuramotoParams, critical_coupling
from experiments.kuramoto.datasets.simulator import (
    SimConfig,
    sample_initial_conditions,
    simulate_one,
)


@eqx.filter_jit
def simulate_batch_jit(
    params: KuramotoParams,
    cfg: SimConfig,
    theta0: jnp.ndarray,
    omega0: jnp.ndarray,
    keys: jr.PRNGKeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled batched simulator.

    Compilation key is `(params, cfg, batch_axis_size)`, recompiles when any
    of these change. The Python-level chunked loop in `simulate_split` keeps
    the batch axis size constant except for a possible smaller final chunk.
    """
    return jax.vmap(simulate_one, in_axes=(None, None, 0, 0, 0))(
        params, cfg, theta0, omega0, keys
    )


def simulate_split(
    params: KuramotoParams,
    cfg: SimConfig,
    n_traj: int,
    omega_scale: float,
    key: jr.PRNGKeyArray,
    batch_sim: int,
    ckpt_dir: Path | None = None,
    split_name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample initial conditions and simulate `n_traj` trajectories in chunks of `batch_sim`.

    If ``ckpt_dir`` is given, each chunk is cached to
    ``ckpt_dir/{split_name}_chunk_NNNN.npz`` after it is simulated and reloaded
    instead of recomputed on a later call. Initial conditions and per-trajectory
    keys are deterministic in ``key``, so a resumed run reproduces identical data.
    """
    k_ic, k_sim = jr.split(key)
    theta0, omega0 = sample_initial_conditions(params, n_traj, omega_scale, k_ic)
    sim_keys = jr.split(k_sim, n_traj)

    theta_chunks: list[np.ndarray] = []
    omega_chunks: list[np.ndarray] = []
    for chunk_idx, start in enumerate(range(0, n_traj, batch_sim)):
        end = min(start + batch_sim, n_traj)
        ckpt = (
            ckpt_dir / f"{split_name}_chunk_{chunk_idx:04d}.npz"
            if ckpt_dir is not None else None
        )
        if ckpt is not None and ckpt.exists():
            with np.load(ckpt) as z:
                th_b, om_b = z["theta"], z["omega"]
        else:
            th_jax, om_jax = simulate_batch_jit(
                params, cfg, theta0[start:end], omega0[start:end], sim_keys[start:end]
            )
            th_b, om_b = np.asarray(th_jax), np.asarray(om_jax)
            if ckpt is not None:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                tmp = ckpt.parent / (ckpt.name + ".tmp")
                with open(tmp, "wb") as fh:  # file object => no auto ".npz" suffix
                    np.savez(fh, theta=th_b, omega=om_b)
                tmp.replace(ckpt)  # atomic: a partial write never looks complete
        theta_chunks.append(th_b)
        omega_chunks.append(om_b)
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
    """Persist the (theta, omega) splits as `.npz` plus a `.json` meta sidecar.

    The `.npz` is written to a temp file and atomically renamed, so a partial
    write from a killed job never leaves a corrupt-but-present output file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / (out_path.name + ".tmp")
    with open(tmp, "wb") as fh:  # file object => no auto ".npz" suffix
        np.savez(
            fh,
            theta_train=splits["train"][0],
            omega_train=splits["train"][1],
            theta_val=splits["val"][0],
            omega_val=splits["val"][1],
            theta_test=splits["test"][0],
            omega_test=splits["test"][1],
            t_grid=t_grid,
        )
    tmp.replace(out_path)
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def generate_one_n(
    N: int,
    P: float,
    K: float,
    m: float,
    cfg: SimConfig,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    batch_sim: int,
    out_dir: Path,
    omega_scale: float,
    render_gif: bool,
    gif_fps: int,
    gif_trail: int,
) -> Path:
    """Generate a single $N$-oscillator dataset and persist it (with optional GIF)."""
    params = KuramotoParams.bimodal(N=N, P=P, K=K, m=m)
    K_c = float(critical_coupling(P))

    print(
        f"\n[pipeline] === generate N={N} dt_fine={cfg.dt_fine:.3e} "
        f"K={K:.3f} (K_c~{K_c:.3f}) P={P} D={cfg.D} ==="
    )
    out_path = out_dir / f"kuramoto_N{N}_seed{seed}.npz"
    if out_path.exists():
        print(f"[pipeline] {out_path.name} already exists; skipping generation.")
        return out_path

    # Per-chunk checkpoints so a timed-out job resumes instead of restarting.
    ckpt_dir = out_dir / f"_ckpt_kuramoto_N{N}_seed{seed}"

    master = jr.PRNGKey(seed + N)
    k_train, k_val, k_test = jr.split(master, 3)

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, n_split, key in [
        ("train", n_train, k_train),
        ("val", n_val, k_val),
        ("test", n_test, k_test),
    ]:
        t0 = time.perf_counter()
        theta, omega = simulate_split(
            params, cfg, n_split, omega_scale, key, batch_sim,
            ckpt_dir=ckpt_dir, split_name=name,
        )
        print(
            f"[pipeline] split={name} n={n_split} shape={theta.shape} "
            f"time={time.perf_counter() - t0:.1f}s"
        )
        splits[name] = (theta, omega)

    t_grid = np.linspace(0.0, cfg.T, cfg.n_obs)
    meta = {
        "model": "stochastic_second_order_kuramoto",
        "reference": "Olmi & Torcini 2024 eq.(1) with K2=0; "
                     "deterministic part following Filatrella, Nielsen & Pedersen 2008.",
        "N": N, "P": P, "K": K, "m": m, "K_c_estimate": K_c,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "D": cfg.D, "T": cfg.T, "n_fine": cfg.n_fine, "n_obs": cfg.n_obs,
        "seed": seed, "omega_scale": omega_scale,
        "library_versions": library_versions(),
    }
    save_split(out_path, splits, t_grid, meta)
    print(f"[pipeline] wrote {out_path} ({out_path.stat().st_size / 2**20:.1f} MiB)")
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # drop the chunk cache now it's assembled

    if render_gif:
        from experiments.kuramoto.datasets.visualize import render_trajectory_gif

        gif_path = out_dir / f"kuramoto_N{N}_seed{seed}.gif"
        render_trajectory_gif(
            theta=splits["test"][0][0],
            t_grid=t_grid,
            out_path=gif_path,
            fps=gif_fps,
            trail_length=gif_trail,
        )
        print(f"[pipeline] wrote {gif_path} ({gif_path.stat().st_size / 2**20:.1f} MiB)")
    return out_path
