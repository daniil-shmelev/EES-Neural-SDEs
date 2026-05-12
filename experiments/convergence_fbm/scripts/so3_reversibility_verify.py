"""Convergence verification for CF-EES(2,5;1/10) on SO(3) driven by fBm.

Purpose
-------
The symbolic MKW check in order_verification verifies antisymmetric
order 5 for CF-EES(2,5;x). For a fractional Brownian driver with Hurst
parameter H, this predicts the reversibility-error slope 6H - 1:
  H = 0.4  ->  slope 1.4
  H = 0.5  ->  slope 2.0
  H = 0.6  ->  slope 2.6

This script fits that empirical slope on an SO(3) test problem.

Experiment design
-----------------
- d = 3 independent fBm drivers of Hurst H (Davies-Harte sampler).
- Three vector fields xi_m : SO(3) -> so(3), constructed as degree-4
  matrix polynomials in Y projected to so(3) via antisym(M)=(M-M^T)/2.
  The polynomial degree ensures that the elementary differentials
  through weight 5 are non-trivial (a linear xi would make trees with
  >= 3 children at the root trivially vanish).
- CF-EES(2,5;1/10) with beta_1 = (1/3,0,0), beta_2 = (-7/16,15/16,0),
  beta_3 = (49/240,-7/16,2/5), implemented *directly* via scipy
  matrix exponential (no georax dependency, avoids API drift).
- Step sizes h = T/2^k for k = 6, 7, 8, 9, 10, 11 (six grids).
- n_trials independent fBm realisations per Hurst; the coarser grids
  sub-sample the finest grid so all step sizes see the SAME sample
  path (Wong--Zakai-consistent -- isolates discretisation error from
  path variation).
- Global reversibility error E(h) = || Y_back - Y_0 ||_F, averaged
  over trials, slope fitted in log-log.

Usage
-----
  uv run python scripts/so3_reversibility_verify.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------------
# CF-EES(2,5; 1/10) weight vectors
# ---------------------------------------------------------------------

BETA = np.array([
    [1/3,     0,       0],
    [-7/16,   15/16,   0],
    [49/240, -7/16,   2/5],
], dtype=np.float64)


# ---------------------------------------------------------------------
# Davies-Harte fBm sampler (exact Gaussian)
# ---------------------------------------------------------------------

def davies_harte_fbm(H: float, N: int, T: float, rng: np.random.Generator) -> np.ndarray:
    """Return fBm sample of Hurst H on [0, T] at N+1 equispaced grid points."""
    def gamma(k):
        return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))

    k_arr = np.arange(N)
    c = np.array([gamma(int(k)) for k in k_arr])
    circ = np.concatenate([c, [0.0], c[:0:-1]])          # length 2N
    lam = np.maximum(np.real(np.fft.fft(circ)), 0.0)
    xi_r = rng.standard_normal(2 * N)
    xi_i = rng.standard_normal(2 * N)
    xi = xi_r + 1j * xi_i
    xi[0] = xi_r[0]
    xi[N] = xi_r[N]
    xi[N + 1:] = np.conj(xi[1:N][::-1])
    W = np.sqrt(lam) * xi / math.sqrt(2 * N)
    fgn = np.real(np.fft.fft(W))[:N]
    dt = T / N
    fgn *= dt ** H                                        # scale increments
    return np.concatenate([[0.0], np.cumsum(fgn)]).astype(np.float64)


def multidim_fbm(H: float, d: int, N: int, T: float, seed: int) -> np.ndarray:
    """Return (N+1, d) array of d independent fBm paths on [0, T]."""
    rng = np.random.default_rng(seed)
    out = np.empty((N + 1, d), dtype=np.float64)
    for m in range(d):
        out[:, m] = davies_harte_fbm(H, N, T, rng)
    return out


# ---------------------------------------------------------------------
# Vector fields on SO(3): xi_m(Y) = antisym(poly_4(Y)) for m = 0..d-1.
# ---------------------------------------------------------------------

def antisym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M - M.T)


def make_so3_vfs(d: int, seed: int, scale: float = 0.1) -> list[callable]:
    """Build d vector fields xi_m : R^{3x3} -> so(3), each a degree-4
    matrix polynomial projected to antisymmetric part."""
    rng = np.random.default_rng(seed)
    vfs = []
    for m in range(d):
        A = [scale * rng.standard_normal((3, 3)) for _ in range(5)]  # A[0..4]

        def xi(Y, A=A):
            Y2 = Y @ Y
            Y3 = Y2 @ Y
            Y4 = Y3 @ Y
            M = A[0] + A[1] @ Y + A[2] @ Y2 + A[3] @ Y3 + A[4] @ Y4
            return antisym(M)

        vfs.append(xi)
    return vfs


def evaluate_vfs(vfs: list[callable], Y: np.ndarray) -> np.ndarray:
    """Return (3, 3, d) tensor whose slab [:,:,m] is xi_m(Y)."""
    return np.stack([vf(Y) for vf in vfs], axis=-1)


# ---------------------------------------------------------------------
# CF-EES(2,5; 1/10) step on SO(3) for RDE dY = sum_m xi_m(Y) Y dX_m
# ---------------------------------------------------------------------

def cfees25_step(Y: np.ndarray,
                 vfs: list[callable],
                 dX: np.ndarray) -> np.ndarray:
    """One step of the CF-EES(2,5; 1/10) 2N-register recurrence.

    dX is the driver-increment vector of shape (d,). The step size h is
    absorbed into dX (since dX = X(t+h) - X(t) already scales as h^H).
    The recurrence is the CF-EES 2N-register form with K reuse.
    """
    Ks = []
    Y_prev = Y
    for l in range(3):
        K_l = evaluate_vfs(vfs, Y_prev)                   # (3, 3, d)
        Ks.append(K_l)
        V_l = np.zeros((3, 3), dtype=np.float64)
        for i in range(l + 1):
            V_l = V_l + BETA[l, i] * np.einsum("abm,m->ab", Ks[i], dX)
        Y_prev = expm(V_l) @ Y_prev
    return Y_prev


def integrate(Y0: np.ndarray, vfs, X_values: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Run CF-EES(2,5; 1/10) over the grid X_values (shape (N+1, d))
    either forward or backward."""
    Y = Y0
    N = X_values.shape[0] - 1
    rng_order = range(N - 1, -1, -1) if reverse else range(N)
    for i in rng_order:
        dX = X_values[i + 1] - X_values[i]
        if reverse:
            dX = -dX
        Y = cfees25_step(Y, vfs, dX)
    return Y


# ---------------------------------------------------------------------
# Slope fit
# ---------------------------------------------------------------------

def fit_slope(hs, errs):
    lh = np.log(np.asarray(hs, dtype=np.float64))
    le = np.log(np.asarray(errs, dtype=np.float64))
    slope, intercept = np.polyfit(lh, le, 1)
    return float(slope), float(intercept)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run(H: float, d: int, step_exponents: list[int], n_trials: int,
        t_final: float, vf_seed: int, base_seed: int,
        Y0: np.ndarray, verbose: bool = True) -> dict:

    K_max = max(step_exponents)
    N_max = 2 ** K_max
    vfs = make_so3_vfs(d, vf_seed)

    # Per-trial FINE fBm paths (all coarser grids sub-sample the fine grid
    # so that every step size sees the same rough path).
    fine_paths = []
    for r in range(n_trials):
        trial_seed = base_seed * 1000003 + r * 997 + int(round(H * 1000))
        fine_paths.append(multidim_fbm(H, d, N_max, t_final, seed=trial_seed))

    log_h, log_err = [], []
    per_grid_errs = []
    for k in step_exponents:
        N = 2 ** k
        h = t_final / N
        stride = N_max // N
        errs = []
        for X_fine in fine_paths:
            X_vals = X_fine[::stride]
            Y_fwd = integrate(Y0, vfs, X_vals, reverse=False)
            Y_back = integrate(Y_fwd, vfs, X_vals, reverse=True)
            e = float(np.linalg.norm(Y_back - Y0, ord="fro"))
            errs.append(e)
        mean_err = float(np.mean(errs))
        std_err = float(np.std(errs))
        log_h.append(math.log(h))
        log_err.append(math.log(max(mean_err, 1e-300)))
        per_grid_errs.append({
            "N": int(N), "h": float(h),
            "mean_err": mean_err, "std_err": std_err,
        })
        if verbose:
            print(f"  H={H:.2f}  N={N:6d}  h={h:.5e}  mean||Y_b - Y_0||={mean_err:.3e}  (std={std_err:.2e})",
                  flush=True)

    slope, intercept = fit_slope(
        [math.exp(x) for x in log_h],
        [math.exp(x) for x in log_err],
    )
    return {
        "H": H, "fitted_slope": slope, "intercept": intercept,
        "predicted_slope": 6 * H - 1,
        "per_grid": per_grid_errs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hursts", nargs="+", type=float, default=[0.4, 0.5, 0.6])
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--step_exponents", nargs="+", type=int,
                    default=[6, 7, 8, 9, 10, 11])
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--t_final", type=float, default=1.0)
    ap.add_argument("--vf_seed", type=int, default=42)
    ap.add_argument("--base_seed", type=int, default=2026)
    ap.add_argument("--output_json", default="so3_reversibility_results.json")
    args = ap.parse_args()

    # Y_0 = a generic rotation near but not equal to the identity.
    rng0 = np.random.default_rng(args.vf_seed + 1)
    V0 = antisym(0.3 * rng0.standard_normal((3, 3)))
    Y0 = expm(V0)

    print("=" * 78)
    print("  CF-EES(2,5; 1/10) reversibility convergence on SO(3) (non-abelian)")
    print(f"  Group: SO(3) (matrix embedding in R^{{3x3}}, non-zero intrinsic torsion)")
    print(f"  Driver: {args.d} independent fBm paths (Davies-Harte)")
    print(f"  Step sizes: h = T / 2^k, k in {args.step_exponents}, T = {args.t_final}")
    print(f"  Trials per Hurst: {args.n_trials}")
    print(f"  Vector fields: degree-4 matrix polynomials projected to so(3)")
    print("=" * 78)
    print()
    print("  Prediction table:")
    print("  +--------+--------------------------+")
    print("  |   H    |  order-5 slope (6H-1)   |")
    print("  +--------+--------------------------+")
    for H in args.hursts:
        print(f"  |  {H:.2f}  |          {6*H-1:>5.2f}           |")
    print("  +--------+--------------------------+")
    print()

    results = []
    for H in args.hursts:
        print(f"[Hurst H = {H:.2f}]")
        res = run(
            H=H, d=args.d,
            step_exponents=args.step_exponents,
            n_trials=args.n_trials,
            t_final=args.t_final,
            vf_seed=args.vf_seed,
            base_seed=args.base_seed,
            Y0=Y0,
        )
        results.append(res)
        print(f"  ==> fitted slope = {res['fitted_slope']:.3f}  "
              f"(predicted 6H-1 = {res['predicted_slope']:.2f})")
        print()

    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'H':>6}  {'fitted slope':>13}  {'6H-1':>10}  {'verdict':>24}")
    for res in results:
        fit = res["fitted_slope"]
        predicted = res["predicted_slope"]
        if abs(fit - predicted) < 0.35:
            verdict = "matches order 5"
        else:
            verdict = "inconclusive"
        print(f"  {res['H']:>6.2f}  {fit:>13.3f}  {predicted:>10.2f}  {verdict:>24}")

    outp = Path(args.output_json)
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nWrote: {outp.resolve()}")


if __name__ == "__main__":
    main()
