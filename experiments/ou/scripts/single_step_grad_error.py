"""Per-step gradient-error sweep for the OU Neural-LSDE.

For each ``(method, h, seed)`` cell we take a single solver step from a
fixed initial state under a fixed Brownian increment, compute the gradient
of a downstream squared-error loss via full autograd
(``torchsde.sdeint``) and via the reversible adjoint
(``torchsde.sdeint_adjoint``), and record the relative L2 distance between
the resulting gradient vectors. Both calls share the same
``BrownianInterval`` entropy so any discrepancy is purely due to the
adjoint method.

Usage::

    PYTHONPATH=. .venv/bin/python -m experiments.ou.scripts.single_step_grad_error \
        --output-dir experiments/ou/results/grad_error/single_step
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchsde

from experiments.ou.OU import generate_data, seed_everything
from experiments.ou.scripts._ou_grad_helpers import (
    NDE_model,
    NeuralLSDEFunc,
    strip_spline_state,
)


DEFAULT_METHODS = ("ees25", "ees27")
DEFAULT_H_VALUES = (
    1.0 / 256, 1.0 / 128, 1.0 / 64, 1.0 / 32,
    1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2,
)
DEFAULT_SEEDS = (10001, 10002, 10003, 10004, 10005)
DEFAULT_T_ANCHOR = 0.5
DEFAULT_EVAL_BATCH = 64
DEFAULT_EVAL_DATA_SEED = 0

REF_CONFIG = {
    "method": "ees25",
    "adjoint": "autograd",
    "dt": 0.05,
    "num_samples": 50000,
    "T": 10.0, "N": 20,
    "theta": 0.2, "mu": 0.1, "sigma": 2.0, "X0": 1.0,
    "train_ratio": 0.8,
    "batch_size": 50000,
    "seed": 42,
    "num_epochs": 250,
    "input_dim": 2, "output_dim": 1,
    "hidden_dim": 32, "num_layers": 1,
    "lr": 1e-3,
}


def _flat_grads(module: torch.nn.Module) -> torch.Tensor:
    parts = []
    for p in module.parameters():
        if p.grad is None:
            parts.append(torch.zeros_like(p).flatten())
        else:
            parts.append(p.grad.detach().flatten().clone())
    return torch.cat(parts) if parts else torch.zeros(0)


def _build_model(config: dict, device: torch.device) -> NDE_model:
    return NDE_model(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
        method=config["method"],
        vector_field=NeuralLSDEFunc,
        dt=config["dt"],
        adjoint=config["adjoint"],
    ).to(device)


def _train_reference_model(
    out_dir: Path,
    *,
    num_epochs: int,
    num_samples: int,
    device: torch.device,
) -> tuple[NDE_model, dict]:
    """Train a default OU NeuralLSDE and persist its state dict."""
    config = dict(REF_CONFIG)
    config.update({"num_epochs": num_epochs, "num_samples": num_samples})

    print(f"[ref] training OU reference (epochs={num_epochs}, samples={num_samples})...")
    seed_everything(config["seed"])
    total_data, coeffs, times = generate_data(config)
    coeffs = coeffs.to(device)
    times = times.to(device)
    targets = total_data[:, :, 1].to(device)

    model = _build_model(config, device)
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.MSELoss()

    n_train = int(config["train_ratio"] * config["num_samples"])
    perm = torch.randperm(coeffs.shape[0])
    train_idx = perm[:n_train]
    coeffs_t = coeffs[train_idx]
    targets_t = targets[train_idx]

    started = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        optim.zero_grad()
        pred = model(coeffs_t, times).squeeze(-1)
        loss = criterion(pred, targets_t)
        loss.backward()
        optim.step()
        if epoch % 25 == 0 or epoch == num_epochs:
            print(f"[ref]   epoch {epoch:>3}/{num_epochs}  train_mse={loss.item():.4f}", flush=True)
    print(f"[ref] done in {time.time() - started:.1f}s, terminal_mse={loss.item():.4f}")

    sd = strip_spline_state(model.state_dict())
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd, "config": config}, out_dir / "ref_model.pt")
    return model, config


def _ensure_reference_model(
    out_dir: Path,
    *,
    num_epochs: int,
    num_samples: int,
    retrain: bool,
    device: torch.device,
) -> tuple[NDE_model, dict]:
    ref_path = out_dir / "ref_model.pt"
    if ref_path.exists() and not retrain:
        print(f"[ref] loading existing model -> {ref_path}")
        ckpt = torch.load(ref_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = _build_model(config, device)
        sd = strip_spline_state(ckpt["state_dict"])
        model.load_state_dict(sd, strict=False)
        return model, config
    return _train_reference_model(
        out_dir, num_epochs=num_epochs, num_samples=num_samples, device=device,
    )


def _make_eval_inputs(
    config: dict,
    *,
    batch_size: int,
    data_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eval_config = dict(config)
    eval_config["num_samples"] = batch_size
    eval_config["batch_size"] = batch_size
    eval_config["seed"] = data_seed
    seed_everything(data_seed)
    total_data, coeffs, times = generate_data(eval_config)
    return total_data.to(device), coeffs.to(device), times.to(device)


def _run_one_cell(
    *,
    model: NDE_model, method: str, h: float, entropy_seed: int,
    t_anchor: float, y0_fixed: torch.Tensor, y_target: torch.Tensor,
    batch_size: int, hidden_dim: int, device: torch.device, dtype: torch.dtype,
) -> dict:
    bm = torchsde.BrownianInterval(
        t0=t_anchor, t1=t_anchor + h, size=(batch_size, hidden_dim),
        dtype=dtype, device=device, entropy=entropy_seed,
    )
    ts = torch.tensor([t_anchor, t_anchor + h], dtype=dtype, device=device)

    model.adjoint = "autograd"
    model.method = method
    model.dt = h
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    y_seq = model.integrate(y0_fixed, ts, bm=bm)
    y1_auto = y_seq[-1]
    pred = model.decoder(y1_auto)
    loss_auto = ((pred - y_target) ** 2).mean()
    loss_auto.backward()
    wall_auto = time.perf_counter() - t0
    g_auto = _flat_grads(model.func)

    model.adjoint = "reversible"
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    y_seq = model.integrate(y0_fixed, ts, bm=bm)
    y1_rev = y_seq[-1]
    pred = model.decoder(y1_rev)
    loss_rev = ((pred - y_target) ** 2).mean()
    loss_rev.backward()
    wall_rev = time.perf_counter() - t0
    g_rev = _flat_grads(model.func)

    g_auto_norm = float(g_auto.norm().item())
    g_rev_norm = float(g_rev.norm().item())
    diff_norm = float((g_rev - g_auto).norm().item())
    rel_err = diff_norm / max(g_auto_norm, 1e-30)
    denom = max(g_auto_norm * g_rev_norm, 1e-30)
    cos_sim = float((g_rev * g_auto).sum().item() / denom)

    return {
        "method": method, "h": h, "entropy": entropy_seed,
        "rel_err": rel_err, "cos_sim": cos_sim,
        "g_auto_norm": g_auto_norm, "g_rev_norm": g_rev_norm,
        "loss_auto": float(loss_auto.item()),
        "loss_rev": float(loss_rev.item()),
        "y1_diff_norm": float((y1_rev - y1_auto).norm().item()),
        "wall_auto": wall_auto, "wall_rev": wall_rev,
        "g_auto": g_auto.detach().cpu().numpy(),
        "g_rev": g_rev.detach().cpu().numpy(),
        "y1_auto": y1_auto.detach().cpu().numpy(),
        "y1_rev": y1_rev.detach().cpu().numpy(),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/ou/results/grad_error/single_step"))
    p.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    p.add_argument("--h-values", type=str, default=None)
    p.add_argument("--num-h", type=int, default=None)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--num-seeds", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=REF_CONFIG["num_epochs"])
    p.add_argument("--num-samples", type=int, default=REF_CONFIG["num_samples"])
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH)
    p.add_argument("--t-anchor", type=float, default=DEFAULT_T_ANCHOR)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    h_values = (
        [float(x) for x in args.h_values.split(",")]
        if args.h_values is not None else list(DEFAULT_H_VALUES)
    )
    if args.num_h is not None:
        h_values = h_values[: args.num_h]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.num_seeds is not None:
        seeds = seeds[: args.num_seeds]

    print(f"[ou-grad] device={device} methods={methods}")
    print(f"[ou-grad] h_values={h_values}")
    print(f"[ou-grad] seeds={seeds}")

    model, ref_config = _ensure_reference_model(
        output_dir,
        num_epochs=args.num_epochs, num_samples=args.num_samples,
        retrain=args.retrain, device=device,
    )
    model.eval()
    for p_ in model.initial.parameters():
        p_.requires_grad_(False)
    for p_ in model.decoder.parameters():
        p_.requires_grad_(False)

    total_data, coeffs, times = _make_eval_inputs(
        ref_config,
        batch_size=args.eval_batch_size,
        data_seed=DEFAULT_EVAL_DATA_SEED,
        device=device,
    )
    model.func.set_X(coeffs, times)
    dtype = total_data.dtype
    t_anchor = float(args.t_anchor)

    with torch.no_grad():
        y0_fixed = model.func.X.evaluate(torch.tensor(t_anchor, dtype=dtype, device=device))
        y0_fixed = model.initial(y0_fixed).detach()

    eval_inputs_path = output_dir / "eval_inputs.npz"
    np.savez_compressed(
        eval_inputs_path,
        total_data=total_data.detach().cpu().numpy(),
        coeffs=coeffs.detach().cpu().numpy(),
        times=times.detach().cpu().numpy(),
        y0_fixed=y0_fixed.detach().cpu().numpy(),
        t_anchor=t_anchor,
    )

    rel_err_table = {m: [[] for _ in h_values] for m in methods}
    cos_sim_table = {m: [[] for _ in h_values] for m in methods}
    g_auto_norm_table = {m: [[] for _ in h_values] for m in methods}
    g_rev_norm_table = {m: [[] for _ in h_values] for m in methods}

    total_cells = len(methods) * len(h_values) * len(seeds)
    cell_idx = 0
    overall_t0 = time.perf_counter()

    for method in methods:
        method_cells_dir = cells_dir / method
        method_cells_dir.mkdir(parents=True, exist_ok=True)
        for h_idx, h in enumerate(h_values):
            for seed in seeds:
                cell_idx += 1
                with torch.no_grad():
                    y_target_full = model.func.X.evaluate(
                        torch.tensor(t_anchor + h, dtype=dtype, device=device)
                    )
                    y_target = y_target_full[:, 1:2].detach()

                cell = _run_one_cell(
                    model=model, method=method, h=h,
                    entropy_seed=int(seed), t_anchor=t_anchor,
                    y0_fixed=y0_fixed, y_target=y_target,
                    batch_size=args.eval_batch_size,
                    hidden_dim=ref_config["hidden_dim"],
                    device=device, dtype=dtype,
                )
                cell_path = method_cells_dir / f"h{h:.6e}_seed{seed}.npz"
                np.savez_compressed(
                    cell_path,
                    g_auto=cell["g_auto"], g_rev=cell["g_rev"],
                    y1_auto=cell["y1_auto"], y1_rev=cell["y1_rev"],
                    loss_auto=cell["loss_auto"], loss_rev=cell["loss_rev"],
                    rel_err=cell["rel_err"], cos_sim=cell["cos_sim"],
                    g_auto_norm=cell["g_auto_norm"], g_rev_norm=cell["g_rev_norm"],
                    y1_diff_norm=cell["y1_diff_norm"],
                    wall_auto=cell["wall_auto"], wall_rev=cell["wall_rev"],
                    h=h, entropy=int(seed), method=method,
                )
                rel_err_table[method][h_idx].append(cell["rel_err"])
                cos_sim_table[method][h_idx].append(cell["cos_sim"])
                g_auto_norm_table[method][h_idx].append(cell["g_auto_norm"])
                g_rev_norm_table[method][h_idx].append(cell["g_rev_norm"])
                print(
                    f"[ou-grad {cell_idx:>3}/{total_cells}] {method} h={h:.4e} "
                    f"seed={seed} rel_err={cell['rel_err']:.3e} "
                    f"cos_sim={cell['cos_sim']:.6f}",
                    flush=True,
                )

    def _stats(table) -> dict:
        out: dict = {}
        for m, rows in table.items():
            arr = np.array(rows, dtype=float)
            mean = arr.mean(axis=1).tolist()
            stderr = ((arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1])).tolist()
                      if arr.shape[1] > 1 else [0.0] * arr.shape[0])
            out[m] = {"mean": mean, "stderr": stderr, "raw": arr.tolist()}
        return out

    summary = {
        "h": list(h_values),
        "methods": methods,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "t_anchor": t_anchor,
        "rel_err": _stats(rel_err_table),
        "cos_sim": _stats(cos_sim_table),
        "g_auto_norm": _stats(g_auto_norm_table),
        "g_rev_norm": _stats(g_rev_norm_table),
        "ref_config": ref_config,
        "elapsed_seconds": time.perf_counter() - overall_t0,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    manifest = {
        "summary_sha256": _sha256(summary_path),
        "ref_model_sha256": _sha256(output_dir / "ref_model.pt"),
        "eval_inputs_sha256": _sha256(eval_inputs_path),
        "torch_version": torch.__version__,
        "torchsde_version": getattr(torchsde, "__version__", "unknown"),
        "n_cells": total_cells,
        "device": str(device),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ou-grad] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
