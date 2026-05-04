"""Aggregate M3 + M4 + M5 results into a single JSON for manuscript fill-in.

Reads all known result files and emits a tidy summary that maps directly
to the placeholders in the .tex file (`\\dan{TBD}` cells, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("experiments/kuramoto/results"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/kuramoto/results/aggregate.json"))
    return p.parse_args()


def _maybe_load(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            return {"_error": f"failed to load {path}: {exc}"}
    return None


def main() -> int:
    args = parse_args()
    out = {"sources": {}, "m3": {}, "m4": {}, "m5": {}}

    # M3: pilot adjoint parity
    pilot = _maybe_load(args.results_dir / "pilot" / "pilot_adjoint_parity.json")
    if pilot:
        out["sources"]["m3"] = "pilot/pilot_adjoint_parity.json"
        m3_1 = pilot.get("m3_1_fidelity_at_train_n_steps", {})
        out["m3"]["fidelity_at_train_n_steps"] = m3_1.get("results", {})
        out["m3"]["fidelity_long_horizon"] = [
            {"n_steps": x.get("n_steps"), "results": x.get("results", {})}
            for x in pilot.get("m3_3_fidelity_long_horizon", [])
        ]
        if "m3_2_training_parity" in pilot:
            tp = pilot["m3_2_training_parity"].get("results", {})
            out["m3"]["training_parity"] = {
                adj: {
                    "n_seeds": len(seeds),
                    "test_loss_mean": mean([s["final_test_loss"] for s in seeds]),
                    "test_loss_std": pstdev([s["final_test_loss"] for s in seeds]) if len(seeds) > 1 else 0.0,
                    "wall_clock_s_mean": mean([s["wall_clock_s"] for s in seeds]),
                }
                for adj, seeds in tp.items()
            }

    # M4: memory sweep (look for any memory_sweep_*.json file)
    m4_files = sorted(args.results_dir.glob("memory_sweep_N*.json"))
    for f in m4_files:
        payload = _maybe_load(f)
        if not payload:
            continue
        N = payload["config"]["N"]
        cells = payload["cells"]
        # group by (solver, adjoint)
        per_pair: dict[str, list[dict]] = {}
        for c in cells:
            per_pair.setdefault(f"{c['solver']}:{c['adjoint']}", []).append(c)
        out["m4"][f"N{N}"] = {
            "config": payload["config"],
            "by_pair": {
                pair: [
                    {"n_steps": c["n_steps"],
                     "peak_mib": (c["peak_bytes"] / 2**20) if c.get("peak_bytes") else None,
                     "wall_s": c.get("wall_clock_s_mean"),
                     "oom": bool(c.get("oom"))}
                    for c in sorted(cells_, key=lambda x: x["n_steps"])
                ]
                for pair, cells_ in per_pair.items()
            },
        }
        out["sources"][f"m4_{f.name}"] = str(f.relative_to(args.results_dir))

    # M5: hero training
    hero_dir = args.results_dir / "hero_N1000"
    hero_summary = _maybe_load(hero_dir / "hero_summary.json")
    if hero_summary:
        out["sources"]["m5"] = "hero_N1000/hero_summary.json"
        # Group by label, and enrich each cell with test_loss_at_best_val
        # by looking up the per-cell history.json (so late-epoch instability
        # doesn't bias the reported number).
        per_label: dict[str, list[dict]] = {}
        for cell in hero_summary.get("cells", []):
            cell_out = Path(cell.get("out_dir", ""))
            hist = _maybe_load(cell_out / "history.json")
            if hist and hist.get("val_loss") and hist.get("test_loss"):
                vals = hist["val_loss"]
                tests = hist["test_loss"]
                if vals and tests and len(vals) == len(tests):
                    best_epoch = min(range(len(vals)), key=lambda i: vals[i])
                    cell["test_loss_at_best_val"] = float(tests[best_epoch])
                    cell["best_val_epoch"] = int(best_epoch)
            per_label.setdefault(cell["label"], []).append(cell)
        def _summary(cells: list[dict]) -> dict:
            def _agg(key: str):
                vals = [c[key] for c in cells if c.get(key) is not None]
                return {
                    "mean": mean(vals) if vals else None,
                    "std": pstdev(vals) if len(vals) > 1 else None,
                    "values": vals,
                }
            return {
                "n_seeds": len(cells),
                "exit_codes": [c.get("exit_code") for c in cells],
                "test_loss_final": _agg("final_test_loss"),
                "test_loss_at_best_val": _agg("test_loss_at_best_val"),
                "test_theta_mae_final": _agg("final_test_theta_mae_mean"),
                "test_omega_mae_final": _agg("final_test_omega_mae_mean"),
                "wall_clock_s": _agg("wall_clock_s"),
                "best_val_epoch": _agg("best_val_epoch"),
            }
        out["m5"]["per_variant"] = {label: _summary(cells) for label, cells in per_label.items()}

    args.output.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")
    print(json.dumps({k: list(v.keys()) if isinstance(v, dict) else None
                      for k, v in out.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
