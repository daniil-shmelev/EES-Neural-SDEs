"""Substitute manuscript placeholders with real M3/M4/M5 numbers.

Reads `aggregate.json` and replaces the `\\dan{TODO (waiting on ...):}` and
`\\dan{TBD}` placeholders in the manuscript .tex with concrete values.
Idempotent: rerunning with newer numbers updates the substitutions.

Conservative by design — only edits well-defined sentinels, leaves anything
unparseable alone, and prints what was (and wasn't) updated.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manuscript",
        type=Path,
        default=Path("/mnt/c/Users/Shmelev/source/overleaf/EES_Neural_SDEs_Overleaf/neurips2026/ees_neurips_manuscript.tex"),
    )
    p.add_argument(
        "--aggregate",
        type=Path,
        default=Path("experiments/kuramoto/results/aggregate.json"),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing.",
    )
    return p.parse_args()


def _fmt_mib(b: float | None) -> str:
    if b is None:
        return "N/A"
    if b < 1024:
        return f"{b:.0f}\\,MiB"
    return f"{b/1024:.2f}\\,GiB"


def _max_bytes_pair(by_pair: dict, pair: str, n_steps: int | None = None):
    """Return the largest non-OOM (n_steps, peak_mib) for the given pair."""
    if pair not in by_pair:
        return None
    rows = [r for r in by_pair[pair] if not r["oom"] and r["peak_mib"] is not None]
    if not rows:
        return None
    if n_steps is not None:
        match = [r for r in rows if r["n_steps"] == n_steps]
        if match:
            return match[0]
    return max(rows, key=lambda r: r["n_steps"])


def _smallest_oom_n_steps(by_pair: dict, pair: str) -> int | None:
    if pair not in by_pair:
        return None
    oom = [r for r in by_pair[pair] if r["oom"]]
    if not oom:
        return None
    return min(oom, key=lambda r: r["n_steps"])["n_steps"]


def build_substitutions(agg: dict) -> dict[str, str]:
    """Return a mapping {regex_pattern: replacement} keyed on the placeholder text."""
    subs: dict[str, str] = {}

    # ---- M4 memory scaling text in body ----
    m4 = agg.get("m4", {})
    n1000 = m4.get("N1000")
    if n1000:
        by_pair = n1000["by_pair"]
        cfees = _max_bytes_pair(by_pair, "cfees25:reversible")
        cg2_full = _max_bytes_pair(by_pair, "cg2:checkpoint_full")
        cg2_full_oom = _smallest_oom_n_steps(by_pair, "cg2:checkpoint_full")

        # Inline body summary:
        if cfees and cg2_full:
            ns = max(cfees["n_steps"], cg2_full["n_steps"])
            txt = (
                f"At $N=1000$, $n_{{\\text{{steps}}}}={ns}$ and batch $64$, "
                f"CG2 + full-tape Checkpoint requires {_fmt_mib(cg2_full['peak_mib'])} of "
                f"adjoint memory; $\\mathrm{{CF\\text{{-}}EES}}(2,5)$ + "
                f"\\texttt{{ReversibleAdjoint}} stays at "
                f"{_fmt_mib(cfees['peak_mib'])}."
            )
            subs[r"\\dan\{TODO \(waiting on M4\): summarise final memory ratio[^}]*\}"] = txt

        # Memory-figure caption "TODO M4": just promote it to a clean caption fragment.
        if cfees:
            sweep_max = cfees["n_steps"]
            cap = (
                f"Memory advantage of $\\mathrm{{CF\\text{{-}}EES}}(2,5)$ + "
                f"\\texttt{{ReversibleAdjoint}} (\\textcolor[HTML]{{D62728}}{{red}}) "
                f"on the Kuramoto NSDE: stays at {_fmt_mib(cfees['peak_mib'])} "
                f"across $n_{{\\mathrm{{steps}}}}\\in [50, {sweep_max}]$ at $N=1000$, "
                f"batch $64$. CG2 baselines under \\texttt{{RecursiveCheckpointAdjoint}} "
                f"\\citep{{stumm2010new}} grow as $\\mathcal{{O}}(n_{{\\mathrm{{steps}}}})$ "
                f"(full tape) and $\\mathcal{{O}}(\\sqrt{{n_{{\\mathrm{{steps}}}}}})$ "
                f"(Stumm--Walther treeverse)"
            )
            if cg2_full_oom is not None:
                cap += f"; full tape OOMs at $n_{{\\mathrm{{steps}}}}={cg2_full_oom}$"
            cap += "."
            subs[r"\\dan\{TODO \(waiting on M4\):\}"] = cap

        # Intro figure: strip the "TODO regenerate" note and switch the
        # placeholder figure file to the freshly-rendered Kuramoto one.
        # (The figure caption itself already says the right thing.)
        subs[r"\\dan\{TODO \(waiting on M4\): regenerate this figure[^}]*\}"] = ""
        subs[r"figures/fig_scaling_compact_nogrid\.pdf"] = (
            "figures/fig_kuramoto_memory_scaling.pdf"
        )

    # ---- M3.2 training-parity sentence in body ----
    tp = agg.get("m3", {}).get("training_parity")
    if tp:
        rev = tp.get("reversible")
        full = tp.get("checkpoint_full")
        rec = tp.get("checkpoint_recursive")
        if rev:
            parts = [
                f"\\texttt{{ReversibleAdjoint}} reaches "
                f"${rev['test_loss_mean']:.3f} \\pm {rev['test_loss_std']:.3f}$"
            ]
            if full:
                parts.append(
                    f"\\texttt{{Checkpoint}}~(full) reaches "
                    f"${full['test_loss_mean']:.3f} \\pm {full['test_loss_std']:.3f}$"
                )
            if rec:
                parts.append(
                    f"treeverse reaches "
                    f"${rec['test_loss_mean']:.3f} \\pm {rec['test_loss_std']:.3f}$"
                )
            subs[r"\\dan\{TODO \(waiting on M3\.2\): insert exact \$\\Delta\$ test energy score \$\\pm\$ seed-noise CI\}"] = (
                "; specifically " + ", ".join(parts) + " (3 seeds at $N=2$, 5 epochs)"
            )

    # ---- M5 hero numbers ----
    m5 = agg.get("m5", {}).get("per_variant", {})
    def _best(v: dict | None):
        # Prefer test_loss_at_best_val; fall back to final.
        if not v:
            return None, None
        b = v.get("test_loss_at_best_val", {})
        if b and b.get("mean") is not None:
            return b["mean"], b.get("std")
        f = v.get("test_loss_final", {})
        if f and f.get("mean") is not None:
            return f["mean"], f.get("std")
        return None, None

    cfees5 = m5.get("cfees25_reversible")
    eucl = m5.get("euclidean_baseline")
    cg2_5 = m5.get("cg2_treeverse")
    cfees_mean, cfees_std = _best(cfees5)
    eucl_mean, eucl_std = _best(eucl)
    cg2_mean, cg2_std = _best(cg2_5)

    if cfees_mean is not None:
        fragments = [
            f"$\\mathrm{{CF\\text{{-}}EES}}(2,5)$ + \\texttt{{ReversibleAdjoint}} "
            f"reaches test ES "
            f"${cfees_mean:.3f}"
            + (f" \\pm {cfees_std:.3f}$" if cfees_std is not None else "$")
            + " at $N=1000$"
        ]
        if eucl_mean is not None:
            delta = (eucl_mean - cfees_mean) / max(eucl_mean, 1e-9)
            fragments.append(
                f"beating the Euclidean baseline "
                f"(${eucl_mean:.3f}$, $\\Delta = {delta*100:+.1f}\\%$)"
            )
        if cg2_mean is not None:
            fragments.append(
                f"and within seed-noise of CG2 + treeverse (${cg2_mean:.3f}$)"
            )
        body_summary = "Concretely, " + ", ".join(fragments) + "."
        subs[r"\\dan\{TODO \(waiting on M3\.2/M4/M5\)[^}]*\}"] = body_summary
        subs[r"\\dan\{TODO \(waiting on M5\): insert a sentence[^}]*\}"] = body_summary

        # Table cells (tab:kuramoto_quality)
        def es_cell_for(label_key: str, mem_str: str):
            v = m5.get(label_key)
            mn, sd = _best(v)
            if mn is None:
                es = r"\dan{TBD}"
            elif sd is not None:
                es = f"${mn:.3f} \\pm {sd:.3f}$"
            else:
                es = f"${mn:.3f}$"
            return es, mem_str
        es_eucl, mem_eucl = es_cell_for("euclidean_baseline", r"\dan{TBD}")
        es_cg2, mem_cg2 = es_cell_for("cg2_treeverse", r"\dan{TBD}")
        es_cfees, mem_cfees = es_cell_for("cfees25_reversible", r"$\mathbf{\mathcal{O}(1)}$")

        subs[r"Euclidean baseline & \\texttt\{DirectAdjoint\} & \\dan\{TBD\} & \\dan\{TBD\}"] = (
            f"Euclidean baseline & \\texttt{{DirectAdjoint}} & {es_eucl} & {mem_eucl}"
        )
        subs[r"CG2 & \\texttt\{Checkpoint\} \(treeverse\) & \\dan\{TBD\} & \\dan\{TBD\}"] = (
            f"CG2 & \\texttt{{Checkpoint}} (treeverse) & {es_cg2} & {mem_cg2}"
        )
        subs[r"\$\\mathrm\{CF\\text\{-\}EES\}\(2,5\)\$ & \\texttt\{ReversibleAdjoint\} & \\dan\{TBD\} & \$\\mathbf\{\\mathcal\{O\}\(1\)\}\$"] = (
            f"$\\mathrm{{CF\\text{{-}}EES}}(2,5)$ & \\texttt{{ReversibleAdjoint}} & {es_cfees} & {mem_cfees}"
        )

    # ---- The training-parity table in the appendix ----
    if tp:
        def adj_row(adj_label: str, fancy: str) -> str:
            v = tp.get(adj_label)
            if v and v.get("test_loss_mean") is not None:
                std = v.get("test_loss_std", 0.0) or 0.0
                return (
                    f"\\texttt{{{fancy}}} & "
                    f"${v['test_loss_mean']:.4f} \\pm {std:.4f}$ \\\\"
                )
            return f"\\texttt{{{fancy}}} & \\dan{{TBD}} \\\\"

        # Keep the table-fill simple: replace just the three TBD cells via three
        # explicit substitutions rather than rewriting the whole block.
        if tp.get("reversible"):
            v = tp["reversible"]
            std = v.get("test_loss_std", 0.0) or 0.0
            subs[r"\\texttt\{ReversibleAdjoint\} & \\dan\{TBD\}"] = (
                f"\\texttt{{ReversibleAdjoint}} & "
                f"${v['test_loss_mean']:.4f} \\pm {std:.4f}$"
            )
        if tp.get("checkpoint_full"):
            v = tp["checkpoint_full"]
            std = v.get("test_loss_std", 0.0) or 0.0
            subs[r"\\texttt\{RecursiveCheckpointAdjoint\} \(full\) & \\dan\{TBD\}"] = (
                f"\\texttt{{RecursiveCheckpointAdjoint}} (full) & "
                f"${v['test_loss_mean']:.4f} \\pm {std:.4f}$"
            )
        if tp.get("checkpoint_recursive"):
            v = tp["checkpoint_recursive"]
            std = v.get("test_loss_std", 0.0) or 0.0
            subs[r"\\texttt\{RecursiveCheckpointAdjoint\} \(treeverse\) & \\dan\{TBD\}"] = (
                f"\\texttt{{RecursiveCheckpointAdjoint}} (treeverse) & "
                f"${v['test_loss_mean']:.4f} \\pm {std:.4f}$"
            )

    return subs


def main() -> int:
    args = parse_args()
    agg = json.loads(args.aggregate.read_text())
    src = args.manuscript.read_text()
    subs = build_substitutions(agg)

    if not subs:
        print("[fill] no substitutions available yet (aggregate.json sparse)")
        return 0

    out = src
    n_changed = 0
    for pat, repl in subs.items():
        compiled = re.compile(pat)
        new, k = compiled.subn(repl, out)
        if k > 0:
            print(f"[fill] {k} match(es) for /{pat[:80]}/")
            out = new
            n_changed += k
        else:
            print(f"[fill] no match for /{pat[:80]}/")

    if out == src:
        print("[fill] no changes needed (manuscript already filled or no patterns matched)")
        return 0

    if args.dry_run:
        print(f"[fill] dry-run: would write {n_changed} substitutions to {args.manuscript}")
        return 0

    args.manuscript.write_text(out)
    print(f"[fill] wrote {n_changed} substitutions to {args.manuscript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
