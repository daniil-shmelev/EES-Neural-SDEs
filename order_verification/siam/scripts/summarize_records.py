"""Summarize learning records and check their tables in the master manuscript.

Reported spreads are sample standard deviations. Use --update to replace the
two tabular environments in ees_simods.tex with values from the records.
"""

import argparse
import json
from pathlib import Path
import re
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiment_records"


def check_or_update_table(source, label, expected, update):
    if source is None:
        return None
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", source, re.S)
    matches = [table for table in tables if rf"\label{{{label}}}" in table]
    assert len(matches) == 1, label
    table = matches[0]
    contents = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", table, re.S)
    assert len(contents) == 1, label
    expected = expected.strip()
    if update:
        return source.replace(table, table.replace(contents[0], expected), 1)
    assert contents[0] == expected, f"{label} differs from the records; use --update to refresh it."
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, help="Also compare the records with these manuscript tables.")
    parser.add_argument("--update", action="store_true", help="Update the two tables in ees_simods.tex.")
    args = parser.parse_args()
    manuscript = args.manuscript
    if args.update and manuscript is None:
        parser.error("--update requires --manuscript")
    original = source = manuscript.read_text() if manuscript is not None else None
    kuramoto = json.loads((RECORDS/"kuramoto_runtime_parity.json").read_text())
    rows, stats = [], {}
    for label, method, adjoint, stages in [
        ("cg2_full", "CG2", "Full", 2),
        ("cg2_recursive", "CG2", "Recursive", 2),
        ("cfees25_reversible", "CF-EES(2,5)", "Reconstruction", 3),
    ]:
        values = [cell for cell in kuramoto["cells"] if cell["label"] == label]
        scores = [cell["final_test_loss"] for cell in values]
        times = [cell["train_walltime_s"] for cell in values]
        assert len(values) == 3
        assert {cell["n_steps"]*stages for cell in values} == {150}
        stats[label] = {"test_mean": mean(scores), "test_sd": stdev(scores),
                        "time_mean": mean(times), "time_sd": stdev(times), "seeds": 3}
        rows.append(f"{method} & {adjoint} & {values[0]['n_steps']} & "
                    f"${mean(scores):.2f} \\pm {stdev(scores):.2f}$ & "
                    f"${mean(times):.0f} \\pm {stdev(times):.0f}$ \\\\")
    source = check_or_update_table(source, "tab:kuramoto_quality",
        "\\begin{tabular}{llrrr}\n\\toprule\n"
        "Method & Adjoint & Steps & Test energy score & Runtime (s) \\\\\n\\midrule\n"
        + "\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n", args.update)

    ablation = json.loads((RECORDS/"kuramoto_adjoint_ablation.json").read_text())
    values = [cell["final_test_loss"] for cell in ablation["cells"]
              if cell["label"] == "cfees25_full_n50"]
    assert len(values) == 3
    stats["cfees25_full_ablation"] = {"test_mean": mean(values), "test_sd": stdev(values), "seeds": 3}

    sphere = [json.loads(path.read_text()) for path in
              sorted((RECORDS/"sphere_training").glob("*/metrics.json"))]
    rows = []
    for solver, method, adjoint, stages, seeds in [
        ("geometric_euler", "Geometric Euler", "Full", 1, 3),
        ("cg2", "CG2", "Full", 2, 3),
        ("cfees25", "CF-EES(2,5)", "Reconstruction", 3, 3),
        ("srkmk_general_shark", "SRKMK--ShARK", "Full", 2, 2),
    ]:
        values = [row for row in sphere if row["solver"] == solver]
        scores = [float(row["test_acc_at_best_val_pct"]) for row in values]
        times = [float(row["train_time_s"]) for row in values]
        assert len(values) == seeds
        assert {row["solve_n_steps"]*stages for row in values} == {30}
        assert {row["actual_forward_nfe"] for row in values} == {30}
        assert {row["completed_epochs"] for row in values} == {200}
        stats[solver] = {"accuracy_mean": mean(scores), "accuracy_sd": stdev(scores),
                         "time_mean": mean(times), "time_sd": stdev(times), "seeds": seeds}
        rows.append(f"{method} & {adjoint} & {values[0]['solve_n_steps']} & "
                    f"${mean(scores):.2f} \\pm {stdev(scores):.2f}$ & "
                    f"${mean(times):.1f} \\pm {stdev(times):.1f}$ \\\\")
    source = check_or_update_table(source, "tab:sphere_parity",
        "\\begin{tabular}{llrrr}\n\\toprule\n"
        "Method & Adjoint & Steps & Test accuracy (\\%) & Runtime (s) \\\\\n\\midrule\n"
        + "\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n", args.update)
    if args.update and source != original:
        manuscript.write_text(source)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
