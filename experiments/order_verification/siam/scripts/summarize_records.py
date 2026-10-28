"""Summarize learning records and check their tables in the master manuscript.

Reported spreads are sample standard deviations. Use --update to replace the
two tabular environments in ees_simods.tex with values from the records.
"""

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiment_records"


def check_or_update_table(source, label, expected, update):
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", source, re.DOTALL)
    matches = [table for table in tables if rf"\label{{{label}}}" in table]
    assert len(matches) == 1, label
    table = matches[0]
    contents = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", table, re.DOTALL)
    assert len(contents) == 1, label
    expected = expected.strip()
    if update:
        return source.replace(table, table.replace(contents[0], expected), 1)
    assert contents[0] == expected, (
        f"{label} differs from the records; use --update to refresh it."
    )
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument(
        "--update", action="store_true", help="Update the two tables in ees_simods.tex."
    )
    args = parser.parse_args()
    manuscript = args.manuscript
    original = source = manuscript.read_text()
    verified = (
        ROOT.parents[1]
        / "kuramoto/results/unified/reversible_second_order_verified_20260925"
    )
    kuramoto = json.loads((verified / "comparison.json").read_text())
    rows, stats = [], {}
    for label, method, adjoint in [
        ("cg2_full", "CG2", "Full"),
        ("cfees25_reversible", "CF-EES(2,5)", "Reconstruction"),
    ]:
        (run,) = [run for run in kuramoto if run["name"] == label]
        assert run["status"] == "completed" and run["exit_code"] == 0
        config = run["metrics"]["config"]
        assert config["seed"] == 0 and config["epochs"] == 15 and config["N"] == 1000
        assert config["drift_kind"] == "second_order_mean_field"
        if label == "cg2_full":
            assert config["adjoint"] == "checkpoint_full" and not config["adaptive"]
            assert config["n_steps"] == 75
        else:
            assert config["adjoint"] == "reversible" and config["adaptive"]
        steps = "Adaptive" if config["adaptive"] else str(config["n_steps"])
        score, runtime = run["metrics"]["final_test_loss"], run["wall_time_s"]
        stats[label] = {"test_loss": score, "end_to_end_seconds": runtime, "seeds": 1}
        rows.append(
            f"{method} & {adjoint} & {steps} & ${score:.2f}$ & ${runtime:.1f}$ " + r"\\"
        )
    source = check_or_update_table(
        source,
        "tab:kuramoto_quality",
        "\\begin{tabular}{lllrr}\n\\toprule\n"
        "Method & Adjoint & Steps & Test energy score & Runtime (s) "
        + r"\\"
        + "\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n",
        args.update,
    )

    sphere = [
        json.loads(path.read_text())
        for path in sorted((RECORDS / "sphere_training").glob("*/metrics.json"))
    ]
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
        assert {row["solve_n_steps"] * stages for row in values} == {30}
        assert {row["actual_forward_nfe"] for row in values} == {30}
        assert {row["completed_epochs"] for row in values} == {200}
        stats[solver] = {
            "accuracy_mean": mean(scores),
            "accuracy_sd": stdev(scores),
            "time_mean": mean(times),
            "time_sd": stdev(times),
            "seeds": seeds,
        }
        rows.append(
            f"{method} & {adjoint} & {values[0]['solve_n_steps']} & "
            f"${mean(scores):.2f} \\pm {stdev(scores):.2f}$ & "
            f"${mean(times):.1f} \\pm {stdev(times):.1f}$ \\\\"
        )
    source = check_or_update_table(
        source,
        "tab:sphere_parity",
        "\\begin{tabular}{llrrr}\n\\toprule\n"
        "Method & Adjoint & Steps & Test accuracy (\\%) & Runtime (s) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n",
        args.update,
    )
    if args.update and source != original:
        manuscript.write_text(source)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
