"""Summarize and compare the post-review experiments across training seeds."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.aggregate_professor_results import (
    mean_std_ci95,
    paired_effects,
    permutation_pvalue,
)


OUT = ROOT / "runs" / "Office_Building" / "review_additions"
BASE = ROOT / "runs" / "Office_Building" / "experiments"
SEEDS = tuple(range(42, 52))
FLEETS = (1, 3, 5, 10)
WEIGHTED_FLEETS = (3, 5, 10)
METRICS = ("success", "spl")
REGIMES = ("static", "dynamic_edge_closure")


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_row(path):
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}")
    return rows[0]


def holm_adjust(rows):
    rows = [dict(row) for row in rows]
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["statistical_family"]].append(index)
    for family, indices in groups.items():
        ordered = sorted(indices, key=lambda index: float(rows[index]["p_value"]))
        running = 0.0
        size = len(ordered)
        for rank, index in enumerate(ordered):
            running = max(running, min(1.0, (size - rank) * float(rows[index]["p_value"])))
            rows[index]["p_value_holm"] = running
            rows[index]["holm_family_size"] = size
            rows[index]["significant_holm_0_05"] = int(running < 0.05)
    return rows


def main():
    descriptive = []
    comparisons = []
    for robots in FLEETS:
        for regime in REGIMES:
            standard = {
                seed: next(
                    row for row in read_csv(
                        BASE / f"seed_{seed}" / f"robots_{robots}" / "evaluation"
                        / ("dynamic_comparison_summary.csv" if regime == "dynamic_edge_closure" else "comparison_summary.csv")
                    )
                    if row["method"] == "fedavg_qlearning"
                    and row["evaluation_regime"] == regime
                )
                for seed in SEEDS
            }
            shortest_path = summary_row(
                OUT / "shortest_path" / f"robots_{robots}"
                / ("dynamic_test_protocol" if regime == "dynamic_edge_closure" else "fixed_test_protocol")
                / f"{regime}_summary.csv"
            )

            method_rows = [("fedavg_qlearning", standard)]
            if robots in WEIGHTED_FLEETS:
                weighted = {}
                for seed in SEEDS:
                    weighted[seed] = summary_row(
                        OUT / "visitation_weighted" / f"seed_{seed}" / f"robots_{robots}"
                        / "evaluation" / "dynamic_test_protocol" / f"{regime}_summary.csv"
                        if regime == "dynamic_edge_closure"
                        else OUT / "visitation_weighted" / f"seed_{seed}" / f"robots_{robots}"
                        / "evaluation" / "fixed_test_protocol" / f"{regime}_summary.csv"
                    )
                method_rows.append(("visitation_weighted_fedavg_qlearning", weighted))
            method_rows.append(("weighted_shortest_path_policy", {seed: shortest_path for seed in SEEDS}))
            for method, seed_rows in method_rows:
                row = {"robots": robots, "regime": regime, "method": method, "n_seeds": len(seed_rows)}
                for metric in METRICS:
                    values = [float(seed_rows[seed][f"{metric}_mean"]) for seed in SEEDS]
                    mean, std, ci95 = mean_std_ci95(values)
                    row.update({
                        f"{metric}_mean_across_seeds": mean,
                        f"{metric}_std_across_seeds": std,
                        f"{metric}_ci95_across_seeds": ci95,
                    })
                descriptive.append(row)

            comparison_rows = []
            if robots in WEIGHTED_FLEETS:
                comparison_rows.append((
                    "post_review_visit_weighted",
                    "visitation_weighted_fedavg_qlearning", weighted,
                    "fedavg_qlearning", standard,
                ))
            comparison_rows.append((
                    "post_review_shortest_path",
                    "weighted_shortest_path_policy",
                    {seed: shortest_path for seed in SEEDS},
                    "fedavg_qlearning", standard,
                ))
            for family, method_a, rows_a, method_b, rows_b in comparison_rows:
                for metric_index, metric in enumerate(METRICS):
                    first = [float(rows_a[seed][f"{metric}_mean"]) for seed in SEEDS]
                    second = [float(rows_b[seed][f"{metric}_mean"]) for seed in SEEDS]
                    comparisons.append({
                        "statistical_family": family,
                        "regime": regime,
                        "robots": robots,
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "test": "exact paired sign-permutation across training seeds",
                        "n_paired_seeds": len(SEEDS),
                        **paired_effects(first, second),
                        "p_value": permutation_pvalue(
                            first, second, seed=55000 + robots * 100 + metric_index
                        ),
                    })

    write_csv(OUT / "aggregate_review_results.csv", descriptive)
    write_csv(OUT / "review_paired_tests.csv", holm_adjust(comparisons))
    print(f"Wrote {OUT / 'aggregate_review_results.csv'}")
    print(f"Wrote {OUT / 'review_paired_tests.csv'}")


if __name__ == "__main__":
    main()
