"""Aggregate final experiment results across independent training seeds."""

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "success",
    "spl",
    "reward",
    "steps",
    "collisions",
    "deadlock",
    "cumulative_risk",
    "detour",
)
PRIMARY_METRICS = {"success", "spl"}

# Two-sided 95% Student critical values. The final protocol uses ten seeds
# (df=9), but the complete table keeps the helper correct for smaller studies.
T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def holm_adjust(
    rows,
    p_key="p_value_paired_permutation",
    group_keys=("robots", "method_a", "method_b", "statistical_family"),
):
    """Apply Holm inside each pre-declared family, never across all tests."""
    adjusted = [dict(row) for row in rows]
    groups = defaultdict(list)
    for index, row in enumerate(adjusted):
        groups[tuple(row.get(key, "") for key in group_keys)].append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(adjusted[index][p_key]))
        running_max = 0.0
        total = len(ordered)
        for rank, index in enumerate(ordered):
            raw = float(adjusted[index][p_key])
            running_max = max(running_max, min(1.0, (total - rank) * raw))
            adjusted[index]["p_value_holm"] = running_max
            adjusted[index]["significant_holm_0_05"] = int(running_max < 0.05)
            adjusted[index]["holm_family_size"] = total
    return adjusted


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean()) if len(values) else 0.0
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    degrees_of_freedom = len(values) - 1
    critical = (
        T_CRITICAL_95.get(degrees_of_freedom, 1.96)
        if degrees_of_freedom > 0 else 0.0
    )
    ci95 = critical * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, std, ci95


def permutation_pvalue(first, second, seed=9876, permutations=50_000):
    """Two-sided paired sign-permutation test, exact whenever feasible."""
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    differences = differences[np.isfinite(differences)]
    observed = abs(float(differences.sum()))
    if len(differences) == 0 or observed == 0:
        return 1.0
    nonzero = differences[~np.isclose(differences, 0.0)]
    if len(nonzero) <= 20:
        extreme = 0
        total = 2 ** len(nonzero)
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            permuted = abs(float(np.sum(nonzero * np.asarray(signs))))
            extreme += permuted >= observed - 1e-15
        return float(extreme / total)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=len(nonzero))
        extreme += abs(float((nonzero * signs).sum())) >= observed - 1e-15
    return float((extreme + 1) / (permutations + 1))


def paired_effects(first, second):
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    differences = differences[np.isfinite(differences)]
    mean, std, ci95 = mean_std_ci95(differences)
    positive = int(np.sum(differences > 0))
    negative = int(np.sum(differences < 0))
    nonzero = positive + negative
    return {
        "mean_difference": mean,
        "difference_ci95_student": ci95,
        "cohen_dz": mean / std if std > 0 else "",
        "paired_rank_biserial": (
            (positive - negative) / nonzero if nonzero else 0.0
        ),
        "positive_seed_differences": positive,
        "negative_seed_differences": negative,
        "zero_seed_differences": int(len(differences) - nonzero),
    }


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["robots"]), row["method"])].append(row)

    output = []
    for (robots, method), group in sorted(grouped.items()):
        result = {
            "robots": robots,
            "method": method,
            "label": group[0]["label"],
            "architecture": group[0]["architecture"],
            "algorithm": group[0]["algorithm"],
            "n_seeds": len({row["training_seed"] for row in group}),
        }
        for metric in METRICS:
            values = [float(row[f"{metric}_mean"]) for row in group]
            mean, std, ci95 = mean_std_ci95(values)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std_across_seeds"] = std
            result[f"{metric}_ci95_across_seeds"] = ci95
        for field in (
            "training_seconds",
            "total_trajectories",
            "total_transitions",
            "model_size_bytes",
            "communication_bytes",
        ):
            values = [
                float(row[field])
                for row in group
                if row.get(field) not in ("", None)
            ]
            result[f"{field}_mean"] = float(np.mean(values)) if values else ""
        output.append(result)
    return output


def across_seed_tests(rows):
    by_key = {
        (int(row["robots"]), int(row["training_seed"]), row["method"]): row
        for row in rows
    }
    robot_counts = sorted({key[0] for key in by_key})
    comparisons = [
        ("local_vs_centralized", "local_independent_qlearning", "centralized_qlearning"),
        ("local_vs_federated", "local_independent_qlearning", "fedavg_qlearning"),
        ("architecture", "centralized_qlearning", "fedavg_qlearning"),
        ("federated_algorithms", "fedavg_qlearning", "fedavg_dqn"),
        ("federated_algorithms", "fedavg_qlearning", "fedavg_ppo"),
        ("federated_algorithms", "fedavg_dqn", "fedavg_ppo"),
    ]
    output = []
    for robots in robot_counts:
        for comparison_index, (family, method_a, method_b) in enumerate(comparisons):
            seeds_a = {
                seed for r, seed, method in by_key
                if r == robots and method == method_a
            }
            seeds_b = {
                seed for r, seed, method in by_key
                if r == robots and method == method_b
            }
            seeds = sorted(seeds_a & seeds_b)
            for metric_index, metric in enumerate(METRICS):
                first = [
                    float(by_key[(robots, seed, method_a)][f"{metric}_mean"])
                    for seed in seeds
                ]
                second = [
                    float(by_key[(robots, seed, method_b)][f"{metric}_mean"])
                    for seed in seeds
                ]
                effects = paired_effects(first, second)
                output.append({
                    "comparison_family": family,
                    "statistical_family": (
                        "primary_navigation"
                        if metric in PRIMARY_METRICS
                        else "secondary_operational"
                    ),
                    "robots": robots,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "n_paired_seeds": len(seeds),
                    "test": (
                        "exact paired sign-permutation"
                        if len(seeds) <= 20
                        else "Monte Carlo paired sign-permutation"
                    ),
                    **effects,
                    "p_value_paired_permutation": permutation_pvalue(
                        first,
                        second,
                        seed=9876 + 100 * comparison_index + metric_index,
                    ),
                })
    return holm_adjust(output)


def static_dynamic_across_seed_tests(static_rows, dynamic_rows):
    static_by_key = {
        (int(row["robots"]), int(row["training_seed"]), row["method"]): row
        for row in static_rows
    }
    dynamic_by_key = {
        (int(row["robots"]), int(row["training_seed"]), row["method"]): row
        for row in dynamic_rows
    }
    common_keys = sorted(set(static_by_key) & set(dynamic_by_key))
    robot_counts = sorted({key[0] for key in common_keys})
    methods = sorted({key[2] for key in common_keys})
    output = []
    for robots in robot_counts:
        for method_index, method in enumerate(methods):
            seeds = sorted(
                seed for r, seed, current_method in common_keys
                if r == robots and current_method == method
            )
            for metric_index, metric in enumerate(METRICS):
                static_values = [
                    float(static_by_key[(robots, seed, method)][f"{metric}_mean"])
                    for seed in seeds
                ]
                dynamic_values = [
                    float(dynamic_by_key[(robots, seed, method)][f"{metric}_mean"])
                    for seed in seeds
                ]
                effects = paired_effects(static_values, dynamic_values)
                output.append({
                    "comparison_family": "static_vs_dynamic",
                    "statistical_family": (
                        "primary_navigation"
                        if metric in PRIMARY_METRICS
                        else "secondary_operational"
                    ),
                    "robots": robots,
                    "method": method,
                    "metric": metric,
                    "n_paired_seeds": len(seeds),
                    "test": (
                        "exact paired sign-permutation"
                        if len(seeds) <= 20
                        else "Monte Carlo paired sign-permutation"
                    ),
                    **effects,
                    "mean_difference_static_minus_dynamic": effects["mean_difference"],
                    "p_value_paired_permutation": permutation_pvalue(
                        static_values,
                        dynamic_values,
                        seed=24680 + 100 * method_index + metric_index,
                    ),
                })
    return holm_adjust(
        output,
        group_keys=("robots", "method", "statistical_family"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(
        Path(args.experiments_dir).glob(
            "seed_*/robots_*/evaluation/comparison_summary.csv"
        )
    )
    if not files:
        raise SystemExit(
            f"Aucun comparison_summary.csv trouve sous {args.experiments_dir}"
        )
    source_rows = [row for path in files for row in read_csv(path)]
    aggregate_rows = aggregate(source_rows)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "all_methods_all_scales.csv", aggregate_rows)
    write_csv(
        output_dir / "table_A_architecture.csv",
        [
            row for row in aggregate_rows
            if row["method"] in {
                "local_independent_qlearning",
                "centralized_qlearning",
                "fedavg_qlearning",
            }
        ],
    )
    write_csv(
        output_dir / "table_B_federated_algorithms.csv",
        [
            row for row in aggregate_rows
            if row["method"] in {
                "fedavg_qlearning",
                "fedavg_dqn",
                "fedavg_ppo",
            }
        ],
    )
    static_tests = across_seed_tests(source_rows)
    write_csv(output_dir / "paired_tests_across_seeds.csv", static_tests)
    write_csv(
        output_dir / "primary_paired_tests_across_seeds.csv",
        [
            row for row in static_tests
            if row["statistical_family"] == "primary_navigation"
        ],
    )
    dynamic_files = sorted(
        Path(args.experiments_dir).glob(
            "seed_*/robots_*/evaluation/dynamic_comparison_summary.csv"
        )
    )
    if dynamic_files:
        dynamic_source_rows = [
            row for path in dynamic_files for row in read_csv(path)
        ]
        dynamic_aggregate_rows = aggregate(dynamic_source_rows)
        write_csv(
            output_dir / "dynamic_methods_all_scales.csv",
            dynamic_aggregate_rows,
        )
        write_csv(
            output_dir / "table_C_dynamic_robustness.csv",
            dynamic_aggregate_rows,
        )
        dynamic_tests = across_seed_tests(dynamic_source_rows)
        write_csv(
            output_dir / "dynamic_paired_tests_across_seeds.csv",
            dynamic_tests,
        )
        write_csv(
            output_dir / "dynamic_primary_paired_tests_across_seeds.csv",
            [
                row for row in dynamic_tests
                if row["statistical_family"] == "primary_navigation"
            ],
        )
        robustness_tests = static_dynamic_across_seed_tests(
            source_rows, dynamic_source_rows
        )
        write_csv(
            output_dir / "static_vs_dynamic_tests_across_seeds.csv",
            robustness_tests,
        )
        write_csv(
            output_dir / "static_vs_dynamic_primary_tests.csv",
            [
                row for row in robustness_tests
                if row["statistical_family"] == "primary_navigation"
            ],
        )
    print(f"Table A: {output_dir / 'table_A_architecture.csv'}")
    print(f"Table B: {output_dir / 'table_B_federated_algorithms.csv'}")
    if dynamic_files:
        print(f"Table C: {output_dir / 'table_C_dynamic_robustness.csv'}")


if __name__ == "__main__":
    main()
