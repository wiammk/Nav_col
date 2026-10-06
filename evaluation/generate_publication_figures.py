"""Generate publication-ready figures from the final aggregated CSV tables."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "local_independent_qlearning": "Independent Q-learning",
    "centralized_qlearning": "Centralised Q-learning",
    "fedavg_qlearning": "FedAvg Q-learning",
    "fedavg_dqn": "FedAvg DQN",
    "fedavg_ppo": "FedAvg PPO",
}
COLORS = {
    "local_independent_qlearning": "#6b7280",
    "centralized_qlearning": "#2563eb",
    "fedavg_qlearning": "#16a34a",
    "fedavg_dqn": "#ea580c",
    "fedavg_ppo": "#dc2626",
}
MARKERS = {
    "local_independent_qlearning": "o",
    "centralized_qlearning": "s",
    "fedavg_qlearning": "^",
    "fedavg_dqn": "D",
    "fedavg_ppo": "X",
}
ARCHITECTURE_METHODS = (
    "local_independent_qlearning",
    "centralized_qlearning",
    "fedavg_qlearning",
)
FEDERATED_METHODS = (
    "fedavg_qlearning",
    "fedavg_dqn",
    "fedavg_ppo",
)
ALL_METHODS = (
    "local_independent_qlearning",
    "centralized_qlearning",
    "fedavg_qlearning",
    "fedavg_dqn",
    "fedavg_ppo",
)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def by_method(rows):
    indexed = {}
    for row in rows:
        indexed.setdefault(row["method"], {})[int(row["robots"])] = row
    return indexed


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
    })


def save_figure(fig, output_dir, stem):
    paths = []
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_metric_panel(ax, indexed, methods, metric, ci_metric, ylabel, scale=1.0):
    robots = [1, 3, 5, 10]
    for method in methods:
        records = indexed[method]
        values = np.asarray([
            float(records[robots_count][metric]) * scale
            for robots_count in robots
        ])
        errors = np.asarray([
            float(records[robots_count][ci_metric]) * scale
            for robots_count in robots
        ])
        ax.errorbar(
            robots,
            values,
            yerr=errors,
            label=LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            markersize=5.5,
            capsize=3,
        )
    ax.set_xticks(robots)
    ax.set_xlabel("Number of robots")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)


def scalability_figure(table_a, table_b, metric, output_dir):
    architecture = by_method(table_a)
    federated = by_method(table_b)
    is_success = metric == "success"
    scale = 100.0 if is_success else 1.0
    ylabel = "Success rate (%)" if is_success else "SPL"
    ci_metric = f"{metric}_ci95_across_seeds"
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25), sharey=True)
    plot_metric_panel(
        axes[0], architecture, ARCHITECTURE_METHODS,
        f"{metric}_mean", ci_metric, ylabel, scale,
    )
    axes[0].set_title("Architecture comparison")
    plot_metric_panel(
        axes[1], federated, FEDERATED_METHODS,
        f"{metric}_mean", ci_metric, ylabel, scale,
    )
    axes[1].set_title("Federated-algorithm comparison")
    if is_success:
        axes[0].set_ylim(0, 100)
    else:
        axes[0].set_ylim(0, 1)
    fig.suptitle(
        "Success-rate scalability"
        if is_success else "Path-efficiency scalability"
    )
    fig.tight_layout()
    stem = "figure_1_success_scalability" if is_success else "figure_2_spl_scalability"
    return save_figure(fig, output_dir, stem)


def dynamic_figure(static_rows, dynamic_rows, robustness_tests, output_dir):
    static = by_method(static_rows)
    dynamic = by_method(dynamic_rows)
    test_index = {
        (row["method"], int(row["robots"]), row["metric"]): row
        for row in robustness_tests
    }
    robots = [1, 3, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))

    method = "fedavg_qlearning"
    for regime, indexed, linestyle, marker in (
        ("Static", static, "-", "o"),
        ("Dynamic edge closure", dynamic, "--", "s"),
    ):
        values = [
            100 * float(indexed[method][robots_count]["success_mean"])
            for robots_count in robots
        ]
        errors = [
            100 * float(indexed[method][robots_count]["success_ci95_across_seeds"])
            for robots_count in robots
        ]
        axes[0].errorbar(
            robots, values, yerr=errors, label=regime,
            color=COLORS[method], linestyle=linestyle, marker=marker,
            linewidth=1.8, markersize=5.5, capsize=3,
        )
    axes[0].set_title("FedAvg Q-learning")
    axes[0].set_xlabel("Number of robots")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_xticks(robots)
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False)

    for current_method in ALL_METHODS:
        differences = []
        errors = []
        for robots_count in robots:
            test = test_index[(current_method, robots_count, "success")]
            differences.append(100 * float(test["mean_difference_static_minus_dynamic"]))
            errors.append(100 * float(test["difference_ci95_student"]))
        axes[1].errorbar(
            robots,
            differences,
            yerr=errors,
            label=LABELS[current_method],
            color=COLORS[current_method],
            marker=MARKERS[current_method],
            linewidth=1.6,
            markersize=5,
            capsize=2.5,
        )
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_title("Loss caused by disruption")
    axes[1].set_xlabel("Number of robots")
    axes[1].set_ylabel("Success decrease (percentage points)")
    axes[1].set_xticks(robots)
    axes[1].legend(frameon=False, ncol=1)
    fig.suptitle("Robustness to dynamic edge closures")
    fig.tight_layout()
    return save_figure(fig, output_dir, "figure_3_static_vs_dynamic")


def cost_figure(all_rows, output_dir):
    indexed = by_method(all_rows)
    robots = [1, 3, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))

    for method in ALL_METHODS:
        minutes = [
            max(float(indexed[method][robots_count]["training_seconds_mean"]) / 60, 1e-4)
            for robots_count in robots
        ]
        axes[0].plot(
            robots, minutes, label=LABELS[method], color=COLORS[method],
            marker=MARKERS[method], linewidth=1.8, markersize=5.5,
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Training time")
    axes[0].set_xlabel("Number of robots")
    axes[0].set_ylabel("Minutes (logarithmic scale)")
    axes[0].set_xticks(robots)
    axes[0].legend(frameon=False)

    for method in FEDERATED_METHODS:
        communication_mb = [
            float(indexed[method][robots_count]["communication_bytes_mean"])
            / (1024 ** 2)
            for robots_count in robots
        ]
        axes[1].plot(
            robots, communication_mb, label=LABELS[method], color=COLORS[method],
            marker=MARKERS[method], linewidth=1.8, markersize=5.5,
        )
    axes[1].set_title("Federated communication")
    axes[1].set_xlabel("Number of robots")
    axes[1].set_ylabel("Estimated volume (MiB)")
    axes[1].set_xticks(robots)
    axes[1].legend(frameon=False)
    fig.suptitle("Computational and communication cost")
    fig.tight_layout()
    return save_figure(fig, output_dir, "figure_4_training_communication_cost")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir or results_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    table_a = read_csv(results_dir / "table_A_architecture.csv")
    table_b = read_csv(results_dir / "table_B_federated_algorithms.csv")
    static_rows = read_csv(results_dir / "all_methods_all_scales.csv")
    dynamic_rows = read_csv(results_dir / "dynamic_methods_all_scales.csv")
    robustness_tests = read_csv(
        results_dir / "static_vs_dynamic_tests_across_seeds.csv"
    )

    generated = []
    generated.extend(scalability_figure(table_a, table_b, "success", output_dir))
    generated.extend(scalability_figure(table_a, table_b, "spl", output_dir))
    generated.extend(dynamic_figure(
        static_rows, dynamic_rows, robustness_tests, output_dir
    ))
    generated.extend(cost_figure(static_rows, output_dir))
    manifest = {
        "source": str(results_dir.resolve()),
        "confidence_intervals": "Student 95% across independent training seeds",
        "files": [path.name for path in generated],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
