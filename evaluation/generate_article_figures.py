"""Generate the two supplementary figures used by the research article.

The script reads only consolidated CSV results. It does not train or evaluate
an agent, so regenerating the figures is fast and deterministic.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


T_95_DF_9 = 2.262157  # Two-sided Student critical value for ten seeds.


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D7DEE8", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_coordination_failures(architecture_csv: Path, output_dir: Path) -> None:
    rows = [
        row
        for row in _read_csv(architecture_csv)
        if row["method"] == "fedavg_qlearning"
    ]
    rows.sort(key=lambda row: int(row["robots"]))

    fleets = np.array([int(row["robots"]) for row in rows])
    collisions = np.array([float(row["collisions_mean"]) for row in rows])
    collisions_ci = np.array(
        [float(row["collisions_ci95_across_seeds"]) for row in rows]
    )
    deadlocks = 100.0 * np.array([float(row["deadlock_mean"]) for row in rows])
    deadlocks_ci = 100.0 * np.array(
        [float(row["deadlock_ci95_across_seeds"]) for row in rows]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    color = "#2F6B9A"
    axes[0].errorbar(
        fleets,
        collisions,
        yerr=collisions_ci,
        color=color,
        marker="o",
        linewidth=2,
        capsize=4,
    )
    axes[0].set_xlabel("Number of robots")
    axes[0].set_ylabel("Mean collision attempts")
    axes[0].set_title("Collision pressure")
    axes[0].set_xticks(fleets)
    axes[0].set_ylim(bottom=0)
    _style_axis(axes[0])

    axes[1].errorbar(
        fleets,
        deadlocks,
        yerr=deadlocks_ci,
        color="#B44C43",
        marker="s",
        linewidth=2,
        capsize=4,
    )
    axes[1].set_xlabel("Number of robots")
    axes[1].set_ylabel("Deadlock rate (\%)")
    axes[1].set_title("Coordination deadlocks")
    axes[1].set_xticks(fleets)
    axes[1].set_ylim(bottom=0)
    _style_axis(axes[1])

    fig.suptitle("Coordination failures of federated Q-learning", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir / "figure_5_coordination_failures")


def _mean_ci(values: list[float]) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=float)
    mean = float(values_array.mean())
    if len(values_array) < 2:
        return mean, 0.0
    standard_error = float(values_array.std(ddof=1) / math.sqrt(len(values_array)))
    return mean, T_95_DF_9 * standard_error


def plot_cross_building_generalization(
    generalization_csv: Path,
    output_dir: Path,
) -> None:
    selected_modes = {
        ("zero_shot", 0): "Zero-shot",
        ("fine_tuning", 100): "Fine-tune\n(100 episodes)",
        ("from_scratch", 100): "Scratch\n(100 episodes)",
    }
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(generalization_csv):
        key = (row["algorithm"], row["mode"], int(row["episodes"]))
        if (key[1], key[2]) in selected_modes:
            grouped[key].append(row)

    categories = list(selected_modes)
    labels = [selected_modes[key] for key in categories]
    algorithms = ["dqn", "ppo"]
    colors = {"dqn": "#2F6B9A", "ppo": "#D17A22"}
    markers = {"dqn": "DQN", "ppo": "PPO"}
    positions = np.arange(len(categories), dtype=float)
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for algorithm_index, algorithm in enumerate(algorithms):
        success_means = []
        success_cis = []
        spl_means = []
        spl_cis = []
        for mode, episodes in categories:
            rows = grouped[(algorithm, mode, episodes)]
            success_mean, success_ci = _mean_ci(
                [100.0 * float(row["success_mean"]) for row in rows]
            )
            spl_mean, spl_ci = _mean_ci([float(row["spl_mean"]) for row in rows])
            success_means.append(success_mean)
            success_cis.append(success_ci)
            spl_means.append(spl_mean)
            spl_cis.append(spl_ci)

        offset = (algorithm_index - 0.5) * width
        axes[0].bar(
            positions + offset,
            success_means,
            width,
            yerr=success_cis,
            capsize=3,
            color=colors[algorithm],
            label=markers[algorithm],
        )
        axes[1].bar(
            positions + offset,
            spl_means,
            width,
            yerr=spl_cis,
            capsize=3,
            color=colors[algorithm],
            label=markers[algorithm],
        )

    axes[0].set_ylabel("Success rate (\%)")
    axes[0].set_title("Navigation success")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("SPL")
    axes[1].set_title("Path efficiency")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Adaptation mode on the Clinic building")
        _style_axis(axis)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    fig.suptitle("Cross-building transfer from Office to Clinic", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir / "figure_6_cross_building_generalization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture-csv",
        type=Path,
        default=Path("runs/Office_Building/final_results/table_A_architecture.csv"),
    )
    parser.add_argument(
        "--generalization-csv",
        type=Path,
        default=Path(
            "runs/Office_Building/generalization/Clinic_Architectural/"
            "all_seeds_detailed.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/Office_Building/final_results/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_coordination_failures(args.architecture_csv, args.output_dir)
    plot_cross_building_generalization(args.generalization_csv, args.output_dir)
    print(f"Generated article figures in {args.output_dir}")


if __name__ == "__main__":
    main()
