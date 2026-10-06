"""Regenerate English per-seed diagnostic plots from consolidated CSV files.

This utility reads existing evaluation summaries only. It does not retrain or
re-evaluate any reinforcement learning model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.compare_federated_algorithms import (
    plot_complexity_bars,
    plot_final_bars,
    plot_metric,
)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def regenerate(data_root: Path, plot_root: Path) -> int:
    generated = 0

    for data_dir in sorted(data_root.glob("seed_*/robots_*")):
        rounds_path = data_dir / "fl_algorithms_rounds.csv"
        summary_path = data_dir / "fl_algorithms_summary.csv"
        if not rounds_path.exists() or not summary_path.exists():
            continue

        round_rows = read_csv(rounds_path)
        summary_rows = read_csv(summary_path)
        histories = {}
        for row in round_rows:
            histories.setdefault(row["algo"], []).append(row)
        for rows in histories.values():
            rows.sort(key=lambda row: int(float(row["round"])))

        output_dir = plot_root / data_dir.relative_to(data_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_metric(
            histories,
            "mean_reward",
            "Mean reward",
            "Mean reward per federated round",
            output_dir / "reward_vs_round.png",
        )
        plot_metric(
            histories,
            "success_rate",
            "Success rate",
            "Success rate per federated round",
            output_dir / "success_rate_vs_round.png",
        )
        plot_final_bars(summary_rows, output_dir / "final_comparison.png")
        plot_complexity_bars(
            summary_rows,
            output_dir / "complexity_communication.png",
        )
        generated += 4

    return generated


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="runs/Office_Building/evaluation/learning_curves",
        help="Directory containing seed_*/robots_* CSV summaries.",
    )
    parser.add_argument(
        "--plot-root",
        default="runs/Office_Building/evaluation/plots/learning_curves",
        help="Destination directory for regenerated PNG figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generated = regenerate(Path(args.data_root), Path(args.plot_root))
    print(f"Generated {generated} English diagnostic figures.")


if __name__ == "__main__":
    main()
