"""Frozen multi-robot scalability evaluation for a global Q-learning policy."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.graph_env import make_env
from evaluation.common_protocol import (
    evaluate_multi_policies,
    evaluate_single_policy,
    generate_protocol,
    save_protocol,
    summarize,
)
from evaluation.compare_qlearning import (
    greedy_action,
    load_qtable,
    provenance_compatibility_error,
    qtable_compatibility_error,
)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def action_fn(qtable, env):
    return lambda obs: greedy_action(
        qtable, env, env.current_node, env.target_node, obs["mask"]
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robot-counts", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qtable = load_qtable(Path(args.model))

    probe, graph, _ = make_env(args.graph, n_robots=1, max_steps=args.max_steps, seed=args.seed)
    error = qtable_compatibility_error(qtable, probe, "modele global FedAvg")
    error = error or provenance_compatibility_error(Path(args.model), probe, "modele global FedAvg")
    if error:
        raise ValueError(error)

    requested = sorted(set(args.robot_counts))
    supported = [count for count in requested if count > 0 and 2 * count <= len(graph)]
    skipped = [count for count in requested if count not in supported]
    if not supported:
        raise ValueError("Aucun nombre de robots compatible avec la taille du graphe.")

    protocol = generate_protocol(
        graph,
        seed=args.seed + 7000,
        n_single=args.scenarios if 1 in supported else 0,
        n_multi=args.scenarios,
        robot_counts=[count for count in supported if count > 1],
    )
    save_protocol(protocol, output_dir / "scalability_protocol.json")

    detailed = []
    summaries = []
    if 1 in supported:
        rows = evaluate_single_policy(probe, action_fn(qtable, probe), protocol["single"])
        for row in rows:
            row.update({"robots": 1, "robot_id": 0})
        detailed.extend(rows)
        summary = summarize(rows)
        summary["robots"] = 1
        summaries.append(summary)

    for count in supported:
        if count == 1:
            continue
        multi_env, _, _ = make_env(args.graph, n_robots=count, max_steps=args.max_steps, seed=args.seed)
        rows = evaluate_multi_policies(
            multi_env,
            [action_fn(qtable, env) for env in multi_env.envs],
            protocol["multi"],
        )
        detailed.extend(rows)
        summary = summarize(rows)
        summary["robots"] = count
        summaries.append(summary)

    write_csv(output_dir / "scalability_detailed.csv", detailed)
    write_csv(output_dir / "scalability_summary.csv", summaries)
    (output_dir / "scalability_metadata.json").write_text(json.dumps({
        "graph": str(Path(args.graph).resolve()),
        "model": str(Path(args.model).resolve()),
        "seed": args.seed,
        "scenarios_per_size": args.scenarios,
        "requested_robot_counts": requested,
        "evaluated_robot_counts": supported,
        "skipped_robot_counts": skipped,
        "learning_during_evaluation": False,
    }, indent=2), encoding="utf-8")
    print(f"Scalability summary: {output_dir / 'scalability_summary.csv'}")


if __name__ == "__main__":
    main()
