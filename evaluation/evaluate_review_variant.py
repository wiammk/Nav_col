"""Evaluate one review-response policy on a previously saved protocol."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.graph_env import make_env
from evaluation.common_protocol import (
    evaluate_multi_policies,
    evaluate_single_policy,
    load_protocol,
    summarize,
)
from evaluation.compare_qlearning import (
    load_qtable,
    qtable_compatibility_error,
)


def qtable_action(qtable, env, obs):
    valid = np.flatnonzero(np.asarray(obs["mask"], dtype=bool))
    if not len(valid):
        return 0
    current = env.node2idx[env.current_node]
    target = env.node2idx[env.target_node]
    values = qtable[current, target]
    return int(valid[np.argmax(values[valid])])


def shortest_path_action(env, obs):
    """Choose a currently valid neighbor with minimum weighted distance to goal."""
    valid = np.flatnonzero(np.asarray(obs["mask"], dtype=bool))
    if not len(valid):
        return 0
    neighbors = env.neighbors_map[env.current_node]
    target = env.target_node
    cache_key = (
        target,
        frozenset(env.blocked_nodes),
        frozenset(env.closed_edges),
    )
    cache = getattr(env, "_review_shortest_path_cache", None)
    if cache is None:
        cache = {}
        env._review_shortest_path_cache = cache
    if cache_key not in cache:
        if not env.blocked_nodes and not env.closed_edges and env._sp is not None:
            distances = env._sp.get(target, {})
        else:
            available = env.graph.copy()
            available.remove_nodes_from(
                node for node in env.blocked_nodes if node in available
            )
            available.remove_edges_from([
                tuple(edge) for edge in env.closed_edges
                if available.has_edge(*tuple(edge))
            ])
            available.remove_edges_from([
                (u, v) for u, v, data in available.edges(data=True)
                if not bool(data.get("passable", True))
                or float(data.get("width", 1.0)) < float(env.rcfg["min_door_width"])
            ])
            distances = nx.single_source_dijkstra_path_length(
                available, target, weight="weight"
            )
        cache[cache_key] = distances
    distances = cache[cache_key]
    return int(min(
        valid.tolist(),
        key=lambda action: (
            float(env.graph.get_edge_data(env.current_node, neighbors[action], default={}).get("weight", 1.0))
            + float(distances.get(neighbors[action], float("inf"))),
            action,
        ),
    ))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--robots", required=True, type=int)
    parser.add_argument("--method", required=True, choices=["shortest_path", "qtable"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=33003)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = load_protocol(protocol_path)
    env, graph, _ = make_env(
        args.graph,
        n_robots=args.robots,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    robot_envs = env.envs if hasattr(env, "envs") else [env]

    qtable = None
    if args.method == "qtable":
        if not args.model:
            parser.error("--model est requis pour --method qtable")
        model_path = Path(args.model)
        probe = robot_envs[0]
        qtable = load_qtable(model_path)
        metadata_path = model_path.parent / "run_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Q-table provenance missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        graph_path = str(Path(args.graph).resolve())
        recorded_graph = str(Path(metadata.get("graph_path", "")).resolve())
        if recorded_graph != graph_path:
            raise ValueError(
                f"Q-table was trained on {recorded_graph}, not evaluation graph {graph_path}."
            )
        if metadata.get("graph_fingerprint") != getattr(env, "graph_fingerprint", None):
            raise ValueError("Q-table graph fingerprint does not match the evaluation graph.")
        error = qtable_compatibility_error(qtable, probe, "review Q-table")
        if error:
            raise ValueError(error)

    if args.method == "shortest_path":
        action_fns = [
            (lambda obs, graph_env=graph_env: shortest_path_action(graph_env, obs))
            for graph_env in robot_envs
        ]
    else:
        action_fns = [
            (lambda obs, graph_env=graph_env: qtable_action(qtable, graph_env, obs))
            for graph_env in robot_envs
        ]

    if args.robots == 1:
        scenarios = protocol.get("single", [])
        rows = evaluate_single_policy(
            robot_envs[0],
            lambda obs: action_fns[0](obs),
            scenarios,
        )
        for row in rows:
            row.update({"robots": 1, "robot_id": 0})
    else:
        scenarios = protocol.get("multi", [])
        rows = evaluate_multi_policies(env, action_fns, scenarios)

    summary = summarize(rows)
    regime = protocol.get("regime", "static")
    output_dir = Path(args.output_dir)
    write_csv(output_dir / f"{regime}_detailed.csv", rows)
    summary.update({
        "method": args.method,
        "robots": args.robots,
        "regime": regime,
        "n_scenarios_in_protocol": len(scenarios),
        "n_robot_scenario_rows": len(rows),
        "n_disruptions_applied": int(sum(row.get("disruption_applied", 0) for row in rows)),
        "graph_path": str(Path(args.graph).resolve()),
        "protocol_path": str(protocol_path.resolve()),
        "model_path": str(Path(args.model).resolve()) if args.model else "",
    })
    write_csv(output_dir / f"{regime}_summary.csv", [summary])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
