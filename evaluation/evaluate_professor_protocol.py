"""Evaluate supervisor comparisons on paired static and dynamic test sets.

Comparison A (architecture):
    independent local Q-learning vs centralized Q-learning vs FedAvg Q-learning.

Comparison B (algorithm, federated architecture only):
    FedAvg Q-learning vs FedAvg DQN vs FedAvg PPO.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.graph_env import make_env
from evaluation.common_protocol import (
    add_dynamic_disruptions,
    evaluate_multi_policies,
    evaluate_single_policy,
    generate_protocol,
    save_protocol,
    summarize,
)
from evaluation.compare_qlearning import (
    load_qtable,
    provenance_compatibility_error,
    qtable_compatibility_error,
)
from scripts.visualization.visualize_federated_navigation import (
    choose_action,
    load_policy,
    validate_model_provenance,
)


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


def holm_adjust(
    rows,
    p_key="p_value",
    group_keys=("method_a", "method_b", "statistical_family"),
):
    """Add Holm-Bonferroni p-values inside each planned test family."""
    adjusted = [dict(row) for row in rows]
    groups = {}
    for index, row in enumerate(adjusted):
        key = tuple(row.get(field, "") for field in group_keys)
        groups.setdefault(key, []).append(index)
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


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_metadata(model_path: Path) -> dict:
    if model_path.is_dir():
        for name in ("run_metadata.json", "train_meta.json"):
            path = model_path / name
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}
    for name in ("run_metadata.json", "train_meta.json"):
        path = model_path.parent / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def model_specifications(args) -> list[dict]:
    return [
        {
            "method": "local_independent_qlearning",
            "label": "Q-learning local independant",
            "architecture": "local_independent",
            "algorithm": "qlearning",
            "path": Path(args.local_q_dir),
            "local_ensemble": True,
        },
        {
            "method": "centralized_qlearning",
            "label": "Q-learning centralise",
            "architecture": "centralized",
            "algorithm": "qlearning",
            "path": Path(args.centralized_q),
        },
        {
            "method": "fedavg_qlearning",
            "label": "FedAvg + Q-learning",
            "architecture": "fedavg",
            "algorithm": "qlearning",
            "path": Path(args.fedavg_q),
        },
        {
            "method": "fedavg_dqn",
            "label": "FedAvg + DQN",
            "architecture": "fedavg",
            "algorithm": "dqn",
            "path": Path(args.fedavg_dqn),
        },
        {
            "method": "fedavg_ppo",
            "label": "FedAvg + PPO",
            "architecture": "fedavg",
            "algorithm": "ppo",
            "path": Path(args.fedavg_ppo),
        },
    ]


def load_and_validate(spec: dict, probe_env, robots: int):
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(f"Modele absent pour {spec['label']}: {path}")
    if spec.get("local_ensemble"):
        policies = []
        for robot_id in range(robots):
            model_path = path / f"robot_{robot_id}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Modele local absent pour robot {robot_id}: {model_path}"
                )
            policy = load_qtable(model_path)
            error = provenance_compatibility_error(
                model_path, probe_env, f"{spec['label']} robot {robot_id}"
            )
            error = error or qtable_compatibility_error(
                policy, probe_env, f"{spec['label']} robot {robot_id}"
            )
            if error:
                raise ValueError(error)
            policies.append(policy)
        return policies
    if spec["algorithm"] == "qlearning":
        policy = load_qtable(path)
        error = provenance_compatibility_error(path, probe_env, spec["label"])
        error = error or qtable_compatibility_error(policy, probe_env, spec["label"])
        if error:
            raise ValueError(error)
        return policy
    validate_model_provenance(path, probe_env, spec["algorithm"])
    return load_policy(
        spec["algorithm"],
        path,
        emb_dim=probe_env.emb_dim,
        hidden_dim=128,
    )


def evaluate_method(spec, policy, graph_path, protocol, robots, max_steps, seed):
    def policy_for(robot_id):
        return policy[robot_id] if spec.get("local_ensemble") else policy

    if robots == 1:
        env, _, _ = make_env(
            graph_path, n_robots=1, max_steps=max_steps, seed=seed
        )
        action_fn = lambda obs: choose_action(
            policy_for(0), spec["algorithm"], env, obs, env.emb_dim
        )
        rows = evaluate_single_policy(env, action_fn, protocol["single"])
        for row in rows:
            row.update({"robots": 1, "robot_id": 0})
        return rows

    multi_env, _, _ = make_env(
        graph_path, n_robots=robots, max_steps=max_steps, seed=seed
    )
    action_fns = [
        (
            lambda obs, graph_env=graph_env, robot_id=robot_id: choose_action(
                policy_for(robot_id),
                spec["algorithm"],
                graph_env,
                obs,
                graph_env.emb_dim,
            )
        )
        for robot_id, graph_env in enumerate(multi_env.envs)
    ]
    return evaluate_multi_policies(multi_env, action_fns, protocol["multi"])


def model_size_bytes(spec):
    path = spec["path"]
    if path.is_dir():
        return int(sum(file.stat().st_size for file in path.glob("robot_*.pkl")))
    return int(path.stat().st_size)


def paired_permutation_test(values_a, values_b, seed, permutations=10_000):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    differences = a - b
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n == 0:
        return {"n_pairs": 0, "mean_difference": 0.0, "effect_size_dz": 0.0, "p_value": 1.0}
    observed = abs(float(differences.mean()))
    std = float(differences.std(ddof=1)) if n > 1 else 0.0
    effect = float(differences.mean() / std) if std > 0 else 0.0
    if observed == 0.0:
        p_value = 1.0
    else:
        rng = np.random.default_rng(seed)
        extreme = 0
        for _ in range(permutations):
            signs = rng.choice((-1.0, 1.0), size=n)
            extreme += abs(float((differences * signs).mean())) >= observed
        p_value = (extreme + 1) / (permutations + 1)
    return {
        "n_pairs": n,
        "mean_difference": float(differences.mean()),
        "effect_size_dz": effect,
        "p_value": float(p_value),
    }


def exact_mcnemar(values_a, values_b):
    a = np.asarray(values_a, dtype=int)
    b = np.asarray(values_b, dtype=int)
    a_wins = int(np.sum((a == 1) & (b == 0)))
    b_wins = int(np.sum((a == 0) & (b == 1)))
    discordant = a_wins + b_wins
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(a_wins, b_wins)
        tail = sum(
            math.comb(discordant, index)
            for index in range(lower + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "n_pairs": int(len(a)),
        "mean_difference": float(np.mean(a - b)) if len(a) else 0.0,
        "effect_size_dz": 0.0,
        "p_value": float(p_value),
        "discordant_a_wins": a_wins,
        "discordant_b_wins": b_wins,
    }


def paired_rows(rows_by_method, method_a, method_b, metric):
    def indexed(method):
        return {
            (row["scenario_id"], row.get("robot_id", 0)): float(row[metric])
            for row in rows_by_method[method]
        }

    first = indexed(method_a)
    second = indexed(method_b)
    keys = sorted(set(first) & set(second))
    return [first[key] for key in keys], [second[key] for key in keys]


def statistical_comparisons(rows_by_method, seed):
    comparisons = [
        ("local_vs_centralized", "local_independent_qlearning", "centralized_qlearning"),
        ("local_vs_federated", "local_independent_qlearning", "fedavg_qlearning"),
        ("architecture", "centralized_qlearning", "fedavg_qlearning"),
        ("federated_algorithms", "fedavg_qlearning", "fedavg_dqn"),
        ("federated_algorithms", "fedavg_qlearning", "fedavg_ppo"),
        ("federated_algorithms", "fedavg_dqn", "fedavg_ppo"),
    ]
    results = []
    for comparison_index, (family, method_a, method_b) in enumerate(comparisons):
        for metric_index, metric in enumerate(METRICS):
            values_a, values_b = paired_rows(
                rows_by_method, method_a, method_b, metric
            )
            if metric == "success":
                test = exact_mcnemar(values_a, values_b)
                test_name = "McNemar exact"
            else:
                test = paired_permutation_test(
                    values_a,
                    values_b,
                    seed + 100 * comparison_index + metric_index,
                )
                test_name = "paired sign-permutation"
            results.append({
                "comparison_family": family,
                "statistical_family": (
                    "primary_navigation"
                    if metric in PRIMARY_METRICS
                    else "secondary_operational"
                ),
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "test": test_name,
                **test,
                "significant_0_05": int(test["p_value"] < 0.05),
            })
    return holm_adjust(results)


def static_dynamic_comparisons(static_rows, dynamic_rows, seed):
    """Paired degradation tests for each frozen method."""
    results = []
    for method_index, method in enumerate(sorted(static_rows)):
        for metric_index, metric in enumerate(METRICS):
            values_static, values_dynamic = paired_rows(
                {
                    f"{method}_static": static_rows[method],
                    f"{method}_dynamic": dynamic_rows[method],
                },
                f"{method}_static",
                f"{method}_dynamic",
                metric,
            )
            if metric == "success":
                test = exact_mcnemar(values_static, values_dynamic)
                test_name = "McNemar exact"
            else:
                test = paired_permutation_test(
                    values_static,
                    values_dynamic,
                    seed + 100 * method_index + metric_index,
                )
                test_name = "paired sign-permutation"
            results.append({
                "comparison_family": "static_vs_dynamic",
                "statistical_family": (
                    "primary_navigation"
                    if metric in PRIMARY_METRICS
                    else "secondary_operational"
                ),
                "method": method,
                "regime_a": "static",
                "regime_b": "dynamic_edge_closure",
                "metric": metric,
                "test": test_name,
                **test,
                "significant_0_05": int(test["p_value"] < 0.05),
            })
    return holm_adjust(
        results,
        group_keys=("method", "statistical_family"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--local-q-dir", required=True)
    parser.add_argument("--centralized-q", required=True)
    parser.add_argument("--fedavg-q", required=True)
    parser.add_argument("--fedavg-dqn", required=True)
    parser.add_argument("--fedavg-ppo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robots", type=int, required=True)
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--test-seed", type=int, default=33003)
    parser.add_argument("--dynamic-seed", type=int, default=44004)
    parser.add_argument("--training-seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe, graph, _ = make_env(
        args.graph, n_robots=1, max_steps=args.max_steps, seed=args.test_seed
    )
    if args.robots > 1 and 2 * args.robots > len(graph):
        raise ValueError(
            f"Le graphe contient {len(graph)} noeuds, insuffisant pour "
            f"{args.robots} departs et cibles distincts."
        )
    static_protocol = generate_protocol(
        graph,
        seed=args.test_seed,
        n_single=args.scenarios if args.robots == 1 else 0,
        n_multi=args.scenarios if args.robots > 1 else 0,
        robot_counts=[args.robots] if args.robots > 1 else [],
    )
    dynamic_protocol = add_dynamic_disruptions(
        graph, static_protocol, seed=args.dynamic_seed
    )
    protocol_path = output_dir / "fixed_test_protocol.json"
    dynamic_protocol_path = output_dir / "dynamic_test_protocol.json"
    save_protocol(static_protocol, protocol_path)
    save_protocol(dynamic_protocol, dynamic_protocol_path)

    specs = model_specifications(args)
    policies = {
        spec["method"]: load_and_validate(spec, probe, args.robots)
        for spec in specs
    }

    def evaluate_regime(protocol, regime):
        detailed = []
        summaries = []
        rows_by_method = {}
        for spec in specs:
            policy = policies[spec["method"]]
            rows = evaluate_method(
                spec,
                policy,
                args.graph,
                protocol,
                args.robots,
                args.max_steps,
                args.test_seed,
            )
            rows_by_method[spec["method"]] = rows
            for row in rows:
                detailed.append({
                    "training_seed": args.training_seed,
                    "evaluation_regime": regime,
                    "method": spec["method"],
                    "architecture": spec["architecture"],
                    "algorithm": spec["algorithm"],
                    **row,
                })
            metadata = read_metadata(spec["path"])
            summary = summarize(rows)
            summary.update({
                "training_seed": args.training_seed,
                "robots": args.robots,
                "evaluation_regime": regime,
                "method": spec["method"],
                "label": spec["label"],
                "architecture": spec["architecture"],
                "algorithm": spec["algorithm"],
                "n_test_scenarios": args.scenarios,
                "total_trajectories": metadata.get("total_trajectories", ""),
                "total_transitions": metadata.get("total_transitions", ""),
                "training_seconds": metadata.get("elapsed_seconds", ""),
                "model_size_bytes": model_size_bytes(spec),
                "communication_bytes": (
                    metadata.get("estimated_communication_bytes", 0)
                    if spec["architecture"] == "fedavg"
                    else 0
                ),
            })
            summaries.append(summary)
        return detailed, summaries, rows_by_method

    detailed, summaries, rows_by_method = evaluate_regime(
        static_protocol, "static"
    )
    dynamic_detailed, dynamic_summaries, dynamic_rows_by_method = evaluate_regime(
        dynamic_protocol, "dynamic_edge_closure"
    )

    tests = statistical_comparisons(rows_by_method, args.test_seed)
    dynamic_tests = statistical_comparisons(
        dynamic_rows_by_method, args.dynamic_seed
    )
    robustness_tests = static_dynamic_comparisons(
        rows_by_method, dynamic_rows_by_method, args.dynamic_seed
    )
    for collection in (tests, dynamic_tests, robustness_tests):
        for row in collection:
            row.update({
                "training_seed": args.training_seed,
                "robots": args.robots,
            })

    write_csv(output_dir / "comparison_detailed.csv", detailed)
    write_csv(output_dir / "comparison_summary.csv", summaries)
    write_csv(output_dir / "paired_statistical_tests.csv", tests)
    write_csv(output_dir / "dynamic_comparison_detailed.csv", dynamic_detailed)
    write_csv(output_dir / "dynamic_comparison_summary.csv", dynamic_summaries)
    write_csv(output_dir / "dynamic_paired_statistical_tests.csv", dynamic_tests)
    write_csv(output_dir / "static_vs_dynamic_tests.csv", robustness_tests)
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps({
            "graph": str(Path(args.graph).resolve()),
            "robots": args.robots,
            "training_seed": args.training_seed,
            "test_seed": args.test_seed,
            "dynamic_seed": args.dynamic_seed,
            "test_scenarios": args.scenarios,
            "max_steps": args.max_steps,
            "learning_during_evaluation": False,
            "protocol": str(protocol_path.resolve()),
            "dynamic_protocol": str(dynamic_protocol_path.resolve()),
            "comparisons": {
                "architecture": [
                    "local_independent_qlearning",
                    "centralized_qlearning",
                    "fedavg_qlearning",
                ],
                "federated_algorithms": [
                    "fedavg_qlearning",
                    "fedavg_dqn",
                    "fedavg_ppo",
                ],
                "robustness": ["static", "dynamic_edge_closure"],
            },
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Tableaux demandes: {output_dir / 'comparison_summary.csv'}")
    print(f"Tests statistiques: {output_dir / 'paired_statistical_tests.csv'}")
    print(f"Robustesse dynamique: {output_dir / 'dynamic_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
