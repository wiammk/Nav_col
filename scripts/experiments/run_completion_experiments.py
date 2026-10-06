"""Complete T5 without retraining already available centralized/FedAvg models.

For every requested seed and fleet size this runner:
1. trains only the missing independent local Q-learning baseline;
2. re-evaluates all frozen policies on paired static/dynamic scenarios;
3. regenerates final aggregate tables and corrected statistical tests.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG, get_nested


ROOT = PROJECT_ROOT


def run(command, execute=False):
    print("> " + " ".join(map(str, command)), flush=True)
    if execute:
        subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def required_existing_models(base):
    return [
        base / "centralized" / "qlearning" / "global_qtable.pkl",
        base / "fedavg" / "qlearning" / "global_qtable_final.pkl",
        base / "fedavg" / "dqn" / "global_model_final.pt",
        base / "fedavg" / "ppo" / "global_model_final.pt",
    ]


def local_training_is_complete(local_dir, args, seed, robots):
    metadata_path = local_dir / "train_meta.json"
    if not metadata_path.exists() or not all(
        (local_dir / f"robot_{index}.pkl").exists()
        for index in range(robots)
    ):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    hyperparameters = metadata.get("hyperparameters", {})
    return (
        metadata.get("architecture") == "local_independent"
        and metadata.get("algorithm") == "qlearning"
        and metadata.get("n_robots") == robots
        and metadata.get("seed") == seed
        and metadata.get("episodes_per_robot") == args.episodes_per_client
        and metadata.get("max_steps") == args.max_steps
        and metadata.get("heterogeneity") == args.heterogeneity
        and hyperparameters.get("alpha") == args.alpha
        and hyperparameters.get("gamma") == args.gamma
        and hyperparameters.get("epsilon_start") == args.epsilon_start
        and hyperparameters.get("epsilon_min") == args.epsilon_min
        and hyperparameters.get("epsilon_decay") == args.epsilon_decay
    )


def evaluation_is_complete(output_dir, args, seed, robots):
    required = (
        "comparison_summary.csv",
        "paired_statistical_tests.csv",
        "dynamic_comparison_summary.csv",
        "dynamic_paired_statistical_tests.csv",
        "static_vs_dynamic_tests.csv",
        "evaluation_metadata.json",
    )
    if not all((output_dir / name).exists() for name in required):
        return False
    try:
        metadata = json.loads(
            (output_dir / "evaluation_metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    return (
        metadata.get("training_seed") == seed
        and metadata.get("robots") == robots
        and metadata.get("test_seed") == args.test_seed
        and metadata.get("dynamic_seed") == args.dynamic_seed
        and metadata.get("test_scenarios") == args.scenarios
        and "local_independent_qlearning"
        in metadata.get("comparisons", {}).get("architecture", [])
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        default="runs/Office_Building/data/processed/graph.gpickle",
    )
    parser.add_argument("--building", default="Office_Building")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=CONFIG.get("seeds", list(range(42, 52)))
    )
    parser.add_argument(
        "--robot-counts", type=int, nargs="+", default=CONFIG.get("robot_counts", [1, 3, 5, 10])
    )
    parser.add_argument(
        "--episodes-per-client",
        type=int,
        default=(
            get_nested("federated", "rounds", default=20)
            * get_nested("federated", "episodes_per_round", default=50)
        ),
    )
    parser.add_argument("--max-steps", type=int, default=CONFIG.get("max_steps", 200))
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--test-seed", type=int, default=33003)
    parser.add_argument("--dynamic-seed", type=int, default=44004)
    parser.add_argument("--heterogeneity", choices=["iid", "spatial"], default="iid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the displayed commands. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graph = Path(args.graph).resolve()
    if not graph.exists():
        raise SystemExit(f"Graphe introuvable: {graph}")

    experiments = ROOT / "runs" / args.building / "experiments"
    for seed in args.seeds:
        for robots in args.robot_counts:
            base = experiments / f"seed_{seed}" / f"robots_{robots}"
            missing = [path for path in required_existing_models(base) if not path.exists()]
            if missing:
                formatted = "\n".join(f"- {path}" for path in missing)
                raise SystemExit(
                    "Les modeles existants requis sont absents. Lancez la campagne "
                    f"complete au lieu du runner incremental:\n{formatted}"
                )

            local_dir = base / "local_independent" / "qlearning"
            local_complete = local_training_is_complete(
                local_dir, args, seed, robots
            )
            if not (args.resume and local_complete):
                run([
                    sys.executable,
                    "scripts/training/train_qlearning.py",
                    "--graph", graph,
                    "--robots", robots,
                    "--episodes", args.episodes_per_client,
                    "--max-steps", args.max_steps,
                    "--seed", seed,
                    "--save-dir", local_dir,
                    "--history", local_dir / "history.csv",
                    "--heterogeneity", args.heterogeneity,
                    "--alpha", args.alpha,
                    "--gamma", args.gamma,
                    "--epsilon-start", args.epsilon_start,
                    "--epsilon-min", args.epsilon_min,
                    "--epsilon-decay", args.epsilon_decay,
                ], execute=args.execute)

            evaluation_dir = base / "evaluation"
            if not (
                args.resume
                and evaluation_is_complete(evaluation_dir, args, seed, robots)
            ):
                run([
                    sys.executable,
                    "evaluation/evaluate_professor_protocol.py",
                    "--graph", graph,
                    "--local-q-dir", local_dir,
                    "--centralized-q", base / "centralized" / "qlearning" / "global_qtable.pkl",
                    "--fedavg-q", base / "fedavg" / "qlearning" / "global_qtable_final.pkl",
                    "--fedavg-dqn", base / "fedavg" / "dqn" / "global_model_final.pt",
                    "--fedavg-ppo", base / "fedavg" / "ppo" / "global_model_final.pt",
                    "--output-dir", evaluation_dir,
                    "--robots", robots,
                    "--scenarios", args.scenarios,
                    "--max-steps", args.max_steps,
                    "--test-seed", args.test_seed,
                    "--dynamic-seed", args.dynamic_seed,
                    "--training-seed", seed,
                ], execute=args.execute)

    run([
        sys.executable,
        "evaluation/aggregate_professor_results.py",
        "--experiments-dir", experiments,
        "--output-dir", ROOT / "runs" / args.building / "final_results",
    ], execute=args.execute)

    if not args.execute:
        print("\nPlan uniquement. Ajoutez --execute pour lancer la campagne.")


if __name__ == "__main__":
    main()
