"""Build or execute the exact experiment matrix requested by the supervisors."""

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


def command_option(command, option):
    return Path(command[command.index(option) + 1])


def command_is_complete(command):
    script = Path(command[1]).as_posix()
    if script.endswith("train_qlearning.py"):
        save_dir = command_option(command, "--save-dir")
        robots = int(command[command.index("--robots") + 1])
        return (
            (save_dir / "train_meta.json").exists()
            and all(
                (save_dir / f"robot_{index}.pkl").exists()
                for index in range(robots)
            )
        )
    if script.endswith("train_centralized_qlearning.py"):
        return command_option(command, "--model").exists()
    if script.endswith("train_federated.py"):
        save_dir = command_option(command, "--save-dir")
        algo = command[command.index("--algo") + 1]
        model = (
            save_dir / "global_qtable_final.pkl"
            if algo == "qlearning"
            else save_dir / "global_model_final.pt"
        )
        return model.exists() and (save_dir / "fl_history.csv").exists()
    if script.endswith("evaluate_professor_protocol.py"):
        output_dir = command_option(command, "--output-dir")
        return (
            (output_dir / "comparison_summary.csv").exists()
            and (output_dir / "paired_statistical_tests.csv").exists()
            and (output_dir / "dynamic_comparison_summary.csv").exists()
        )
    return False


def command_matrix(args, graph: Path):
    commands = []
    records = []
    for seed in args.seeds:
        for robots in args.robot_counts:
            base = (
                ROOT
                / "runs"
                / args.building
                / "experiments"
                / f"seed_{seed}"
                / f"robots_{robots}"
            )
            total_trajectories = robots * args.rounds * args.episodes_per_round
            common_q = [
                "--alpha", str(args.alpha),
                "--gamma", str(args.gamma),
                "--epsilon-start", str(args.epsilon_start),
                "--epsilon-min", str(args.epsilon_min),
                "--epsilon-decay", str(args.epsilon_decay),
            ]

            local_q_dir = base / "local_independent" / "qlearning"
            commands.append([
                sys.executable,
                "scripts/training/train_qlearning.py",
                "--graph", str(graph),
                "--robots", str(robots),
                "--episodes", str(args.rounds * args.episodes_per_round),
                "--max-steps", str(args.max_steps),
                "--seed", str(seed),
                "--save-dir", str(local_q_dir),
                "--history", str(local_q_dir / "history.csv"),
                "--heterogeneity", args.heterogeneity,
                *common_q,
            ])

            centralized_q = [
                sys.executable,
                "scripts/training/train_centralized_qlearning.py",
                "--graph", str(graph),
                "--robots", str(robots),
                "--total-trajectories", str(total_trajectories),
                "--max-steps", str(args.max_steps),
                "--seed", str(seed),
                "--save-dir", str(base / "centralized" / "qlearning"),
                "--model", str(base / "centralized" / "qlearning" / "global_qtable.pkl"),
                "--history", str(base / "centralized" / "qlearning" / "history.csv"),
                *common_q,
            ]
            commands.append(centralized_q)

            fed_paths = {}
            for algo in ("qlearning", "dqn", "ppo"):
                save_dir = base / "fedavg" / algo
                command = [
                    sys.executable,
                    "scripts/training/train_federated.py",
                    "--algo", algo,
                    "--graph", str(graph),
                    "--robots", str(robots),
                    "--rounds", str(args.rounds),
                    "--episodes", str(args.episodes_per_round),
                    "--max-steps", str(args.max_steps),
                    "--eval-episodes", str(args.validation_scenarios),
                    "--validation-seed", str(args.validation_seed),
                    "--seed", str(seed),
                    "--save-every", str(args.save_every),
                    "--heterogeneity", args.heterogeneity,
                    "--save-dir", str(save_dir),
                ]
                if algo == "qlearning":
                    command.extend(common_q)
                elif algo == "dqn":
                    command.extend([
                        "--lr", str(args.dqn_lr),
                        "--gamma", str(args.neural_gamma),
                        "--batch-size", str(args.dqn_batch_size),
                        "--buffer-size", str(args.dqn_buffer_size),
                        "--tau", str(args.dqn_tau),
                    ])
                else:
                    command.extend([
                        "--lr", str(args.ppo_lr),
                        "--gamma", str(args.neural_gamma),
                        "--rollout-len", str(args.ppo_rollout_len),
                        "--minibatch-size", str(args.ppo_minibatch_size),
                        "--ppo-epochs", str(args.ppo_epochs),
                        "--clip-eps", str(args.ppo_clip),
                        "--entropy-coef", str(args.ppo_entropy_coef),
                    ])
                commands.append(command)
                fed_paths[algo] = save_dir

            evaluation_dir = base / "evaluation"
            commands.append([
                sys.executable,
                "evaluation/evaluate_professor_protocol.py",
                "--graph", str(graph),
                "--local-q-dir", str(local_q_dir),
                "--centralized-q", str(base / "centralized" / "qlearning" / "global_qtable.pkl"),
                "--fedavg-q", str(fed_paths["qlearning"] / "global_qtable_final.pkl"),
                "--fedavg-dqn", str(fed_paths["dqn"] / "global_model_final.pt"),
                "--fedavg-ppo", str(fed_paths["ppo"] / "global_model_final.pt"),
                "--output-dir", str(evaluation_dir),
                "--robots", str(robots),
                "--scenarios", str(args.test_scenarios),
                "--max-steps", str(args.max_steps),
                "--test-seed", str(args.test_seed),
                "--dynamic-seed", str(args.dynamic_seed),
                "--training-seed", str(seed),
            ])
            records.append({
                "training_seed": seed,
                "robots": robots,
                "rounds": args.rounds,
                "episodes_per_round_per_client": args.episodes_per_round,
                "total_trajectories_per_method": total_trajectories,
                "test_seed": args.test_seed,
                "dynamic_test_seed": args.dynamic_seed,
                "validation_seed": args.validation_seed,
                "test_scenarios": args.test_scenarios,
                "architecture_comparison": [
                    "local_independent_qlearning",
                    "centralized_qlearning",
                    "fedavg_qlearning",
                ],
                "federated_algorithm_comparison": [
                    "fedavg_qlearning",
                    "fedavg_dqn",
                    "fedavg_ppo",
                ],
                "output_dir": str(base),
            })

    experiments_dir = ROOT / "runs" / args.building / "experiments"
    commands.append([
        sys.executable,
        "evaluation/aggregate_professor_results.py",
        "--experiments-dir", str(experiments_dir),
        "--output-dir", str(ROOT / "runs" / args.building / "final_results"),
    ])
    return commands, records


def parse_args():
    protocol = CONFIG.get("scientific_protocol", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--building", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ignorer les entrainements/evaluations deja termines.",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=CONFIG.get("seeds", list(range(42, 52))),
    )
    parser.add_argument(
        "--robot-counts",
        type=int,
        nargs="+",
        default=CONFIG.get("robot_counts", [1, 3, 5, 10]),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=get_nested("federated", "rounds", default=10),
    )
    parser.add_argument(
        "--episodes-per-round",
        type=int,
        default=get_nested("federated", "episodes_per_round", default=20),
    )
    parser.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    parser.add_argument(
        "--validation-scenarios",
        type=int,
        default=50,
        help="Scenarios fixes suivis pendant l'entrainement; ils ne sont pas le test final.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=protocol.get("validation_seed", 22002),
    )
    parser.add_argument(
        "--test-scenarios",
        type=int,
        default=get_nested("common_evaluation", "single_robot_scenarios", default=200),
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=protocol.get("test_seed", 33003),
    )
    parser.add_argument(
        "--dynamic-seed",
        type=int,
        default=44004,
        help="Seed du protocole dynamique apparie au protocole statique.",
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--heterogeneity", choices=["iid", "spatial"], default="iid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--neural-gamma", type=float, default=0.99)
    parser.add_argument("--dqn-lr", type=float, default=3e-4)
    parser.add_argument("--dqn-batch-size", type=int, default=64)
    parser.add_argument("--dqn-buffer-size", type=int, default=20000)
    parser.add_argument("--dqn-tau", type=float, default=0.005)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-rollout-len", type=int, default=256)
    parser.add_argument("--ppo-minibatch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    graph = Path(args.graph).resolve()
    if not graph.exists():
        raise SystemExit(f"Graphe introuvable: {graph}")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Les seeds doivent etre distinctes.")
    if any(value <= 0 for value in args.robot_counts):
        raise SystemExit("Chaque nombre de robots doit etre strictement positif.")
    commands, records = command_matrix(args, graph)
    manifest_path = Path(
        args.manifest
        or ROOT / "runs" / args.building / "final_experiment_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({
            "protocol_version": 3,
            "purpose": "comparaisons demandees par les encadrants",
            "budget_formula": "robots * rounds * episodes_per_round",
            "learning_during_test": False,
            "commands": commands,
            "experiments": records,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"{len(commands)} commandes ecrites dans {manifest_path}")
    print(
        "Comparaison A: Q-learning centralise vs Q-learning FedAvg\n"
        "Comparaison T5: Q-learning local independant vs centralise vs FedAvg\n"
        "Comparaison B: FedAvg Q-learning vs FedAvg DQN vs FedAvg PPO\n"
        "Robustesse: scenarios statiques vs fermetures dynamiques d'aretes"
    )
    if args.execute:
        for index, command in enumerate(commands, start=1):
            if args.resume and command_is_complete(command):
                print(f"[{index}/{len(commands)}] deja termine, ignore", flush=True)
                continue
            print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
