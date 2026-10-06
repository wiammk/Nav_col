"""Train independent local Q-learning policies without model sharing.

Each robot receives the same per-client trajectory budget as a FedAvg client,
but its Q-table is never broadcast or aggregated.  The resulting models form
the literal ``local learning vs federated learning`` baseline requested in T5.
"""

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG, get_nested
from config.run_layout import layout_for_graph
from environment.graph_env import make_env
from federated.client import QLearningClient
from federated.multi_robot_training import train_clients_multi_robot


def _configure_spatial_profiles(clients, ordered_nodes):
    partitions = np.array_split(
        np.asarray(ordered_nodes, dtype=object), len(clients)
    )
    for index, client in enumerate(clients):
        local_nodes = partitions[index].tolist()
        if len(local_nodes) < 2:
            local_nodes = ordered_nodes
        client.configure_data_profile(
            start_nodes=local_nodes,
            target_nodes=local_nodes,
            label=f"spatial_partition_{index}",
        )


def train_qlearning_multi(
    graph_path: str = CONFIG["graph_path"],
    n_robots: int = CONFIG["n_robots"],
    episodes: int = get_nested("qlearning", "episodes", default=200),
    max_steps: int = CONFIG["max_steps"],
    save_dir: str | None = None,
    history_path: str | None = None,
    seed: int = CONFIG["seed"],
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    heterogeneity: str = "iid",
):
    if n_robots <= 0:
        raise ValueError("n_robots doit etre strictement positif.")
    if episodes <= 0:
        raise ValueError("episodes doit etre strictement positif.")
    if heterogeneity not in {"iid", "spatial"}:
        raise ValueError("heterogeneity doit etre 'iid' ou 'spatial'.")

    layout = layout_for_graph(graph_path)
    save_dir = Path(save_dir) if save_dir else layout.local_models / "qlearning"
    history_path = Path(
        history_path
        or layout.results / "local" / "qlearning" / "local_history.csv"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    environment, _, _ = make_env(
        graph_path,
        n_robots=n_robots,
        max_steps=max_steps,
        seed=seed,
    )
    robot_envs = environment.envs if hasattr(environment, "envs") else [environment]
    clients = [
        QLearningClient(
            robot_env,
            robot_id=index,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed + index,
        )
        for index, robot_env in enumerate(robot_envs)
    ]

    if heterogeneity == "spatial":
        ordered_nodes = sorted(
            robot_envs[0].nodes,
            key=lambda node: (
                float(robot_envs[0].graph.nodes[node].get("x_norm", 0.0)),
                float(robot_envs[0].graph.nodes[node].get("y_norm", 0.0)),
                str(node),
            ),
        )
        _configure_spatial_profiles(clients, ordered_nodes)

    started_at = time.perf_counter()
    if hasattr(environment, "envs"):
        results = train_clients_multi_robot(
            clients,
            environment,
            episodes_per_client=episodes,
        )
    else:
        results = [clients[0].train_local(episodes)]
        results[0].setdefault("collisions", 0)
    elapsed_seconds = time.perf_counter() - started_at

    history_rows = []
    for episode_index in range(episodes):
        rewards = [result["rewards"][episode_index] for result in results]
        successes = [result["successes"][episode_index] for result in results]
        row = {
            "episode": episode_index + 1,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "n_robots": n_robots,
        }
        for robot_id, (reward, success) in enumerate(zip(rewards, successes)):
            row[f"robot_{robot_id}_reward"] = float(reward)
            row[f"robot_{robot_id}_success"] = float(success)
        history_rows.append(row)

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)

    model_paths = []
    for index, client in enumerate(clients):
        model_path = save_dir / f"robot_{index}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(
                {
                    "Q": client.Q,
                    "visit_counts": client.visit_counts,
                    "alpha": client.alpha,
                    "gamma": client.gamma,
                    "epsilon": client.epsilon,
                },
                handle,
            )
        model_paths.append(model_path)

    total_transitions = int(sum(result.get("n_transitions", 0) for result in results))
    total_collisions = int(sum(result.get("collisions", 0) for result in results))
    metadata = {
        "architecture": "local_independent",
        "algorithm": "qlearning",
        "graph_path": str(Path(graph_path).resolve()),
        "graph_fingerprint": environment.graph_fingerprint,
        "n_robots": n_robots,
        "seed": seed,
        "episodes_per_robot": episodes,
        "total_trajectories": n_robots * episodes,
        "total_transitions": total_transitions,
        "training_collisions": total_collisions,
        "elapsed_seconds": elapsed_seconds,
        "estimated_communication_bytes": 0,
        "max_steps": max_steps,
        "heterogeneity": heterogeneity,
        "hyperparameters": {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
        },
        "history_path": str(history_path.resolve()),
        "models": [path.name for path in model_paths],
        "model_size_bytes": int(sum(path.stat().st_size for path in model_paths)),
    }
    (save_dir / "train_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (save_dir / "robots_list.json").write_text(
        json.dumps(metadata["models"], indent=2), encoding="utf-8"
    )

    print(f"Modeles locaux independants sauvegardes: {save_dir}")
    print(f"Historique local sauvegarde: {history_path}")
    return clients


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train independent local multi-robot Q-learning baselines."
    )
    parser.add_argument("--graph", type=str, default=CONFIG["graph_path"])
    parser.add_argument("--robots", type=int, default=CONFIG["n_robots"])
    parser.add_argument(
        "--episodes",
        type=int,
        default=get_nested("qlearning", "episodes", default=200),
        help="Trajectoires d'entrainement par robot.",
    )
    parser.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--history", type=str, default=None)
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument(
        "--epsilon-min", "--epsilon-end", dest="epsilon_min", type=float, default=0.05
    )
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument(
        "--heterogeneity", choices=["iid", "spatial"], default="iid"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_qlearning_multi(
        graph_path=args.graph,
        n_robots=args.robots,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        history_path=args.history,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        heterogeneity=args.heterogeneity,
    )
