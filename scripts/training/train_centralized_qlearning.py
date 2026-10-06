import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.q_learning import QLearningAgent
from config import CONFIG, get_nested
from config.run_layout import layout_for_graph
from environment.graph_env import make_env


def train_centralized_qlearning(
    graph_path: str = CONFIG["graph_path"],
    n_robots: int = CONFIG["n_robots"],
    episodes: int = get_nested("centralized", "episodes", default=200),
    max_steps: int = CONFIG["max_steps"],
    save_dir: str | None = None,
    model_path: str | None = None,
    history_path: str | None = None,
    seed: int = CONFIG["seed"],
    total_trajectories: int | None = None,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
):
    if total_trajectories is None:
        total_trajectories = episodes * n_robots
    if total_trajectories <= 0:
        raise ValueError("total_trajectories doit etre strictement positif.")
    if total_trajectories % n_robots != 0:
        raise ValueError(
            "Pour un entrainement centralise multi-robot equitable, "
            "total_trajectories doit etre divisible par n_robots."
        )

    layout = layout_for_graph(graph_path)
    save_dir = str(save_dir or layout.centralized_models)
    model_path = str(model_path or os.path.join(save_dir, "global_qtable.pkl"))
    history_path = str(
        history_path
        or layout.results / "centralized" / "qlearning" / "centralized_history.csv"
    )
    env, G, emb = make_env(
        graph_path,
        n_robots=n_robots,
        max_steps=max_steps,
        seed=seed,
    )
    robot_envs = env.envs if hasattr(env, "envs") else [env]
    reference_env = robot_envs[0]

    agent = QLearningAgent(
        num_nodes=len(reference_env.nodes),
        max_degree=reference_env.max_degree,
        lr=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_min,
        epsilon_decay_steps=1,
        seed=seed,
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    history_rows = []

    joint_episodes = total_trajectories // n_robots
    trajectories_used = 0
    transitions_used = 0
    started_at = time.perf_counter()

    for ep in range(joint_episodes):
        reset_result = env.reset()
        obs_list = (
            reset_result
            if hasattr(env, "envs")
            else [reset_result[0]]
        )
        total_rewards = [0.0 for _ in range(n_robots)]
        done_steps = [max_steps for _ in range(n_robots)]
        successes = [0 for _ in range(n_robots)]
        done_flags = [False for _ in range(n_robots)]

        for step in range(max_steps):
            actions = []
            current_nodes = [None for _ in range(n_robots)]
            target_nodes = [None for _ in range(n_robots)]

            for i in range(n_robots):
                if done_flags[i]:
                    actions.append(0)
                    continue
                current_node = robot_envs[i].current_node
                target_node = robot_envs[i].target_node
                current_nodes[i] = current_node
                target_nodes[i] = target_node

                mask = np.asarray(obs_list[i]["mask"], dtype=bool)
                valid_actions = np.where(mask)[0]
                if len(valid_actions) == 0:
                    action = 0
                elif agent.rng.random() < agent.epsilon:
                    action = int(agent.rng.choice(valid_actions))
                else:
                    action = agent.greedy_action(
                        current_idx=robot_envs[i].node2idx[current_node],
                        target_idx=robot_envs[i].node2idx[target_node],
                        mask=mask,
                    )
                actions.append(action)

            step_result = env.step(actions if hasattr(env, "envs") else actions[0])
            if hasattr(env, "envs"):
                next_obs_list, rewards, dones, infos = step_result
            else:
                next_obs, reward, terminated, truncated, info = step_result
                next_obs_list = [next_obs]
                rewards = [reward]
                dones = [terminated or truncated]
                infos = [info]

            for i in range(n_robots):
                if done_flags[i]:
                    continue
                next_node = robot_envs[i].current_node
                agent.update(
                    current_idx=robot_envs[i].node2idx[current_nodes[i]],
                    target_idx=robot_envs[i].node2idx[target_nodes[i]],
                    action=actions[i],
                    reward=rewards[i],
                    next_idx=robot_envs[i].node2idx[next_node],
                    next_mask=next_obs_list[i]["mask"],
                    done=dones[i],
                )
                transitions_used += 1
                total_rewards[i] += rewards[i]

                if dones[i] and done_steps[i] == max_steps:
                    done_steps[i] = step + 1
                    successes[i] = int(
                        robot_envs[i].current_node == robot_envs[i].target_node
                    )
                done_flags[i] = bool(dones[i])

            obs_list = next_obs_list

            if all(done_flags):
                break

        row = {
            "episode": ep + 1,
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_steps": float(np.mean(done_steps)),
            "n_robots": n_robots,
        }
        row["trajectories_used"] = min(
            total_trajectories, trajectories_used + n_robots
        )
        row["transitions_used"] = transitions_used
        history_rows.append(row)
        trajectories_used += n_robots
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        print(
            f"[Centralized Episode {ep + 1}/{joint_episodes}] "
            f"mean_reward={row['mean_reward']:+.2f} "
            f"success={row['success_rate']:.0%} "
            f"mean_steps={row['mean_steps']:.1f}"
        )

    agent.save(model_path)

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    elapsed_seconds = time.perf_counter() - started_at
    meta = {
        "architecture": "centralized",
        "algorithm": "qlearning",
        "graph_path": graph_path,
        "graph_fingerprint": env.graph_fingerprint,
        "n_robots": n_robots,
        "episodes": episodes,
        "total_trajectories": trajectories_used,
        "total_transitions": transitions_used,
        "elapsed_seconds": elapsed_seconds,
        "max_steps": max_steps,
        "seed": seed,
        "hyperparameters": {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
        },
        "model_path": model_path,
        "history_path": history_path,
    }
    with open(os.path.join(save_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Modele centralise sauvegarde: {model_path}")
    print(f"Historique centralise sauvegarde: {history_path}")
    return agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train centralized Q-learning baseline.")
    parser.add_argument("--graph", type=str, default=CONFIG["graph_path"])
    parser.add_argument("--robots", type=int, default=CONFIG["n_robots"])
    parser.add_argument("--episodes", type=int, default=get_nested("centralized", "episodes", default=200))
    parser.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--history",
        type=str,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--total-trajectories", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_centralized_qlearning(
        graph_path=args.graph,
        n_robots=args.robots,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        model_path=args.model,
        history_path=args.history,
        seed=args.seed,
        total_trajectories=args.total_trajectories,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )
