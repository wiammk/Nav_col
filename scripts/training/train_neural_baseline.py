"""Train comparable local or centralized DQN/PPO baselines.

Local: one independent model per robot, ``episodes`` trajectories per robot.
Centralized: one shared model trained on exactly ``total_trajectories`` sampled
round-robin from the robot environments.  Evaluation is intentionally separate.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environment.graph_env import make_env
from config.run_layout import layout_for_graph


def parse_args():
    parser = argparse.ArgumentParser(description="DQN/PPO local and centralized baselines")
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--mode", choices=["local", "centralized"], required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument(
        "--additional-train-graphs",
        nargs="*",
        default=[],
        help="Bâtiments supplémentaires pour l'entraînement multi-domaine; jamais le test final.",
    )
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--total-trajectories", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def make_agent(args, emb_dim):
    if args.algo == "dqn":
        return DQNAgent(
            emb_dim=emb_dim,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
        )
    return PPOAgent(
        emb_dim=emb_dim,
        lr=args.lr,
        gamma=args.gamma,
        rollout_len=args.rollout_len,
        minibatch_size=args.minibatch_size,
        n_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
    )


def metric_rows(metrics, robot_id, episode_offset):
    rows = []
    for index, (reward, success) in enumerate(
        zip(metrics["rewards"], metrics["successes"]), start=1
    ):
        row = {
            "episode": episode_offset + index,
            "robot_id": robot_id,
            "reward": float(reward),
            "success": float(success),
        }
        for key in ("loss_actor", "loss_critic", "entropy", "kl", "clip_frac", "grad_norm", "epsilon"):
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    return rows


def main():
    args = parse_args()
    np.random.seed(args.seed)
    graph_path = Path(args.graph)
    layout = layout_for_graph(graph_path)
    save_dir = Path(args.save_dir) if args.save_dir else layout.run_dir / "results" / args.mode / args.algo
    model_dir = layout.run_dir / "models" / args.mode / args.algo
    save_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    primary_env = make_env(
        str(graph_path),
        n_robots=1,
        max_steps=args.max_steps,
        seed=args.seed,
    )[0]
    domain_envs = [primary_env]
    shared_gcn = graph_path.with_name("gcn_encoder.pt")
    for index, additional_graph in enumerate(args.additional_train_graphs, start=1):
        domain_envs.append(
            make_env(
                str(Path(additional_graph)),
                n_robots=1,
                max_steps=args.max_steps,
                seed=args.seed + 1000 * index,
                shared_gcn_checkpoint_path=str(shared_gcn),
                force_rebuild_embeddings=True,
            )[0]
        )
    envs = [
        domain_envs[index % len(domain_envs)]
        for index in range(args.robots)
    ]
    rows = []
    if args.mode == "local":
        for robot_id, env in enumerate(envs):
            agent = make_agent(args, env.emb_dim)
            metrics = agent.train(env, args.episodes)
            rows.extend(metric_rows(metrics, robot_id, 0))
            agent.save(model_dir / f"robot_{robot_id}.pt")
        total_trajectories = args.episodes * args.robots
    else:
        agent = make_agent(args, envs[0].emb_dim)
        for trajectory in range(args.total_trajectories):
            robot_id = trajectory % args.robots
            metrics = agent.train(envs[robot_id], 1)
            rows.extend(metric_rows(metrics, robot_id, trajectory))
        agent.save(model_dir / "global_model.pt")
        total_trajectories = args.total_trajectories

    fieldnames = sorted({key for row in rows for key in row})
    with open(save_dir / "training_history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "algorithm": args.algo,
        "mode": args.mode,
        "graph_path": str(graph_path.resolve()),
        "graph_fingerprint": envs[0].graph_fingerprint,
        "robots": args.robots,
        "seed": args.seed,
        "episodes_per_robot": args.episodes if args.mode == "local" else None,
        "total_trajectories": total_trajectories,
        "max_steps": args.max_steps,
        "hyperparameters": vars(args),
        "training_graphs": [
            str(graph_path.resolve()),
            *[str(Path(path).resolve()) for path in args.additional_train_graphs],
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (save_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
