import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG, get_nested
from config.run_layout import layout_for_graph


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GRAPH_PKL = Path(CONFIG["graph_path"])


def parse_args():
    parser = argparse.ArgumentParser(description="Federated RL for robot navigation.")
    parser.add_argument(
        "--algo",
        type=str,
        default=get_nested("federated", "algo", default="qlearning"),
        choices=["qlearning", "dqn", "ppo"],
        help="RL algorithm: qlearning | dqn | ppo",
    )
    parser.add_argument("--robots", type=int, default=CONFIG["n_robots"], help="Number of robots")
    parser.add_argument(
        "--rounds",
        type=int,
        default=get_nested("federated", "rounds", default=20),
        help="Number of federated rounds",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=get_nested("federated", "episodes_per_round", default=50),
        help="Local episodes per round",
    )
    parser.add_argument("--max-steps", type=int, default=CONFIG["max_steps"], help="Max steps per episode")
    parser.add_argument("--graph", type=str, default=str(GRAPH_PKL))
    parser.add_argument(
        "--save-every",
        type=int,
        default=get_nested("federated", "save_every", default=5),
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Dossier des checkpoints, historiques et metadonnees federes.",
    )
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=get_nested("scientific_protocol", "validation_seed", default=22002),
        help="Seed du jeu de validation, distincte de l'entrainement et du test.",
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None)
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
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument(
        "--q-aggregation",
        choices=["standard", "visitation_weighted"],
        default="standard",
        help="Q-table aggregator; visitation weighting is an experimental variant.",
    )
    parser.add_argument(
        "--heterogeneity",
        choices=["iid", "spatial"],
        default="iid",
        help="Distribution locale des couples départ-cible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.algo != "qlearning" and args.q_aggregation != "standard":
        raise ValueError("--q-aggregation s'applique uniquement à Q-learning.")
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log.info("=" * 55)
    log.info(
        "FEDERATED RL | algo=%s robots=%s rounds=%s",
        args.algo.upper(),
        args.robots,
        args.rounds,
    )
    log.info("=" * 55)

    graph_path = Path(args.graph)
    if not graph_path.exists():
        log.error("graph.gpickle introuvable: %s", graph_path)
        log.error("Lancez: python data/ifc_parser.py --synthetic puis python data/graph_builder.py")
        sys.exit(1)

    from environment.graph_env import make_env

    log.info("Creation des environnements")
    multi_env, _, _ = make_env(
        graph_path=str(graph_path),
        n_robots=args.robots,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    envs = multi_env.envs if hasattr(multi_env, "envs") else [multi_env]

    log.info(
        "%s environnements crees (emb_dim=%s, max_degree=%s)",
        args.robots,
        envs[0].emb_dim,
        envs[0].max_degree,
    )

    layout = layout_for_graph(graph_path)
    save_dir = Path(args.save_dir) if args.save_dir else layout.federated_results / args.algo
    save_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "algorithm": args.algo,
        "aggregation": (
            "fedavg_visitation_weighted_q"
            if args.algo == "qlearning" and args.q_aggregation == "visitation_weighted"
            else "fedavg_standard_sample_weighted"
        ),
        "model_variant": "standard_dqn" if args.algo == "dqn" else args.algo,
        "graph_path": str(graph_path.resolve()),
        "graph_fingerprint": multi_env.graph_fingerprint,
        "embedding_metadata": multi_env.embedding_metadata,
        "n_robots": args.robots,
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "rounds": args.rounds,
        "episodes_per_round_per_client": args.episodes,
        "total_episodes_per_client": args.rounds * args.episodes,
        "total_trajectories": args.robots * args.rounds * args.episodes,
        "max_steps": args.max_steps,
        "evaluation_episodes_per_round": args.eval_episodes,
        "hyperparameters": {
            "lr": args.lr,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "tau": args.tau,
            "rollout_len": args.rollout_len,
            "minibatch_size": args.minibatch_size,
            "ppo_epochs": args.ppo_epochs,
            "clip_eps": args.clip_eps,
            "entropy_coef": args.entropy_coef,
            "max_grad_norm": args.max_grad_norm,
            "alpha": args.alpha,
            "epsilon_start": args.epsilon_start,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "heterogeneity": args.heterogeneity,
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (save_dir / "run_metadata.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    from federated.client import make_client

    log.info("Creation des clients %s", args.algo.upper())
    if args.algo == "dqn":
        client_kwargs = {
            "emb_dim": envs[0].emb_dim,
            "lr": args.lr or 3e-4,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "tau": args.tau,
        }
    elif args.algo == "ppo":
        client_kwargs = {
            "emb_dim": envs[0].emb_dim,
            "lr": args.lr or 3e-4,
            "gamma": args.gamma,
            "rollout_len": args.rollout_len,
            "minibatch_size": args.minibatch_size,
            "n_epochs": args.ppo_epochs,
            "clip_eps": args.clip_eps,
            "entropy_coef": args.entropy_coef,
            "max_grad_norm": args.max_grad_norm,
        }
    else:
        client_kwargs = {
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon": args.epsilon_start,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
        }
    clients = [
        make_client(
            args.algo,
            envs[i],
            robot_id=i,
            **client_kwargs,
            **({"seed": args.seed + i} if args.algo == "qlearning" else {}),
        )
        for i in range(args.robots)
    ]
    if args.heterogeneity == "spatial":
        ordered_nodes = sorted(
            envs[0].nodes,
            key=lambda node: (
                float(envs[0].graph.nodes[node].get("x_norm", 0.0)),
                float(envs[0].graph.nodes[node].get("y_norm", 0.0)),
                str(node),
            ),
        )
        partitions = __import__("numpy").array_split(
            __import__("numpy").asarray(ordered_nodes, dtype=object),
            args.robots,
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

    rng = __import__("numpy").random.default_rng(args.validation_seed)
    nodes = list(envs[0].nodes)
    evaluation_scenarios = []
    while len(evaluation_scenarios) < args.eval_episodes:
        start, target = rng.choice(nodes, size=2, replace=False)
        evaluation_scenarios.append((start.item() if hasattr(start, "item") else start,
                                     target.item() if hasattr(target, "item") else target))

    from federated.server import FedAvgServer

    server = FedAvgServer(save_dir=save_dir, q_aggregation=args.q_aggregation)
    started_at = time.perf_counter()
    history = server.run(
        clients=clients,
        n_rounds=args.rounds,
        episodes_per_round=args.episodes,
        save_every=args.save_every,
        evaluation_scenarios=evaluation_scenarios,
        multi_env=multi_env if hasattr(multi_env, "envs") else None,
    )
    server.save_history()
    provenance["elapsed_seconds"] = time.perf_counter() - started_at
    provenance["total_transitions"] = int(
        sum(int(item.get("n_transitions", 0)) for item in history)
    )
    provenance["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    model_name = (
        "global_qtable_final.pkl"
        if args.algo == "qlearning"
        else "global_model_final.pt"
    )
    model_path = save_dir / model_name
    provenance["model_size_bytes"] = (
        int(model_path.stat().st_size) if model_path.exists() else 0
    )
    provenance["estimated_communication_bytes"] = (
        2 * args.robots * args.rounds * provenance["model_size_bytes"]
    )
    (save_dir / "run_metadata.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rewards = [h["global_mean_reward"] for h in history]
    successes = [h["global_success_rate"] for h in history]

    log.info("=" * 55)
    log.info("RESULTATS FINAUX | %s federe", args.algo.upper())
    log.info("=" * 55)
    log.info("Reward: debut=%+.2f fin=%+.2f max=%+.2f", rewards[0], rewards[-1], max(rewards))
    log.info("Succes: debut=%.0f%% fin=%.0f%% max=%.0f%%", successes[0] * 100, successes[-1] * 100, max(successes) * 100)
    log.info("Resultats: %s", save_dir)
    log.info("Entrainement federe termine.")


if __name__ == "__main__":
    main()
