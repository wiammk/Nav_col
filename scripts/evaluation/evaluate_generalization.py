"""Office-to-target-building zero-shot and fine-tuning protocol for DQN/PPO.

The script evaluates a frozen Office model directly on a target graph, then
compares optional target-building fine-tuning with training from scratch under
the same episode budget.  Every CLI seed initializes Python, NumPy and PyTorch
so that a complete invocation is reproducible on the same software stack.
"""

import argparse
import csv
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environment.graph_env import make_env
from evaluation.common_protocol import (
    evaluate_single_policy,
    generate_protocol,
    summarize,
)


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the target-building training protocol."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CPU is used by the reported experiments.  ``warn_only`` keeps the
    # protocol usable if a future CUDA operation has no deterministic kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_agent(algo: str):
    if algo == "dqn":
        return DQNAgent(lr=3e-4, batch_size=64, buffer_size=20_000)
    if algo == "ppo":
        return PPOAgent(
            lr=3e-4,
            rollout_len=256,
            minibatch_size=64,
            n_epochs=10,
            clip_eps=0.2,
            entropy_coef=0.01,
            max_grad_norm=0.5,
        )
    raise ValueError(f"Unsupported algorithm: {algo}")


def greedy_fn(agent, algo: str):
    if algo == "dqn":
        return lambda observation: agent.act(observation, greedy=True)
    return lambda observation: agent.act(observation, greedy=True)[0]


def load_transfer_checkpoint(agent, path: str | Path) -> str:
    """Load either an agent checkpoint or raw FedAvg model weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        agent.load(path)
        return "agent_checkpoint"
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f"Checkpoint de transfert invalide: {path}")
    agent.set_weights(checkpoint, reset_optimizer=True)
    return "fedavg_weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-graph", required=True)
    parser.add_argument("--shared-gcn", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--fine-tuning",
        type=int,
        nargs="+",
        default=[0, 10, 50, 100],
        help="Target-building episode budgets; 0 denotes zero-shot evaluation.",
    )
    return parser.parse_args()


def _write_outputs(
    output_dir: Path,
    summaries: list[dict],
    metadata: dict,
) -> None:
    with (output_dir / "generalization_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    (output_dir / "generalization_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if any(budget < 0 for budget in args.fine_tuning):
        raise ValueError("Fine-tuning budgets must be non-negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    env, graph, _ = make_env(
        args.target_graph,
        n_robots=1,
        max_steps=args.max_steps,
        seed=args.seed,
        shared_gcn_checkpoint_path=args.shared_gcn,
        embedding_cache_path=str(output_dir / "shared_gcn_target_embeddings.npz"),
        force_rebuild_embeddings=True,
    )
    protocol = generate_protocol(
        graph,
        seed=args.seed + 9000,
        n_single=200,
        n_multi=0,
        robot_counts=(),
    )

    summaries: list[dict] = []
    checkpoint_format: str | None = None
    for budget in args.fine_tuning:
        transferred = make_agent(args.algo)
        checkpoint_format = load_transfer_checkpoint(transferred, args.source_model)
        if budget:
            transferred.train(env, budget)

        summary = summarize(
            evaluate_single_policy(env, greedy_fn(transferred, args.algo), protocol["single"])
        )
        summary.update(
            {
                "mode": "zero_shot" if budget == 0 else "fine_tuning",
                "episodes": budget,
            }
        )
        summaries.append(summary)

        if budget == 0:
            continue

        scratch = make_agent(args.algo)
        scratch.train(env, budget)
        scratch_summary = summarize(
            evaluate_single_policy(env, greedy_fn(scratch, args.algo), protocol["single"])
        )
        scratch_summary.update({"mode": "from_scratch", "episodes": budget})
        summaries.append(scratch_summary)

    _write_outputs(
        output_dir,
        summaries,
        {
            "algorithm": args.algo,
            "source_model": str(Path(args.source_model).resolve()),
            "target_graph": str(Path(args.target_graph).resolve()),
            "shared_gcn": str(Path(args.shared_gcn).resolve()),
            "seed": args.seed,
            "fine_tuning_episodes": args.fine_tuning,
            "checkpoint_format": checkpoint_format,
            "learning_during_evaluation": False,
            "random_seed_policy": "python_numpy_torch_seeded_from_cli_seed",
            "torch_deterministic_algorithms": True,
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "torch_version": torch.__version__,
        },
    )


if __name__ == "__main__":
    main()
