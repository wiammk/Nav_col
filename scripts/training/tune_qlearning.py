"""Reproducible train/validation/test search for standard tabular Q-learning."""

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.q_learning import QLearningAgent
from environment.graph_env import make_env
from evaluation.common_protocol import evaluate_single_policy, generate_protocol, summarize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--validation-scenarios", type=int, default=100)
    parser.add_argument("--test-scenarios", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.95, 0.99])
    parser.add_argument("--epsilon-starts", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--epsilon-ends", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--max-steps", type=int, default=200)
    return parser.parse_args()


def reset_from_scenario(env, scenario):
    return env.reset(
        start=scenario["start"],
        target=scenario["target"],
        options={
            "closed_edges": scenario.get("closed_edges", []),
            "blocked_nodes": scenario.get("blocked_nodes", []),
        },
    )


def train_candidate(env, params, scenarios, episodes, seed):
    agent = QLearningAgent(
        num_nodes=len(env.nodes),
        max_degree=env.max_degree,
        lr=params["alpha"],
        gamma=params["gamma"],
        epsilon_start=params["epsilon_start"],
        epsilon_end=params["epsilon_end"],
        epsilon_decay_steps=max(1, episodes * 20),
        seed=seed,
    )
    for episode in range(episodes):
        scenario = scenarios[episode % len(scenarios)]
        obs, _ = reset_from_scenario(env, scenario)
        done = False
        while not done:
            current = env.current_node
            target = env.target_node
            action = agent.choose_action(
                env.node2idx[current],
                env.node2idx[target],
                obs["mask"],
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_node = env.current_node
            done = terminated or truncated
            agent.update(
                env.node2idx[current],
                env.node2idx[target],
                action,
                reward,
                env.node2idx[next_node],
                next_obs["mask"],
                done,
            )
            obs = next_obs
    return agent


def action_function(agent, env):
    def act(obs):
        return agent.greedy_action(
            env.node2idx[env.current_node],
            env.node2idx[env.target_node],
            obs["mask"],
        )
    return act


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env, graph, _ = make_env(
        args.graph,
        n_robots=1,
        max_steps=args.max_steps,
        seed=min(args.seeds),
    )
    train_protocol = generate_protocol(
        graph, seed=11_001, n_single=max(args.train_episodes, 100), n_multi=0
    )["single"]
    validation_protocol = generate_protocol(
        graph, seed=22_002, n_single=args.validation_scenarios, n_multi=0
    )["single"]
    test_protocol = generate_protocol(
        graph, seed=33_003, n_single=args.test_scenarios, n_multi=0
    )["single"]

    rows = []
    combinations = itertools.product(
        args.alphas,
        args.gammas,
        args.epsilon_starts,
        args.epsilon_ends,
    )
    for alpha, gamma, epsilon_start, epsilon_end in combinations:
        params = {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
        }
        seed_summaries = []
        for seed in args.seeds:
            agent = train_candidate(
                env, params, train_protocol, args.train_episodes, seed
            )
            validation_rows = evaluate_single_policy(
                env, action_function(agent, env), validation_protocol
            )
            seed_summaries.append(summarize(validation_rows))
        row = dict(params)
        for metric in ("success_mean", "spl_mean", "steps_mean", "reward_mean"):
            row[f"validation_{metric}"] = float(
                np.mean([summary[metric] for summary in seed_summaries])
            )
            row[f"validation_{metric}_seed_std"] = float(
                np.std([summary[metric] for summary in seed_summaries], ddof=1)
                if len(seed_summaries) > 1 else 0.0
            )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["validation_success_mean"],
            row["validation_spl_mean"],
            -row["validation_steps_mean"],
        ),
        reverse=True,
    )
    best = rows[0]
    best_params = {
        key: best[key]
        for key in ("alpha", "gamma", "epsilon_start", "epsilon_end")
    }

    test_summaries = []
    for seed in args.seeds:
        agent = train_candidate(
            env, best_params, train_protocol, args.train_episodes, seed
        )
        test_summaries.append(
            summarize(
                evaluate_single_policy(
                    env, action_function(agent, env), test_protocol
                )
            )
        )
    test_summary = {
        metric: float(np.mean([summary[metric] for summary in test_summaries]))
        for metric in test_summaries[0]
        if metric != "n_scenarios"
    }

    with open(output_dir / "validation_search.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "selection_rule": "validation_success_then_spl_then_steps",
        "best_hyperparameters": best_params,
        "validation": best,
        "held_out_test": test_summary,
        "seeds": args.seeds,
        "train_episodes": args.train_episodes,
        "protocol_seeds": {"train": 11001, "validation": 22002, "test": 33003},
    }
    (output_dir / "best_result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
