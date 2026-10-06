import argparse
import csv
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG, get_nested
from config.run_layout import RunLayout, layout_for_graph
from environment.graph_env import make_env
from scripts.visualization.visualize_federated_navigation import (
    GIF_FPS,
    build_pos_per_floor,
    choose_action,
    load_policy,
    make_colors,
    render_static_graph,
    resolve_path,
    validate_model_provenance,
)


ROOT = PROJECT_ROOT
def default_single_model_path(
    algo: str,
    source: str,
    layout: RunLayout | None = None,
) -> Path:
    layout = layout or layout_for_graph(CONFIG["graph_path"])
    if algo == "qlearning":
        if source == "fedavg":
            return layout.federated_results / "qlearning" / "global_qtable_final.pkl"
        return layout.centralized_models / "global_qtable.pkl"
    if source != "fedavg":
        raise FileNotFoundError(
            f"Aucun entrainement {source} n'est defini par defaut pour {algo}. "
            "Utilisez --source fedavg ou donnez un chemin avec --model."
        )
    return layout.federated_results / algo / "global_model_final.pt"


def default_local_q_paths(
    n_robots: int,
    local_dir: str | None = None,
    layout: RunLayout | None = None,
) -> list[Path]:
    layout = layout or layout_for_graph(CONFIG["graph_path"])
    q_dir = resolve_path(local_dir) if local_dir else layout.local_models
    return [q_dir / f"robot_{i}.pkl" for i in range(n_robots)]


def load_qtable(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "Q" in data:
        return data["Q"]
    if isinstance(data, np.ndarray):
        return data
    raise TypeError(f"Format Q-table non supporte: {path}")


def load_trained_policy(
    algo: str,
    source: str,
    model_path: Path | None,
    n_robots: int,
    emb_dim: int,
    hidden_dim: int,
    local_dir: str | None = None,
    layout: RunLayout | None = None,
):
    if algo == "qlearning" and source == "local" and model_path is None:
        paths = default_local_q_paths(n_robots, local_dir, layout)
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Q-tables locales manquantes. Lancez d'abord: python train_qlearning.py\n"
                + "\n".join(missing)
            )
        return [load_qtable(p) for p in paths], "local_qtables"

    path = model_path if model_path else default_single_model_path(algo, source, layout)
    if not path.exists():
        raise FileNotFoundError(f"Modele introuvable: {path}")
    return load_policy(algo, path, emb_dim, hidden_dim), path.stem


def choose_trained_action(policy, algo: str, source: str, robot_id: int, graph_env, obs: dict, emb_dim: int):
    if algo == "qlearning" and source == "local" and isinstance(policy, list):
        return choose_action(policy[robot_id], algo, graph_env, obs, emb_dim)
    return choose_action(policy, algo, graph_env, obs, emb_dim)


def simulate_episode(env, policy, algo: str, source: str, emb_dim: int, max_steps: int):
    obs_list = env.reset()
    paths = [[env.envs[i].current_node] for i in range(env.n_robots)]
    successes = [0 for _ in range(env.n_robots)]
    done_steps = [max_steps for _ in range(env.n_robots)]

    for step in range(max_steps):
        actions = []
        for i, graph_env in enumerate(env.envs):
            if graph_env._terminated or graph_env._truncated:
                actions.append(0)
                continue
            action = choose_trained_action(policy, algo, source, i, graph_env, obs_list[i], emb_dim)
            actions.append(action)

        obs_list, rewards, dones, infos = env.step(actions)

        for i, graph_env in enumerate(env.envs):
            paths[i].append(graph_env.current_node)
            if dones[i] and done_steps[i] == max_steps:
                done_steps[i] = step + 1
                successes[i] = int(graph_env.current_node == graph_env.target_node)

        if all(dones):
            break

    return paths, successes, done_steps


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize final trained navigation policies.")
    parser.add_argument("--algo", choices=["qlearning", "ppo", "dqn"], required=True)
    parser.add_argument(
        "--source",
        choices=["local", "centralized", "fedavg"],
        default=None,
        help="Default: local for qlearning, fedavg for ppo/dqn.",
    )
    parser.add_argument("--model", default=None, help="Optional model path.")
    parser.add_argument("--local-dir", default=None, help="Dossier des Q-tables locales.")
    parser.add_argument("--graph", default=CONFIG["graph_path"])
    parser.add_argument("--robots", type=int, default=CONFIG["n_robots"])
    parser.add_argument("--episodes", type=int, default=get_nested("visualization", "episodes", default=10))
    parser.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source or ("local" if args.algo == "qlearning" else "fedavg")
    model_path = resolve_path(args.model) if args.model else None
    graph_path = resolve_path(args.graph)
    layout = layout_for_graph(graph_path)
    out_dir = (
        resolve_path(args.out_dir)
        if args.out_dir
        else layout.visualizations / "trained" / args.algo / source
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env, G, emb = make_env(
        str(graph_path),
        n_robots=args.robots,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    policy, model_label = load_trained_policy(
        args.algo,
        source,
        model_path,
        args.robots,
        args.emb_dim,
        args.hidden_dim,
        args.local_dir,
        layout,
    )
    provenance_model = model_path
    if provenance_model is None:
        if args.algo == "qlearning" and source == "local":
            provenance_model = default_local_q_paths(args.robots, args.local_dir, layout)[0]
        else:
            provenance_model = default_single_model_path(args.algo, source, layout)
    validate_model_provenance(provenance_model, env, args.algo)

    colors = make_colors(args.robots)
    pos = build_pos_per_floor(G)

    frames = []
    summary_rows = []
    for ep in range(args.episodes):
        paths, successes, done_steps = simulate_episode(
            env,
            policy,
            args.algo,
            source,
            args.emb_dim,
            args.max_steps,
        )
        success_rate = float(np.mean(successes))
        mean_steps = float(np.mean(done_steps))
        title = (
            f"{args.algo.upper()} trained | source={source} | {model_label} | "
            f"episode {ep + 1} | success={success_rate:.0%} | mean_steps={mean_steps:.1f}"
        )
        frames.append(render_static_graph(G, pos, paths, colors, title=title))
        summary_rows.append(
            {
                "episode": ep + 1,
                "algo": args.algo,
                "source": source,
                "model": model_label,
                "success_rate": success_rate,
                "mean_steps": mean_steps,
            }
        )

    gif_path = out_dir / "episodes.gif"
    imageio.mimsave(gif_path, frames, fps=GIF_FPS)

    summary_path = out_dir / "visualization_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {gif_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
