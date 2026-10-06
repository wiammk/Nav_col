import argparse
import csv
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
from config.run_layout import RunLayout, layout_for_graph
from environment.graph_env import make_env
from scripts.visualization.visualize_federated_navigation import (
    GIF_FPS,
    build_pos_per_floor,
    load_policy,
    make_colors,
    render_static_graph,
    resolve_path,
    simulate_episode,
)


ROOT = PROJECT_ROOT
def checkpoint_dir(
    algo: str,
    base_dir: Path | None = None,
    layout: RunLayout | None = None,
) -> Path:
    if base_dir is not None:
        return base_dir
    layout = layout or layout_for_graph(CONFIG["graph_path"])
    return layout.federated_results / algo


def checkpoint_pattern(algo: str) -> str:
    if algo == "qlearning":
        return "global_qtable_*.pkl"
    return "global_model_*.pt"


def checkpoint_label(path: Path) -> tuple[int, str]:
    if "final" in path.stem:
        return 10**9, "final"
    match = re.search(r"round_(\d+)", path.stem)
    if match:
        round_n = int(match.group(1))
        return round_n, f"round_{round_n:03d}"
    return 10**8, path.stem


def find_checkpoints(
    algo: str,
    explicit_paths: list[str] | None = None,
    base_dir: Path | None = None,
    layout: RunLayout | None = None,
) -> list[tuple[str, Path]]:
    if explicit_paths:
        paths = [resolve_path(p) for p in explicit_paths]
    else:
        paths = sorted(checkpoint_dir(algo, base_dir, layout).glob(checkpoint_pattern(algo)))

    existing = [p for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"Aucun checkpoint FedAvg trouve pour {algo}. "
            f"Lancez d'abord: python train_federated.py --algo {algo} --save-every 1"
        )

    ordered = sorted(existing, key=lambda p: checkpoint_label(p)[0])
    return [(checkpoint_label(p)[1], p) for p in ordered]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize FedAvg global model progression across rounds.")
    parser.add_argument("--algo", choices=["qlearning", "ppo", "dqn"], required=True)
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Optional explicit checkpoint paths.")
    parser.add_argument("--checkpoint-dir", default=None, help="Dossier des checkpoints de cet algorithme.")
    parser.add_argument("--graph", default=CONFIG["graph_path"])
    parser.add_argument("--robots", type=int, default=CONFIG["n_robots"])
    parser.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    graph_path = resolve_path(args.graph)
    layout = layout_for_graph(graph_path)
    out_dir = (
        resolve_path(args.out_dir)
        if args.out_dir
        else layout.visualizations / "fedavg_progress" / args.algo
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_base = resolve_path(args.checkpoint_dir) if args.checkpoint_dir else None
    checkpoints = find_checkpoints(args.algo, args.checkpoints, checkpoint_base, layout)
    env, G, emb = make_env(
        str(graph_path),
        n_robots=args.robots,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    colors = make_colors(args.robots)
    pos = build_pos_per_floor(G)

    frames = []
    summary_rows = []
    for label, model_path in checkpoints:
        policy = load_policy(args.algo, model_path, args.emb_dim, args.hidden_dim)
        paths, successes, done_steps = simulate_episode(
            env,
            policy,
            args.algo,
            args.emb_dim,
            args.max_steps,
        )
        success_rate = float(np.mean(successes))
        mean_steps = float(np.mean(done_steps))
        title = (
            f"{args.algo.upper()} FedAvg progress | {label} | "
            f"success={success_rate:.0%} | mean_steps={mean_steps:.1f}"
        )
        frames.append(render_static_graph(G, pos, paths, colors, title=title))
        summary_rows.append(
            {
                "checkpoint": label,
                "model_path": str(model_path),
                "success_rate": success_rate,
                "mean_steps": mean_steps,
            }
        )

    gif_path = out_dir / "fedavg_progress.gif"
    imageio.mimsave(gif_path, frames, fps=GIF_FPS)

    summary_path = out_dir / "fedavg_progress_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {gif_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
