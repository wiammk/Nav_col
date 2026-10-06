import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.graph_env import make_env
from config import CONFIG, get_nested
from config.run_layout import layout_for_graph, resolve_project_path

DEFAULT_SUCCESS_TARGET = 0.95
DEFAULT_STABILITY_WINDOW = 10


def load_qtable(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "Q" in data:
        return data["Q"]
    if isinstance(data, np.ndarray):
        return data
    raise TypeError(f"Unsupported Q-table format: {path}")


def qtable_compatibility_error(qtable: np.ndarray, env, model_name: str) -> str:
    expected_nodes = len(env.nodes)
    expected_actions = env.max_degree

    if not isinstance(qtable, np.ndarray):
        return f"{model_name}: Q-table invalide ({type(qtable).__name__}), ndarray attendu."

    if qtable.ndim != 3:
        return f"{model_name}: Q-table invalide shape={qtable.shape}, shape attendu=(n_nodes, n_nodes, max_degree)."

    if qtable.shape[0] != expected_nodes or qtable.shape[1] != expected_nodes:
        return (
            f"{model_name}: modele entraine sur {qtable.shape[0]} noeuds, "
            f"mais le graphe d'evaluation contient {expected_nodes} noeuds. "
            "Regenerer le graphe puis reentrainer les modeles Q-learning avec le meme fichier IFC."
        )

    if qtable.shape[2] < expected_actions:
        return (
            f"{model_name}: Q-table avec {qtable.shape[2]} actions max, "
            f"mais le graphe demande {expected_actions} actions max."
        )

    return ""


def provenance_compatibility_error(model_path: Path, env, model_name: str) -> str:
    metadata_path = model_path.parent / "run_metadata.json"
    if not metadata_path.exists():
        metadata_path = model_path.parent / "train_meta.json"
    if not metadata_path.exists():
        return (
            f"{model_name}: provenance absente. Reentrainez ce modele avec la version actuelle "
            "pour verifier le graphe et l'ordre des actions."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{model_name}: provenance illisible ({exc})."
    if metadata.get("graph_fingerprint") != getattr(env, "graph_fingerprint", None):
        return (
            f"{model_name}: modele entraine sur un autre graphe ou un autre ordre de voisins. "
            "Reentrainez-le avant evaluation."
        )
    return ""


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        if fieldnames:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return int(sum(p.stat().st_size for p in path.glob("*.pkl") if p.is_file()))
    return 0


def make_scenarios(nodes, n_eval: int, seed: int):
    rng = np.random.default_rng(seed)
    nodes = list(nodes)
    scenarios = []
    for scenario_id in range(n_eval):
        start = rng.choice(nodes)
        target = rng.choice(nodes)
        while target == start:
            target = rng.choice(nodes)
        scenarios.append((scenario_id + 1, int(start), int(target)))
    return scenarios


def greedy_action(qtable, env, current_node, target_node, mask):
    valid = np.where(mask.astype(bool))[0]
    if len(valid) == 0:
        return 0

    current_idx = env.node2idx.get(current_node, current_node)
    target_idx = env.node2idx.get(target_node, target_node)
    qvals = qtable[current_idx, target_idx]

    masked = np.full_like(qvals, -np.inf, dtype=np.float32)
    masked[valid] = qvals[valid]
    return int(np.argmax(masked))


def evaluate_qtable(qtable, graph_path: str, scenarios, max_steps: int, seed: int):
    env, _, _ = make_env(graph_path, n_robots=1, max_steps=max_steps, seed=seed)
    error = qtable_compatibility_error(qtable, env, "qtable")
    if error:
        raise ValueError(error)

    rows = []

    for scenario_id, start, target in scenarios:
        obs, _ = env.reset(start=start, target=target)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = greedy_action(
                qtable=qtable,
                env=env,
                current_node=env.current_node,
                target_node=env.target_node,
                mask=obs["mask"],
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        rows.append(
            {
                "scenario": scenario_id,
                "start": start,
                "target": target,
                "reward": float(total_reward),
                "success": int(env.current_node == env.target_node),
                "steps": int(steps),
            }
        )

    return rows


def summarize(rows, method: str, model_name: str):
    rewards = np.array([float(r["reward"]) for r in rows], dtype=np.float32)
    successes = np.array([float(r["success"]) for r in rows], dtype=np.float32)
    steps = np.array([float(r["steps"]) for r in rows], dtype=np.float32)
    return {
        "method": method,
        "model": model_name,
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "std_reward": float(rewards.std()) if len(rewards) else 0.0,
        "success_rate": float(successes.mean()) if len(successes) else 0.0,
        "mean_steps": float(steps.mean()) if len(steps) else 0.0,
        "n_eval_episodes": int(len(rows)),
    }


def rolling(values, window=10):
    if len(values) == 0:
        return values
    window = min(window, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def first_step_at(values: np.ndarray, steps: np.ndarray, threshold: float):
    hits = np.where(values >= threshold)[0]
    if len(hits) == 0:
        return ""
    return int(steps[int(hits[0])])


def training_metric(row: dict, metric: str, architecture: str) -> float:
    """Read training metrics without confusing local FedAvg data with global data."""
    candidates = []
    if architecture == "federated":
        candidates.append(f"global_{metric}")
    candidates.append(metric)
    for key in candidates:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise ValueError(
        f"Colonne de métrique absente pour {architecture}: "
        f"attendu l'une de {candidates}, trouvé {sorted(row.keys())}"
    )


def summarize_training_history(
    name: str,
    architecture: str,
    rows,
    model_file: Path,
    success_target: float,
    stability_window: int,
):
    if not rows:
        return {
            "method": name,
            "architecture": architecture,
            "points": 0,
            "initial_reward": "",
            "final_reward": "",
            "reward_gain": "",
            "best_reward": "",
            "step_to_success_target": "",
            "step_to_100_success": "",
            "final_success_rate": "",
            "mean_std_reward": "",
            "reward_volatility": "",
            "last_window_reward_std": "",
            "model_size_bytes": file_size_bytes(model_file),
        }

    step_key = "round" if "round" in rows[0] else "episode"
    steps = np.array([int(float(r[step_key])) for r in rows], dtype=np.int32)
    rewards = np.array(
        [training_metric(r, "mean_reward", architecture) for r in rows],
        dtype=np.float32,
    )
    successes = np.array(
        [training_metric(r, "success_rate", architecture) for r in rows],
        dtype=np.float32,
    )
    std_rewards = np.array(
        [training_metric(r, "std_reward", architecture) for r in rows],
        dtype=np.float32,
    )
    reward_deltas = np.diff(rewards)
    window = min(stability_window, len(rewards))
    last_rewards = rewards[-window:]

    return {
        "method": name,
        "architecture": architecture,
        "points": len(rows),
        "initial_reward": float(rewards[0]),
        "final_reward": float(rewards[-1]),
        "reward_gain": float(rewards[-1] - rewards[0]),
        "best_reward": float(rewards.max()),
        "step_to_success_target": first_step_at(successes, steps, success_target),
        "step_to_100_success": first_step_at(successes, steps, 1.0),
        "final_success_rate": float(successes[-1]),
        "mean_std_reward": float(std_rewards.mean()) if len(std_rewards) else 0.0,
        "reward_volatility": float(np.std(reward_deltas)) if len(reward_deltas) else 0.0,
        "last_window_reward_std": float(np.std(last_rewards)) if len(last_rewards) else 0.0,
        "model_size_bytes": file_size_bytes(model_file),
    }


def plot_training_curves(local_history, centralized_history, fl_history, out_dir: Path):
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed; skipping training curve plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if local_history or centralized_history or fl_history:
        plt.figure(figsize=(9, 5))
        if local_history:
            x = np.array([int(r["episode"]) for r in local_history])
            y = np.array([float(r["mean_reward"]) for r in local_history])
            yr = rolling(y, window=10)
            xr = x[-len(yr) :]
            plt.plot(xr, yr, alpha=0.45, label="Independent local Q-learning")
        if centralized_history:
            x = np.array([int(r["episode"]) for r in centralized_history])
            y = np.array([float(r["mean_reward"]) for r in centralized_history])
            yr = rolling(y, window=10)
            xr = x[-len(yr) :]
            plt.plot(xr, yr, label="Centralized Q-learning")
        if fl_history:
            x = np.array([int(r["round"]) for r in fl_history])
            y = np.array([
                training_metric(r, "mean_reward", "federated") for r in fl_history
            ])
            plt.plot(x, y, marker="o", label="Q-learning FedAvg")
        plt.xlabel("Local episode / federated round")
        plt.ylabel("Mean reward")
        plt.title("Mean reward during training")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "reward_vs_episode_round.png", dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        if local_history:
            x = np.array([int(r["episode"]) for r in local_history])
            y = np.array([float(r["success_rate"]) for r in local_history])
            yr = rolling(y, window=10)
            xr = x[-len(yr) :]
            plt.plot(xr, yr, alpha=0.45, label="Independent local Q-learning")
        if centralized_history:
            x = np.array([int(r["episode"]) for r in centralized_history])
            y = np.array([float(r["success_rate"]) for r in centralized_history])
            yr = rolling(y, window=10)
            xr = x[-len(yr) :]
            plt.plot(xr, yr, label="Centralized Q-learning")
        if fl_history:
            x = np.array([int(r["round"]) for r in fl_history])
            y = np.array([
                training_metric(r, "success_rate", "federated") for r in fl_history
            ])
            plt.plot(x, y, marker="o", label="Q-learning FedAvg")
        plt.xlabel("Local episode / federated round")
        plt.ylabel("Success rate")
        plt.ylim(-0.05, 1.05)
        plt.title("Success rate during training")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "success_rate_vs_episode_round.png", dpi=200)
        plt.close()


def plot_eval_bars(summary_rows, out_dir: Path):
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed; skipping evaluation bar plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in summary_rows if r["model"] in ("centralized_global", "fedavg_final")]
    if len(rows) < 2:
        rows = [r for r in summary_rows if r["model"] in ("local_mean", "fedavg_final")]
    if len(rows) < 2:
        rows = summary_rows

    labels = [r["method"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, [float(r["mean_steps"]) for r in rows], color=["#4C78A8", "#F58518"][: len(rows)])
    plt.ylabel("Mean steps")
    plt.title("Mean episode length during evaluation")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_steps_comparison.png", dpi=200)
    plt.close()

    metrics = [
        ("Mean reward", "mean_reward"),
        ("Success rate", "success_rate"),
        ("Mean steps", "mean_steps"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#4C78A8", "#F58518"][: len(rows)]
    for ax, (title, key) in zip(axes, metrics):
        ax.bar(labels, [float(r[key]) for r in rows], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Final comparison: Local vs FedAvg")
    fig.tight_layout()
    fig.savefig(out_dir / "final_bar_comparison.png", dpi=200)
    plt.close(fig)


def plot_training_summary(training_rows, out_dir: Path):
    if not HAS_MATPLOTLIB or not training_rows:
        return

    rows = [r for r in training_rows if r["points"]]
    if len(rows) < 2:
        return

    labels = [r["method"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(labels, [float(r["reward_gain"]) for r in rows], color="#4C78A8")
    axes[0].set_title("Reward improvement")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(labels, [float(r["last_window_reward_std"]) for r in rows], color="#F58518")
    axes[1].set_title("Final-stage stability")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].bar(
        labels,
        [float(r["model_size_bytes"]) / 1024 for r in rows],
        color="#54A24B",
    )
    axes[2].set_title("Model size")
    axes[2].set_ylabel("KiB")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle("Centralized vs FedAvg analysis: convergence, stability, and complexity")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "training_analysis_summary.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare local Q-learning and FedAvg Q-learning.")
    parser.add_argument("--graph", default=CONFIG["graph_path"])
    parser.add_argument("--local-dir", default=None)
    parser.add_argument(
        "--centralized-model",
        default=None,
    )
    parser.add_argument("--fedavg-model", default=None)
    parser.add_argument(
        "--centralized-history",
        default=None,
    )
    parser.add_argument(
        "--local-history",
        default=None,
    )
    parser.add_argument("--fedavg-history", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--eval-episodes", type=int, default=get_nested("evaluation", "eval_episodes", default=200))
    parser.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--robots", type=int, default=CONFIG["n_robots"])
    parser.add_argument("--success-target", type=float, default=DEFAULT_SUCCESS_TARGET)
    parser.add_argument("--stability-window", type=int, default=DEFAULT_STABILITY_WINDOW)
    args = parser.parse_args()

    graph = resolve_project_path(args.graph)
    layout = layout_for_graph(graph)
    args.graph = str(graph)
    args.local_dir = str(resolve_project_path(args.local_dir) if args.local_dir else layout.local_models)
    args.centralized_model = str(
        resolve_project_path(args.centralized_model)
        if args.centralized_model
        else layout.centralized_models / "global_qtable.pkl"
    )
    args.fedavg_model = str(
        resolve_project_path(args.fedavg_model)
        if args.fedavg_model
        else layout.federated_results / "qlearning" / "global_qtable_final.pkl"
    )
    args.local_history = str(
        resolve_project_path(args.local_history)
        if args.local_history
        else layout.results / "local" / "qlearning" / "local_history.csv"
    )
    args.centralized_history = str(
        resolve_project_path(args.centralized_history)
        if args.centralized_history
        else layout.results / "centralized" / "qlearning" / "centralized_history.csv"
    )
    args.fedavg_history = str(
        resolve_project_path(args.fedavg_history)
        if args.fedavg_history
        else layout.federated_results / "qlearning" / "fl_history.csv"
    )
    args.out_dir = str(
        resolve_project_path(args.out_dir)
        if args.out_dir
        else layout.evaluation_results / "qlearning"
    )
    args.plots_dir = str(
        resolve_project_path(args.plots_dir)
        if args.plots_dir
        else layout.evaluation_plots / "qlearning"
    )

    graph_path = str(ROOT / args.graph)
    probe_env, _, _ = make_env(graph_path, n_robots=1, max_steps=args.max_steps, seed=args.seed)
    scenarios = make_scenarios(probe_env.nodes, args.eval_episodes, args.seed)

    out_dir = ROOT / args.out_dir
    plots_dir = ROOT / args.plots_dir
    detailed_rows = []
    summary_rows = []
    skipped_rows = []

    local_files = [ROOT / args.local_dir / f"robot_{i}.pkl" for i in range(args.robots)]
    missing_local = [p.name for p in local_files if not p.exists()]
    if missing_local:
        print(
            f"Missing local Q-tables for n_robots={args.robots}: {missing_local}. "
            "Run train_qlearning.py after changing config/experiment_config.json."
        )
        local_files = [p for p in local_files if p.exists()]
    local_summaries = []
    for q_path in local_files:
        qtable = load_qtable(q_path)
        error = provenance_compatibility_error(q_path, probe_env, q_path.stem)
        error = error or qtable_compatibility_error(qtable, probe_env, q_path.stem)
        if error:
            print(f"Evaluation ignoree: {error}")
            skipped_rows.append({"model": q_path.stem, "path": str(q_path), "reason": error})
            continue
        rows = evaluate_qtable(qtable, graph_path, scenarios, args.max_steps, args.seed)
        for row in rows:
            detailed_rows.append({"method": "local", "model": q_path.stem, **row})
        summary = summarize(rows, "Independent local Q-learning", q_path.stem)
        summary_rows.append(summary)
        local_summaries.append(summary)

    if local_summaries:
        summary_rows.append(
            {
                "method": "Independent local Q-learning",
                "model": "local_mean",
                "mean_reward": float(np.mean([r["mean_reward"] for r in local_summaries])),
                "std_reward": float(np.mean([r["std_reward"] for r in local_summaries])),
                "success_rate": float(np.mean([r["success_rate"] for r in local_summaries])),
                "mean_steps": float(np.mean([r["mean_steps"] for r in local_summaries])),
                "n_eval_episodes": int(args.eval_episodes * len(local_summaries)),
            }
        )

    centralized_path = ROOT / args.centralized_model
    if centralized_path.exists():
        centralized_q = load_qtable(centralized_path)
        error = provenance_compatibility_error(centralized_path, probe_env, "centralized_global")
        error = error or qtable_compatibility_error(centralized_q, probe_env, "centralized_global")
        if error:
            print(f"Evaluation ignoree: {error}")
            skipped_rows.append({"model": "centralized_global", "path": str(centralized_path), "reason": error})
        else:
            rows = evaluate_qtable(centralized_q, graph_path, scenarios, args.max_steps, args.seed)
            for row in rows:
                detailed_rows.append({"method": "centralized", "model": "centralized_global", **row})
            summary_rows.append(summarize(rows, "Centralized Q-learning", "centralized_global"))
    else:
        print(
            f"Centralized Q-table not found: {centralized_path}. "
            "Run train_centralized_qlearning.py to compare centralized vs FedAvg."
        )

    fed_path = ROOT / args.fedavg_model
    if fed_path.exists():
        fed_q = load_qtable(fed_path)
        error = provenance_compatibility_error(fed_path, probe_env, "fedavg_final")
        error = error or qtable_compatibility_error(fed_q, probe_env, "fedavg_final")
        if error:
            print(f"Evaluation ignoree: {error}")
            skipped_rows.append({"model": "fedavg_final", "path": str(fed_path), "reason": error})
        else:
            rows = evaluate_qtable(fed_q, graph_path, scenarios, args.max_steps, args.seed)
            for row in rows:
                detailed_rows.append({"method": "fedavg", "model": "fedavg_final", **row})
            summary_rows.append(summarize(rows, "Q-learning FedAvg", "fedavg_final"))

    if skipped_rows and not summary_rows:
        print(
            "Aucune Q-table compatible avec le graphe courant. "
            "Les fichiers d'evaluation sont vides; consultez evaluation_skipped.csv."
        )

    write_csv(
        out_dir / "evaluation_detailed.csv",
        detailed_rows,
        fieldnames=["method", "model", "scenario", "start", "target", "reward", "success", "steps"],
    )
    write_csv(
        out_dir / "evaluation_summary.csv",
        summary_rows,
        fieldnames=[
            "method",
            "model",
            "mean_reward",
            "std_reward",
            "success_rate",
            "mean_steps",
            "n_eval_episodes",
        ],
    )
    write_csv(
        out_dir / "evaluation_skipped.csv",
        skipped_rows,
        fieldnames=["model", "path", "reason"],
    )

    local_history = read_csv(ROOT / args.local_history)
    centralized_history = read_csv(ROOT / args.centralized_history)
    fl_history = read_csv(ROOT / args.fedavg_history)

    training_rows = [
        summarize_training_history(
            "Independent local Q-learning",
            "local_independent",
            local_history,
            ROOT / args.local_dir,
            args.success_target,
            args.stability_window,
        ),
        summarize_training_history(
            "Centralized Q-learning",
            "centralized",
            centralized_history,
            ROOT / args.centralized_model,
            args.success_target,
            args.stability_window,
        ),
        summarize_training_history(
            "Q-learning FedAvg",
            "federated",
            fl_history,
            ROOT / args.fedavg_model,
            args.success_target,
            args.stability_window,
        ),
    ]
    write_csv(out_dir / "training_comparison_summary.csv", training_rows)

    plot_training_curves(
        local_history=local_history,
        centralized_history=centralized_history,
        fl_history=fl_history,
        out_dir=plots_dir,
    )
    plot_eval_bars(summary_rows, plots_dir)
    plot_training_summary(training_rows, plots_dir)

    print(f"Detailed evaluation: {out_dir / 'evaluation_detailed.csv'}")
    print(f"Summary evaluation:  {out_dir / 'evaluation_summary.csv'}")
    print(f"Training summary:    {out_dir / 'training_comparison_summary.csv'}")
    print(f"Plots:               {plots_dir}")


if __name__ == "__main__":
    main()
