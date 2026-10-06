import argparse
import csv
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

from config import get_nested
from config.run_layout import latest_run_layout, layout_from_run_dir, resolve_project_path

DEFAULT_ALGOS = ["qlearning", "ppo", "dqn"]
DEFAULT_SUCCESS_TARGET = 0.95
DEFAULT_STABILITY_WINDOW = 5


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def history_path(algo: str, federated_results_dir: Path) -> Path:
    return federated_results_dir / algo / "fl_history.csv"


def model_path(algo: str, federated_results_dir: Path) -> Path:
    if algo == "qlearning":
        return federated_results_dir / algo / "global_qtable_final.pkl"
    return federated_results_dir / algo / "global_model_final.pt"


def dqn_model_compatibility_error(federated_results_dir: Path) -> str | None:
    """Retourne une explication si le résultat DQN vient de l'ancien modèle."""
    path = model_path("dqn", federated_results_dir)
    if not path.exists():
        return f"modèle DQN absent: {path}"
    try:
        import torch

        from models.dqn_network import validate_standard_dqn_state_dict

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        validate_standard_dqn_state_dict(state_dict, str(path))
    except (ImportError, OSError, TypeError, ValueError, RuntimeError, KeyError) as exc:
        return str(exc)
    return None


def file_size_bytes(path: Path) -> int:
    return int(path.stat().st_size) if path.is_file() else 0


def first_round_at(values: np.ndarray, rows, threshold: float):
    hits = np.where(values >= threshold)[0]
    if len(hits) == 0:
        return ""
    return int(float(rows[int(hits[0])]["round"]))


def estimate_num_clients(rows, episodes_per_round: int) -> int:
    if episodes_per_round <= 0:
        return 0
    n_episodes = np.array([int(float(r["n_episodes"])) for r in rows], dtype=np.int32)
    return int(round(float(np.median(n_episodes)) / float(episodes_per_round)))


def load_histories(algos, federated_results_dir: Path):
    histories = {}
    missing = []
    for algo in algos:
        path = history_path(algo, federated_results_dir)
        rows = read_csv(path)
        if not rows:
            missing.append(str(path))
            continue
        if algo == "dqn":
            compatibility_error = dqn_model_compatibility_error(federated_results_dir)
            if compatibility_error:
                missing.append(
                    f"{path} ignoré: {compatibility_error}"
                )
                continue
        histories[algo] = [canonical_history_row(row) for row in rows]
    return histories, missing


def canonical_history_row(row):
    """Accept both legacy and current FedAvg history column names.

    Current training records frozen global-policy evaluation metrics with the
    ``global_`` prefix. Older experiment archives used the shorter names.
    Downstream comparisons operate on one stable schema so both remain usable.
    """
    aliases = {
        "mean_reward": ("global_mean_reward", "mean_reward"),
        "std_reward": ("global_std_reward", "std_reward"),
        "success_rate": ("global_success_rate", "success_rate"),
    }
    normalized = dict(row)
    for target, candidates in aliases.items():
        value = next((row[name] for name in candidates if row.get(name) not in (None, "")), None)
        if value is None:
            raise ValueError(
                f"Historique FedAvg invalide: colonne {target!r} absente "
                f"(alias acceptes: {', '.join(candidates)})"
            )
        normalized[target] = value
    return normalized


def summarize_history(
    algo: str,
    rows,
    federated_results_dir: Path,
    episodes_per_round: int,
    success_target: float,
    stability_window: int,
):
    rows = [canonical_history_row(row) for row in rows]
    rewards = np.array([float(r["mean_reward"]) for r in rows], dtype=np.float32)
    successes = np.array([float(r["success_rate"]) for r in rows], dtype=np.float32)
    std_rewards = np.array([float(r["std_reward"]) for r in rows], dtype=np.float32)
    episodes = np.array([int(float(r["n_episodes"])) for r in rows], dtype=np.int32)

    best_reward_idx = int(np.argmax(rewards))
    best_success_idx = int(np.argmax(successes))
    model_bytes = file_size_bytes(model_path(algo, federated_results_dir))
    num_clients = estimate_num_clients(rows, episodes_per_round)
    total_upload_bytes = model_bytes * len(rows) * num_clients
    total_download_bytes = model_bytes * len(rows) * num_clients
    reward_deltas = np.diff(rewards)
    window = min(stability_window, len(rewards))
    last_rewards = rewards[-window:]

    return {
        "algo": algo,
        "rounds": len(rows),
        "estimated_clients": num_clients,
        "total_local_episodes": int(episodes.sum()),
        "initial_reward": float(rewards[0]),
        "final_reward": float(rewards[-1]),
        "reward_gain": float(rewards[-1] - rewards[0]),
        "best_reward": float(rewards[best_reward_idx]),
        "best_reward_round": int(float(rows[best_reward_idx]["round"])),
        "initial_success_rate": float(successes[0]),
        "final_success_rate": float(successes[-1]),
        "best_success_rate": float(successes[best_success_idx]),
        "best_success_round": int(float(rows[best_success_idx]["round"])),
        "round_to_success_target": first_round_at(successes, rows, success_target),
        "round_to_100_success": first_round_at(successes, rows, 1.0),
        "final_std_reward": float(std_rewards[-1]),
        "mean_std_reward": float(std_rewards.mean()),
        "reward_volatility": float(np.std(reward_deltas)) if len(reward_deltas) else 0.0,
        "last_window_reward_std": float(np.std(last_rewards)) if len(last_rewards) else 0.0,
        "model_size_bytes": model_bytes,
        "estimated_upload_bytes": int(total_upload_bytes),
        "estimated_download_bytes": int(total_download_bytes),
        "estimated_total_comm_bytes": int(total_upload_bytes + total_download_bytes),
        "final_model_exists": int(model_path(algo, federated_results_dir).exists()),
    }


def combined_round_rows(histories):
    rows = []
    for algo, history in histories.items():
        for raw_row in history:
            row = canonical_history_row(raw_row)
            rows.append(
                {
                    "algo": algo,
                    "round": int(float(row["round"])),
                    "mean_reward": float(row["mean_reward"]),
                    "std_reward": float(row["std_reward"]),
                    "success_rate": float(row["success_rate"]),
                    "n_episodes": int(float(row["n_episodes"])),
                }
            )
    return rows


def plot_metric(histories, metric: str, ylabel: str, title: str, out_path: Path):
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(9, 5))
    for algo, rows in histories.items():
        x = np.array([int(float(r["round"])) for r in rows])
        y = np.array([float(r[metric]) for r in rows])
        plt.plot(x, y, marker="o", label=algo.upper())

    if metric == "success_rate":
        plt.ylim(-0.05, 1.05)
    plt.xlabel("Federated round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_final_bars(summary_rows, out_path: Path):
    if not HAS_MATPLOTLIB or not summary_rows:
        return

    labels = [r["algo"].upper() for r in summary_rows]
    rewards = [float(r["final_reward"]) for r in summary_rows]
    successes = [float(r["final_success_rate"]) for r in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, rewards, color="#4C78A8")
    axes[0].set_title("Final reward")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(labels, successes, color="#F58518")
    axes[1].set_title("Final success rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Federated comparison: Q-learning vs DQN vs PPO")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_complexity_bars(summary_rows, out_path: Path):
    if not HAS_MATPLOTLIB or not summary_rows:
        return

    labels = [r["algo"].upper() for r in summary_rows]
    model_mb = [float(r["model_size_bytes"]) / (1024 * 1024) for r in summary_rows]
    comm_mb = [float(r["estimated_total_comm_bytes"]) / (1024 * 1024) for r in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, model_mb, color="#54A24B")
    axes[0].set_title("Final model size")
    axes[0].set_ylabel("MiB")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(labels, comm_mb, color="#B279A2")
    axes[1].set_title("Estimated federated\ncommunication")
    axes[1].set_ylabel("MiB")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Practical complexity of federated models", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=3.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare RL algorithms inside the same Federated Learning architecture."
    )
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS, choices=DEFAULT_ALGOS)
    parser.add_argument(
        "--out-dir",
        default=None,
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
    )
    parser.add_argument(
        "--federated-results-dir",
        default=None,
        help="Dossier racine contenant qlearning/, ppo/ et dqn/ federes.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Dossier runs/<batiment> utilise pour les chemins non specifies.",
    )
    parser.add_argument(
        "--episodes-per-round",
        type=int,
        default=get_nested("federated", "episodes_per_round", default=50),
        help="Episodes locaux par robot et par round, utilise pour estimer le nombre de clients.",
    )
    parser.add_argument(
        "--success-target",
        type=float,
        default=DEFAULT_SUCCESS_TARGET,
        help="Seuil de succes utilise pour mesurer la convergence.",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=DEFAULT_STABILITY_WINDOW,
        help="Nombre de derniers rounds utilises pour mesurer la stabilite finale.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    layout = layout_from_run_dir(args.run_dir) if args.run_dir else latest_run_layout()
    out_dir = (
        resolve_project_path(args.out_dir)
        if args.out_dir
        else layout.evaluation_results / "federated_algorithms"
    )
    plots_dir = (
        resolve_project_path(args.plots_dir)
        if args.plots_dir
        else layout.evaluation_plots / "federated_algorithms"
    )
    federated_results_dir = (
        resolve_project_path(args.federated_results_dir)
        if args.federated_results_dir
        else layout.federated_results
    )

    histories, missing = load_histories(args.algos, federated_results_dir)
    if missing:
        print("Historiques manquants ou vides:")
        for path in missing:
            print(f"  - {path}")
        print("Lancez d'abord train_federated.py pour les algorithmes manquants.")

    if not histories:
        raise RuntimeError("Aucun historique FL disponible pour la comparaison.")

    summary_rows = [
        summarize_history(
            algo,
            rows,
            federated_results_dir,
            args.episodes_per_round,
            args.success_target,
            args.stability_window,
        )
        for algo, rows in histories.items()
    ]
    round_rows = combined_round_rows(histories)

    write_csv(out_dir / "fl_algorithms_summary.csv", summary_rows)
    write_csv(out_dir / "fl_algorithms_rounds.csv", round_rows)

    plot_metric(
        histories,
        "mean_reward",
        "Mean reward",
        "Mean reward per federated round",
        plots_dir / "reward_vs_round.png",
    )
    plot_metric(
        histories,
        "success_rate",
        "Success rate",
        "Success rate per federated round",
        plots_dir / "success_rate_vs_round.png",
    )
    plot_final_bars(summary_rows, plots_dir / "final_comparison.png")
    plot_complexity_bars(summary_rows, plots_dir / "complexity_communication.png")

    print(f"Summary: {out_dir / 'fl_algorithms_summary.csv'}")
    print(f"Rounds:  {out_dir / 'fl_algorithms_rounds.csv'}")
    print(f"Plots:   {plots_dir}")


if __name__ == "__main__":
    main()
