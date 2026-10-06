import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG, get_nested
from config.run_layout import RunLayout, layout_for_graph
from environment.graph_env import make_env


ROOT = PROJECT_ROOT
FIG_DPI = 150
FIG_SIZE = (22, 10)
GIF_FPS = 1
COLORMAP_NAME = "tab20"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def default_model_path(algo: str, layout: RunLayout | None = None) -> Path:
    layout = layout or layout_for_graph(CONFIG["graph_path"])
    if algo == "qlearning":
        return layout.federated_results / "qlearning" / "global_qtable_final.pkl"
    return layout.federated_results / algo / "global_model_final.pt"


def validate_model_provenance(model_path: Path, env, algo: str) -> None:
    """Refuse a GIF when the model and the graph/GCN artifact do not match."""
    metadata_path = model_path.parent / "run_metadata.json"
    if not metadata_path.exists():
        metadata_path = model_path.parent / "train_meta.json"
    if not metadata_path.exists():
        raise ValueError(
            f"Modele sans provenance: {model_path}. Reentrainez-le avec la version actuelle "
            "avant de produire une evaluation ou un GIF."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_graph = metadata.get("graph_fingerprint")
    if expected_graph != getattr(env, "graph_fingerprint", None):
        raise ValueError(
            "Le modele a ete entraine sur un autre graphe. Reentrainez-le avant la visualisation."
        )

    if algo in {"dqn", "ppo"}:
        expected_embedding = metadata.get("embedding_metadata", {}).get("embedding_checksum")
        actual_embedding = getattr(env, "embedding_metadata", {}).get("embedding_checksum")
        if not expected_embedding or expected_embedding != actual_embedding:
            raise ValueError(
                "Les embeddings GCN du modele et ceux de la visualisation different. "
                "Reentrainez le modele avec l'artefact GCN actuel."
            )


def load_qtable(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "Q" in data:
        return data["Q"]
    if isinstance(data, np.ndarray):
        return data
    raise TypeError(f"Format Q-table non supporte: {path}")


def load_torch_state(path: Path):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    if isinstance(data, dict):
        return data
    raise TypeError(f"Format modele PyTorch non supporte: {path}")


def load_policy(algo: str, model_path: Path, emb_dim: int, hidden_dim: int):
    if algo == "qlearning":
        return load_qtable(model_path)

    state = load_torch_state(model_path)

    if algo == "ppo":
        from models.ppo_network import PPOActorCritic

        model = PPOActorCritic(emb_dim=emb_dim, hidden_dim=hidden_dim, ortho_init=False)
        model.load_state_dict(state)
        model.eval()
        return model

    if algo == "dqn":
        from models.dqn_network import DQN, validate_standard_dqn_state_dict

        validate_standard_dqn_state_dict(state, str(model_path))
        model = DQN(emb_dim=emb_dim, hidden_dim=hidden_dim)
        model.online.load_state_dict(state)
        model.target.load_state_dict(state)
        model.eval()
        return model

    raise ValueError(f"Algorithme non supporte: {algo}")


def make_colors(n: int, cmap_name: str = COLORMAP_NAME):
    cmap = plt.get_cmap(cmap_name)
    vals = cmap(np.linspace(0, 1, max(n, 1)))
    colors = []
    for rgba in vals:
        r, g, b, _ = rgba
        colors.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return colors


def qlearning_action(qtable, graph_env, target_node, mask):
    valid = np.where(mask.astype(bool))[0]
    if len(valid) == 0:
        return 0

    current_idx = graph_env.node2idx[graph_env.current_node]
    target_idx = graph_env.node2idx[target_node]
    qvals = qtable[current_idx, target_idx]

    masked = np.full_like(qvals, -np.inf, dtype=np.float32)
    masked[valid] = qvals[valid]
    return int(np.argmax(masked))


def neural_action(policy, algo: str, obs: dict, emb_dim: int):
    import torch

    valid = np.where(obs["mask"].astype(bool))[0]
    if len(valid) == 0:
        return 0

    state = torch.tensor(obs["state"], dtype=torch.float32)
    neighbors = torch.tensor(obs["neighbors"], dtype=torch.float32)
    current_emb = state[:emb_dim]
    target_emb = state[emb_dim : 2 * emb_dim]
    valid_neighbors = neighbors[valid]

    with torch.no_grad():
        if algo == "ppo":
            dist, _ = policy(current_emb, target_emb, valid_neighbors)
            local_action = int(dist.probs.argmax().item())
        elif algo == "dqn":
            q_values = policy.online(current_emb, target_emb, valid_neighbors)
            local_action = int(q_values.argmax().item())
        else:
            raise ValueError(f"Action neuronale non supportee pour: {algo}")

    return int(valid[local_action])


def choose_action(policy, algo: str, graph_env, obs: dict, emb_dim: int):
    if algo == "qlearning":
        return qlearning_action(policy, graph_env, graph_env.target_node, obs["mask"])
    return neural_action(policy, algo, obs, emb_dim)


def simulate_episode(env, policy, algo: str, emb_dim: int, max_steps: int):
    # make_env returns GraphEnv for one robot and MultiRobotEnv otherwise.
    # Normalize both APIs here so the visualization code can treat them alike.
    is_multi_robot = hasattr(env, "envs")
    if is_multi_robot:
        robot_envs = env.envs
        obs_list = env.reset()
    else:
        robot_envs = [env]
        observation, _ = env.reset()
        obs_list = [observation]

    n_robots = len(robot_envs)
    paths = [[graph_env.current_node] for graph_env in robot_envs]
    successes = [0 for _ in range(n_robots)]
    done_steps = [max_steps for _ in range(n_robots)]

    for step in range(max_steps):
        actions = []
        for i, graph_env in enumerate(robot_envs):
            if graph_env._terminated or graph_env._truncated:
                actions.append(0)
                continue
            actions.append(choose_action(policy, algo, graph_env, obs_list[i], emb_dim))

        if is_multi_robot:
            obs_list, rewards, dones, infos = env.step(actions)
        else:
            observation, reward, terminated, truncated, info = env.step(actions[0])
            obs_list = [observation]
            rewards = [reward]
            dones = [bool(terminated or truncated)]
            infos = [info]

        for i, graph_env in enumerate(robot_envs):
            paths[i].append(graph_env.current_node)
            if dones[i] and done_steps[i] == max_steps:
                done_steps[i] = step + 1
                successes[i] = int(graph_env.current_node == graph_env.target_node)

        if all(dones):
            break

    return paths, successes, done_steps


def build_pos_per_floor(G, scale=10.0, gap_x=25):
    floors = {}
    for node in G.nodes():
        floor = G.nodes[node].get("floor", "Floor_0")
        floors.setdefault(floor, []).append(node)

    pos = {}
    for idx, floor in enumerate(sorted(floors.keys())):
        sub_nodes = floors[floor]
        sub_pos = nx.spring_layout(G.subgraph(sub_nodes), seed=42, k=1.8, iterations=200)
        for node, (x, y) in sub_pos.items():
            pos[node] = (x * scale + idx * gap_x, y * scale)
    return pos


def render_static_graph(G, pos, paths, colors, title):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_facecolor("#FAFAFA")

    corridor_edges, stair_edges, door_edges = [], [], []
    for u, v, data in G.edges(data=True):
        edge_type = data.get("type", "corridor")
        if edge_type == "stair":
            stair_edges.append((u, v))
        elif edge_type == "door":
            door_edges.append((u, v))
        else:
            corridor_edges.append((u, v))

    nx.draw_networkx_edges(G, pos, edgelist=corridor_edges, edge_color="#B8E0B8", alpha=0.35, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=door_edges, edge_color="#34495E", alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=stair_edges, edge_color="#E67E22", style="dashed", alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color="#cccccc", ax=ax)
    nx.draw_networkx_labels(
        G,
        {n: (x, y + 0.6) for n, (x, y) in pos.items()},
        {n: str(n) for n in G.nodes()},
        font_size=7,
        ax=ax,
    )

    for i, path in enumerate(paths):
        color = colors[i] if i < len(colors) else "#ff0000"
        coords = [pos[node] for node in path if node in pos]
        if len(coords) >= 2:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.95)
            ax.scatter(xs[0], ys[0], color=color, s=80, edgecolor="black")
            ax.scatter(xs[-1], ys[-1], color=color, marker="X", s=80, edgecolor="black")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize federated RL navigation policies.")
    parser.add_argument("--algo", choices=["qlearning", "ppo", "dqn"], required=True)
    parser.add_argument(
        "--model",
        default=None,
        help="Modele global. Par defaut: runs/<batiment>/results/federated/<algo>.",
    )
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
    graph_path = resolve_path(args.graph)
    layout = layout_for_graph(graph_path)
    model_path = resolve_path(args.model) if args.model else default_model_path(args.algo, layout)
    out_dir = (
        resolve_path(args.out_dir)
        if args.out_dir
        else layout.visualizations / "federated" / args.algo
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modele introuvable: {model_path}. "
            f"Lancez d'abord: python train_federated.py --algo {args.algo}"
        )

    env, G, emb = make_env(
        str(graph_path),
        n_robots=args.robots,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    validate_model_provenance(model_path, env, args.algo)
    policy = load_policy(args.algo, model_path, args.emb_dim, args.hidden_dim)
    colors = make_colors(args.robots)
    pos = build_pos_per_floor(G)

    frames = []
    summary_rows = []
    for ep in range(args.episodes):
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
            f"{args.algo.upper()} FedAvg | Episode {ep + 1} | "
            f"success={success_rate:.0%} | mean_steps={mean_steps:.1f}"
        )
        frames.append(render_static_graph(G, pos, paths, colors, title=title))
        summary_rows.append((ep + 1, success_rate, mean_steps))

    out_path = out_dir / "episodes.gif"
    imageio.mimsave(out_path, frames, fps=GIF_FPS)

    summary_path = out_dir / "visualization_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("episode,success_rate,mean_steps\n")
        for episode, success_rate, mean_steps in summary_rows:
            f.write(f"{episode},{success_rate},{mean_steps}\n")

    print(f"Saved {out_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
