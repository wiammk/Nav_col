import io
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG, get_nested
from config.run_layout import layout_for_graph
from environment.graph_env import make_env


GRAPH_PATH = CONFIG["graph_path"]
RUN_LAYOUT = layout_for_graph(GRAPH_PATH)
Q_DIR = RUN_LAYOUT.local_models
OUT_DIR = RUN_LAYOUT.visualizations / "trained" / "qlearning" / "local"
N_EPISODES = get_nested("visualization", "episodes", default=10)
MAX_STEPS = CONFIG["max_steps"]
N_ROBOTS = CONFIG["n_robots"]
GIF_FPS_EPISODES = 1
FIG_DPI = 150
FIG_SIZE_EPISODE = (22, 10)
COLORMAP_NAME = "tab20"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


def load_q(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "Q" in data:
        return data["Q"]
    return data


def q_files_from_config():
    model_dir = Path(Q_DIR)
    files = [model_dir / f"robot_{i}.pkl" for i in range(int(N_ROBOTS))]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Q-tables manquantes pour n_robots={N_ROBOTS}: {missing}. "
            "Relance train_qlearning.py apres modification de config/experiment_config.json."
        )
    return files


def make_colors(n, cmap_name=COLORMAP_NAME):
    cmap = plt.get_cmap(cmap_name)
    vals = cmap(np.linspace(0, 1, max(n, 1)))
    colors = []
    for rgba in vals:
        r, g, b, _ = rgba
        colors.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return colors


def greedy_action_from_q(qtable, current_idx, target_idx, mask):
    valid = np.where(mask.astype(bool))[0]
    if len(valid) == 0:
        return 0
    qvals = qtable[current_idx, target_idx]
    masked = np.full_like(qvals, -np.inf, dtype=np.float32)
    masked[valid] = qvals[valid]
    return int(np.argmax(masked))


def simulate_episode(env, agents_q, max_steps=MAX_STEPS):
    obs_list = env.reset()
    paths = [[env.envs[i].current_node] for i in range(env.n_robots)]

    for _ in range(max_steps):
        actions = []
        for i in range(env.n_robots):
            obs = obs_list[i]
            cur = env.envs[i].current_node
            tgt = env.envs[i].target_node
            actions.append(greedy_action_from_q(agents_q[i], cur, tgt, obs["mask"]))

        obs_list, rewards, dones, infos = env.step(actions)
        for i in range(env.n_robots):
            paths[i].append(env.envs[i].current_node)
        if all(dones):
            break

    return paths


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
    fig, ax = plt.subplots(figsize=FIG_SIZE_EPISODE)
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


def main():
    q_files = q_files_from_config()
    n_robots = int(N_ROBOTS)
    agents_q = [load_q(path) for path in q_files]
    colors = make_colors(n_robots)

    env, G, emb = make_env(GRAPH_PATH, n_robots=n_robots, max_steps=MAX_STEPS, seed=CONFIG["seed"])
    pos = build_pos_per_floor(G)

    frames = []
    for ep in range(int(N_EPISODES)):
        paths = simulate_episode(env, agents_q)
        frames.append(render_static_graph(G, pos, paths, colors, title=f"Episode {ep + 1}"))

    out_path = Path(OUT_DIR) / "episodes.gif"
    imageio.mimsave(out_path, frames, fps=GIF_FPS_EPISODES)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
