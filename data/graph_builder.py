"""
data/graph_builder.py
=====================
Étape 2 du pipeline de données.

Lit nodes.csv + edges.csv et construit un graphe NetworkX avec :
  - Vérification de connexité (+ réparation automatique)
  - Calcul des centralités (betweenness, closeness, degree)
  - Normalisation des features
  - Sauvegarde en graph.gpickle + nodes_with_centrality.csv
  - Rapport de qualité
  - Visualisation propre : étages côte à côte

Usage :
    python data/graph_builder.py
    python data/graph_builder.py --visualize
"""

import math
import pickle
import logging
import argparse
import datetime
import sys
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.console import configure_console_encoding
from config.run_layout import latest_run_layout

configure_console_encoding()

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── Chemins ─────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent
PROCESSED_DIR = latest_run_layout().processed
NODES_CSV     = PROCESSED_DIR / "nodes.csv"
EDGES_CSV     = PROCESSED_DIR / "edges.csv"
GRAPH_PKL     = PROCESSED_DIR / "graph.gpickle"

# ─── Couleurs ─────────────────────────────────────────────────────────────────
TYPE_COLORS = {
    "room"    : "#4A90D9",
    "corridor": "#7ED321",
    "stair"   : "#F5A623",
    "storage" : "#9B59B6",
    "office"  : "#1ABC9C",
    "hall"    : "#E74C3C",
    "exit"    : "#2ECC71",
    "toilet"  : "#95A5A6",
}

TYPE_LABELS_EN = {
    "room"    : "Room",
    "corridor": "Corridor",
    "stair"   : "Staircase",
    "storage" : "Storage",
    "office"  : "Office",
    "hall"    : "Hall",
    "exit"    : "Exit",
    "toilet"  : "Toilet",
}

FLOOR_LABELS_EN = {
    "Keller"          : "Basement",
    "Erdgeschoss"     : "Ground Floor",
    "1. Obergeschoss" : "First Floor",
    "2. Obergeschoss" : "Second Floor",
    "3. Obergeschoss" : "Third Floor",
    "Dachgeschoss"    : "Roof Level",
}

FLOOR_ORDER_EN = {
    "Basement"     : 0,
    "Ground Floor" : 1,
    "First Floor"  : 2,
    "Second Floor" : 3,
    "Third Floor"  : 4,
    "Roof Level"   : 5,
}


def translate_floor_name(floor_name: str) -> str:
    """Return an English publication label for a building storey."""
    original = str(floor_name).strip()
    return FLOOR_LABELS_EN.get(original, original)


def floor_sort_key(floor_name: str):
    """Sort known floors from the lowest level to the highest level."""
    english_name = translate_floor_name(floor_name)
    return FLOOR_ORDER_EN.get(english_name, 100), english_name.casefold()

EDGE_COLORS = {
    "door"    : "#2C3E50",
    "stair"   : "#E67E22",
    "corridor": "#27AE60",
    "proximity": "#95A5A6",
    "repair"  : "#C0392B",
}

EDGE_GCN_WEIGHTS = {
    "door": 1.0,
    "stair": 0.9,
    "corridor": 0.8,
    "proximity": 0.35,
    "repair": 0.15,
}


# ─────────────────────────────────────────────────────────────────────────────
#  CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate_nodes(path: Path) -> pd.DataFrame:
    log.info(f" ---Chargement nodes.csv--- : {path}")
    df = pd.read_csv(path)

    required = ["id", "x", "y"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    defaults = {
        "z": 0.0, "area": 25.0, "type": "room", "floor": "Floor_0",
        "floor_elev": 0.0, "risk": 0.05, "capacity": 2,
        "name": None, "guid": None,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    df["id"]         = df["id"].astype(int)
    df["name"]       = df["name"].fillna(df["id"].apply(lambda i: f"Space_{i}"))

    df["z"]          = df["z"].fillna(0.0)
    df["area"]       = df["area"].fillna(25.0).clip(lower=1.0)
    df["type"]       = df["type"].fillna("room")
    df["floor"]      = df["floor"].fillna("Floor_0")
    df["floor_elev"] = df["floor_elev"].fillna(0.0)
    df["risk"]       = df["risk"].fillna(0.05).clip(0.0, 1.0)
    df["capacity"]   = df["capacity"].fillna(2).astype(int).clip(lower=1)

    log.info(f"  {len(df)} nœuds chargés")
    log.info(f"  Types : {df['type'].value_counts().to_dict()}")
    return df


def load_and_validate_edges(path: Path, valid_ids: set) -> pd.DataFrame:
    log.info(f" ---Chargement edges.csv--- : {path}")
    df = pd.read_csv(path)

    required = ["from", "to"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    defaults = {"weight": None, "type": "door", "passable": True, "width": 1.0}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    df["from"]     = df["from"].astype(int)
    df["to"]       = df["to"].astype(int)
    df["passable"] = df["passable"].astype(bool)
    df["width"]    = df["width"].fillna(1.0).astype(float)
    df["type"]     = df["type"].fillna("door")

    n_before = len(df)
    df = df[df["from"].isin(valid_ids) & df["to"].isin(valid_ids)]
    df = df[df["from"] != df["to"]]
    if n_before != len(df):
        log.warning(f"  {n_before - len(df)} arêtes invalides supprimées")

    log.info(f"  {len(df)} arêtes chargées")
    log.info(f"  Types : {df['type'].value_counts().to_dict()}")
    return df


def validate_spatial_quality(
    df_nodes: pd.DataFrame,
    df_edges: pd.DataFrame,
    max_origin_ratio: float = 0.10,
    max_zero_distance_ratio: float = 0.05,
    min_unique_xy_ratio: float = 0.25,
) -> None:
    """Fail fast instead of building/training on a degenerate spatial graph."""
    xy = df_nodes[["x", "y"]].astype(float)
    xyz = df_nodes[["x", "y", "z"]].astype(float)
    if not np.isfinite(xyz.to_numpy()).all():
        raise ValueError("Qualité spatiale invalide : coordonnées non finies")

    origin_ratio = ((xy["x"].abs() <= 1e-9) & (xy["y"].abs() <= 1e-9)).mean()
    unique_xy_ratio = len(xy.round(6).drop_duplicates()) / len(df_nodes)
    positions = {
        int(row.id): np.asarray([row.x, row.y, row.z], dtype=float)
        for row in df_nodes.itertuples()
    }
    zero_edges = 0
    for edge in df_edges.itertuples():
        if np.linalg.norm(positions[int(edge[1])] - positions[int(edge[2])]) <= 1e-6:
            zero_edges += 1
    zero_distance_ratio = zero_edges / len(df_edges) if len(df_edges) else 0.0

    errors = []
    if origin_ratio > max_origin_ratio:
        errors.append(f"x=y=0 pour {origin_ratio:.1%} des nœuds")
    if unique_xy_ratio < min_unique_xy_ratio:
        errors.append(f"positions XY distinctes insuffisantes ({unique_xy_ratio:.1%})")
    if zero_distance_ratio > max_zero_distance_ratio:
        errors.append(f"distances d'arêtes nulles trop nombreuses ({zero_distance_ratio:.1%})")

    floor_by_id = {int(row.id): row.floor for row in df_nodes.itertuples()}
    floor_elevations = df_nodes.groupby("floor")["floor_elev"].median().to_dict()
    invalid_horizontal = []
    stair_pairs = set()
    for edge in df_edges.itertuples():
        floor_from = floor_by_id[int(edge[1])]
        floor_to = floor_by_id[int(edge[2])]
        edge_type = str(edge.type)
        if edge_type in {"door", "proximity", "corridor"} and floor_from != floor_to:
            invalid_horizontal.append((int(edge[1]), int(edge[2])))
        if edge_type == "stair":
            stair_pairs.add(frozenset((floor_from, floor_to)))
    if invalid_horizontal:
        errors.append(f"{len(invalid_horizontal)} connexions horizontales traversent des étages")

    ordered_floors = sorted(floor_elevations, key=floor_elevations.get)
    missing_floor_links = [
        (ordered_floors[idx], ordered_floors[idx + 1])
        for idx in range(len(ordered_floors) - 1)
        if frozenset((ordered_floors[idx], ordered_floors[idx + 1])) not in stair_pairs
    ]
    if missing_floor_links:
        errors.append(f"escaliers absents entre {len(missing_floor_links)} étages adjacents")
    if errors:
        raise ValueError(
            "Graphe refusé avant entraînement : " + " ; ".join(errors)
            + ". Corrigez l'extraction IFC."
        )

    log.info(
        "  Qualité spatiale validée : origine_xy=%.1f%%, xy_uniques=%.1f%%, distances_nulles=%.1f%%",
        origin_ratio * 100,
        unique_xy_ratio * 100,
        zero_distance_ratio * 100,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(df_nodes: pd.DataFrame, df_edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, row in df_nodes.iterrows():
        node_id = int(row["id"])
        attrs   = row.to_dict()
        attrs.pop("id", None)
        attrs["current_robots"] = 0
        attrs["color"] = TYPE_COLORS.get(attrs.get("type", "room"), "#4A90D9")
        G.add_node(node_id, **attrs)

    pos = {int(r["id"]): (r["x"], r["y"], r["z"]) for _, r in df_nodes.iterrows()}

    for _, row in df_edges.iterrows():
        u, v   = int(row["from"]), int(row["to"])
        weight = row["weight"]
        if pd.isna(weight) or weight <= 0:
            pu, pv = pos.get(u, (0, 0, 0)), pos.get(v, (0, 0, 0))
            weight = max(math.sqrt(sum((a - b)**2 for a, b in zip(pu, pv))), 0.1)
        G.add_edge(u, v,
                   weight  = round(float(weight), 3),
                   type    = str(row["type"]),
                   passable= bool(row["passable"]),
                   width   = float(row["width"]),
                   gcn_weight= EDGE_GCN_WEIGHTS.get(str(row["type"]), 0.8),
                   color   = EDGE_COLORS.get(str(row["type"]), "#95A5A6"))

    log.info(f"  Graphe initial : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    return G


# ─────────────────────────────────────────────────────────────────────────────
#  CONNEXITÉ
# ─────────────────────────────────────────────────────────────────────────────

def ensure_connectivity(G: nx.Graph) -> nx.Graph:
    if nx.is_connected(G):
        log.info("  ✅ Graphe connexe")
        return G

    components = list(nx.connected_components(G))
    log.warning(f"  ⚠️  Graphe non connexe : {len(components)} composantes")

    def dist3d(a, b):
        pa = (G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[a]["z"])
        pb = (G.nodes[b]["x"], G.nodes[b]["y"], G.nodes[b]["z"])
        return math.sqrt(sum((x - y)**2 for x, y in zip(pa, pb)))

    repairs = 0
    for i in range(1, len(components)):
        comp_a = sorted(components[0])
        comp_b = sorted(components[i])
        best_dist, best_pair = float("inf"), (comp_a[0], comp_b[0])
        for u in comp_a:
            for v in comp_b:
                d = dist3d(u, v)
                if d < best_dist:
                    best_dist, best_pair = d, (u, v)
        u, v = best_pair
        G.add_edge(u, v,
                   weight  = round(best_dist, 3),
                   type    = "repair",
                   passable= True,
                   width   = 1.5,
                   repaired= True,
                   gcn_weight= EDGE_GCN_WEIGHTS["repair"],
                   color   = EDGE_COLORS["repair"])
        components[0] = components[0] | components[i]
        repairs += 1
        log.info(f"  ---Connexion réparatrice--- : {u} ↔ {v} (dist={best_dist:.1f}m)")

    log.info(f"  {repairs} arêtes de réparation ajoutées")

    if not nx.is_connected(G):
        raise RuntimeError(
            "Réparation de connexité échouée — vérifier les données CSV."
        )
    log.info("  ✅ Graphe connexe après réparation")
    return G


# ─────────────────────────────────────────────────────────────────────────────
#  CENTRALITÉS & FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_centralities(G: nx.Graph) -> nx.Graph:
    log.info(" ---Calcul des centralités---")
    n = G.number_of_nodes()
    if n > 500:
        bet = nx.betweenness_centrality(G, weight="weight", k=100)
    else:
        bet = nx.betweenness_centrality(G, weight="weight")
    clo = nx.closeness_centrality(G, distance="weight")
    deg = nx.degree_centrality(G)
    for node in G.nodes():
        G.nodes[node]["betweenness"] = round(bet[node], 4)
        G.nodes[node]["closeness"]   = round(clo[node], 4)
        G.nodes[node]["degree_c"]    = round(deg[node], 4)
    log.info(f"  Betweenness max={max(bet.values()):.3f}, mean={np.mean(list(bet.values())):.3f}")
    log.info(f"  Closeness   max={max(clo.values()):.3f}, mean={np.mean(list(clo.values())):.3f}")
    return G


def normalize_features(G: nx.Graph) -> nx.Graph:
    log.info(" ---Normalisation des features---")
    nodes = list(G.nodes())

    def normalize(values):
        mn, mx = min(values), max(values)
        return [0.5] * len(values) if mx - mn < 1e-8 else [(v - mn) / (mx - mn) for v in values]

    for attr in ["x", "y", "z", "area"]:
        vals  = [G.nodes[n].get(attr, 0) for n in nodes]
        norms = normalize(vals)
        for node, v in zip(nodes, norms):
            G.nodes[node][f"{attr}_norm"] = round(v, 4)

    weights = [G.edges[e]["weight"] for e in G.edges()]
    if weights:
        w_max = max(weights)
        for e in G.edges():
            G.edges[e]["weight_norm"] = round(G.edges[e]["weight"] / w_max, 4)

    log.info("  ✅ Normalisation terminée")
    return G


def add_graph_metadata(G: nx.Graph, source: str = "ifc") -> nx.Graph:
    G.graph["source"]       = source
    G.graph["created_at"]   = str(datetime.datetime.now())
    G.graph["n_nodes"]      = G.number_of_nodes()
    G.graph["n_edges"]      = G.number_of_edges()
    G.graph["is_connected"] = nx.is_connected(G)
    floors = set(G.nodes[n].get("floor", "unknown") for n in G.nodes())
    G.graph["floors"]       = sorted(floors)
    G.graph["n_floors"]     = len(floors)
    type_counts = {}
    for n in G.nodes():
        t = G.nodes[n].get("type", "room")
        type_counts[t] = type_counts.get(t, 0) + 1
    G.graph["node_types"] = type_counts
    return G


# ─────────────────────────────────────────────────────────────────────────────
#  RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_graph_report(G: nx.Graph):
    sep = "=" * 55
    log.info(sep)
    log.info(" ---RAPPORT DU GRAPHE---")
    log.info(sep)
    log.info(f"  Nœuds       : {G.number_of_nodes()}")
    log.info(f"  Arêtes      : {G.number_of_edges()}")
    log.info(f"  Connexe     : {nx.is_connected(G)}")

    if nx.is_connected(G) and G.number_of_nodes() < 300:
        log.info(f"  Diamètre    : {nx.diameter(G)}")
    else:
        log.info(f"  Diamètre    : N/A (graphe trop grand ou non connexe)")

    log.info(f"  Densité     : {nx.density(G):.4f}")
    avg_deg = sum(d for _, d in G.degree()) / G.number_of_nodes()
    log.info(f"  Degré moyen : {avg_deg:.2f}")

    log.info("🔷 Types de nœuds :")
    for t, cnt in sorted(G.graph.get("node_types", {}).items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / G.number_of_nodes() * 20)
        log.info(f"  {t:<12} : {cnt:4d}  {bar}")

    edge_types = {}
    for u, v, d in G.edges(data=True):
        t = d.get("type", "unknown")
        edge_types[t] = edge_types.get(t, 0) + 1
    log.info("🔷 Types d'arêtes :")
    for t, cnt in sorted(edge_types.items(), key=lambda x: -x[1]):
        log.info(f"  {t:<12} : {cnt:4d}")

    log.info(f"🔷 Étages ({G.graph.get('n_floors', '?')}) :")
    for floor in G.graph.get("floors", []):
        n_floor = sum(1 for n in G.nodes() if G.nodes[n].get("floor") == floor)
        log.info(f"  {floor:<15} : {n_floor} espaces")

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    if weights:
        log.info(f"🔷 Distances : min={min(weights):.2f}m  max={max(weights):.2f}m  moy={np.mean(weights):.2f}m")

    top_bw = sorted(G.nodes(), key=lambda n: G.nodes[n].get("betweenness", 0), reverse=True)[:3]
    log.info("🔷 Top 3 nœuds (betweenness) :")
    for n in top_bw:
        log.info(f"  Nœud {n:3d} ({G.nodes[n].get('name', '?'):<20}) bw={G.nodes[n].get('betweenness', 0):.3f}")

    log.info(sep)
    log.info("✅ Graphe prêt pour l'environnement RL")
    log.info(sep)


# ─────────────────────────────────────────────────────────────────────────────
#  SAUVEGARDE / CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(G: nx.Graph, path: Path) -> None:
    """Sauvegarde uniquement graph.gpickle."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  ✅ graph.gpickle → {output_path}")


def save_nodes_with_centrality(G: nx.Graph, dir_path: Path) -> None:
    """
    Exporte nodes_with_centrality.csv depuis le graphe enrichi.
    Appelée explicitement dans build_pipeline() — pas un effet de bord.
    """
    nodes_updated = []
    for n in G.nodes():
        d = dict(G.nodes[n])
        d["id"] = n
        d.pop("current_robots", None)
        nodes_updated.append(d)

    df_nodes = pd.DataFrame(nodes_updated)
    priority = ["id", "name", "x", "y", "z", "area", "type", "floor", "floor_elev",
                "risk", "capacity", "betweenness", "closeness", "degree_c"]
    cols = priority + [c for c in df_nodes.columns if c not in priority]
    cols = [c for c in cols if c in df_nodes.columns]
    out  = Path(dir_path) / "nodes_with_centrality.csv"
    df_nodes[cols].sort_values("id").reset_index(drop=True).to_csv(out, index=False)
    log.info(f"  ✅ nodes_with_centrality.csv → {out}")


def load_graph(path: Path) -> nx.Graph:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"graph.gpickle introuvable : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_graph(
    G: nx.Graph,
    save_path: Optional[Path] = None,
    building_name: str = "Building",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
    except ImportError:
        log.warning("matplotlib is unavailable; graph visualization skipped.")
        return

    floors = {}
    for n in G.nodes():
        floor = G.nodes[n].get("floor", "Floor_0")
        floors.setdefault(floor, []).append(n)

    floor_names = sorted(floors.keys(), key=floor_sort_key)
    n_floors = len(floor_names)
    floor_gap_x = 25

    pos = {}
    for idx, floor in enumerate(floor_names):
        floor_nodes = floors[floor]
        subgraph = G.subgraph(floor_nodes)
        sub_pos = nx.spring_layout(subgraph, seed=42, k=1.8, iterations=200)
        for n, (x, y) in sub_pos.items():
            pos[n] = (x * 10 + idx * floor_gap_x, y * 10)

    fig, ax = plt.subplots(figsize=(22, 10))
    ax.set_facecolor("#FAFAFA")

    # Reserve independent bands for the main title, graph statistics,
    # floor labels, and graph content so that text cannot overlap.
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.06, top=0.82)

    edge_groups = {}
    for u, v, data in G.edges(data=True):
        edge_type = str(data.get("type", "proximity")).lower()
        edge_groups.setdefault(edge_type, []).append((u, v))

    edge_styles = {
        "corridor": {
            "color": "#27AE60", "width": 1.4, "alpha": 0.45,
            "style": "solid", "label": "Corridor",
        },
        "proximity": {
            "color": "#B8E0B8", "width": 1.2, "alpha": 0.35,
            "style": "solid", "label": "Proximity",
        },
        "door": {
            "color": "#34495E", "width": 2.0, "alpha": 0.70,
            "style": "solid", "label": "Door",
        },
        "stair": {
            "color": "#E67E22", "width": 2.5, "alpha": 0.90,
            "style": "dashed", "label": "Stair",
        },
        "repair": {
            "color": "#C0392B", "width": 2.0, "alpha": 0.85,
            "style": "dotted", "label": "Connectivity repair",
        },
    }

    for edge_type, edges in edge_groups.items():
        style = edge_styles.get(edge_type, edge_styles["proximity"])
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=style["color"],
            width=style["width"],
            alpha=style["alpha"],
            style=style["style"],
            ax=ax,
        )

    for node_type, color in TYPE_COLORS.items():
        node_list = [
            n for n in G.nodes()
            if G.nodes[n].get("type") == node_type
        ]
        if not node_list:
            continue
        sizes = [350 + G.degree(n) * 40 for n in node_list]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_list,
            node_color=color,
            node_size=sizes,
            edgecolors="black",
            linewidths=1.3,
            alpha=0.95,
            ax=ax,
        )

    label_pos = {n: (x, y + 0.6) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G,
        label_pos,
        {n: str(n) for n in G.nodes()},
        font_size=7,
        font_weight="bold",
        bbox=dict(
            facecolor="white", edgecolor="none", alpha=0.75, pad=0.15,
        ),
        ax=ax,
    )

    for idx, floor in enumerate(floor_names):
        floor_nodes = floors[floor]
        x_center = float(np.mean([pos[node][0] for node in floor_nodes]))
        if idx > 0:
            separator_x = idx * floor_gap_x - floor_gap_x / 2
            ax.axvline(
                x=separator_x,
                color="#D6D6D6",
                linestyle="--",
                linewidth=1.5,
            )
        ax.text(
            x_center,
            1.015,
            translate_floor_name(floor),
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(
                facecolor="#EBF5FB",
                edgecolor="#5DADE2",
                boxstyle="round,pad=0.3",
            ),
        )

    node_patches = [
        mpatches.Patch(
            color=color,
            label=TYPE_LABELS_EN.get(
                node_type, node_type.replace("_", " ").title(),
            ),
        )
        for node_type, color in TYPE_COLORS.items()
        if any(
            G.nodes[n].get("type") == node_type for n in G.nodes()
        )
    ]

    edge_lines = []
    for edge_type in ("corridor", "proximity", "door", "stair", "repair"):
        if edge_type not in edge_groups:
            continue
        style = edge_styles[edge_type]
        edge_lines.append(
            mlines.Line2D(
                [],
                [],
                color=style["color"],
                linewidth=2,
                linestyle=style["style"],
                label=style["label"],
            )
        )

    node_legend = ax.legend(
        handles=node_patches,
        loc="upper left",
        fontsize=8,
        title="Space types",
        title_fontsize=9,
    )
    ax.add_artist(node_legend)
    ax.legend(
        handles=edge_lines,
        loc="lower left",
        fontsize=8,
        title="Connection types",
        title_fontsize=9,
    )

    fig.suptitle(
        f"IFC-Derived {building_name} Graph",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.925,
        (
            f"{G.number_of_nodes()} nodes  |  "
            f"{G.number_of_edges()} edges  |  "
            f"{n_floors} floors"
        ),
        ha="center",
        va="center",
        fontsize=12,
        color="#34495E",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        log.info(f"  ---Graph visualization saved--- : {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    nodes_path  : Path = NODES_CSV,
    edges_path  : Path = EDGES_CSV,
    output_path : Path = GRAPH_PKL,
    visualize   : bool = False,
    source      : str  = "ifc",
) -> nx.Graph:

    log.info("=" * 55)
    log.info(" ---CONSTRUCTION DU GRAPHE---")
    log.info("=" * 55)

    df_nodes  = load_and_validate_nodes(nodes_path)
    valid_ids = set(df_nodes["id"].astype(int))
    df_edges  = load_and_validate_edges(edges_path, valid_ids)
    validate_spatial_quality(df_nodes, df_edges)

    log.info("🔷 Construction du graphe NetworkX...")
    G = build_graph(df_nodes, df_edges)

    log.info("🔷 Vérification de la connexité...")
    G = ensure_connectivity(G)

    G = compute_centralities(G)
    G = normalize_features(G)
    G = add_graph_metadata(G, source=source)

    log.info(" ---Sauvegarde du graphe---")
    output_path = Path(output_path)
    save_graph(G, output_path)                              # .gpickle
    save_nodes_with_centrality(G, output_path.parent)      # FIX 5 : appel explicite

    print_graph_report(G)

    if visualize:
        log.info(" ---Visualisation du graphe---")
        visualize_graph(G, output_path.parent / "graph_visualization.png")

    return G


def main():
    parser = argparse.ArgumentParser(description="Graph Builder")
    parser.add_argument("--nodes",     type=str, default=str(NODES_CSV))
    parser.add_argument("--edges",     type=str, default=str(EDGES_CSV))
    parser.add_argument("--output",    type=str, default=str(GRAPH_PKL))
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--source",    type=str, default="ifc",
                        choices=["ifc", "synthetic"])
    args = parser.parse_args()

    build_pipeline(
        nodes_path  = Path(args.nodes),
        edges_path  = Path(args.edges),
        output_path = Path(args.output),
        visualize   = args.visualize,
        source      = args.source,
    )


if __name__ == "__main__":
    main()
