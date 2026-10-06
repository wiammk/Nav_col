"""Dense PyTorch GCN encoding, link-prediction training and frozen embedding caches."""

import hashlib
import json
import pickle
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR    = Path(__file__).parent
PROCESSED_DIR = MODELS_DIR.parent / "data" / "processed"
GRAPH_PKL     = PROCESSED_DIR / "graph.gpickle"

SPACE_TYPES = ["room", "corridor", "stair", "storage",
               "office", "hall", "exit", "toilet"]
TYPE2IDX    = {t: i for i, t in enumerate(SPACE_TYPES)}
N_TYPES     = len(SPACE_TYPES)   # 8

CONTINUOUS_FEATS = [
    "x_norm", "y_norm", "z_norm", "area_norm",
    "risk", "betweenness", "closeness", "degree_c",
]
N_CONTINUOUS  = len(CONTINUOUS_FEATS)   # 8
NODE_FEAT_DIM = N_TYPES + N_CONTINUOUS  # 16

DEFAULT_HIDDEN  = 64
DEFAULT_OUT_DIM = 64
DEFAULT_DROPOUT = 0.10
DEFAULT_LAYERS  = 2
EMBEDDING_ARTIFACT_VERSION = 1
EDGE_RELIABILITY = {
    "door": 1.0,
    "stair": 0.9,
    "corridor": 0.8,
    "proximity": 0.35,
    "repair": 0.15,
}


class GraphDataConverter:
    """
    Convertit un nx.Graph (issu de graph_builder.py) en tenseurs PyTorch.

    Méthode principale : nx_to_tensors(G) → (x, A_norm, node_map)

    node_map {nx_id → tensor_index} permet de retrouver l'embedding
    d'un nœud par son id NetworkX original.
    """

    def __init__(self):
        self._node_map: Optional[Dict[int, int]] = None


    def node_features(self, G) -> torch.Tensor:
        """
        Construit x [N, 16].

        Colonnes 0-7  : one-hot du type d'espace
        Colonnes 8-15 : x_norm, y_norm, z_norm, area_norm,
                        risk, betweenness, closeness, degree_c
        """
        nodes = sorted(G.nodes())
        rows  = []
        for n in nodes:
            attr     = G.nodes[n]
            one_hot  = [0.0] * N_TYPES
            one_hot[TYPE2IDX.get(attr.get("type", "room"), 0)] = 1.0
            cont     = [float(attr.get(f, 0.0)) for f in CONTINUOUS_FEATS]
            rows.append(one_hot + cont)

        x = torch.tensor(rows, dtype=torch.float32)

        if torch.isnan(x).any():
            log.warning("  ⚠️  NaN détectés dans les features → remplacés par 0")
            x = torch.nan_to_num(x, nan=0.0)

        return x   # [N, 16]


    def adjacency_matrix(self, G, sparse_threshold: int = 1000) -> torch.Tensor:
        """
        Calcule Ã = D^{-1/2}(A + I)D^{-1/2} — matrice dense [N, N].

        A + I : auto-boucles (chaque nœud se lit lui-même)
        D     : matrice diagonale des degrés de A + I
        Ã     : normalisation symétrique de Kipf & Welling (2017)

        Dense car N ≤ 1 000 pour ce projet.
        Voir NOTE SCALABILITÉ pour N > 1 000.
        """
        nodes   = sorted(G.nodes())
        n       = len(nodes)
        idx_map = {nd: i for i, nd in enumerate(nodes)}

        if n >= sparse_threshold:
            return self._sparse_normalized_adjacency(G, idx_map, n)

        A = torch.zeros(n, n)
        for u, v, data in G.edges(data=True):
            i, j       = idx_map[u], idx_map[v]
            reliability = float(
                data.get("gcn_weight", EDGE_RELIABILITY.get(data.get("type"), 0.8))
            )
            A[i, j]    = reliability
            A[j, i]    = reliability

        A_hat = A + torch.eye(n)                   # A + I

        deg         = A_hat.sum(dim=1)             # degré de chaque nœud
        d_inv_sqrt  = deg.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # nœud isolé → 0
        D           = torch.diag(d_inv_sqrt)

        return D @ A_hat @ D                       # Ã : [N, N]


    def _sparse_normalized_adjacency(self, G, idx_map: Dict[Any, int], n: int) -> torch.Tensor:
        """Build the same normalized adjacency without an N x N allocation."""
        rows, cols, values = [], [], []
        for u, v, data in G.edges(data=True):
            i, j = idx_map[u], idx_map[v]
            reliability = float(
                data.get("gcn_weight", EDGE_RELIABILITY.get(data.get("type"), 0.8))
            )
            rows.extend((i, j))
            cols.extend((j, i))
            values.extend((reliability, reliability))

        rows.extend(range(n))
        cols.extend(range(n))
        values.extend([1.0] * n)
        indices = torch.tensor([rows, cols], dtype=torch.long)
        A_hat = torch.sparse_coo_tensor(
            indices,
            torch.tensor(values, dtype=torch.float32),
            (n, n),
        ).coalesce()
        degrees = torch.sparse.sum(A_hat, dim=1).to_dense()
        inv_sqrt = degrees.pow(-0.5)
        inv_sqrt[torch.isinf(inv_sqrt)] = 0.0
        norm_values = (
            A_hat.values()
            * inv_sqrt[A_hat.indices()[0]]
            * inv_sqrt[A_hat.indices()[1]]
        )
        log.info("  Adjacence creuse activee pour %s noeuds", n)
        return torch.sparse_coo_tensor(A_hat.indices(), norm_values, (n, n)).coalesce()

    def nx_to_tensors(
        self, G
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
        """
        Convertit G en (x, A_norm, node_map).

        Returns
        -------
        x        : [N, NODE_FEAT_DIM]    features des nœuds
        A_norm   : [N, N]                adjacence normalisée
        node_map : {nx_id → idx}         correspondance id→index tenseur
        """
        nodes          = sorted(G.nodes())
        self._node_map = {n: i for i, n in enumerate(nodes)}

        x      = self.node_features(G)
        A_norm = self.adjacency_matrix(G)

        log.info(f"  x={tuple(x.shape)}  A_norm={tuple(A_norm.shape)}"
                 f"  ({G.number_of_edges()} arêtes)")
        return x, A_norm, self._node_map

    def neighbor_indices(self, G, nx_id: int) -> torch.Tensor:
        """
        Retourne les indices tenseur des voisins d'un nœud nx_id.
        Utilisé par les agents pour construire l'espace d'actions.
        """
        if self._node_map is None:
            raise RuntimeError("Appeler nx_to_tensors() d'abord.")
        return torch.tensor(
            [self._node_map[nb] for nb in G.neighbors(nx_id)],
            dtype=torch.long,
        )


class GCNLayer(nn.Module):
    """
    Une couche de convolution de graphe :

        H' = A_norm · H · W     (multiplication matricielle)

    où A_norm est la matrice normalisée précomputée par GraphDataConverter.

    Note : l'activation est appliquée dans GCNEncoder, pas ici,
    pour garder la dernière couche sans contrainte.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.W.weight)
        if bias:
            nn.init.zeros_(self.W.bias)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x      : [N, in_dim]
        A_norm : [N, N]  adjacence normalisée (sur le même device que x)

        Returns
        -------
        [N, out_dim]
        """
        propagated = torch.sparse.mm(A_norm, x) if A_norm.is_sparse else A_norm @ x
        return self.W(propagated)


class GCNEncoder(nn.Module):
    """
    Encodeur GCN complet.

    - n_layers couches GCNLayer (2 ou 3)
    - BatchNorm + ReLU + Dropout entre les couches internes
    - Dernière couche : pas d'activation (embeddings réels non bornés)
    - Skip connection : projection(x_input) ajoutée à la sortie finale
      → évite la sur-lissage (over-smoothing) des features

    Input  : (x [N, node_feat_dim],  A_norm [N, N])
    Output : embeddings [N, out_dim]
    """

    def __init__(
        self,
        node_feat_dim : int   = NODE_FEAT_DIM,
        hidden_dim    : int   = DEFAULT_HIDDEN,
        out_dim       : int   = DEFAULT_OUT_DIM,
        n_layers      : int   = DEFAULT_LAYERS,
        dropout       : float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        assert n_layers in (2, 3), "n_layers doit être 2 ou 3"

        self.n_layers  = n_layers
        self.dropout_p = dropout
        self.out_dim   = out_dim

        dims = [node_feat_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]

        self.convs = nn.ModuleList([
            GCNLayer(dims[i], dims[i + 1])
            for i in range(n_layers)
        ])
        # BatchNorm seulement sur les couches internes (pas la dernière)
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(dims[i + 1])
            for i in range(n_layers - 1)
        ])
        # Skip : projette x_input dans l'espace de sortie
        self.skip = (
            nn.Linear(node_feat_dim, out_dim, bias=False)
            if node_feat_dim != out_dim
            else nn.Identity()
        )


    def forward(
        self,
        x      : torch.Tensor,
        A_norm : torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
        x      : [N, node_feat_dim]
        A_norm : [N, N]

        Returns
        -------
        embeddings : [N, out_dim]
        """
        residual = self.skip(x)   # skip depuis l'entrée
        h = x

        for i, conv in enumerate(self.convs):
            h = conv(h, A_norm)

            if i < len(self.bns):               # couches internes
                if h.shape[0] > 1:              # BatchNorm requiert N ≥ 2
                    h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout_p, training=self.training)
            # Dernière couche : ni ReLU ni BN

        return h + residual   # [N, out_dim]


    @torch.no_grad()
    def encode_graph(
        self,
        G,
        converter : Optional[GraphDataConverter] = None,
        device    : str = "cpu",
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Encode un graphe NetworkX complet.

        Returns
        -------
        embeddings : [N, out_dim]       sur `device`
        node_map   : {nx_id → idx}
        """
        if converter is None:
            converter = GraphDataConverter()
        x, A_norm, node_map = converter.nx_to_tensors(G)
        self.eval()
        emb = self(x.to(device), A_norm.to(device))
        log.info(f"  Embeddings : {tuple(emb.shape)}  device={device}")
        return emb, node_map


    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        layers = " → ".join(
            str(c.W.in_features) for c in self.convs
        ) + f" → {self.convs[-1].W.out_features}"
        return (f"GCNEncoder({layers})  "
                f"n_layers={self.n_layers}  "
                f"params={params:,}")


def save_gcn(model: GCNEncoder, path: Path, extra: dict = None) -> None:
    """Sauvegarde le modèle avec sa config pour le rechargement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict"   : model.state_dict(),
        "config"       : {
            "node_feat_dim": model.convs[0].W.in_features,
            "hidden_dim"   : model.convs[0].W.out_features,
            "out_dim"      : model.out_dim,
            "n_layers"     : model.n_layers,
            "dropout"      : model.dropout_p,
        },
        "extra"        : extra or {},
    }, path)
    log.info(f"  ✅ GCNEncoder → {path}")


def load_gcn(path: Path, device: str = "cpu") -> GCNEncoder:
    """Charge un GCNEncoder depuis un checkpoint."""
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = GCNEncoder(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info(f"  ✅ GCNEncoder chargé depuis {path}")
    return model.to(device)


def _node_key(node: Any) -> str:
    return f"{type(node).__name__}:{node!r}"


def graph_fingerprint(G) -> str:
    """Stable identifier for the topology and the node attributes seen by the GCN."""
    attrs = ["type", *CONTINUOUS_FEATS]
    node_payload = []
    for node in sorted(G.nodes()):
        node_payload.append(
            (_node_key(node), {name: G.nodes[node].get(name, 0.0) for name in attrs})
        )
    edge_payload = []
    for u, v, data in G.edges(data=True):
        left, right = sorted((_node_key(u), _node_key(v)))
        edge_payload.append(
            (
                left,
                right,
                data.get("type", ""),
                data.get("weight", 0.0),
                data.get("gcn_weight", 0.0),
                data.get("passable", True),
                data.get("width", 0.0),
            )
        )
    edge_payload.sort()
    raw = json.dumps(
        {"nodes": node_payload, "edges": edge_payload},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def embedding_metadata_path(cache_path: Path) -> Path:
    cache_path = Path(cache_path)
    return cache_path.with_suffix(".meta.json")


def embedding_checksum(embeddings: np.ndarray) -> str:
    array = np.ascontiguousarray(embeddings, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sample_negative_pairs(G, node_map: Dict[Any, int], count: int, seed: int):
    n = len(node_map)
    existing = {
        tuple(sorted((node_map[u], node_map[v])))
        for u, v in G.edges()
    }
    available = n * (n - 1) // 2 - len(existing)
    target = min(count, available)
    if target <= 0:
        return []

    rng = np.random.default_rng(seed)
    pairs = set()
    attempts = 0
    max_attempts = max(1000, target * 50)
    while len(pairs) < target and attempts < max_attempts:
        i, j = sorted(rng.choice(n, size=2, replace=False).tolist())
        pair = (int(i), int(j))
        if pair not in existing:
            pairs.add(pair)
        attempts += 1

    if len(pairs) < target:
        for i in range(n):
            for j in range(i + 1, n):
                pair = (i, j)
                if pair not in existing:
                    pairs.add(pair)
                    if len(pairs) == target:
                        return sorted(pairs)
    return sorted(pairs)


def train_gcn_link_prediction(
    model: GCNEncoder,
    G,
    converter: GraphDataConverter,
    epochs: int = 150,
    lr: float = 1e-2,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> float:
    """Train the encoder once by reconstructing graph links from node embeddings."""
    if epochs <= 0 or G.number_of_edges() == 0:
        model.eval()
        return 0.0

    x, A_norm, node_map = converter.nx_to_tensors(G)
    positive_pairs = [
        (node_map[u], node_map[v])
        for u, v in G.edges()
    ]
    negative_pairs = _sample_negative_pairs(
        G,
        node_map,
        max(1, int(len(positive_pairs) * negative_ratio)),
        seed,
    )
    if not negative_pairs:
        model.eval()
        return 0.0

    pos_idx = torch.tensor(positive_pairs, dtype=torch.long)
    neg_idx = torch.tensor(negative_pairs, dtype=torch.long)
    labels = torch.cat((torch.ones(len(pos_idx)), torch.zeros(len(neg_idx))))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scale = float(model.out_dim) ** 0.5
    final_loss = 0.0

    model.train()
    for _ in range(epochs):
        embeddings = model(x, A_norm)
        pos_logits = (embeddings[pos_idx[:, 0]] * embeddings[pos_idx[:, 1]]).sum(dim=1) / scale
        neg_logits = (embeddings[neg_idx[:, 0]] * embeddings[neg_idx[:, 1]]).sum(dim=1) / scale
        logits = torch.cat((pos_logits, neg_logits))
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        final_loss = float(loss.detach().item())

    model.eval()
    log.info("  GCN link-prediction: epochs=%s loss=%.4f", epochs, final_loss)
    return final_loss


def load_or_create_embeddings(
    G,
    cache_path: Path,
    checkpoint_path: Path,
    emb_dim: int = DEFAULT_OUT_DIM,
    hidden_dim: int = DEFAULT_HIDDEN,
    seed: int = 42,
    train_epochs: int = 150,
    force_rebuild: bool = False,
    shared_checkpoint_path: Optional[Path] = None,
) -> Tuple[Dict[Any, np.ndarray], dict]:
    """Return one validated embedding artifact for training, evaluation and GIFs."""
    cache_path = Path(cache_path)
    checkpoint_path = Path(checkpoint_path)
    metadata_path = embedding_metadata_path(cache_path)
    fingerprint = graph_fingerprint(G)
    nodes = sorted(G.nodes())

    if not force_rebuild and cache_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            data = np.load(cache_path, allow_pickle=False)
            array = data["embeddings"].astype(np.float32, copy=False)
            valid = (
                metadata.get("version") == EMBEDDING_ARTIFACT_VERSION
                and metadata.get("graph_fingerprint") == fingerprint
                and metadata.get("node_keys") == [_node_key(node) for node in nodes]
                and array.shape == (len(nodes), emb_dim)
                and metadata.get("embedding_checksum") == embedding_checksum(array)
            )
            if valid:
                log.info("  Embeddings GCN charges depuis %s", cache_path)
                return {node: array[i].copy() for i, node in enumerate(nodes)}, metadata
            log.warning("  Cache GCN incompatible avec le graphe actuel: regeneration")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            log.warning("  Cache GCN illisible (%s): regeneration", exc)

    if shared_checkpoint_path is not None:
        shared_checkpoint_path = Path(shared_checkpoint_path)
        if not shared_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint GCN partagé introuvable: {shared_checkpoint_path}")
        converter = GraphDataConverter()
        model = load_gcn(shared_checkpoint_path)
        if model.out_dim != emb_dim:
            raise ValueError(
                f"Dimension GCN partagée incompatible: {model.out_dim} != {emb_dim}"
            )
        emb_tensor, node_map = model.encode_graph(G, converter=converter)
        array = np.stack([
            emb_tensor[node_map[node]].cpu().numpy() for node in nodes
        ]).astype(np.float32)
        metadata = {
            "version": EMBEDDING_ARTIFACT_VERSION,
            "graph_fingerprint": fingerprint,
            "node_keys": [_node_key(node) for node in nodes],
            "node_count": len(nodes),
            "embedding_dim": emb_dim,
            "embedding_checksum": embedding_checksum(array),
            "seed": seed,
            "train_epochs": 0,
            "shared_gcn_checkpoint": str(shared_checkpoint_path.resolve()),
            "transfer_mode": "frozen_shared_gcn",
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=array)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return {node: array[i].copy() for i, node in enumerate(nodes)}, metadata

    torch.manual_seed(seed)
    np.random.seed(seed)
    converter = GraphDataConverter()
    model = GCNEncoder(
        node_feat_dim=NODE_FEAT_DIM,
        hidden_dim=hidden_dim,
        out_dim=emb_dim,
        n_layers=DEFAULT_LAYERS,
        dropout=DEFAULT_DROPOUT,
    )
    final_loss = train_gcn_link_prediction(
        model,
        G,
        converter,
        epochs=train_epochs,
        seed=seed,
    )
    emb_tensor, node_map = model.encode_graph(G, converter=converter)
    array = np.stack([emb_tensor[node_map[node]].cpu().numpy() for node in nodes]).astype(np.float32)
    checksum = embedding_checksum(array)
    metadata = {
        "version": EMBEDDING_ARTIFACT_VERSION,
        "graph_fingerprint": fingerprint,
        "node_keys": [_node_key(node) for node in nodes],
        "node_count": len(nodes),
        "embedding_dim": emb_dim,
        "embedding_checksum": checksum,
        "seed": seed,
        "train_epochs": train_epochs,
        "final_link_prediction_loss": final_loss,
        "checkpoint_path": str(checkpoint_path),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=array)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    save_gcn(model, checkpoint_path, extra=metadata)
    log.info("  Embeddings GCN sauvegardes: %s", cache_path)
    return {node: array[i].copy() for i, node in enumerate(nodes)}, metadata


def _demo_synthetic():
    """Teste le pipeline complet avec un graphe synthétique."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    log.info("🏗️  Génération d'un bâtiment synthétique...")
    try:
        from data.ifc_parser import SyntheticBuildingGenerator, save_nodes_csv, save_edges_csv, deduplicate_edges
        from data.graph_builder import build_pipeline
        import tempfile, os

        gen         = SyntheticBuildingGenerator(n_floors=3, rooms_per_floor=8, seed=42)
        spaces, edges = gen.generate()
        edges       = deduplicate_edges(edges)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_nodes_csv(spaces, tmp / "nodes.csv")
            save_edges_csv(edges,  tmp / "edges.csv")
            G = build_pipeline(
                tmp / "nodes.csv", tmp / "edges.csv",
                tmp / "graph.gpickle", visualize=False
            )
    except ImportError:
        log.warning("data/ non trouvé — création d'un graphe NetworkX minimal")
        import networkx as nx
        G = nx.grid_2d_graph(5, 5)
        G = nx.convert_node_labels_to_integers(G)
        for n in G.nodes():
            G.nodes[n].update({
                "type": "room", "x_norm": float(n % 5) / 4,
                "y_norm": float(n // 5) / 4, "z_norm": 0.0,
                "area_norm": 0.5, "risk": 0.05,
                "betweenness": 0.1, "closeness": 0.5, "degree_c": 0.3,
            })

    log.info(f"  Graphe : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    conv  = GraphDataConverter()
    x, A_norm, node_map = conv.nx_to_tensors(G)
    log.info(f"  x={tuple(x.shape)}  A_norm={tuple(A_norm.shape)}")
    assert not torch.isnan(x).any(),      "NaN dans x !"
    assert not torch.isnan(A_norm).any(), "NaN dans A_norm !"

    model = GCNEncoder(node_feat_dim=NODE_FEAT_DIM, hidden_dim=64,
                       out_dim=64, n_layers=2, dropout=0.1)
    log.info(f"  {model.summary()}")

    model.train()
    emb_train = model(x, A_norm)
    assert emb_train.shape == (G.number_of_nodes(), 64), f"Shape inattendu : {emb_train.shape}"
    assert not torch.isnan(emb_train).any(), "NaN dans les embeddings !"
    log.info(f"  Embeddings (train) : {tuple(emb_train.shape)}  ✅")

    emb, nm = model.encode_graph(G)
    log.info(f"  Embeddings (eval)  : {tuple(emb.shape)}  ✅")

    # Vérification skip connection : embeddings ≠ 0
    assert emb.abs().mean() > 1e-4, "Embeddings tous nuls — problème de skip !"

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)
    save_gcn(model, tmp_path, extra={"test": True})
    model2 = load_gcn(tmp_path)
    emb2, _ = model2.encode_graph(G)
    assert torch.allclose(emb, emb2, atol=1e-5), "Embeddings avant/après chargement différents !"
    os.unlink(tmp_path)
    log.info("  Sauvegarde/chargement ✅")

    log.info("=" * 50)
    log.info("✅ gcn_model.py — tous les tests passent")
    log.info("=" * 50)
    return model, emb, node_map


def main():
    parser = argparse.ArgumentParser(description="GCN Encoder")
    parser.add_argument("--graph",     type=str, default=str(GRAPH_PKL))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--hidden",    type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--out-dim",   type=int, default=DEFAULT_OUT_DIM)
    parser.add_argument("--layers",    type=int, default=DEFAULT_LAYERS, choices=[2, 3])
    parser.add_argument("--save",      type=str, default=None)
    args = parser.parse_args()

    if args.synthetic:
        model, emb, node_map = _demo_synthetic()
        return

    graph_path = Path(args.graph)
    if not graph_path.exists():
        log.error(f"graph.gpickle introuvable : {graph_path}")
        log.error("Lancez d'abord : python data/graph_builder.py")
        log.error("Ou utilisez --synthetic pour tester sans graphe")
        return

    log.info(f"📂 Chargement : {graph_path}")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    log.info(f"  {G.number_of_nodes()} nœuds · {G.number_of_edges()} arêtes")

    conv  = GraphDataConverter()
    x, A_norm, node_map = conv.nx_to_tensors(G)

    model = GCNEncoder(
        node_feat_dim = NODE_FEAT_DIM,
        hidden_dim    = args.hidden,
        out_dim       = args.out_dim,
        n_layers      = args.layers,
    )
    log.info(f"  {model.summary()}")

    emb, _ = model.encode_graph(G)
    log.info(f"  Embeddings : {tuple(emb.shape)}")
    log.info(f"  Norme moy  : {emb.norm(dim=1).mean():.4f}")

    if args.save:
        save_gcn(model, Path(args.save))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    main()
