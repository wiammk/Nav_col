"""Graph navigation environments with action masks, occupancy and conflict resolution."""

import pickle
import logging
import argparse
import sys
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import CONFIG
except ImportError:
    CONFIG = {}

log = logging.getLogger(__name__)

ENV_DIR       = Path(__file__).parent
GRAPH_PKL     = ENV_DIR.parent / "data" / "processed" / "graph.gpickle"


class GraphEnv(gym.Env):
    """
    Environnement Gymnasium pour un seul robot navigant sur un graphe.

    Paramètres
    ----------
    graph       : nx.Graph  — le graphe du bâtiment (nodes ont attribut "risk")
    embeddings  : dict[node_id → np.ndarray]  — embeddings GCN [emb_dim]
    max_steps   : nombre max de steps par épisode avant troncature
    reward_cfg  : dict de reward, voir REWARD_DEFAULTS ci-dessous
    seed        : graine aléatoire
    robot_id    : identifiant du robot (pour les logs FL)
    """

    metadata = {"render_modes": ["human"]}

    REWARD_DEFAULTS = {
        "goal_reward"    :  10.0,   # arrivée à destination
        "step_penalty"   :  -0.1,   # pénalité par step
        "shaping_factor" :   1.0,   # bonus si on réduit la distance graphe
        "shaping_gamma"  :   0.99,
        "risk_factor"    :   0.5,   # pénalité * risque de la pièce visitée
        "stuck_penalty"  :  -2.0,   # pénalité si bloqué trop longtemps
        "stuck_limit"    :  10,     # steps consécutifs sans mouvement → stuck
        "min_door_width" :   0.6,
    }

    def __init__(
        self,
        graph      : nx.Graph,
        embeddings : Dict[Any, np.ndarray],
        max_steps  : int = 200,
        reward_cfg : Optional[Dict] = None,
        seed       : Optional[int]  = None,
        robot_id   : int = 0,
    ):
        super().__init__()

        self.graph    = graph
        self.emb = {
            node: np.asarray(value, dtype=np.float32).copy()
            for node, value in embeddings.items()
        }
        self.robot_id = robot_id

        # Indexation déterministe des nœuds (critique pour FL : même index partout)
        self.nodes      = sorted(graph.nodes())
        self.node2idx   = {n: i for i, n in enumerate(self.nodes)}
        self.idx2node   = {i: n for i, n in enumerate(self.nodes)}
        self.n_nodes    = len(self.nodes)

        if self.n_nodes == 0:
            raise ValueError("Le graphe est vide.")

        # Voisins triés de façon déterministe
        self.neighbors_map = {
            n: sorted(list(graph.neighbors(n)), key=lambda x: self.node2idx[x])
            for n in self.nodes
        }

        sample_emb      = next(iter(embeddings.values()))
        self.emb_dim    = int(sample_emb.shape[0])
        self.max_degree = max(len(v) for v in self.neighbors_map.values())
        if self.max_degree == 0:
            self.max_degree = 1

        self.observation_space = spaces.Dict({
            "state"    : spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(2 * self.emb_dim,), dtype=np.float32,
            ),
            "neighbors": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_degree, self.emb_dim), dtype=np.float32,
            ),
            "mask"     : spaces.MultiBinary(self.max_degree),
            "occupancy": spaces.Box(
                low=0.0, high=1.0, shape=(self.max_degree,), dtype=np.float32
            ),
        })
        self.action_space = spaces.Discrete(self.max_degree)

        self.max_steps  = max_steps
        cfg             = dict(self.REWARD_DEFAULTS)
        if reward_cfg:
            cfg.update(reward_cfg)
        self.rcfg       = cfg
        self.closed_edges = set()
        self.blocked_nodes = set()
        self._sp_dist_cache = {}

        try:
            available = graph.copy()
            available.remove_edges_from([
                (u, v) for u, v, data in available.edges(data=True)
                if not bool(data.get("passable", True))
                or float(data.get("width", 1.0)) < float(self.rcfg["min_door_width"])
            ])
            self._sp = dict(nx.all_pairs_dijkstra_path_length(available, weight="weight"))
        except Exception:
            self._sp = None

        self.np_random = np.random.default_rng(seed)
        self._state_init()


    def _state_init(self):
        self.current_node : Any  = None
        self.target_node  : Any  = None
        self._prev_dist   : float = 0.0
        self._initial_dist: float = 1.0
        self._step_count  : int  = 0
        self._stuck_count : int  = 0
        self._terminated  : bool = False
        self._truncated   : bool = False
        self.path_taken   : List = []
        self.cumulative_risk: float = 0.0
        self.path_length: float = 0.0

    @staticmethod
    def _edge_key(a, b):
        return frozenset((a, b))

    def edge_is_available(self, a, b) -> bool:
        data = self.graph.get_edge_data(a, b, default={})
        return (
            b not in self.blocked_nodes
            and self._edge_key(a, b) not in self.closed_edges
            and bool(data.get("passable", True))
            and float(data.get("width", 1.0)) >= float(self.rcfg["min_door_width"])
        )

    def set_disruptions(self, blocked_nodes=None, closed_edges=None):
        next_blocked = set(blocked_nodes or ())
        next_closed = {self._edge_key(a, b) for a, b in (closed_edges or ())}
        if next_blocked != self.blocked_nodes or next_closed != self.closed_edges:
            self._sp_dist_cache.clear()
        self.blocked_nodes = next_blocked
        self.closed_edges = next_closed

    def _sp_dist(self, a, b) -> float:
        if a == b:
            return 0
        if self._sp is not None and not self.blocked_nodes and not self.closed_edges:
            return float(self._sp.get(a, {}).get(b, 10_000.0))
        cache_key = (a, b)
        if cache_key in self._sp_dist_cache:
            return self._sp_dist_cache[cache_key]
        try:
            available = self.graph.copy()
            available.remove_nodes_from(
                node for node in self.blocked_nodes
                if node in available and node not in {a, b}
            )
            available.remove_edges_from(
                tuple(edge) for edge in self.closed_edges
                if len(edge) == 2 and available.has_edge(*tuple(edge))
            )
            available.remove_edges_from([
                (u, v) for u, v, data in available.edges(data=True)
                if not bool(data.get("passable", True))
                or float(data.get("width", 1.0)) < float(self.rcfg["min_door_width"])
            ])
            distance = float(nx.shortest_path_length(available, a, b, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            distance = 10_000.0
        self._sp_dist_cache[cache_key] = distance
        return distance

    def _build_obs(self) -> Dict:
        cur  = self.current_node
        tgt  = self.target_node
        nbs  = self.neighbors_map.get(cur, [])

        state = np.concatenate([
            self.emb[cur].astype(np.float32),
            self.emb[tgt].astype(np.float32),
        ])                                          # [2 * emb_dim]

        neigh_emb = np.zeros((self.max_degree, self.emb_dim), dtype=np.float32)
        mask      = np.zeros(self.max_degree, dtype=np.int8)
        for i, nb in enumerate(nbs):
            if i >= self.max_degree:
                break
            neigh_emb[i] = self.emb[nb].astype(np.float32)
            mask[i] = int(self.edge_is_available(cur, nb))

        return {
            "state": state,
            "neighbors": neigh_emb,
            "mask": mask,
            "occupancy": np.zeros(self.max_degree, dtype=np.float32),
        }


    def reset(
        self,
        *,
        seed   : Optional[int] = None,
        start  : Optional[Any] = None,
        target : Optional[Any] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Réinitialise l'épisode.

        start / target : nœuds NetworkX (optionnels).
        Si non fournis, tirés aléatoirement.

        Retourne (obs, info).
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._state_init()
        options = options or {}
        self.set_disruptions(
            blocked_nodes=options.get("blocked_nodes"),
            closed_edges=options.get("closed_edges"),
        )

        if start is not None and start in self.node2idx:
            self.current_node = start
        else:
            self.current_node = self.np_random.choice(self.nodes)

        if target is not None and target in self.node2idx and target != self.current_node:
            self.target_node = target
        else:
            candidates = [n for n in self.nodes if n != self.current_node]
            self.target_node = self.np_random.choice(candidates)

        self._prev_dist = self._sp_dist(self.current_node, self.target_node)
        self._initial_dist = max(self._prev_dist, 1.0)
        self.path_taken = [self.current_node]

        obs  = self._build_obs()
        info = {"robot_id": self.robot_id, "start": self.current_node,
                "target": self.target_node, "sp_dist": self._prev_dist}
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Exécute une action.

        action : int  — indice dans neighbors_map[current_node] (0..max_degree-1)

        Retourne (obs, reward, terminated, truncated, info).
        """
        assert not (self._terminated or self._truncated), \
            "reset() requis avant step() après la fin d'un épisode."

        self._step_count += 1
        nbs    = self.neighbors_map.get(self.current_node, [])
        valid = (
            0 <= action < len(nbs)
            and self.edge_is_available(self.current_node, nbs[action])
        )

        next_node = nbs[action] if valid else self.current_node
        moved     = next_node != self.current_node

        reward = self.rcfg["step_penalty"]

        # Shaping : distance graphe → récompense si on se rapproche
        new_dist = self._sp_dist(next_node, self.target_node)
        shaping  = (
            (
                self._prev_dist
                - float(self.rcfg["shaping_gamma"]) * new_dist
            ) / self._initial_dist
        ) * self.rcfg["shaping_factor"]
        reward  += shaping

        if moved:
            risk = float(self.graph.nodes[next_node].get("risk", 0.0))
            reward -= self.rcfg["risk_factor"] * risk
            self.cumulative_risk += risk
            self.path_length += float(
                self.graph.get_edge_data(self.current_node, next_node, default={}).get("weight", 1.0)
            )

        if moved:
            self._stuck_count = 0
        else:
            self._stuck_count += 1
            if self._stuck_count >= int(self.rcfg["stuck_limit"]):
                reward         += self.rcfg["stuck_penalty"]
                self._terminated = True

        terminated = self._terminated
        if next_node == self.target_node:
            reward    += self.rcfg["goal_reward"]
            terminated = True

        self.current_node = next_node
        self._prev_dist   = new_dist
        self.path_taken.append(next_node)

        truncated = self._step_count >= self.max_steps

        self._terminated = terminated
        self._truncated  = truncated

        obs  = self._build_obs()
        info = {
            "robot_id"    : self.robot_id,
            "current_node": self.current_node,
            "target_node" : self.target_node,
            "step"        : self._step_count,
            "dist_to_goal": new_dist,
            "valid_action": valid,
            "moved"       : moved,
            "cumulative_risk": self.cumulative_risk,
            "path_length": self.path_length,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "human"):
        dist = self._sp_dist(self.current_node, self.target_node)
        print(f"[Robot {self.robot_id}] step={self._step_count:3d} | "
              f"pos={self.current_node} → goal={self.target_node} | dist={dist}")

    def close(self):
        pass


    @property
    def n_actions(self) -> int:
        """Nombre max d'actions (= max_degree)."""
        return self.max_degree

    def valid_actions(self, node=None) -> List[int]:
        """Indices des actions valides depuis le nœud courant (ou `node`)."""
        n   = node if node is not None else self.current_node
        nbs = self.neighbors_map.get(n, [])
        return [
            index for index, neighbor in enumerate(nbs)
            if self.edge_is_available(n, neighbor)
        ]


class MultiRobotEnv:
    """
    Orchestre N instances GraphEnv indépendantes.

    Pourquoi N instances séparées et pas un seul env multi-agent ?
    → Le Federated Learning demande que chaque robot ait son propre
      modèle local entraîné sur SES propres expériences. N envs séparés
      garantit que les observations et histoires ne se mélangent pas.

    Utilisation :
        env = MultiRobotEnv(graph, embeddings, n_robots=3)
        obs_list = env.reset()                   # list[dict], len=3
        obs_list, rews, terms, infos = env.step([0, 2, 1])
    """

    def __init__(
        self,
        graph      : nx.Graph,
        embeddings : Dict[Any, np.ndarray],
        n_robots   : int = 3,
        max_steps  : int = 200,
        reward_cfg : Optional[Dict] = None,
        seed       : Optional[int]  = None,
        avoid_collisions: bool = True,
        collision_penalty: float = -1.0,
    ):
        self.n_robots = n_robots
        self.avoid_collisions = avoid_collisions
        self.collision_penalty = float(collision_penalty)
        seeds = [None] * n_robots if seed is None else [seed + i for i in range(n_robots)]

        self.envs : List[GraphEnv] = [
            GraphEnv(
                graph      = graph,
                embeddings = embeddings,
                max_steps  = max_steps,
                reward_cfg = reward_cfg,
                seed       = seeds[i],
                robot_id   = i,
            )
            for i in range(n_robots)
        ]

        self.emb_dim        = self.envs[0].emb_dim
        self.max_degree     = self.envs[0].max_degree
        self.n_actions      = self.envs[0].n_actions
        self.nodes          = self.envs[0].nodes
        self.neighbors_map  = self.envs[0].neighbors_map

    def _mask_occupied_neighbors(self, obs_list: List[Dict]) -> List[Dict]:
        if not self.avoid_collisions:
            return obs_list
        occupancy_counts = {}
        for robot_env in self.envs:
            occupancy_counts[robot_env.current_node] = (
                occupancy_counts.get(robot_env.current_node, 0) + 1
            )
        for env, obs in zip(self.envs, obs_list):
            blocked = 0
            occupancy = np.zeros(env.max_degree, dtype=np.float32)
            for action, neighbor in enumerate(env.neighbors_map[env.current_node]):
                capacity = int(env.graph.nodes[neighbor].get("capacity", 1))
                load = occupancy_counts.get(neighbor, 0)
                occupancy[action] = min(1.0, load / max(1, capacity))
                if load >= capacity and neighbor != env.current_node:
                    obs["mask"][action] = 0
                    blocked += 1
            obs["occupancy"] = occupancy
            if blocked:
                obs["occupied_neighbors"] = blocked
        return obs_list

    def _sample_unique_start(self, env: GraphEnv, occupied: set):
        candidates = [node for node in env.nodes if node not in occupied]
        if not candidates:
            raise RuntimeError("Plus assez de noeuds pour donner un depart unique a chaque robot.")
        return env.np_random.choice(candidates)

    def _sample_unique_target(self, env: GraphEnv, start, reserved: set):
        candidates = [node for node in env.nodes if node != start and node not in reserved]
        if not candidates:
            raise RuntimeError("Plus assez de noeuds pour donner une cible unique a chaque robot.")
        return env.np_random.choice(candidates)

    def reset(
        self,
        starts  : Optional[List] = None,
        targets : Optional[List] = None,
        options : Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Réinitialise tous les robots.

        starts / targets : listes de nœuds (optionnelles, longueur = n_robots).
        Retourne list[obs], len = n_robots.
        """
        obs_list = []
        occupied_starts = set()
        reserved_targets = set()
        for i, env in enumerate(self.envs):
            start = starts[i] if starts and i < len(starts) else None
            if self.avoid_collisions and (start is None or start in occupied_starts):
                start = self._sample_unique_start(env, occupied_starts)
            if self.avoid_collisions:
                occupied_starts.add(start)

            target = targets[i] if targets and i < len(targets) else None
            if self.avoid_collisions and (target is None or target == start or target in reserved_targets):
                target = self._sample_unique_target(env, start, reserved_targets)
            if self.avoid_collisions:
                reserved_targets.add(target)

            robot_options = options[i] if options and i < len(options) else None
            obs, _ = env.reset(start=start, target=target, options=robot_options)
            obs_list.append(obs)
        return self._mask_occupied_neighbors(obs_list)

    def step(
        self,
        actions: List[int],
    ) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
        """
    Avance chaque robot d'un step.

    actions : list[int], len = n_robots.

    Retourne :
        obs_list
        rewards
        dones
        infos
    """
        if len(actions) != self.n_robots:
            raise ValueError(
                f"actions doit avoir {self.n_robots} éléments, reçu {len(actions)}"
            )

        resolved_actions = list(actions)
        collision_reasons: Dict[int, str] = {}
        if self.avoid_collisions:
            current = [env.current_node for env in self.envs]
            proposals = list(current)
            active = []
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                if env._terminated or env._truncated:
                    continue
                neighbors = env.neighbors_map[env.current_node]
                if 0 <= action < len(neighbors):
                    proposals[i] = neighbors[action]
                active.append(i)

            for i in active:
                if proposals[i] == current[i]:
                    continue
                same_destination = [
                    j for j in active
                    if proposals[j] == proposals[i] and proposals[j] != current[j]
                ]
                capacity = int(self.envs[i].graph.nodes[proposals[i]].get("capacity", 1))
                if len(same_destination) > capacity:
                    collision_reasons[i] = "same_destination"
                elif any(
                    proposals[i] == current[j] and proposals[j] == current[j]
                    for j in range(self.n_robots) if j != i
                ):
                    collision_reasons[i] = "occupied_destination"

            for i in active:
                for j in active:
                    if i < j and proposals[i] == current[j] and proposals[j] == current[i]:
                        collision_reasons[i] = "edge_swap"
                        collision_reasons[j] = "edge_swap"

            for robot_id in collision_reasons:
                resolved_actions[robot_id] = -1

        obs_list = []
        rewards = []
        dones = []
        infos = []
        for robot_id, (env, action) in enumerate(zip(self.envs, resolved_actions)):
            if env._terminated or env._truncated:
                obs_list.append(env._build_obs())
                rewards.append(0.0)
                dones.append(True)
                infos.append({
                    "robot_id": env.robot_id,
                    "already_done": True,
                    "current_node": env.current_node,
                    "target_node": env.target_node,
                    "step": env._step_count,
                    "dist_to_goal": env._prev_dist,
                    "valid_action": False,
                    "moved": False,
                    
                })

                continue

            obs, reward, terminated, truncated, info = env.step(action)
            if robot_id in collision_reasons:
                reward += self.collision_penalty
                info["collision"] = True
                info["collision_reason"] = collision_reasons[robot_id]
            info["terminated"] = bool(terminated)
            info["truncated"] = bool(truncated)
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)

        return self._mask_occupied_neighbors(obs_list), rewards, dones, infos

    def reset_done(self) -> None:
        """
        Réinitialise uniquement les robots dont l'épisode est terminé.
        Appelé entre les épisodes pour les robots plus rapides.
        """
        for env in self.envs:
            if env._terminated or env._truncated:
                env.reset()

    def render(self):
        for env in self.envs:
            env.render()

    def close(self):
        for env in self.envs:
            env.close()


    def get_env(self, robot_id: int) -> GraphEnv:
        return self.envs[robot_id]


def make_env(
    graph_path : str,
    n_robots   : int = 1,
    max_steps  : int = 200,
    reward_cfg : Optional[Dict] = None,
    seed       : Optional[int]  = None,
    emb_dim    : int = 64,
    hidden_dim : int = 64,
    embedding_cache_path: Optional[str] = None,
    gcn_checkpoint_path: Optional[str] = None,
    shared_gcn_checkpoint_path: Optional[str] = None,
    gcn_train_epochs: Optional[int] = None,
    force_rebuild_embeddings: bool = False,
):
    """
    Charge le graphe + GCN + crée l'env en une seule fonction.

    Retourne :
        env  : GraphEnv (n_robots=1) ou MultiRobotEnv (n_robots>1)
        G    : nx.Graph
        emb_dict : dict[node → np.ndarray]
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    log.info(f"  Graphe chargé : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    try:
        import torch
        from models.gcn_model import GCNEncoder, GraphDataConverter, NODE_FEAT_DIM

        gcn_cfg = CONFIG.get("gcn", {})
        graph_file = Path(graph_path)
        cache_path = (
            Path(embedding_cache_path)
            if embedding_cache_path is not None
            else graph_file.with_name("gcn_embeddings.npz")
        )
        checkpoint_path = (
            Path(gcn_checkpoint_path)
            if gcn_checkpoint_path is not None
            else graph_file.with_name("gcn_encoder.pt")
        )
        from models.gcn_model import load_or_create_embeddings

        emb_dict, embedding_metadata = load_or_create_embeddings(
            G,
            cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            seed=int(gcn_cfg.get("seed", seed if seed is not None else 42)),
            train_epochs=(
                int(gcn_train_epochs)
                if gcn_train_epochs is not None
                else int(gcn_cfg.get("train_epochs", 150))
            ),
            force_rebuild=force_rebuild_embeddings,
            shared_checkpoint_path=(
                Path(shared_gcn_checkpoint_path)
                if shared_gcn_checkpoint_path is not None else None
            ),
        )
        if n_robots == 1:
            env = GraphEnv(G, emb_dict, max_steps=max_steps, reward_cfg=reward_cfg, seed=seed)
        else:
            env = MultiRobotEnv(
                G,
                emb_dict,
                n_robots=n_robots,
                max_steps=max_steps,
                reward_cfg=reward_cfg,
                seed=seed,
            )
        env.embedding_metadata = embedding_metadata
        env.graph_fingerprint = embedding_metadata["graph_fingerprint"]
        return env, G, emb_dict

        conv = GraphDataConverter()
        x, A_norm, node_map = conv.nx_to_tensors(G)

        gcn = GCNEncoder(
            node_feat_dim = NODE_FEAT_DIM,
            hidden_dim    = hidden_dim,
            out_dim       = emb_dim,
            n_layers      = 2,
        )
        gcn.eval()
        with torch.no_grad():
            emb_tensor = gcn(x, A_norm)   # [N, emb_dim]

        emb_dict = {
            node: emb_tensor[idx].numpy()
            for node, idx in node_map.items()
        }
        log.info(f"  Embeddings GCN : {emb_tensor.shape}  emb_dim={emb_dim}")

    except ImportError:
        log.warning("  torch/models non disponibles → embeddings aléatoires")
        nodes    = sorted(G.nodes())
        rng      = np.random.default_rng(0)
        emb_dict = {n: rng.standard_normal(emb_dim).astype(np.float32) for n in nodes}

    if n_robots == 1:
        env = GraphEnv(G, emb_dict, max_steps=max_steps, reward_cfg=reward_cfg, seed=seed)
    else:
        env = MultiRobotEnv(G, emb_dict, n_robots=n_robots,
                            max_steps=max_steps, reward_cfg=reward_cfg, seed=seed)

    # Q-learning can still run without PyTorch; retain a graph identity so its
    # saved tables cannot be evaluated on a different topology by accident.
    if not hasattr(env, "graph_fingerprint"):
        import hashlib
        graph_bytes = Path(graph_path).read_bytes()
        env.graph_fingerprint = hashlib.sha256(graph_bytes).hexdigest()
        env.embedding_metadata = {
            "backend": "deterministic_numpy_fallback",
            "graph_fingerprint": env.graph_fingerprint,
        }

    return env, G, emb_dict


def _test(graph_path: str, n_robots: int = 3, n_episodes: int = 3):
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
        datefmt= "%H:%M:%S",
    )

    log.info(f"🧪 Test GraphEnv / MultiRobotEnv — {n_robots} robots")

    env, G, emb_dict = make_env(graph_path, n_robots=n_robots, seed=42)

    log.info(f"  emb_dim={env.emb_dim}  max_degree={env.max_degree}  n_actions={env.n_actions}")
    log.info(f"  observation_space={env.envs[0].observation_space if n_robots > 1 else env.observation_space}")

    sep = "=" * 55

    for ep in range(n_episodes):
        obs_list = env.reset() if n_robots > 1 else [env.reset()[0]]
        log.info(f"{sep}")
        log.info(f"  Épisode {ep+1} — positions initiales :")

        envs_list = env.envs if n_robots > 1 else [env]
        for i, (e, obs) in enumerate(zip(envs_list, obs_list)):
            log.info(f"    Robot {i} : pos={e.current_node} → cible={e.target_node}  dist={e._prev_dist}")
            assert obs["state"].shape    == (2 * env.emb_dim,)
            assert obs["neighbors"].shape == (env.max_degree, env.emb_dim)
            assert obs["mask"].shape      == (env.max_degree,)
            assert obs["mask"].sum() > 0, "Aucun voisin accessible !"

        total_rewards = [0.0] * n_robots
        step_count    = 0

        for _ in range(50):
            # Action aléatoire parmi les actions valides
            actions = []
            for e in envs_list:
                if e._terminated or e._truncated:
                    actions.append(0)  
                    continue

                
                valid = e.valid_actions()
                actions.append(np.random.choice(valid) if valid else 0)

            if n_robots > 1:
                obs_list, rews, dones, infos = env.step(actions)
            else:
                obs_list_r, rew, term, trunc, info = env.step(actions[0])
                obs_list  = [obs_list_r]
                rews      = [rew]
                dones     = [term or trunc]
                infos     = [info]

            for i, r in enumerate(rews):
                total_rewards[i] += r
            step_count += 1

            if all(dones):
                break

        for i in range(n_robots):
            log.info(f"    Robot {i} : steps={step_count}  total_reward={total_rewards[i]:.2f}")

    log.info(sep)
    log.info("✅ GraphEnv — tous les tests passent")
    log.info("   Interface correcte pour Q-Learning + DQN + PPO + FedAvg")
    log.info(sep)


def main():
    parser = argparse.ArgumentParser(description="GraphEnv — test")
    parser.add_argument("--graph",    type=str, default=CONFIG.get("graph_path", str(GRAPH_PKL)))
    parser.add_argument("--robots",   type=int, default=CONFIG.get("n_robots", 3))
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if not Path(args.graph).exists():
        print(f"graph.gpickle non trouvé : {args.graph}")
        print("Lancez d'abord : python data/ifc_parser.py --synthetic --floors 3 --rooms 8")
        print("                 python data/graph_builder.py")
        sys.exit(1)

    _test(args.graph, args.robots, args.episodes)


if __name__ == "__main__":
    main()
