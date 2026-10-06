"""
federated/server.py
===================
Serveur FedAvg — commun pour Q-Learning, DQN et PPO.

Pourquoi un seul FedAvg pour les 3 modèles ?
    La formule est identique : global = Σ(n_i × w_i) / Σ(n_i)
    Seul le format change :
      - Q-Learning : numpy array   shape (n_nodes, n_nodes, max_degree)
      - DQN / PPO  : dict PyTorch  {layer_name → tensor}
    Le serveur détecte automatiquement le format et agrège.

Flux FedAvg (1 round) :
    1. broadcast   : serveur envoie le modèle global à chaque robot
    2. local_train : chaque robot s'entraîne N épisodes localement
    3. collect     : chaque robot envoie ses poids + n_samples au serveur
    4. aggregate   : serveur calcule la moyenne pondérée
    5. → round suivant

Usage :
    server = FedAvgServer()
    server.run(clients, n_rounds=10, episodes_per_round=50)
"""

import copy
import logging
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from config.run_layout import latest_run_layout

if TYPE_CHECKING:
    from federated.client import FedClient

log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — AGRÉGATION (cœur mathématique)
# ═════════════════════════════════════════════════════════════════════════════

def fedavg(
    weights_list : List[Any],
    n_samples    : List[int],
) -> Any:
    """
    Moyenne pondérée FedAvg.

    Formule : global = Σ(n_i × w_i) / Σ(n_i)

    Accepte automatiquement :
      - list[np.ndarray]  → Q-tables
      - list[dict]        → state_dict PyTorch (DQN / PPO)

    Paramètres
    ----------
    weights_list : poids locaux de chaque robot
    n_samples    : nombre d'épisodes joués par chaque robot ce round

    Retourne
    --------
    global_weights : même type que weights_list[0]
    """
    assert len(weights_list) == len(n_samples) > 0, "Listes de taille différente ou vides"

    total = sum(n_samples)
    if total == 0:
        log.warning("  n_samples tous nuls → agrégation uniforme")
        n_samples = [1] * len(weights_list)
        total     = len(weights_list)

    # ── Cas Q-Learning : numpy arrays ─────────────────────────────────────────
    if isinstance(weights_list[0], np.ndarray):
        global_w = np.zeros_like(weights_list[0], dtype=np.float64)
        for w, n in zip(weights_list, n_samples):
            global_w += (n / total) * w.astype(np.float64)
        return global_w.astype(weights_list[0].dtype)

    # ── Cas DQN / PPO : dicts de tenseurs PyTorch ─────────────────────────────
    if isinstance(weights_list[0], dict):
        try:
            import torch
        except ImportError:
            raise ImportError("torch requis pour agréger des state_dicts.")

        global_w = {}
        for key in weights_list[0].keys():
            # Empiler : [n_clients, *param_shape]
            stacked = torch.stack([w[key].float() for w in weights_list], dim=0)
            factors = torch.tensor(
                [n / total for n in n_samples], dtype=torch.float32
            ).view(-1, *([1] * (stacked.dim() - 1)))
            global_w[key] = (stacked * factors).sum(dim=0).to(weights_list[0][key].dtype)
        return global_w

    raise TypeError(f"Format de poids non supporté : {type(weights_list[0])}")


def visitation_weighted_q_aggregate(
    q_tables: List[np.ndarray],
    visit_counts: List[np.ndarray],
    previous_global: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Aggregate each Q(s,a) only from clients that actually visited it."""
    if len(q_tables) != len(visit_counts) or not q_tables:
        raise ValueError("Une table Q et un compteur de visites sont requis par client.")
    numerator = np.zeros_like(q_tables[0], dtype=np.float64)
    denominator = np.zeros_like(q_tables[0], dtype=np.float64)
    for table, counts in zip(q_tables, visit_counts):
        if table.shape != q_tables[0].shape or counts.shape != q_tables[0].shape:
            raise ValueError("Les Q-tables et compteurs doivent avoir la même forme.")
        non_negative = np.maximum(counts, 0).astype(np.float64)
        numerator += table.astype(np.float64) * non_negative
        denominator += non_negative
    fallback = (
        np.zeros_like(q_tables[0], dtype=np.float64)
        if previous_global is None else previous_global.astype(np.float64)
    )
    result = np.divide(
        numerator,
        denominator,
        out=fallback.copy(),
        where=denominator > 0,
    )
    return result.astype(q_tables[0].dtype)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SERVEUR FEDAVG
# ═════════════════════════════════════════════════════════════════════════════

class FedAvgServer:
    """
    Serveur central FedAvg.

    Orchestre les rounds d'entraînement fédéré :
      - reçoit les poids locaux des clients
      - agrège via fedavg()
      - redistribue le modèle global

    Utilisé de la même façon pour Q-Learning, DQN et PPO.
    """

    def __init__(
        self,
        save_dir: Optional[Path] = None,
        q_aggregation: str = "standard",
    ):
        if q_aggregation not in {"standard", "visitation_weighted"}:
            raise ValueError(f"Aggregation Q inconnue: {q_aggregation}")
        self.save_dir       = Path(save_dir) if save_dir else latest_run_layout().federated_results
        self.q_aggregation  = q_aggregation
        self.global_weights : Optional[Any]  = None
        self.round_history  : List[dict]     = []
        self._round         : int            = 0

    # ── Interface principale ──────────────────────────────────────────────────

    def run(
        self,
        clients          : List["FedClient"],
        n_rounds         : int = 10,
        episodes_per_round: int = 50,
        save_every       : int = 5,
        evaluation_scenarios=None,
        multi_env=None,
    ) -> List[dict]:
        """
        Lance N rounds de Federated Learning.

        Paramètres
        ----------
        clients            : liste des FedClient (un par robot)
        n_rounds           : nombre de rounds FL
        episodes_per_round : épisodes d'entraînement local par round
        save_every         : sauvegarder le modèle global tous les X rounds

        Retourne
        --------
        round_history : liste de métriques par round
        """
        log.info("=" * 55)
        log.info(f"🔗 FEDERATED LEARNING — {len(clients)} robots, {n_rounds} rounds")
        log.info(f"   Algorithme : {clients[0].algo_name}  |  FedAvg")
        log.info("=" * 55)

        # Initialiser le modèle global depuis le premier client
        self.global_weights = copy.deepcopy(clients[0].get_weights())

        for r in range(1, n_rounds + 1):
            self._round = r
            metrics = self._run_round(
                clients,
                episodes_per_round,
                evaluation_scenarios=evaluation_scenarios,
                multi_env=multi_env,
            )
            self.round_history.append(metrics)

            log.info(
                f"  Round {r:3d}/{n_rounds} | "
                f"global_reward={metrics['global_mean_reward']:+.2f} | "
                f"global_succès={metrics['global_success_rate']:.0%} | "
                f"clients={len(clients)}"
            )

            if r % save_every == 0:
                self._save_global(r)

        self._save_global(n_rounds, final=True)
        log.info("✅ Entraînement fédéré terminé")
        return self.round_history

    # ── Round unique ──────────────────────────────────────────────────────────

    def _run_round(
        self,
        clients           : List["FedClient"],
        episodes_per_round: int,
        evaluation_scenarios=None,
        multi_env=None,
    ) -> dict:
        """Exécute un round complet : broadcast → train → collect → aggregate."""

        # 1. Broadcast : envoyer le modèle global à chaque robot
        for client in clients:
            client.set_weights(copy.deepcopy(self.global_weights))

        # 2. Entraînement local (chaque robot s'entraîne indépendamment)
        if multi_env is not None:
            from federated.multi_robot_training import train_clients_multi_robot
            local_results = train_clients_multi_robot(
                clients, multi_env, episodes_per_round
            )
        else:
            local_results = [
                client.train_local(episodes_per_round)
                for client in clients
            ]

        # 3. Collect : récupérer poids locaux + n_samples
        weights_list = [client.get_weights()   for client in clients]
        n_samples    = [r["n_episodes"]        for r in local_results]
        n_transitions = [r.get("n_transitions", 0) for r in local_results]

        # 4. Aggregate. Keep standard FedAvg as the default and expose the
        # visit-weighted Q-table variant as a separate review experiment.
        if (
            self.q_aggregation == "visitation_weighted"
            and clients[0].algo_name == "qlearning"
        ):
            visit_counts = [client.get_visit_counts() for client in clients]
            self.global_weights = visitation_weighted_q_aggregate(
                weights_list,
                visit_counts,
                previous_global=self.global_weights,
            )
        else:
            self.global_weights = fedavg(weights_list, n_samples)

        # 5. Métriques du round
        all_rewards   = [r for res in local_results for r in res.get("rewards", [])]
        all_successes = [r for res in local_results for r in res.get("successes", [])]

        global_eval = {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "n_eval": 0,
        }
        if evaluation_scenarios:
            clients[0].set_weights(copy.deepcopy(self.global_weights))
            global_eval = clients[0].evaluate(evaluation_scenarios)

        result = {
            "round"       : self._round,
            "local_mean_reward" : float(np.mean(all_rewards)) if all_rewards else 0.0,
            "local_std_reward"  : float(np.std(all_rewards)) if all_rewards else 0.0,
            "local_success_rate": float(np.mean(all_successes)) if all_successes else 0.0,
            "global_mean_reward": global_eval["mean_reward"],
            "global_std_reward": global_eval["std_reward"],
            "global_success_rate": global_eval["success_rate"],
            "global_mean_steps": global_eval["mean_steps"],
            "global_n_eval": global_eval["n_eval"],
            "n_episodes"  : sum(n_samples),
            "n_transitions": int(sum(n_transitions)),
            "collisions": int(sum(res.get("collisions", 0) for res in local_results)),
            "per_client"  : [
                {
                    "robot_id"    : res.get("robot_id", i),
                    "mean_reward" : float(np.mean(res.get("rewards", [0]))),
                    "success_rate": float(np.mean(res.get("successes", [0]))),
                }
                for i, res in enumerate(local_results)
            ],
        }
        for metric in ("loss_actor", "loss_critic", "entropy", "kl", "clip_frac", "grad_norm"):
            values = [res[metric] for res in local_results if metric in res]
            result[metric] = float(np.mean(values)) if values else 0.0
        return result

    # ── Sauvegarde ────────────────────────────────────────────────────────────

    def _save_global(self, round_n: int, final: bool = False) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "final" if final else f"round_{round_n:03d}"

        # Q-table (numpy)
        if isinstance(self.global_weights, np.ndarray):
            import pickle
            path = self.save_dir / f"global_qtable_{suffix}.pkl"
            with open(path, "wb") as f:
                pickle.dump(self.global_weights, f)

        # DQN/PPO state_dict (torch)
        elif isinstance(self.global_weights, dict):
            import torch
            path = self.save_dir / f"global_model_{suffix}.pt"
            torch.save(self.global_weights, path)
        else:
            return

        tag = "✅ FINAL" if final else "💾"
        log.info(f"  {tag} Modèle global → {path.name}")

    def save_history(self) -> None:
        """Sauvegarde l'historique des rounds en CSV."""
        import csv
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / "fl_history.csv"
        keys = [
            "round", "local_mean_reward", "local_std_reward", "local_success_rate",
            "global_mean_reward", "global_std_reward", "global_success_rate",
            "global_mean_steps", "global_n_eval", "n_episodes", "loss_actor",
            "n_transitions",
            "loss_critic", "entropy", "kl", "clip_frac", "grad_norm", "collisions",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.round_history)
        log.info(f"  📊 Historique FL → {path.name}")
