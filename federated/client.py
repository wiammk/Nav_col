"""
federated/client.py
===================
FedClient — interface commune pour Q-Learning, DQN et PPO.

Pourquoi une seule classe ?
    Le serveur FedAvg n'a besoin que de 3 opérations :
      get_weights()         → retourner les poids du modèle local
      set_weights(w)        → charger les poids globaux reçus
      train_local(episodes) → s'entraîner N épisodes, retourner métriques

    Que le modèle soit une Q-table ou un réseau de neurones, ces 3
    méthodes suffisent. FedClient les adapte automatiquement.

Classes disponibles :
    QLearningClient   : wrappeur pour QLearningAgent
    DQNClient         : wrappeur pour le DQN standard
    PPOClient         : wrappeur pour PPOActorCritic
    make_client()     : factory qui retourne le bon client selon l'algo

Usage :
    from federated.client import make_client
    client = make_client("qlearning", env, robot_id=0)
    client = make_client("dqn",       env, robot_id=1)
    client = make_client("ppo",       env, robot_id=2)
"""

import logging
import numpy as np
from typing import Any

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — BASE CLIENT
# ═════════════════════════════════════════════════════════════════════════════

class FedClient:
    """
    Interface de base — à hériter par chaque type d'agent.

    Chaque sous-classe implémente :
      get_weights()         → Any   (Q-table ou state_dict)
      set_weights(w)        → None
      train_local(n)        → dict  {rewards, successes, n_episodes, robot_id}
    """

    def __init__(self, env, robot_id: int = 0, algo_name: str = "base"):
        self.env       = env          # GraphEnv (1 robot)
        self.robot_id  = robot_id
        self.algo_name = algo_name
        self._episode_count = 0
        self.data_profile = None
        self.profile_rng = np.random.default_rng(10_000 + robot_id)

    def configure_data_profile(self, start_nodes, target_nodes, label: str) -> None:
        """Assign a reproducible client-specific trajectory distribution."""
        self.data_profile = {
            "label": str(label),
            "start_nodes": list(start_nodes),
            "target_nodes": list(target_nodes),
        }

    def sample_start_target(self):
        if not self.data_profile:
            return None, None
        starts = self.data_profile["start_nodes"]
        targets = self.data_profile["target_nodes"]
        if not starts or not targets:
            return None, None
        start = self.profile_rng.choice(starts)
        candidates = [node for node in targets if node != start]
        if not candidates:
            candidates = [node for node in self.env.nodes if node != start]
        return start, self.profile_rng.choice(candidates)

    def get_weights(self) -> Any:
        raise NotImplementedError

    def set_weights(self, weights: Any) -> None:
        raise NotImplementedError

    def train_local(self, n_episodes: int) -> dict:
        raise NotImplementedError

    def greedy_action(self, obs: dict) -> int:
        raise NotImplementedError

    def evaluate(self, scenarios) -> dict:
        """Evaluate without learning on an externally fixed scenario list."""
        rewards, successes, steps = [], [], []
        for start, target in scenarios:
            obs, _ = self.env.reset(start=start, target=target)
            total_reward = 0.0
            done = False
            n_steps = 0
            while not done:
                obs, reward, terminated, truncated, _ = self.env.step(
                    self.greedy_action(obs)
                )
                total_reward += float(reward)
                n_steps += 1
                done = terminated or truncated
            rewards.append(total_reward)
            successes.append(float(self.env.current_node == self.env.target_node))
            steps.append(n_steps)
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "mean_steps": float(np.mean(steps)) if steps else 0.0,
            "n_eval": len(rewards),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CLIENT Q-LEARNING
# ═════════════════════════════════════════════════════════════════════════════

class QLearningClient(FedClient):
    """
    FedClient pour Q-Learning tabulaire.

    Poids partagés = Q-table numpy array shape (n_nodes, n_nodes, max_degree).
    FedAvg moyenne directement les valeurs Q de chaque robot.
    """

    def __init__(
        self,
        env,
        robot_id   : int   = 0,
        alpha      : float = 0.1,    # taux d'apprentissage
        gamma      : float = 0.99,   # discount
        epsilon    : float = 1.0,    # exploration initiale
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 0,
    ):
        super().__init__(env, robot_id, algo_name="qlearning")

        n     = len(env.nodes)
        k     = env.max_degree
        self.Q          = np.zeros((n, n, k), dtype=np.float32)
        self.visit_counts = np.zeros((n, n, k), dtype=np.int64)
        self.node2idx   = env.node2idx
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.rng = np.random.default_rng(seed)

    # ── Interface FedClient ───────────────────────────────────────────────────

    def get_weights(self) -> np.ndarray:
        """Retourne une copie de la Q-table."""
        return self.Q.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        """Charge les poids globaux dans la Q-table locale."""
        self.Q = weights.copy().astype(np.float32)
        self.visit_counts.fill(0)

    def get_visit_counts(self) -> np.ndarray:
        return self.visit_counts.copy()

    def train_local(self, n_episodes: int) -> dict:
        """
        Entraîne l'agent Q-Learning pour n_episodes épisodes.

        Retourne métriques pour le serveur.
        """
        rewards_ep, successes = [], []
        n_transitions = 0

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            cur    = self.env.current_node
            tgt    = self.env.target_node
            total_r = 0.0
            done    = False

            while not done:
                action  = self._choose_action(cur, tgt, obs["mask"])
                obs, r, term, trunc, _ = self.env.step(action)
                next_cur = self.env.current_node
                done     = term or trunc

                self._update(cur, tgt, action, r, next_cur, tgt, obs["mask"], done)

                cur      = next_cur
                total_r += r
                n_transitions += 1

            rewards_ep.append(total_r)
            successes.append(1.0 if self.env.current_node == self.env.target_node else 0.0)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._episode_count += n_episodes
        return {
            "robot_id"  : self.robot_id,
            "rewards"   : rewards_ep,
            "successes" : successes,
            "n_episodes": n_episodes,
            "n_transitions": n_transitions,
            "epsilon"   : self.epsilon,
        }

    # ── Q-Learning interne ────────────────────────────────────────────────────

    def _choose_action(self, cur, tgt, mask: np.ndarray) -> int:
        valid = [i for i, m in enumerate(mask) if m == 1]
        if not valid:
            return 0
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))
        ci = self.node2idx[cur]
        ti = self.node2idx[tgt]
        q  = self.Q[ci, ti, :]
        q_valid = [(q[a], a) for a in valid]
        return max(q_valid, key=lambda x: x[0])[1]

    def greedy_action(self, obs: dict) -> int:
        valid = [i for i, value in enumerate(obs["mask"]) if value == 1]
        if not valid:
            return 0
        current = self.node2idx[self.env.current_node]
        target = self.node2idx[self.env.target_node]
        return max(valid, key=lambda action: self.Q[current, target, action])

    def _update(self, cur, tgt, action, reward, next_cur, next_tgt,
                next_mask, done):
        ci    = self.node2idx[cur]
        ti    = self.node2idx[tgt]
        ni    = self.node2idx[next_cur]
        nti   = self.node2idx[next_tgt]

        valid_next = [i for i, m in enumerate(next_mask) if m == 1]
        max_next_q = (
            0.0 if done or not valid_next
            else float(self.Q[ni, nti, valid_next].max())
        )

        target = reward + (0.0 if done else self.gamma * max_next_q)
        self.Q[ci, ti, action] += self.alpha * (target - self.Q[ci, ti, action])
        self.visit_counts[ci, ti, action] += 1


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CLIENT DQN
# ═════════════════════════════════════════════════════════════════════════════

class DQNClient(FedClient):
    """
    FedClient pour le DQN standard avec réseau online et réseau cible.

    Poids partagés = state_dict du réseau online (dict de tenseurs PyTorch).
    FedAvg moyenne les paramètres couche par couche.
    """

    def __init__(
        self,
        env,
        robot_id   : int   = 0,
        emb_dim    : int   = 64,
        hidden_dim : int   = 128,
        lr         : float = 1e-3,
        gamma      : float = 0.99,
        epsilon    : float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size : int   = 32,
        buffer_size: int   = 5000,
        tau        : float = 0.005,
    ):
        super().__init__(env, robot_id, algo_name="dqn")
        from agents.dqn_agent import DQNAgent

        self.agent = DQNAgent(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            buffer_size=buffer_size,
            tau=tau,
        )

    def get_weights(self) -> dict:
        return self.agent.get_weights()

    def set_weights(self, weights: dict) -> None:
        self.agent.set_weights(weights, reset_optimizer=True)

    def train_local(self, n_episodes: int) -> dict:
        metrics = self.agent.train(self.env, n_episodes)
        self._episode_count += n_episodes
        metrics["robot_id"] = self.robot_id
        return metrics

    def greedy_action(self, obs: dict) -> int:
        return self.agent.act(obs, greedy=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — CLIENT PPO
# ═════════════════════════════════════════════════════════════════════════════

class PPOClient(FedClient):
    """
    FedClient pour PPO (Actor-Critic).

    Poids partagés = state_dict complet du PPOActorCritic.
    FedAvg moyenne les paramètres acteur ET critique ensemble.
    """

    def __init__(
        self,
        env,
        robot_id   : int   = 0,
        emb_dim    : int   = 64,
        hidden_dim : int   = 128,
        lr         : float = 3e-4,
        gamma      : float = 0.99,
        lam        : float = 0.95,
        clip_eps   : float = 0.2,
        n_epochs   : int   = 10,
        rollout_len: int   = 256,
        minibatch_size: int = 64,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.03,
    ):
        super().__init__(env, robot_id, algo_name="ppo")
        from agents.ppo_agent import PPOAgent

        self.agent = PPOAgent(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            lam=lam,
            clip_eps=clip_eps,
            n_epochs=n_epochs,
            rollout_len=rollout_len,
            minibatch_size=minibatch_size,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
        )

    def get_weights(self) -> dict:
        return self.agent.get_weights()

    def set_weights(self, weights: dict) -> None:
        self.agent.set_weights(weights, reset_optimizer=True)

    def train_local(self, n_episodes: int) -> dict:
        metrics = self.agent.train(self.env, n_episodes)
        self._episode_count += n_episodes
        metrics["robot_id"] = self.robot_id
        return metrics

    def greedy_action(self, obs: dict) -> int:
        action, _, _ = self.agent.act(obs, greedy=True)
        return action


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def make_client(
    algo    : str,
    env,
    robot_id: int = 0,
    **kwargs,
) -> FedClient:
    """
    Crée le bon FedClient selon l'algorithme.

    Paramètres
    ----------
    algo     : "qlearning" | "dqn" | "ppo"
    env      : GraphEnv (un seul robot)
    robot_id : identifiant du robot

    Usage :
        client = make_client("qlearning", env, robot_id=0)
        client = make_client("dqn",       env, robot_id=1, lr=1e-3)
        client = make_client("ppo",       env, robot_id=2, n_epochs=4)
    """
    algo = algo.lower().strip()
    mapping = {
        "qlearning": QLearningClient,
        "q_learning": QLearningClient,
        "dqn"      : DQNClient,
        "ppo"      : PPOClient,
    }
    if algo not in mapping:
        raise ValueError(f"Algo inconnu : {algo}. Choix : {list(mapping.keys())}")

    client = mapping[algo](env, robot_id=robot_id, **kwargs)
    log.info(f"  ✅ {algo.upper()} client créé — robot {robot_id}")
    return client
