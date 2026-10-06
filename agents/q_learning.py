"""Masked tabular Q-learning over current-node and target-node indices."""

import os
import numpy as np
import pickle
from typing import Tuple


class QLearningAgent:
    """
    Q-learning tabulaire compatible avec GraphEnv.

    État utilisé : (current_node_idx, target_node_idx)
    Q-table de forme : (num_nodes, num_nodes, max_degree)
      - axe 0 : noeud courant
      - axe 1 : noeud cible
      - axe 2 : index d'action (voisin parmi max_degree)
    """

    def __init__(
        self,
        num_nodes: int,
        max_degree: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        seed: int = 0,
    ):
        self.num_nodes = int(num_nodes)
        self.max_degree = int(max_degree)

        self.lr = float(lr)
        self.gamma = float(gamma)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.epsilon_step = 0
        self.epsilon = self.epsilon_start

        self.rng = np.random.default_rng(seed)

        # Q[s, t, a]
        self.Q = np.zeros(
            (self.num_nodes, self.num_nodes, self.max_degree),
            dtype=np.float32
        )
        self.visit_counts = np.zeros_like(self.Q, dtype=np.int64)

        self.total_updates = 0

    def state_to_idx(self, current_idx: int, target_idx: int) -> Tuple[int, int]:
        return int(current_idx), int(target_idx)

    def _decay_epsilon(self):
        if self.epsilon_step < self.epsilon_decay_steps:
            frac = self.epsilon_step / max(1, self.epsilon_decay_steps)
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
            self.epsilon_step += 1
        else:
            self.epsilon = self.epsilon_end

    def choose_action(
        self,
        current_idx: int,
        target_idx: int,
        mask: np.ndarray,
    ) -> int:
        """
        Politique epsilon-greedy avec masque d'actions valides.

        mask : array binaire de taille max_degree (1 = action valide, 0 = invalide)
        """
        mask = mask.astype(bool)
        valid_actions = np.where(mask)[0]

        if len(valid_actions) == 0:
            # fallback de sécurité (ne devrait pas arriver si l'env est bien défini)
            return 0

        self._decay_epsilon()

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))

        s_i, t_i = self.state_to_idx(current_idx, target_idx)
        qvals = self.Q[s_i, t_i]  # (max_degree,)

        masked_q = np.full_like(qvals, -np.inf, dtype=np.float32)
        masked_q[valid_actions] = qvals[valid_actions]

        return int(np.argmax(masked_q))

    def greedy_action(
        self,
        current_idx: int,
        target_idx: int,
        mask: np.ndarray,
    ) -> int:
        mask = np.asarray(mask, dtype=bool)
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            return 0
        qvals = self.Q[int(current_idx), int(target_idx)]
        return int(valid_actions[np.argmax(qvals[valid_actions])])

    def update(
        self,
        current_idx: int,
        target_idx: int,
        action: int,
        reward: float,
        next_idx: int,
        next_mask: np.ndarray,
        done: bool,
    ):
        """
        Mise à jour Q-learning :

          Q(s,a) ← Q(s,a) + lr * (r + γ max_a' Q(s',a') - Q(s,a))

        next_mask est utilisé pour ne prendre que les actions valides dans s'.
        """
        s_i, t_i = self.state_to_idx(current_idx, target_idx)
        q_sa = self.Q[s_i, t_i, action]

        if done:
            target = reward
        else:
            next_mask = next_mask.astype(bool)
            valid_next = np.where(next_mask)[0]

            if len(valid_next) == 0:
                max_next = 0.0
            else:
                max_next = float(
                    np.max(self.Q[next_idx, target_idx, valid_next])
                )

            target = reward + self.gamma * max_next

        self.Q[s_i, t_i, action] = q_sa + self.lr * (target - q_sa)
        self.visit_counts[s_i, t_i, action] += 1
        self.total_updates += 1

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "Q": self.Q,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_step": self.epsilon_step,
                    "visit_counts": self.visit_counts,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.Q = data.get("Q", self.Q)
        self.lr = data.get("lr", self.lr)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_step = data.get("epsilon_step", self.epsilon_step)
        self.visit_counts = data.get("visit_counts", self.visit_counts)
