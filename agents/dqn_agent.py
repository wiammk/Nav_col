import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.dqn_network import (
    DQN,
    STANDARD_DQN_ARCHITECTURE,
    validate_standard_dqn_state_dict,
)


class ReplayBuffer:
    """Buffer circulaire pour l'apprentissage DQN."""

    def __init__(self, capacity: int = 5000):
        self.capacity = int(capacity)
        self._buffer = []
        self._ptr = 0

    def push(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        item = (obs, int(action), float(reward), next_obs, bool(done))
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
        else:
            self._buffer[self._ptr] = item
        self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, n: int) -> List[Tuple]:
        size = min(int(n), len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=size, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class DQNAgent:
    """
    Agent DQN standard local pour GraphEnv.

    Le reseau est defini dans models/dqn_network.py. Cette classe ajoute la
    strategie epsilon-greedy, le replay buffer, la cible TD et les updates.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 5000,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = DQN(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            target_update_mode="soft",
            tau=tau,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.online.parameters(), lr=lr)

        self.emb_dim = emb_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)

        self.total_episodes = 0
        self.total_updates = 0

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.model.online.state_dict())

    def set_weights(self, weights: Dict[str, torch.Tensor], reset_optimizer: bool = True) -> None:
        validate_standard_dqn_state_dict(weights, "poids FedAvg DQN")
        self.model.online.load_state_dict(copy.deepcopy(weights))
        self.model.target.load_state_dict(copy.deepcopy(weights))
        if reset_optimizer:
            # Adam moments belong to the previous local model and must not leak
            # across FedAvg broadcasts.
            self.optimizer = optim.Adam(self.model.online.parameters(), lr=self.lr)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": STANDARD_DQN_ARCHITECTURE,
                "state_dict": self.model.online.state_dict(),
                "emb_dim": self.emb_dim,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt["state_dict"]
        validate_standard_dqn_state_dict(state_dict, str(path))
        architecture = ckpt.get("architecture")
        if architecture not in {None, STANDARD_DQN_ARCHITECTURE}:
            raise ValueError(f"Architecture DQN non supportée: {architecture}")
        self.model.online.load_state_dict(state_dict)
        self.model.target.load_state_dict(state_dict)
        self.epsilon = ckpt.get("epsilon", self.epsilon)

    def train(self, env, n_episodes: int) -> dict:
        rewards_ep: List[float] = []
        successes: List[float] = []
        n_transitions = 0

        for _ in range(n_episodes):
            reward, success = self._run_episode(env)
            rewards_ep.append(float(reward))
            successes.append(float(success))
            n_transitions += int(env._step_count)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.total_episodes += n_episodes
        return {
            "rewards": rewards_ep,
            "successes": successes,
            "n_episodes": n_episodes,
            "n_transitions": n_transitions,
            "epsilon": self.epsilon,
        }

    def act(self, obs: dict, greedy: bool = False) -> int:
        current_emb, target_emb, neighbor_embs, context, valid = self._obs_to_tensors(obs)
        if not valid:
            return 0

        explore = (not greedy) and (np.random.rand() < self.epsilon)
        if explore:
            return int(np.random.choice(valid))

        was_training = self.model.online.training
        self.model.online.eval()
        with torch.no_grad():
            q_values = self.model.online(
                current_emb, target_emb, neighbor_embs, context
            )
            local_action = int(q_values.argmax().item())
        if was_training:
            self.model.online.train()
        return valid[local_action]

    def _run_episode(self, env):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

            if len(self.buffer) >= self.batch_size:
                self._learn()
            self.model.update_target()

        success = 1.0 if env.current_node == env.target_node else 0.0
        return total_reward, success

    def observe_transition(self, obs, action, reward, next_obs, done) -> None:
        """Consume one transition produced by a joint MultiRobotEnv step."""
        self.buffer.push(obs, action, reward, next_obs, done)
        if len(self.buffer) >= self.batch_size:
            self._learn()
        self.model.update_target()

    def end_episode(self) -> None:
        self.total_episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _learn(self) -> None:
        batch = self.buffer.sample(self.batch_size)
        losses = []

        self.model.online.train()
        for obs, action, reward, next_obs, done in batch:
            current_emb, target_emb, neighbor_embs, context, valid = self._obs_to_tensors(obs)
            if not valid or action not in valid:
                continue

            (
                next_current_emb,
                next_target_emb,
                next_neighbor_embs,
                next_context,
                _,
            ) = self._obs_to_tensors(next_obs)

            local_action = valid.index(action)
            q_pred = self.model.online(
                current_emb, target_emb, neighbor_embs, context
            )[local_action]
            td_target = self.model.compute_td_target(
                next_current_emb,
                next_target_emb,
                next_neighbor_embs,
                reward,
                done,
                self.gamma,
                next_neighbor_context=next_context,
            ).to(self.device)
            losses.append(F.smooth_l1_loss(q_pred, td_target))

        if not losses:
            return

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.total_updates += 1

    def _obs_to_tensors(self, obs: dict):
        state = torch.tensor(obs["state"], dtype=torch.float32, device=self.device)
        neighbors = torch.tensor(obs["neighbors"], dtype=torch.float32, device=self.device)
        occupancy = torch.tensor(
            obs.get("occupancy", np.zeros(len(obs["mask"]))),
            dtype=torch.float32,
            device=self.device,
        )
        mask = obs["mask"]
        valid = [i for i, m in enumerate(mask) if m == 1]

        current_emb = state[: self.emb_dim]
        target_emb = state[self.emb_dim :]
        neighbor_embs = neighbors[valid] if valid else neighbors[:0]
        neighbor_context = occupancy[valid].unsqueeze(-1) if valid else occupancy[:0].reshape(0, 1)
        return current_emb, target_emb, neighbor_embs, neighbor_context, valid
