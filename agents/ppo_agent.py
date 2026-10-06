import copy
from pathlib import Path
from typing import Dict, List

import torch
import torch.optim as optim

from models.ppo_network import PPOActorCritic, compute_gae, ppo_loss


class PPOAgent:
    """
    Agent PPO local pour la navigation sur GraphEnv.

    Le modele est defini dans models/ppo_network.py. Cette classe gere la
    partie apprentissage: interaction avec l'environnement, collecte des
    trajectoires, calcul GAE et mise a jour PPO-Clip.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        rollout_len: int = 256,
        minibatch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.03,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = PPOActorCritic(emb_dim, hidden_dim, ortho_init=True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.emb_dim = emb_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.rollout_len = rollout_len
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.total_episodes = 0
        self.total_updates = 0

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, weights: Dict[str, torch.Tensor], reset_optimizer: bool = True) -> None:
        self.model.load_state_dict(copy.deepcopy(weights))
        if reset_optimizer:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "emb_dim": self.emb_dim,
                "gamma": self.gamma,
                "lam": self.lam,
                "clip_eps": self.clip_eps,
                "n_epochs": self.n_epochs,
                "rollout_len": self.rollout_len,
                "minibatch_size": self.minibatch_size,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
                "target_kl": self.target_kl,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])

    def train(self, env, n_episodes: int) -> dict:
        rewards_ep: List[float] = []
        successes: List[float] = []
        update_metrics: List[dict] = []
        n_transitions = 0

        self.model.train()
        for _ in range(n_episodes):
            reward, success, episode_updates = self._run_episode(env)
            rewards_ep.append(float(reward))
            successes.append(float(success))
            update_metrics.extend(episode_updates)
            n_transitions += int(env._step_count)

        self.total_episodes += n_episodes
        result = {
            "rewards": rewards_ep,
            "successes": successes,
            "n_episodes": n_episodes,
            "n_transitions": n_transitions,
        }
        for key in ("loss_actor", "loss_critic", "entropy", "kl", "clip_frac", "grad_norm"):
            values = [metric[key] for metric in update_metrics if key in metric]
            result[key] = float(sum(values) / len(values)) if values else 0.0
        result["n_updates"] = len(update_metrics)
        return result

    def act(self, obs: dict, greedy: bool = False):
        current_emb, target_emb, neighbor_embs, context, valid = self._obs_to_tensors(obs)
        if not valid:
            return 0, None, None

        with torch.no_grad():
            local_action, log_prob, entropy, value = self.model.get_action(
                current_emb,
                target_emb,
                neighbor_embs,
                greedy=greedy,
                neighbor_context=context,
            )
        return valid[local_action], log_prob, value

    def _run_episode(self, env):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        update_metrics = []

        buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }

        while not done:
            current_emb, target_emb, neighbor_embs, context, valid = self._obs_to_tensors(obs)

            if not valid:
                obs, reward, terminated, truncated, _ = env.step(0)
                done = terminated or truncated
                total_reward += reward
                continue

            with torch.no_grad():
                dist, value = self.model(
                    current_emb, target_emb, neighbor_embs, context
                )
                local_action = dist.sample()
                log_prob = dist.log_prob(local_action)
                global_action = valid[int(local_action.item())]

            next_obs, reward, terminated, truncated, _ = env.step(global_action)
            done = terminated or truncated

            buffer["obs"].append(obs)
            buffer["actions"].append(local_action.detach())
            buffer["rewards"].append(float(reward))
            buffer["values"].append(value.detach())
            buffer["log_probs"].append(log_prob.detach())
            # A time-limit truncation is not a terminal MDP state and must be bootstrapped.
            buffer["dones"].append(float(terminated))

            obs = next_obs
            total_reward += reward

            if len(buffer["rewards"]) >= self.rollout_len or done:
                metrics = self._update(buffer, next_obs=obs, terminal=terminated)
                if metrics:
                    update_metrics.extend(metrics)
                for values in buffer.values():
                    values.clear()

        success = 1.0 if env.current_node == env.target_node else 0.0
        return total_reward, success, update_metrics

    def _bootstrap_value(self, obs: dict, terminal: bool) -> torch.Tensor:
        if terminal:
            return torch.zeros((), device=self.device)
        current_emb, target_emb, neighbor_embs, context, valid = self._obs_to_tensors(obs)
        if not valid:
            return torch.zeros((), device=self.device)
        with torch.no_grad():
            _, value = self.model(
                current_emb, target_emb, neighbor_embs, context
            )
        return value.detach().view(())

    def _update(self, buffer: dict, next_obs: dict, terminal: bool) -> List[dict]:
        if len(buffer["rewards"]) < 2:
            return []

        rewards = torch.tensor(buffer["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(buffer["dones"], dtype=torch.float32, device=self.device)
        values = torch.stack(buffer["values"]).to(self.device).view(-1)
        old_log_probs = torch.stack(buffer["log_probs"]).to(self.device).detach()
        actions = torch.stack(buffer["actions"]).to(self.device)

        bootstrap_value = self._bootstrap_value(next_obs, terminal)
        values_ext = torch.cat([values, bootstrap_value.view(1)])
        advantages, returns = compute_gae(
            rewards.cpu(),
            values_ext.detach().cpu(),
            dones.cpu(),
            self.gamma,
            self.lam,
        )
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        n_samples = len(buffer["obs"])
        all_metrics: List[dict] = []
        stop_early = False
        for _ in range(self.n_epochs):
            permutation = torch.randperm(n_samples).tolist()
            for start in range(0, n_samples, self.minibatch_size):
                minibatch = permutation[start : start + self.minibatch_size]
                log_probs_new = []
                entropies = []
                values_new = []
                kept_indices = []

                for i in minibatch:
                    obs = buffer["obs"][i]
                    current_emb, target_emb, neighbor_embs, context, valid = self._obs_to_tensors(obs)
                    if not valid or int(actions[i].item()) >= len(valid):
                        continue

                    dist, value = self.model(
                        current_emb, target_emb, neighbor_embs, context
                    )
                    log_probs_new.append(dist.log_prob(actions[i]))
                    entropies.append(dist.entropy())
                    values_new.append(value)
                    kept_indices.append(i)

                if len(log_probs_new) < 2:
                    continue

                idx = torch.tensor(kept_indices, dtype=torch.long, device=self.device)
                loss, _ = ppo_loss(
                    torch.stack(log_probs_new),
                    old_log_probs[idx],
                    advantages[idx],
                    torch.stack(values_new).view(-1),
                    returns[idx],
                    torch.stack(entropies),
                    self.clip_eps,
                    vf_coef=self.value_coef,
                    ent_coef=self.entropy_coef,
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm
                )
                self.optimizer.step()
                self.total_updates += 1

                with torch.no_grad():
                    approx_kl = (
                        old_log_probs[idx] - torch.stack(log_probs_new)
                    ).mean().abs().item()
                _, metrics = ppo_loss(
                    torch.stack(log_probs_new).detach(),
                    old_log_probs[idx],
                    advantages[idx],
                    torch.stack(values_new).view(-1).detach(),
                    returns[idx],
                    torch.stack(entropies).detach(),
                    self.clip_eps,
                    vf_coef=self.value_coef,
                    ent_coef=self.entropy_coef,
                )
                metrics["kl"] = float(approx_kl)
                metrics["grad_norm"] = float(grad_norm)
                all_metrics.append(metrics)
                if self.target_kl > 0 and approx_kl > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        return all_metrics

    def _obs_to_tensors(self, obs: dict):
        state = torch.tensor(obs["state"], dtype=torch.float32, device=self.device)
        neighbors = torch.tensor(obs["neighbors"], dtype=torch.float32, device=self.device)
        occupancy = torch.tensor(
            obs.get("occupancy", [0.0] * len(obs["mask"])),
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
