import numpy as np
import torch

import agents.ppo_agent as ppo_module
from agents.ppo_agent import PPOAgent


def observation():
    return {
        "state": np.zeros(128, dtype=np.float32),
        "neighbors": np.stack([
            np.ones(64, dtype=np.float32),
            -np.ones(64, dtype=np.float32),
        ]),
        "mask": np.asarray([1, 1], dtype=np.int8),
        "occupancy": np.asarray([0, 0], dtype=np.int8),
    }


def rollout_buffer(agent, length=4):
    obs = observation()
    buffer = {key: [] for key in ("obs", "actions", "rewards", "values", "log_probs", "dones")}
    for _ in range(length):
        current, target, neighbors, context, _ = agent._obs_to_tensors(obs)
        with torch.no_grad():
            dist, value = agent.model(current, target, neighbors, context)
            action = torch.tensor(0)
        buffer["obs"].append(obs)
        buffer["actions"].append(action)
        buffer["rewards"].append(1.0)
        buffer["values"].append(value.detach())
        buffer["log_probs"].append(dist.log_prob(action).detach())
        buffer["dones"].append(0.0)
    return buffer


def test_non_terminal_rollout_bootstraps_next_value(monkeypatch):
    agent = PPOAgent(n_epochs=1, minibatch_size=2, rollout_len=4)
    captured = {}
    original = ppo_module.compute_gae

    def capture(rewards, values, dones, gamma, lam):
        captured["bootstrap"] = float(values[-1])
        return original(rewards, values, dones, gamma, lam)

    monkeypatch.setattr(ppo_module, "compute_gae", capture)
    monkeypatch.setattr(agent, "_bootstrap_value", lambda obs, terminal: torch.tensor(2.5))
    metrics = agent._update(rollout_buffer(agent), observation(), terminal=False)
    assert captured["bootstrap"] == 2.5
    assert metrics
    assert {"loss_actor", "loss_critic", "entropy", "kl", "grad_norm"} <= metrics[0].keys()


def test_fedavg_weight_reception_resets_ppo_optimizer():
    agent = PPOAgent()
    previous = agent.optimizer
    agent.set_weights(agent.get_weights(), reset_optimizer=True)
    assert agent.optimizer is not previous
