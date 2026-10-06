"""Experiment-only batched execution of the existing scalar DQN update.

The replay sampling, target network, terminal handling, Huber loss, Adam and
gradient clipping are unchanged. The original agent remains unchanged.
"""
import numpy as np
import torch
import torch.nn.functional as F


def batched_learn(self):
    batch = self.buffer.sample(self.batch_size)
    chosen = []
    next_inputs = []
    next_masks = []
    rewards = []
    bootstrap = []
    for obs, action, reward, next_obs, done in batch:
        if action < 0 or action >= len(obs['mask']) or obs['mask'][action] != 1:
            continue
        chosen.append(_inputs(obs)[action])
        next_inputs.append(_inputs(next_obs))
        mask = np.asarray(next_obs['mask']) == 1
        next_masks.append(mask)
        bootstrap.append(not done and bool(mask.any()))
        rewards.append(reward)
    if not chosen:
        return
    self.model.online.train()
    x = torch.as_tensor(np.asarray(chosen,dtype=np.float32),device=self.device)
    q = self.model.online.q_network(x).squeeze(-1)
    targets = torch.as_tensor(rewards,dtype=torch.float32,device=self.device)
    all_next = torch.as_tensor(np.asarray(next_inputs),dtype=torch.float32,device=self.device)
    mask = torch.as_tensor(np.asarray(next_masks),dtype=torch.bool,device=self.device)
    eligible = torch.as_tensor(bootstrap,dtype=torch.bool,device=self.device)
    with torch.no_grad():
        all_values = self.model.target.q_network(all_next).squeeze(-1)
        maxima = all_values.masked_fill(~mask,float('-inf')).max(dim=1).values
        targets += self.gamma * torch.where(eligible,maxima,torch.zeros_like(maxima))
    loss = F.smooth_l1_loss(q,targets)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.online.parameters(),max_norm=10.)
    self.optimizer.step()
    self.total_updates += 1


def _inputs(obs):
    cached = obs.get('_review_dqn_inputs')
    if cached is None:
        state = np.asarray(obs['state'],dtype=np.float32)
        neighbors = np.asarray(obs['neighbors'],dtype=np.float32)
        context = np.asarray(obs.get('occupancy',np.zeros(len(obs['mask']))),dtype=np.float32)[:,None]
        cached = np.concatenate((np.broadcast_to(state,(len(neighbors),len(state))),neighbors,context),axis=1)
        obs['_review_dqn_inputs'] = cached
    return cached


def install():
    from agents.dqn_agent import DQNAgent
    DQNAgent._learn = batched_learn
