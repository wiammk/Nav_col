"""Experiment-only batched PPO minibatches; same masks, loss and update schedule."""
import numpy as np
import torch
from torch.distributions import Categorical
from models.ppo_network import compute_gae, ppo_loss


def minibatch_forward(agent, observations):
    states=torch.as_tensor(np.asarray([o['state'] for o in observations]),dtype=torch.float32,device=agent.device)
    neighbors=torch.as_tensor(np.asarray([o['neighbors'] for o in observations]),dtype=torch.float32,device=agent.device)
    contexts=torch.as_tensor(np.asarray([o.get('occupancy',np.zeros(len(o['mask']))) for o in observations]),dtype=torch.float32,device=agent.device).unsqueeze(-1)
    masks=torch.as_tensor(np.asarray([o['mask'] for o in observations]),dtype=torch.bool,device=agent.device)
    shared=agent.model.backbone(states)
    values=agent.model.critic_head(shared).squeeze(-1)
    actor_input=torch.cat((shared[:,None,:].expand(-1,neighbors.shape[1],-1),neighbors,contexts),dim=-1)
    logits=agent.model.actor_head(actor_input).squeeze(-1).masked_fill(~masks,float('-inf'))
    return Categorical(logits=logits),values


def batched_update(self, buffer, next_obs, terminal):
    if len(buffer['rewards'])<2:
        return []
    rewards=torch.tensor(buffer['rewards'],dtype=torch.float32,device=self.device)
    dones=torch.tensor(buffer['dones'],dtype=torch.float32,device=self.device)
    values=torch.stack(buffer['values']).to(self.device).view(-1)
    old_log_probs=torch.stack(buffer['log_probs']).to(self.device).detach()
    actions=torch.stack(buffer['actions']).to(self.device)
    bootstrap=self._bootstrap_value(next_obs,terminal)
    advantages,returns=compute_gae(rewards.cpu(),torch.cat((values,bootstrap.view(1))).detach().cpu(),dones.cpu(),self.gamma,self.lam)
    advantages=advantages.to(self.device);returns=returns.to(self.device)
    n_samples=len(buffer['obs']);all_metrics=[];stop=False
    for _ in range(self.n_epochs):
        permutation=torch.randperm(n_samples).tolist()
        for start in range(0,n_samples,self.minibatch_size):
            kept=[];global_actions=[]
            for i in permutation[start:start+self.minibatch_size]:
                valid=np.flatnonzero(np.asarray(buffer['obs'][i]['mask'])==1)
                local=int(actions[i].item())
                if len(valid) and local<len(valid):
                    kept.append(i);global_actions.append(int(valid[local]))
            if len(kept)<2:
                continue
            idx=torch.tensor(kept,dtype=torch.long,device=self.device)
            dist,value=minibatch_forward(self,[buffer['obs'][i] for i in kept])
            log_prob=dist.log_prob(torch.tensor(global_actions,dtype=torch.long,device=self.device))
            entropy=dist.entropy()
            loss,_=ppo_loss(log_prob,old_log_probs[idx],advantages[idx],value,returns[idx],entropy,self.clip_eps,vf_coef=self.value_coef,ent_coef=self.entropy_coef)
            self.optimizer.zero_grad();loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.max_grad_norm)
            self.optimizer.step();self.total_updates+=1
            with torch.no_grad():
                kl=(old_log_probs[idx]-log_prob).mean().abs().item()
            _,metrics=ppo_loss(log_prob.detach(),old_log_probs[idx],advantages[idx],value.detach(),returns[idx],entropy.detach(),self.clip_eps,vf_coef=self.value_coef,ent_coef=self.entropy_coef)
            metrics['kl']=float(kl);metrics['grad_norm']=float(grad_norm);all_metrics.append(metrics)
            if self.target_kl>0 and kl>self.target_kl:
                stop=True;break
        if stop:
            break
    return all_metrics


def install():
    from agents.ppo_agent import PPOAgent
    PPOAgent._update=batched_update
