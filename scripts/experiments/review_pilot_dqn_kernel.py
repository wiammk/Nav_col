"""Frozen first batched DQN kernel used in the three tuning pilot runs.

Final matched representation/budget runs use review_batched_dqn.py. Preserve
this version so that the validation-only hyperparameter selection is reproducible.
"""
import numpy as np
import torch
import torch.nn.functional as F


def batched_learn(self):
    batch=self.buffer.sample(self.batch_size)
    chosen=[];next_inputs=[];rewards=[]
    for obs,action,reward,next_obs,done in batch:
        valid=np.flatnonzero(np.asarray(obs['mask'])==1)
        if action not in valid:
            continue
        state=np.asarray(obs['state'],dtype=np.float32)
        neighbor=np.asarray(obs['neighbors'][action],dtype=np.float32)
        occupancy=float(obs.get('occupancy',np.zeros(len(obs['mask'])))[action])
        chosen.append(np.concatenate((state,neighbor,[occupancy])))
        valid_next=np.flatnonzero(np.asarray(next_obs['mask'])==1)
        if done or len(valid_next)==0:
            next_inputs.append(np.empty((0,len(chosen[-1])),dtype=np.float32))
        else:
            state_next=np.asarray(next_obs['state'],dtype=np.float32)
            neighbors=np.asarray(next_obs['neighbors'][valid_next],dtype=np.float32)
            context=np.asarray(next_obs.get('occupancy',np.zeros(len(next_obs['mask']))),dtype=np.float32)[valid_next,None]
            next_inputs.append(np.concatenate((np.broadcast_to(state_next,(len(valid_next),len(state_next))),neighbors,context),axis=1))
        rewards.append(reward)
    if not chosen:
        return
    self.model.online.train()
    x=torch.as_tensor(np.asarray(chosen,dtype=np.float32),device=self.device)
    q=self.model.online.q_network(x).squeeze(-1)
    targets=torch.as_tensor(rewards,dtype=torch.float32,device=self.device)
    lengths=[len(x) for x in next_inputs]
    if sum(lengths):
        all_next=torch.as_tensor(np.concatenate([x for x in next_inputs if len(x)]),dtype=torch.float32,device=self.device)
        with torch.no_grad():
            all_values=self.model.target.q_network(all_next).squeeze(-1)
            offset=0
            for i,length in enumerate(lengths):
                if length:
                    targets[i]+=self.gamma*all_values[offset:offset+length].max()
                    offset+=length
    loss=F.smooth_l1_loss(q,targets)
    self.optimizer.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.online.parameters(),max_norm=10.)
    self.optimizer.step();self.total_updates+=1


def install():
    from agents.dqn_agent import DQNAgent
    DQNAgent._learn=batched_learn
