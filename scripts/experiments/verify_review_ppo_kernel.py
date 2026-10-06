"""Verify batched PPO against the original update with identical rollout/RNG."""
import copy
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(1)
from agents.ppo_agent import PPOAgent
from scripts.experiments.review_followup_worker import factory
from scripts.experiments.review_batched_ppo import batched_update, minibatch_forward


def main():
    results=[]
    for seed in (18001,18002,18003):
        np.random.seed(seed);torch.manual_seed(seed)
        env,_,_=factory(ROOT/'runs/Office_Building/data/processed/graph.gpickle',representation='gcn',seed=seed)
        agent=PPOAgent(emb_dim=64,hidden_dim=128,lr=3e-4,n_epochs=10,minibatch_size=64)
        buffer={k:[] for k in ('obs','actions','rewards','values','log_probs','dones')}
        obs,_=env.reset()
        for _ in range(64):
            c,g,n,x,valid=agent._obs_to_tensors(obs)
            if not valid:
                obs=env.reset()[0];continue
            with torch.no_grad():
                dist,value=agent.model(c,g,n,x);local=dist.sample();log_prob=dist.log_prob(local)
            nxt,r,t,tr,_=env.step(valid[int(local.item())])
            for k,v in dict(obs=obs,actions=local.detach(),rewards=r,values=value.detach(),log_probs=log_prob.detach(),dones=float(t)).items():
                buffer[k].append(v)
            obs=env.reset()[0] if t or tr else nxt
        state=copy.deepcopy(agent.model.state_dict());opt=copy.deepcopy(agent.optimizer.state_dict());rng=torch.get_rng_state()
        distribution,value_batch=minibatch_forward(agent,buffer['obs'])
        scalar_log=[];scalar_value=[];global_actions=[]
        for o,action in zip(buffer['obs'],buffer['actions']):
            c,g,n,x,valid=agent._obs_to_tensors(o)
            dist,v=agent.model(c,g,n,x)
            scalar_log.append(dist.log_prob(action));scalar_value.append(v)
            global_actions.append(valid[int(action.item())])
        log_error=float((torch.stack(scalar_log)-distribution.log_prob(torch.tensor(global_actions))).abs().max().detach())
        value_error=float((torch.stack(scalar_value)-value_batch).abs().max().detach())
        tick=time.perf_counter();reference_metrics=agent._update(buffer,obs,False);scalar=time.perf_counter()-tick
        reference=copy.deepcopy(agent.model.state_dict())
        agent.model.load_state_dict(state);agent.optimizer.load_state_dict(opt);torch.set_rng_state(rng)
        tick=time.perf_counter();metrics=batched_update(agent,buffer,obs,False);batched=time.perf_counter()-tick
        diff=max(float((reference[k]-agent.model.state_dict()[k]).abs().max()) for k in reference)
        metric_error=max(abs(reference_metrics[i][k]-metrics[i][k]) for i in range(len(metrics))
            for k in ('loss_actor','loss_critic','entropy','kl'))
        results.append(dict(seed=seed,max_parameter_difference_after_ten_updates=diff,
            max_log_probability_difference=log_error,max_value_difference=value_error,
            max_reported_loss_entropy_kl_difference=metric_error,
            numerical_not_bitwise_equivalence=True,
            same_update_count=len(reference_metrics)==len(metrics),
            passed=log_error<1e-6 and value_error<1e-6 and metric_error<1e-4 and len(reference_metrics)==len(metrics),
            scalar_seconds=scalar,batched_seconds=batched,speedup=scalar/batched))
    (ROOT/'runs/Office_Building/review_followup/ppo_kernel_equivalence.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results));assert all(r['passed'] for r in results)


if __name__=='__main__':
    main()
