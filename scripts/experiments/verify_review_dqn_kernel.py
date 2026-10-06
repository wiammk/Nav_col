"""Compare scalar and batched DQN updates on exactly the same replay samples."""
import copy
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(1)
from agents.dqn_agent import DQNAgent
from scripts.experiments.review_followup_worker import factory
from scripts.experiments.review_batched_dqn import batched_learn

def main():
    results=[]
    for seed in (18001,18002,18003):
        np.random.seed(seed);torch.manual_seed(seed)
        env,_,_=factory(ROOT/'runs/Office_Building/data/processed/graph.gpickle',representation='gcn',seed=seed)
        agent=DQNAgent(emb_dim=64,hidden_dim=128,lr=3e-4,batch_size=64,buffer_size=20000)
        obs,_=env.reset()
        for _ in range(128):
            valid=np.flatnonzero(obs['mask']);action=int(np.random.choice(valid)) if len(valid) else 0
            nxt,r,t,tr,_=env.step(action)
            agent.buffer.push(obs,action,r,nxt,t or tr)
            obs=env.reset()[0] if t or tr else nxt
        state=copy.deepcopy(agent.model.state_dict());opt=copy.deepcopy(agent.optimizer.state_dict());rng=np.random.get_state()
        tick=time.perf_counter();agent._learn();scalar=time.perf_counter()-tick
        reference=copy.deepcopy(agent.model.online.state_dict())
        agent.model.load_state_dict(state);agent.optimizer.load_state_dict(opt);np.random.set_state(rng)
        tick=time.perf_counter();batched_learn(agent);batched=time.perf_counter()-tick
        diff=max(float((reference[k]-agent.model.online.state_dict()[k]).abs().max()) for k in reference)
        results.append(dict(seed=seed,max_parameter_difference=diff,absolute_tolerance=2e-6,passed=diff<2e-6,
            scalar_seconds=scalar,batched_seconds=batched,speedup=scalar/batched))
    out=ROOT/'runs/Office_Building/review_followup/kernel_equivalence.json'
    out.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results))
    assert all(r['passed'] for r in results)

if __name__=='__main__':
    main()
