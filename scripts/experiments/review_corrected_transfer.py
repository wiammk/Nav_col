"""Recompute Clinic transfer after the IFC width correction, preserving old files."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'runs/Clinic_Architectural_width_corrected/review_transfer'


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def worker(algo, seed):
    sys.path.insert(0,str(ROOT))
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from environment.graph_env import make_env
    from evaluation.common_protocol import evaluate_single_policy,load_protocol,summarize
    from scripts.evaluation.evaluate_generalization import make_agent,load_transfer_checkpoint,greedy_fn
    if algo=='dqn':
        from scripts.experiments.review_batched_dqn import install
    else:
        from scripts.experiments.review_batched_ppo import install
    install()
    out=BASE/f'seed_{seed}'/algo;out.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat();clock=time.perf_counter()
    graph=ROOT/'runs/Clinic_Architectural_width_corrected/data/processed/graph.gpickle'
    encoder=ROOT/'runs/Office_Building/data/processed/gcn_encoder.pt'
    source=ROOT/f'runs/Office_Building/experiments/seed_{seed}/robots_1/fedavg/{algo}/global_model_final.pt'
    protocol_file=graph.parents[2]/'review_followup/protocols/fixed_test_protocol.json'
    protocol=load_protocol(protocol_file)
    summaries=[];histories=[];embedding=None
    for budget,mode in [(0,'zero_shot'),(10,'fine_tuning'),(10,'from_scratch'),
                        (50,'fine_tuning'),(50,'from_scratch'),(100,'fine_tuning'),(100,'from_scratch')]:
        run_seed=seed+1000*budget
        random.seed(run_seed);np.random.seed(run_seed);torch.manual_seed(run_seed)
        env,g,_=make_env(str(graph),n_robots=1,max_steps=200,seed=run_seed,
            shared_gcn_checkpoint_path=str(encoder),
            embedding_cache_path=str(out/'shared_gcn_target_embeddings.npz'))
        embedding=env.embedding_metadata
        # The seed is reset after encoder loading so both modes consume the same
        # initial RNG stream; target training never changes the frozen encoder.
        random.seed(run_seed);np.random.seed(run_seed);torch.manual_seed(run_seed)
        agent=make_agent(algo)
        if mode!='from_scratch':load_transfer_checkpoint(agent,source)
        training=agent.train(env,budget) if budget else {}
        rows=evaluate_single_policy(env,greedy_fn(agent,algo),protocol['single'])
        write_csv(out/f'{mode}_{budget}_detailed.csv',rows)
        summary=summarize(rows);summary.update(algorithm=algo,seed=seed,mode=mode,episodes=budget)
        summaries.append(summary);histories.append(dict(mode=mode,episodes=budget,training=training))
        print(json.dumps(dict(algorithm=algo,seed=seed,mode=mode,budget=budget,
                              success=summary['success_mean'])),flush=True)
    write_csv(out/'generalization_summary.csv',summaries)
    (out/'training_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    metadata=dict(algorithm=algo,seed=seed,target_graph=str(graph),source_model=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        target_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
        encoder_sha256=hashlib.sha256(encoder.read_bytes()).hexdigest(),
        test_protocol_sha256=hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
        fixed_test_seed=33003,learning_rate=.0003,embedding_metadata=embedding,
        budgets=[0,10,50,100],paired_mode_initial_rng=True,
        source_comparison='initial 1,000-episode Office configuration; not the validation-selected follow-up',
        numerical_kernel='same separately verified batched DQN/PPO kernels as review follow-up',
        started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-clock,python=sys.version,torch=torch.__version__,
        source_checksums={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ROOT/'environment/graph_env.py',ROOT/'evaluation/common_protocol.py',
                      ROOT/f'scripts/experiments/review_batched_{algo}.py']})
    (out/'complete.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


def run_job(algo,seed):
    out=BASE/f'seed_{seed}'/algo
    if (out/'complete.json').exists():return dict(algorithm=algo,seed=seed,resumed=True,returncode=0)
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['PYTHONIOENCODING']='utf-8'
    with (out/'execution.log').open('w',encoding='utf-8') as f:
        result=subprocess.run([sys.executable,'-u',str(Path(__file__)),
            '--worker','--algo',algo,'--seed',str(seed)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    return dict(algorithm=algo,seed=seed,returncode=result.returncode)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--algo',choices=['dqn','ppo'])
    p.add_argument('--seed',type=int);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if a.worker:return worker(a.algo,a.seed)
    BASE.mkdir(parents=True,exist_ok=True)
    manifest=dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Replacement of invalid legacy Clinic width-based transfer interpretation',
        methods=['zero_shot','fine_tuning','from_scratch'],algorithms=['dqn','ppo'],
        training_seeds=list(range(42,52)),budgets=[0,10,50,100],test_seed=33003,
        same_frozen_protocol_across_modes_and_seeds=True,statistical_tests='descriptive seed-level intervals only')
    (BASE/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_job,algo,seed) for seed in range(42,52) for algo in ('dqn','ppo')]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result());print(json.dumps(dict(done=len(results),total=20,**results[-1])),flush=True)
    (BASE/'batch_report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if any(r['returncode'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
