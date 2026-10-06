"""Run the declared post-review experiment batches with resumable subprocesses."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OFFICE=ROOT/'runs/Office_Building/review_followup'
CLINIC=ROOT/'runs/Clinic_Architectural_width_corrected/review_followup'


def jobs(phase,algorithm=None):
    result=[]
    graph='runs/Office_Building/data/processed/graph.gpickle'
    if phase=='pilot':
        for algo in ((algorithm,) if algorithm else ('dqn','ppo')):
            for lr in (1e-4,3e-4,1e-3):
                out=OFFICE/'pilot'/algo/f'lr_{lr}'
                result.append((out,['--graph',graph,'--algo',algo,'--seed','11001',
                    '--robots','1','--lr',str(lr),'--validation-only','--batched-dqn','--batched-ppo','--pilot-kernel']))
    elif phase=='neural':
        choices={}
        import csv
        for algo in ((algorithm,) if algorithm else ('dqn','ppo')):
            candidates=[]
            for lr in (1e-4,3e-4,1e-3):
                d=OFFICE/'pilot'/algo/f'lr_{lr}'
                if not (d/'complete.json').exists():
                    raise RuntimeError('Tuning runs must complete before selecting configurations')
                with (d/'validation_summary.csv').open() as f:
                    r=next(csv.DictReader(f))
                candidates.append((float(r['success_mean']),float(r['spl_mean']),-float(r['steps_mean']),-lr,lr))
            choices[algo]=max(candidates)[-1]
        for algo,lr in choices.items():
            (OFFICE/f'selected_hyperparameters_{algo}.json').write_text(json.dumps({algo:lr},indent=2))
        merged={}
        for algo in ('dqn','ppo'):
            selected=OFFICE/f'selected_hyperparameters_{algo}.json'
            if selected.exists():merged.update(json.loads(selected.read_text()))
        temporary=OFFICE/f'selected_hyperparameters_{algorithm or "all"}.tmp'
        temporary.write_text(json.dumps(merged,indent=2));temporary.replace(OFFICE/'selected_hyperparameters.json')
        for seed in range(42,52):
            for k in (1,3):
                for algo in choices:
                    for representation in ('gcn','raw'):
                        out=OFFICE/'neural'/algo/representation/f'seed_{seed}'/f'robots_{k}'
                        result.append((out,['--graph',graph,'--algo',algo,'--seed',str(seed),
                            '--robots',str(k),'--lr',str(choices[algo]),'--representation',representation,
                            '--rounds','40','--batched-dqn','--batched-ppo']))
    elif phase=='clinic':
        for seed in range(42,52):
            for k in (3,5,10):
                for architecture in ('fedavg','local','centralized'):
                    out=CLINIC/architecture/f'seed_{seed}'/f'robots_{k}'
                    result.append((out,['--graph','runs/Clinic_Architectural_width_corrected/data/processed/graph.gpickle',
                        '--robots',str(k),'--seed',str(seed),'--architecture',architecture,'--representation','zero']))
    elif phase=='hotspots':
        for seed in range(42,52):
            for k in (3,5,10):
                out=OFFICE/'hotspots'/f'seed_{seed}'/f'robots_{k}'
                model=ROOT/f'runs/Office_Building/experiments/seed_{seed}/robots_{k}/fedavg/qlearning/global_qtable_final.pkl'
                result.append((out,['--graph',graph,'--robots',str(k),'--seed',str(seed),
                    '--representation','zero','--hotspots','--model',str(model)]))
    elif phase=='spatial':
        for seed in range(42,52):
            for arch in ('local','fedavg'):
                out=OFFICE/'spatial'/arch/f'seed_{seed}'/'robots_5'
                result.append((out,['--graph',graph,'--robots','5','--seed',str(seed),
                    '--architecture',arch,'--representation','zero','--spatial']))
    return result


def run_job(job):
    output,args=job
    if (output/'complete.json').exists() and ('--hotspots' not in args or json.loads((output/'complete.json').read_text()).get('instrumentation_version')==2):
        return {'output':str(output),'resumed':True,'returncode':0}
    output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['PYTHONPATH']=str(ROOT);env['PYTHONIOENCODING']='utf-8'
    command=[sys.executable,'-u',str(ROOT/'scripts/experiments/review_followup_worker.py'),
        '--output',str(output),*args]
    with (output/'execution.log').open('w',encoding='utf-8') as log:
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    return {'output':str(output),'returncode':completed.returncode}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['pilot','neural','clinic','hotspots','spatial'])
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--algo',choices=['dqn','ppo'])
    a=p.parse_args()
    if a.algo and a.phase not in ('pilot','neural'):
        p.error('--algo only applies to pilot or neural phases')
    todo=jobs(a.phase,a.algo);results=[];start=time.time()
    label=a.phase+('_'+a.algo if a.algo else '')
    print(json.dumps({'phase':label,'jobs':len(todo),'workers':a.workers}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run_job,j) for j in todo]):
            r=future.result();results.append(r)
            print(json.dumps({'done':len(results),'total':len(todo),**r}),flush=True)
    report={'phase':label,'seconds':time.time()-start,'results':results}
    (OFFICE/f'{label}_batch_report.json').write_text(json.dumps(report,indent=2))
    if any(r['returncode'] for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
