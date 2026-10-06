"""Continue long review jobs and aggregate only when complete; resume safely."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OFFICE=ROOT/'runs/Office_Building/review_followup'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['neural','hotspots'],required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--algo',choices=['dqn','ppo'])
    a=p.parse_args()
    if a.algo:
        if a.phase!='neural':p.error('--algo requires --phase neural')
        prerequisites=[OFFICE/f'pilot/{a.algo}/lr_{lr}/complete.json' for lr in (1e-4,3e-4,1e-3)]
        print(json.dumps({'waiting_for':[str(x) for x in prerequisites],'next_algorithm':a.algo}),flush=True)
        while not all(x.exists() for x in prerequisites):time.sleep(30)
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/experiments/run_review_followup.py'),
            '--phase',a.phase,'--algo',a.algo,'--workers',str(a.workers)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'evaluation/aggregate_review_followup.py')],cwd=ROOT,check=True)
        return
    prerequisite=OFFICE/('pilot_batch_report.json' if a.phase=='neural' else 'hotspots_batch_report.json')
    print(json.dumps({'waiting_for':str(prerequisite),'next_phase':a.phase}),flush=True)
    while not prerequisite.exists():
        time.sleep(30)
    report=json.loads(prerequisite.read_text())
    if any(r['returncode'] for r in report['results']):
        raise RuntimeError('Prerequisite phase failed; inspect saved logs')
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/experiments/run_review_followup.py'),
        '--phase',a.phase,'--workers',str(a.workers)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'evaluation/aggregate_review_followup.py')],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
