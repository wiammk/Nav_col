"""Audit and aggregate complete ten-seed width-corrected Clinic transfer runs."""
import csv
import hashlib
import json
import math
import pickle
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import networkx as nx
from evaluation.aggregate_professor_results import mean_std_ci95
from evaluation.common_protocol import _available_graph

BASE=ROOT/'runs/Clinic_Architectural_width_corrected/review_transfer'
GRAPH=ROOT/'runs/Clinic_Architectural_width_corrected/data/processed/graph.gpickle'
PROTOCOL=GRAPH.parents[2]/'review_followup/protocols/fixed_test_protocol.json'
METRICS=('success','spl','steps','reward','cumulative_risk')


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    if not BASE.exists():return
    protocol=json.loads(PROTOCOL.read_text())
    with GRAPH.open('rb') as f:graph=pickle.load(f)
    reference={}
    for scenario in protocol['single']:
        legal=_available_graph(graph,scenario.get('blocked_nodes',()),scenario.get('closed_edges',()))
        reference[scenario['scenario_id']]=nx.shortest_path_length(legal,scenario['start'],scenario['target'],weight='weight')
    expected_modes={('zero_shot',0),('fine_tuning',10),('from_scratch',10),
                    ('fine_tuning',50),('from_scratch',50),('fine_tuning',100),('from_scratch',100)}
    complete=[];values={};checked_rows=0
    for algo in ('dqn','ppo'):
        for seed in range(42,52):
            folder=BASE/f'seed_{seed}'/algo
            if not (folder/'complete.json').exists():continue
            meta=json.loads((folder/'complete.json').read_text())
            assert meta['test_protocol_sha256']==hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
            assert meta['target_sha256']==hashlib.sha256(GRAPH.read_bytes()).hexdigest()
            summary=read_csv(folder/'generalization_summary.csv')
            assert {(r['mode'],int(r['episodes'])) for r in summary}==expected_modes
            history=json.loads((folder/'training_history.json').read_text())
            for h in history:
                if h['episodes']:
                    assert h['training']['n_episodes']==h['episodes']
                    assert len(h['training']['rewards'])==h['episodes']
                    assert len(h['training']['successes'])==h['episodes']
            for row in summary:
                mode,budget=row['mode'],int(row['episodes'])
                detailed=read_csv(folder/f'{mode}_{budget}_detailed.csv')
                assert len(detailed)==200 and len({int(r['scenario_id']) for r in detailed})==200
                for r in detailed:
                    ref=reference[int(r['scenario_id'])]
                    assert math.isclose(float(r['shortest_path_length']),ref,abs_tol=1e-9)
                    spl=ref/max(ref,float(r['path_length'])) if int(float(r['success'])) else 0.
                    assert math.isclose(float(r['spl']),spl,abs_tol=1e-9)
                for metric in METRICS:
                    assert math.isclose(float(row[f'{metric}_mean']),statistics.mean(float(r[metric]) for r in detailed),abs_tol=1e-10)
                values.setdefault((algo,mode,budget),[]).append(row)
                checked_rows+=len(detailed)
            complete.append(dict(algorithm=algo,seed=seed))
    aggregate=[]
    for (algo,mode,budget),rows in values.items():
        if len(rows)!=10:continue
        record=dict(algorithm=algo,mode=mode,episodes=budget,n_training_seeds=10,n_distinct_scenarios=200)
        for metric in METRICS:
            mean,std,ci=mean_std_ci95([float(r[f'{metric}_mean']) for r in rows])
            record.update({f'{metric}_mean':mean,f'{metric}_std':std,f'{metric}_ci95':ci})
        aggregate.append(record)
    if aggregate:
        with (BASE/'aggregate_transfer.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    report=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),completed_runs=len(complete),
        expected_runs=20,complete=len(complete)==20,aggregate_cells=len(aggregate),
        detailed_rows_checked=checked_rows,reference_distances_checked_independently=True,passed=True,runs=complete)
    (BASE/'output_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'}))


if __name__=='__main__':main()
