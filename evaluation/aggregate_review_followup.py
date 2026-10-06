"""Aggregate completed post-review runs and declared paired comparison families."""
import csv
import json
import math
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import networkx as nx
from evaluation.aggregate_professor_results import mean_std_ci95, paired_effects, permutation_pvalue
from evaluation.aggregate_review_experiments import holm_adjust, write_csv
OFFICE=ROOT/'runs/Office_Building/review_followup'
CLINIC=ROOT/'runs/Clinic_Architectural_width_corrected/review_followup'
SEEDS=tuple(range(42,52));REGIMES=('static','dynamic_edge_closure')
METRICS=('success','spl','collisions','deadlock','steps')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def row(path):
    rows=read_csv(path)
    if len(rows)!=1:
        raise ValueError(f'Expected one summary: {path}')
    return {k:float(v) if k.endswith(('_mean','_std','_ci95')) else v for k,v in rows[0].items()}


def main():
    aggregate=[];tests=[];values={}
    def add(building,k,regime,method,rows,budget=1000,representation='indices',deterministic=False):
        if len(rows)!=len(SEEDS):
            raise ValueError('A complete ten-seed set is required')
        key=(building,k,regime,method,budget,representation)
        values[key]=rows
        out=dict(building=building,robots=k,regime=regime,method=method,budget=budget,
            representation=representation,n_training_seeds=0 if deterministic else 10,
            n_distinct_scenarios=200,deterministic_reference=int(deterministic))
        for metric in METRICS:
            mean,std,ci=mean_std_ci95([r[f'{metric}_mean'] for r in rows])
            out.update({f'{metric}_mean':mean,f'{metric}_std':std,f'{metric}_ci95':ci})
        aggregate.append(out)
        return key
    def compare(a,b,family):
        for metric in ('success','spl'):
            first=[r[f'{metric}_mean'] for r in values[a]]
            second=[r[f'{metric}_mean'] for r in values[b]]
            tests.append(dict(statistical_family=family,building=a[0],robots=a[1],regime=a[2],
                method_a=a[3],method_b=b[3],budget_a=a[4],budget_b=b[4],representation_a=a[5],representation_b=b[5],
                metric=metric,n_paired_training_seeds=10,
                test='exact paired sign-permutation across training seeds',
                **paired_effects(first,second),p_value=permutation_pvalue(first,second,seed=55001)))
    for k in (1,3,5,10):
        for regime in REGIMES:
            name='comparison_summary.csv' if regime=='static' else 'dynamic_comparison_summary.csv'
            old=[]
            for seed in SEEDS:
                rows=read_csv(ROOT/f'runs/Office_Building/experiments/seed_{seed}/robots_{k}/evaluation'/name)
                old.append({f'{m}_mean':float(next(r for r in rows if r['method']=='fedavg_qlearning')[f'{m}_mean']) for m in METRICS})
            standard=add('Office',k,regime,'fedavg_qlearning',old)
            summary=OFFICE/f'prioritized/robots_{k}/{regime}_summary.csv'
            if summary.exists() and (OFFICE/'prioritized/complete.json').exists():
                planner=add('Office',k,regime,'prioritized_planning',[row(summary)]*10,representation='full_graph',budget=0,deterministic=True)
                compare(planner,standard,'post_review_prioritized_planning')
    for k in (3,5,10):
        for regime in REGIMES:
            keys={}
            for method in ('fedavg','local','centralized'):
                paths=[CLINIC/f'{method}/seed_{seed}/robots_{k}' for seed in SEEDS]
                if all((p/'complete.json').exists() for p in paths):
                    keys[method]=add('Clinic',k,regime,method,[row(p/f'{regime}_summary.csv') for p in paths])
            if 'fedavg' in keys:
                for other in ('local','centralized'):
                    if other in keys:
                        compare(keys['fedavg'],keys[other],f'post_review_clinic_fedavg_vs_{other}')
    for regime in REGIMES:
        keys={}
        for method in ('fedavg','local'):
            paths=[OFFICE/f'spatial/{method}/seed_{seed}/robots_5' for seed in SEEDS]
            if all((p/'complete.json').exists() for p in paths):
                keys[method]=add('Office',5,regime,method+'_spatial',[row(p/f'{regime}_summary.csv') for p in paths])
        if len(keys)==2:
            compare(keys['fedavg'],keys['local'],'post_review_spatial_fedavg_vs_local')
    if (OFFICE/'spatial_matched/complete.json').exists():
        for regime in REGIMES:
            keys={}
            for method in ('fedavg','local'):
                paths=[OFFICE/f'spatial_matched/{method}/seed_{seed}/robots_5' for seed in SEEDS]
                if all((p/'complete.json').exists() for p in paths):
                    keys[method]=add('Office',5,regime,method+'_spatial_matched',[row(p/f'{regime}_summary.csv') for p in paths])
            if len(keys)==2:
                compare(keys['fedavg'],keys['local'],'post_review_spatial_matched_fedavg_vs_local')
    neural={}
    for algo in ('dqn','ppo'):
        for k in (1,3):
            for representation in ('gcn','raw'):
                paths=[OFFICE/f'neural/{algo}/{representation}/seed_{seed}/robots_{k}' for seed in SEEDS]
                if not all((p/'complete.json').exists() for p in paths):
                    continue
                for budget in (1000,2000):
                    for regime in REGIMES:
                        paths_b=[p/'budget_1000' if budget==1000 else p for p in paths]
                        key=add('Office',k,regime,'fedavg_'+algo,[row(p/f'{regime}_summary.csv') for p in paths_b],budget,representation)
                        neural[algo,k,representation,budget,regime]=key
    for algo in ('dqn','ppo'):
        for k in (1,3):
            for budget in (1000,2000):
                for regime in REGIMES:
                    a=neural.get((algo,k,'gcn',budget,regime));b=neural.get((algo,k,'raw',budget,regime))
                    if a and b:
                        compare(a,b,f'post_review_{algo}_representation')
            for representation in ('gcn','raw'):
                for regime in REGIMES:
                    a=neural.get((algo,k,representation,2000,regime));b=neural.get((algo,k,representation,1000,regime))
                    if a and b:
                        compare(a,b,f'post_review_{algo}_longer_budget')
    write_csv(OFFICE/'aggregate_followup_results.csv',aggregate)
    write_csv(OFFICE/'followup_paired_tests.csv',holm_adjust(tests))
    with (ROOT/'runs/Office_Building/data/processed/graph.gpickle').open('rb') as f:
        graph=pickle.load(f)
    bc=nx.betweenness_centrality(graph,weight='weight',normalized=True)
    stair_nodes={n for u,v,d in graph.edges(data=True) if d.get('type')=='stair' for n in (u,v)}
    hotspot_report=[]
    for k in (3,5,10):
        paths=[OFFICE/f'hotspots/seed_{seed}/robots_{k}' for seed in SEEDS]
        if not all((p/'complete.json').exists() and json.loads((p/'complete.json').read_text()).get('instrumentation_version')==2 for p in paths):
            continue
        counts={n:dict(node=n,betweenness=bc[n],degree=graph.degree(n),stair_endpoint=int(n in stair_nodes),proposals=0,conflicts=0,deadlocks=0) for n in graph}
        fidelity=[]
        for seed,path in zip(SEEDS,paths):
            for r in read_csv(path/'hotspots.csv'):
                n=int(r['node']);counts[n]['proposals']+=int(r['proposals']);counts[n]['deadlocks']+=int(r['deadlocks'])
                counts[n]['conflicts']+=sum(int(r[c]) for c in ('same_destination','occupied_destination','edge_swap'))
            for regime in REGIMES:
                old_name='comparison_detailed.csv' if regime=='static' else 'dynamic_comparison_detailed.csv'
                old=read_csv(ROOT/f'runs/Office_Building/experiments/seed_{seed}/robots_{k}/evaluation'/old_name)
                old={(r['scenario_id'],r['robot_id']):r for r in old if r['method']=='fedavg_qlearning'}
                new=read_csv(path/f'{regime}_detailed.csv');differences=0
                for r in new:
                    o=old[r['scenario_id'],r['robot_id']]
                    differences+=sum(abs(float(r[m])-float(o[m]))>1e-8 for m in ('success','spl','collisions','deadlock','path_length','steps','reward','shortest_path_length'))
                fidelity.append(dict(seed=seed,regime=regime,rows=len(new),differences=differences))
        for c in counts.values():
            c['rejection_rate']=c['conflicts']/c['proposals'] if c['proposals'] else 0.
        write_csv(OFFICE/f'hotspot_nodes_K{k}.csv',counts.values())
        selected=[c for c in counts.values() if c['proposals']]
        def spearman(field):
            def ranks(values):
                order=sorted(range(len(values)),key=lambda i:values[i]);result=np.zeros(len(values))
                i=0
                while i<len(order):
                    j=i+1
                    while j<len(order) and values[order[j]]==values[order[i]]:j+=1
                    for index in order[i:j]:result[index]=(i+j-1)/2
                    i=j
                return result
            return float(np.corrcoef(ranks([c['betweenness'] for c in selected]),ranks([c[field] for c in selected]))[0,1])
        total=sum(c['conflicts'] for c in selected);deadlocks=sum(c['deadlocks'] for c in selected)
        hotspot_report.append(dict(robots=k,pooled_regimes=list(REGIMES),fidelity=fidelity,
            total_proposals=sum(c['proposals'] for c in selected),total_conflict_involvements=total,total_deadlocks=deadlocks,
            stair_endpoint_nodes=sorted(stair_nodes),stair_endpoint_conflict_share=sum(c['conflicts'] for c in selected if c['stair_endpoint'])/total,
            stair_endpoint_deadlock_share=sum(c['deadlocks'] for c in selected if c['stair_endpoint'])/deadlocks,
            betweenness_conflict_count_spearman=spearman('conflicts'),
            betweenness_rejection_rate_spearman=spearman('rejection_rate'),
            top_conflict_nodes=sorted(selected,key=lambda c:c['conflicts'],reverse=True)[:8]))
        assert all(r['differences']==0 for r in fidelity)
    (OFFICE/'hotspot_analysis.json').write_text(json.dumps(hotspot_report,indent=2))
    print(json.dumps({'aggregate_rows':len(aggregate),'tests':len(tests),'hotspot_fleets':len(hotspot_report)}))


if __name__=='__main__':
    main()
