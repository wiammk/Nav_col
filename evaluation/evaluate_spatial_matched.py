"""Post-hoc control: evaluate saved spatial policies on their own task regions.

No training or parameter selection occurs. Protocols are frozen before scores
are read. The original whole-building spatial test remains unchanged.
"""
import csv
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environment.graph_env import MultiRobotEnv
from evaluation.common_protocol import (add_dynamic_disruptions, evaluate_multi_policies,
                                         load_protocol, save_protocol, summarize)
from evaluation.evaluate_review_variant import qtable_action
from evaluation.compare_qlearning import load_qtable, qtable_compatibility_error

BASE = ROOT/'runs/Office_Building/review_followup'
OUT = BASE/'spatial_matched'


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    graph_path = ROOT/'runs/Office_Building/data/processed/graph.gpickle'
    with graph_path.open('rb') as handle:
        graph = pickle.load(handle)
    nodes = sorted(graph, key=lambda n:(float(graph.nodes[n].get('x_norm',0)),
                                        float(graph.nodes[n].get('y_norm',0)),str(n)))
    partitions = [p.tolist() for p in np.array_split(np.asarray(nodes,dtype=object),5)]
    rng = np.random.default_rng(33003)
    scenarios = []
    for i in range(200):
        pairs = [rng.choice(p,size=2,replace=False).tolist() for p in partitions]
        scenarios.append(dict(scenario_id=i,robots=5,starts=[p[0] for p in pairs],
                              targets=[p[1] for p in pairs]))
    static = dict(seed=33003,regime='static',single=[],multi=scenarios,
                  task_distribution='start and target inside each robot training partition')
    dynamic = add_dynamic_disruptions(graph,static,seed=44004)
    protocol_paths = [OUT/'protocols/fixed_test_protocol.json',OUT/'protocols/dynamic_test_protocol.json']
    for protocol,path in zip((static,dynamic),protocol_paths):
        if path.exists():
            assert load_protocol(path) == protocol, 'Previously frozen protocol changed'
        else:
            save_protocol(protocol,path)
    protocol_hashes = {p.name:digest(p) for p in protocol_paths}
    before = dict(declared_before_evaluation_utc=datetime.now(timezone.utc).isoformat(),
                  purpose='control for whole-building task shift in the spatial non-IID analysis',
                  post_hoc=True,training_seeds=list(range(42,52)),robots=5,
                  partition_sizes=[len(p) for p in partitions],partitions=partitions,
                  test_seed=33003,dynamic_seed=44004,protocol_hashes=protocol_hashes,
                  graph_sha256=digest(graph_path),
                  eligible_closures=sum(bool(s.get('dynamic_closed_edges')) for s in dynamic['multi']),
                  holm_family='post_review_spatial_matched_fedavg_vs_local',holm_family_size=4)
    manifest = OUT/'experiment_manifest.json'
    if not manifest.exists():
        manifest.write_text(json.dumps(before,indent=2),encoding='utf-8')
    else:
        previous = json.loads(manifest.read_text())
        assert previous['graph_sha256'] == before['graph_sha256']
        assert previous['protocol_hashes'] == protocol_hashes
    for method in ('local','fedavg'):
        for seed in range(42,52):
            saved = BASE/f'spatial/{method}/seed_{seed}/robots_5'
            metadata = json.loads((saved/'complete.json').read_text())
            assert metadata['spatial'] and metadata['robots'] == 5 and metadata['seed'] == seed
            target = OUT/f'{method}/seed_{seed}/robots_5'
            if (target/'complete.json').exists():
                continue
            target.mkdir(parents=True,exist_ok=True)
            start = time.perf_counter()
            started_at = datetime.now(timezone.utc).isoformat()
            embeddings = {n:np.zeros(64,dtype=np.float32) for n in graph}
            env = MultiRobotEnv(graph,embeddings,n_robots=5,max_steps=200,seed=seed)
            model_paths = [saved/'models/global_qtable_final.pkl']*5 if method=='fedavg' else [saved/f'models/robot_{i}.pkl' for i in range(5)]
            tables = [load_qtable(p) for p in model_paths]
            for table,e in zip(tables,env.envs):
                assert not qtable_compatibility_error(table,e,str(saved))
            policies = [lambda obs,t=t,e=e:qtable_action(t,e,obs) for t,e in zip(tables,env.envs)]
            for protocol in (static,dynamic):
                regime = protocol['regime']
                detailed = evaluate_multi_policies(env,policies,protocol['multi'])
                assert len(detailed) == 1000
                write_csv(target/f'{regime}_detailed.csv',detailed)
                write_csv(target/f'{regime}_summary.csv',[summarize(detailed)])
            metadata.update(evaluation_only=True,training_source=str(saved.relative_to(ROOT)),
                            output=str(target),started_at_utc=started_at,
                            model_checksums={str(p.relative_to(ROOT)):digest(p) for p in model_paths},
                            matched_protocol_hashes=protocol_hashes,graph_sha256=digest(graph_path),
                            evaluation_source_sha256=digest(Path(__file__)),
                            elapsed_seconds=time.perf_counter()-start)
            metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            (target/'complete.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            print(json.dumps(dict(method=method,seed=seed,seconds=metadata['elapsed_seconds'])),flush=True)
    (OUT/'complete.json').write_text(json.dumps(dict(evaluations=20,manifest_sha256=digest(manifest),
                                                     completed_at_utc=datetime.now(timezone.utc).isoformat()),indent=2))


if __name__ == '__main__':
    main()
