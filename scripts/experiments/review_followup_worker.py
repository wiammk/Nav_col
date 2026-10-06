"""Isolated, reproducible post-review runs; preserves the original experiment files.

Use Python 3.11 with the installed CPU PyTorch runtime. Raw node features are
zero-padded to 64 dimensions to keep neural network sizes and initializations
identical to the frozen-GCN condition. Q agents never consume these features.
"""
import argparse
import csv
import hashlib
import json
import logging
import pickle
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from environment.graph_env import GraphEnv, MultiRobotEnv
from evaluation.common_protocol import (generate_protocol, add_dynamic_disruptions,
    evaluate_single_policy, evaluate_multi_policies, load_protocol, save_protocol, summarize)
from evaluation.evaluate_review_variant import qtable_action, shortest_path_action
from federated.client import make_client
from federated.server import FedAvgServer
from models.gcn_model import GraphDataConverter, graph_fingerprint


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def factory(graph_path, n_robots=1, max_steps=200, seed=None, representation='zero', **kwargs):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    nodes = sorted(graph)
    if representation == 'gcn':
        matrix = np.load(Path(graph_path).with_name('gcn_embeddings.npz'))['embeddings']
    elif representation == 'raw':
        matrix = np.zeros((len(nodes), 64), dtype=np.float32)
        matrix[:, :16] = GraphDataConverter().node_features(graph).numpy()
    else:
        matrix = np.zeros((len(nodes), 64), dtype=np.float32)
    embeddings = dict(zip(nodes, matrix))
    if n_robots == 1:
        env = GraphEnv(graph, embeddings, max_steps=max_steps, seed=seed)
    else:
        env = MultiRobotEnv(graph, embeddings, n_robots=n_robots, max_steps=max_steps, seed=seed)
    env.graph_fingerprint = graph_fingerprint(graph)
    env.embedding_metadata = {'representation': representation,
        'checksum': hashlib.sha256(matrix.tobytes()).hexdigest()}
    return env, graph, embeddings


def protocol_paths(graph_path, robots):
    building = graph_path.parents[2].name
    if building == 'Office_Building':
        d = ROOT / 'runs/Office_Building/experiments/seed_42' / f'robots_{robots}/evaluation'
        return [d / 'fixed_test_protocol.json', d / 'dynamic_test_protocol.json']
    d = ROOT / 'runs' / building / 'review_followup/protocols'
    paths = [d / 'fixed_test_protocol.json', d / 'dynamic_test_protocol.json']
    if not paths[0].exists():
        with graph_path.open('rb') as f:
            graph = pickle.load(f)
        p = generate_protocol(graph, seed=33003, robot_counts=(3, 5, 10))
        save_protocol(p, paths[0])
        save_protocol(add_dynamic_disruptions(graph, p, seed=44004), paths[1])
    return paths


def evaluate(env, fns, protocol):
    if hasattr(env, 'envs'):
        return evaluate_multi_policies(env, fns, protocol['multi'])
    rows = evaluate_single_policy(env, fns[0], protocol['single'])
    for row in rows:
        row.update(robot_id=0, robots=1)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--algo', choices=['dqn','ppo','qlearning'], default='qlearning')
    p.add_argument('--architecture', choices=['fedavg','local','centralized'], default='fedavg')
    p.add_argument('--robots', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--representation', choices=['gcn','raw','zero'], default='gcn')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ppo-epochs', type=int, default=10)
    p.add_argument('--entropy', type=float, default=.01)
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--spatial', action='store_true')
    p.add_argument('--hotspots', action='store_true')
    p.add_argument('--model', type=Path)
    p.add_argument('--batched-dqn', action='store_true')
    p.add_argument('--batched-ppo', action='store_true')
    p.add_argument('--pilot-kernel', action='store_true')
    a = p.parse_args()
    if a.batched_dqn and a.algo=='ppo':
        a.batched_ppo=True
    a.graph = a.graph.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    done = a.output / 'complete.json'
    if done.exists() and (not a.hotspots or json.loads(done.read_text()).get('instrumentation_version')==2):
        print('Already complete:', a.output, flush=True)
        return
    logging.basicConfig(level=logging.WARNING)
    if a.batched_dqn:
        if a.pilot_kernel:
            from scripts.experiments.review_pilot_dqn_kernel import install
        else:
            from scripts.experiments.review_batched_dqn import install
        install()
    if a.batched_ppo:
        from scripts.experiments.review_batched_ppo import install
        install()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    env, graph, _ = factory(a.graph, a.robots, seed=a.seed, representation=a.representation)
    envs = env.envs if hasattr(env,'envs') else [env]
    start = time.perf_counter()
    metadata = vars(a).copy()
    metadata = {k: str(v) if isinstance(v, Path) else v for k,v in metadata.items()}
    metadata.update(graph_fingerprint=env.graph_fingerprint,
        embedding=env.embedding_metadata, python=sys.version, torch=torch.__version__,
        source_checksums={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),ROOT/'federated/client.py',ROOT/'federated/server.py',
                ROOT/'federated/multi_robot_training.py',ROOT/'environment/graph_env.py',
                ROOT/'scripts/experiments/review_batched_dqn.py',ROOT/'scripts/experiments/review_batched_ppo.py']},
        started_at_utc=datetime.now(timezone.utc).isoformat())
    if a.hotspots:
        metadata['instrumentation_version']=2
    (a.output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if a.hotspots:
        if not a.model or not hasattr(env, 'envs'):
            raise ValueError('Hotspots require a saved Q-table and K > 1')
        with a.model.open('rb') as f:
            weights = pickle.load(f)
        counts = {n: dict(node=n, proposals=0, same_destination=0,
            occupied_destination=0, edge_swap=0, deadlocks=0) for n in graph}
        original_step = env.step
        def tracked_step(actions):
            dest = {}
            for i, e in enumerate(env.envs):
                if not e._terminated and not e._truncated:
                    neighbors = e.neighbors_map[e.current_node]
                    if 0 <= actions[i] < len(neighbors):
                        dest[i] = neighbors[actions[i]]
                        counts[dest[i]]['proposals'] += 1
            result = original_step(actions)
            for i, info in enumerate(result[3]):
                if info.get('collision') and i in dest:
                    reason = info['collision_reason']
                    counts[dest[i]][reason] += 1
                e=env.envs[i]
                if not info.get('already_done',False) and result[2][i] and e.current_node!=e.target_node and e._stuck_count>=e.rcfg['stuck_limit']:
                    counts[e.current_node]['deadlocks']+=1
            return result
        env.step = tracked_step
        fns = [lambda obs, e=e: qtable_action(weights,e,obs) for e in envs]
    elif a.architecture in ['local', 'centralized']:
        if a.algo != 'qlearning':
            raise ValueError('Additional-building baselines use Q-learning')
        if a.architecture == 'local':
            from scripts.training import train_qlearning as m
            m.make_env = factory
            clients = m.train_qlearning_multi(str(a.graph), n_robots=a.robots,
                episodes=a.rounds*a.episodes, save_dir=str(a.output/'models'),
                history_path=str(a.output/'training.csv'), seed=a.seed,
                heterogeneity='spatial' if a.spatial else 'iid')
            fns = [c.greedy_action for c in clients]
            envs = [c.env for c in clients]
            # The returned clients reference the training environment.
            # Evaluation uses a fresh environment with the same node/action map.
            fns = [lambda obs,e=e,c=c: qtable_action(c.Q,e,obs) for e,c in zip(env.envs,clients)]
        else:
            from scripts.training import train_centralized_qlearning as m
            m.make_env = factory
            agent = m.train_centralized_qlearning(str(a.graph), n_robots=a.robots,
                episodes=a.rounds*a.episodes, save_dir=str(a.output/'models'),
                history_path=str(a.output/'training.csv'), seed=a.seed)
            fns = [lambda obs,e=e: qtable_action(agent.Q,e,obs) for e in envs]
    else:
        kwargs = {'seed': a.seed} if a.algo == 'qlearning' else {
            'emb_dim': 64, 'hidden_dim':128, 'lr':a.lr}
        if a.algo == 'dqn':
            kwargs.update(batch_size=64, buffer_size=20000, tau=.005)
        elif a.algo == 'ppo':
            kwargs.update(n_epochs=a.ppo_epochs, entropy_coef=a.entropy,
                rollout_len=256, minibatch_size=64, clip_eps=.2, max_grad_norm=.5)
        clients = [make_client(a.algo,e,i,**{**kwargs,**({'seed':a.seed+i} if a.algo=='qlearning' else {})}) for i,e in enumerate(envs)]
        if a.spatial:
            nodes = sorted(graph, key=lambda n:(float(graph.nodes[n].get('x_norm',0)),float(graph.nodes[n].get('y_norm',0)),str(n)))
            for i,partition in enumerate(np.array_split(np.asarray(nodes,dtype=object),a.robots)):
                clients[i].configure_data_profile(partition.tolist(), partition.tolist(),f'spatial_{i}')
        server = FedAvgServer(a.output/'models')
        rng = np.random.default_rng(22002)
        validation = [tuple(rng.choice(envs[0].nodes,size=2,replace=False)) for _ in range(50)]
        server.run(clients, n_rounds=a.rounds, episodes_per_round=a.episodes,
            save_every=5, evaluation_scenarios=validation,
            multi_env=env if hasattr(env,'envs') else None)
        server.save_history()
        for c in clients:
            c.set_weights(server.global_weights)
        fns = [c.greedy_action for c in clients]
    if not a.hotspots:
        val = generate_protocol(graph, seed=22002, robot_counts=(a.robots,) if a.robots>1 else ())
        rows = evaluate(env,fns,val)
        write_csv(a.output/'validation_detailed.csv',rows)
        write_csv(a.output/'validation_summary.csv',[summarize(rows)])
    if not a.validation_only:
        for path in protocol_paths(a.graph,a.robots):
            protocol = load_protocol(path)
            regime = protocol.get('regime','static')
            rows = evaluate(env,fns,protocol)
            write_csv(a.output/f'{regime}_detailed.csv',rows)
            write_csv(a.output/f'{regime}_summary.csv',[summarize(rows)])
        if a.hotspots:
            write_csv(a.output/'hotspots.csv',counts.values())
        elif a.algo in ('dqn','ppo') and a.rounds==40:
            checkpoint=torch.load(a.output/'models/global_model_round_020.pt',map_location='cpu',weights_only=False)
            for c in clients:
                c.set_weights(checkpoint)
            for path in protocol_paths(a.graph,a.robots):
                protocol=load_protocol(path);regime=protocol.get('regime','static')
                rows=evaluate(env,fns,protocol)
                write_csv(a.output/'budget_1000'/f'{regime}_detailed.csv',rows)
                write_csv(a.output/'budget_1000'/f'{regime}_summary.csv',[summarize(rows)])
    metadata.update(elapsed_seconds=time.perf_counter()-start,
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    done.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'complete':str(a.output),'seconds':metadata['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
