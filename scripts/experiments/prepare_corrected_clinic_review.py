"""Build a separate width-corrected Clinic graph and freeze its review protocol."""
import hashlib
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import networkx as nx
import numpy as np
from data.graph_builder import build_pipeline
from environment.graph_env import GraphEnv
from evaluation.common_protocol import generate_protocol, add_dynamic_disruptions, save_protocol
from models.gcn_model import GraphDataConverter, graph_fingerprint

base = ROOT / 'runs/Clinic_Architectural_width_corrected'
processed = base / 'data/processed'
graph = build_pipeline(processed/'nodes.csv', processed/'edges.csv', processed/'graph.gpickle')
with (ROOT/'runs/Clinic_Architectural/data/processed/graph.gpickle').open('rb') as f:
    old = pickle.load(f)
assert set(graph) == set(old), 'Unexpected node identity change'
assert set(map(frozenset, graph.edges())) == set(map(frozenset,old.edges())), 'Unexpected topology change'
assert all(graph.nodes[n] == old.nodes[n] for n in graph), 'Unexpected node attribute change'
changed = []
for u,v,d in graph.edges(data=True):
    before = old[u][v]
    assert {k:v for k,v in d.items() if k!='width'} == {k:v for k,v in before.items() if k!='width'}
    if d['width'] != before['width']:
        changed.append({'from':u,'to':v,'old_width':before['width'],'corrected_width':d['width']})
features_equal = np.array_equal(GraphDataConverter().node_features(old).numpy(),
                              GraphDataConverter().node_features(graph).numpy())
assert features_equal, 'Unexpected neural input change'
legal = graph.copy()
legal.remove_edges_from([(u,v) for u,v,d in graph.edges(data=True)
                        if not d.get('passable',True) or float(d.get('width',1.)) < .6])
assert nx.is_connected(legal), 'Corrected protocol requires a connected traversable graph'
# Office has no permanently invalid edges: filtering its reference graph is an identity.
with (ROOT/'runs/Office_Building/data/processed/graph.gpickle').open('rb') as f:
    office = pickle.load(f)
env = GraphEnv(office,{n:np.zeros(64,dtype=np.float32) for n in office})
assert env._sp == dict(nx.all_pairs_dijkstra_path_length(office,weight='weight'))
protocol = generate_protocol(graph,seed=33003,robot_counts=(3,5,10))
dynamic = add_dynamic_disruptions(graph,protocol,seed=44004)
directory = base/'review_followup/protocols'
save_protocol(protocol,directory/'fixed_test_protocol.json')
save_protocol(dynamic,directory/'dynamic_test_protocol.json')
report = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    reason='Legacy substring width matching selected Trim Width (0.076 m) instead of IfcDoor.OverallWidth; cached references also failed to filter permanently unavailable edges.',
    graph_fingerprint=graph_fingerprint(graph),legacy_graph_fingerprint=graph_fingerprint(old),
    graph_file_sha256=hashlib.sha256((processed/'graph.gpickle').read_bytes()).hexdigest(),
    nodes=graph.number_of_nodes(),edges=graph.number_of_edges(),changed_widths=changed,
    topology_unchanged=True,node_attributes_unchanged=True,gcn_input_features_unchanged=bool(features_equal),
    traversable_component_sizes=sorted(map(len,nx.connected_components(legal)),reverse=True),
    narrow_edges=sum(float(d.get('width',1.))<.6 for u,v,d in graph.edges(data=True)),
    door_widths=[dict(width=w,count=c) for w,c in sorted(Counter(float(d['width']) for u,v,d in graph.edges(data=True) if d.get('type')=='door').items())],
    office_reference_distances_unchanged=True,legacy_results_preserved=True,
    dynamic_eligible={str(k):sum(bool(s.get('dynamic_closed_edges')) for s in dynamic['multi'] if s['robots']==k) for k in (3,5,10)},
    source_checksums={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'data/ifc_parser.py',ROOT/'environment/graph_env.py',ROOT/'evaluation/common_protocol.py']},
    width_definition_source='https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifcdoor.htm')
(base/'review_followup/graph_correction_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
plan_path = ROOT/'runs/Office_Building/review_followup/experiment_plan.json'
plan = json.loads(plan_path.read_text())
plan['additional_building']['graph_correction'] = {k:report[k] for k in
    ('reason','recorded_at_utc','graph_fingerprint','topology_unchanged','legacy_results_preserved')}
plan['additional_building']['graph_path'] = str((processed/'graph.gpickle').relative_to(ROOT))
plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('changed_widths','door_widths','source_checksums')}))
