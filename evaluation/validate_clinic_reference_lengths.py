"""Independently check initial reference distances in completed Clinic runs.

Does not call the environment's distance cache or the protocol graph filter.
Dynamic references deliberately exclude closures introduced after reset.
"""
import csv
import hashlib
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/Clinic_Architectural_width_corrected/review_followup'
GRAPH = BASE.parent/'data/processed/graph.gpickle'


def references(graph, scenarios):
    result = {}
    for scenario in scenarios:
        if scenario['robots'] not in (3, 5, 10):
            continue
        legal = graph.copy()
        legal.remove_nodes_from(scenario.get('blocked_nodes', []))
        legal.remove_edges_from(scenario.get('closed_edges', []))
        legal.remove_edges_from([
            (u, v) for u, v, attributes in legal.edges(data=True)
            if not attributes.get('passable', True)
            or float(attributes.get('width', 1.0)) < .6
        ])
        for robot, (start, target) in enumerate(zip(scenario['starts'], scenario['targets'])):
            key = scenario['robots'], scenario['scenario_id'], robot
            assert key not in result
            result[key] = nx.shortest_path_length(legal, start, target, weight='weight')
    assert len(result) == 200*(3+5+10)
    return result


def main():
    manifest = json.loads((BASE/'graph_correction_manifest.json').read_text())
    assert hashlib.sha256(GRAPH.read_bytes()).hexdigest() == manifest['graph_file_sha256']
    with GRAPH.open('rb') as handle:
        graph = pickle.load(handle)
    tables, hashes = {}, {}
    for regime, name in [('static', 'fixed_test_protocol.json'),
                         ('dynamic_edge_closure', 'dynamic_test_protocol.json')]:
        path = BASE/'protocols'/name
        protocol = json.loads(path.read_text())
        tables[regime] = references(graph, protocol['multi'])
        hashes[regime] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert tables['static'] == tables['dynamic_edge_closure']
    checked, runs = 0, []
    for method in ('local', 'centralized', 'fedavg'):
        for marker in sorted((BASE/method).rglob('complete.json')):
            metadata = json.loads(marker.read_text())
            assert metadata['graph_fingerprint'] == manifest['graph_fingerprint']
            robots = metadata['robots']
            for regime, table in tables.items():
                with (marker.parent/f'{regime}_detailed.csv').open(newline='', encoding='utf-8') as handle:
                    rows = list(csv.DictReader(handle))
                keys = {(robots, int(r['scenario_id']), int(r['robot_id'])) for r in rows}
                assert len(rows) == len(keys) == 200*robots
                assert keys == {key for key in table if key[0] == robots}
                for row in rows:
                    key = robots, int(row['scenario_id']), int(row['robot_id'])
                    assert math.isclose(float(row['shortest_path_length']), table[key],
                                        rel_tol=1e-10, abs_tol=1e-9), (marker.parent, regime, key)
                checked += len(rows)
            runs.append(str(marker.parent.relative_to(ROOT)))
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  completed_runs=len(runs), expected_runs=90, complete=len(runs)==90,
                  distinct_robot_scenario_references_per_regime=3600,
                  detailed_reference_rows_checked=checked,
                  initial_reference_equal_across_paired_regimes=True,
                  permanent_width_and_passability_filtered=True,
                  protocol_sha256=hashes, graph_sha256=manifest['graph_file_sha256'],
                  passed=True, runs=runs)
    (BASE/'reference_length_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'runs'}))


if __name__ == '__main__':
    main()
