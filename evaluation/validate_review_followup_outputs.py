"""Audit saved post-review outputs without running or changing training jobs.

This uses the standard library, independently recomputes detailed-file means,
seed-level intervals, and Holm adjustments, and records unfinished families.
"""
import csv
import json
import itertools
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICE = ROOT / 'runs/Office_Building/review_followup'
CLINIC = ROOT / 'runs/Clinic_Architectural_width_corrected/review_followup'
SEEDS = range(42, 52)
REGIMES = ('static', 'dynamic_edge_closure')
METRICS = ('success', 'spl', 'collisions', 'deadlock', 'steps')
T95_DF9 = 2.262  # Same rounded critical value as the experiment aggregator.


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def one_row(path):
    rows = read_csv(path)
    assert len(rows) == 1, str(path)
    return rows[0]


def close(first, second, context):
    assert math.isclose(float(first), float(second), rel_tol=1e-10,
                        abs_tol=1e-10), context


def inspect_run(folder):
    metadata = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
    k = metadata['robots']
    budget = metadata['rounds'] * metadata['episodes']
    assert metadata['finished_at_utc'] >= metadata['started_at_utc']
    subsets = [folder]
    if metadata['validation_only']:
        subsets = []
    elif metadata['algo'] in ('dqn', 'ppo') and metadata['rounds'] == 40:
        subsets.append(folder / 'budget_1000')
    inspected_rows = 0
    for subset in subsets:
        for regime in REGIMES:
            rows = read_csv(subset / f'{regime}_detailed.csv')
            summary = one_row(subset / f'{regime}_summary.csv')
            assert len(rows) == 200 * k, str(subset)
            pairs = {(int(r['scenario_id']), int(r['robot_id'])) for r in rows}
            assert len(pairs) == len(rows), str(subset)
            assert set(Counter(int(r['scenario_id']) for r in rows).values()) == {k}
            assert {int(r['robot_id']) for r in rows} == set(range(k))
            for r in rows:
                success = float(r['success'])
                assert success in (0., 1.)
                length = float(r['path_length'])
                reference = float(r['shortest_path_length'])
                spl = reference / max(reference, length) if success and reference > 0 else success
                close(r['spl'], spl, (subset, regime, r['scenario_id'], r['robot_id'], 'SPL'))
            for metric in METRICS:
                close(summary[f'{metric}_mean'], statistics.mean(float(r[metric]) for r in rows),
                      (subset, regime, metric))
            inspected_rows += len(rows)
    if not metadata['hotspots'] and not metadata.get('evaluation_only',False):
        if metadata['architecture'] == 'fedavg':
            history = read_csv(folder / 'models/fl_history.csv')
            assert len(history) == metadata['rounds']
            assert sum(int(r['n_episodes']) for r in history) == budget * k
            extension = 'pkl' if metadata['algo'] == 'qlearning' else 'pt'
            stem = 'global_qtable' if metadata['algo'] == 'qlearning' else 'global_model'
            assert (folder / f'models/{stem}_final.{extension}').is_file()
        else:
            history = read_csv(folder / 'training.csv')
            assert len(history) == budget
            if metadata['architecture'] == 'local':
                training = json.loads((folder / 'models/train_meta.json').read_text())
                assert training['episodes_per_robot'] == budget
                assert training['total_trajectories'] == budget * k
                assert len(training['models']) == k
    return dict(folder=str(folder.relative_to(ROOT)), seed=metadata['seed'],
                robots=k, architecture=metadata['architecture'], algorithm=metadata['algo'],
                budget=budget, rows_checked=inspected_rows,
                graph_fingerprint=metadata['graph_fingerprint'])


def seed_summaries(row):
    building, method = row['building'], row['method']
    k, budget = int(row['robots']), int(row['budget'])
    regime, representation = row['regime'], row['representation']
    if method == 'prioritized_planning':
        return [one_row(OFFICE / f'prioritized/robots_{k}/{regime}_summary.csv')] * 10
    if building == 'Clinic':
        return [one_row(CLINIC / f'{method}/seed_{s}/robots_{k}/{regime}_summary.csv') for s in SEEDS]
    if '_spatial' in method:
        architecture = method.split('_spatial',1)[0]
        folder = 'spatial_matched' if method.endswith('_matched') else 'spatial'
        return [one_row(OFFICE / f'{folder}/{architecture}/seed_{s}/robots_5/{regime}_summary.csv') for s in SEEDS]
    if representation in ('gcn', 'raw'):
        algo = method.removeprefix('fedavg_')
        subfolder = 'budget_1000/' if budget == 1000 else ''
        return [one_row(OFFICE / f'neural/{algo}/{representation}/seed_{s}/robots_{k}/{subfolder}{regime}_summary.csv') for s in SEEDS]
    name = 'comparison_summary.csv' if regime == 'static' else 'dynamic_comparison_summary.csv'
    return [next(r for r in read_csv(ROOT / f'runs/Office_Building/experiments/seed_{s}/robots_{k}/evaluation/{name}')
                 if r['method'] == method) for s in SEEDS]


def inspect_test(row):
    values = []
    for side in ('a', 'b'):
        descriptor = dict(building=row['building'], robots=row['robots'], regime=row['regime'],
                          method=row[f'method_{side}'], budget=row[f'budget_{side}'],
                          representation=row[f'representation_{side}'])
        values.append([float(r[f"{row['metric']}_mean"]) for r in seed_summaries(descriptor)])
    differences = [a-b for a,b in zip(*values)]
    close(row['mean_difference'], statistics.mean(differences), (row['statistical_family'], 'effect'))
    close(row['difference_ci95_student'], T95_DF9 * statistics.stdev(differences) / math.sqrt(10),
          (row['statistical_family'], 'effect interval'))
    nonzero = [x for x in differences if not math.isclose(x, 0., abs_tol=1e-8)]
    observed = abs(math.fsum(differences))
    if not nonzero or observed == 0:
        p = 1.
    else:
        extreme = sum(abs(math.fsum(sign*x for sign,x in zip(signs, nonzero))) >= observed-1e-12
                      for signs in itertools.product((-1.,1.), repeat=len(nonzero)))
        p = extreme / (2**len(nonzero))
    close(row['p_value'], p, (row['statistical_family'], row['metric'], 'paired permutation'))


def main():
    completed = []
    for base in (OFFICE / 'pilot', OFFICE / 'neural', OFFICE / 'spatial',
                 OFFICE / 'hotspots', CLINIC):
        for marker in sorted(base.rglob('complete.json')):
            completed.append(inspect_run(marker.parent))
    for method in ('local','fedavg'):
        for marker in sorted((OFFICE/f'spatial_matched/{method}').rglob('complete.json')):
            completed.append(inspect_run(marker.parent))
    aggregate = read_csv(OFFICE / 'aggregate_followup_results.csv')
    for row in aggregate:
        seed_rows = seed_summaries(row)
        for metric in METRICS:
            values = [float(r[f'{metric}_mean']) for r in seed_rows]
            mean, std = statistics.mean(values), statistics.stdev(values)
            close(row[f'{metric}_mean'], mean, (row['method'], metric, 'mean'))
            close(row[f'{metric}_std'], std, (row['method'], metric, 'std'))
            close(row[f'{metric}_ci95'], T95_DF9 * std / math.sqrt(10),
                  (row['method'], metric, 'interval'))
    tests = read_csv(OFFICE / 'followup_paired_tests.csv')
    families = defaultdict(list)
    for row in tests:
        inspect_test(row)
        families[row['statistical_family']].append(row)
    plan = json.loads((OFFICE / 'experiment_plan.json').read_text())
    family_report = []
    for name, rows in families.items():
        previous = 0.
        for rank, row in enumerate(sorted(rows, key=lambda r: float(r['p_value']))):
            previous = min(1., max(previous, (len(rows) - rank) * float(row['p_value'])))
            close(row['p_value_holm'], previous, (name, 'Holm'))
            assert int(row['holm_family_size']) == len(rows)
        expected = plan['holm_families'][name]
        family_report.append(dict(family=name, present=len(rows), expected=expected,
                                  complete=len(rows) == expected))
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  completed_runs=len(completed), detailed_rows_checked=sum(r['rows_checked'] for r in completed),
                  aggregate_rows_checked=len(aggregate), holm_tests_checked=len(tests),
                  families=family_report, runs=completed, passed=True)
    (OFFICE / 'output_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'runs'}))


if __name__ == '__main__':
    main()
