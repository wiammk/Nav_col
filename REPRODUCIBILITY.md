# Reproducing the recorded study

Run commands from the repository root. Install `requirements.txt` first; manuscript asset generation also needs `reportlab` and `pypdf` when the corresponding script imports them. Existing CSVs can be read without any model training. Long training batches are optional and can take hours.

## Frozen inputs

- Initial Office: `runs/Office_Building/data/processed/graph.gpickle`, SHA-256 `bbb155bc3c33c01f61d9fecbe732a08ca2e80c61268ed169ca25a2f602e17339`.
- Corrected Clinic: `runs/Clinic_Architectural_width_corrected/data/processed/graph.gpickle`, SHA-256 `e1b696035c275c3bee914d97ea22a4ff49ffec2937a7c1805c9eaa39eb50b9cc`.
- Office frozen GCN: `runs/Office_Building/data/processed/gcn_encoder.pt`, embeddings and metadata alongside it.
- Initial frozen scenarios: `runs/Office_Building/experiments/seed_*/robots_*/evaluation/*test_protocol.json`.
- Corrected Clinic scenarios: `runs/Clinic_Architectural_width_corrected/review_followup/protocols/`.
- Follow-up plans and selected rates: `runs/Office_Building/review_followup/experiment_plan.json` and `selected_hyperparameters*.json`.

Use the frozen Office graph for reproducing the recorded results: it retains the historical default door width 0.9 m. The corrected parser reads 1.0 m in the raw Office file, which changes the graph hash without changing Office action masks or reference distances. Clinic results use the corrected opening width, not the earlier frame-width interpretation. Only corrected Clinic outputs are included.

## Saved models

```sh
python scripts/publication/restore_models.py
```

This restores original model bytes and relative paths from the archives listed in `PUBLICATION_MANIFEST.json`, checking SHA-256 hashes. Different local model files are preserved and cause an explanatory error. Intermediate round 5/10/15 checkpoints are omitted; neural round 20/40 checkpoints needed for the budget comparison are included. The Clinic transfer script saves evaluations and training histories rather than checkpoints for each target budget.

## Initial Office study

Entry point: `scripts/experiments/run_final_experiments.py`. Seeds 42-51, K=1,3,5,10; five configurations, static and closure evaluations. The configurations and exact arguments are defined by that script and saved per-run metadata. Per-robot evaluations, summaries and frozen protocols are under `runs/Office_Building/experiments/`. Final aggregate results and Holm comparisons are under `runs/Office_Building/final_results/`.

## Visit-weighted aggregation and greedy planner

Entry points:

```sh
python scripts/experiments/run_review_experiments.py --help
python evaluation/aggregate_review_experiments.py --help
```

Use `review_experiment_manifest.json` for the recorded arguments. `federated/server.py` contains both standard aggregation and the visit-weighted variant. The training option is `--q-aggregation standard` (default) or `--q-aggregation visitation_weighted`; the rest of the command is specified in the experiment runner. `evaluation/evaluate_review_variant.py` contains the greedy planner using immediate edge cost plus remaining shortest-path distance.

## Follow-up experiments

Entry point: `scripts/experiments/run_review_followup.py` with phases `pilot`, `neural`, `clinic`, `hotspots` and `spatial`. These runners skip completed runs; use a separate checkout/output tree without completed results for a fresh training replication. Saved hashes and timestamps describe the original execution, including code changes made during the Clinic correction.

- Six validation pilots, seed 11001, selected rate 0.001 for DQN and PPO.
- Eighty neural runs, GCN/raw representations, K=1/3, seeds 42-51, checkpoints at 1,000/2,000 episodes per robot.
- Ninety corrected Clinic runs, local/centralized/FedAvg Q-learning, K=3/5/10, seeds 42-51.
- Twenty spatial-training runs plus within-region evaluation of the same policies, `evaluation/evaluate_spatial_matched.py`.
- Thirty conflict hotspot replays of saved initial Office policies.
- Prioritized planner: `evaluation/review_prioritized_planner.py`; outcomes under `review_followup/prioritized/`.
- Corrected transfer: `scripts/experiments/review_corrected_transfer.py`, twenty source-agent/seed combinations, target budgets 0/10/50/100.

Aggregation entry points are `evaluation/aggregate_review_followup.py` and `evaluation/aggregate_corrected_transfer.py`. The main follow-up CSV combines Office and corrected Clinic results. Tests for transfer are not claimed: transfer outputs are descriptive means and confidence intervals.

The neural follow-up uses batched update kernels in `review_batched_dqn.py` and `review_batched_ppo.py`. All new neural comparisons share these kernels; the original study remains a separate configuration comparison. See the saved numerical equivalence reports and manuscript limitations.

## Figures and aggregate tables

The `evaluation/` package provides publication and diagnostic figure generators. Final aggregate CSVs and corrected tests are included in the result tree. The camera-ready manuscript and final figure files are distributed separately.

## Units and interpretation

Success is per robot, not joint mission success. SPL uses initial legal weighted shortest-path distance, before the scheduled closure. Intervals use ten training-seed means conditional on frozen scenarios. Deterministic planners have no across-training-seed interval; this does not eliminate uncertainty over scenarios. Statistical families for post-review analyses are separate from the initial comparison families.
