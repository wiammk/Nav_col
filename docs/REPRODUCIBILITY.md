# Reproduction instructions

Run commands from the repository root after installing `requirements.txt`. Saved evaluation CSVs are available without model training. The recorded follow-up runs used Python 3.11.9 and PyTorch 2.12.0+cpu; individual `complete.json` files record their execution configuration.

## 1. Inputs and trained models

| Input | Location |
| --- | --- |
| Evaluated Office graph and frozen GCN | `runs/Office_Building/data/processed/` |
| Evaluated Clinic graph | `runs/Clinic_Architectural_width_corrected/data/processed/` |
| Initial Office scenarios | `runs/Office_Building/experiments/seed_*/robots_*/evaluation/*test_protocol.json` |
| Clinic scenarios | `runs/Clinic_Architectural_width_corrected/review_followup/protocols/` |
| Follow-up plan and selected learning rates | `runs/Office_Building/review_followup/experiment_plan.json` and `selected_hyperparameters*.json` |

Use the supplied graphs to reproduce the recorded study. Office retains its historical 0.9 m default door width. Regenerating it with the current parser reads 1.0 m and changes the graph hash, while its legal moves and reference distances remain the same. Clinic uses corrected opening widths. Graph hashes and raw IFC sources are documented in [DATA_SOURCES.md](DATA_SOURCES.md).

Restore trained models with:

```sh
python scripts/publication/restore_models.py
```

The command restores the original relative paths using `artifacts/manifest.json` and checks model hashes. It preserves existing files with different contents. Final policies and neural checkpoints at rounds 20 and 40 are included; other intermediate checkpoints are omitted. Target-transfer runs provide evaluation outputs and training histories.

## 2. Initial Office comparison

Entry point: `scripts/experiments/run_final_experiments.py`.

- Training seeds: 42–51; fleet sizes: 1, 3, 5 and 10.
- Methods: independent local Q-learning, centralized Q-learning, FedAvg Q-learning, FedAvg DQN and FedAvg PPO.
- Budget: 20 rounds × 50 episodes per robot, with the trajectory budget matched for local and centralized Q-learning.
- Evaluation: frozen static and edge-closure scenarios.

Exact arguments are defined by the runner and stored metadata. Per-seed outputs are in `runs/Office_Building/experiments/`; aggregate tables and Holm-adjusted tests are in `runs/Office_Building/final_results/`.

## 3. Visit weighting and greedy planning

Runner: `scripts/experiments/run_review_experiments.py`.

Aggregator: `evaluation/aggregate_review_experiments.py`.

The saved arguments are in `runs/Office_Building/review_additions/review_experiment_manifest.json`. Standard aggregation is the default; visit weighting is selected with `--q-aggregation visitation_weighted` in the training command. Both implementations are in `federated/server.py`.

The greedy planner in `evaluation/evaluate_review_variant.py` minimizes immediate edge cost plus remaining weighted shortest-path distance. Results and paired tests are in `runs/Office_Building/review_additions/`.

## 4. Additional experiments

The main runner is `scripts/experiments/run_review_followup.py`. Its phases are `pilot`, `neural`, `clinic`, `hotspots` and `spatial`.

| Analysis | Recorded design |
| --- | --- |
| Learning-rate selection | Six validation pilots; seed 11001; selected rate 0.001 for DQN and PPO |
| Neural representation and budget | 80 runs; GCN/raw inputs; K=1/3; seeds 42–51; 1,000/2,000 episodes per robot |
| Clinic multi-robot navigation | 90 runs; local/centralized/FedAvg Q-learning; K=3/5/10; seeds 42–51 |
| Spatial training | 20 runs; local/FedAvg; K=5; global and matched within-region evaluation |
| Conflict hotspots | 30 replays of saved Office policies |
| Prioritized planning | Static and closure evaluations for K=1/3/5/10 |
| Corrected transfer | 20 agent/seed combinations; target budgets 0/10/50/100 |

Additional entry points:

- Matched spatial evaluation: `evaluation/evaluate_spatial_matched.py`.
- Prioritized planner: `evaluation/review_prioritized_planner.py`.
- Transfer: `scripts/experiments/review_corrected_transfer.py`.
- Aggregation: `evaluation/aggregate_review_followup.py` and `evaluation/aggregate_corrected_transfer.py`.

Follow-up neural comparisons share the batched update kernels in `review_batched_dqn.py` and `review_batched_ppo.py`. The original Office comparison uses its recorded configurations. Numerical kernel diagnostics and result audits remain alongside the saved outputs.

The runners skip completed jobs. For a fresh training replication, use a separate checkout/output tree without completion markers. The broad defaults in `config/experiment_config.json` are exploratory settings; the experiment runners and saved plans define the paper's comparisons.

## 5. Statistics and figures

Success is measured per robot. SPL uses the initial legal weighted shortest-path distance, before the scheduled closure. Confidence intervals use ten training-seed means conditional on the fixed scenarios. Transfer estimates are descriptive. Post-review comparisons have their own Holm families.

Recorded execution metadata retains the original source hashes and local path strings. The publication manifest describes the current repository files; documentation and comment cleanup do not change the scientific results.

`evaluation/generate_publication_figures.py` generates diagnostic plots from the initial aggregate CSVs. Its closure plot uses percentage-point differences; it is separate from the paper's final relative-decrease figure.
