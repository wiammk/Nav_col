# Results index

Every building has its own folder: `runs/<building>/`. The folders below name
the result produced at each stage of the project.

| Result name | Location | Contents |
| --- | --- | --- |
| **01 — Prepared building data** | `data/processed/` | Parsed nodes, edges, graph, centrality values, and graph preview. |
| **02 — Trained models** | `models/` | Local and centralized Q-learning models plus graph-encoder artifacts. |
| **03 — Federated training results** | `results/federated/` | Final Q-learning, DQN, and PPO FedAvg models and their round histories. |
| **04 — Evaluation tables** | `evaluation/results/` | Success, SPL, reward, collision, and statistical-comparison CSV files. |
| **05 — Evaluation figures** | `evaluation/plots/` | Learning curves and comparison charts. |
| **06 — Navigation visualizations** | `visualizations/` | Navigation GIFs, progress GIFs, and visualization summaries. |
| **07 — Experiment protocol** | `experiments/` | Per-seed and per-robot training artifacts and paired test outputs. |
| **08 — Final report results** | `final_results/` | Aggregated tables, corrected tests, and report-ready figures. |
| **09 — Cross-building generalization** | `generalization*/` | Zero-shot and fine-tuning results for a target building. |
| **10 — Review figures** | `figures_for_review/` | Curated figures used for review or presentation. |

The folders are intentionally kept in their existing locations so stored
experiment paths and reproducibility scripts remain valid.
