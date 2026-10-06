# Federated Multi-Robot Navigation

Code and recorded results for **Federated Reinforcement Learning for Multi-Robot Navigation on Building Graphs**.

IFC building models are converted into navigation graphs. Robots learn with tabular Q-learning, DQN or PPO and are evaluated in a shared simulator with occupancy constraints, conflicting moves and edge closures.

## Project structure

| Directory | Purpose |
| --- | --- |
| `agents/`, `models/` | Learning agents, neural networks and GCN encoder |
| `data/`, `environment/` | IFC parsing, graph construction and navigation simulator |
| `federated/` | Local clients, FedAvg and visit-weighted aggregation |
| `config/`, `scripts/` | Configuration, training and experiment entry points |
| `evaluation/`, `tests/` | Evaluation, statistical analysis, figures and existing checks |
| `runs/` | Evaluated graphs, frozen scenarios, training histories and results |
| `artifacts/` | Compressed trained models and their manifest |
| `docs/` | Reproduction instructions and data sources |

## Installation

Use Python 3.11 and install the dependencies in a virtual environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The general project entry point is `python run_project.py --help`.

## Results

The saved CSVs can be read without retraining. Start with the [results index](runs/RESULTS_INDEX.md).

| Study | Aggregate results |
| --- | --- |
| Initial Office comparison | [Static](runs/Office_Building/final_results/all_methods_all_scales.csv), [edge closure](runs/Office_Building/final_results/dynamic_methods_all_scales.csv) |
| Visit weighting and greedy shortest path | [Results](runs/Office_Building/review_additions/aggregate_review_results.csv) |
| Neural ablation, budgets, Clinic, planning and spatial analyses | [Results](runs/Office_Building/review_followup/aggregate_followup_results.csv), [paired tests](runs/Office_Building/review_followup/followup_paired_tests.csv) |
| Corrected Office-to-Clinic transfer | [Results](runs/Clinic_Architectural_width_corrected/review_transfer/aggregate_transfer.csv) |

Federation improves over independent local learning in several evaluated settings. Its advantage over centralized learning depends on the setting, and planning remains a strong baseline. The GCN effect depends on the neural agent; visit weighting and cross-building transfer do not establish a general improvement.

## Reproduction

Evaluated graphs, fixed scenarios and per-seed outputs are included. Before an evaluation that loads trained policies, restore the model archives:

```sh
python scripts/publication/restore_models.py
```

See [reproduction instructions](docs/REPRODUCIBILITY.md) for the experiment entry points, budgets and statistics. See [data sources](docs/DATA_SOURCES.md) for the original IFC models and graph provenance.

The study covers two abstract building graphs and simulated federation in one process. Real network conditions, privacy guarantees and physical robot deployment are outside the evaluated scope.
