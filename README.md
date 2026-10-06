# Federated Reinforcement Learning for Multi-Robot Navigation on Building Graphs

Code and recorded results for navigation on IFC-derived Office and Clinic building graphs, using tabular Q-learning, DQN and PPO with local, centralized or federated training. The simulator handles occupancy, node capacity, conflicting moves and an edge closure during navigation.

## Quick start

Use Python 3.11. The recorded follow-up experiments used Python 3.11.9 and PyTorch 2.12.0+cpu; metadata records the environment for individual runs. Install dependencies in a virtual environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The evaluated graphs, original Office encoder, fixed scenarios, per-seed evaluation CSVs and aggregate statistics are included. Reading the CSV results does not require training again. Trained policies are stored in compressed archives; restore them before an evaluation that loads saved policies:

```sh
python scripts/publication/restore_models.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the experiment entry points and exact result groups, and [DATA_SOURCES.md](DATA_SOURCES.md) for the IFC sources and graph provenance.

## Repository contents

| Path | Contents |
| --- | --- |
| `agents/`, `models/` | Q-learning, DQN, PPO and GCN implementations |
| `data/`, `environment/`, `federated/` | IFC parsing, graph construction, simulator and aggregation |
| `scripts/`, `evaluation/`, `config/` | Training, evaluation, experiment matrices and configurations |
| `runs/Office_Building/experiments/` | Initial ten-seed Office study, K=1,3,5,10 |
| `runs/Office_Building/review_additions/` | Visit weighting and greedy shortest-path comparisons |
| `runs/Office_Building/review_followup/` | Neural tuning/ablation/budgets, prioritized planning, conflict hotspots and spatial experiments |
| `runs/Clinic_Architectural_width_corrected/` | Corrected Clinic graph, multi-robot experiments and transfer |
| `artifacts/` | Final trained policies and neural budget checkpoints, compressed with original paths |
| `PUBLICATION_MANIFEST.json` | File hashes, archived policy hashes and publication scope |

## Reading the findings

Federation improves over independent local Q-learning in several evaluated settings, but does not consistently outperform centralized learning or planning. Visit-count weighting did not improve the results. GCN versus raw features depends on the neural agent. Corrected cross-building transfer remains weak. A matched spatial control narrows the scope of the large gains on globally sampled tasks.

The study evaluates two abstract building graphs and simulated federation in one process. It does not measure real network delays, client dropout, privacy guarantees or physical robot deployment. See the manuscript for the statistical families and limitations.

The original experiment configuration also lists exploratory defaults and potential buildings. The paper's experiments are defined by the experiment scripts and saved manifests; a listed option is not evidence that an experiment was run.

Raw IFC files and historical Clinic results from the incorrect door-width parser are excluded. Experiment data and model bytes are preserved. Recorded metadata contains historical local path strings; run commands from the repository root so the scripts construct the current paths.
