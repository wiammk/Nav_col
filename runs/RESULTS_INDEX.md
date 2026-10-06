# Results index

Each study includes frozen scenarios, per-seed outputs and aggregate estimates. All paths below are relative to `runs/`.

| Study | Directory | Key files |
| --- | --- | --- |
| Office inputs | `Office_Building/data/processed/` | `graph.gpickle`, node/edge CSVs, frozen GCN and embeddings |
| Initial Office experiment | `Office_Building/experiments/` | `seed_42`–`seed_51`, K=1/3/5/10, training histories and paired evaluations |
| Initial Office aggregates | `Office_Building/final_results/` | `all_methods_all_scales.csv`, `dynamic_methods_all_scales.csv`, primary paired tests |
| Visit weighting and greedy planner | `Office_Building/review_additions/` | `aggregate_review_results.csv`, `review_paired_tests.csv`, experiment manifest |
| Neural, planning, hotspots and spatial analyses | `Office_Building/review_followup/` | `aggregate_followup_results.csv`, `followup_paired_tests.csv`, experiment plan |
| Clinic inputs | `Clinic_Architectural_width_corrected/data/processed/` | Corrected `graph.gpickle` and node/edge CSVs |
| Clinic multi-robot experiment | `Clinic_Architectural_width_corrected/review_followup/` | `local/`, `centralized/`, `fedavg/`, frozen protocols and distance audit |
| Office-to-Clinic transfer | `Clinic_Architectural_width_corrected/review_transfer/` | `aggregate_transfer.csv`, per-seed budget results and validation report |

Clinic multi-robot aggregate estimates are included in the shared Office follow-up aggregate CSV. Transfer results are descriptive and have a separate aggregate file.

Original comparison tables report confidence intervals across training seeds. Detailed CSVs contain robot-level evaluation outcomes. Saved completion metadata retains configurations and original execution hashes.

Trained policies are stored under `artifacts/` at the repository root. Restore them with `python scripts/publication/restore_models.py` from that root before loading a saved policy.
