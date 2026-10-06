from pathlib import Path

import pytest

from evaluation.compare_federated_algorithms import (
    canonical_history_row,
    combined_round_rows,
    summarize_history,
)


def current_row(round_number=1):
    return {
        "round": str(round_number),
        "global_mean_reward": "2.5",
        "global_std_reward": "0.4",
        "global_success_rate": "0.75",
        "n_episodes": "6",
    }


def test_current_global_metric_schema_is_normalized():
    row = canonical_history_row(current_row())
    assert row["mean_reward"] == "2.5"
    assert row["std_reward"] == "0.4"
    assert row["success_rate"] == "0.75"


def test_legacy_metric_schema_remains_supported():
    row = canonical_history_row({
        "mean_reward": "1.0", "std_reward": "0.2", "success_rate": "0.5"
    })
    assert row["mean_reward"] == "1.0"


def test_missing_evaluation_metric_is_reported():
    with pytest.raises(ValueError, match="success_rate"):
        canonical_history_row({"global_mean_reward": "1", "global_std_reward": "2"})


def test_summary_and_round_export_accept_current_schema(tmp_path: Path):
    rows = [current_row(1), current_row(2)]
    summary = summarize_history("ppo", rows, tmp_path, 2, 0.8, 2)
    combined = combined_round_rows({"ppo": rows})
    assert summary["estimated_clients"] == 3
    assert summary["final_success_rate"] == pytest.approx(0.75)
    assert combined[0]["mean_reward"] == pytest.approx(2.5)
