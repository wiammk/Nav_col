from argparse import Namespace
from pathlib import Path

import pytest

from evaluation.aggregate_professor_results import (
    holm_adjust as aggregate_holm_adjust,
    mean_std_ci95,
    permutation_pvalue,
)
from evaluation.evaluate_professor_protocol import (
    exact_mcnemar,
    paired_permutation_test,
)
from scripts.experiments.run_final_experiments import command_is_complete, command_matrix


def experiment_args():
    return Namespace(
        building="test_building",
        seeds=[42],
        robot_counts=[3],
        rounds=4,
        episodes_per_round=5,
        max_steps=20,
        validation_scenarios=3,
        validation_seed=22002,
        test_scenarios=7,
        test_seed=33003,
        dynamic_seed=44004,
        save_every=2,
        heterogeneity="iid",
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        neural_gamma=0.99,
        dqn_lr=3e-4,
        dqn_batch_size=64,
        dqn_buffer_size=20000,
        dqn_tau=0.005,
        ppo_lr=3e-4,
        ppo_rollout_len=256,
        ppo_minibatch_size=64,
        ppo_epochs=10,
        ppo_clip=0.2,
        ppo_entropy_coef=0.01,
    )


def option_value(command, option):
    return command[command.index(option) + 1]


def test_final_matrix_matches_supervisor_comparisons_and_budget(tmp_path: Path):
    graph = tmp_path / "graph.gpickle"
    commands, records = command_matrix(experiment_args(), graph)

    training = commands[:-2]
    assert len(training) == 5
    assert "train_qlearning.py" in training[0]
    assert option_value(training[0], "--episodes") == "20"
    assert "train_centralized_qlearning.py" in training[1]
    assert option_value(training[1], "--total-trajectories") == "60"

    fed_commands = training[2:]
    assert {
        option_value(command, "--algo") for command in fed_commands
    } == {"qlearning", "dqn", "ppo"}
    assert all(option_value(command, "--rounds") == "4" for command in fed_commands)
    assert all(option_value(command, "--episodes") == "5" for command in fed_commands)

    assert records[0]["total_trajectories_per_method"] == 60
    assert records[0]["architecture_comparison"] == [
        "local_independent_qlearning",
        "centralized_qlearning",
        "fedavg_qlearning",
    ]
    assert records[0]["federated_algorithm_comparison"] == [
        "fedavg_qlearning",
        "fedavg_dqn",
        "fedavg_ppo",
    ]
    assert not any("train_neural_baseline.py" in command for command in commands)
    assert sum("train_qlearning.py" in command for command in commands) == 1


def test_paired_permutation_detects_identical_results():
    result = paired_permutation_test([1, 2, 3], [1, 2, 3], seed=42)
    assert result["mean_difference"] == 0.0
    assert result["p_value"] == 1.0


def test_exact_mcnemar_reports_discordant_pairs():
    result = exact_mcnemar([1, 1, 1, 0], [0, 0, 1, 1])
    assert result["discordant_a_wins"] == 2
    assert result["discordant_b_wins"] == 1
    assert 0.0 <= result["p_value"] <= 1.0


def test_across_seed_permutation_is_exact_for_ten_seeds():
    first = [1.0] * 10
    second = [0.0] * 10
    assert permutation_pvalue(first, second) == pytest.approx(2 / (2 ** 10))


def test_seed_confidence_interval_uses_student_t():
    mean, std, ci95 = mean_std_ci95(range(1, 11))
    assert mean == 5.5
    assert ci95 == pytest.approx(2.262 * std / (10 ** 0.5))


def test_holm_is_applied_inside_each_planned_family():
    rows = [
        {"robots": 3, "method_a": "a", "method_b": "b",
         "statistical_family": "primary", "p": 0.01},
        {"robots": 3, "method_a": "a", "method_b": "b",
         "statistical_family": "primary", "p": 0.04},
        {"robots": 5, "method_a": "a", "method_b": "b",
         "statistical_family": "primary", "p": 0.01},
        {"robots": 5, "method_a": "a", "method_b": "b",
         "statistical_family": "primary", "p": 0.04},
    ]
    adjusted = aggregate_holm_adjust(rows, p_key="p")
    assert [row["p_value_holm"] for row in adjusted] == [0.02, 0.04, 0.02, 0.04]
    assert all(row["holm_family_size"] == 2 for row in adjusted)


def test_centralized_budget_must_be_divisible_by_robot_count(tmp_path: Path):
    from scripts.training.train_centralized_qlearning import train_centralized_qlearning

    with pytest.raises(ValueError, match="divisible"):
        train_centralized_qlearning(
            graph_path=str(tmp_path / "unused.gpickle"),
            n_robots=3,
            total_trajectories=10,
        )


def test_resume_detects_completed_centralized_model(tmp_path: Path):
    model = tmp_path / "global_qtable.pkl"
    model.write_bytes(b"model")
    command = [
        "python",
        "train_centralized_qlearning.py",
        "--model",
        str(model),
    ]
    assert command_is_complete(command)
