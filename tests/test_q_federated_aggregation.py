import numpy as np

from agents.q_learning import QLearningAgent
from federated.server import visitation_weighted_q_aggregate


def test_q_aggregation_ignores_unvisited_zero_values():
    learned = np.array([[[8.0, 2.0]]], dtype=np.float32)
    unvisited = np.zeros_like(learned)
    learned_counts = np.array([[[4, 2]]], dtype=np.int64)
    zero_counts = np.zeros_like(learned_counts)
    result = visitation_weighted_q_aggregate(
        [learned, unvisited], [learned_counts, zero_counts]
    )
    np.testing.assert_allclose(result, learned)


def test_q_aggregation_keeps_previous_value_when_no_client_visited():
    previous = np.array([[[3.0]]], dtype=np.float32)
    result = visitation_weighted_q_aggregate(
        [np.zeros_like(previous), np.zeros_like(previous)],
        [np.zeros_like(previous, dtype=np.int64), np.zeros_like(previous, dtype=np.int64)],
        previous_global=previous,
    )
    np.testing.assert_allclose(result, previous)


def test_standard_q_learning_update_and_visitation():
    agent = QLearningAgent(
        num_nodes=2,
        max_degree=1,
        lr=0.5,
        gamma=0.9,
        epsilon_start=0.0,
        epsilon_end=0.0,
    )
    agent.Q[1, 1, 0] = 2.0
    agent.update(
        current_idx=0,
        target_idx=1,
        action=0,
        reward=1.0,
        next_idx=1,
        next_mask=np.array([1]),
        done=False,
    )
    # Q <- 0 + 0.5 * (1 + 0.9 * 2 - 0) = 1.4
    assert np.isclose(agent.Q[0, 1, 0], 1.4)
    assert agent.visit_counts[0, 1, 0] == 1
