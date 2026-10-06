import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from agents.dqn_agent import DQNAgent
from models.dqn_network import (
    DQN,
    DQNNetwork,
    load_dqn,
    save_dqn,
    validate_standard_dqn_state_dict,
)
from evaluation.compare_federated_algorithms import dqn_model_compatibility_error


class FixedQNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, current_emb, goal_emb, neighbor_embs):
        return self.values[: neighbor_embs.shape[0]]


class StandardDQNTests(unittest.TestCase):
    def test_network_outputs_one_q_value_per_neighbor(self):
        model = DQNNetwork(emb_dim=8, hidden_dim=16)
        q_values = model(torch.randn(8), torch.randn(8), torch.randn(5, 8))
        self.assertEqual(tuple(q_values.shape), (5,))
        self.assertFalse(torch.isnan(q_values).any())
        self.assertFalse(
            any(
                "value_stream" in key or "adv_stream" in key
                for key in model.state_dict()
            )
        )

    def test_td_target_uses_target_network_maximum(self):
        model = DQN(emb_dim=2, hidden_dim=4)
        # Online préfère l'action 0, target préfère l'action 1. Un Double DQN
        # utiliserait target[0]. Le DQN standard doit utiliser max(target)=4.
        model.online = FixedQNetwork([9.0, 0.0, 0.0])
        model.target = FixedQNetwork([1.0, 4.0, 2.0])
        target = model.compute_td_target(
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(3, 2),
            reward=1.0,
            done=False,
            gamma=0.5,
        )
        self.assertAlmostEqual(target.item(), 3.0, places=6)

    def test_agent_and_checkpoint_use_standard_architecture(self):
        agent = DQNAgent(emb_dim=8, hidden_dim=16, batch_size=2)
        validate_standard_dqn_state_dict(agent.get_weights())

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "standard_dqn.pt"
            save_dqn(agent.model, checkpoint)
            loaded = load_dqn(checkpoint)
            self.assertIsInstance(loaded, DQN)
            self.assertEqual(
                set(agent.model.online.state_dict()),
                set(loaded.online.state_dict()),
            )

    def test_legacy_dueling_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Double Dueling DQN"):
            validate_standard_dqn_state_dict(
                {"value_stream.0.weight": torch.zeros(1, 1)},
                "ancien modèle",
            )

    def test_federated_comparison_rejects_legacy_dqn_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir) / "dqn"
            result_dir.mkdir(parents=True)
            torch.save(
                {"value_stream.0.weight": torch.zeros(1, 1)},
                result_dir / "global_model_final.pt",
            )
            error = dqn_model_compatibility_error(Path(temp_dir))
            self.assertIsNotNone(error)
            self.assertIn("Double Dueling DQN", error)


if __name__ == "__main__":
    unittest.main()
