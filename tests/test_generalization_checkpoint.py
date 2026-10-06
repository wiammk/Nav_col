import torch

from agents.dqn_agent import DQNAgent
from scripts.evaluation.evaluate_generalization import load_transfer_checkpoint


def test_raw_fedavg_weights_are_loaded(tmp_path):
    source = DQNAgent()
    path = tmp_path / "global_model_final.pt"
    torch.save(source.get_weights(), path)
    target = DQNAgent()
    assert load_transfer_checkpoint(target, path) == "fedavg_weights"
    for key, value in source.get_weights().items():
        assert torch.equal(value, target.get_weights()[key])
