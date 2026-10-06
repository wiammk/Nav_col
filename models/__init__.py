"""Public exports for graph encoders and navigation networks."""

from models.gcn_model import (
    GCNEncoder,
    GCNLayer,
    GraphDataConverter,
    NODE_FEAT_DIM,
    save_gcn,
    load_gcn,
)
from models.dqn_network import (
    DQN,
    DQNNetwork,
    build_dqn,
    save_dqn,
    load_dqn,
)
from models.ppo_network import (
    PPOActorCritic,
    build_ppo,
    save_ppo,
    load_ppo,
    compute_gae,
    ppo_loss,
)


def build_gcn(node_feat_dim=NODE_FEAT_DIM, hidden_dim=64,
              out_dim=64, n_layers=2, dropout=0.1) -> GCNEncoder:
    """Raccourci : crée et log un GCNEncoder."""
    import logging
    model = GCNEncoder(node_feat_dim, hidden_dim, out_dim, n_layers, dropout)
    logging.getLogger(__name__).info(f"  ✅ {model.summary()}")
    return model


__all__ = [
    "GCNEncoder", "GCNLayer", "GraphDataConverter",
    "NODE_FEAT_DIM", "save_gcn", "load_gcn", "build_gcn",
    "DQN", "DQNNetwork", "build_dqn", "save_dqn", "load_dqn",
    "PPOActorCritic", "build_ppo", "save_ppo", "load_ppo",
    "compute_gae", "ppo_loss",
]
