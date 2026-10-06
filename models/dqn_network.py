"""
Réseau DQN standard pour la navigation sur un graphe.

L'espace d'actions dépend du nœud courant. Le réseau estime donc une valeur
Q(s, a_k) pour chaque voisin valide en concaténant les embeddings du nœud
courant, de la cible et du voisin candidat.

Cette implémentation est volontairement un DQN standard :
  - un réseau online ;
  - un réseau target ;
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


log = logging.getLogger(__name__)

DEFAULT_EMB_DIM = 64
DEFAULT_HIDDEN_DIM = 128
STANDARD_DQN_ARCHITECTURE = "standard_dqn_congestion_v2"


class DQNNetwork(nn.Module):
    """Approxime directement Q(s, a) pour chaque voisin candidat."""

    def __init__(
        self,
        emb_dim: int = DEFAULT_EMB_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = 1

        # Entrée par action : [nœud courant || objectif || voisin candidat].
        self.q_network = nn.Sequential(
            nn.Linear(3 * self.emb_dim + self.context_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        current_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        neighbor_embs: torch.Tensor,
        neighbor_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Retourne un vecteur de K valeurs Q, une par voisin valide."""
        current = current_emb.reshape(self.emb_dim)
        goal = goal_emb.reshape(self.emb_dim)
        neighbors = neighbor_embs.reshape(-1, self.emb_dim)
        n_actions = neighbors.shape[0]

        if n_actions == 0:
            return torch.empty(0, dtype=current.dtype, device=current.device)

        state = torch.cat((current, goal), dim=0)
        state_batch = state.unsqueeze(0).expand(n_actions, -1)
        if neighbor_context is None:
            context = torch.zeros(
                (n_actions, self.context_dim),
                dtype=neighbors.dtype,
                device=neighbors.device,
            )
        else:
            context = neighbor_context.reshape(n_actions, self.context_dim)
        state_action = torch.cat((state_batch, neighbors, context), dim=1)
        return self.q_network(state_action).squeeze(-1)

    @torch.no_grad()
    def select_action(
        self,
        current_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        neighbor_embs: torch.Tensor,
        epsilon: float = 0.0,
    ) -> Tuple[int, torch.Tensor]:
        q_values = self(current_emb, goal_emb, neighbor_embs)
        if q_values.numel() == 0:
            raise ValueError("Aucune action valide pour le DQN.")
        if epsilon > 0.0 and torch.rand(1).item() < epsilon:
            action = torch.randint(q_values.shape[0], (1,)).item()
        else:
            action = q_values.argmax().item()
        return int(action), q_values

    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"DQNNetwork(emb={self.emb_dim}, hidden={self.hidden_dim}) "
            f"params={params:,}"
        )


def validate_standard_dqn_state_dict(state_dict: dict, source: str = "modèle") -> None:
    """Refuse explicitement les anciens checkpoints Double/Dueling DQN."""
    keys = set(state_dict)
    if any("value_stream" in key or "adv_stream" in key for key in keys):
        raise ValueError(
            f"{source} contient l'ancienne architecture Double Dueling DQN. "
            "Réentraînez --algo dqn pour produire un DQN standard."
        )
    if not any(key.startswith("q_network.") for key in keys):
        raise ValueError(
            f"{source} n'est pas un checkpoint DQN standard compatible."
        )


class DQN(nn.Module):
    """DQN standard avec réseaux online et target."""

    def __init__(
        self,
        emb_dim: int = DEFAULT_EMB_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        target_update_mode: str = "soft",
        tau: float = 0.005,
        update_freq: int = 500,
    ):
        super().__init__()
        if target_update_mode not in {"soft", "hard"}:
            raise ValueError("target_update_mode doit être 'soft' ou 'hard'.")

        self.online = DQNNetwork(emb_dim, hidden_dim)
        self.target = DQNNetwork(emb_dim, hidden_dim)
        self.mode = target_update_mode
        self.tau = float(tau)
        self.freq = int(update_freq)
        self._steps = 0

        self.target.load_state_dict(self.online.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.online(*args, **kwargs)

    @torch.no_grad()
    def update_target(self, force: bool = False) -> None:
        self._steps += 1
        if self.mode == "soft":
            for online_parameter, target_parameter in zip(
                self.online.parameters(), self.target.parameters()
            ):
                target_parameter.copy_(
                    self.tau * online_parameter
                    + (1.0 - self.tau) * target_parameter
                )
        elif force or self._steps % self.freq == 0:
            self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def compute_td_target(
        self,
        next_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        next_neighbors: torch.Tensor,
        reward: float,
        done: bool,
        gamma: float = 0.99,
        next_neighbor_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cible DQN standard : r + gamma * max_a Q_target(s', a)."""
        device = next_emb.device
        if done or next_neighbors.shape[0] == 0:
            return torch.tensor(reward, dtype=torch.float32, device=device)

        if next_neighbor_context is None:
            next_q_values = self.target(next_emb, goal_emb, next_neighbors)
        else:
            next_q_values = self.target(
                next_emb, goal_emb, next_neighbors, next_neighbor_context
            )
        return torch.tensor(reward, dtype=torch.float32, device=device) + (
            float(gamma) * next_q_values.max()
        )

    def summary(self) -> str:
        params = sum(p.numel() for p in self.online.parameters() if p.requires_grad)
        return (
            f"DQN standard(mode={self.mode}, tau={self.tau}) "
            f"online_params={params:,}"
        )


def build_dqn(
    emb_dim: int = DEFAULT_EMB_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    **kwargs,
) -> DQN:
    model = DQN(emb_dim, hidden_dim, **kwargs)
    log.info("  %s", model.summary())
    return model


def save_dqn(model: DQN, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": STANDARD_DQN_ARCHITECTURE,
            "state_dict": model.online.state_dict(),
            "emb_dim": model.online.emb_dim,
            "hidden_dim": model.online.hidden_dim,
        },
        path,
    )
    log.info("  DQN standard -> %s", path)


def load_dqn(path: Path, device: str = "cpu") -> DQN:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Format de checkpoint DQN non supporté: {path}")

    state_dict = checkpoint.get("state_dict", checkpoint)
    validate_standard_dqn_state_dict(state_dict, str(path))
    architecture = checkpoint.get("architecture")
    if architecture not in {None, STANDARD_DQN_ARCHITECTURE}:
        raise ValueError(f"Architecture DQN non supportée: {architecture}")

    model = build_dqn(
        emb_dim=int(checkpoint.get("emb_dim", DEFAULT_EMB_DIM)),
        hidden_dim=int(checkpoint.get("hidden_dim", DEFAULT_HIDDEN_DIM)),
    )
    model.online.load_state_dict(state_dict)
    model.target.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def _test() -> None:
    emb_dim = 64
    n_actions = 4
    current = torch.randn(emb_dim)
    goal = torch.randn(emb_dim)
    neighbors = torch.randn(n_actions, emb_dim)

    model = DQN(emb_dim=emb_dim, hidden_dim=128)
    q_values = model.online(current, goal, neighbors)
    assert q_values.shape == (n_actions,)
    assert not torch.isnan(q_values).any()

    with torch.no_grad():
        expected = 1.0 + 0.99 * model.target(current, goal, neighbors).max()
    actual = model.compute_td_target(
        current, goal, neighbors, reward=1.0, done=False, gamma=0.99
    )
    assert torch.allclose(actual, expected)
    assert not any(
        "value_stream" in name or "adv_stream" in name
        for name in model.online.state_dict()
    )

    log.info("dqn_network.py: DQN standard validé")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test()
