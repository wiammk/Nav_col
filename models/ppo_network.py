"""PPO actor-critic, generalized advantage estimation and clipped surrogate loss."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_EMB_DIM    = 64
DEFAULT_HIDDEN_DIM = 128


class PPOActorCritic(nn.Module):
    """
    Actor-Critic partagé pour PPO.

    Backbone commun → tête acteur + tête critique.
    L'acteur produit des logits par voisin (actions à espace variable).
    Le critique prédit V(s) pour le calcul des avantages GAE.

    Paramètres
    ----------
    emb_dim    : dimension des embeddings GCN
    hidden_dim : neurones dans le backbone et les têtes
    ortho_init : initialisation orthogonale (recommandé pour PPO)
    """

    def __init__(
        self,
        emb_dim    : int  = DEFAULT_EMB_DIM,
        hidden_dim : int  = DEFAULT_HIDDEN_DIM,
        ortho_init : bool = True,
    ):
        super().__init__()
        self.emb_dim    = emb_dim
        self.hidden_dim = hidden_dim
        self.context_dim = 1
        state_dim       = emb_dim * 2   # [current || goal]

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Prend [shared_emb || action_emb] → logit scalaire
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + emb_dim + self.context_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if ortho_init:
            self._ortho_init()

    def _ortho_init(self):
        """
        Initialisation orthogonale (Mnih et al., OpenAI Baselines).
        Améliore la stabilité de PPO en évitant les grands gradients initiaux.
        """
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        for head in [self.actor_head, self.critic_head]:
            for i, m in enumerate(head):
                if isinstance(m, nn.Linear):
                    # Petits gains pour les couches de sortie
                    gain = 0.01 if i == len(head) - 1 else 1.0
                    nn.init.orthogonal_(m.weight, gain=gain)
                    nn.init.zeros_(m.bias)


    def forward(
        self,
        current_emb   : torch.Tensor,
        goal_emb      : torch.Tensor,
        neighbor_embs : torch.Tensor,
        neighbor_context: torch.Tensor | None = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        Paramètres
        ----------
        current_emb   : [emb_dim]
        goal_emb      : [emb_dim]
        neighbor_embs : [K, emb_dim]

        Retourne
        --------
        dist  : Categorical(probs)   distribution sur K actions
        value : scalar tensor        V(s)
        """
        c = current_emb.view(self.emb_dim)
        g = goal_emb.view(self.emb_dim)
        K = neighbor_embs.shape[0]

        state  = torch.cat([c, g], dim=0)   # [2·emb]
        shared = self.backbone(state)        # [hidden_dim]

        value  = self.critic_head(shared).squeeze(-1)   # scalaire

        shared_exp = shared.unsqueeze(0).expand(K, -1)            # [K, hidden]
        if neighbor_context is None:
            neighbor_context = torch.zeros(
                (K, self.context_dim),
                dtype=neighbor_embs.dtype,
                device=neighbor_embs.device,
            )
        else:
            neighbor_context = neighbor_context.reshape(K, self.context_dim)
        actor_in   = torch.cat(
            [shared_exp, neighbor_embs, neighbor_context], dim=-1
        )
        logits     = self.actor_head(actor_in).squeeze(-1)        # [K]

        dist = Categorical(logits=logits)
        return dist, value


    def get_action(
        self,
        current_emb   : torch.Tensor,
        goal_emb      : torch.Tensor,
        neighbor_embs : torch.Tensor,
        greedy        : bool = False,
        neighbor_context: torch.Tensor | None = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Échantillonne (ou sélectionne greedily) une action.

        Retourne
        --------
        action_idx  : int        indice du voisin choisi
        log_prob    : tensor     log π(a|s)
        entropy     : tensor     H[π(·|s)]
        value       : tensor     V(s)
        """
        dist, value = self(
            current_emb, goal_emb, neighbor_embs, neighbor_context
        )

        if greedy:
            action = dist.probs.argmax()
        else:
            action = dist.sample()

        return (
            int(action.item()),
            dist.log_prob(action),
            dist.entropy(),
            value,
        )

    @torch.no_grad()
    def evaluate_actions(
        self,
        current_emb   : torch.Tensor,
        goal_emb      : torch.Tensor,
        neighbor_embs : torch.Tensor,
        actions       : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Réévalue des actions déjà prises (pour l'update PPO).
        Utilisé dans la phase d'apprentissage de l'agent PPO.

        Paramètres
        ----------
        actions : [T]  indices des actions prises dans le buffer

        Retourne
        --------
        log_probs  : [T]
        entropies  : [T]
        values     : [T]
        """
        dist, value = self(current_emb, goal_emb, neighbor_embs)
        return (
            dist.log_prob(actions),
            dist.entropy(),
            value.expand(actions.shape[0]),
        )


    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"PPOActorCritic(emb={self.emb_dim}, "
                f"hidden={self.hidden_dim})  params={params:,}")


def compute_gae(
    rewards  : torch.Tensor,
    values   : torch.Tensor,
    dones    : torch.Tensor,
    gamma    : float = 0.99,
    lam      : float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (Schulman et al., 2016).

    GAE(λ) offre un meilleur compromis biais/variance que TD(0) pur.

    Paramètres
    ----------
    rewards  : [T]      récompenses de la trajectoire
    values   : [T+1]    V(s_t) pour t=0..T  (inclut s_{T+1})
    dones    : [T]      1.0 si épisode terminé à t
    gamma    : discount
    lam      : paramètre λ de GAE

    Retourne
    --------
    advantages : [T]   avantages normalisés
    returns    : [T]   cibles pour le critique (V-target)
    """
    T           = len(rewards)
    advantages  = torch.zeros(T)
    gae         = 0.0

    for t in reversed(range(T)):
        next_value  = values[t + 1] * (1.0 - dones[t].item())
        delta       = rewards[t] + gamma * next_value - values[t]
        gae         = delta + gamma * lam * (1.0 - dones[t].item()) * gae
        advantages[t] = gae

    returns = advantages + values[:T]

    # Normalisation des avantages → stabilité PPO
    if T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def ppo_loss(
    log_probs_new : torch.Tensor,
    log_probs_old : torch.Tensor,
    advantages    : torch.Tensor,
    values_new    : torch.Tensor,
    returns       : torch.Tensor,
    entropies     : torch.Tensor,
    clip_eps      : float = 0.2,
    vf_coef       : float = 0.5,
    ent_coef      : float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """
    Perte PPO-Clip avec coefficient valeur et bonus d'entropie.

        L = L_clip - vf_coef · L_value + ent_coef · H

    Paramètres
    ----------
    log_probs_new : [T]   log π_θ(a|s) avec les nouveaux poids
    log_probs_old : [T]   log π_θ_old(a|s) depuis le rollout
    advantages    : [T]   avantages GAE normalisés
    values_new    : [T]   V(s) avec les nouveaux poids
    returns       : [T]   cibles V-target
    entropies     : [T]   entropie de la distribution courante
    clip_eps      : ε pour le clipping du ratio de probabilité
    vf_coef       : coefficient de la loss critique
    ent_coef      : coefficient du bonus d'entropie

    Retourne
    --------
    loss    : scalaire    perte totale (à minimiser)
    metrics : dict        sous-pertes pour le logging
    """
    # Ratio de probabilité r_t(θ) = π_new / π_old
    ratio          = (log_probs_new - log_probs_old.detach()).exp()

    # Perte acteur avec clipping
    surr1          = ratio * advantages
    surr2          = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    actor_loss     = -torch.min(surr1, surr2).mean()

    # Perte critique (MSE)
    critic_loss    = F.mse_loss(values_new, returns.detach())

    # Bonus d'entropie (encourage l'exploration)
    entropy_bonus  = entropies.mean()

    loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy_bonus

    metrics = {
        "loss_total"  : loss.item(),
        "loss_actor"  : actor_loss.item(),
        "loss_critic" : critic_loss.item(),
        "entropy"     : entropy_bonus.item(),
        "ratio_mean"  : ratio.mean().item(),
        "clip_frac"   : ((ratio - 1.0).abs() > clip_eps).float().mean().item(),
    }
    return loss, metrics


def build_ppo(
    emb_dim    : int  = DEFAULT_EMB_DIM,
    hidden_dim : int  = DEFAULT_HIDDEN_DIM,
    ortho_init : bool = True,
) -> PPOActorCritic:
    model = PPOActorCritic(emb_dim, hidden_dim, ortho_init)
    log.info(f"  ✅ {model.summary()}")
    return model


def save_ppo(model: PPOActorCritic, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "emb_dim"   : model.emb_dim,
        "hidden_dim": model.hidden_dim,
    }, path)
    log.info(f"  ✅ PPOActorCritic → {path}")


def load_ppo(path: Path, device: str = "cpu") -> PPOActorCritic:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = build_ppo(ckpt["emb_dim"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info(f"  ✅ PPOActorCritic chargé depuis {path}")
    return model.to(device)


def _test():
    log.info("🧪 Test PPOActorCritic + GAE + PPO Loss")
    EMB, K, T = 64, 5, 16

    emb_cur  = torch.randn(EMB)
    emb_goal = torch.randn(EMB)
    emb_nb   = torch.randn(K, EMB)

    model = PPOActorCritic(emb_dim=EMB, hidden_dim=128, ortho_init=True)
    log.info(f"  {model.summary()}")

    dist, value = model(emb_cur, emb_goal, emb_nb)
    assert dist.probs.shape == (K,),        f"probs shape: {dist.probs.shape}"
    assert dist.probs.sum().item() - 1 < 1e-5, "Probs ne somment pas à 1 !"
    assert value.ndim == 0,                 "value doit être scalaire"
    log.info(f"  Probs K={K} : {dist.probs.detach().round(decimals=3).tolist()}")
    log.info(f"  V(s) = {value.item():.4f}")

    a, logp, ent, v = model.get_action(emb_cur, emb_goal, emb_nb, greedy=False)
    assert 0 <= a < K
    assert logp.ndim == 0
    log.info(f"  Action={a}  logp={logp.item():.4f}  ent={ent.item():.4f}")

    rewards  = torch.rand(T)
    values_t = torch.rand(T + 1)
    dones    = torch.zeros(T);  dones[-1] = 1.0
    adv, ret = compute_gae(rewards, values_t, dones, gamma=0.99, lam=0.95)
    assert adv.shape == (T,)
    assert abs(adv.mean().item()) < 0.1, "Avantages non normalisés !"
    log.info(f"  GAE : adv_mean={adv.mean():.4f}  ret_mean={ret.mean():.4f}")

    lp_old = torch.randn(T)
    lp_new = torch.randn(T)
    v_new  = torch.randn(T)
    ents   = torch.rand(T) * 2
    loss, metrics = ppo_loss(lp_new, lp_old, adv, v_new, ret, ents)
    assert not torch.isnan(loss), "NaN dans la perte PPO !"
    log.info(f"  PPO Loss = {loss.item():.4f}  "
             f"clip_frac={metrics['clip_frac']:.2f}")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        p = Path(f.name)
    save_ppo(model, p)
    model2      = load_ppo(p)
    dist2, val2 = model2(emb_cur, emb_goal, emb_nb)
    assert torch.allclose(dist.probs, dist2.probs, atol=1e-5), "Poids diff !"
    os.unlink(p)
    log.info("  Sauvegarde/chargement ✅")

    log.info("=" * 50)
    log.info("✅ ppo_network.py — tous les tests passent")
    log.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    _test()
