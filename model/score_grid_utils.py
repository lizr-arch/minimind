"""
P1.13A: ScoreGrid utilities — soft target smoothing, marginal losses.

Key idea: instead of one-hot label for exact score (e.g., 2-1),
distribute probability mass to nearby scorelines to account for
scoreline adjacency and uncertainty.
"""
import torch
import torch.nn.functional as F
from typing import Optional


# ── Soft target smoothing ────────────────────────────────────────────────

def build_soft_target(
    home_goals: int,
    away_goals: int,
    grid_size: int = 8,
    self_weight: float = 0.75,
    neighbor_weight: float = 0.25,
) -> torch.Tensor:
    """
    Build a soft probability distribution over the score grid.

    The true scoreline gets self_weight, and neighboring scorelines
    (±1 in either home or away) split neighbor_weight.

    Neighbors considered:
        (h±1, a), (h, a±1), (h±1, a±1) — 8 neighbors total
        Each gets equal share of neighbor_weight.

    Args:
        home_goals, away_goals: true integer goals
        grid_size: bucket count per team (default 8)
        self_weight: probability mass for exact scoreline
        neighbor_weight: total mass distributed to neighbors

    Returns:
        [grid_size * grid_size] normalized probability tensor
    """
    G = grid_size
    num_classes = G * G
    target = torch.zeros(num_classes)

    hb = min(home_goals, G - 1)
    ab = min(away_goals, G - 1)

    # Collect neighbors
    neighbors = []
    for dh in [-1, 0, 1]:
        for da in [-1, 0, 1]:
            if dh == 0 and da == 0:
                continue
            nh = hb + dh
            na = ab + da
            if 0 <= nh < G and 0 <= na < G:
                neighbors.append(nh * G + na)

    # Assign weights
    center_idx = hb * G + ab
    target[center_idx] = self_weight

    if neighbors and neighbor_weight > 0:
        per_neighbor = neighbor_weight / len(neighbors)
        for idx in neighbors:
            target[idx] += per_neighbor

    # Normalize
    s = target.sum()
    if s > 0:
        target = target / s
    else:
        target[center_idx] = 1.0

    return target


# ── Marginal loss computation ─────────────────────────────────────────────

def compute_score_grid_loss(
    score_grid_logits: torch.Tensor,       # [B, 64]
    score_labels: torch.Tensor,             # [B, 2] (home, away goals)
    grid_size: int = 8,
    ce_weight: float = 1.0,
    result_weight: float = 0.3,
    total_weight: float = 0.3,
    diff_weight: float = 0.2,
    soft_target: bool = True,
    soft_self_weight: float = 0.75,
) -> dict:
    """
    Compute combined loss for ScoreGridHead.

    Loss = ce_weight * CE(score_grid_logits, soft/hard target)
         + result_weight * marginal_1X2_CE
         + total_weight * marginal_total_CE
         + diff_weight * marginal_diff_CE

    Returns dict with individual loss components.
    """
    B = score_grid_logits.shape[0]
    device = score_grid_logits.device
    G = grid_size

    probs = torch.softmax(score_grid_logits, dim=-1)  # [B, 64]
    probs_grid = probs.reshape(B, G, G)  # [B, 8, 8]

    # ── Main CE loss ──
    if soft_target:
        soft_targets = torch.zeros(B, G * G, device=device)
        for n in range(B):
            hg = int(score_labels[n, 0].item())
            ag = int(score_labels[n, 1].item())
            soft_targets[n] = build_soft_target(
                hg, ag, grid_size=G,
                self_weight=soft_self_weight,
                neighbor_weight=1.0 - soft_self_weight,
            )
        main_ce = -(soft_targets * F.log_softmax(score_grid_logits, dim=-1)).sum(-1).mean()
    else:
        score_class = ScoreGridHead_goals_to_class(score_labels[:, 0], score_labels[:, 1], G)
        main_ce = F.cross_entropy(score_grid_logits, score_class)

    losses = {"score_grid_ce": main_ce.item()}
    total_loss = ce_weight * main_ce

    # ── Marginal 1X2 loss ──
    if result_weight > 0:
        result_logits = torch.zeros(B, 3, device=device)
        for i in range(G):
            for j in range(G):
                p = probs_grid[:, i, j]
                if i > j:
                    result_logits[:, 0] += p
                elif i == j:
                    result_logits[:, 1] += p
                else:
                    result_logits[:, 2] += p
        result_logits = torch.log(result_logits + 1e-10)

        true_home = score_labels[:, 0]
        true_away = score_labels[:, 1]
        true_result = torch.where(true_home > true_away, torch.tensor(0, device=device),
                        torch.where(true_home < true_away, torch.tensor(2, device=device),
                                    torch.tensor(1, device=device)))
        result_ce = F.cross_entropy(result_logits, true_result)
        losses["marginal_result_ce"] = result_ce.item()
        total_loss = total_loss + result_weight * result_ce

    # ── Marginal total goals loss ──
    if total_weight > 0:
        total_logits = torch.zeros(B, 15, device=device)
        for i in range(G):
            for j in range(G):
                t = min(i + j, 14)
                total_logits[:, t] += probs_grid[:, i, j]
        total_logits = torch.log(total_logits + 1e-10)

        true_total = (score_labels[:, 0] + score_labels[:, 1]).clamp(0, 14).long()
        total_ce = F.cross_entropy(total_logits, true_total)
        losses["marginal_total_ce"] = total_ce.item()
        total_loss = total_loss + total_weight * total_ce

    # ── Marginal goal diff loss ──
    if diff_weight > 0:
        diff_logits = torch.zeros(B, 15, device=device)
        for i in range(G):
            for j in range(G):
                d = i - j + 7  # shift -7..+7 to 0..14
                diff_logits[:, d] += probs_grid[:, i, j]
        diff_logits = torch.log(diff_logits + 1e-10)

        true_diff = (score_labels[:, 0] - score_labels[:, 1]).clamp(-7, 7).long() + 7
        diff_ce = F.cross_entropy(diff_logits, true_diff)
        losses["marginal_diff_ce"] = diff_ce.item()
        total_loss = total_loss + diff_weight * diff_ce

    losses["total"] = total_loss.item()
    return total_loss, losses


# ── Helper (module-level to avoid circular import) ──────────────────────

def ScoreGridHead_goals_to_class(home_goals: torch.Tensor, away_goals: torch.Tensor, G: int = 8) -> torch.Tensor:
    """Convert goal tensors to flat class indices."""
    hb = home_goals.clamp(0, G - 1).long()
    ab = away_goals.clamp(0, G - 1).long()
    return hb * G + ab


# ── Score grid eval metrics ──────────────────────────────────────────────

def score_grid_predictions(
    probs_grid: torch.Tensor,  # [B, 8, 8]
    k: int = 5,
) -> dict:
    """
    Extract predictions from score grid.

    Returns dict with:
        top_k_scorelines: list of (score_str, prob) for top k
        expected_home_goals: [B]
        expected_away_goals: [B]
        home_win_prob, draw_prob, away_win_prob: [B]
    """
    B = probs_grid.shape[0]
    G = probs_grid.shape[1]
    device = probs_grid.device

    home_win = torch.zeros(B, device=device)
    draw = torch.zeros(B, device=device)
    away_win = torch.zeros(B, device=device)
    expected_home = torch.zeros(B, device=device)
    expected_away = torch.zeros(B, device=device)

    for i in range(G):
        for j in range(G):
            p = probs_grid[:, i, j]
            expected_home += p * i
            expected_away += p * j
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p

    return {
        "expected_home_goals": expected_home,
        "expected_away_goals": expected_away,
        "home_win_prob": home_win,
        "draw_prob": draw,
        "away_win_prob": away_win,
    }
