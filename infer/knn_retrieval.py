"""
P2.3 KNN Retrieval Component — reusable similar-match retrieval for inference.

Usage:
    from infer.knn_retrieval import build_index, retrieve_similar, compute_explanation

    index = build_index(matches, cutoff_minutes=0)
    similar = retrieve_similar(target_match, index, K=100)
    explanation = compute_explanation(similar)
"""

import json
from typing import Dict, List

import torch

from eval.domain_transfer.domain_taxonomy import get_domain
from eval.domain_transfer.market_quality import compute_market_quality_features
from eval.odds_baselines import close_no_vig_euro, euro_probs_to_tensor
from dataset.odds_dataset import EURO_MAP

EPS = 1e-9

# ── Index building ─────────────────────────────────────────────────────

def build_index(matches: List[dict], cutoff_minutes: float = 0) -> List[dict]:
    """Build retrieval index from training matches.

    Args:
        matches: list of match dicts with odds_timeline, league_id, label, match_id.
        cutoff_minutes: minimum minutes_before_kickoff for events used.

    Returns:
        List of index entries: {match_id, baseline_probs, market_quality,
                                domain_family, league_id, label, cutoff_minutes}
    """
    index = []
    for m in matches:
        tl = m.get("odds_timeline", [])
        filtered = [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff_minutes]
        if not filtered:
            continue

        try:
            bl = euro_probs_to_tensor(close_no_vig_euro(filtered))
        except (ValueError, KeyError):
            bl = torch.full((3,), 1.0 / 3.0)

        mq = compute_market_quality_features(tl, cutoff_minutes)
        domain = get_domain(m.get("league_id", ""))
        label = EURO_MAP[m["label"]["euro_result"]]

        index.append({
            "match_id": m["match_id"],
            "baseline_probs": bl.tolist(),       # [home, draw, away]
            "market_quality": mq,                  # dict of float features
            "domain_family": domain["domain_family"],
            "league_id": m.get("league_id", ""),
            "label": label,                        # 0=home, 1=draw, 2=away
            "cutoff_minutes": cutoff_minutes,
        })
    return index


# ── Retrieval ───────────────────────────────────────────────────────────

def retrieve_similar(target_match: dict, index: List[dict], K: int = 100) -> List[dict]:
    """Retrieve K most similar matches from index.

    Similarity: euclidean distance on baseline probs + domain match bonus (0.0 if same domain_family, else 0.1).

    Args:
        target_match: dict with odds_timeline, cutoff_minutes, league_id.
        index: list of index entries from build_index().
        K: number of matches to retrieve.

    Returns:
        List of K index entries, sorted by similarity (best first).
    """
    tl = target_match.get("odds_timeline", [])
    cutoff = target_match.get("cutoff_minutes", 0)

    # Compute target features
    filtered = [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff]
    try:
        t_bl = euro_probs_to_tensor(close_no_vig_euro(filtered))
    except (ValueError, KeyError):
        t_bl = torch.full((3,), 1.0 / 3.0)

    t_domain = get_domain(target_match.get("league_id", ""))["domain_family"]

    scored = []
    for entry in index:
        bl = torch.tensor(entry["baseline_probs"])
        bl_dist = (t_bl - bl).pow(2).sum().item()
        domain_match = 0.0 if t_domain == entry["domain_family"] else 0.1
        score = bl_dist + domain_match
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0])
    return [e for _, e in scored[:K]]


# ── Explanation ─────────────────────────────────────────────────────────

def compute_explanation(retrieved: List[dict]) -> dict:
    """Generate explanation from retrieved similar matches.

    Returns:
        {support_level, historical_distribution, top_matches}
    """
    K = len(retrieved)
    counts = {0: 0, 1: 0, 2: 0}
    for r in retrieved:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    total = max(1, sum(counts.values()))
    probs = {
        "home": round(counts[0] / total, 4),
        "draw": round(counts[1] / total, 4),
        "away": round(counts[2] / total, 4),
    }

    # Support level
    if K >= 80:
        support = "high"
    elif K >= 40:
        support = "medium"
    elif K >= 10:
        support = "low"
    else:
        support = "insufficient"

    # Top matches (up to 10)
    top = []
    for r in retrieved[:10]:
        top.append({
            "match_id": r["match_id"],
            "league_id": r.get("league_id", ""),
            "domain_family": r.get("domain_family", ""),
            "result": ["home", "draw", "away"][r["label"]],
        })

    return {
        "k": K,
        "support_level": support,
        "historical_distribution": probs,
        "top_matches": top,
    }


def retrieval_predict(retrieved: List[dict], _unused=None) -> torch.Tensor:
    """Predict from retrieved matches: majority vote distribution.

    Args:
        retrieved: list of index entries from retrieve_similar().
        _unused: ignored (for API compatibility with eval scripts).

    Returns:
        torch.Tensor [3] with home/draw/away probabilities.
    """
    counts = {0: 0, 1: 0, 2: 0}
    for r in retrieved:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    total = max(1, sum(counts.values()))
    return torch.tensor([counts[0]/total, counts[1]/total, counts[2]/total])


# ── I/O ─────────────────────────────────────────────────────────────────

def load_index(path: str) -> List[dict]:
    """Load retrieval index from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index: List[dict], path: str):
    """Save retrieval index to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, default=str)
