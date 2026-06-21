"""
Multi-Bookmaker Feature Extractor (P1.6)

Transforms raw multi-bookmaker odds into 5-layer structured features:
  1. Consensus (avg, median of implied probabilities)
  2. Dispersion (std, range of implied probabilities)
  3. Sharp anchors (Pinnacle/Betfair vs market avg)
  4. Soft deviation (William Hill/Bwin/1XBet vs Pinnacle)
  5. Rank / extremes (who is most optimistic/pessimistic)

All computations use no-vig implied probabilities, not raw odds.
"""

import math
from typing import Dict, List, Optional, Tuple

# ── Bookmaker groups ──────────────────────────────────────────────────

SHARP_BOOKMAKERS = ["Pinnacle", "Betfair Exchange"]
MAINSTREAM_BOOKMAKERS = ["Bet365"]
SOFT_BOOKMAKERS = ["William Hill", "1XBet", "Bwin"]
ALL_BOOKMAKERS = SHARP_BOOKMAKERS + MAINSTREAM_BOOKMAKERS + SOFT_BOOKMAKERS

EURO_MARKETS = ["home", "draw", "away"]
AH_MARKETS = ["line", "home_water", "away_water"]


# ── Odds → Probability ───────────────────────────────────────────────

def implied_probability(odds_h: float, odds_d: float, odds_a: float) -> Tuple[float, float, float]:
    """
    Convert decimal odds to no-vig implied probabilities.
    Returns (p_home, p_draw, p_away) where sum ≈ 1.
    """
    raw_h = 1.0 / max(odds_h, 1.01)
    raw_d = 1.0 / max(odds_d, 1.01)
    raw_a = 1.0 / max(odds_a, 1.01)
    total = raw_h + raw_d + raw_a
    return (raw_h / total, raw_d / total, raw_a / total)


def bookmaker_margin(odds_h: float, odds_d: float, odds_a: float) -> float:
    """Bookmaker overround (margin)."""
    return 1.0 / odds_h + 1.0 / odds_d + 1.0 / odds_a - 1.0


# ── Per-match feature builder ─────────────────────────────────────────

def extract_bookmaker_features(
    per_bookmaker: Dict[str, dict],
    fill_missing: bool = True,
) -> dict:
    """
    Extract 5-layer features from one match's multi-bookmaker data.

    per_bookmaker: dict keyed by bookmaker_id, each value is the CLOSING
                   (last) event dict with keys:
                   euro_h, euro_d, euro_a, asian_line, upper_water, lower_water

    Returns a flat dict of feature_name → float.
    """
    feats = {}

    # ── Step 1: per-bookmaker implied probabilities ──
    bk_probs: Dict[str, Tuple[float, float, float]] = {}
    bk_margins: Dict[str, float] = {}
    for bk_id, event in per_bookmaker.items():
        h, d, a = event["euro_h"], event["euro_d"], event["euro_a"]
        bk_probs[bk_id] = implied_probability(h, d, a)
        bk_margins[bk_id] = bookmaker_margin(h, d, a)

    if not bk_probs:
        return feats

    # ── Collect probability vectors ──
    home_probs = {bk: p[0] for bk, p in bk_probs.items()}
    draw_probs = {bk: p[1] for bk, p in bk_probs.items()}
    away_probs = {bk: p[2] for bk, p in bk_probs.items()}

    # ── Layer 1: Consensus ──
    for market, prob_dict in [("home", home_probs), ("draw", draw_probs), ("away", away_probs)]:
        vals = list(prob_dict.values())
        feats[f"{market}_prob_avg"] = sum(vals) / len(vals)
        feats[f"{market}_prob_median"] = sorted(vals)[len(vals) // 2]

    # ── Layer 2: Dispersion ──
    for market, prob_dict in [("home", home_probs), ("draw", draw_probs), ("away", away_probs)]:
        vals = list(prob_dict.values())
        avg = sum(vals) / len(vals)
        feats[f"{market}_prob_std"] = math.sqrt(sum((v - avg) ** 2 for v in vals) / len(vals))
        feats[f"{market}_prob_range"] = max(vals) - min(vals)

    # ── Layer 3: Sharp anchors ──
    soft_avg_home = _safe_avg([home_probs.get(b, None) for b in SOFT_BOOKMAKERS + MAINSTREAM_BOOKMAKERS])
    soft_avg_draw = _safe_avg([draw_probs.get(b, None) for b in SOFT_BOOKMAKERS + MAINSTREAM_BOOKMAKERS])
    soft_avg_away = _safe_avg([away_probs.get(b, None) for b in SOFT_BOOKMAKERS + MAINSTREAM_BOOKMAKERS])

    for sharp in SHARP_BOOKMAKERS:
        if sharp in bk_probs:
            for market, prob_dict, soft_avg in [
                ("home", home_probs, soft_avg_home),
                ("draw", draw_probs, soft_avg_draw),
                ("away", away_probs, soft_avg_away),
            ]:
                key = sharp.lower().replace(" ", "_")
                feats[f"{market}_prob_{key}"] = prob_dict.get(sharp, 0)
                if soft_avg is not None:
                    feats[f"{market}_prob_{key}_vs_soft_avg"] = prob_dict.get(sharp, 0) - soft_avg

    # ── Layer 4: Soft deviation ──
    for soft in SOFT_BOOKMAKERS:
        if soft in bk_probs:
            key = soft.lower().replace(" ", "_")
            for sharp in SHARP_BOOKMAKERS:
                if sharp in bk_probs:
                    sk = sharp.lower().replace(" ", "_")
                    for market, prob_dict in [("home", home_probs), ("draw", draw_probs), ("away", away_probs)]:
                        feats[f"{market}_prob_{key}_vs_{sk}"] = prob_dict.get(soft, 0) - prob_dict.get(sharp, 0)

    # Bet365 vs sharp
    for sharp in SHARP_BOOKMAKERS:
        if "Bet365" in bk_probs and sharp in bk_probs:
            sk = sharp.lower().replace(" ", "_")
            for market, prob_dict in [("home", home_probs), ("draw", draw_probs), ("away", away_probs)]:
                feats[f"{market}_prob_b365_vs_{sk}"] = prob_dict["Bet365"] - prob_dict[sharp]

    # ── Layer 5: Rank / extremes ──
    for market, prob_dict in [("home", home_probs), ("draw", draw_probs), ("away", away_probs)]:
        sorted_bks = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        feats[f"{market}_prob_max"] = sorted_bks[0][1]
        feats[f"{market}_prob_min"] = sorted_bks[-1][1]
        feats[f"{market}_prob_max_bookmaker"] = _bk_index(sorted_bks[0][0])
        feats[f"{market}_prob_min_bookmaker"] = _bk_index(sorted_bks[-1][0])
        # Bet365 rank
        if "Bet365" in prob_dict:
            b365_val = prob_dict["Bet365"]
            rank = sum(1 for _, v in sorted_bks if v > b365_val) + 1
            feats[f"{market}_prob_b365_rank"] = rank

    # ── Bookmaker count ──
    feats["num_bookmakers"] = len(bk_probs)

    # ── Asian Handicap features ──
    _add_ah_features(feats, per_bookmaker)

    # Fill missing features with 0 for consistent dimensionality
    if fill_missing:
        names = get_feature_names()  # ensures cache is populated
        for name in names:
            feats.setdefault(name, 0.0)

    return feats


def _add_ah_features(feats: dict, per_bookmaker: Dict[str, dict]):
    """Add Asian Handicap multi-bookmaker features (line-aware)."""
    # Per-bookmaker AH data
    bk_lines = {}
    bk_home_water = {}
    bk_away_water = {}
    for bk_id, event in per_bookmaker.items():
        line = event.get("asian_line", 0)
        if line != 0:
            bk_lines[bk_id] = line
            bk_home_water[bk_id] = event.get("upper_water", 1.0)
            bk_away_water[bk_id] = event.get("lower_water", 1.0)

    if not bk_lines:
        return

    # Line consensus
    lines = list(bk_lines.values())
    feats["ah_line_avg"] = sum(lines) / len(lines)
    feats["ah_line_median"] = sorted(lines)[len(lines) // 2]
    feats["ah_line_std"] = math.sqrt(sum((l - feats["ah_line_avg"]) ** 2 for l in lines) / len(lines))
    feats["ah_line_range"] = max(lines) - min(lines)

    # Water comparison: only compare when lines match
    for sharp in SHARP_BOOKMAKERS:
        if sharp not in bk_lines:
            continue
        s_line = bk_lines[sharp]
        s_hw = bk_home_water[sharp]
        s_aw = bk_away_water[sharp]
        feats[f"ah_line_{sharp.lower().replace(' ','_')}"] = s_line

        # Compare other bookmakers ONLY if they have the same line
        same_line_bks = [b for b in bk_lines if abs(bk_lines[b] - s_line) < 0.01 and b != sharp]
        if same_line_bks:
            avg_hw = sum(bk_home_water[b] for b in same_line_bks) / len(same_line_bks)
            avg_aw = sum(bk_away_water[b] for b in same_line_bks) / len(same_line_bks)
            sk = sharp.lower().replace(" ", "_")
            feats[f"ah_home_water_{sk}_vs_same_line"] = s_hw - avg_hw
            feats[f"ah_away_water_{sk}_vs_same_line"] = s_aw - avg_aw

    # Home water spread
    hw_vals = list(bk_home_water.values())
    if hw_vals:
        feats["ah_home_water_avg"] = sum(hw_vals) / len(hw_vals)
        feats["ah_home_water_range"] = max(hw_vals) - min(hw_vals)

    # Away water spread
    aw_vals = list(bk_away_water.values())
    if aw_vals:
        feats["ah_away_water_avg"] = sum(aw_vals) / len(aw_vals)
        feats["ah_away_water_range"] = max(aw_vals) - min(aw_vals)


def _safe_avg(vals: List[Optional[float]]) -> Optional[float]:
    """Average of non-None values."""
    valid = [v for v in vals if v is not None]
    return sum(valid) / len(valid) if valid else None


_BK_INDEX = {b: i for i, b in enumerate(ALL_BOOKMAKERS)}


def _bk_index(name: str) -> int:
    return _BK_INDEX.get(name, -1)


# ── Batch feature names ──────────────────────────────────────────────

_FEATURE_NAMES_CACHE = None


def get_feature_names() -> List[str]:
    global _FEATURE_NAMES_CACHE
    if _FEATURE_NAMES_CACHE is not None:
        return _FEATURE_NAMES_CACHE
    dummy = {b: {"euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
                 "asian_line": -0.5, "upper_water": 0.9, "lower_water": 0.9}
             for b in ALL_BOOKMAKERS}
    feats = extract_bookmaker_features(dummy, fill_missing=False)
    _FEATURE_NAMES_CACHE = sorted(feats.keys())
    return _FEATURE_NAMES_CACHE


def feature_dim() -> int:
    return len(get_feature_names())
