"""
P2.2 Market-quality proxy features — computed from odds timeline only.

No external data (betting volume, liquidity). All features derived from
odds prices, bookmaker counts, and temporal patterns in the timeline.
"""

import math
from typing import Dict, List


def compute_market_quality_features(timeline: List[dict], cutoff_minutes: float = 0) -> Dict[str, float]:
    """
    Compute market-quality proxy features from an odds timeline.

    Args:
        timeline: list of event dicts with euro_h/d/a, asian_line, waters, etc.
        cutoff_minutes: only events with minutes_before_kickoff >= cutoff are used.

    Returns dict of float features (all >= 0, missing → 0).
    """
    filtered = [e for e in timeline if e.get("minutes_before_kickoff", 0) >= cutoff_minutes]
    if not filtered:
        return _empty_features()

    # Sort by time (earliest first)
    sorted_tl = sorted(filtered, key=lambda e: e["minutes_before_kickoff"], reverse=True)

    # ── Bookmaker coverage ──
    # Since our data is single-bookmaker per sample, we use the timeline
    # to estimate coverage: more events = more updates = better coverage.
    num_events = len(sorted_tl)

    # How many events have euro/asian/ou
    has_euro = sum(1 for e in sorted_tl if _has_euro(e))
    has_asian = sum(1 for e in sorted_tl if _has_asian(e))
    has_ou = sum(1 for e in sorted_tl if _has_ou(e))

    # ── Overround (implied probability sum - 1) ──
    open_ev = sorted_tl[0]
    close_ev = sorted_tl[-1]

    open_or = _overround(open_ev)
    close_or = _overround(close_ev)
    or_delta = close_or - open_or

    # ── Bookmaker disagreement (std of implied probs across time) ──
    home_imps = []
    draw_imps = []
    away_imps = []
    for e in sorted_tl:
        if _has_euro(e):
            imp = _implied_probs(e)
            if imp:
                home_imps.append(imp[0])
                draw_imps.append(imp[1])
                away_imps.append(imp[2])

    if len(home_imps) >= 2:
        import statistics
        disagreement_home = statistics.stdev(home_imps) if len(home_imps) >= 2 else 0.0
        disagreement_draw = statistics.stdev(draw_imps) if len(draw_imps) >= 2 else 0.0
        disagreement_away = statistics.stdev(away_imps) if len(away_imps) >= 2 else 0.0
    else:
        disagreement_home = disagreement_draw = disagreement_away = 0.0

    # ── Update counts by time windows ──
    update_total = num_events
    update_24h = sum(1 for e in sorted_tl if e.get("minutes_before_kickoff", 0) <= 1440)
    update_6h = sum(1 for e in sorted_tl if e.get("minutes_before_kickoff", 0) <= 360)
    update_1h = sum(1 for e in sorted_tl if e.get("minutes_before_kickoff", 0) <= 60)

    # ── Movement magnitude ──
    if len(home_imps) >= 2:
        movement = abs(home_imps[-1] - home_imps[0]) + abs(draw_imps[-1] - draw_imps[0]) + abs(away_imps[-1] - away_imps[0])
    else:
        movement = 0.0

    # Late movement: last 6h vs earlier
    early_events = [e for e in sorted_tl if e.get("minutes_before_kickoff", 0) > 360]
    late_events = [e for e in sorted_tl if e.get("minutes_before_kickoff", 0) <= 360]
    if early_events and late_events:
        early_imp = _implied_probs(early_events[-1])  # latest early
        late_imp = _implied_probs(late_events[-1])     # latest late
        if early_imp and late_imp:
            late_movement = abs(late_imp[0] - early_imp[0]) + abs(late_imp[1] - early_imp[1]) + abs(late_imp[2] - early_imp[2])
        else:
            late_movement = 0.0
    else:
        late_movement = 0.0

    # ── Reversal count (direction changes) ──
    reversals = 0
    if len(home_imps) >= 3:
        for i in range(1, len(home_imps) - 1):
            prev_dir = home_imps[i] - home_imps[i-1]
            next_dir = home_imps[i+1] - home_imps[i]
            if (prev_dir > 0 and next_dir < 0) or (prev_dir < 0 and next_dir > 0):
                reversals += 1  # sign change = reversal

    # ── Missingness rate ──
    total_fields = num_events * 3  # euro, asian, ou
    present = has_euro + has_asian + has_ou
    missingness = 1.0 - (present / max(1, total_fields))

    # ── Market coverage score (0-1 composite) ──
    coverage = (min(has_euro / max(1, num_events), 1.0) * 0.4 +
                min(has_asian / max(1, num_events), 1.0) * 0.35 +
                min(has_ou / max(1, num_events), 1.0) * 0.25)

    return {
        "num_bookmakers_total": 1.0,  # single-bookmaker data
        "num_bookmakers_with_euro": 1.0 if has_euro > 0 else 0.0,
        "num_bookmakers_with_asian": 1.0 if has_asian > 0 else 0.0,
        "num_bookmakers_with_ou": 1.0 if has_ou > 0 else 0.0,
        "market_coverage_score": round(coverage, 4),
        "open_overround": round(open_or, 6),
        "latest_overround": round(close_or, 6),
        "overround_delta": round(or_delta, 6),
        "bookmaker_disagreement_home": round(disagreement_home, 6),
        "bookmaker_disagreement_draw": round(disagreement_draw, 6),
        "bookmaker_disagreement_away": round(disagreement_away, 6),
        "update_count_total": update_total,
        "update_count_last_24h": update_24h,
        "update_count_last_6h": update_6h,
        "update_count_last_1h": update_1h,
        "movement_magnitude": round(movement, 6),
        "late_movement_magnitude": round(late_movement, 6),
        "reversal_count": reversals,
        "missingness_rate": round(missingness, 4),
    }


def _has_euro(e: dict) -> bool:
    return (e.get("euro_h", 0) > 1.0 and e.get("euro_d", 0) > 1.0 and e.get("euro_a", 0) > 1.0)


def _has_asian(e: dict) -> bool:
    return ("asian_line" in e and
            (e.get("upper_water", 0) > 0 or e.get("upper_water", 0) < 0) and
            (e.get("lower_water", 0) > 0 or e.get("lower_water", 0) < 0))


def _has_ou(e: dict) -> bool:
    return (e.get("over_under_line", 0) > 0 and
            e.get("over_water", 0) > 0 and e.get("under_water", 0) > 0)


def _overround(e: dict) -> float:
    if not _has_euro(e): return 0.0
    imp = 1.0/e["euro_h"] + 1.0/e["euro_d"] + 1.0/e["euro_a"]
    return imp - 1.0


def _implied_probs(e: dict):
    if not _has_euro(e): return None
    rh = 1.0/e["euro_h"]; rd = 1.0/e["euro_d"]; ra = 1.0/e["euro_a"]
    total = rh + rd + ra
    if total <= 0: return None
    return (rh/total, rd/total, ra/total)


def _empty_features() -> Dict[str, float]:
    return {
        "num_bookmakers_total": 0.0, "num_bookmakers_with_euro": 0.0,
        "num_bookmakers_with_asian": 0.0, "num_bookmakers_with_ou": 0.0,
        "market_coverage_score": 0.0,
        "open_overround": 0.0, "latest_overround": 0.0, "overround_delta": 0.0,
        "bookmaker_disagreement_home": 0.0, "bookmaker_disagreement_draw": 0.0,
        "bookmaker_disagreement_away": 0.0,
        "update_count_total": 0, "update_count_last_24h": 0,
        "update_count_last_6h": 0, "update_count_last_1h": 0,
        "movement_magnitude": 0.0, "late_movement_magnitude": 0.0,
        "reversal_count": 0, "missingness_rate": 1.0,
    }
