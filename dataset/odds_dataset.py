"""
OddsMind Dataset — loads JSONL odds timeline data.

P0.2 adds cutoff modes:
  - none (default):       one sample per JSONL line (P0.1 behaviour)
  - exhaustive:           expand each match into one sample per cutoff
  - random:               at each __getitem__ randomly pick a cutoff

Label mapping:
    euro_result:  home=0, draw=1, away=2
    asian_result: upper=0, push=1, lower=2
"""

import json
import random
import warnings
from typing import Dict, List, Optional, Set, Union

import torch
from torch.utils.data import Dataset

from dataset.odds_cutoff import filter_timeline_by_cutoff, build_exhaustive_cutoff_dataset, assert_cutoff_integrity
from dataset.odds_labels import (
    encode_euro_label,
    encode_asian_label,
    get_asian_num_classes,
    EURO_MAP,
    ASIAN_MAP,
    ASIAN_MAP_5CLASS,
    ASIAN_REV_5CLASS,
    ASIAN_REV,
)
# P1.5 bookmaker embedding — defined here, not in odds_labels
BOOKMAKER_MAP = {k: i for i, k in enumerate(sorted([
    "Bet365", "Pinnacle", "William Hill", "1XBet", "Bwin", "Betfair Exchange"
]))}
BOOKMAKER_COUNT = len(BOOKMAKER_MAP)

LEAGUE_MAP = {k: i for i, k in enumerate(sorted([
    "Bundesliga", "EPL", "LaLiga", "Ligue1", "SerieA"
]))}
LEAGUE_COUNT = len(LEAGUE_MAP)

# Fixed feature order (must match OddsEventEncoder.feature_dim)
FEATURE_KEYS = [
    "minutes_before_kickoff",
    "euro_h",
    "euro_d",
    "euro_a",
    "asian_line",
    "upper_water",
    "lower_water",
    # P1.15: over/under (goal-line) market features
    "over_under_line",
    "over_water",
    "under_water",
]


def _event_to_features(event: dict, schema_version: str = "v1") -> List[float]:
    """Extract a fixed-order feature vector from an odds event dict.
    
    Missing keys default to 0.0 for backward compatibility with older data
    that lacks over/under fields.
    
    Schema versions:
        v1 (default): 10 features, no availability mask
        v2:            13 features, includes has_euro/has_asian/has_over_under
        v3:            13 features same as v2, plus parallel missing_mask tensor
    """
    base = [float(event.get(k, 0.0)) for k in FEATURE_KEYS]
    
    if schema_version in ("v2", "v3"):
        has_euro  = 1.0 if _event_has_euro(event) else 0.0
        has_asian = 1.0 if _event_has_asian(event) else 0.0
        has_ou    = 1.0 if _event_has_over_under(event) else 0.0
        return base + [has_euro, has_asian, has_ou]
    
    return base


# ── P1.16: Per-event missing mask (v3 schema) ────────────────────────────

def _event_to_missing_mask(event: dict) -> List[float]:
    """
    Return per-feature availability mask for a single odds event.

    Returns a list of 13 floats (matching v2/v3 feature dim):
        1.0 = feature is from real data
        0.0 = feature is a default/placeholder (missing)

    Rules:
      - minutes_before_kickoff: always present (structural field)
      - euro_h/d/a:          present if _event_has_euro()
      - asian_line + waters:  present if _event_has_asian()
        asian_line=0 with water prices → present (real flat handicap)
        asian_line=0 without water → missing
      - over_under_line + waters: present if _event_has_over_under()
        over_under_line=0 → always missing (no real goal line is 0)
      - has_euro/has_asian/has_over_under: always present (computed)
    """
    has_euro = _event_has_euro(event)
    has_asian = _event_has_asian(event)
    has_ou = _event_has_over_under(event)

    return [
        1.0,                              # minutes_before_kickoff
        (1.0 if has_euro else 0.0),       # euro_h
        (1.0 if has_euro else 0.0),       # euro_d
        (1.0 if has_euro else 0.0),       # euro_a
        (1.0 if has_asian else 0.0),      # asian_line
        (1.0 if has_asian else 0.0),      # upper_water
        (1.0 if has_asian else 0.0),      # lower_water
        (1.0 if has_ou else 0.0),         # over_under_line
        (1.0 if has_ou else 0.0),         # over_water
        (1.0 if has_ou else 0.0),         # under_water
        1.0,                               # has_euro (computed)
        1.0,                               # has_asian (computed)
        1.0,                               # has_over_under (computed)
    ]


# ── P1.15A: Market availability detection ────────────────────────────────

def _event_has_euro(event: dict) -> bool:
    """True if this event has real Euro odds (not just zero)."""
    return (event.get("euro_h", 0) > 1.0
            and event.get("euro_d", 0) > 1.0
            and event.get("euro_a", 0) > 1.0)


def _event_has_asian(event: dict) -> bool:
    """True if this event has real Asian handicap data.

    P1.1B: If the exporter provides 'asian_source' (v4 data), trust it:
        - 'raw_update' or 'forward_fill' → real data (True)
        - 'missing' → missing (False)
    P1.0F fallback for v3 data: Detect default placeholder values.
    """
    # v4 source-based detection (authoritative)
    if "asian_source" in event:
        src = event["asian_source"]
        if src == "missing":
            return False
        if src in ("raw_update", "forward_fill"):
            return True
    # v3 heuristic fallback
    al = event.get("asian_line", None)
    if al is None:
        return False
    uw = event.get("upper_water", 0)
    lw = event.get("lower_water", 0)
    if al == 0.0 and uw == 1.0 and lw == 1.0:
        return False
    return uw != 0 or lw != 0


def _event_has_over_under(event: dict) -> bool:
    """True if this event has real over/under goal-line data.

    P1.1B: If the exporter provides 'over_under_source' (v4 data), trust it:
        - 'raw_update' or 'forward_fill' → real data (True)
        - 'missing' → missing (False)
    P1.0F fallback for v3 data: Detect default placeholder values.
    """
    # v4 source-based detection (authoritative)
    if "over_under_source" in event:
        src = event["over_under_source"]
        if src == "missing":
            return False
        if src in ("raw_update", "forward_fill"):
            return True
    # v3 heuristic fallback
    ou_line = event.get("over_under_line", 0)
    ou_over = event.get("over_water", 0)
    ou_under = event.get("under_water", 0)
    if ou_line == 2.5 and ou_over == 1.0 and ou_under == 1.0:
        return False
    return ou_line > 0 and ou_over > 0 and ou_under > 0


def compute_market_availability(timeline: list) -> dict:
    """Compute market availability stats from an odds timeline.
    
    Returns dict suitable for inference output and eval reports.
    """
    total = len(timeline)
    if total == 0:
        return {
            "has_euro": False, "has_asian": False, "has_over_under": False,
            "euro_event_count": 0, "asian_event_count": 0, "over_under_event_count": 0,
            "asian_zero_line_count": 0, "ou_zero_line_count": 0,
            "missing_market_warning": ["no_timeline_data"],
        }
    
    euro_count = sum(1 for e in timeline if _event_has_euro(e))
    asian_count = sum(1 for e in timeline if _event_has_asian(e))
    ou_count = sum(1 for e in timeline if _event_has_over_under(e))
    
    # Count events where asian_line=0 but might be real flat handicap
    asian_zero = sum(1 for e in timeline 
                     if e.get("asian_line", 0) == 0 and _event_has_asian(e))
    ou_zero = sum(1 for e in timeline 
                  if e.get("over_under_line", 0) == 0)
    
    warnings = []
    if euro_count == 0:
        warnings.append("euro_missing")
    if asian_count == 0:
        warnings.append("asian_missing")
    if ou_count == 0:
        warnings.append("over_under_missing")
    
    return {
        "has_euro": euro_count > 0,
        "has_asian": asian_count > 0,
        "has_over_under": ou_count > 0,
        "euro_event_count": euro_count,
        "asian_event_count": asian_count,
        "over_under_event_count": ou_count,
        "asian_zero_line_count": asian_zero,
        "ou_zero_line_count": ou_zero,
        "missing_market_warning": warnings if warnings else ["none"],
    }


# ── P1.0G: Safe odds math utilities ──────────────────────────────────────

def safe_inverse_odds(x: float) -> float:
    """1/odds, safe for missing or invalid odds (<=1). Returns 0.0 for invalid."""
    if x <= 1.0:
        return 0.0
    return 1.0 / x


def safe_implied_probs(h: float, d: float, a: float) -> tuple:
    """3-way implied probabilities normalized to sum=1. Safe for missing odds."""
    rh, rd, ra = safe_inverse_odds(h), safe_inverse_odds(d), safe_inverse_odds(a)
    total = rh + rd + ra
    if total <= 0:
        return (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return (rh / total, rd / total, ra / total)


def safe_novig_probs(h: float, d: float, a: float) -> tuple:
    """No-vig probabilities (same as implied, normalized). Safe wrapper."""
    return safe_implied_probs(h, d, a)


def safe_overround(values: list) -> float:
    """Sum(1/odds) - 1. Returns 0.0 for missing/invalid odds."""
    inv_sum = sum(safe_inverse_odds(v) for v in values)
    if inv_sum <= 0:
        return 0.0
    return inv_sum - 1.0


def safe_2way_implied(a: float, b: float) -> tuple:
    """2-way implied probabilities normalized to sum=1."""
    ra, rb = safe_inverse_odds(a), safe_inverse_odds(b)
    total = ra + rb
    if total <= 0:
        return (0.5, 0.5)
    return (ra / total, rb / total)


def safe_2way_novig(a: float, b: float) -> tuple:
    """2-way no-vig probabilities. Safe wrapper."""
    return safe_2way_implied(a, b)


def safe_overround_2way(a: float, b: float) -> float:
    """2-way overround. Safe wrapper."""
    return safe_overround([a, b])


# ── P1.0G: v4 feature schema ─────────────────────────────────────────────

V4_FEATURE_NAMES = [
    # Time (1)
    "minutes_before_kickoff",
    # Euro raw (3)
    "euro_h", "euro_d", "euro_a",
    # Euro implied (3)
    "euro_h_implied", "euro_d_implied", "euro_a_implied",
    # Euro overround (1)
    "euro_overround",
    # Euro no-vig (3)
    "euro_h_novig", "euro_d_novig", "euro_a_novig",
    # Euro availability (1)
    "has_euro",
    # Asian raw (3)
    "asian_line", "upper_water", "lower_water",
    # Asian implied (2)
    "asian_upper_implied", "asian_lower_implied",
    # Asian overround (1)
    "asian_overround",
    # Asian no-vig (2)
    "asian_upper_novig", "asian_lower_novig",
    # Asian water spread (1)
    "asian_water_spread",
    # Asian availability (1)
    "has_asian",
    # Over/Under raw (3)
    "over_under_line", "over_water", "under_water",
    # Over/Under implied (2)
    "ou_over_implied", "ou_under_implied",
    # Over/Under overround (1)
    "ou_overround",
    # Over/Under no-vig (2)
    "ou_over_novig", "ou_under_novig",
    # Over/Under water spread (1)
    "ou_water_spread",
    # Over/Under availability (1)
    "has_over_under",
]

V4_FEATURE_DIM = len(V4_FEATURE_NAMES)  # 32

V5_FEATURE_NAMES = V4_FEATURE_NAMES + [
    "euro_change_rate", "asian_change_rate", "ou_change_rate",
]
V5_FEATURE_DIM = len(V5_FEATURE_NAMES)  # 35


def _event_to_features_v4(event: dict) -> List[float]:
    """Extract v4 feature vector: raw odds + implied + no-vig + overround + activity."""
    # Time
    minutes = float(event.get("minutes_before_kickoff", 0))

    # Euro
    euro_h = float(event.get("euro_h", 0))
    euro_d = float(event.get("euro_d", 0))
    euro_a = float(event.get("euro_a", 0))
    h_impl, d_impl, a_impl = safe_implied_probs(euro_h, euro_d, euro_a)
    h_nv, d_nv, a_nv = safe_novig_probs(euro_h, euro_d, euro_a)
    euro_or = safe_overround([euro_h, euro_d, euro_a])
    has_euro = 1.0 if _event_has_euro(event) else 0.0

    # Asian
    asian_line = float(event.get("asian_line", 0))
    uw = float(event.get("upper_water", 0))
    lw = float(event.get("lower_water", 0))
    has_asian = 1.0 if _event_has_asian(event) else 0.0
    ah_u_impl, ah_l_impl = safe_2way_implied(uw, lw)
    ah_u_nv, ah_l_nv = safe_2way_novig(uw, lw)
    ah_or = safe_overround_2way(uw, lw)
    ah_water_spread = abs(uw - lw) if has_asian else 0.0

    # Over/Under
    ou_line = float(event.get("over_under_line", 0))
    ou_o = float(event.get("over_water", 0))
    ou_u = float(event.get("under_water", 0))
    has_ou = 1.0 if _event_has_over_under(event) else 0.0
    ou_o_impl, ou_u_impl = safe_2way_implied(ou_o, ou_u)
    ou_o_nv, ou_u_nv = safe_2way_novig(ou_o, ou_u)
    ou_or = safe_overround_2way(ou_o, ou_u)
    ou_water_spread = abs(ou_o - ou_u) if has_ou else 0.0

    return [
        minutes,
        euro_h, euro_d, euro_a,
        h_impl, d_impl, a_impl,
        euro_or,
        h_nv, d_nv, a_nv,
        has_euro,
        asian_line, uw, lw,
        ah_u_impl, ah_l_impl,
        ah_or,
        ah_u_nv, ah_l_nv,
        ah_water_spread,
        has_asian,
        ou_line, ou_o, ou_u,
        ou_o_impl, ou_u_impl,
        ou_or,
        ou_o_nv, ou_u_nv,
        ou_water_spread,
        has_ou,
    ]


def _event_to_missing_mask_v4(event: dict) -> List[float]:
    """Per-feature availability mask for v4 schema (32 features)."""
    has_euro = _event_has_euro(event)
    has_asian = _event_has_asian(event)
    has_ou = _event_has_over_under(event)

    return [
        1.0,                                    # 0: minutes_before_kickoff
        (1.0 if has_euro else 0.0),             # 1: euro_h
        (1.0 if has_euro else 0.0),             # 2: euro_d
        (1.0 if has_euro else 0.0),             # 3: euro_a
        (1.0 if has_euro else 0.0),             # 4: euro_h_implied
        (1.0 if has_euro else 0.0),             # 5: euro_d_implied
        (1.0 if has_euro else 0.0),             # 6: euro_a_implied
        (1.0 if has_euro else 0.0),             # 7: euro_overround
        (1.0 if has_euro else 0.0),             # 8: euro_h_novig
        (1.0 if has_euro else 0.0),             # 9: euro_d_novig
        (1.0 if has_euro else 0.0),             # 10: euro_a_novig
        1.0,                                     # 11: has_euro
        (1.0 if has_asian else 0.0),            # 12: asian_line
        (1.0 if has_asian else 0.0),            # 13: upper_water
        (1.0 if has_asian else 0.0),            # 14: lower_water
        (1.0 if has_asian else 0.0),            # 15: asian_upper_implied
        (1.0 if has_asian else 0.0),            # 16: asian_lower_implied
        (1.0 if has_asian else 0.0),            # 17: asian_overround
        (1.0 if has_asian else 0.0),            # 18: asian_upper_novig
        (1.0 if has_asian else 0.0),            # 19: asian_lower_novig
        (1.0 if has_asian else 0.0),            # 20: asian_water_spread
        1.0,                                     # 21: has_asian
        (1.0 if has_ou else 0.0),               # 22: over_under_line
        (1.0 if has_ou else 0.0),               # 23: over_water
        (1.0 if has_ou else 0.0),               # 24: under_water
        (1.0 if has_ou else 0.0),               # 25: ou_over_implied
        (1.0 if has_ou else 0.0),               # 26: ou_under_implied
        (1.0 if has_ou else 0.0),               # 27: ou_overround
        (1.0 if has_ou else 0.0),               # 28: ou_over_novig
        (1.0 if has_ou else 0.0),               # 29: ou_under_novig
        (1.0 if has_ou else 0.0),               # 30: ou_water_spread
        1.0,                                     # 31: has_over_under
    ]


def _compute_odds_change_rate(sorted_timeline: list, current_idx: int, key: str) -> float:
    """Rate of change: (curr - prev) / prev for an odds field.
    Returns 0.0 for first event or missing/invalid values."""
    if current_idx <= 0:
        return 0.0
    prev = sorted_timeline[current_idx - 1].get(key, 0)
    curr = sorted_timeline[current_idx].get(key, 0)
    if prev <= 0 or curr <= 0:
        return 0.0
    return (curr - prev) / prev


def _event_to_features_v5(event: dict, change_rates: list) -> List[float]:
    """v5 = v4 (32-dim) + 3 change rates."""
    return _event_to_features_v4(event) + change_rates


def _event_to_missing_mask_v5(event: dict) -> List[float]:
    """v5 mask = v4 mask (32-dim) + 3 ones (derived, always present)."""
    return _event_to_missing_mask_v4(event) + [1.0, 1.0, 1.0]


# ── P1.0E: Consensus feature modes ──────────────────────────────────────

CONSENSUS_MODES = ("none", "visible_only", "legacy_full_timeline")


def _compute_consensus_feats(
    sample: dict,
    consensus_mode: str = "none",
    visible_timeline: Optional[list] = None,
) -> torch.Tensor:
    """
    Compute consensus features [6] based on the chosen mode.

    Modes:
        none:                  zeros(6) — safe default, no leakage risk
        visible_only:          compute from cutoff-filtered visible_timeline
        legacy_full_timeline:  compute from full-match bookmaker_features (DEBUG ONLY)

    Args:
        sample: raw match dict (may contain 'bookmaker_features')
        consensus_mode: one of CONSENSUS_MODES
        visible_timeline: the cutoff-filtered events (for visible_only mode)

    Returns:
        Tensor of shape [6]: [home_avg, home_med, draw_avg, draw_med, away_avg, away_med]
    """
    if consensus_mode == "none":
        return torch.zeros(6, dtype=torch.float32)

    if consensus_mode == "legacy_full_timeline":
        warnings.warn(
            "consensus_mode='legacy_full_timeline' uses full-match bookmaker_features "
            "which may leak future information. Use only for debugging.",
            UserWarning,
            stacklevel=3,
        )
        bk_feats = sample.get("bookmaker_features", {})
        return torch.tensor([
            bk_feats.get("home_prob_avg", 0.0), bk_feats.get("home_prob_median", 0.0),
            bk_feats.get("draw_prob_avg", 0.0), bk_feats.get("draw_prob_median", 0.0),
            bk_feats.get("away_prob_avg", 0.0), bk_feats.get("away_prob_median", 0.0),
        ], dtype=torch.float32)

    if consensus_mode == "visible_only":
        if visible_timeline is None or len(visible_timeline) == 0:
            return torch.zeros(6, dtype=torch.float32)
        # Compute from visible events' euro odds only (asian/ou are defaults in v3 data)
        from dataset.odds_bookmaker_features import implied_probability
        home_ps, draw_ps, away_ps = [], [], []
        for e in visible_timeline:
            h, d, a = e.get("euro_h", 0), e.get("euro_d", 0), e.get("euro_a", 0)
            if h > 1.0 and d > 1.0 and a > 1.0:
                hp, dp, ap = implied_probability(h, d, a)
                home_ps.append(hp)
                draw_ps.append(dp)
                away_ps.append(ap)
        if not home_ps:
            return torch.zeros(6, dtype=torch.float32)
        def _avg_med(vals):
            s = sorted(vals)
            n = len(s)
            avg = sum(s) / n
            med = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
            return avg, med
        ha, hm = _avg_med(home_ps)
        da, dm = _avg_med(draw_ps)
        aa, am = _avg_med(away_ps)
        return torch.tensor([ha, hm, da, dm, aa, am], dtype=torch.float32)

    raise ValueError(
        f"Unknown consensus_mode '{consensus_mode}'. "
        f"Allowed: {CONSENSUS_MODES}"
    )


def _build_features_and_labels(sample: dict, max_seq_len: int, asian_label_mode: str = "3class",
                               feature_schema_version: str = "v1",
                               validate_cutoff: bool = True,
                               consensus_mode: str = "none") -> dict:
    """
    Core item builder: given a sample dict (which may be an original match
    or a pre-built cutoff sample), extract features, labels, and metadata.
    """
    timeline = sample["odds_timeline"]
    cutoff = sample.get("cutoff_minutes", 0)

    # P1.16: optional cutoff integrity assertion
    if validate_cutoff and cutoff > 0:
        sid = sample.get("sample_id", sample.get("match_id", ""))
        assert_cutoff_integrity(timeline, cutoff, sample_id=sid)

    # Sort descending by minutes_before_kickoff (earliest first)
    sorted_timeline = sorted(
        timeline,
        key=lambda e: e["minutes_before_kickoff"],
        reverse=True,
    )

    # Truncate to max_seq_len (keep most recent = end of list)
    if len(sorted_timeline) > max_seq_len:
        sorted_timeline = sorted_timeline[-max_seq_len:]

    # P1.18: handle empty timelines (e.g. all events filtered by cutoff)
    if feature_schema_version == "v5":
        fdim = V5_FEATURE_DIM
    elif feature_schema_version == "v4":
        fdim = V4_FEATURE_DIM
    elif feature_schema_version in ("v2", "v3"):
        fdim = len(FEATURE_KEYS) + 3
    else:
        fdim = len(FEATURE_KEYS)

    if len(sorted_timeline) == 0:
        features = torch.zeros(0, fdim, dtype=torch.float32)
    elif feature_schema_version == "v5":
        # v5: v4 features + 3 change rates (need timeline context)
        feature_rows = []
        for i, e in enumerate(sorted_timeline):
            cr_h = _compute_odds_change_rate(sorted_timeline, i, "euro_h")
            cr_d = _compute_odds_change_rate(sorted_timeline, i, "euro_d")
            cr_a = _compute_odds_change_rate(sorted_timeline, i, "euro_a")
            feature_rows.append(_event_to_features_v5(e, [cr_h, cr_d, cr_a]))
        features = torch.tensor(feature_rows, dtype=torch.float32)
    elif feature_schema_version == "v4":
        features = torch.tensor(
            [_event_to_features_v4(e) for e in sorted_timeline],
            dtype=torch.float32,
        )
    else:
        features = torch.tensor(
            [_event_to_features(e, schema_version=feature_schema_version) for e in sorted_timeline],
            dtype=torch.float32,
        )

    # P1.16: per-event missing mask (v3/v4/v5 schema)
    missing_mask = None
    if feature_schema_version in ("v3", "v4", "v5"):
        if len(sorted_timeline) == 0:
            missing_mask = torch.zeros(0, fdim, dtype=torch.float32)
        elif feature_schema_version == "v5":
            missing_mask = torch.tensor(
                [_event_to_missing_mask_v5(e) for e in sorted_timeline],
                dtype=torch.float32,
            )
        elif feature_schema_version == "v4":
            missing_mask = torch.tensor(
                [_event_to_missing_mask_v4(e) for e in sorted_timeline],
                dtype=torch.float32,
            )
        else:
            missing_mask = torch.tensor(
                [_event_to_missing_mask(e) for e in sorted_timeline],
                dtype=torch.float32,
            )

    euro_label = encode_euro_label(sample["label"]["euro_result"])

    # P1.1B: asian_label with missing_handicap support
    label_data = sample["label"]
    asian_label_status = label_data.get("asian_label_status", "ok")
    asian_result = label_data.get("asian_result")
    if asian_label_status == "missing_handicap" or asian_result is None:
        asian_label = -100  # ignore_index placeholder
        asian_label_mask = 0.0
    else:
        asian_label = encode_asian_label(asian_result, mode=asian_label_mode)
        asian_label_mask = 1.0

    match_id = sample.get("sample_id", sample.get("match_id", ""))

    # P1.4: score labels [home_goals, away_goals]
    hg = sample["label"].get("home_goals", 0) or 0
    ag = sample["label"].get("away_goals", 0) or 0
    score_label = torch.tensor([float(hg), float(ag)], dtype=torch.float32)

    # P1.5: bookmaker id
    bookmaker_id = BOOKMAKER_MAP.get(sample.get("bookmaker_id", "Bet365"), 0)

    # P1.6: consensus features from multi-bookmaker data
    # P1.0E: use consensus_mode to control leakage
    consensus_feats = _compute_consensus_feats(
        sample, consensus_mode=consensus_mode, visible_timeline=sorted_timeline,
    )

    result = {
        "features": features,
        "euro_label": euro_label,
        "asian_label": asian_label,
        "asian_label_mask": asian_label_mask,
        "score_label": score_label,
        "bookmaker_id": bookmaker_id,
        "consensus_feats": consensus_feats,
        "match_id": match_id,
        "league_id": sample.get("league_id", ""),
        "seq_len": len(sorted_timeline),
    }
    if missing_mask is not None:
        result["missing_mask"] = missing_mask
    return result


class OddsDataset(Dataset):
    """
    Dataset for odds time-series matches.

    Modes:
        cutoff_mode="none" (default): one sample per JSONL line.
            Uses each sample's own 'cutoff_minutes' field or the global
            cutoff_minutes arg to filter the timeline.

        cutoff_mode="exhaustive": each match is expanded into one sample
            per cutoff value.  Filtering happens at init time.

        cutoff_mode="random": each __getitem__ randomly selects a cutoff
            from the cutoffs list and filters on the fly.

    Args:
        jsonl_path:      Path to JSONL file.
        max_seq_len:     Max time-steps per sample.
        cutoff_minutes:  Global cutoff override (used in "none" mode).
        cutoffs:         List of cutoff values for "exhaustive" / "random".
        cutoff_mode:     "none" | "exhaustive" | "random".
        min_events:      Minimum events required after filtering.
        asian_label_mode: "3class" (default) or "5class" (P0.3).
        seed:            Random seed for "random" mode.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_seq_len: int = 64,
        cutoff_minutes: Optional[float] = None,
        cutoffs: Optional[List[float]] = None,
        cutoff_mode: str = "none",
        min_events: int = 1,
        asian_label_mode: str = "3class",
        allowed_match_ids: Optional[Set[str]] = None,
        seed: int = 42,
        feature_schema_version: str = "v1",  # P1.15B: "v1"=10dim, "v2"=13dim, "v3"=13dim+mask
        consensus_mode: str = "none",  # P1.0E: "none" | "visible_only" | "legacy_full_timeline"
    ):
        if cutoff_mode not in ("none", "exhaustive", "random"):
            raise ValueError(f"Unknown cutoff_mode: {cutoff_mode}")
        if asian_label_mode not in ("3class", "5class"):
            raise ValueError(f"Unknown asian_label_mode: {asian_label_mode}")
        if consensus_mode not in CONSENSUS_MODES:
            raise ValueError(
                f"Unknown consensus_mode: '{consensus_mode}'. "
                f"Allowed: {CONSENSUS_MODES}"
            )

        self.max_seq_len = max_seq_len
        self.cutoff_minutes = cutoff_minutes
        self.cutoffs = cutoffs or []
        self.cutoff_mode = cutoff_mode
        self.min_events = min_events
        self.asian_label_mode = asian_label_mode
        self._asian_map = ASIAN_MAP if asian_label_mode == "3class" else ASIAN_MAP_5CLASS
        self._jsonl_path = jsonl_path
        self._allowed_ids = allowed_match_ids
        self.feature_schema_version = feature_schema_version  # P1.15B
        self.consensus_mode = consensus_mode  # P1.0E

        # Load raw matches
        raw_matches: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = json.loads(line)
                if allowed_match_ids is not None and m["match_id"] not in allowed_match_ids:
                    continue
                raw_matches.append(m)

        self.num_raw_matches = len(raw_matches)

        if cutoff_mode == "exhaustive":
            # Pre-expand all matches
            self.samples = build_exhaustive_cutoff_dataset(
                raw_matches, self.cutoffs, min_events=min_events
            )
            # Count per cutoff
            self.cutoff_counts: Dict[str, int] = {}
            for s in self.samples:
                c = str(int(s["cutoff_minutes"]))
                self.cutoff_counts[c] = self.cutoff_counts.get(c, 0) + 1
        else:
            # Keep raw matches (filtering deferred to __getitem__)
            self.samples = raw_matches
            self.cutoff_counts = {}

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        if self.cutoff_mode == "exhaustive":
            # Sample is already a filtered cutoff sample
            return _build_features_and_labels(sample, self.max_seq_len, self.asian_label_mode,
                                                feature_schema_version=self.feature_schema_version,
                                                consensus_mode=self.consensus_mode)

        elif self.cutoff_mode == "random":
            # Pick a random cutoff and filter
            match = sample
            valid_cutoffs = []
            for c in self.cutoffs:
                filtered = filter_timeline_by_cutoff(match["odds_timeline"], c)
                if len(filtered) >= self.min_events:
                    valid_cutoffs.append((c, filtered))

            if not valid_cutoffs:
                # Fallback: use the match's own cutoff or 0
                cutoff = self.cutoff_minutes or match.get("cutoff_minutes", 0)
                filtered = filter_timeline_by_cutoff(match["odds_timeline"], cutoff)
            else:
                cutoff, filtered = self._rng.choice(valid_cutoffs)

            # Build a temporary cutoff sample
            cutoff_sample = {
                "sample_id": f"{match['match_id']}__cutoff_{int(cutoff)}",
                "match_id": match["match_id"],
                "cutoff_minutes": cutoff,
                "odds_timeline": filtered,
                "label": match["label"],
                "league_id": match.get("league_id", ""),
                "bookmaker_id": match.get("bookmaker_id", "Bet365"),
            }
            return _build_features_and_labels(cutoff_sample, self.max_seq_len, self.asian_label_mode,
                                                feature_schema_version=self.feature_schema_version,
                                                consensus_mode=self.consensus_mode)

        else:
            # "none" mode — P0.1 behaviour
            cutoff = (
                self.cutoff_minutes
                if self.cutoff_minutes is not None
                else sample.get("cutoff_minutes", 0)
            )

            filtered = filter_timeline_by_cutoff(sample["odds_timeline"], cutoff)

            cutoff_sample = {
                "match_id": sample["match_id"],
                "odds_timeline": filtered,
                "label": sample["label"],
        "league_id": sample.get("league_id", ""),
                "bookmaker_id": sample.get("bookmaker_id", "Bet365"),
            }
            return _build_features_and_labels(cutoff_sample, self.max_seq_len, self.asian_label_mode,
                                                feature_schema_version=self.feature_schema_version,
                                                consensus_mode=self.consensus_mode)
