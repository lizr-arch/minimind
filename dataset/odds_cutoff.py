"""
OddsMind Cutoff Sample Generator (P0.2)

Converts a single match with a full pre-match odds timeline into
multiple training samples, each representing the information available
at a specific cutoff time before kickoff.

Rules:
  - cutoff=90:  only events with minutes_before_kickoff >= 90
  - cutoff=60:  only events with minutes_before_kickoff >= 60
  - cutoff=30:  only events with minutes_before_kickoff >= 30

Each cutoff sample keeps the same final label (euro_result, asian_result)
because the outcome never changes — only the available information shrinks.
"""

from typing import List, Optional, Dict

# ── Cutoff bucket constants ──────────────────────────────────────────

CUTOFF_BUCKETS_BASE = [1440, 720, 360, 180, 120, 60, 30, 15, 5]
CUTOFF_BUCKETS_LAST30_1MIN = list(range(30, 0, -1))
CUTOFF_BUCKETS_V1 = [1440, 720, 360, 180, 120, 60, *list(range(30, 0, -1))]


def assert_cutoff_integrity(
    timeline: List[dict],
    cutoff_minutes: float,
    sample_id: str = "",
) -> None:
    """
    P1.16: Assert that every event in a filtered timeline respects the cutoff.

    Raises AssertionError if any event has minutes_before_kickoff < cutoff_minutes.

    Args:
        timeline: filtered list of event dicts.
        cutoff_minutes: the cutoff that was used for filtering.
        sample_id: optional identifier for error messages.
    """
    for e in timeline:
        mbk = e.get("minutes_before_kickoff", 0)
        if mbk < cutoff_minutes:
            raise AssertionError(
                f"Cutoff violation in sample '{sample_id}': "
                f"event has minutes_before_kickoff={mbk} < cutoff={cutoff_minutes}"
            )


def filter_timeline_by_cutoff(
    timeline: List[dict],
    cutoff_minutes: float,
) -> List[dict]:
    """
    Return a filtered timeline containing only events where
    minutes_before_kickoff >= cutoff_minutes.

    The result is sorted by minutes_before_kickoff DESC (earliest first,
    closest to kickoff last).  The input timeline is assumed pre-sorted;
    if not, we sort it explicitly.

    Args:
        timeline: list of event dicts, each with 'minutes_before_kickoff'.
        cutoff_minutes: minimum minutes-before-kickoff to include.

    Returns:
        Filtered list (may be empty if no events qualify).
    """
    # Ensure descending sort (largest minutes_before_kickoff first = earliest)
    sorted_timeline = sorted(
        timeline,
        key=lambda e: e["minutes_before_kickoff"],
        reverse=True,
    )

    filtered = [
        e for e in sorted_timeline
        if e["minutes_before_kickoff"] >= cutoff_minutes
    ]
    return filtered


def build_cutoff_samples(
    match: dict,
    cutoffs: List[float],
    min_events: int = 1,
) -> List[dict]:
    """
    Expand a single match dict into multiple cutoff samples.

    Args:
        match: dict with keys: match_id, odds_timeline, label,
               and optionally league_id, bookmaker_id, kickoff_time.
        cutoffs: list of cutoff minutes, e.g. [90, 60, 30].
        min_events: minimum number of events required after filtering
                    for a cutoff sample to be included.

    Returns:
        List of sample dicts, each with a unique sample_id and the
        filtered odds_timeline for that cutoff.

    Each output sample has:
        sample_id:       f"{match_id}__cutoff_{cutoff_minutes}"
        source_match_id: original match_id
        cutoff_minutes:  the cutoff value used
        odds_timeline:   filtered list
        label:           unchanged from input match
        league_id, bookmaker_id, kickoff_time: copied if present
    """
    if not cutoffs:
        return []

    match_id = match["match_id"]
    timeline = match["odds_timeline"]
    label = match["label"]

    samples = []
    for cutoff in cutoffs:
        filtered = filter_timeline_by_cutoff(timeline, cutoff)
        if len(filtered) < min_events:
            continue

        sample = {
            "sample_id": f"{match_id}__cutoff_{int(cutoff)}",
            "source_match_id": match_id,
            "cutoff_minutes": cutoff,
            "odds_timeline": filtered,
            "label": label,
        }

        # Copy optional metadata fields
        for key in ("league_id", "bookmaker_id", "kickoff_time"):
            if key in match:
                sample[key] = match[key]

        samples.append(sample)

    return samples


def build_exhaustive_cutoff_dataset(
    matches: List[dict],
    cutoffs: List[float],
    min_events: int = 1,
) -> List[dict]:
    """
    Expand a list of matches into a flat list of cutoff samples.

    Returns:
        List of cutoff-sample dicts ready for Dataset consumption.
    """
    all_samples = []
    for match in matches:
        samples = build_cutoff_samples(match, cutoffs, min_events=min_events)
        all_samples.extend(samples)
    return all_samples
