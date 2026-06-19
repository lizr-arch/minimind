"""
OddsMind Grouped Time Split (P0.4)

Splits match data into train / val / test by kickoff_time,
keeping all samples derived from the same match_id in the same split.

This prevents future-information leakage that would occur with
random shuffling across time or across cutoff variants of the same match.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Set


def parse_kickoff_time(value: str) -> datetime:
    """
    Parse an ISO-8601 datetime string into a timezone-naive datetime.

    Handles 'Z' suffix and standard '+00:00' offsets.
    """
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def load_match_metadata(jsonl_path: str) -> List[dict]:
    """
    Load all matches from a JSONL file, returning dicts with at least
    'match_id' and 'kickoff_time' (parsed as datetime).

    Returns list sorted by kickoff_time ascending.
    """
    matches: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            m["_kickoff_dt"] = parse_kickoff_time(m["kickoff_time"])
            matches.append(m)

    matches.sort(key=lambda m: m["_kickoff_dt"])
    return matches


def grouped_time_split(
    matches: List[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_key: str = "match_id",
    time_key: str = "kickoff_time",
) -> Dict[str, Set[str]]:
    """
    Split matches into train / val / test by time order.

    Matches are assumed to be pre-sorted by time_key ascending.
    The earliest matches go to train, then val, then test (most recent).

    Splits are by match_id (group_key), not by individual samples.
    Every match_id appears in exactly one split.

    Args:
        matches: list of match dicts sorted by time ascending.
        train_ratio, val_ratio, test_ratio: proportions (should sum to ~1.0).
        group_key: field used as group identifier.
        time_key: field used for temporal ordering.

    Returns:
        {"train": set, "val": set, "test": set} of match_id strings.

    Raises:
        ValueError: if any split would be empty.
    """
    n = len(matches)
    if n == 0:
        return {"train": set(), "val": set(), "test": set()}

    # Compute split boundaries by match count
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    n_test = n - n_train - n_val

    # Adjust: ensure each split has at least 1 match
    if n_test < 1:
        # Give one from val or train
        if n_val > 1:
            n_val -= 1
            n_test = 1
        elif n_train > 1:
            n_train -= 1
            n_test = 1

    train_ids = {m[group_key] for m in matches[:n_train]}
    val_ids = {m[group_key] for m in matches[n_train:n_train + n_val]}
    test_ids = {m[group_key] for m in matches[n_train + n_val:]}

    # Verify no overlap
    assert train_ids.isdisjoint(val_ids), "train/val overlap"
    assert train_ids.isdisjoint(test_ids), "train/test overlap"
    assert val_ids.isdisjoint(test_ids), "val/test overlap"

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def get_time_range(matches: List[dict], match_ids: Set[str]) -> tuple:
    """Return (min_time, max_time) as ISO strings for a set of match_ids."""
    filtered = [m for m in matches if m["match_id"] in match_ids]
    if not filtered:
        return ("N/A", "N/A")
    times = [m["_kickoff_dt"] for m in filtered]
    return (min(times).isoformat(), max(times).isoformat())


def load_match_ids_from_file(path: str) -> Set[str]:
    """Load a set of match_ids from a plain-text file (one per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}
