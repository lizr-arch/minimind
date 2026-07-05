"""
Phase 1a Baseline Evaluation Script (sub-step 2a)

Reads the flat v6_phase1a.jsonl, deduplicates by match_id using
bookmaker priority, filters by split, computes the joint euro+asian
baseline (方案 C), and outputs Track 2 metrics.

Usage:
    python eval/eval_phase1a_baselines.py \
        --data data/odds_real/v6_phase1a.jsonl \
        --split-match-ids data/odds_real/splits_v6/test_match_ids.txt \
        --bookmaker-priority Pinnacle,Bet365 \
        --out-json eval/reports/phase1a_baseline_v2.json
"""

import argparse
import json
import os
import sys
from collections import Counter

import torch

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_baselines_v2 import Phase1Baseline, kl_divergence
from eval.odds_metrics import (
    accuracy_from_probs,
    logloss_from_probs,
    brier_from_probs,
    ece_from_probs,
    class_counts,
    prediction_counts_from_probs,
)

EURO_MAP = {"home": 0, "draw": 1, "away": 2}
EURO_REVERSE = {0: "home", 1: "draw", 2: "away"}


def load_split_ids(path: str) -> set:
    """Load match_ids from a split file (one per line)."""
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mid = line.strip()
            if mid:
                ids.add(mid)
    return ids


def load_flat_jsonl(path: str) -> list:
    """Load all rows from a flat Phase 1a JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def deduplicate_by_bookmaker(
    rows: list, priority: list
) -> list:
    """
    Keep one row per match_id, picking the highest-priority bookmaker.

    priority: list of bookmaker_id strings, first = highest priority.
    Rows whose bookmaker_id is not in the priority list are kept if no
    higher-priority row exists for that match_id.
    """
    priority_rank = {bk: i for i, bk in enumerate(priority)}
    best: dict = {}  # match_id → (rank, row)

    for row in rows:
        mid = row["match_id"]
        bk = row.get("bookmaker_id", "")
        rank = priority_rank.get(bk, len(priority))  # unlisted = lowest

        if mid not in best or rank < best[mid][0]:
            best[mid] = (rank, row)

    return [row for _, row in best.values()]


def categorise_handicap(row: dict) -> str:
    """Return a handicap category label for coverage tracking."""
    line = row.get("close_asian_line")
    uw = row.get("close_asian_upper_water")
    lw = row.get("close_asian_lower_water")

    if line is None or uw is None or lw is None:
        return "no_asian"

    try:
        line = float(line)
    except (TypeError, ValueError):
        return "no_asian"

    if line == 0.0:
        return "line_zero"
    elif line in (-0.5, 0.5):
        return "line_half"
    elif line in (-1.0, 1.0):
        return "line_one"
    elif line in (-0.25, 0.25, -0.75, 0.75):
        return "line_quarter"
    else:
        return "other_line"


def evaluate(rows: list, baseline: Phase1Baseline) -> dict:
    """
    Run baseline on all rows and compute Track 2 metrics.

    Also computes a pure-euro reference for comparison.
    """
    all_probs_joint = []
    all_probs_pure = []
    all_labels = []
    league_counter = Counter()
    handicap_counter = Counter()

    for row in rows:
        # Joint baseline
        p_joint = baseline.compute(
            close_euro_h=row.get("close_euro_h"),
            close_euro_d=row.get("close_euro_d"),
            close_euro_a=row.get("close_euro_a"),
            close_asian_line=row.get("close_asian_line"),
            close_asian_upper_water=row.get("close_asian_upper_water"),
            close_asian_lower_water=row.get("close_asian_lower_water"),
        )

        # Pure euro reference (for delta comparison)
        if baseline._has_euro(
            row.get("close_euro_h"),
            row.get("close_euro_d"),
            row.get("close_euro_a"),
        ):
            q = baseline.compute_euro_prior(
                float(row["close_euro_h"]),
                float(row["close_euro_d"]),
                float(row["close_euro_a"]),
            )
            p_pure = torch.tensor(q)
        else:
            p_pure = torch.full((3,), 1.0 / 3.0)

        # Label
        euro_label = EURO_MAP.get(row["label"]["euro_result"], 0)

        all_probs_joint.append(p_joint)
        all_probs_pure.append(p_pure)
        all_labels.append(euro_label)

        league_counter[row.get("league_id", "unknown")] += 1
        handicap_counter[categorise_handicap(row)] += 1

    if not all_probs_joint:
        return {"num_samples": 0, "error": "No valid samples"}

    # Stack tensors
    probs_j = torch.stack(all_probs_joint)  # [N, 3]
    probs_p = torch.stack(all_probs_pure)    # [N, 3]
    labels_t = torch.tensor(all_labels)       # [N]
    N = probs_j.shape[0]
    C = 3

    # ── Track 2 metrics ──
    def _m(probs, label):
        return {
            "accuracy": accuracy_from_probs(probs, labels_t),
            "logloss": logloss_from_probs(probs, labels_t),
            "brier": brier_from_probs(probs, labels_t, C),
            "ece": ece_from_probs(probs, labels_t, C)["ece"],
            "label_counts": class_counts(labels_t, C),
            "prediction_counts": prediction_counts_from_probs(probs, C),
        }

    result = {
        "num_samples": N,
        "baseline_version": "v2_joint_euro_asian",
        "coverage": {
            "total": N,
            "by_handicap_type": dict(handicap_counter),
            "by_league": dict(league_counter),
        },
        "track2": {
            "joint_euro_asian": _m(probs_j, labels_t),
            "pure_euro_reference": _m(probs_p, labels_t),
        },
        "comparison": {
            "delta_accuracy": round(
                _m(probs_j, labels_t)["accuracy"]
                - _m(probs_p, labels_t)["accuracy"],
                6,
            ),
            "delta_logloss": round(
                _m(probs_j, labels_t)["logloss"]
                - _m(probs_p, labels_t)["logloss"],
                6,
            ),
            "delta_brier": round(
                _m(probs_j, labels_t)["brier"]
                - _m(probs_p, labels_t)["brier"],
                6,
            ),
            "delta_ece": round(
                _m(probs_j, labels_t)["ece"]
                - _m(probs_p, labels_t)["ece"],
                6,
            ),
        },
    }

    # ── By handicap type ──
    by_type = {}
    for htype in sorted(handicap_counter.keys()):
        indices = [
            i for i, r in enumerate(rows) if categorise_handicap(r) == htype
        ]
        if len(indices) < 10:
            continue
        p_j = probs_j[indices]
        p_p = probs_p[indices]
        l_t = labels_t[indices]
        by_type[htype] = {
            "count": len(indices),
            "joint": {
                "accuracy": accuracy_from_probs(p_j, l_t),
                "logloss": logloss_from_probs(p_j, l_t),
            },
            "pure_euro": {
                "accuracy": accuracy_from_probs(p_p, l_t),
                "logloss": logloss_from_probs(p_p, l_t),
            },
        }
    result["by_handicap_type"] = by_type

    # ── By league ──
    by_league = {}
    for league in sorted(league_counter.keys()):
        indices = [
            i
            for i, r in enumerate(rows)
            if r.get("league_id", "unknown") == league
        ]
        if len(indices) < 10:
            continue
        p_j = probs_j[indices]
        l_t = labels_t[indices]
        by_league[league] = {
            "count": len(indices),
            "accuracy": accuracy_from_probs(p_j, l_t),
            "logloss": logloss_from_probs(p_j, l_t),
            "brier": brier_from_probs(p_j, l_t, C),
            "ece": ece_from_probs(p_j, l_t, C)["ece"],
        }
    result["by_league"] = by_league

    # ── Track 1 self-consistency (how much does asian change euro?) ──
    kl_vals = []
    for i in range(N):
        kl = kl_divergence(probs_j[i].unsqueeze(0), probs_p[i].unsqueeze(0))
        kl_vals.append(kl)
    result["track1_self_consistency"] = {
        "mean_kl_joint_vs_pure_euro": round(
            sum(kl_vals) / max(len(kl_vals), 1), 6
        ),
        "description": "KL(joint || pure_euro): how much the asian constraint "
                       "shifts the euro prior.  Larger = asian adds more info.",
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1a Baseline v2 Evaluation"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to v6_phase1a.jsonl")
    parser.add_argument("--split-match-ids", type=str, default="",
                        help="Path to split text file (one match_id per line)")
    parser.add_argument("--bookmaker-priority", type=str,
                        default="Pinnacle,Bet365",
                        help="Comma-separated bookmaker priority list")
    parser.add_argument("--out-json", type=str, default="",
                        help="Output JSON path")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit to first N matches (for debugging)")
    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading {args.data} ...")
    all_rows = load_flat_jsonl(args.data)
    print(f"  Total rows: {len(all_rows)}")

    # ── Deduplicate by bookmaker ──
    priority = [b.strip() for b in args.bookmaker_priority.split(",") if b.strip()]
    print(f"  Bookmaker priority: {priority}")
    deduped = deduplicate_by_bookmaker(all_rows, priority)
    print(f"  After dedup: {len(deduped)} unique matches")

    # ── Filter by split ──
    if args.split_match_ids:
        split_ids = load_split_ids(args.split_match_ids)
        print(f"  Split IDs loaded: {len(split_ids)}")
        before = len(deduped)
        deduped = [r for r in deduped if r["match_id"] in split_ids]
        print(f"  After split filter: {len(deduped)} (removed {before - len(deduped)})")

    if args.max_samples and args.max_samples > 0:
        deduped = deduped[: args.max_samples]
        print(f"  Limited to {len(deduped)} samples")

    if not deduped:
        print("ERROR: No samples after filtering. Check split file and data.")
        sys.exit(1)

    # ── Evaluate ──
    print("Computing baselines ...")
    baseline = Phase1Baseline()
    result = evaluate(deduped, baseline)

    # ── Output ──
    output = json.dumps(result, indent=2, default=str)
    print(output)

    if args.out_json:
        out_dir = os.path.dirname(args.out_json) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"\nSaved to {args.out_json}")

    # ── Quick summary ──
    t2 = result.get("track2", {})
    joint = t2.get("joint_euro_asian", {})
    pure = t2.get("pure_euro_reference", {})
    print(f"\n=== Quick Summary ===")
    print(f"  Samples: {result.get('num_samples', 0)}")
    print(f"  Joint  | acc={joint.get('accuracy', 'N/A')}  "
          f"logloss={joint.get('logloss', 'N/A')}  "
          f"brier={joint.get('brier', 'N/A')}  "
          f"ece={joint.get('ece', 'N/A')}")
    print(f"  Pure   | acc={pure.get('accuracy', 'N/A')}  "
          f"logloss={pure.get('logloss', 'N/A')}  "
          f"brier={pure.get('brier', 'N/A')}  "
          f"ece={pure.get('ece', 'N/A')}")
    comp = result.get("comparison", {})
    print(f"  Δ      | acc={comp.get('delta_accuracy', 'N/A')}  "
          f"logloss={comp.get('delta_logloss', 'N/A')}  "
          f"brier={comp.get('delta_brier', 'N/A')}  "
          f"ece={comp.get('delta_ece', 'N/A')}")


if __name__ == "__main__":
    main()
