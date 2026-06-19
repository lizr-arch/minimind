"""
OddsMind Baseline Evaluation Script (P0.5A)

Runs all deterministic baselines on a split and outputs metrics JSON.

Usage:
    python eval/eval_odds_baselines.py \
        --data data/odds_fixtures/sample_odds_matches_5class.jsonl \
        --split-match-ids data/odds_fixtures/splits_p0_4/test_match_ids.txt \
        --cutoffs 90,60,30 --cutoff-mode exhaustive --asian-label-mode 5class \
        --out-json runs/baseline_eval.json
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from dataset.odds_dataset import OddsDataset, EURO_MAP, ASIAN_MAP_5CLASS
from dataset.odds_split import load_match_ids_from_file
from eval.odds_metrics import (
    accuracy_from_probs,
    logloss_from_probs,
    brier_from_probs,
    class_counts,
    prediction_counts_from_probs,
)
from eval.odds_baselines import (
    lowest_odds_euro,
    implied_prob_euro,
    low_water_asian_3class,
    low_water_asian_5class,
    uniform_baseline,
    euro_probs_to_tensor,
    asian_probs_to_tensor,
)

EURO_KEYS = ["home", "draw", "away"]
ASIAN_3_KEYS = ["upper", "push", "lower"]
ASIAN_5_KEYS = ["upper_full_win", "upper_half_win", "push",
                "upper_half_loss", "upper_full_loss"]


def evaluate_baselines(
    dataset: OddsDataset,
    asian_label_mode: str,
) -> dict:
    """
    Run all baselines over a dataset, returning metrics dict.
    """
    asian_num_classes = 5 if asian_label_mode == "5class" else 3

    # Accumulators
    results = {
        "euro_lowest_odds": [],
        "euro_implied_prob": [],
        "asian_low_water": [],
        "uniform_euro": [],
        "uniform_asian": [],
    }
    all_euro_labels = []
    all_asian_labels = []
    all_cutoffs = []

    for i in range(len(dataset)):
        item = dataset[i]  # dict with 'features' and labels
        # We need the raw sample to get the events.  The dataset stores samples
        # differently based on cutoff_mode.  We'll build a helper that reads
        # the underlying raw match and filters.
        # Simpler: use the dataset's internal samples to get the original match.
        # But the item only has features tensor + labels.  We need the events.
        # Workaround: re-read the match from the stored internal samples.

    # Actually, we'll iterate the dataset's internal sample list directly
    # (bypassing features tensor building) and do the baseline logic ourselves.

    return {}


def evaluate_baselines_direct(
    dataset: OddsDataset,
    asian_label_mode: str,
    cutoffs: list,
) -> dict:
    """
    Evaluate baselines by iterating the dataset's internal sample list
    and applying baseline functions to each sample's timeline.
    """
    asian_num_classes = 5 if asian_label_mode == "5class" else 3
    asian_map = ASIAN_MAP_5CLASS if asian_label_mode == "5class" else {
        "upper": 0, "push": 1, "lower": 2}

    # Collect per-sample results
    records = []

    for sample in dataset.samples:
        timeline = sample.get("odds_timeline", [])

        # Determine the cutoff for this sample
        cutoff = sample.get("cutoff_minutes", 0)
        cutoff_key = str(int(cutoff))

        # Filter timeline by cutoff (in case sample is raw match, not pre-cutoff)
        filtered = [e for e in timeline if e["minutes_before_kickoff"] >= cutoff]
        if not filtered:
            continue

        # Labels
        euro_label = EURO_MAP[sample["label"]["euro_result"]]
        asian_label = asian_map[sample["label"]["asian_result"]]

        # Euro baselines
        eur_low = lowest_odds_euro(filtered)
        eur_imp = implied_prob_euro(filtered)
        eur_low_probs = euro_probs_to_tensor(eur_low)
        eur_imp_probs = euro_probs_to_tensor(eur_imp)

        # Asian baselines
        if asian_label_mode == "5class":
            asn_low = low_water_asian_5class(filtered)
            asn_probs = asian_probs_to_tensor(asn_low, 5)
        else:
            asn_low = low_water_asian_3class(filtered)
            asn_probs = asian_probs_to_tensor(asn_low, 3)

        # Uniform
        unif_eur = torch.full((3,), 1.0/3.0)
        unif_asn = torch.full((asian_num_classes,), 1.0/asian_num_classes)

        records.append({
            "cutoff_key": cutoff_key,
            "euro_label": euro_label,
            "asian_label": asian_label,
            "eur_low_probs": eur_low_probs,
            "eur_imp_probs": eur_imp_probs,
            "asn_probs": asn_probs,
            "unif_eur": unif_eur,
            "unif_asn": unif_asn,
        })

    if not records:
        return {"num_samples": 0, "num_matches": 0}

    # Aggregate
    def agg_metrics(probs, labels, num_classes):
        return {
            "accuracy": accuracy_from_probs(probs, labels),
            "logloss": logloss_from_probs(probs, labels),
            "brier": brier_from_probs(probs, labels, num_classes),
        }

    eur_low_p = torch.stack([r["eur_low_probs"] for r in records])
    eur_imp_p = torch.stack([r["eur_imp_probs"] for r in records])
    asn_p = torch.stack([r["asn_probs"] for r in records])
    unif_eur_p = torch.stack([r["unif_eur"] for r in records])
    unif_asn_p = torch.stack([r["unif_asn"] for r in records])

    eur_labels_t = torch.tensor([r["euro_label"] for r in records])
    asn_labels_t = torch.tensor([r["asian_label"] for r in records])

    result = {
        "num_samples": len(records),
        "baselines": {
            "euro_lowest_odds": {
                **agg_metrics(eur_low_p, eur_labels_t, 3),
                "label_counts": class_counts(eur_labels_t, 3),
                "prediction_counts": prediction_counts_from_probs(eur_low_p, 3),
            },
            "euro_implied_prob": {
                **agg_metrics(eur_imp_p, eur_labels_t, 3),
                "label_counts": class_counts(eur_labels_t, 3),
                "prediction_counts": prediction_counts_from_probs(eur_imp_p, 3),
            },
            "asian_low_water": {
                **agg_metrics(asn_p, asn_labels_t, asian_num_classes),
                "label_counts": class_counts(asn_labels_t, asian_num_classes),
                "prediction_counts": prediction_counts_from_probs(asn_p, asian_num_classes),
            },
            "uniform": {
                "euro": agg_metrics(unif_eur_p, eur_labels_t, 3),
                "asian": agg_metrics(unif_asn_p, asn_labels_t, asian_num_classes),
            },
        },
    }

    # By cutoff
    # Collect all unique cutoff keys from records
    all_cutoff_keys = sorted(set(r["cutoff_key"] for r in records), key=int)
    by_cutoff = {}
    for ck in all_cutoff_keys:
        c_records = [r for r in records if r["cutoff_key"] == ck]
        if not c_records:
            continue
        ce_p = torch.stack([r["eur_low_probs"] for r in c_records])
        ca_p = torch.stack([r["asn_probs"] for r in c_records])
        ce_l = torch.tensor([r["euro_label"] for r in c_records])
        ca_l = torch.tensor([r["asian_label"] for r in c_records])
        ci_p = torch.stack([r["eur_imp_probs"] for r in c_records])
        by_cutoff[ck] = {
            "num_samples": len(c_records),
            "euro_lowest_odds_accuracy": accuracy_from_probs(ce_p, ce_l),
            "euro_implied_prob_accuracy": accuracy_from_probs(ci_p, ce_l),
            "asian_low_water_accuracy": accuracy_from_probs(ca_p, ca_l),
        }
    result["by_cutoff"] = by_cutoff

    # Unique matches
    match_ids = set()
    for s in dataset.samples:
        mid = s.get("match_id", s.get("source_match_id", ""))
        match_ids.add(mid)
    result["num_matches"] = len(match_ids)

    return result


def main():
    parser = argparse.ArgumentParser(description="OddsMind Baseline Evaluation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split-match-ids", type=str, default="")
    parser.add_argument("--cutoffs", type=str, default="")
    parser.add_argument("--cutoff-mode", type=str, default="exhaustive")
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--min-events", type=int, default=1)
    args = parser.parse_args()

    allowed_ids = None
    if args.split_match_ids:
        allowed_ids = load_match_ids_from_file(args.split_match_ids)

    cutoffs = None
    if args.cutoffs:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    ds = OddsDataset(
        args.data,
        max_seq_len=args.max_seq_len,
        cutoffs=cutoffs,
        cutoff_mode=args.cutoff_mode,
        asian_label_mode=args.asian_label_mode,
        allowed_match_ids=allowed_ids,
        min_events=args.min_events,
    )

    result = evaluate_baselines_direct(ds, args.asian_label_mode, cutoffs or [])

    output = json.dumps(result, indent=2, default=str)
    print(output)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"Saved to {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
