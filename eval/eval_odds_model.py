"""
OddsMind Model Evaluation Script (P0.4)

Loads a trained (or untrained) model, runs evaluation on a split,
and outputs a JSON summary with per-cutoff breakdown.

Usage:
    python eval/eval_odds_model.py \
        --data data/odds_fixtures/sample_odds_matches_5class.jsonl \
        --model runs/oddsmind_p0_3_5class_smoke/oddsmind_smoke.pth \
        --split-match-ids data/odds_fixtures/splits_p0_4/test_match_ids.txt \
        --cutoffs 90,60,30 --cutoff-mode exhaustive --asian-label-mode 5class \
        --hidden-size 64 --num-layers 2 --num-heads 4 --device cpu
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from eval.odds_metrics import (
    accuracy_from_logits,
    logloss_from_logits,
    brier_from_logits,
    class_counts,
    prediction_counts,
)


def evaluate_model(
    model: OddsMindModel,
    loader: DataLoader,
    device: str,
    asian_num_classes: int,
) -> dict:
    """
    Run evaluation over a DataLoader and return metrics dict.
    """
    model.eval()

    all_euro_logits = []
    all_asian_logits = []
    all_euro_labels = []
    all_asian_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            euro_labels = batch["euro_labels"].to(device)
            asian_labels = batch["asian_labels"].to(device)

            out = model(features, attention_mask=attention_mask)

            all_euro_logits.append(out["euro_logits"].cpu())
            all_asian_logits.append(out["asian_logits"].cpu())
            all_euro_labels.append(euro_labels.cpu())
            all_asian_labels.append(asian_labels.cpu())

    euro_logits = torch.cat(all_euro_logits, dim=0)
    asian_logits = torch.cat(all_asian_logits, dim=0)
    euro_labels = torch.cat(all_euro_labels, dim=0)
    asian_labels = torch.cat(all_asian_labels, dim=0)

    return {
        "num_samples": int(euro_logits.shape[0]),
        "euro": {
            "accuracy": accuracy_from_logits(euro_logits, euro_labels),
            "logloss": logloss_from_logits(euro_logits, euro_labels),
            "brier": brier_from_logits(euro_logits, euro_labels, 3),
            "label_counts": class_counts(euro_labels, 3),
            "prediction_counts": prediction_counts(euro_logits, 3),
        },
        "asian": {
            "accuracy": accuracy_from_logits(asian_logits, asian_labels),
            "logloss": logloss_from_logits(asian_logits, asian_labels),
            "brier": brier_from_logits(asian_logits, asian_labels, asian_num_classes),
            "label_counts": class_counts(asian_labels, asian_num_classes),
            "prediction_counts": prediction_counts(asian_logits, asian_num_classes),
        },
    }


def evaluate_by_cutoff(
    model: OddsMindModel,
    dataset: OddsDataset,
    collator: OddsCollator,
    device: str,
    asian_num_classes: int,
    cutoffs: list,
) -> dict:
    """Evaluate per cutoff by building separate datasets for each cutoff."""
    by_cutoff = {}
    for cutoff in cutoffs:
        ds_cutoff = OddsDataset(
            dataset._jsonl_path,
            max_seq_len=dataset.max_seq_len,
            cutoff_minutes=cutoff,
            cutoff_mode="none",
            asian_label_mode=dataset.asian_label_mode,
            allowed_match_ids=dataset._allowed_ids,
        )
        loader = DataLoader(ds_cutoff, batch_size=8, shuffle=False, collate_fn=collator)
        if len(ds_cutoff) == 0:
            by_cutoff[str(int(cutoff))] = {"num_samples": 0}
            continue
        result = evaluate_model(model, loader, device, asian_num_classes)
        by_cutoff[str(int(cutoff))] = {
            "num_samples": result["num_samples"],
            "euro_accuracy": result["euro"]["accuracy"],
            "asian_accuracy": result["asian"]["accuracy"],
        }
    return by_cutoff


def main():
    parser = argparse.ArgumentParser(description="OddsMind Model Evaluation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--split-match-ids", type=str, default="")
    parser.add_argument("--cutoffs", type=str, default="")
    parser.add_argument("--cutoff-mode", type=str, default="exhaustive")
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    asian_num_classes = 5 if args.asian_label_mode == "5class" else 3

    # Load model
    config = OddsMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        asian_num_classes=asian_num_classes,
    )

    if args.untrained or not args.model:
        model = OddsMindModel(config).to(device)
        model.eval()
    else:
        model = OddsMindModel(config).to(device)
        state_dict = torch.load(args.model, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    # Load match IDs if split specified
    allowed_ids = None
    if args.split_match_ids:
        allowed_ids = load_match_ids_from_file(args.split_match_ids)

    cutoffs = None
    if args.cutoffs:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    # Dataset for the split
    ds = OddsDataset(
        args.data,
        max_seq_len=args.max_seq_len,
        cutoffs=cutoffs,
        cutoff_mode=args.cutoff_mode,
        asian_label_mode=args.asian_label_mode,
        allowed_match_ids=allowed_ids,
    )

    collator = OddsCollator()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    result = evaluate_model(model, loader, device, asian_num_classes)

    # By-cutoff breakdown
    if cutoffs and args.cutoff_mode == "exhaustive":
        result["by_cutoff"] = evaluate_by_cutoff(
            model, ds, collator, device, asian_num_classes, cutoffs
        )

    # Count unique matches
    unique_matches = len(allowed_ids) if allowed_ids else ds.num_raw_matches
    result["num_matches"] = unique_matches

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
