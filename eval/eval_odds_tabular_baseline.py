"""
OddsMind Tabular Baseline Evaluation Script (P0.5B)

Evaluates a trained logistic/tiny_mlp model on a split.

Usage:
    python eval/eval_odds_tabular_baseline.py \
        --data ...5class.jsonl --model runs/.../odds_tabular_logistic.pth \
        --model-type logistic --split-match-ids .../test_match_ids.txt \
        --cutoffs 90,60,30 --cutoff-mode exhaustive --asian-label-mode 5class \
        --device cpu --out-json runs/tabular_eval.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from dataset.odds_tabular_dataset import OddsTabularDataset
from dataset.odds_split import load_match_ids_from_file
from model.odds_tabular_baselines import OddsLogisticRegression, OddsTinyMLP
from eval.odds_metrics import (
    accuracy_from_logits,
    logloss_from_logits,
    brier_from_logits,
    class_counts,
    prediction_counts,
)


def collate_tabular(batch):
    features = torch.stack([item["features"] for item in batch])
    euro_labels = torch.tensor([item["euro_label"] for item in batch])
    asian_labels = torch.tensor([item["asian_label"] for item in batch])
    return {"features": features, "euro_labels": euro_labels, "asian_labels": asian_labels}


def evaluate(model, loader, device, asian_num_classes):
    model.eval()
    eu_logits, as_logits = [], []
    eu_labels, as_labels = [], []

    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(device)
            out = model(f)
            eu_logits.append(out["euro_logits"].cpu())
            as_logits.append(out["asian_logits"].cpu())
            eu_labels.append(batch["euro_labels"])
            as_labels.append(batch["asian_labels"])

    eu_l = torch.cat(eu_logits, 0)
    as_l = torch.cat(as_logits, 0)
    eu_lb = torch.cat(eu_labels, 0)
    as_lb = torch.cat(as_labels, 0)

    return {
        "num_samples": int(eu_l.shape[0]),
        "euro": {
            "accuracy": accuracy_from_logits(eu_l, eu_lb),
            "logloss": logloss_from_logits(eu_l, eu_lb),
            "brier": brier_from_logits(eu_l, eu_lb, 3),
            "label_counts": class_counts(eu_lb, 3),
            "prediction_counts": prediction_counts(eu_l, 3),
        },
        "asian": {
            "accuracy": accuracy_from_logits(as_l, as_lb),
            "logloss": logloss_from_logits(as_l, as_lb),
            "brier": brier_from_logits(as_l, as_lb, asian_num_classes),
            "label_counts": class_counts(as_lb, asian_num_classes),
            "prediction_counts": prediction_counts(as_l, asian_num_classes),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="OddsMind Tabular Baseline Eval")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--model-type", type=str, default="logistic", choices=["logistic", "tiny_mlp"])
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--split-match-ids", type=str, default="")
    parser.add_argument("--cutoffs", type=str, default="")
    parser.add_argument("--cutoff-mode", type=str, default="exhaustive")
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--min-events", type=int, default=1)
    args = parser.parse_args()

    device = args.device
    asian_num_classes = 5 if args.asian_label_mode == "5class" else 3

    if args.untrained or not args.model:
        if args.model_type == "logistic":
            model = OddsLogisticRegression(asian_num_classes=asian_num_classes).to(device)
        else:
            model = OddsTinyMLP(hidden_size=args.hidden_size, asian_num_classes=asian_num_classes).to(device)
        model.eval()
    else:
        if args.model_type == "logistic":
            model = OddsLogisticRegression(asian_num_classes=asian_num_classes).to(device)
        else:
            model = OddsTinyMLP(hidden_size=args.hidden_size, asian_num_classes=asian_num_classes).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

    cutoffs = None
    if args.cutoffs:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    allowed_ids = None
    if args.split_match_ids:
        allowed_ids = load_match_ids_from_file(args.split_match_ids)

    ds = OddsTabularDataset(
        args.data, max_seq_len=args.max_seq_len,
        cutoffs=cutoffs, cutoff_mode=args.cutoff_mode,
        min_events=args.min_events, asian_label_mode=args.asian_label_mode,
        allowed_match_ids=allowed_ids,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_tabular)
    result = evaluate(model, loader, device, asian_num_classes)

    if cutoffs:
        by_cutoff = {}
        for cutoff in cutoffs:
            ds_c = OddsTabularDataset(
                args.data, max_seq_len=args.max_seq_len,
                cutoff_minutes=cutoff, cutoff_mode="none",
                asian_label_mode=args.asian_label_mode,
                allowed_match_ids=allowed_ids,
            )
            if len(ds_c) == 0:
                by_cutoff[str(int(cutoff))] = {"num_samples": 0}
                continue
            loader_c = DataLoader(ds_c, batch_size=args.batch_size, shuffle=False, collate_fn=collate_tabular)
            r = evaluate(model, loader_c, device, asian_num_classes)
            by_cutoff[str(int(cutoff))] = {
                "num_samples": r["num_samples"],
                "euro_accuracy": r["euro"]["accuracy"],
                "asian_accuracy": r["asian"]["accuracy"],
            }
        result["by_cutoff"] = by_cutoff

    result["num_matches"] = len(allowed_ids) if allowed_ids else ds.num_raw_matches

    output = json.dumps(result, indent=2, default=str)
    print(output)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(output + "\n")


if __name__ == "__main__":
    main()
