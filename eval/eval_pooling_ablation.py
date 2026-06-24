"""
P1.16 Pooling Ablation Script

Evaluates a single checkpoint with all three pooling modes on the same
test split and outputs a comparison JSON, without retraining.

Usage:
    python eval/eval_pooling_ablation.py \
        --data data/odds_fixtures/sample_odds_matches_5class.jsonl \
        --model runs/oddsmind_p0_3_5class_smoke/oddsmind_smoke.pth \
        --split-match-ids data/odds_fixtures/splits_p0_4/test_match_ids.txt \
        --cutoffs 90,60,30 --cutoff-mode exhaustive --asian-label-mode 5class \
        --hidden-size 256 --num-layers 4 --num-heads 8 --device cpu
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
    ece_from_logits,
)


def evaluate_single(model, loader, device, asian_num_classes) -> dict:
    """Run evaluation and return scalar metrics."""
    model.eval()
    all_eu_log, all_as_log = [], []
    all_eu_lbl, all_as_lbl = [], []

    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(device)
            m = batch["attention_mask"].to(device)
            el = batch["euro_labels"].to(device)
            al = batch["asian_labels"].to(device)

            out = model(f, attention_mask=m)
            all_eu_log.append(out["euro_logits"].cpu())
            all_as_log.append(out["asian_logits"].cpu())
            all_eu_lbl.append(el.cpu())
            all_as_lbl.append(al.cpu())

    eu_log = torch.cat(all_eu_log, dim=0)
    as_log = torch.cat(all_as_log, dim=0)
    eu_lbl = torch.cat(all_eu_lbl, dim=0)
    as_lbl = torch.cat(all_as_lbl, dim=0)

    return {
        "num_samples": int(eu_log.shape[0]),
        "euro_accuracy": accuracy_from_logits(eu_log, eu_lbl),
        "euro_logloss": logloss_from_logits(eu_log, eu_lbl),
        "euro_brier": brier_from_logits(eu_log, eu_lbl, 3),
        "euro_ece": ece_from_logits(eu_log, eu_lbl, 3)["ece"],
        "asian_accuracy": accuracy_from_logits(as_log, as_lbl),
        "asian_logloss": logloss_from_logits(as_log, as_lbl),
        "asian_brier": brier_from_logits(as_log, as_lbl, asian_num_classes),
        "asian_ece": ece_from_logits(as_log, as_lbl, asian_num_classes)["ece"],
    }


def main():
    parser = argparse.ArgumentParser(description="P1.16 Pooling Ablation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
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
    parser.add_argument("--transformer-backend", type=str, default="odds_native")
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    device = args.device
    asian_num_classes = 5 if args.asian_label_mode == "5class" else 3

    # Load model weights once
    ckp = torch.load(args.model, map_location=device)
    if isinstance(ckp, dict) and "model_state_dict" in ckp:
        state_dict = ckp["model_state_dict"]
    else:
        state_dict = ckp

    # Dataset
    allowed_ids = None
    if args.split_match_ids:
        allowed_ids = load_match_ids_from_file(args.split_match_ids)

    cutoffs = None
    if args.cutoffs:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    collator = OddsCollator()

    results = {}
    for mode in ["mean", "attention", "cls"]:
        print(f"\n=== Pooling mode: {mode} ===")
        config = OddsMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            asian_num_classes=asian_num_classes,
            transformer_backend=args.transformer_backend,
            pooling_mode=mode,
        )

        model = OddsMindModel(config).to(device)
        # Load compatible weights (pooling-specific keys are new and won't match)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            pooling_keys = [k for k in missing if "pool" in k or "cls" in k]
            non_pooling_keys = [k for k in missing if k not in pooling_keys]
            if non_pooling_keys:
                print(f"  WARNING: {len(non_pooling_keys)} missing non-pooling keys: {non_pooling_keys[:5]}...")
            if pooling_keys:
                print(f"  New pooling keys (expected): {pooling_keys}")
        model.eval()

        ds = OddsDataset(
            args.data,
            max_seq_len=args.max_seq_len,
            cutoffs=cutoffs,
            cutoff_mode=args.cutoff_mode,
            asian_label_mode=args.asian_label_mode,
            allowed_match_ids=allowed_ids,
            feature_schema_version="v2",
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        metrics = evaluate_single(model, loader, device, asian_num_classes)
        results[mode] = metrics
        print(f"  Euro acc={metrics['euro_accuracy']:.4f} loss={metrics['euro_logloss']:.4f} ece={metrics['euro_ece']:.4f}")
        print(f"  Asian acc={metrics['asian_accuracy']:.4f} loss={metrics['asian_logloss']:.4f} ece={metrics['asian_ece']:.4f}")

    # Print comparison table
    print("\n=== Ablation Summary ===")
    print(f"{'Metric':<25} {'mean':>10} {'attention':>10} {'cls':>10}")
    print("-" * 57)
    for metric in ["euro_accuracy", "euro_logloss", "euro_brier", "euro_ece",
                    "asian_accuracy", "asian_logloss", "asian_brier", "asian_ece"]:
        vals = [str(round(results[m][metric], 4)) for m in ["mean", "attention", "cls"]]
        print(f"{metric:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    output = {"pooling_ablation": results}
    print("\n" + json.dumps(output, indent=2, default=str))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(output, indent=2, default=str) + "\n")
        print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
