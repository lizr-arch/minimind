"""
OddsMind Pretrain Evaluation Script (P0.7)

Usage:
    python eval/eval_odds_pretrain.py --task masked_reconstruction ...
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from model.model_oddsmind_pretrain import OddsPretrainConfig, OddsMindPretrainModel
from dataset.odds_pretrain_dataset import OddsPretrainDataset
from dataset.odds_pretrain_collator import OddsPretrainCollator
from dataset.odds_split import load_match_ids_from_file


def prepare_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="OddsMind Pretrain Eval")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--task", type=str, default="masked_reconstruction")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--split-match-ids", type=str, default="")
    parser.add_argument("--cutoffs", type=str, default="")
    parser.add_argument("--cutoff-mode", type=str, default="exhaustive")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--transformer-backend", type=str, default="odds_native")
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--min-events", type=int, default=2)
    args = parser.parse_args()

    device = args.device
    config = OddsPretrainConfig(
        task=args.task, hidden_size=args.hidden_size,
        num_layers=args.num_layers, num_heads=args.num_heads,
        transformer_backend=args.transformer_backend,
        max_seq_len=args.max_seq_len,
    )

    if args.untrained or not args.model:
        model = OddsMindPretrainModel(config).to(device)
    else:
        model = OddsMindPretrainModel(config).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    cutoffs = None
    if args.cutoffs:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    allowed_ids = load_match_ids_from_file(args.split_match_ids) if args.split_match_ids else None

    ds = OddsPretrainDataset(args.data, max_seq_len=args.max_seq_len,
                             cutoffs=cutoffs, cutoff_mode=args.cutoff_mode,
                             allowed_match_ids=allowed_ids, task=args.task,
                             mask_ratio=args.mask_ratio, min_events=args.min_events)
    collator = OddsPretrainCollator()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch(batch, device)
            out = model(features=batch["features"], attention_mask=batch.get("attention_mask"),
                        target_features=batch.get("target_features"),
                        target_mask=batch.get("target_mask"),
                        target_event=batch.get("target_event"))
            losses.append(out["loss"].item())

    result = {
        "task": args.task,
        "num_samples": len(ds),
        "num_matches": len(allowed_ids) if allowed_ids else ds.num_raw_matches,
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "mse": sum(losses) / len(losses) if losses else 0.0,
    }

    if cutoffs:
        by_cutoff = {}
        for cutoff in cutoffs:
            ds_c = OddsPretrainDataset(args.data, max_seq_len=args.max_seq_len,
                                       cutoff_minutes=cutoff, cutoff_mode="none",
                                       allowed_match_ids=allowed_ids, task=args.task,
                                       mask_ratio=args.mask_ratio, min_events=args.min_events)
            if len(ds_c) == 0:
                by_cutoff[str(int(cutoff))] = {"num_samples": 0}
                continue
            loader_c = DataLoader(ds_c, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
            l = []
            with torch.no_grad():
                for batch in loader_c:
                    batch = prepare_batch(batch, device)
                    out = model(features=batch["features"], attention_mask=batch.get("attention_mask"),
                                target_features=batch.get("target_features"),
                                target_mask=batch.get("target_mask"),
                                target_event=batch.get("target_event"))
                    l.append(out["loss"].item())
            by_cutoff[str(int(cutoff))] = {"num_samples": len(ds_c), "loss": sum(l)/len(l) if l else 0}
        result["by_cutoff"] = by_cutoff

    output = json.dumps(result, indent=2, default=str)
    print(output)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(output + "\n")


if __name__ == "__main__":
    main()
