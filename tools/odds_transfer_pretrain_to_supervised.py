"""
OddsMind Pretrain-to-Supervised Transfer Tool (P0.7B)

Usage:
    # Dry-run (no checkpoint written)
    python tools/odds_transfer_pretrain_to_supervised.py \
        --pretrain-checkpoint runs/.../oddsmind_pretrain_masked_reconstruction.pth \
        --out runs/.../supervised_init.pth --dry-run \
        --hidden-size 64 --num-layers 2 --num-heads 4 \
        --asian-label-mode 5class --transformer-backend odds_native

    # Actual transfer
    python tools/odds_transfer_pretrain_to_supervised.py \
        --pretrain-checkpoint ... --out ... \
        --hidden-size 64 --num-layers 2 --num-heads 4 \
        --asian-label-mode 5class --transformer-backend odds_native
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from model.oddsmind_weight_transfer import load_pretrained_encoder_transformer


def main():
    parser = argparse.ArgumentParser(description="OddsMind Pretrain → Supervised Transfer")
    parser.add_argument("--pretrain-checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--transformer-backend", type=str, default="odds_native")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asian_num_classes = 5 if args.asian_label_mode == "5class" else 3

    config = OddsMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        asian_num_classes=asian_num_classes,
        transformer_backend=args.transformer_backend,
    )

    model = OddsMindModel(config)

    report = load_pretrained_encoder_transformer(
        model,
        args.pretrain_checkpoint,
        backend=args.transformer_backend,
        strict_shapes=True,
    )

    print("=== Transfer Report ===")
    for k in ["transferred", "skipped", "missing", "unexpected", "shape_mismatch"]:
        print(f"  {k}: {report[k]}")
    if report["transferred_keys"]:
        print(f"  transferred_keys ({len(report['transferred_keys'])}):")
        for k in report["transferred_keys"][:5]:
            print(f"    {k}")
        if len(report["transferred_keys"]) > 5:
            print(f"    ... and {len(report['transferred_keys']) - 5} more")
    if report["skipped_keys"]:
        print(f"  skipped_keys ({len(report['skipped_keys'])}):")
        for k in report["skipped_keys"]:
            print(f"    {k}")
    if report["shape_mismatch_details"]:
        print(f"  shape_mismatches:")
        for k, ps, ss in report["shape_mismatch_details"]:
            print(f"    {k}: pretrain {list(ps)} vs supervised {list(ss)}")

    if args.dry_run:
        print("\n[Dry-run] No checkpoint written.")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_data = {
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "asian_num_classes": asian_num_classes,
            "transformer_backend": args.transformer_backend,
        },
        "source_pretrain_checkpoint": args.pretrain_checkpoint,
        "transfer_report": report,
    }
    torch.save(save_data, args.out)
    print(f"\nSaved supervised init to {args.out}")


if __name__ == "__main__":
    main()
