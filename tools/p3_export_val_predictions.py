"""Export validation predictions from saved P3 checkpoints."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.odds_dataset import OddsDataset
from model.odds_patch_itransformer import OddsPatchITransformer
from model.odds_patch_itransformer_v2 import OddsPatchITransformerV2
from tools.p3_save_raw_pooled_mlp_baseline import (
    EURO_MAP as RAW_EURO_MAP,
    RawPooledMLP,
    build_dataset as build_raw_dataset,
    load_rows_for_ids,
    load_split_ids,
)
from tools.p3_train_patch_itransformer import build_bucketed_dataset
from tools.p3b_train_patch_itransformer import build_bucketed_dataset_v2


INV_LABEL = {0: "home", 1: "draw", 2: "away"}


def _load_checkpoint(run_dir: str | Path, device: torch.device) -> dict:
    path = Path(run_dir) / "best_model.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _load_report(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _model_config(ckpt: dict, report: dict) -> dict:
    config = dict(ckpt.get("config") or {})
    report_config = report.get("config") or {}
    for key in ["d_model", "n_heads", "n_layers", "d_ff", "dropout"]:
        if key not in config and key in report_config:
            config[key] = report_config[key]
    return config


def _p3b_preprocessing_config(ckpt: dict, report: dict, args) -> tuple[str, bool]:
    ckpt_config = ckpt.get("config") or {}
    report_config = report.get("config") or {}
    feature_groups = args.feature_groups or ckpt_config.get("feature_groups", report_config.get("feature_groups"))
    clean_missing = ckpt_config.get("clean_missing_markets", report_config.get("clean_missing_markets"))
    if args.clean_missing_markets:
        clean_missing = True
    elif args.no_clean_missing_markets:
        clean_missing = False
    if feature_groups is None or clean_missing is None:
        raise ValueError(
            "P3b checkpoint/report does not record feature_groups and clean_missing_markets; "
            "refusing to export predictions with guessed preprocessing."
        )
    return str(feature_groups), bool(clean_missing)


@torch.no_grad()
def _predict_batches(model, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        logits = model(x[start : start + batch_size].to(device))
        chunks.append(F.softmax(logits, dim=-1).clamp(1e-9, 1 - 1e-9).cpu())
    return torch.cat(chunks, dim=0)


def _write_predictions(path: str | Path, match_ids: list[str], labels: torch.Tensor, probs: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["match_id", "y_true", "p_home", "p_draw", "p_away", "pred_class", "correct"],
        )
        writer.writeheader()
        pred = probs.argmax(dim=-1)
        for idx, mid in enumerate(match_ids):
            writer.writerow(
                {
                    "match_id": mid,
                    "y_true": int(labels[idx].item()),
                    "p_home": float(probs[idx, 0].item()),
                    "p_draw": float(probs[idx, 1].item()),
                    "p_away": float(probs[idx, 2].item()),
                    "pred_class": int(pred[idx].item()),
                    "correct": int(pred[idx].item() == labels[idx].item()),
                }
            )


def export_raw_mlp(args, device: torch.device) -> int:
    ckpt = _load_checkpoint(args.run_dir, device)
    val_ids = load_split_ids(args.val_ids)
    rows = load_rows_for_ids(args.data, val_ids)
    x_val, y_val = build_raw_dataset(rows)
    match_ids = [row.get("match_id", "") for row in rows if row.get("label", {}).get("euro_result") in RAW_EURO_MAP]
    model = RawPooledMLP(input_dim=x_val.shape[1]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    probs = _predict_batches(model, x_val, args.batch_size, device)
    _write_predictions(args.out, match_ids, y_val, probs)
    return len(match_ids)


def _load_odds_dataset(args) -> OddsDataset:
    val_ids = load_split_ids(args.val_ids)
    return OddsDataset(
        args.data,
        max_seq_len=args.max_seq_len,
        feature_schema_version="v6_event",
        allowed_match_ids=val_ids,
        min_events=1,
    )


def export_p3a(args, device: torch.device) -> int:
    ckpt = _load_checkpoint(args.run_dir, device)
    report = _load_report(args.run_dir)
    config = _model_config(ckpt, report)
    val_ds = _load_odds_dataset(args)
    x_val, y_val = build_bucketed_dataset(val_ds.samples, val_ds)
    match_ids = [sample.get("match_id", "") for sample in val_ds.samples]
    model = OddsPatchITransformer(
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 2),
        d_ff=config.get("d_ff", 128),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    probs = _predict_batches(model, x_val, args.batch_size, device)
    _write_predictions(args.out, match_ids, y_val, probs)
    return len(match_ids)


def export_p3b(args, device: torch.device) -> int:
    ckpt = _load_checkpoint(args.run_dir, device)
    report = _load_report(args.run_dir)
    config = _model_config(ckpt, report)
    feature_groups, clean_missing_markets = _p3b_preprocessing_config(ckpt, report, args)
    val_ds = _load_odds_dataset(args)
    x_val, y_val = build_bucketed_dataset_v2(
        val_ds.samples,
        val_ds,
        feature_groups=feature_groups,
        clean_missing_markets=clean_missing_markets,
    )
    match_ids = [sample.get("match_id", "") for sample in val_ds.samples]
    model = OddsPatchITransformerV2(
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 2),
        d_ff=config.get("d_ff", 128),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    probs = _predict_batches(model, x_val, args.batch_size, device)
    _write_predictions(args.out, match_ids, y_val, probs)
    return len(match_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export P3 validation predictions")
    parser.add_argument("--model", choices=["raw_mlp", "p3a", "p3b"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--feature-groups", type=str, default=None)
    clean_group = parser.add_mutually_exclusive_group()
    clean_group.add_argument("--clean-missing-markets", action="store_true")
    clean_group.add_argument("--no-clean-missing-markets", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.model == "raw_mlp":
        count = export_raw_mlp(args, device)
    elif args.model == "p3a":
        count = export_p3a(args, device)
    else:
        count = export_p3b(args, device)
    print(f"Exported {count} predictions to {args.out}")


if __name__ == "__main__":
    main()
