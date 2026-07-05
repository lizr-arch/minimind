"""P3.4 out-of-fold market fusion diagnostic.

OOF stacking flow:
1. Split official train rows into K folds.
2. For each market and fold, train on K-1 folds and predict the held-out fold.
3. Train a fusion head on train-only OOF predictions.
4. Train final market models on full train and evaluate base/fusion variants on official val.

The script never reads test ids and never fits fusion on official val.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p3_3_market_fusion_diagnostic import (
    MARKETS,
    apply_standardizer,
    average_blend,
    build_dataset,
    build_report_payload as _base_report_payload,
    evaluate_probs,
    make_fusion_inputs,
    predict_probs,
    run_one_seed as run_holdout_seed,
    summarize_runs as summarize_holdout_runs,
    train_fusion_head,
    train_market_model,
    write_report as write_holdout_style_report,
)
from tools.p3_save_raw_pooled_mlp_baseline import load_rows_for_ids, load_split_ids


def assign_folds(rows: list[dict], n_folds: int, seed: int) -> list[int]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_folds > len(rows):
        raise ValueError("n_folds cannot exceed row count")
    order = list(range(len(rows)))
    order.sort(key=lambda idx: str(rows[idx].get("match_id", "")))
    rng = random.Random(seed)
    rng.shuffle(order)
    folds = [None for _ in rows]
    for position, row_idx in enumerate(order):
        folds[row_idx] = position % n_folds
    return [int(fold) for fold in folds]


def _select_rows(rows: list[dict], indices: list[int]) -> list[dict]:
    return [rows[idx] for idx in indices]


def build_oof_prediction_frame(
    fold_predictions: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    market_order: list[str],
    labels: torch.Tensor,
) -> dict:
    probs_by_market = {}
    missing = torch.zeros(len(labels), dtype=torch.bool)
    for market in market_order:
        market_probs = torch.full((len(labels), 3), float("nan"))
        for _, (indices, probs) in fold_predictions[market].items():
            market_probs[indices.long()] = probs
        missing |= torch.isnan(market_probs).any(dim=1)
        probs_by_market[market] = market_probs
    if torch.any(missing):
        keep = ~missing
        probs_by_market = {market: probs[keep] for market, probs in probs_by_market.items()}
        labels = labels[keep]
    features = make_fusion_inputs(probs_by_market, market_order)
    return {
        "features": features,
        "labels": labels,
        "missing_count": int(missing.sum().item()),
    }


def _train_oof_market_predictions(
    rows: list[dict],
    market: str,
    folds: list[int],
    seed: int,
    args,
    device: torch.device,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    full_x, full_y, _ = build_dataset(rows, market)
    fold_predictions = {}
    for fold in range(args.n_folds):
        train_indices = [idx for idx, row_fold in enumerate(folds) if row_fold != fold]
        holdout_indices = [idx for idx, row_fold in enumerate(folds) if row_fold == fold]
        x_train = full_x[train_indices]
        y_train = full_y[train_indices]
        x_holdout = full_x[holdout_indices]
        y_holdout = full_y[holdout_indices]
        model, scaler, _ = train_market_model(
            x_train,
            y_train,
            x_holdout,
            y_holdout,
            seed=seed + fold,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        probs = predict_probs(model, apply_standardizer(x_holdout, scaler), args.batch_size * 2, device)
        fold_predictions[fold] = (torch.tensor(holdout_indices, dtype=torch.long), probs)
    return fold_predictions, full_y


def _train_full_market_model(
    train_rows: list[dict],
    val_rows: list[dict],
    market: str,
    seed: int,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    x_train, y_train, _ = build_dataset(train_rows, market)
    x_val, y_val, _ = build_dataset(val_rows, market)
    model, scaler, train_info = train_market_model(
        x_train,
        y_train,
        x_val,
        y_val,
        seed=seed,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    probs = predict_probs(model, apply_standardizer(x_val, scaler), args.batch_size * 2, device)
    return probs, y_val, train_info


def run_oof_seed(args, seed: int, train_rows: list[dict], val_rows: list[dict], device: torch.device) -> dict:
    folds = assign_folds(train_rows, n_folds=args.n_folds, seed=seed)
    market_order = list(MARKETS)
    fold_predictions = {}
    labels_by_market = {}
    for market in market_order:
        fold_predictions[market], labels_by_market[market] = _train_oof_market_predictions(
            train_rows, market, folds, seed, args, device
        )

    oof_frame = build_oof_prediction_frame(fold_predictions, market_order, labels_by_market[market_order[0]])
    val_probs_by_market = {}
    market_results = []
    val_labels = None
    for market in market_order:
        probs, labels, train_info = _train_full_market_model(train_rows, val_rows, market, seed, args, device)
        val_probs_by_market[market] = probs
        val_labels = labels
        market_results.append(
            {
                "variant": market,
                "seed": seed,
                "metrics": evaluate_probs(probs, labels),
                "train_info": train_info,
            }
        )

    val_fusion_inputs = make_fusion_inputs(val_probs_by_market, market_order)
    _, fusion_info = train_fusion_head(
        oof_frame["features"],
        oof_frame["labels"],
        val_fusion_inputs,
        val_labels,
        seed=seed,
        device=device,
        epochs=args.fusion_epochs,
        lr=args.fusion_lr,
    )
    avg_probs = average_blend(val_probs_by_market, market_order)
    return {
        "seed": seed,
        "n_folds": args.n_folds,
        "oof_missing_count": oof_frame["missing_count"],
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "market_results": market_results,
        "average_blend": {"variant": "average_blend", "seed": seed, "metrics": evaluate_probs(avg_probs, val_labels)},
        "stacking_fusion": {"variant": "oof_stacking_fusion", "seed": seed, **fusion_info},
    }


def summarize_runs(runs: list[dict]) -> dict:
    adapted = []
    for run in runs:
        adapted.append(
            {
                "market_results": run["market_results"],
                "average_blend": run["average_blend"],
                "stacking_fusion": {
                    "variant": "stacking_fusion",
                    "seed": run["seed"],
                    "val_metrics": run["stacking_fusion"]["val_metrics"],
                },
            }
        )
    summary = summarize_holdout_runs(adapted)
    for row in summary.get("variants", []):
        if row["variant"] == "stacking_fusion":
            row["variant"] = "oof_stacking_fusion"
    if summary.get("best_variant") == "stacking_fusion":
        summary["best_variant"] = "oof_stacking_fusion"

    euro = next((row for row in summary.get("variants", []) if row["variant"] == "euro"), None)
    fusion = next((row for row in summary.get("variants", []) if row["variant"] == "oof_stacking_fusion"), None)
    verdict = []
    if fusion and euro and fusion["mean_val_logloss"] < euro["mean_val_logloss"] - 0.0001:
        verdict.append("OOF_MARKET_FUSION_IMPROVES_VAL_LOGLOSS")
    elif fusion and euro:
        verdict.append("OOF_FUSION_NO_GAIN_OVER_EURO")
    best = summary.get("best_variant")
    if best == "euro":
        verdict.append("EURO_ONLY_STILL_BEST")
    summary["verdict"] = verdict
    return summary


def build_report_payload(runs: list[dict], summary: dict, inputs: dict, verdict: list[str]) -> dict:
    payload = _base_report_payload(runs=runs, summary=summary, inputs=inputs, verdict=verdict)
    if "OOF_MARKET_FUSION_IMPROVES_VAL_LOGLOSS" in verdict:
        payload["next_actions"] = [
            "Compare OOF fusion against RawPooledMLP and P3.2 robust-Euro with the same seeds.",
            "If stable, promote OOF fusion to a reusable training script with persisted fold metadata.",
            "Keep fusion training train-only; never fit fusion weights on official val.",
        ]
    elif "OOF_FUSION_NO_GAIN_OVER_EURO" in verdict:
        payload["next_actions"] = [
            "Do not merge market fusion yet; inspect Asian/OU features and market coverage first.",
            "Use Euro-only or RawPooledMLP as the baseline until OOF fusion beats them.",
        ]
    return payload


def write_report(out_dir: str, payload: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oof_market_fusion_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = [
        "# P3.4 OOF Market Fusion Diagnostic",
        "",
        "| Variant | mean_val_logloss | std | best | mean_Brier | mean_ECE | mean_val_acc | draw_recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"].get("variants", []):
        lines.append(
            f"| {row['variant']} | {row['mean_val_logloss']:.4f} | {row['std_val_logloss']:.4f} | "
            f"{row['best_val_logloss']:.4f} | {row['mean_brier']:.4f} | {row['mean_ECE']:.4f} | "
            f"{row['mean_val_acc']:.4f} | {row['mean_draw_recall']:.4f} |"
        )
    lines.extend(["", "## Verdict"])
    lines.extend(f"- `{item}`" for item in payload.get("verdict", []))
    lines.extend(["", "## Next Actions"])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload.get("next_actions", []), start=1))
    (out / "oof_market_fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    out = Path(out_dir)
    existing = [out / "oof_market_fusion_report.json", out / "oof_market_fusion_report.md"]
    if not allow_overwrite and any(path.exists() for path in existing):
        raise FileExistsError(f"Refusing to overwrite existing P3.4 outputs in {out_dir}")
    out.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3.4 OOF market fusion diagnostic")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--out-dir", default="runs/p3_4_oof_market_fusion/formal")
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fusion-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fusion-lr", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    prepare_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
    started = time.time()
    train_ids = load_split_ids(args.train_ids)
    val_ids = load_split_ids(args.val_ids)
    train_rows = load_rows_for_ids(args.data, train_ids)
    val_rows = load_rows_for_ids(args.data, val_ids)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    runs = []
    for seed in seeds:
        print(f"Running OOF seed {seed} with {args.n_folds} folds on {device}...")
        runs.append(run_oof_seed(args, seed, train_rows, val_rows, device))

    summary = summarize_runs(runs)
    verdict = list(summary.get("verdict", []))
    payload = build_report_payload(
        runs=runs,
        summary=summary,
        inputs={
            "data": args.data,
            "train_ids": args.train_ids,
            "val_ids": args.val_ids,
            "test_ids_used": False,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "seeds": seeds,
            "n_folds": args.n_folds,
            "runtime_seconds": time.time() - started,
        },
        verdict=verdict,
    )
    write_report(args.out_dir, payload)
    print(f"Report json path: {Path(args.out_dir) / 'oof_market_fusion_report.json'}")
    print(f"Report markdown path: {Path(args.out_dir) / 'oof_market_fusion_report.md'}")
    print(f"Best variant: {summary.get('best_variant')}")
    print(f"Verdict: {', '.join(verdict)}")


if __name__ == "__main__":
    main()
