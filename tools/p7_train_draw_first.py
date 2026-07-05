"""Train the P7 draw-first factorized Patch-iTransformer probe.

P7 intentionally preserves the P6 data path while replacing the final 3-way
softmax head with a draw-first factorized 1X2 head.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_metrics import logloss_from_probs
from model.odds_patch_itransformer_v2 import FEATURE_DEFS
from model.p7_draw_first_patch_itransformer import P7DrawFirstPatchITransformer
from tools.p4_goal_diff_utils import compute_anchor_only_metrics
from tools.p4_train_residual_goal_diff import (
    SCALING_CHOICES,
    apply_p4_feature_scaler,
    build_p4_dataset,
    fit_p4_feature_scaler,
    load_rows_for_ids,
    load_split_ids,
    save_feature_scaler,
    selected_feature_names,
    set_deterministic_seed,
)
from tools.p6_train_residual_patch_itransformer import compute_draw_auxiliary_metrics


FEATURE_MARKET_TYPES = {name: market for name, market in FEATURE_DEFS}


def draw_first_probs(draw_logit: torch.Tensor, home_away_logit: torch.Tensor) -> torch.Tensor:
    """Convert draw-first binary logits to [home, draw, away] probabilities."""
    p_draw = torch.sigmoid(draw_logit)
    p_home_given_not_draw = torch.sigmoid(home_away_logit)
    p_not_draw = 1.0 - p_draw
    return torch.stack(
        [
            p_not_draw * p_home_given_not_draw,
            p_draw,
            p_not_draw * (1.0 - p_home_given_not_draw),
        ],
        dim=-1,
    )


def factorized_nll(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-8)
    return F.nll_loss(probs.log(), labels.long())


def conditional_home_away_aux_loss(home_away_logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    mask = labels != 1
    if int(mask.sum().item()) == 0:
        return home_away_logit.new_tensor(0.0)
    target = (labels[mask] == 0).float()
    return F.binary_cross_entropy_with_logits(home_away_logit[mask], target)


def compute_p7_losses(
    out: dict[str, torch.Tensor | str],
    y_1x2: torch.Tensor,
    lambda_draw_bce: float = 0.0,
    lambda_cond_ha_aux: float = 0.0,
) -> dict[str, torch.Tensor]:
    draw_logit = out["draw_logit"]
    home_away_logit = out["home_away_logit"]
    if not isinstance(draw_logit, torch.Tensor) or not isinstance(home_away_logit, torch.Tensor):
        raise TypeError("P7 logits must be tensors")
    probs = draw_first_probs(draw_logit, home_away_logit)
    l_1x2 = factorized_nll(probs, y_1x2)

    if float(lambda_draw_bce) > 0:
        draw_target = (y_1x2.long() == 1).float()
        l_draw_bce = F.binary_cross_entropy_with_logits(draw_logit, draw_target)
    else:
        l_draw_bce = l_1x2.new_tensor(0.0)

    if float(lambda_cond_ha_aux) > 0:
        l_cond_ha = conditional_home_away_aux_loss(home_away_logit, y_1x2)
    else:
        l_cond_ha = l_1x2.new_tensor(0.0)

    loss = l_1x2 + float(lambda_draw_bce) * l_draw_bce + float(lambda_cond_ha_aux) * l_cond_ha
    return {
        "loss": loss,
        "L_1x2": l_1x2,
        "L_draw_bce": l_draw_bce,
        "L_cond_ha_aux": l_cond_ha,
        "p_final": probs,
    }


def p7_selected_feature_names(feature_groups: str) -> list[str]:
    return selected_feature_names(feature_groups)


def p7_selected_feature_market_type_ids(feature_groups: str) -> list[int]:
    return [int(FEATURE_MARKET_TYPES[name]) for name in p7_selected_feature_names(feature_groups)]


def prepare_p7_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing_outputs = [
        Path(out_dir) / "report.json",
        Path(out_dir) / "best_model.pth",
        Path(out_dir) / "val_predictions.csv",
        Path(out_dir) / "val_metrics.json",
        Path(out_dir) / "training_manifest.json",
    ]
    if not allow_overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(f"Refusing to overwrite existing P7 outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def _binary_ece(confidence: torch.Tensor, target: torch.Tensor, n_bins: int = 10) -> float:
    confidence = confidence.detach().cpu().float().clamp(0.0, 1.0)
    target = target.detach().cpu().float()
    if confidence.numel() == 0:
        return 0.0
    ece = 0.0
    for i in range(int(n_bins)):
        low = i / n_bins
        high = (i + 1) / n_bins
        mask = (confidence >= low) & (confidence <= high if i == n_bins - 1 else confidence < high)
        if int(mask.sum().item()) == 0:
            continue
        ece += float(mask.float().mean().item()) * abs(
            float(target[mask].mean().item()) - float(confidence[mask].mean().item())
        )
    return float(ece)


def multiclass_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    probs = probs.detach().cpu().float()
    confidence, preds = probs.max(dim=-1)
    return _binary_ece(confidence, (preds == labels.detach().cpu().long()).float(), n_bins=n_bins)


def brier_from_probs(probs: torch.Tensor, labels: torch.Tensor, n_classes: int = 3) -> float:
    one_hot = F.one_hot(labels.detach().cpu().long(), num_classes=n_classes).float()
    return float(((probs.detach().cpu().float() - one_hot) ** 2).sum(dim=-1).mean().item())


def classwise_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probs = probs.detach().cpu().float().clamp_min(1e-8)
    labels = labels.detach().cpu().long()
    names = ["home", "draw", "away"]
    result: dict[str, Any] = {}
    for idx, name in enumerate(names):
        mask = labels == idx
        if int(mask.sum().item()) == 0:
            result[f"{name}_n"] = 0
            result[f"{name}_nll"] = 0.0
            result[f"{name}_recall"] = 0.0
            continue
        preds = probs.argmax(dim=-1)
        result[f"{name}_n"] = int(mask.sum().item())
        result[f"{name}_nll"] = float((-torch.log(probs[mask, idx])).mean().item())
        result[f"{name}_recall"] = float((preds[mask] == idx).float().mean().item())
    return result


def draw_rank_margin_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().long()
    order = torch.argsort(probs, dim=-1, descending=True)
    draw_ranks = (order == 1).nonzero(as_tuple=False)[:, 1] + 1
    top_prob = probs.max(dim=-1).values
    draw_margin_to_top = top_prob - probs[:, 1]
    true_draw = labels == 1
    pred_draw = probs.argmax(dim=-1) == 1
    out = {
        "draw_rank1_count": int((draw_ranks == 1).sum().item()),
        "draw_rank2_count": int((draw_ranks == 2).sum().item()),
        "draw_rank3_count": int((draw_ranks == 3).sum().item()),
        "mean_draw_margin_to_top": float(draw_margin_to_top.mean().item()),
        "argmax_draw_count": int(pred_draw.sum().item()),
    }
    if int(true_draw.sum().item()) > 0:
        out["mean_draw_margin_to_top_true_draw"] = float(draw_margin_to_top[true_draw].mean().item())
    else:
        out["mean_draw_margin_to_top_true_draw"] = 0.0
    return out


@torch.no_grad()
def predict_outputs(
    model: P7DrawFirstPatchITransformer,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    probs, draw_logits, ha_logits = [], [], []
    for start in range(0, len(data["X"]), batch_size):
        xb = data["X"][start : start + batch_size].to(device)
        out = model(xb)
        draw_logit = out["draw_logit"]
        ha_logit = out["home_away_logit"]
        if not isinstance(draw_logit, torch.Tensor) or not isinstance(ha_logit, torch.Tensor):
            raise TypeError("P7 logits must be tensors")
        probs.append(draw_first_probs(draw_logit, ha_logit).cpu())
        draw_logits.append(draw_logit.cpu())
        ha_logits.append(ha_logit.cpu())
    return {
        "p_final": torch.cat(probs, dim=0),
        "draw_logit": torch.cat(draw_logits, dim=0),
        "home_away_logit": torch.cat(ha_logits, dim=0),
    }


@torch.no_grad()
def evaluate(
    model: P7DrawFirstPatchITransformer,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    preds = predict_outputs(model, data, batch_size, device)
    probs = preds["p_final"].clamp_min(1e-8)
    labels = data["y_1x2"]
    val_metrics = compute_draw_auxiliary_metrics(probs, labels)
    val_metrics.update(
        {
            "accuracy": float((probs.argmax(dim=-1) == labels).float().mean().item()),
            "logloss": logloss_from_probs(probs, labels),
            "brier": brier_from_probs(probs, labels, 3),
            "ece": multiclass_ece(probs, labels, 10),
        }
    )
    val_metrics.update(draw_rank_margin_metrics(probs, labels))
    return val_metrics, preds


def train_epoch(
    model: P7DrawFirstPatchITransformer,
    data: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    lambda_draw_bce: float,
    lambda_cond_ha_aux: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "L_1x2": 0.0, "L_draw_bce": 0.0, "L_cond_ha_aux": 0.0}
    n_batches = 0
    n_correct = 0
    n_total = 0
    perm = torch.randperm(len(data["X"]))
    for start in range(0, len(perm), batch_size):
        idx = perm[start : start + batch_size]
        xb = data["X"][idx].to(device)
        yb = data["y_1x2"][idx].to(device)
        out = model(xb)
        losses = compute_p7_losses(out, yb, lambda_draw_bce=lambda_draw_bce, lambda_cond_ha_aux=lambda_cond_ha_aux)
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu().item())
        n_correct += int((losses["p_final"].argmax(dim=-1) == yb).sum().item())
        n_total += int(len(idx))
        n_batches += 1
    metrics = {key: value / max(n_batches, 1) for key, value in totals.items()}
    metrics["accuracy"] = n_correct / max(n_total, 1)
    return metrics


def write_val_predictions_csv(
    path: str,
    match_ids: list[str],
    labels: torch.Tensor,
    probs: torch.Tensor,
    draw_logit: torch.Tensor,
    home_away_logit: torch.Tensor,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "match_id",
        "y_true",
        "p_home",
        "p_draw",
        "p_away",
        "pred_class",
        "correct",
        "draw_logit",
        "home_away_logit",
        "head_type",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, prob in enumerate(probs):
            pred = int(prob.argmax().item())
            y_true = int(labels[idx].item())
            writer.writerow(
                {
                    "match_id": match_ids[idx] if idx < len(match_ids) else str(idx),
                    "y_true": y_true,
                    "p_home": float(prob[0].item()),
                    "p_draw": float(prob[1].item()),
                    "p_away": float(prob[2].item()),
                    "pred_class": pred,
                    "correct": int(pred == y_true),
                    "draw_logit": float(draw_logit[idx].item()),
                    "home_away_logit": float(home_away_logit[idx].item()),
                    "head_type": "draw_first_factorized",
                }
            )


def maybe_write_val_predictions_parquet(csv_path: Path, parquet_path: Path) -> bool:
    """Write parquet only when optional pyarrow is installed."""
    if importlib.util.find_spec("pyarrow") is None:
        return False
    import pyarrow.csv as pa_csv  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    table = pa_csv.read_csv(csv_path)
    pq.write_table(table, parquet_path)
    return True


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train P7 draw-first factorized Patch-iTransformer")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--feature-groups", default="euro")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lambda-draw-bce", type=float, default=0.0)
    parser.add_argument("--lambda-cond-ha-aux", type=float, default=0.0)
    parser.add_argument("--scaling", default="robust", choices=SCALING_CHOICES)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--variant", default="manual")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if "test" in Path(args.train_ids).name.lower() or "test" in Path(args.val_ids).name.lower():
        raise SystemExit("P7 refuses any test split path")

    feature_names = p7_selected_feature_names(args.feature_groups)
    market_type_ids = p7_selected_feature_market_type_ids(args.feature_groups)
    set_deterministic_seed(args.seed)
    prepare_p7_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids = load_split_ids(args.train_ids)
    val_ids = load_split_ids(args.val_ids)
    train_rows = load_rows_for_ids(args.data, train_ids)
    val_rows = load_rows_for_ids(args.data, val_ids)
    if args.max_samples > 0:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: min(args.max_samples, len(val_rows))]
    print(f"Rows: train={len(train_rows)} val={len(val_rows)}")

    train_data = build_p4_dataset(train_rows, args.feature_groups)
    val_data = build_p4_dataset(val_rows, args.feature_groups)
    if train_data["feature_names"] != feature_names or val_data["feature_names"] != feature_names:
        raise ValueError("P7 selected feature names drifted from dataset construction")
    scaler = fit_p4_feature_scaler(train_data["X"], feature_names, args.scaling)
    scaler_path = save_feature_scaler(scaler, args.out_dir)
    train_data["X"] = apply_p4_feature_scaler(train_data["X"], scaler)
    val_data["X"] = apply_p4_feature_scaler(val_data["X"], scaler)

    anchor_only = compute_anchor_only_metrics(val_data["p_euro_anchor"], val_data["y_1x2"])
    model = P7DrawFirstPatchITransformer(
        n_features=len(feature_names),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        market_type_ids=market_type_ids,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}, features={len(feature_names)}")

    best_val_logloss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_data,
            optimizer,
            args.batch_size,
            device,
            lambda_draw_bce=args.lambda_draw_bce,
            lambda_cond_ha_aux=args.lambda_cond_ha_aux,
        )
        val_metrics, _ = evaluate(model, val_data, args.batch_size * 2, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_L_1x2": train_metrics["L_1x2"],
            "train_L_draw_bce": train_metrics["L_draw_bce"],
            "train_L_cond_ha_aux": train_metrics["L_cond_ha_aux"],
            "train_acc": train_metrics["accuracy"],
            "val_logloss": val_metrics["logloss"],
            "val_brier": val_metrics["brier"],
            "val_ece": val_metrics["ece"],
            "val_accuracy": val_metrics["accuracy"],
            "val_draw_class_nll": val_metrics["draw_class_nll"],
            "val_draw_precision": val_metrics["draw_precision"],
            "val_draw_recall": val_metrics["draw_recall"],
            "val_draw_top2_recall": val_metrics["draw_top2_recall"],
            "val_mean_p_draw_on_true_draw": val_metrics["mean_p_draw_on_true_draw"],
            "val_mean_draw_margin_to_top": val_metrics["mean_draw_margin_to_top"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        is_best = val_metrics["logloss"] < best_val_logloss
        if is_best:
            best_val_logloss = val_metrics["logloss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "feature_names": feature_names,
                    "market_type_ids": market_type_ids,
                    "scaler": scaler,
                    "best_epoch": best_epoch,
                    "best_val_logloss": best_val_logloss,
                    "head_type": "draw_first_factorized",
                },
                Path(args.out_dir) / "best_model.pth",
            )
        print(
            f"Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
            f"val_logloss={val_metrics['logloss']:.4f} "
            f"draw_recall={val_metrics['draw_recall']:.4f} "
            f"draw_precision={val_metrics['draw_precision']:.4f}"
            + (" *" if is_best else "")
        )

    ckpt_path = Path(args.out_dir) / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_val, preds = evaluate(model, val_data, args.batch_size * 2, device)

    out_dir = Path(args.out_dir)
    pred_csv = out_dir / "val_predictions.csv"
    write_val_predictions_csv(
        str(pred_csv),
        val_data["match_ids"],
        val_data["y_1x2"],
        preds["p_final"],
        preds["draw_logit"],
        preds["home_away_logit"],
    )
    parquet_written = maybe_write_val_predictions_parquet(pred_csv, out_dir / "val_predictions.parquet")
    classwise = classwise_metrics(preds["p_final"], val_data["y_1x2"])
    draw_rank = draw_rank_margin_metrics(preds["p_final"], val_data["y_1x2"])
    calibration = {"ece": final_val["ece"], "draw_ece": final_val["classwise_ece_draw"]}
    manifest = {
        "phase": "P7 Draw-First Factorized 1X2 Head",
        "model": "P7DrawFirstPatchITransformer",
        "head_type": "draw_first_factorized",
        "variant": args.variant,
        "seed": args.seed,
        "feature_groups": args.feature_groups,
        "test_ids_used": False,
        "validation_fit_used": False,
        "posthoc_val_fit_used": False,
        "parquet_written": parquet_written,
        "config": vars(args),
    }

    data_summary = {
        "train_samples": int(len(train_data["X"])),
        "val_samples": int(len(val_data["X"])),
        "test_ids_used": False,
        "validation_fit_used": False,
        "posthoc_val_fit_used": False,
        "checkpoint_selection": "best_val_logloss",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "train_skipped_rows": int(train_data["skipped_rows"]),
        "val_skipped_rows": int(val_data["skipped_rows"]),
        "anchor_fallback_count": int(val_data["anchor_fallback_count"]),
        "negative_time_valid_euro_count": int(val_data["negative_time_valid_euro_count"]),
    }
    report = {
        "phase": "P7 Draw-First Factorized 1X2 Head",
        "run_mode": "smoke" if args.max_samples else "formal",
        "model": "P7DrawFirstPatchITransformer",
        "head_type": "draw_first_factorized",
        "config": vars(args),
        "data": data_summary,
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "anchor_only_baseline": anchor_only,
        "train_metrics": history[-1] if history else {},
        "val_metrics": final_val,
        "classwise_metrics": classwise,
        "draw_rank_metrics": draw_rank,
        "draw_margin_metrics": {
            "mean_draw_margin_to_top": draw_rank["mean_draw_margin_to_top"],
            "mean_draw_margin_to_top_true_draw": draw_rank["mean_draw_margin_to_top_true_draw"],
        },
        "calibration_metrics": calibration,
        "history": history,
        "warnings": [],
        "params": n_params,
        "scaler": {"path": scaler_path, **scaler},
        "run_path": args.out_dir,
        "artifacts": {
            "best_model": str(ckpt_path),
            "val_predictions_csv": str(pred_csv),
            "val_predictions_parquet": str(out_dir / "val_predictions.parquet") if parquet_written else None,
            "val_metrics": str(out_dir / "val_metrics.json"),
            "training_manifest": str(out_dir / "training_manifest.json"),
        },
    }
    write_json(out_dir / "val_metrics.json", final_val)
    write_json(out_dir / "classwise_metrics.json", classwise)
    write_json(out_dir / "draw_rank_metrics.json", draw_rank)
    write_json(out_dir / "draw_margin_metrics.json", report["draw_margin_metrics"])
    write_json(out_dir / "calibration_metrics.json", calibration)
    write_json(out_dir / "training_manifest.json", manifest)
    write_json(out_dir / "report.json", report)
    print(f"Report saved to {out_dir / 'report.json'}")
    print(f"Best val_logloss: {best_val_logloss:.6f}")


if __name__ == "__main__":
    main()
