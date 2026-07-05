"""Train P18.1 AH-cover auxiliary objective on the frozen P14/P16 backbone.

P18.1 keeps the P6ResidualPatchITransformer backbone and P14 market-replication
objective. It adds one optional 5-class Asian-handicap cover auxiliary head:
upper_full_win / upper_half_win / push / upper_half_loss / upper_full_loss.
Rows without a valid pre-kickoff AH line keep their 1X2/market/diff labels but
use ignore_index for the AH auxiliary loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.asian_handicap import settle_asian_5class
from eval.odds_metrics import logloss_from_probs
from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
from tools.p14_train_market_replication import (
    EPS,
    _brier_from_probs,
    _multiclass_ece,
    apply_feature_scaler,
    assert_no_test_paths,
    build_epoch_history_row,
    build_p14_dataset,
    compute_anchor_only_metrics,
    compute_draw_auxiliary_metrics,
    compute_market_replication_metrics,
    compute_p14_losses,
    fit_feature_scaler,
    load_best_model_state,
    load_rows_for_ids,
    load_split_ids,
    p14_selected_feature_market_type_ids,
    p14_selected_feature_names,
    save_feature_scaler,
    select_checkpoint_row,
    set_deterministic_seed,
    should_replace_checkpoint,
    write_market_replication_csv,
    write_val_predictions_csv,
)
from tools.p18_ah_cover_diagnostics import latest_pre_kickoff_asian_snapshot


AH_COVER_LABELS = ["upper_full_win", "upper_half_win", "push", "upper_half_loss", "upper_full_loss"]
AH_LABEL_TO_IDX = {name: idx for idx, name in enumerate(AH_COVER_LABELS)}
AH_UNIT_VALUES = [1.0, 0.5, 0.0, -0.5, -1.0]
AH_SIDE_LABELS = ["upper", "push", "lower"]
AH_SIDE_TARGETS = [0, 0, 1, 2, 2]
IGNORE_INDEX = -100


def ah_cover_label_from_row(row: dict[str, Any]) -> int:
    label = row.get("label", {}) or {}
    snapshot = latest_pre_kickoff_asian_snapshot(row)
    if not snapshot.get("valid_asian"):
        return IGNORE_INDEX
    try:
        home_goals = int(label["home_goals"])
        away_goals = int(label["away_goals"])
    except (KeyError, TypeError, ValueError):
        return IGNORE_INDEX
    result = settle_asian_5class(home_goals, away_goals, float(snapshot["asian_line"]))
    return AH_LABEL_TO_IDX[result]


def _is_valid_p14_label(row: dict[str, Any]) -> bool:
    label = row.get("label", {}) or {}
    result = label.get("euro_result")
    if result not in {"home", "draw", "away"}:
        return False
    if "home_goals" not in label or "away_goals" not in label:
        return False
    try:
        int(label["home_goals"])
        int(label["away_goals"])
    except (TypeError, ValueError):
        return False
    return True


def build_p18_dataset(rows: list[dict[str, Any]], feature_groups: str) -> dict[str, Any]:
    data = build_p14_dataset(rows, feature_groups)
    ah_labels = [ah_cover_label_from_row(row) for row in rows if _is_valid_p14_label(row)]
    if len(ah_labels) != len(data["X"]):
        raise ValueError(f"P18 AH label alignment failed: ah={len(ah_labels)} p14={len(data['X'])}")
    data["y_ah_cover"] = torch.tensor(ah_labels, dtype=torch.long)
    data["ah_cover_label_counts"] = _label_counts(ah_labels)
    data["ah_cover_valid_rows"] = int(sum(1 for value in ah_labels if value != IGNORE_INDEX))
    return data


def _label_counts(labels: list[int]) -> dict[str, int]:
    counts = {name: 0 for name in AH_COVER_LABELS}
    counts["ignore"] = 0
    for label in labels:
        if label == IGNORE_INDEX:
            counts["ignore"] += 1
        elif 0 <= int(label) < len(AH_COVER_LABELS):
            counts[AH_COVER_LABELS[int(label)]] += 1
    return counts


def ah_unit_weights_for(logits: torch.Tensor) -> torch.Tensor:
    return logits.new_tensor(AH_UNIT_VALUES)


def ah_side_targets_for(labels: torch.Tensor) -> torch.Tensor:
    return labels.new_tensor(AH_SIDE_TARGETS)[labels]


def ah_side_logits_from_cover_logits(logits: torch.Tensor) -> torch.Tensor:
    upper = torch.logsumexp(logits[:, [0, 1]], dim=-1)
    push = logits[:, 2]
    lower = torch.logsumexp(logits[:, [3, 4]], dim=-1)
    return torch.stack([upper, push, lower], dim=-1)


def compute_p18_losses(
    out: dict[str, torch.Tensor],
    p_market: torch.Tensor,
    y_1x2: torch.Tensor,
    y_goal_diff: torch.Tensor,
    y_ah_cover: torch.Tensor,
    ah_cover_loss_weight: float = 0.0,
    ah_unit_loss_weight: float = 0.0,
    ah_side_loss_weight: float = 0.0,
    ah_direct_unit_loss_weight: float = 0.0,
    **p14_loss_config: float,
) -> dict[str, torch.Tensor]:
    losses = compute_p14_losses(out, p_market, y_1x2, y_goal_diff, **p14_loss_config)
    l_ah_cover = losses["loss"].new_tensor(0.0)
    l_ah_unit = losses["loss"].new_tensor(0.0)
    l_ah_side = losses["loss"].new_tensor(0.0)
    l_ah_direct_unit = losses["loss"].new_tensor(0.0)
    valid = y_ah_cover != IGNORE_INDEX
    if float(ah_cover_loss_weight) > 0 or float(ah_unit_loss_weight) > 0 or float(ah_side_loss_weight) > 0:
        if "ah_cover_logits" not in out:
            raise ValueError("ah_cover_logits are required when any AH auxiliary loss weight > 0")
        if bool(valid.any().item()):
            logits_v = out["ah_cover_logits"][valid]
            labels_v = y_ah_cover[valid]
            if float(ah_cover_loss_weight) > 0:
                l_ah_cover = F.cross_entropy(logits_v, labels_v)
            if float(ah_unit_loss_weight) > 0:
                unit_weights = ah_unit_weights_for(logits_v)
                pred_units = (F.softmax(logits_v, dim=-1) * unit_weights).sum(dim=-1)
                target_units = unit_weights[labels_v]
                l_ah_unit = F.smooth_l1_loss(pred_units, target_units)
            if float(ah_side_loss_weight) > 0:
                l_ah_side = F.cross_entropy(ah_side_logits_from_cover_logits(logits_v), ah_side_targets_for(labels_v))
    if float(ah_direct_unit_loss_weight) > 0:
        if "ah_unit_pred" not in out:
            raise ValueError("ah_unit_pred is required when ah_direct_unit_loss_weight > 0")
        if bool(valid.any().item()):
            unit_weights = out["ah_unit_pred"].new_tensor(AH_UNIT_VALUES)
            target_units = unit_weights[y_ah_cover[valid]]
            l_ah_direct_unit = F.smooth_l1_loss(out["ah_unit_pred"][valid].float(), target_units)
    losses["L_ah_cover"] = l_ah_cover
    losses["L_ah_unit"] = l_ah_unit
    losses["L_ah_side"] = l_ah_side
    losses["L_ah_direct_unit"] = l_ah_direct_unit
    losses["loss"] = (
        losses["loss"]
        + float(ah_cover_loss_weight) * l_ah_cover
        + float(ah_unit_loss_weight) * l_ah_unit
        + float(ah_side_loss_weight) * l_ah_side
        + float(ah_direct_unit_loss_weight) * l_ah_direct_unit
    )
    return losses


def compute_ah_cover_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()
    valid = labels != IGNORE_INDEX
    if not bool(valid.any().item()):
        return {
            "ah_cover_n": 0,
            "ah_cover_acc": 0.0,
            "ah_cover_nll": 0.0,
            "ah_unit_mae": 0.0,
            "ah_side_acc": 0.0,
            "ah_expected_upper_units_mean": 0.0,
            "ah_cover_counts": {name: 0 for name in AH_COVER_LABELS},
            "ah_cover_pred_counts": {name: 0 for name in AH_COVER_LABELS},
            "ah_side_pred_counts": {name: 0 for name in AH_SIDE_LABELS},
        }
    logits_v = logits[valid]
    labels_v = labels[valid]
    probs = F.softmax(logits_v, dim=-1).clamp_min(EPS)
    preds = probs.argmax(dim=-1)
    unit_weights = ah_unit_weights_for(logits_v)
    pred_units = (probs * unit_weights).sum(dim=-1)
    target_units = unit_weights[labels_v]
    side_logits = ah_side_logits_from_cover_logits(logits_v)
    side_preds = side_logits.argmax(dim=-1)
    side_targets = ah_side_targets_for(labels_v)
    counts = {name: int((labels_v == idx).sum().item()) for idx, name in enumerate(AH_COVER_LABELS)}
    pred_counts = {name: int((preds == idx).sum().item()) for idx, name in enumerate(AH_COVER_LABELS)}
    side_pred_counts = {name: int((side_preds == idx).sum().item()) for idx, name in enumerate(AH_SIDE_LABELS)}
    return {
        "ah_cover_n": int(labels_v.numel()),
        "ah_cover_acc": float((preds == labels_v).float().mean().item()),
        "ah_cover_nll": logloss_from_probs(probs, labels_v),
        "ah_unit_mae": float((pred_units - target_units).abs().mean().item()),
        "ah_side_acc": float((side_preds == side_targets).float().mean().item()),
        "ah_expected_upper_units_mean": float(pred_units.mean().item()),
        "ah_cover_counts": counts,
        "ah_cover_pred_counts": pred_counts,
        "ah_side_pred_counts": side_pred_counts,
    }


def compute_ah_direct_unit_metrics(pred_units: torch.Tensor | None, labels: torch.Tensor) -> dict[str, Any]:
    if pred_units is None:
        return {
            "ah_direct_unit_n": 0,
            "ah_direct_unit_mae": 0.0,
            "ah_direct_side_acc": 0.0,
            "ah_direct_expected_upper_units_mean": 0.0,
        }
    pred_units = pred_units.detach().cpu().float().clamp(-1.0, 1.0)
    labels = labels.detach().cpu().long()
    valid = labels != IGNORE_INDEX
    if not bool(valid.any().item()):
        return {
            "ah_direct_unit_n": 0,
            "ah_direct_unit_mae": 0.0,
            "ah_direct_side_acc": 0.0,
            "ah_direct_expected_upper_units_mean": 0.0,
        }
    pred_v = pred_units[valid]
    unit_weights = pred_v.new_tensor(AH_UNIT_VALUES)
    target = unit_weights[labels[valid]]
    pred_side = torch.where(pred_v > 1e-9, 1, torch.where(pred_v < -1e-9, -1, 0))
    target_side = torch.where(target > 1e-9, 1, torch.where(target < -1e-9, -1, 0))
    non_push = target_side != 0
    return {
        "ah_direct_unit_n": int(pred_v.numel()),
        "ah_direct_unit_mae": float((pred_v - target).abs().mean().item()),
        "ah_direct_side_acc": float((pred_side[non_push] == target_side[non_push]).float().mean().item()) if bool(non_push.any().item()) else 0.0,
        "ah_direct_expected_upper_units_mean": float(pred_v.mean().item()),
    }


def train_epoch(
    model: P6ResidualPatchITransformer,
    data: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    config: dict[str, float],
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "L_1x2": 0.0,
        "L_market_kl": 0.0,
        "L_diff": 0.0,
        "L_consistency": 0.0,
        "L_delta": 0.0,
        "L_ah_cover": 0.0,
        "L_ah_unit": 0.0,
        "L_ah_side": 0.0,
        "L_ah_direct_unit": 0.0,
    }
    n_batches = 0
    n_correct = 0
    n_total = 0
    perm = torch.randperm(len(data["X"]))
    for start in range(0, len(perm), batch_size):
        idx = perm[start : start + batch_size]
        xb = data["X"][idx].to(device)
        yb = data["y_1x2"][idx].to(device)
        ygb = data["y_goal_diff"][idx].to(device)
        yab = data["y_ah_cover"][idx].to(device)
        market = data["p_market"][idx].to(device)
        losses = compute_p18_losses(model(xb), market, yb, ygb, yab, **config)
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


@torch.no_grad()
def predict_outputs(model: P6ResidualPatchITransformer, data: dict[str, Any], batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    p_final, p_from_diff, q_diff, delta_logits, ah_logits, ah_unit_pred = [], [], [], [], [], []
    for start in range(0, len(data["X"]), batch_size):
        xb = data["X"][start : start + batch_size].to(device)
        market = data["p_market"][start : start + batch_size].to(device)
        out = model(xb)
        losses = compute_p18_losses(
            out,
            market,
            data["y_1x2"][start : start + batch_size].to(device),
            data["y_goal_diff"][start : start + batch_size].to(device),
            data["y_ah_cover"][start : start + batch_size].to(device),
            ah_cover_loss_weight=0.0,
            market_loss_weight=0.0,
            label_loss_weight=1.0,
            diff_loss_weight=0.0,
            consistency_loss_weight=0.0,
            delta_l2_weight=0.0,
        )
        p_final.append(losses["p_final"].cpu())
        p_from_diff.append(losses["p_from_diff"].cpu())
        q_diff.append(losses["q_diff"].cpu())
        delta_logits.append(out["delta_logits"].cpu())
        ah_logits.append(out["ah_cover_logits"].cpu())
        if "ah_unit_pred" in out:
            ah_unit_pred.append(out["ah_unit_pred"].cpu())
    result = {
        "p_final": torch.cat(p_final, dim=0),
        "p_from_diff": torch.cat(p_from_diff, dim=0),
        "q_diff": torch.cat(q_diff, dim=0),
        "delta_logits": torch.cat(delta_logits, dim=0),
        "ah_cover_logits": torch.cat(ah_logits, dim=0),
    }
    if ah_unit_pred:
        result["ah_unit_pred"] = torch.cat(ah_unit_pred, dim=0)
    return result


@torch.no_grad()
def evaluate(model: P6ResidualPatchITransformer, data: dict[str, Any], batch_size: int, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    preds = predict_outputs(model, data, batch_size, device)
    labels = data["y_1x2"]
    goal_labels = data["y_goal_diff"]
    p_final = _normalize_probs(preds["p_final"].clamp_min(EPS))
    q = _normalize_probs(preds["q_diff"].clamp_min(EPS))
    p_from_diff = _normalize_probs(preds["p_from_diff"].clamp_min(EPS))
    val_metrics = compute_draw_auxiliary_metrics(p_final, labels)
    val_metrics.update(
        {
            "accuracy": float((p_final.argmax(dim=-1) == labels).float().mean().item()),
            "logloss": logloss_from_probs(p_final, labels),
            "brier": _brier_from_probs(p_final, labels, 3),
            "ece": _multiclass_ece(p_final, labels, 10),
        }
    )
    true_zero = goal_labels == 3
    pred_zero = q.argmax(dim=-1) == 3
    goal_diff_metrics = {
        "diff_nll": logloss_from_probs(q, goal_labels),
        "diff_acc": float((q.argmax(dim=-1) == goal_labels).float().mean().item()),
        "zero_recall": float((true_zero & pred_zero).sum().item() / true_zero.sum().item()) if int(true_zero.sum().item()) else 0.0,
        "p_from_diff_logloss": logloss_from_probs(p_from_diff, labels),
        "final_vs_diff_kl": float((p_from_diff * (p_from_diff.log() - p_final.log())).sum(dim=-1).mean().item()),
    }
    market_metrics = compute_market_replication_metrics(p_final, data["p_market"], labels)
    ah_metrics = compute_ah_cover_metrics(preds["ah_cover_logits"], data["y_ah_cover"])
    ah_metrics.update(compute_ah_direct_unit_metrics(preds.get("ah_unit_pred"), data["y_ah_cover"]))
    return val_metrics, market_metrics, goal_diff_metrics, ah_metrics, preds


def _normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.float().clamp_min(EPS)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)


def prepare_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    outputs = [
        Path(out_dir) / "report.json",
        Path(out_dir) / "report.md",
        Path(out_dir) / "best_model.pth",
        Path(out_dir) / "val_predictions.csv",
        Path(out_dir) / "val_market_replication.csv",
        Path(out_dir) / "val_ah_cover_predictions.csv",
        Path(out_dir) / "scaler.json",
    ]
    if not allow_overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Refusing to overwrite existing P18 outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def write_ah_cover_predictions_csv(path: Path, data: dict[str, Any], preds: dict[str, torch.Tensor]) -> None:
    logits = preds["ah_cover_logits"].detach().cpu().float()
    probs = F.softmax(logits, dim=-1)
    labels = data["y_ah_cover"].detach().cpu().long()
    pred = probs.argmax(dim=-1)
    direct_units = preds.get("ah_unit_pred")
    if direct_units is not None:
        direct_units = direct_units.detach().cpu().float().clamp(-1.0, 1.0)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["match_id", "y_ah_cover", "pred_ah_cover", *[f"p_{name}" for name in AH_COVER_LABELS]]
        if direct_units is not None:
            fieldnames.append("direct_expected_upper_units")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, match_id in enumerate(data["match_ids"]):
            row = {
                "match_id": match_id,
                "y_ah_cover": int(labels[idx].item()),
                "pred_ah_cover": int(pred[idx].item()),
            }
            for cls_idx, name in enumerate(AH_COVER_LABELS):
                row[f"p_{name}"] = float(probs[idx, cls_idx].item())
            if direct_units is not None:
                row["direct_expected_upper_units"] = float(direct_units[idx].item())
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train P18.1 AH-cover auxiliary probe")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--feature-groups", default="euro,asian,ou")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--label-loss-weight", type=float, default=1.0)
    parser.add_argument("--market-loss-weight", type=float, default=0.30)
    parser.add_argument("--diff-loss-weight", type=float, default=0.30)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--delta-l2-weight", type=float, default=0.01)
    parser.add_argument("--ah-cover-loss-weight", type=float, default=0.10)
    parser.add_argument("--ah-unit-loss-weight", type=float, default=0.0)
    parser.add_argument("--ah-side-loss-weight", type=float, default=0.0)
    parser.add_argument("--ah-direct-unit-loss-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-selection", default="balanced", choices=("logloss", "balanced"))
    parser.add_argument("--balanced-logloss-ceiling", type=float, default=0.9375)
    parser.add_argument("--scaling", default="robust")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    assert_no_test_paths([args.data, args.train_ids, args.val_ids, args.out_dir])
    set_deterministic_seed(args.seed)
    prepare_output_dir(args.out_dir, args.allow_overwrite)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    feature_names = p14_selected_feature_names(args.feature_groups)
    market_type_ids = p14_selected_feature_market_type_ids(args.feature_groups)
    train_ids = load_split_ids(args.train_ids)
    val_ids = load_split_ids(args.val_ids)
    train_rows = load_rows_for_ids(args.data, train_ids)
    val_rows = load_rows_for_ids(args.data, val_ids)
    if args.max_samples > 0:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: min(args.max_samples, len(val_rows))]
    train_data = build_p18_dataset(train_rows, args.feature_groups)
    val_data = build_p18_dataset(val_rows, args.feature_groups)
    scaler = fit_feature_scaler(train_data["X"], feature_names, args.scaling)
    scaler_path = save_feature_scaler(scaler, args.out_dir)
    train_data["X"] = apply_feature_scaler(train_data["X"], scaler)
    val_data["X"] = apply_feature_scaler(val_data["X"], scaler)
    anchor_only = compute_anchor_only_metrics(val_data["p_market"], val_data["y_1x2"])
    model = P6ResidualPatchITransformer(
        n_features=len(feature_names),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        market_type_ids=market_type_ids,
        enable_ah_cover_head=True,
        enable_ah_unit_head=float(args.ah_direct_unit_loss_weight) > 0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_config = {
        "market_loss_weight": float(args.market_loss_weight),
        "label_loss_weight": float(args.label_loss_weight),
        "diff_loss_weight": float(args.diff_loss_weight),
        "consistency_loss_weight": float(args.consistency_loss_weight),
        "delta_l2_weight": float(args.delta_l2_weight),
        "ah_cover_loss_weight": float(args.ah_cover_loss_weight),
        "ah_unit_loss_weight": float(args.ah_unit_loss_weight),
        "ah_side_loss_weight": float(args.ah_side_loss_weight),
        "ah_direct_unit_loss_weight": float(args.ah_direct_unit_loss_weight),
    }
    selected_checkpoint_row: dict[str, Any] | None = None
    best_logloss = float("inf")
    best_logloss_epoch = 0
    history: list[dict[str, Any]] = []
    train_metrics: dict[str, Any] = {}
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_data, optimizer, args.batch_size, device, loss_config)
        val_metrics, market_metrics, goal_diff_metrics, ah_metrics, _ = evaluate(model, val_data, args.batch_size, device)
        scheduler.step()
        row = build_epoch_history_row(epoch, train_metrics, val_metrics, market_metrics, goal_diff_metrics)
        row.update(
            {
                "train_L_ah_cover": float(train_metrics.get("L_ah_cover", 0.0)),
                "train_L_ah_unit": float(train_metrics.get("L_ah_unit", 0.0)),
                "train_L_ah_side": float(train_metrics.get("L_ah_side", 0.0)),
                "train_L_ah_direct_unit": float(train_metrics.get("L_ah_direct_unit", 0.0)),
                "val_ah_cover_acc": float(ah_metrics["ah_cover_acc"]),
                "val_ah_cover_nll": float(ah_metrics["ah_cover_nll"]),
                "val_ah_unit_mae": float(ah_metrics["ah_unit_mae"]),
                "val_ah_side_acc": float(ah_metrics["ah_side_acc"]),
                "val_ah_direct_unit_mae": float(ah_metrics["ah_direct_unit_mae"]),
                "val_ah_direct_side_acc": float(ah_metrics["ah_direct_side_acc"]),
                "val_ah_cover_n": int(ah_metrics["ah_cover_n"]),
            }
        )
        history.append(row)
        if val_metrics["logloss"] < best_logloss:
            best_logloss = val_metrics["logloss"]
            best_logloss_epoch = epoch
        if should_replace_checkpoint(row, selected_checkpoint_row, args.checkpoint_selection, args.balanced_logloss_ceiling):
            selected_checkpoint_row = dict(row)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "feature_names": feature_names,
                    "market_type_ids": market_type_ids,
                    "scaler_path": scaler_path,
                    "checkpoint_selection": args.checkpoint_selection,
                    "balanced_logloss_ceiling": float(args.balanced_logloss_ceiling),
                    "selected_checkpoint_metrics": selected_checkpoint_row,
                    "best_epoch": int(selected_checkpoint_row["epoch"]),
                    "best_val_logloss": float(selected_checkpoint_row["val_logloss"]),
                    "best_logloss_epoch": int(best_logloss_epoch),
                    "best_logloss": float(best_logloss),
                },
                Path(args.out_dir) / "best_model.pth",
            )
        print(
            f"Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
            f"val_logloss={val_metrics['logloss']:.4f} "
            f"ah_acc={ah_metrics['ah_cover_acc']:.4f} "
            f"ah_side={ah_metrics['ah_side_acc']:.4f} "
            f"ah_direct_mae={ah_metrics['ah_direct_unit_mae']:.4f} "
            f"market_kl={market_metrics['market_kl']:.5f}",
            flush=True,
        )
    best_checkpoint_path = Path(args.out_dir) / "best_model.pth"
    best_checkpoint = load_best_model_state(model, best_checkpoint_path, device)
    selected_checkpoint_metrics = best_checkpoint.get("selected_checkpoint_metrics") or select_checkpoint_row(
        history, args.checkpoint_selection, args.balanced_logloss_ceiling
    )
    final_val, final_market, final_goal_diff, final_ah, preds = evaluate(model, val_data, args.batch_size, device)
    write_val_predictions_csv(Path(args.out_dir) / "val_predictions.csv", val_data["match_ids"], val_data["y_1x2"], preds["p_final"])
    write_market_replication_csv(Path(args.out_dir) / "val_market_replication.csv", val_data, preds)
    write_ah_cover_predictions_csv(Path(args.out_dir) / "val_ah_cover_predictions.csv", val_data, preds)
    report = {
        "phase": "P18.1 AH-cover auxiliary probe",
        "run_mode": "smoke" if args.max_samples else "formal",
        "model": "P6ResidualPatchITransformer",
        "config": vars(args),
        "data": {
            "train_rows": len(train_data["X"]),
            "val_rows": len(val_data["X"]),
            "train_ah_cover_valid_rows": train_data["ah_cover_valid_rows"],
            "val_ah_cover_valid_rows": val_data["ah_cover_valid_rows"],
            "train_ah_cover_label_counts": train_data["ah_cover_label_counts"],
            "val_ah_cover_label_counts": val_data["ah_cover_label_counts"],
            "feature_names": feature_names,
            "scaler_path": scaler_path,
        },
        "checkpoint_selection": args.checkpoint_selection,
        "balanced_logloss_ceiling": float(args.balanced_logloss_ceiling),
        "best_epoch": int(selected_checkpoint_metrics["epoch"]),
        "best_val_logloss": float(selected_checkpoint_metrics["val_logloss"]),
        "selected_checkpoint_metrics": selected_checkpoint_metrics,
        "best_logloss_epoch": best_logloss_epoch,
        "best_logloss": best_logloss,
        "anchor_only_baseline": anchor_only,
        "train_metrics": train_metrics,
        "val_metrics": final_val,
        "market_replication_metrics": final_market,
        "goal_diff_metrics": final_goal_diff,
        "ah_cover_metrics": final_ah,
        "history": history,
        "artifacts": {
            "best_model": str(best_checkpoint_path),
            "val_predictions": str(Path(args.out_dir) / "val_predictions.csv"),
            "val_market_replication": str(Path(args.out_dir) / "val_market_replication.csv"),
            "val_ah_cover_predictions": str(Path(args.out_dir) / "val_ah_cover_predictions.csv"),
            "scaler": scaler_path,
        },
    }
    report_path = Path(args.out_dir) / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (Path(args.out_dir) / "report.md").write_text(
        "\n".join(
            [
                "# P18.1 AH-cover auxiliary probe",
                "",
                f"- val_logloss: `{final_val['logloss']}`",
                f"- market_kl: `{final_market['market_kl']}`",
                f"- market_draw_mae: `{final_market['market_draw_mae']}`",
                f"- ah_cover_acc: `{final_ah['ah_cover_acc']}`",
                f"- ah_cover_nll: `{final_ah['ah_cover_nll']}`",
                f"- ah_unit_mae: `{final_ah['ah_unit_mae']}`",
                f"- ah_side_acc: `{final_ah['ah_side_acc']}`",
                f"- ah_direct_unit_mae: `{final_ah['ah_direct_unit_mae']}`",
                f"- ah_direct_side_acc: `{final_ah['ah_direct_side_acc']}`",
                f"- ah_cover_n: `{final_ah['ah_cover_n']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
