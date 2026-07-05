"""Train P10 market draw pairwise ranking variants.

P10 keeps the P6 residual Patch-iTransformer architecture and adds only
market-close draw ranking losses. It never uses the test split and never fits
posthoc calibration on validation predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_metrics import logloss_from_probs
from model.odds_patch_itransformer_v2 import FEATURE_DEFS
from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
from tools.p3b_train_patch_itransformer import bucketize_sample_v2_for_training
from tools.p4_goal_diff_utils import compute_anchor_only_metrics, extract_euro_anchor_probs, fold_goal_diff_probs, goal_diff_to_bucket
from tools.p4_train_residual_goal_diff import (
    EURO_MAP,
    FEATURE_INDEX,
    SCALING_CHOICES,
    apply_p4_feature_scaler,
    fit_p4_feature_scaler,
    load_rows_for_ids,
    load_split_ids,
    save_feature_scaler,
    selected_feature_names,
    set_deterministic_seed,
    write_val_goal_diff_predictions_csv,
    write_val_predictions_csv,
)
from tools.p6_train_residual_patch_itransformer import (
    _brier_from_probs,
    _multiclass_ece,
    compute_draw_auxiliary_metrics,
    compute_p6_losses,
)
from tools.p8_draw_signal_forensics import valid_pre_kickoff_euro_events


EPS = 1e-8
FEATURE_MARKET_TYPES = {name: market for name, market in FEATURE_DEFS}


def p10_selected_feature_names(feature_groups: str) -> list[str]:
    return selected_feature_names(feature_groups)


def p10_selected_feature_market_type_ids(feature_groups: str) -> list[int]:
    return [int(FEATURE_MARKET_TYPES[name]) for name in p10_selected_feature_names(feature_groups)]


def prepare_p10_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing_outputs = [
        Path(out_dir) / "report.json",
        Path(out_dir) / "best_model.pth",
        Path(out_dir) / "val_predictions.csv",
        Path(out_dir) / "teacher_coverage.json",
        Path(out_dir) / "scaler.json",
    ]
    if not allow_overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(f"Refusing to overwrite existing P10 outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def _valid_teacher_mask(market_probs: torch.Tensor) -> torch.Tensor:
    probs = market_probs.detach().float()
    finite = torch.isfinite(probs).all(dim=-1)
    positive = (probs > 0).all(dim=-1)
    sums = probs.sum(dim=-1)
    return finite & positive & torch.isfinite(sums) & (sums > 0)


def _normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.float().clamp_min(EPS)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)


def extract_market_close_teacher(row: dict[str, Any]) -> tuple[torch.Tensor, bool, str]:
    events = valid_pre_kickoff_euro_events(row)
    if not events:
        return torch.zeros(3, dtype=torch.float32), False, "no_valid_pre_kickoff_euro"
    event = min(events, key=lambda item: item["_minute"])
    probs = torch.tensor(event["_probs"], dtype=torch.float32)
    if not bool(_valid_teacher_mask(probs.view(1, 3))[0].item()):
        return torch.zeros(3, dtype=torch.float32), False, "invalid_market_probability"
    return _normalize_probs(probs), True, "valid"


def build_p10_dataset(rows: list[dict[str, Any]], feature_groups: str) -> dict[str, Any]:
    names = selected_feature_names(feature_groups)
    indices = [FEATURE_INDEX[name] for name in names]
    X, y_1x2, y_goal_diff, anchors, market_teachers, valid_market, match_ids = [], [], [], [], [], [], []
    teacher_reasons: list[str] = []
    fallback_count = 0
    negative_time_valid_euro_count = 0
    skipped = 0

    for row_idx, row in enumerate(rows):
        label = row.get("label", {})
        result = label.get("euro_result")
        if result not in EURO_MAP or "home_goals" not in label or "away_goals" not in label:
            skipped += 1
            continue
        try:
            home_goals = int(label["home_goals"])
            away_goals = int(label["away_goals"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        timeline = row.get("raw_timeline", []) or row.get("odds_timeline", [])
        xb_full = bucketize_sample_v2_for_training(timeline, clean_missing_markets=True)
        anchor, anchor_diag = extract_euro_anchor_probs(row)
        market, valid, reason = extract_market_close_teacher(row)

        X.append(xb_full[:, indices, :])
        y_1x2.append(EURO_MAP[result])
        y_goal_diff.append(goal_diff_to_bucket(home_goals - away_goals))
        anchors.append(anchor)
        market_teachers.append(market)
        valid_market.append(valid)
        teacher_reasons.append(reason)
        match_ids.append(str(row.get("match_id", row_idx)))
        fallback_count += int(anchor_diag["used_fallback"])
        negative_time_valid_euro_count += int(anchor_diag["negative_time_valid_euro_count"])

    if not X:
        raise ValueError("No P10 labelled rows available")
    return {
        "X": torch.stack(X),
        "y_1x2": torch.tensor(y_1x2, dtype=torch.long),
        "y_goal_diff": torch.tensor(y_goal_diff, dtype=torch.long),
        "p_euro_anchor": torch.stack(anchors),
        "p_market_close": torch.stack(market_teachers),
        "valid_market_teacher": torch.tensor(valid_market, dtype=torch.bool),
        "teacher_reasons": teacher_reasons,
        "match_ids": match_ids,
        "feature_names": names,
        "anchor_fallback_count": fallback_count,
        "negative_time_valid_euro_count": negative_time_valid_euro_count,
        "skipped_rows": skipped,
    }


def build_teacher_coverage(
    train_teacher: torch.Tensor,
    val_teacher: torch.Tensor,
    train_invalid_reasons: list[str] | None = None,
    val_invalid_reasons: list[str] | None = None,
) -> dict[str, Any]:
    train_valid = _valid_teacher_mask(train_teacher)
    val_valid = _valid_teacher_mask(val_teacher)
    train_reasons = train_invalid_reasons or ["valid" if bool(item) else "invalid_market_probability" for item in train_valid]
    val_reasons = val_invalid_reasons or ["valid" if bool(item) else "invalid_market_probability" for item in val_valid]

    def invalid_counts(reasons: list[str]) -> dict[str, int]:
        return dict(sorted(Counter(reason for reason in reasons if reason != "valid").items()))

    train_total = int(train_teacher.shape[0])
    val_total = int(val_teacher.shape[0])
    train_valid_n = int(train_valid.sum().item())
    val_valid_n = int(val_valid.sum().item())
    return {
        "train_total_rows": train_total,
        "train_valid_teacher_rows": train_valid_n,
        "train_invalid_teacher_rows": train_total - train_valid_n,
        "train_teacher_coverage": train_valid_n / max(train_total, 1),
        "val_total_rows": val_total,
        "val_valid_teacher_rows": val_valid_n,
        "val_invalid_teacher_rows": val_total - val_valid_n,
        "val_teacher_coverage": val_valid_n / max(val_total, 1),
        "invalid_reason_counts_train": invalid_counts(train_reasons),
        "invalid_reason_counts_val": invalid_counts(val_reasons),
    }


def market_pairwise_targets(market_probs: torch.Tensor) -> dict[str, torch.Tensor]:
    market = _normalize_probs(market_probs)
    draw_vs_home = market[:, 1] / (market[:, 1] + market[:, 0]).clamp_min(EPS)
    draw_vs_away = market[:, 1] / (market[:, 1] + market[:, 2]).clamp_min(EPS)
    return {"draw_vs_home": draw_vs_home, "draw_vs_away": draw_vs_away}


def market_draw_rank_tie_aware(market_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    market = _normalize_probs(market_probs)
    draw = market[:, 1:2]
    return 1 + (market > draw + float(eps)).sum(dim=-1)


def _ordinary_draw_rank(probs: torch.Tensor) -> torch.Tensor:
    probs = _normalize_probs(probs)
    return (torch.argsort(probs, dim=-1, descending=True) == 1).nonzero(as_tuple=False)[:, 1] + 1


def true_draw_top2_margin_loss(
    final_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    active = (labels == 1) & mask.bool()
    if int(active.sum().item()) == 0:
        return final_logits.new_tensor(0.0)
    logits = final_logits[active]
    min_opponent = torch.minimum(logits[:, 0], logits[:, 2])
    return F.softplus(min_opponent - logits[:, 1] + float(margin)).mean()


def _market_pairwise_loss(
    final_logits: torch.Tensor,
    market_probs: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = mask.bool()
    if int(active.sum().item()) == 0:
        return final_logits.new_tensor(0.0), final_logits.new_tensor(0.0)
    targets = market_pairwise_targets(market_probs)
    logits = final_logits[active]
    draw_home = logits[:, 1] - logits[:, 0]
    draw_away = logits[:, 1] - logits[:, 2]
    loss = 0.5 * F.binary_cross_entropy_with_logits(draw_home, targets["draw_vs_home"][active])
    loss = loss + 0.5 * F.binary_cross_entropy_with_logits(draw_away, targets["draw_vs_away"][active])
    return loss, final_logits.new_tensor(float(active.sum().item()))


def compute_p10_losses(
    out: dict[str, torch.Tensor],
    p_euro_anchor: torch.Tensor,
    y_1x2: torch.Tensor,
    y_goal_diff: torch.Tensor,
    p_market_close: torch.Tensor,
    valid_market_teacher: torch.Tensor,
    diff_loss_weight: float,
    consistency_loss_weight: float,
    delta_l2_weight: float,
    market_draw_pairwise_weight: float = 0.0,
    market_draw_pairwise_mode: str = "all",
    true_draw_top2_margin_weight: float = 0.0,
    true_draw_top2_margin: float = 0.05,
) -> dict[str, torch.Tensor]:
    losses = compute_p6_losses(
        out,
        p_euro_anchor,
        y_1x2,
        y_goal_diff,
        diff_loss_weight=diff_loss_weight,
        consistency_loss_weight=consistency_loss_weight,
        delta_l2_weight=delta_l2_weight,
    )
    valid = valid_market_teacher.bool()
    if market_draw_pairwise_mode == "all":
        pairwise_mask = valid
    elif market_draw_pairwise_mode == "market_draw_top2":
        pairwise_mask = valid & (market_draw_rank_tie_aware(p_market_close) <= 2)
    else:
        raise ValueError(f"Unknown market_draw_pairwise_mode: {market_draw_pairwise_mode}")

    if float(market_draw_pairwise_weight) > 0:
        l_pair, pairwise_n = _market_pairwise_loss(losses["final_logits"], p_market_close, pairwise_mask)
        losses["loss"] = losses["loss"] + float(market_draw_pairwise_weight) * l_pair
    else:
        l_pair = losses["loss"].new_tensor(0.0)
        pairwise_n = losses["loss"].new_tensor(float(pairwise_mask.sum().item()))

    top2_margin_mask = valid & (market_draw_rank_tie_aware(p_market_close) <= 2)
    if float(true_draw_top2_margin_weight) > 0:
        l_margin = true_draw_top2_margin_loss(
            losses["final_logits"],
            y_1x2,
            top2_margin_mask,
            margin=true_draw_top2_margin,
        )
        losses["loss"] = losses["loss"] + float(true_draw_top2_margin_weight) * l_margin
    else:
        l_margin = losses["loss"].new_tensor(0.0)

    losses["L_market_draw_pairwise"] = l_pair
    losses["L_true_draw_top2_margin"] = l_margin
    losses["market_pairwise_n"] = pairwise_n
    losses["market_focus_n"] = losses["loss"].new_tensor(float(top2_margin_mask.sum().item()))
    return losses


def _pearson(xs: torch.Tensor, ys: torch.Tensor) -> float | None:
    if xs.numel() < 2 or xs.numel() != ys.numel():
        return None
    x = xs.float()
    y = ys.float()
    dx = x - x.mean()
    dy = y - y.mean()
    denom = torch.sqrt((dx * dx).sum() * (dy * dy).sum())
    if float(denom.item()) == 0.0:
        return None
    return float(((dx * dy).sum() / denom).item())


def compute_market_pairwise_metrics(probs: torch.Tensor, market_probs: torch.Tensor, valid_teacher: torch.Tensor) -> dict[str, Any]:
    probs = _normalize_probs(probs.detach().cpu())
    market = _normalize_probs(market_probs.detach().cpu())
    valid = valid_teacher.detach().cpu().bool() & _valid_teacher_mask(market)
    out: dict[str, Any] = {
        "valid_n": int(valid.sum().item()),
        "skipped_n": int((~valid).sum().item()),
    }
    if int(valid.sum().item()) == 0:
        out.update(
            {
                "market_pairwise_agreement": 0.0,
                "model_vs_market_draw_corr": None,
                "model_vs_market_draw_mae": 0.0,
                "pairwise_bce": 0.0,
            }
        )
        return out

    p = probs[valid]
    m = market[valid]
    model_draw_home = p[:, 1] >= p[:, 0]
    model_draw_away = p[:, 1] >= p[:, 2]
    market_draw_home = m[:, 1] >= m[:, 0]
    market_draw_away = m[:, 1] >= m[:, 2]
    agreement = torch.cat([(model_draw_home == market_draw_home).float(), (model_draw_away == market_draw_away).float()]).mean()
    targets = market_pairwise_targets(m)
    pairwise_bce = 0.5 * F.binary_cross_entropy((p[:, 1] / (p[:, 1] + p[:, 0]).clamp_min(EPS)).clamp(EPS, 1 - EPS), targets["draw_vs_home"])
    pairwise_bce = pairwise_bce + 0.5 * F.binary_cross_entropy(
        (p[:, 1] / (p[:, 1] + p[:, 2]).clamp_min(EPS)).clamp(EPS, 1 - EPS),
        targets["draw_vs_away"],
    )
    out.update(
        {
            "market_pairwise_agreement": float(agreement.item()),
            "model_vs_market_draw_corr": _pearson(p[:, 1], m[:, 1]),
            "model_vs_market_draw_mae": float((p[:, 1] - m[:, 1]).abs().mean().item()),
            "pairwise_bce": float(pairwise_bce.item()),
            "market_draw_top2_tie_aware_n": int((market_draw_rank_tie_aware(m) <= 2).sum().item()),
        }
    )
    return out


def _classwise_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probs = _normalize_probs(probs.detach().cpu())
    labels = labels.detach().cpu().long()
    preds = probs.argmax(dim=-1)
    out = {}
    for idx, name in [(0, "home"), (1, "draw"), (2, "away")]:
        mask = labels == idx
        out[f"{name}_n"] = int(mask.sum().item())
        out[f"{name}_nll"] = float((-torch.log(probs[mask, idx].clamp_min(EPS))).mean().item()) if int(mask.sum().item()) else 0.0
        out[f"{name}_recall"] = float((preds[mask] == idx).float().mean().item()) if int(mask.sum().item()) else 0.0
    return out


def _draw_rank_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probs = _normalize_probs(probs.detach().cpu())
    labels = labels.detach().cpu().long()
    ranks = _ordinary_draw_rank(probs)
    true_draw = labels == 1
    top2 = torch.topk(probs, k=2, dim=-1).indices
    draw_margin = probs.max(dim=-1).values - probs[:, 1]
    return {
        "draw_rank1_count": int((ranks == 1).sum().item()),
        "draw_rank2_count": int((ranks == 2).sum().item()),
        "draw_rank3_count": int((ranks == 3).sum().item()),
        "draw_top2": float((top2[true_draw] == 1).any(dim=-1).float().mean().item()) if int(true_draw.sum().item()) else 0.0,
        "mean_draw_margin_to_top": float(draw_margin.mean().item()) if probs.numel() else 0.0,
    }


def train_epoch(
    model: P6ResidualPatchITransformer,
    data: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    diff_loss_weight: float,
    consistency_loss_weight: float,
    delta_l2_weight: float,
    market_draw_pairwise_weight: float,
    market_draw_pairwise_mode: str,
    true_draw_top2_margin_weight: float,
    true_draw_top2_margin: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "L_1x2": 0.0,
        "L_diff": 0.0,
        "L_consistency": 0.0,
        "L_delta": 0.0,
        "L_market_draw_pairwise": 0.0,
        "L_true_draw_top2_margin": 0.0,
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
        ab = data["p_euro_anchor"][idx].to(device)
        mb = data["p_market_close"][idx].to(device)
        vb = data["valid_market_teacher"][idx].to(device)
        out = model(xb)
        losses = compute_p10_losses(
            out,
            ab,
            yb,
            ygb,
            mb,
            vb,
            diff_loss_weight=diff_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            delta_l2_weight=delta_l2_weight,
            market_draw_pairwise_weight=market_draw_pairwise_weight,
            market_draw_pairwise_mode=market_draw_pairwise_mode,
            true_draw_top2_margin_weight=true_draw_top2_margin_weight,
            true_draw_top2_margin=true_draw_top2_margin,
        )
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
    p_final, p_from_diff, q_diff, delta_logits = [], [], [], []
    for start in range(0, len(data["X"]), batch_size):
        xb = data["X"][start : start + batch_size].to(device)
        ab = data["p_euro_anchor"][start : start + batch_size].to(device)
        out = model(xb)
        final_logits = torch.log(ab.clamp_min(EPS)) + out["delta_logits"]
        p_final.append(torch.softmax(final_logits, dim=-1).cpu())
        q = torch.softmax(out["goal_diff_logits"], dim=-1)
        q_diff.append(q.cpu())
        p_from_diff.append(fold_goal_diff_probs(q).cpu())
        delta_logits.append(out["delta_logits"].cpu())
    return {
        "p_final": torch.cat(p_final, dim=0),
        "p_from_diff": torch.cat(p_from_diff, dim=0),
        "q_diff": torch.cat(q_diff, dim=0),
        "delta_logits": torch.cat(delta_logits, dim=0),
    }


@torch.no_grad()
def evaluate(model: P6ResidualPatchITransformer, data: dict[str, Any], batch_size: int, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    preds = predict_outputs(model, data, batch_size, device)
    labels = data["y_1x2"]
    goal_labels = data["y_goal_diff"]
    p_final = preds["p_final"].clamp_min(EPS)
    q = preds["q_diff"].clamp_min(EPS)
    p_from_diff = preds["p_from_diff"].clamp_min(EPS)
    val_metrics = compute_draw_auxiliary_metrics(p_final, labels)
    val_metrics.update(
        {
            "accuracy": float((p_final.argmax(dim=-1) == labels).float().mean().item()),
            "logloss": logloss_from_probs(p_final, labels),
            "brier": _brier_from_probs(p_final, labels, 3),
            "ece": _multiclass_ece(p_final, labels, 10),
        }
    )
    val_metrics.update(_draw_rank_metrics(p_final, labels))
    true_zero = goal_labels == 3
    pred_zero = q.argmax(dim=-1) == 3
    zero_recall = float((true_zero & pred_zero).sum().item() / true_zero.sum().item()) if int(true_zero.sum().item()) else 0.0
    goal_diff_metrics = {
        "diff_nll": logloss_from_probs(q, goal_labels),
        "diff_acc": float((q.argmax(dim=-1) == goal_labels).float().mean().item()),
        "bucket_acc": float((q.argmax(dim=-1) == goal_labels).float().mean().item()),
        "zero_recall": zero_recall,
        "p_from_diff_logloss": logloss_from_probs(p_from_diff, labels),
        "final_vs_diff_kl": float((p_from_diff * (p_from_diff.log() - p_final.log())).sum(dim=-1).mean().item()),
    }
    market_pairwise = compute_market_pairwise_metrics(p_final, data["p_market_close"], data["valid_market_teacher"])
    return val_metrics, goal_diff_metrics, market_pairwise, preds


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_market_teacher_csv(path: Path, match_ids: list[str], teacher: torch.Tensor, valid: torch.Tensor, reasons: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match_id", "valid_teacher", "reason", "m_home", "m_draw", "m_away", "market_draw_rank_tie_aware"])
        writer.writeheader()
        ranks = market_draw_rank_tie_aware(teacher)
        for idx, probs in enumerate(teacher):
            writer.writerow(
                {
                    "match_id": match_ids[idx],
                    "valid_teacher": int(bool(valid[idx].item())),
                    "reason": reasons[idx],
                    "m_home": float(probs[0].item()),
                    "m_draw": float(probs[1].item()),
                    "m_away": float(probs[2].item()),
                    "market_draw_rank_tie_aware": int(ranks[idx].item()) if bool(valid[idx].item()) else "",
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train P10 market draw pairwise ranking variants")
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
    parser.add_argument("--diff-loss-weight", type=float, default=0.30)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--delta-l2-weight", type=float, default=0.01)
    parser.add_argument("--enable-market-draw-pairwise", action="store_true")
    parser.add_argument("--market-draw-pairwise-weight", type=float, default=0.0)
    parser.add_argument("--market-draw-pairwise-mode", default="all", choices=("all", "market_draw_top2"))
    parser.add_argument("--market-pairwise-tau", type=float, default=1.0)
    parser.add_argument("--enable-true-draw-top2-margin", action="store_true")
    parser.add_argument("--true-draw-top2-margin-weight", type=float, default=0.0)
    parser.add_argument("--true-draw-top2-margin", type=float, default=0.05)
    parser.add_argument("--scaling", default="robust", choices=SCALING_CHOICES)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--variant", default="manual")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if "test" in Path(args.train_ids).name.lower() or "test" in Path(args.val_ids).name.lower():
        raise SystemExit("P10 refuses any test split path")
    if args.market_pairwise_tau != 1.0:
        raise SystemExit("P10 keeps market_pairwise_tau fixed at 1.0 in this phase")
    feature_names = p10_selected_feature_names(args.feature_groups)
    market_type_ids = p10_selected_feature_market_type_ids(args.feature_groups)
    set_deterministic_seed(args.seed)
    prepare_p10_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_rows = load_rows_for_ids(args.data, load_split_ids(args.train_ids))
    val_rows = load_rows_for_ids(args.data, load_split_ids(args.val_ids))
    if args.max_samples > 0:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: min(args.max_samples, len(val_rows))]
    print(f"Rows: train={len(train_rows)} val={len(val_rows)}")

    train_data = build_p10_dataset(train_rows, args.feature_groups)
    val_data = build_p10_dataset(val_rows, args.feature_groups)
    if train_data["feature_names"] != feature_names or val_data["feature_names"] != feature_names:
        raise ValueError("P10 selected feature names drifted from dataset construction")
    coverage = build_teacher_coverage(
        train_data["p_market_close"],
        val_data["p_market_close"],
        train_data["teacher_reasons"],
        val_data["teacher_reasons"],
    )
    write_json(Path(args.out_dir) / "teacher_coverage.json", coverage)

    scaler = fit_p4_feature_scaler(train_data["X"], feature_names, args.scaling)
    scaler_path = save_feature_scaler(scaler, args.out_dir)
    train_data["X"] = apply_p4_feature_scaler(train_data["X"], scaler)
    val_data["X"] = apply_p4_feature_scaler(val_data["X"], scaler)
    anchor_only = compute_anchor_only_metrics(val_data["p_euro_anchor"], val_data["y_1x2"])

    model = P6ResidualPatchITransformer(
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
    best_val_logloss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    pairwise_weight = args.market_draw_pairwise_weight if args.enable_market_draw_pairwise else 0.0
    margin_weight = args.true_draw_top2_margin_weight if args.enable_true_draw_top2_margin else 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_data,
            optimizer,
            args.batch_size,
            device,
            args.diff_loss_weight,
            args.consistency_loss_weight,
            args.delta_l2_weight,
            pairwise_weight,
            args.market_draw_pairwise_mode,
            margin_weight,
            args.true_draw_top2_margin,
        )
        val_metrics, goal_diff_metrics, market_pairwise_metrics, _ = evaluate(model, val_data, args.batch_size * 2, device)
        train_pairwise_metrics = compute_market_pairwise_metrics(
            predict_outputs(model, train_data, args.batch_size * 2, device)["p_final"],
            train_data["p_market_close"],
            train_data["valid_market_teacher"],
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "train_pairwise_bce": train_pairwise_metrics.get("pairwise_bce", 0.0),
            "val_logloss": val_metrics["logloss"],
            "val_ece": val_metrics["ece"],
            "val_accuracy": val_metrics["accuracy"],
            "val_draw_class_nll": val_metrics["draw_class_nll"],
            "val_draw_recall": val_metrics["draw_recall"],
            "val_draw_precision": val_metrics["draw_precision"],
            "val_draw_top2_recall": val_metrics["draw_top2_recall"],
            "val_mean_p_draw_on_true_draw": val_metrics["mean_p_draw_on_true_draw"],
            "val_mean_draw_margin_to_top": val_metrics["mean_draw_margin_to_top"],
            "val_pairwise_bce": market_pairwise_metrics.get("pairwise_bce", 0.0),
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
                },
                Path(args.out_dir) / "best_model.pth",
            )
        print(
            f"Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
            f"val_logloss={val_metrics['logloss']:.4f} "
            f"draw_top2={val_metrics['draw_top2_recall']:.4f}"
            + (" *" if is_best else "")
        )

    ckpt_path = Path(args.out_dir) / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_val, final_goal_diff, final_market_pairwise, preds = evaluate(model, val_data, args.batch_size * 2, device)
    final_train_pairwise = compute_market_pairwise_metrics(
        predict_outputs(model, train_data, args.batch_size * 2, device)["p_final"],
        train_data["p_market_close"],
        train_data["valid_market_teacher"],
    )
    write_val_predictions_csv(str(Path(args.out_dir) / "val_predictions.csv"), val_data["match_ids"], val_data["y_1x2"], preds["p_final"])
    write_val_goal_diff_predictions_csv(
        str(Path(args.out_dir) / "val_goal_diff_predictions.csv"),
        val_data["match_ids"],
        val_data["y_goal_diff"],
        preds["q_diff"],
        preds["p_from_diff"],
    )
    write_market_teacher_csv(Path(args.out_dir) / "val_market_teacher.csv", val_data["match_ids"], val_data["p_market_close"], val_data["valid_market_teacher"], val_data["teacher_reasons"])

    classwise_metrics = _classwise_metrics(preds["p_final"], val_data["y_1x2"])
    draw_rank = _draw_rank_metrics(preds["p_final"], val_data["y_1x2"])
    calibration = {"ece": final_val["ece"], "classwise_ece_draw": final_val["classwise_ece_draw"]}
    goal_diff_probe = {
        **final_goal_diff,
        "available": True,
        "unavailable_reason": "",
    }
    write_json(Path(args.out_dir) / "val_metrics.json", final_val)
    write_json(Path(args.out_dir) / "classwise_metrics.json", classwise_metrics)
    write_json(Path(args.out_dir) / "draw_rank_metrics.json", draw_rank)
    write_json(Path(args.out_dir) / "calibration_metrics.json", calibration)
    write_json(Path(args.out_dir) / "market_pairwise_metrics.json", {"train": final_train_pairwise, "val": final_market_pairwise})
    write_json(Path(args.out_dir) / "goal_diff_structure_probe.json", goal_diff_probe)

    data_summary = {
        "train_samples": int(len(train_data["X"])),
        "val_samples": int(len(val_data["X"])),
        "test_ids_used": False,
        "validation_fit_used": False,
        "posthoc_val_fit_used": False,
        "anchor_type": "euro_close_de_vig",
        "teacher_type": "market_close_draw_pairwise_de_vig",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "train_skipped_rows": int(train_data["skipped_rows"]),
        "val_skipped_rows": int(val_data["skipped_rows"]),
        "anchor_fallback_count": int(val_data["anchor_fallback_count"]),
        "negative_time_valid_euro_count": int(val_data["negative_time_valid_euro_count"]),
    }
    training_manifest = {
        "phase": "P10",
        "variant": args.variant,
        "seed": args.seed,
        "pairwise_weight": pairwise_weight,
        "pairwise_mode": args.market_draw_pairwise_mode,
        "true_draw_top2_margin_weight": margin_weight,
        "p6_default_behavior_changed": False,
    }
    write_json(Path(args.out_dir) / "training_manifest.json", training_manifest)
    report = {
        "phase": "P10 Market Draw Pairwise Ranking",
        "run_mode": "smoke" if args.max_samples else "formal",
        "model": "P6ResidualPatchITransformer",
        "config": vars(args),
        "data": data_summary,
        "teacher_coverage": coverage,
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "anchor_only_baseline": anchor_only,
        "train_metrics": history[-1] if history else {},
        "val_metrics": final_val,
        "classwise_metrics": classwise_metrics,
        "draw_rank_metrics": draw_rank,
        "goal_diff_metrics": final_goal_diff,
        "goal_diff_structure_probe": goal_diff_probe,
        "market_pairwise_metrics": {"train": final_train_pairwise, "val": final_market_pairwise},
        "history": history,
        "warnings": [],
        "params": n_params,
        "scaler": {"path": scaler_path, **scaler},
        "run_path": args.out_dir,
        "artifacts": {
            "best_model": str(ckpt_path),
            "val_predictions": str(Path(args.out_dir) / "val_predictions.csv"),
            "val_goal_diff_predictions": str(Path(args.out_dir) / "val_goal_diff_predictions.csv"),
            "teacher_coverage": str(Path(args.out_dir) / "teacher_coverage.json"),
        },
    }
    write_json(Path(args.out_dir) / "report.json", report)
    (Path(args.out_dir) / "report.md").write_text(
        "\n".join(
            [
                "# P10 Run Report",
                "",
                f"- variant: `{args.variant}`",
                f"- seed: `{args.seed}`",
                f"- val_logloss: `{final_val['logloss']}`",
                f"- draw_top2: `{final_val['draw_top2_recall']}`",
                f"- mean_p_draw_true_draw: `{final_val['mean_p_draw_on_true_draw']}`",
                f"- train_teacher_coverage: `{coverage['train_teacher_coverage']}`",
                f"- val_teacher_coverage: `{coverage['val_teacher_coverage']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Report saved to {Path(args.out_dir) / 'report.json'}")
    print(f"Best val_logloss: {best_val_logloss:.6f}")
    print(f"Anchor-only val_logloss: {anchor_only['anchor_only_val_logloss']:.6f}")


if __name__ == "__main__":
    main()
