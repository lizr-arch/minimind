"""Train P14 market-replication objective probes.

P14 keeps the P6 residual Patch-iTransformer backbone and adds one explicit
diagnostic objective: preserve/replicate the closing no-vig Euro market
probabilities while still training on the realized 1X2 label. It reads train
and val splits only and never performs posthoc validation fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_metrics import logloss_from_probs
from model.odds_patch_itransformer_v2 import FEATURE_DEFS
from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
from tools.p3b_train_patch_itransformer import bucketize_sample_v2_for_training
from tools.p4_goal_diff_utils import compute_anchor_only_metrics, extract_euro_anchor_probs, fold_goal_diff_probs, goal_diff_to_bucket
from tools.p4_train_residual_goal_diff import EURO_MAP, FEATURE_INDEX, load_rows_for_ids, load_split_ids
from tools.p6_train_residual_patch_itransformer import _brier_from_probs, _multiclass_ece, compute_draw_auxiliary_metrics


EPS = 1e-8
SCALING_CHOICES = ("none", "standard", "robust")
SCALER_EXCLUDED_FEATURES = {"has_euro", "has_asian", "has_ou", "bucket_valid_count"}
FEATURE_MARKET_TYPES = {name: market for name, market in FEATURE_DEFS}
P14_FEATURES = {
    "euro": ["euro_h", "euro_d", "euro_a", "imp_h", "imp_d", "imp_a", "euro_margin", "has_euro"],
    "asian": [
        "asian_line",
        "upper_water",
        "lower_water",
        "asian_upper_implied",
        "asian_lower_implied",
        "asian_margin",
        "has_asian",
    ],
    "ou": ["over_under_line", "over_water", "under_water", "over_implied", "under_implied", "ou_margin", "has_ou"],
    "time": ["time_log", "bucket_valid_count"],
}


class P14ProtocolError(RuntimeError):
    pass


def assert_no_test_paths(paths: Iterable[str | Path]) -> None:
    for path in paths:
        parts = [part.lower() for part in Path(path).parts]
        if any(part == "test" or part.startswith("test_") or part.startswith("test-") for part in parts):
            raise P14ProtocolError(f"P14 refuses test split or test artifact path: {path}")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def p14_selected_feature_names(feature_groups: str) -> list[str]:
    selected = {part.strip().lower() for part in feature_groups.split(",") if part.strip()}
    allowed = ({"euro"}, {"euro", "asian"}, {"euro", "ou"}, {"euro", "asian", "ou"})
    if selected not in allowed:
        raise ValueError("P14 feature_groups must be one of: euro, euro,asian, euro,ou, euro,asian,ou")
    names = list(P14_FEATURES["euro"])
    if "asian" in selected:
        names.extend(P14_FEATURES["asian"])
    if "ou" in selected:
        names.extend(P14_FEATURES["ou"])
    names.extend(P14_FEATURES["time"])
    return names


def p14_selected_feature_market_type_ids(feature_groups: str) -> list[int]:
    return [int(FEATURE_MARKET_TYPES[name]) for name in p14_selected_feature_names(feature_groups)]


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _latest_pre_kickoff_market_value(row: dict[str, Any], field: str, has_field: str) -> float | None:
    values: list[tuple[float, float]] = []
    for event in row.get("raw_timeline", []) or row.get("odds_timeline", []) or []:
        minute = _safe_float(event.get("minutes_before_kickoff"))
        value = _safe_float(event.get(field))
        if minute is None or minute < 0 or value is None:
            continue
        if has_field and not bool(event.get(has_field)):
            continue
        values.append((minute, value))
    if not values:
        return None
    return min(values, key=lambda item: item[0])[1]


def draw_risk_bucket(asian_line: float | None, ou_line: float | None) -> int:
    """Return low/mid/high draw-risk class from interpretable AH/OU buckets."""
    score = 0
    if asian_line is not None:
        abs_line = abs(float(asian_line))
        if 0.0 < abs_line <= 0.50:
            score += 2
        elif abs_line >= 1.25:
            score -= 1
    if ou_line is not None:
        ou = float(ou_line)
        if ou <= 2.50:
            score += 1
        elif ou >= 3.00:
            score -= 1
    if score >= 2:
        return 2
    if score <= -1:
        return 0
    return 1


def prepare_p14_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing_outputs = [
        Path(out_dir) / "report.json",
        Path(out_dir) / "report.md",
        Path(out_dir) / "best_model.pth",
        Path(out_dir) / "val_predictions.csv",
        Path(out_dir) / "val_market_replication.csv",
        Path(out_dir) / "val_goal_diff_predictions.csv",
        Path(out_dir) / "scaler.json",
    ]
    if not allow_overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(f"Refusing to overwrite existing P14 outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def build_p14_dataset(rows: list[dict[str, Any]], feature_groups: str) -> dict[str, Any]:
    names = p14_selected_feature_names(feature_groups)
    indices = [FEATURE_INDEX[name] for name in names]
    x_rows, y_1x2, y_goal_diff, y_draw_risk, anchors, match_ids = [], [], [], [], [], []
    fallback_count = 0
    negative_time_valid_euro_count = 0
    skipped_rows = 0

    for row_idx, row in enumerate(rows):
        label = row.get("label", {}) or {}
        result = label.get("euro_result")
        if result not in EURO_MAP or "home_goals" not in label or "away_goals" not in label:
            skipped_rows += 1
            continue
        try:
            home_goals = int(label["home_goals"])
            away_goals = int(label["away_goals"])
        except (TypeError, ValueError):
            skipped_rows += 1
            continue

        timeline = row.get("raw_timeline", []) or row.get("odds_timeline", [])
        xb_full = bucketize_sample_v2_for_training(timeline, clean_missing_markets=True)
        anchor, anchor_diag = extract_euro_anchor_probs(row)

        x_rows.append(xb_full[:, indices, :])
        y_1x2.append(EURO_MAP[result])
        y_goal_diff.append(goal_diff_to_bucket(home_goals - away_goals))
        asian_line = _latest_pre_kickoff_market_value(row, "asian_line", "has_asian")
        ou_line = _latest_pre_kickoff_market_value(row, "over_under_line", "has_over_under")
        y_draw_risk.append(draw_risk_bucket(asian_line, ou_line))
        anchors.append(anchor)
        match_ids.append(str(row.get("match_id", row_idx)))
        fallback_count += int(anchor_diag["used_fallback"])
        negative_time_valid_euro_count += int(anchor_diag["negative_time_valid_euro_count"])

    if not x_rows:
        raise ValueError("No P14 labelled rows available")
    return {
        "X": torch.stack(x_rows),
        "y_1x2": torch.tensor(y_1x2, dtype=torch.long),
        "y_goal_diff": torch.tensor(y_goal_diff, dtype=torch.long),
        "y_draw_risk": torch.tensor(y_draw_risk, dtype=torch.long),
        "p_market": torch.stack(anchors),
        "p_euro_anchor": torch.stack(anchors),
        "match_ids": match_ids,
        "feature_names": names,
        "anchor_fallback_count": fallback_count,
        "negative_time_valid_euro_count": negative_time_valid_euro_count,
        "skipped_rows": skipped_rows,
        "draw_risk_counts": _label_counts(y_draw_risk, ("low", "mid", "high")),
    }


def fit_feature_scaler(x_train: torch.Tensor, feature_names: list[str], scaling: str) -> dict[str, Any]:
    if scaling not in SCALING_CHOICES:
        raise ValueError(f"Unknown scaling: {scaling}")
    scaler = {
        "scaling": scaling,
        "fit_split": "train",
        "feature_names": feature_names,
        "excluded_features": [name for name in feature_names if name in SCALER_EXCLUDED_FEATURES],
        "scaled_features": [],
        "features": {},
    }
    if scaling == "none":
        return scaler
    for idx, name in enumerate(feature_names):
        if name in SCALER_EXCLUDED_FEATURES:
            continue
        values = x_train[..., idx, :].reshape(-1).float()
        if values.numel() == 0:
            center, scale = 0.0, 1.0
            stats = {"index": idx, "center": center, "scale": scale}
        elif scaling == "standard":
            center = _tensor_float(values.mean())
            scale = _safe_scale(values)
            stats = {"index": idx, "center": center, "scale": scale}
        else:
            center = _tensor_float(torch.quantile(values, 0.5))
            q25 = _tensor_float(torch.quantile(values, 0.25))
            q75 = _tensor_float(torch.quantile(values, 0.75))
            scale = q75 - q25
            fallback = None
            if not math.isfinite(scale) or scale <= 1e-6:
                scale = _safe_scale(values)
                fallback = "std_or_unit"
            stats = {"index": idx, "center": center, "scale": scale, "q25": q25, "q75": q75, "fallback": fallback}
        scaler["scaled_features"].append(name)
        scaler["features"][name] = stats
    return scaler


def apply_feature_scaler(x: torch.Tensor, scaler: dict[str, Any]) -> torch.Tensor:
    if scaler.get("scaling") == "none":
        return x
    for stats in scaler.get("features", {}).values():
        idx = int(stats["index"])
        x[..., idx, :] = (x[..., idx, :] - float(stats["center"])) / float(stats["scale"])
    return x


def save_feature_scaler(scaler: dict[str, Any], out_dir: str) -> str:
    path = Path(out_dir) / "scaler.json"
    path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    return str(path)


def load_best_model_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def checkpoint_sort_key(row: dict[str, Any], checkpoint_selection: str, balanced_logloss_ceiling: float) -> tuple[Any, ...]:
    if checkpoint_selection == "logloss":
        return (
            float(row["val_logloss"]),
            float(row.get("market_kl", 0.0)),
            -float(row.get("val_draw_top2", 0.0)),
            int(row.get("epoch", 0)),
        )
    if checkpoint_selection != "balanced":
        raise ValueError(f"Unknown checkpoint_selection: {checkpoint_selection}")
    ceiling = float(balanced_logloss_ceiling)
    eligible = float(row["val_logloss"]) <= ceiling if ceiling > 0 else True
    if eligible:
        return (
            0,
            float(row.get("market_kl", 0.0)),
            -float(row.get("val_draw_top2", 0.0)),
            float(row["val_logloss"]),
            int(row.get("epoch", 0)),
        )
    return (
        1,
        float(row["val_logloss"]),
        float(row.get("market_kl", 0.0)),
        -float(row.get("val_draw_top2", 0.0)),
        int(row.get("epoch", 0)),
    )


def select_checkpoint_row(
    history: list[dict[str, Any]],
    checkpoint_selection: str,
    balanced_logloss_ceiling: float,
) -> dict[str, Any]:
    if not history:
        raise ValueError("Cannot select checkpoint from empty history")
    return min(history, key=lambda row: checkpoint_sort_key(row, checkpoint_selection, balanced_logloss_ceiling))


def should_replace_checkpoint(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    checkpoint_selection: str,
    balanced_logloss_ceiling: float,
) -> bool:
    if current is None:
        return True
    return checkpoint_sort_key(candidate, checkpoint_selection, balanced_logloss_ceiling) < checkpoint_sort_key(
        current,
        checkpoint_selection,
        balanced_logloss_ceiling,
    )


def build_epoch_history_row(
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    market_metrics: dict[str, Any],
    goal_diff_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_metrics["loss"]),
        "train_L_market_kl": float(train_metrics["L_market_kl"]),
        "train_L_draw_risk": float(train_metrics.get("L_draw_risk", 0.0)),
        "val_logloss": float(val_metrics["logloss"]),
        "val_ece": float(val_metrics.get("ece", 0.0)),
        "val_draw_class_nll": float(val_metrics.get("draw_class_nll", 0.0)),
        "val_draw_top2": float(val_metrics["draw_top2_recall"]),
        "val_mean_p_draw": float(val_metrics.get("mean_p_draw", 0.0)),
        "val_mean_p_draw_on_true_draw": float(val_metrics["mean_p_draw_on_true_draw"]),
        "market_kl": float(market_metrics["market_kl"]),
        "market_draw_mae": float(market_metrics["market_draw_mae"]),
        "model_minus_market_p_draw": float(market_metrics["model_minus_market_p_draw"]),
        "market_draw_top2": float(market_metrics.get("market_draw_top2", 0.0)),
        "final_mean_p_draw": float(market_metrics.get("final_mean_p_draw", 0.0)),
        "draw_risk_acc": float(goal_diff_metrics.get("draw_risk_acc", 0.0)),
    }


def compute_p14_losses(
    out: dict[str, torch.Tensor],
    p_market: torch.Tensor,
    y_1x2: torch.Tensor,
    y_goal_diff: torch.Tensor,
    y_draw_risk: torch.Tensor | None = None,
    market_loss_weight: float = 0.10,
    label_loss_weight: float = 1.0,
    diff_loss_weight: float = 0.30,
    consistency_loss_weight: float = 0.10,
    delta_l2_weight: float = 0.01,
    draw_risk_loss_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    p_market = p_market.float().clamp_min(EPS)
    p_market = p_market / p_market.sum(dim=-1, keepdim=True).clamp_min(EPS)
    delta_logits = out["delta_logits"]
    final_logits = torch.log(p_market) + delta_logits
    p_final = F.softmax(final_logits, dim=-1)
    q_diff = F.softmax(out["goal_diff_logits"], dim=-1)
    p_from_diff = fold_goal_diff_probs(q_diff).clamp_min(EPS)
    p_from_diff = p_from_diff / p_from_diff.sum(dim=-1, keepdim=True).clamp_min(EPS)

    l_1x2 = F.cross_entropy(final_logits, y_1x2)
    l_market_kl = F.kl_div(F.log_softmax(final_logits, dim=-1), p_market.detach(), reduction="batchmean")
    l_diff = F.cross_entropy(out["goal_diff_logits"], y_goal_diff)
    l_consistency = F.kl_div(F.log_softmax(final_logits, dim=-1), p_from_diff.detach(), reduction="batchmean")
    l_delta = delta_logits.pow(2).mean()
    if float(draw_risk_loss_weight) > 0:
        if "draw_risk_logits" not in out:
            raise ValueError("draw_risk_logits are required when draw_risk_loss_weight > 0")
        if y_draw_risk is None:
            raise ValueError("y_draw_risk is required when draw_risk_loss_weight > 0")
        l_draw_risk = F.cross_entropy(out["draw_risk_logits"], y_draw_risk)
    else:
        l_draw_risk = final_logits.new_tensor(0.0)
    loss = (
        float(label_loss_weight) * l_1x2
        + float(market_loss_weight) * l_market_kl
        + float(diff_loss_weight) * l_diff
        + float(consistency_loss_weight) * l_consistency
        + float(delta_l2_weight) * l_delta
        + float(draw_risk_loss_weight) * l_draw_risk
    )
    return {
        "loss": loss,
        "L_1x2": l_1x2,
        "L_market_kl": l_market_kl,
        "L_diff": l_diff,
        "L_consistency": l_consistency,
        "L_delta": l_delta,
        "L_draw_risk": l_draw_risk,
        "p_final": p_final,
        "p_from_diff": p_from_diff,
        "q_diff": q_diff,
        "final_logits": final_logits,
    }


def compute_market_replication_metrics(p_final: torch.Tensor, p_market: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    p_final = _normalize_probs(p_final.detach().cpu().float())
    p_market = _normalize_probs(p_market.detach().cpu().float())
    labels = labels.detach().cpu().long()
    true_draw = labels == 1
    market_kl = (p_market * (p_market.clamp_min(EPS).log() - p_final.clamp_min(EPS).log())).sum(dim=-1)
    draw_delta = p_final[:, 1] - p_market[:, 1]
    return {
        "market_kl": float(market_kl.mean().item()) if market_kl.numel() else 0.0,
        "market_draw_mae": float(draw_delta.abs().mean().item()) if draw_delta.numel() else 0.0,
        "model_minus_market_p_draw": float(draw_delta.mean().item()) if draw_delta.numel() else 0.0,
        "model_market_draw_corr": _pearson(p_final[:, 1], p_market[:, 1]),
        "market_logloss": logloss_from_probs(p_market, labels),
        "market_draw_top2": _draw_top2_recall(p_market, labels),
        "final_draw_top2": _draw_top2_recall(p_final, labels),
        "market_mean_p_draw": float(p_market[:, 1].mean().item()) if p_market.numel() else 0.0,
        "final_mean_p_draw": float(p_final[:, 1].mean().item()) if p_final.numel() else 0.0,
        "market_mean_p_draw_on_true_draw": float(p_market[true_draw, 1].mean().item()) if int(true_draw.sum().item()) else 0.0,
        "final_mean_p_draw_on_true_draw": float(p_final[true_draw, 1].mean().item()) if int(true_draw.sum().item()) else 0.0,
    }


def compute_draw_risk_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()
    probs = torch.softmax(logits, dim=-1).clamp_min(EPS)
    preds = probs.argmax(dim=-1)
    return {
        "draw_risk_acc": float((preds == labels).float().mean().item()) if labels.numel() else 0.0,
        "draw_risk_nll": logloss_from_probs(probs, labels),
        "draw_risk_counts": _label_counts(labels.tolist(), ("low", "mid", "high")),
        "draw_risk_pred_counts": _label_counts(preds.tolist(), ("low", "mid", "high")),
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
        "L_draw_risk": 0.0,
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
        yrb = data["y_draw_risk"][idx].to(device)
        market = data["p_market"][idx].to(device)
        losses = compute_p14_losses(model(xb), market, yb, ygb, y_draw_risk=yrb, **config)
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
    draw_risk_logits = []
    for start in range(0, len(data["X"]), batch_size):
        xb = data["X"][start : start + batch_size].to(device)
        market = data["p_market"][start : start + batch_size].to(device)
        out = model(xb)
        losses = compute_p14_losses(
            out,
            market,
            data["y_1x2"][start : start + batch_size].to(device),
            data["y_goal_diff"][start : start + batch_size].to(device),
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
        if "draw_risk_logits" in out:
            draw_risk_logits.append(out["draw_risk_logits"].cpu())
    preds = {
        "p_final": torch.cat(p_final, dim=0),
        "p_from_diff": torch.cat(p_from_diff, dim=0),
        "q_diff": torch.cat(q_diff, dim=0),
        "delta_logits": torch.cat(delta_logits, dim=0),
    }
    if draw_risk_logits:
        preds["draw_risk_logits"] = torch.cat(draw_risk_logits, dim=0)
    return preds


@torch.no_grad()
def evaluate(model: P6ResidualPatchITransformer, data: dict[str, Any], batch_size: int, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
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
    if "draw_risk_logits" in preds:
        goal_diff_metrics.update(compute_draw_risk_metrics(preds["draw_risk_logits"], data["y_draw_risk"]))
    return val_metrics, market_metrics, goal_diff_metrics, preds


def write_val_predictions_csv(path: Path, match_ids: Sequence[str], labels: torch.Tensor, probs: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probs = _normalize_probs(probs.detach().cpu().float())
    labels = labels.detach().cpu().long()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match_id", "y_true", "p_home", "p_draw", "p_away", "pred_class", "correct"])
        writer.writeheader()
        for idx, match_id in enumerate(match_ids):
            pred_class = int(probs[idx].argmax().item())
            y_true = int(labels[idx].item())
            writer.writerow(
                {
                    "match_id": match_id,
                    "y_true": y_true,
                    "p_home": float(probs[idx, 0].item()),
                    "p_draw": float(probs[idx, 1].item()),
                    "p_away": float(probs[idx, 2].item()),
                    "pred_class": pred_class,
                    "correct": int(pred_class == y_true),
                }
            )


def write_market_replication_csv(path: Path, data: dict[str, Any], preds: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p_final = _normalize_probs(preds["p_final"].detach().cpu().float())
    p_market = _normalize_probs(data["p_market"].detach().cpu().float())
    labels = data["y_1x2"].detach().cpu().long()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "match_id",
                "y_true",
                "market_home",
                "market_draw",
                "market_away",
                "final_home",
                "final_draw",
                "final_away",
                "final_minus_market_draw",
            ],
        )
        writer.writeheader()
        for idx, match_id in enumerate(data["match_ids"]):
            writer.writerow(
                {
                    "match_id": match_id,
                    "y_true": int(labels[idx].item()),
                    "market_home": float(p_market[idx, 0].item()),
                    "market_draw": float(p_market[idx, 1].item()),
                    "market_away": float(p_market[idx, 2].item()),
                    "final_home": float(p_final[idx, 0].item()),
                    "final_draw": float(p_final[idx, 1].item()),
                    "final_away": float(p_final[idx, 2].item()),
                    "final_minus_market_draw": float((p_final[idx, 1] - p_market[idx, 1]).item()),
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train P14 market-replication objective probe")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--feature-groups", default="euro,asian,ou")
    parser.add_argument("--epochs", type=int, default=50)
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
    parser.add_argument("--market-loss-weight", type=float, default=0.10)
    parser.add_argument("--checkpoint-selection", default="logloss", choices=("logloss", "balanced"))
    parser.add_argument("--balanced-logloss-ceiling", type=float, default=0.9375)
    parser.add_argument("--diff-loss-weight", type=float, default=0.30)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--delta-l2-weight", type=float, default=0.01)
    parser.add_argument("--draw-risk-loss-weight", type=float, default=0.0)
    parser.add_argument("--scaling", default="robust", choices=SCALING_CHOICES)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    assert_no_test_paths([args.data, args.train_ids, args.val_ids, args.out_dir])
    feature_names = p14_selected_feature_names(args.feature_groups)
    market_type_ids = p14_selected_feature_market_type_ids(args.feature_groups)
    set_deterministic_seed(args.seed)
    prepare_p14_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
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

    train_data = build_p14_dataset(train_rows, args.feature_groups)
    val_data = build_p14_dataset(val_rows, args.feature_groups)
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
        enable_draw_aux_head=False,
        enable_draw_risk_head=args.draw_risk_loss_weight > 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}, features={len(feature_names)}")

    loss_config = {
        "market_loss_weight": float(args.market_loss_weight),
        "label_loss_weight": float(args.label_loss_weight),
        "diff_loss_weight": float(args.diff_loss_weight),
        "consistency_loss_weight": float(args.consistency_loss_weight),
        "delta_l2_weight": float(args.delta_l2_weight),
        "draw_risk_loss_weight": float(args.draw_risk_loss_weight),
    }
    best_logloss = float("inf")
    best_logloss_epoch = 0
    selected_checkpoint_row: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    train_metrics: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_data, optimizer, args.batch_size, device, loss_config)
        val_metrics, market_metrics, goal_diff_metrics, _ = evaluate(model, val_data, args.batch_size, device)
        scheduler.step()
        row = build_epoch_history_row(epoch, train_metrics, val_metrics, market_metrics, goal_diff_metrics)
        history.append(row)
        if val_metrics["logloss"] < best_logloss:
            best_logloss = val_metrics["logloss"]
            best_logloss_epoch = epoch
        if should_replace_checkpoint(
            row,
            selected_checkpoint_row,
            args.checkpoint_selection,
            args.balanced_logloss_ceiling,
        ):
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
            f"draw_top2={val_metrics['draw_top2_recall']:.4f} "
            f"market_kl={market_metrics['market_kl']:.5f}"
        )

    best_checkpoint_path = Path(args.out_dir) / "best_model.pth"
    best_checkpoint = load_best_model_state(model, best_checkpoint_path, device)
    selected_checkpoint_metrics = best_checkpoint.get("selected_checkpoint_metrics") or select_checkpoint_row(
        history,
        args.checkpoint_selection,
        args.balanced_logloss_ceiling,
    )
    final_val, final_market, final_goal_diff, preds = evaluate(model, val_data, args.batch_size, device)
    write_val_predictions_csv(Path(args.out_dir) / "val_predictions.csv", val_data["match_ids"], val_data["y_1x2"], preds["p_final"])
    write_market_replication_csv(Path(args.out_dir) / "val_market_replication.csv", val_data, preds)

    phase_name = (
        "P15 draw-risk auxiliary market-replication probe"
        if float(args.draw_risk_loss_weight) > 0
        else "P14 market-replication objective probe"
    )
    report = {
        "phase": phase_name,
        "run_mode": "smoke" if args.max_samples else "formal",
        "model": "P6ResidualPatchITransformer",
        "diagnostic_note": "Market replication is an objective probe, not value-detection evidence.",
        "config": vars(args),
        "data": {
            "train_rows": len(train_data["X"]),
            "val_rows": len(val_data["X"]),
            "train_unique_match_ids": len(set(train_data["match_ids"])),
            "val_unique_match_ids": len(set(val_data["match_ids"])),
            "feature_names": feature_names,
            "scaler_path": scaler_path,
            "anchor_fallback_count": int(val_data["anchor_fallback_count"]),
            "negative_time_valid_euro_count": int(val_data["negative_time_valid_euro_count"]),
            "skipped_train_rows": int(train_data["skipped_rows"]),
            "skipped_val_rows": int(val_data["skipped_rows"]),
            "train_draw_risk_counts": train_data["draw_risk_counts"],
            "val_draw_risk_counts": val_data["draw_risk_counts"],
        },
        "checkpoint_selection": args.checkpoint_selection,
        "balanced_logloss_ceiling": float(args.balanced_logloss_ceiling),
        "best_epoch": int(selected_checkpoint_metrics["epoch"]),
        "best_val_logloss": float(selected_checkpoint_metrics["val_logloss"]),
        "selected_checkpoint_metrics": selected_checkpoint_metrics,
        "best_logloss_epoch": best_logloss_epoch,
        "best_logloss": best_logloss,
        "final_evaluation_checkpoint": {
            "path": str(best_checkpoint_path),
            "checkpoint_selection": best_checkpoint.get("checkpoint_selection", args.checkpoint_selection),
            "best_epoch": int(best_checkpoint.get("best_epoch", selected_checkpoint_metrics["epoch"])),
            "best_val_logloss": float(best_checkpoint.get("best_val_logloss", selected_checkpoint_metrics["val_logloss"])),
        },
        "anchor_only_baseline": anchor_only,
        "train_metrics": train_metrics,
        "val_metrics": final_val,
        "market_replication_metrics": final_market,
        "goal_diff_metrics": final_goal_diff,
        "history": history,
        "artifacts": {
            "best_model": str(Path(args.out_dir) / "best_model.pth"),
            "val_predictions": str(Path(args.out_dir) / "val_predictions.csv"),
            "val_market_replication": str(Path(args.out_dir) / "val_market_replication.csv"),
            "scaler": scaler_path,
        },
    }
    report_path = Path(args.out_dir) / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (Path(args.out_dir) / "report.md").write_text(
        "\n".join(
            [
                f"# {phase_name}",
                "",
                f"- val_logloss: `{final_val['logloss']}`",
                f"- anchor_only_val_logloss: `{anchor_only['anchor_only_val_logloss']}`",
                f"- draw_top2: `{final_val['draw_top2_recall']}`",
                f"- market_draw_top2: `{final_market['market_draw_top2']}`",
                f"- market_kl: `{final_market['market_kl']}`",
                f"- market_draw_mae: `{final_market['market_draw_mae']}`",
                f"- model_minus_market_p_draw: `{final_market['model_minus_market_p_draw']}`",
                f"- draw_risk_acc: `{final_goal_diff.get('draw_risk_acc', '')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Report saved to {report_path}")
    print(f"Selected checkpoint epoch: {int(selected_checkpoint_metrics['epoch'])}")
    print(f"Selected val_logloss: {float(selected_checkpoint_metrics['val_logloss']):.6f}")
    print(f"Best logloss epoch: {best_logloss_epoch}, best_logloss={best_logloss:.6f}")
    print(f"Market KL: {final_market['market_kl']:.6f}")


def _normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.float().clamp_min(EPS)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)


def _draw_top2_recall(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs = _normalize_probs(probs.detach().cpu().float())
    labels = labels.detach().cpu().long()
    true_draw = labels == 1
    if int(true_draw.sum().item()) == 0:
        return 0.0
    top2 = torch.topk(probs, k=2, dim=-1).indices
    return float((top2[true_draw] == 1).any(dim=-1).float().mean().item())


def _label_counts(labels: Sequence[int], names: Sequence[str]) -> dict[str, int]:
    counts = {name: 0 for name in names}
    for label in labels:
        idx = int(label)
        if 0 <= idx < len(names):
            counts[names[idx]] += 1
    return counts


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


def _tensor_float(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def _safe_scale(values: torch.Tensor) -> float:
    scale = _tensor_float(values.std(unbiased=False))
    return scale if math.isfinite(scale) and scale > 1e-6 else 1.0


if __name__ == "__main__":
    main()
