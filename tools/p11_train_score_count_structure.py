"""Train P11 minimal score-count structure probes.

P11 keeps the P6 residual Patch-iTransformer backbone and final 1X2 contract.
It optionally adds a two-scalar independent Poisson goal-count head and a
detached score-derived 1X2 consistency loss. It never reads the test split and
never performs posthoc validation fitting.
"""

from __future__ import annotations

import argparse
import csv
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
from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
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
from tools.p3b_train_patch_itransformer import bucketize_sample_v2_for_training


EPS = 1e-8
FEATURE_MARKET_TYPES = {name: market for name, market in FEATURE_DEFS}
POISSON_MIN_RATE = 0.05
POISSON_MAX_RATE = 5.50


def p11_selected_feature_names(feature_groups: str) -> list[str]:
    return selected_feature_names(feature_groups)


def p11_selected_feature_market_type_ids(feature_groups: str) -> list[int]:
    return [int(FEATURE_MARKET_TYPES[name]) for name in p11_selected_feature_names(feature_groups)]


def prepare_p11_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing_outputs = [
        Path(out_dir) / "report.json",
        Path(out_dir) / "best_model.pth",
        Path(out_dir) / "val_predictions.csv",
        Path(out_dir) / "val_goal_diff_predictions.csv",
        Path(out_dir) / "val_score_predictions.csv",
        Path(out_dir) / "score_label_audit.json",
        Path(out_dir) / "scaler.json",
    ]
    if not allow_overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(f"Refusing to overwrite existing P11 outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _parse_nonnegative_int(value: Any) -> int | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float) or as_float < 0 or not as_float.is_integer():
        return None
    return int(as_float)


def score_label_audit(rows: list[dict[str, Any]], min_coverage: float = 0.999) -> dict[str, Any]:
    eligible = 0
    valid = 0
    invalid_reasons: dict[str, int] = {}
    for row in rows:
        label = row.get("label", {}) or {}
        if label.get("euro_result") not in EURO_MAP:
            continue
        eligible += 1
        home = _parse_nonnegative_int(label.get("home_goals"))
        away = _parse_nonnegative_int(label.get("away_goals"))
        if home is None or away is None:
            invalid_reasons["missing_or_invalid_score"] = invalid_reasons.get("missing_or_invalid_score", 0) + 1
            continue
        valid += 1
    coverage = valid / max(eligible, 1)
    return {
        "eligible_1x2_rows": eligible,
        "valid_score_label_rows": valid,
        "invalid_score_label_rows": eligible - valid,
        "score_label_coverage": coverage,
        "min_required_coverage": float(min_coverage),
        "passes_min_coverage": coverage >= float(min_coverage),
        "invalid_reason_counts": invalid_reasons,
    }


def build_p11_dataset(rows: list[dict[str, Any]], feature_groups: str) -> dict[str, Any]:
    names = selected_feature_names(feature_groups)
    indices = [FEATURE_INDEX[name] for name in names]
    X, y_1x2, y_goal_diff, y_home, y_away, anchors, match_ids = [], [], [], [], [], [], []
    fallback_count = 0
    negative_time_valid_euro_count = 0
    skipped_rows = 0

    for row_idx, row in enumerate(rows):
        label = row.get("label", {}) or {}
        result = label.get("euro_result")
        if result not in EURO_MAP:
            skipped_rows += 1
            continue
        home_goals = _parse_nonnegative_int(label.get("home_goals"))
        away_goals = _parse_nonnegative_int(label.get("away_goals"))
        if home_goals is None or away_goals is None:
            skipped_rows += 1
            continue

        timeline = row.get("raw_timeline", []) or row.get("odds_timeline", [])
        xb_full = bucketize_sample_v2_for_training(timeline, clean_missing_markets=True)
        anchor, anchor_diag = extract_euro_anchor_probs(row)

        X.append(xb_full[:, indices, :])
        y_1x2.append(EURO_MAP[result])
        y_goal_diff.append(goal_diff_to_bucket(home_goals - away_goals))
        y_home.append(float(home_goals))
        y_away.append(float(away_goals))
        anchors.append(anchor)
        match_ids.append(str(row.get("match_id", row_idx)))
        fallback_count += int(anchor_diag["used_fallback"])
        negative_time_valid_euro_count += int(anchor_diag["negative_time_valid_euro_count"])

    if not X:
        raise ValueError("No P11 labelled rows available")
    return {
        "X": torch.stack(X),
        "y_1x2": torch.tensor(y_1x2, dtype=torch.long),
        "y_goal_diff": torch.tensor(y_goal_diff, dtype=torch.long),
        "y_home_goals": torch.tensor(y_home, dtype=torch.float32),
        "y_away_goals": torch.tensor(y_away, dtype=torch.float32),
        "p_euro_anchor": torch.stack(anchors),
        "match_ids": match_ids,
        "feature_names": names,
        "score_label_audit": score_label_audit(rows),
        "anchor_fallback_count": fallback_count,
        "negative_time_valid_euro_count": negative_time_valid_euro_count,
        "skipped_rows": skipped_rows,
    }


def poisson_count_nll(lambdas: torch.Tensor, home_goals: torch.Tensor, away_goals: torch.Tensor) -> torch.Tensor:
    if lambdas.shape[-1] != 2:
        raise ValueError("goal-count lambdas must have shape [B, 2]")
    goals = torch.stack([home_goals.float(), away_goals.float()], dim=-1)
    lambdas = lambdas.float().clamp_min(EPS)
    nll = lambdas - goals * torch.log(lambdas) + torch.lgamma(goals + 1.0)
    return 0.5 * nll.sum(dim=-1).mean()


def constant_mean_rate_count_nll(
    train_home_goals: torch.Tensor,
    train_away_goals: torch.Tensor,
    val_home_goals: torch.Tensor,
    val_away_goals: torch.Tensor,
) -> float:
    mean_home = train_home_goals.float().mean().clamp_min(EPS)
    mean_away = train_away_goals.float().mean().clamp_min(EPS)
    lambdas = torch.stack([mean_home, mean_away]).view(1, 2).expand(int(val_home_goals.numel()), 2)
    return float(poisson_count_nll(lambdas, val_home_goals.float(), val_away_goals.float()).item())


def constant_class_prior_logloss(train_labels: torch.Tensor, val_labels: torch.Tensor) -> float:
    counts = torch.bincount(train_labels.detach().cpu().long(), minlength=3).float()
    prior = counts.clamp_min(EPS) / counts.sum().clamp_min(EPS)
    probs = prior.view(1, 3).expand(int(val_labels.numel()), 3)
    return logloss_from_probs(probs, val_labels.detach().cpu().long())


def score_rates_to_1x2(lambdas: torch.Tensor, max_goals: int = 10) -> dict[str, torch.Tensor]:
    if lambdas.shape[-1] != 2:
        raise ValueError("goal-count lambdas must have shape [B, 2]")
    if int(max_goals) < 1:
        raise ValueError("max_goals must be >= 1")
    lambdas = lambdas.float().clamp_min(EPS)
    goals = torch.arange(int(max_goals) + 1, device=lambdas.device, dtype=lambdas.dtype)
    log_factorial = torch.lgamma(goals + 1.0)
    log_lambdas = torch.log(lambdas)

    home_log_pmf = goals.view(1, -1) * log_lambdas[:, 0:1] - lambdas[:, 0:1] - log_factorial.view(1, -1)
    away_log_pmf = goals.view(1, -1) * log_lambdas[:, 1:2] - lambdas[:, 1:2] - log_factorial.view(1, -1)
    home_pmf = torch.exp(home_log_pmf)
    away_pmf = torch.exp(away_log_pmf)
    score_probs = home_pmf.unsqueeze(2) * away_pmf.unsqueeze(1)

    home_idx = goals.view(-1, 1)
    away_idx = goals.view(1, -1)
    home_mask = home_idx > away_idx
    draw_mask = home_idx == away_idx
    away_mask = home_idx < away_idx
    folded = torch.stack(
        [
            score_probs[:, home_mask].sum(dim=-1),
            score_probs[:, draw_mask].sum(dim=-1),
            score_probs[:, away_mask].sum(dim=-1),
        ],
        dim=-1,
    )
    folded_mass = folded.sum(dim=-1)
    tail_mass = (1.0 - folded_mass).clamp_min(0.0)
    p_score_1x2 = folded / folded_mass.unsqueeze(-1).clamp_min(EPS)
    return {
        "p_score_1x2": p_score_1x2,
        "tail_mass": tail_mass,
        "folded_mass": folded_mass,
        "score_probs": score_probs,
        "home_pmf": home_pmf,
        "away_pmf": away_pmf,
    }


def compute_p11_losses(
    out: dict[str, torch.Tensor],
    p_euro_anchor: torch.Tensor,
    y_1x2: torch.Tensor,
    y_goal_diff: torch.Tensor,
    y_home_goals: torch.Tensor,
    y_away_goals: torch.Tensor,
    diff_loss_weight: float,
    consistency_loss_weight: float,
    delta_l2_weight: float,
    count_loss_weight: float = 0.0,
    score_kl_weight: float = 0.0,
    close_game_kl_only: bool = False,
    score_grid_max_goals: int = 10,
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
    zero = losses["loss"].new_tensor(0.0)
    needs_score_head = float(count_loss_weight) > 0 or float(score_kl_weight) > 0
    if not needs_score_head and "goal_count_lambdas" not in out:
        losses.update(
            {
                "L_count": zero,
                "L_score_1x2_kl": zero,
                "score_kl_n": zero,
                "score_tail_mass_mean": zero,
                "score_tail_mass_p95": zero,
            }
        )
        return losses
    if "goal_count_lambdas" not in out:
        raise ValueError("goal_count_lambdas are required for P11 count or score KL losses")

    lambdas = out["goal_count_lambdas"]
    folded = score_rates_to_1x2(lambdas, max_goals=score_grid_max_goals)
    p_score_1x2 = folded["p_score_1x2"].clamp_min(EPS)
    l_count = poisson_count_nll(lambdas, y_home_goals, y_away_goals) if float(count_loss_weight) > 0 else zero

    if float(score_kl_weight) > 0:
        if close_game_kl_only:
            kl_mask = (y_home_goals.long() - y_away_goals.long()).abs() <= 1
        else:
            kl_mask = torch.ones_like(y_1x2, dtype=torch.bool)
        if int(kl_mask.sum().item()) > 0:
            l_score_kl = F.kl_div(
                F.log_softmax(losses["final_logits"][kl_mask], dim=-1),
                p_score_1x2[kl_mask].detach(),
                reduction="batchmean",
            )
            score_kl_n = losses["loss"].new_tensor(float(kl_mask.sum().item()))
        else:
            l_score_kl = zero
            score_kl_n = zero
    else:
        l_score_kl = zero
        score_kl_n = zero

    losses["loss"] = losses["loss"] + float(count_loss_weight) * l_count + float(score_kl_weight) * l_score_kl
    tail_mass = folded["tail_mass"]
    losses.update(
        {
            "L_count": l_count,
            "L_score_1x2_kl": l_score_kl,
            "score_kl_n": score_kl_n,
            "p_score_1x2": p_score_1x2,
            "score_tail_mass_mean": tail_mass.mean(),
            "score_tail_mass_p95": torch.quantile(tail_mass.detach(), 0.95),
        }
    )
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


def _draw_rank_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probs = probs.detach().cpu().float().clamp_min(EPS)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    labels = labels.detach().cpu().long()
    true_draw = labels == 1
    top2 = torch.topk(probs, k=2, dim=-1).indices
    draw_margin = probs.max(dim=-1).values - probs[:, 1]
    return {
        "draw_top2": float((top2[true_draw] == 1).any(dim=-1).float().mean().item()) if int(true_draw.sum().item()) else 0.0,
        "mean_draw_margin_to_top": float(draw_margin.mean().item()) if probs.numel() else 0.0,
    }


def _classwise_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probs = probs.detach().cpu().float().clamp_min(EPS)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    labels = labels.detach().cpu().long()
    preds = probs.argmax(dim=-1)
    out: dict[str, Any] = {}
    for idx, name in [(0, "home"), (1, "draw"), (2, "away")]:
        mask = labels == idx
        pred_mask = preds == idx
        true_positive = pred_mask & mask
        out[f"{name}_n"] = int(mask.sum().item())
        out[f"{name}_nll"] = float((-torch.log(probs[mask, idx].clamp_min(EPS))).mean().item()) if int(mask.sum().item()) else 0.0
        out[f"{name}_recall"] = float(true_positive.sum().item() / mask.sum().item()) if int(mask.sum().item()) else 0.0
        out[f"{name}_precision"] = float(true_positive.sum().item() / pred_mask.sum().item()) if int(pred_mask.sum().item()) else 0.0
    return out


def _score_topk_accuracy(score_probs: torch.Tensor, home_goals: torch.Tensor, away_goals: torch.Tensor, k: int) -> float:
    max_goal = score_probs.shape[-1] - 1
    flat = score_probs.detach().cpu().float().reshape(score_probs.shape[0], -1)
    topk = torch.topk(flat, k=min(int(k), flat.shape[-1]), dim=-1).indices
    home = home_goals.detach().cpu().long()
    away = away_goals.detach().cpu().long()
    valid = (home >= 0) & (away >= 0) & (home <= max_goal) & (away <= max_goal)
    target = home * (max_goal + 1) + away
    hit = (topk == target.view(-1, 1)).any(dim=-1) & valid
    return float(hit.float().mean().item()) if hit.numel() else 0.0


def compute_score_count_metrics(
    lambdas: torch.Tensor,
    folded: dict[str, torch.Tensor],
    home_goals: torch.Tensor,
    away_goals: torch.Tensor,
    constant_count_nll: float,
) -> dict[str, Any]:
    lambdas = lambdas.detach().cpu().float()
    home = home_goals.detach().cpu().float()
    away = away_goals.detach().cpu().float()
    score_probs = folded["score_probs"].detach().cpu().float()
    tail_mass = folded["tail_mass"].detach().cpu().float()
    home_pmf = folded["home_pmf"].detach().cpu().float()
    away_pmf = folded["away_pmf"].detach().cpu().float()
    pred_home = lambdas[:, 0]
    pred_away = lambdas[:, 1]
    pred_total = pred_home + pred_away
    true_total = home + away
    pred_diff = pred_home - pred_away
    true_diff = home - away
    btts_prob = (1.0 - home_pmf[:, 0]) * (1.0 - away_pmf[:, 0])
    btts_pred = btts_prob >= 0.5
    btts_true = (home > 0) & (away > 0)
    return {
        "available": True,
        "val_count_nll": float(poisson_count_nll(lambdas, home, away).item()),
        "constant_train_mean_rate_count_nll": float(constant_count_nll),
        "home_goal_mae": float((pred_home - home).abs().mean().item()),
        "away_goal_mae": float((pred_away - away).abs().mean().item()),
        "total_goal_mae": float((pred_total - true_total).abs().mean().item()),
        "goal_diff_mae": float((pred_diff - true_diff).abs().mean().item()),
        "exact_score_top1": _score_topk_accuracy(score_probs, home, away, 1),
        "exact_score_top3": _score_topk_accuracy(score_probs, home, away, 3),
        "btts_accuracy": float((btts_pred == btts_true).float().mean().item()) if btts_true.numel() else 0.0,
        "total_goals_mae": float((pred_total - true_total).abs().mean().item()),
        "tail_mass_mean": float(tail_mass.mean().item()),
        "tail_mass_p95": float(torch.quantile(tail_mass, 0.95).item()),
        "lambda_min_saturation_rate": float((lambdas <= POISSON_MIN_RATE * 1.01).float().mean().item()),
        "lambda_max_saturation_rate": float((lambdas >= POISSON_MAX_RATE * 0.99).float().mean().item()),
    }


def compute_score_derived_1x2_metrics(
    p_score_1x2: torch.Tensor,
    p_final: torch.Tensor,
    labels: torch.Tensor,
    constant_class_prior_ll: float,
) -> dict[str, Any]:
    p_score = p_score_1x2.detach().cpu().float().clamp_min(EPS)
    p_score = p_score / p_score.sum(dim=-1, keepdim=True).clamp_min(EPS)
    p_final = p_final.detach().cpu().float().clamp_min(EPS)
    p_final = p_final / p_final.sum(dim=-1, keepdim=True).clamp_min(EPS)
    labels = labels.detach().cpu().long()
    true_draw = labels == 1
    top2 = torch.topk(p_score, k=2, dim=-1).indices
    margin = p_score.max(dim=-1).values - p_score[:, 1]
    return {
        "available": True,
        "score_derived_1x2_logloss": logloss_from_probs(p_score, labels),
        "constant_class_prior_1x2_logloss": float(constant_class_prior_ll),
        "score_derived_draw_nll": float((-torch.log(p_score[true_draw, 1].clamp_min(EPS))).mean().item()) if int(true_draw.sum().item()) else 0.0,
        "score_derived_draw_top2": float((top2[true_draw] == 1).any(dim=-1).float().mean().item()) if int(true_draw.sum().item()) else 0.0,
        "score_derived_mean_p_draw_true_draw": float(p_score[true_draw, 1].mean().item()) if int(true_draw.sum().item()) else 0.0,
        "score_derived_draw_margin_to_top": float(margin.mean().item()) if p_score.numel() else 0.0,
        "score_vs_final_draw_corr": _pearson(p_score[:, 1], p_final[:, 1]),
        "score_vs_final_draw_mae": float((p_score[:, 1] - p_final[:, 1]).abs().mean().item()),
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
    count_loss_weight: float,
    score_kl_weight: float,
    close_game_kl_only: bool,
    score_grid_max_goals: int,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "L_1x2": 0.0,
        "L_diff": 0.0,
        "L_consistency": 0.0,
        "L_delta": 0.0,
        "L_count": 0.0,
        "L_score_1x2_kl": 0.0,
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
        hgb = data["y_home_goals"][idx].to(device)
        agb = data["y_away_goals"][idx].to(device)
        ab = data["p_euro_anchor"][idx].to(device)
        out = model(xb)
        losses = compute_p11_losses(
            out,
            ab,
            yb,
            ygb,
            hgb,
            agb,
            diff_loss_weight=diff_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            delta_l2_weight=delta_l2_weight,
            count_loss_weight=count_loss_weight,
            score_kl_weight=score_kl_weight,
            close_game_kl_only=close_game_kl_only,
            score_grid_max_goals=score_grid_max_goals,
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
def predict_outputs(model: P6ResidualPatchITransformer, data: dict[str, Any], batch_size: int, device: torch.device, score_grid_max_goals: int = 10) -> dict[str, torch.Tensor]:
    model.eval()
    p_final, p_from_diff, q_diff, delta_logits = [], [], [], []
    lambdas, p_score_1x2, tail_mass, score_probs, home_pmf, away_pmf = [], [], [], [], [], []
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
        if "goal_count_lambdas" in out:
            folded = score_rates_to_1x2(out["goal_count_lambdas"], max_goals=score_grid_max_goals)
            lambdas.append(out["goal_count_lambdas"].cpu())
            p_score_1x2.append(folded["p_score_1x2"].cpu())
            tail_mass.append(folded["tail_mass"].cpu())
            score_probs.append(folded["score_probs"].cpu())
            home_pmf.append(folded["home_pmf"].cpu())
            away_pmf.append(folded["away_pmf"].cpu())
    preds = {
        "p_final": torch.cat(p_final, dim=0),
        "p_from_diff": torch.cat(p_from_diff, dim=0),
        "q_diff": torch.cat(q_diff, dim=0),
        "delta_logits": torch.cat(delta_logits, dim=0),
    }
    if lambdas:
        preds.update(
            {
                "goal_count_lambdas": torch.cat(lambdas, dim=0),
                "p_score_1x2": torch.cat(p_score_1x2, dim=0),
                "tail_mass": torch.cat(tail_mass, dim=0),
                "score_probs": torch.cat(score_probs, dim=0),
                "home_pmf": torch.cat(home_pmf, dim=0),
                "away_pmf": torch.cat(away_pmf, dim=0),
            }
        )
    return preds


@torch.no_grad()
def evaluate(
    model: P6ResidualPatchITransformer,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
    score_grid_max_goals: int,
    constant_count_nll: float | None = None,
    constant_prior_ll: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    preds = predict_outputs(model, data, batch_size, device, score_grid_max_goals=score_grid_max_goals)
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
    val_metrics.update(_classwise_metrics(p_final, labels))
    if "draw_nll" not in val_metrics:
        val_metrics["draw_nll"] = val_metrics.get("draw_class_nll", 0.0)

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
    score_metrics: dict[str, Any] = {"available": False}
    score_derived: dict[str, Any] = {"available": False}
    if "goal_count_lambdas" in preds:
        folded = {
            "p_score_1x2": preds["p_score_1x2"],
            "tail_mass": preds["tail_mass"],
            "score_probs": preds["score_probs"],
            "home_pmf": preds["home_pmf"],
            "away_pmf": preds["away_pmf"],
        }
        score_metrics = compute_score_count_metrics(
            preds["goal_count_lambdas"],
            folded,
            data["y_home_goals"],
            data["y_away_goals"],
            float(constant_count_nll or 0.0),
        )
        score_derived = compute_score_derived_1x2_metrics(
            preds["p_score_1x2"],
            p_final,
            labels,
            float(constant_prior_ll or 0.0),
        )
    return val_metrics, goal_diff_metrics, score_metrics, score_derived, preds


def write_score_predictions_csv(path: Path, data: dict[str, Any], preds: dict[str, torch.Tensor]) -> None:
    if "goal_count_lambdas" not in preds:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lambdas = preds["goal_count_lambdas"]
    p_score = preds["p_score_1x2"]
    tail = preds["tail_mass"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "match_id",
                "home_goals",
                "away_goals",
                "lambda_home",
                "lambda_away",
                "score_p_home",
                "score_p_draw",
                "score_p_away",
                "tail_mass",
            ],
        )
        writer.writeheader()
        for idx, match_id in enumerate(data["match_ids"]):
            writer.writerow(
                {
                    "match_id": match_id,
                    "home_goals": int(data["y_home_goals"][idx].item()),
                    "away_goals": int(data["y_away_goals"][idx].item()),
                    "lambda_home": float(lambdas[idx, 0].item()),
                    "lambda_away": float(lambdas[idx, 1].item()),
                    "score_p_home": float(p_score[idx, 0].item()),
                    "score_p_draw": float(p_score[idx, 1].item()),
                    "score_p_away": float(p_score[idx, 2].item()),
                    "tail_mass": float(tail[idx].item()),
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train P11 score-count structure probes")
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
    parser.add_argument("--enable-goal-count-head", action="store_true")
    parser.add_argument("--count-loss-weight", type=float, default=0.0)
    parser.add_argument("--score-kl-weight", type=float, default=0.0)
    parser.add_argument("--close-game-kl-only", action="store_true")
    parser.add_argument("--score-grid-max-goals", type=int, default=10)
    parser.add_argument("--scaling", default="robust", choices=SCALING_CHOICES)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--variant", default="manual")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if "test" in Path(args.train_ids).name.lower() or "test" in Path(args.val_ids).name.lower():
        raise SystemExit("P11 refuses any test split path")
    set_deterministic_seed(args.seed)
    prepare_p11_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feature_names = p11_selected_feature_names(args.feature_groups)
    market_type_ids = p11_selected_feature_market_type_ids(args.feature_groups)
    train_rows = load_rows_for_ids(args.data, load_split_ids(args.train_ids))
    val_rows = load_rows_for_ids(args.data, load_split_ids(args.val_ids))
    if args.max_samples > 0:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: min(args.max_samples, len(val_rows))]
    print(f"Rows: train={len(train_rows)} val={len(val_rows)}")

    train_label_audit = score_label_audit(train_rows)
    val_label_audit = score_label_audit(val_rows)
    write_json(Path(args.out_dir) / "score_label_audit.json", {"train": train_label_audit, "val": val_label_audit})
    if not train_label_audit["passes_min_coverage"] or not val_label_audit["passes_min_coverage"]:
        blocked = {
            "phase": "P11 minimal score-count structure probe",
            "variant": args.variant,
            "verdict": "P11_BLOCKED_BY_LABEL_AUDIT",
            "score_label_audit": {"train": train_label_audit, "val": val_label_audit},
            "hard_fail_reasons": ["P11_BLOCKED_BY_LABEL_AUDIT"],
        }
        write_json(Path(args.out_dir) / "report.json", blocked)
        raise SystemExit("P11_BLOCKED_BY_LABEL_AUDIT")

    train_data = build_p11_dataset(train_rows, args.feature_groups)
    val_data = build_p11_dataset(val_rows, args.feature_groups)
    if train_data["feature_names"] != feature_names or val_data["feature_names"] != feature_names:
        raise ValueError("P11 selected feature names drifted from dataset construction")

    scaler = fit_p4_feature_scaler(train_data["X"], feature_names, args.scaling)
    scaler_path = save_feature_scaler(scaler, args.out_dir)
    train_data["X"] = apply_p4_feature_scaler(train_data["X"], scaler)
    val_data["X"] = apply_p4_feature_scaler(val_data["X"], scaler)
    anchor_only = compute_anchor_only_metrics(val_data["p_euro_anchor"], val_data["y_1x2"])
    constant_count_nll = constant_mean_rate_count_nll(
        train_data["y_home_goals"],
        train_data["y_away_goals"],
        val_data["y_home_goals"],
        val_data["y_away_goals"],
    )
    constant_prior_ll = constant_class_prior_logloss(train_data["y_1x2"], val_data["y_1x2"])

    score_head_enabled = bool(args.enable_goal_count_head or args.count_loss_weight > 0 or args.score_kl_weight > 0)
    model = P6ResidualPatchITransformer(
        n_features=len(feature_names),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        market_type_ids=market_type_ids,
        enable_goal_count_head=score_head_enabled,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}, features={len(feature_names)}, score_head_enabled={score_head_enabled}")

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
            args.diff_loss_weight,
            args.consistency_loss_weight,
            args.delta_l2_weight,
            args.count_loss_weight,
            args.score_kl_weight,
            args.close_game_kl_only,
            args.score_grid_max_goals,
        )
        val_metrics, goal_diff_metrics, score_metrics, score_derived_metrics, _ = evaluate(
            model,
            val_data,
            args.batch_size * 2,
            device,
            score_grid_max_goals=args.score_grid_max_goals,
            constant_count_nll=constant_count_nll,
            constant_prior_ll=constant_prior_ll,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "val_logloss": val_metrics["logloss"],
            "val_ece": val_metrics["ece"],
            "val_accuracy": val_metrics["accuracy"],
            "val_draw_class_nll": val_metrics["draw_class_nll"],
            "val_draw_top2_recall": val_metrics["draw_top2_recall"],
            "val_mean_p_draw_on_true_draw": val_metrics["mean_p_draw_on_true_draw"],
            "val_mean_draw_margin_to_top": val_metrics["mean_draw_margin_to_top"],
            "val_count_nll": score_metrics.get("val_count_nll", 0.0),
            "val_score_derived_1x2_logloss": score_derived_metrics.get("score_derived_1x2_logloss", 0.0),
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
                    "enable_goal_count_head": score_head_enabled,
                },
                Path(args.out_dir) / "best_model.pth",
            )
        print(
            f"Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
            f"val_logloss={val_metrics['logloss']:.4f} "
            f"draw_top2={val_metrics['draw_top2_recall']:.4f}"
            + (f" count_nll={score_metrics.get('val_count_nll', 0.0):.4f}" if score_head_enabled else "")
            + (" *" if is_best else "")
        )

    ckpt_path = Path(args.out_dir) / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_val, final_goal_diff, final_score_metrics, final_score_derived, preds = evaluate(
        model,
        val_data,
        args.batch_size * 2,
        device,
        score_grid_max_goals=args.score_grid_max_goals,
        constant_count_nll=constant_count_nll,
        constant_prior_ll=constant_prior_ll,
    )
    write_val_predictions_csv(str(Path(args.out_dir) / "val_predictions.csv"), val_data["match_ids"], val_data["y_1x2"], preds["p_final"])
    write_val_goal_diff_predictions_csv(
        str(Path(args.out_dir) / "val_goal_diff_predictions.csv"),
        val_data["match_ids"],
        val_data["y_goal_diff"],
        preds["q_diff"],
        preds["p_from_diff"],
    )
    write_score_predictions_csv(Path(args.out_dir) / "val_score_predictions.csv", val_data, preds)
    write_json(Path(args.out_dir) / "val_metrics.json", final_val)
    write_json(Path(args.out_dir) / "goal_diff_metrics.json", final_goal_diff)
    write_json(Path(args.out_dir) / "score_count_metrics.json", final_score_metrics)
    write_json(Path(args.out_dir) / "score_derived_1x2_metrics.json", final_score_derived)

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
        "score_label_audit": {"train": train_label_audit, "val": val_label_audit},
    }
    training_manifest = {
        "phase": "P11",
        "variant": args.variant,
        "seed": args.seed,
        "score_head_enabled": score_head_enabled,
        "count_loss_weight": float(args.count_loss_weight),
        "score_kl_weight": float(args.score_kl_weight),
        "close_game_kl_only": bool(args.close_game_kl_only),
        "score_grid_max_goals": int(args.score_grid_max_goals),
        "p6_default_behavior_changed": False,
        "forbidden_large_score_grid_head": False,
        "forbidden_draw_specific_loss": False,
    }
    write_json(Path(args.out_dir) / "training_manifest.json", training_manifest)
    report = {
        "phase": "P11 minimal score-count structure probe",
        "run_mode": "smoke" if args.max_samples else "formal",
        "model": "P6ResidualPatchITransformer",
        "config": vars(args),
        "data": data_summary,
        "training_manifest": training_manifest,
        "score_baselines": {
            "constant_train_mean_rate_count_nll": constant_count_nll,
            "constant_class_prior_1x2_logloss": constant_prior_ll,
        },
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "anchor_only_baseline": anchor_only,
        "train_metrics": history[-1] if history else {},
        "val_metrics": final_val,
        "goal_diff_metrics": final_goal_diff,
        "score_count_metrics": final_score_metrics,
        "score_derived_1x2_metrics": final_score_derived,
        "history": history,
        "warnings": [],
        "params": n_params,
        "scaler": {"path": scaler_path, **scaler},
        "run_path": args.out_dir,
        "artifacts": {
            "best_model": str(ckpt_path),
            "val_predictions": str(Path(args.out_dir) / "val_predictions.csv"),
            "val_goal_diff_predictions": str(Path(args.out_dir) / "val_goal_diff_predictions.csv"),
            "val_score_predictions": str(Path(args.out_dir) / "val_score_predictions.csv") if score_head_enabled else "",
            "score_label_audit": str(Path(args.out_dir) / "score_label_audit.json"),
        },
    }
    write_json(Path(args.out_dir) / "report.json", report)
    (Path(args.out_dir) / "report.md").write_text(
        "\n".join(
            [
                "# P11 Run Report",
                "",
                f"- variant: `{args.variant}`",
                f"- seed: `{args.seed}`",
                f"- score_head_enabled: `{score_head_enabled}`",
                f"- val_logloss: `{final_val['logloss']}`",
                f"- draw_top2: `{final_val['draw_top2_recall']}`",
                f"- mean_p_draw_true_draw: `{final_val['mean_p_draw_on_true_draw']}`",
                f"- val_count_nll: `{final_score_metrics.get('val_count_nll', '')}`",
                f"- score_derived_draw_top2: `{final_score_derived.get('score_derived_draw_top2', '')}`",
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
