"""Train P12 explicit score-prior residual probes."""

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
from model.p6_residual_patch_itransformer import GoalCountRateHead
from model.p12_score_prior_residual import P12ScorePriorResidualITransformer
from tools.p4_goal_diff_utils import compute_anchor_only_metrics
from tools.p4_train_residual_goal_diff import (
    SCALING_CHOICES,
    apply_p4_feature_scaler,
    fit_p4_feature_scaler,
    load_rows_for_ids,
    load_split_ids,
    save_feature_scaler,
    set_deterministic_seed,
    write_val_predictions_csv,
)
from tools.p6_train_residual_patch_itransformer import _brier_from_probs, _multiclass_ece, compute_draw_auxiliary_metrics
from tools.p11_train_score_count_structure import (
    EPS,
    build_p11_dataset,
    compute_score_count_metrics,
    compute_score_derived_1x2_metrics,
    constant_class_prior_logloss,
    constant_mean_rate_count_nll,
    p11_selected_feature_market_type_ids,
    p11_selected_feature_names,
    poisson_count_nll,
    score_label_audit,
    score_rates_to_1x2,
    write_json,
)


PRIOR_MODES = ("score", "blend")


def initialize_goal_count_head_from_means(head: GoalCountRateHead, mean_home_goals: float, mean_away_goals: float) -> None:
    head.initialize_from_means(mean_home_goals, mean_away_goals)


def residual_centered_l2(residual_delta: torch.Tensor) -> torch.Tensor:
    centered = residual_delta - residual_delta.mean(dim=-1, keepdim=True)
    return centered.pow(2).mean()


def geometric_blend_prior(p_anchor: torch.Tensor, p_score: torch.Tensor, alpha: float) -> torch.Tensor:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    anchor = p_anchor.float().clamp_min(EPS)
    score = p_score.float().clamp_min(EPS)
    blended = (anchor ** float(alpha)) * (score ** (1.0 - float(alpha)))
    return blended / blended.sum(dim=-1, keepdim=True).clamp_min(EPS)


def compute_p12_losses(
    out: dict[str, torch.Tensor],
    y_1x2: torch.Tensor,
    y_home_goals: torch.Tensor,
    y_away_goals: torch.Tensor,
    prior_mode: str,
    blend_alpha: float = 1.0,
    count_loss_weight: float = 0.10,
    residual_l2_weight: float = 0.0,
    anchor_ce_weight: float = 0.0,
    score_grid_max_goals: int = 10,
) -> dict[str, torch.Tensor]:
    if prior_mode not in PRIOR_MODES:
        raise ValueError(f"Unknown P12 prior_mode: {prior_mode}")
    if "goal_count_lambdas" not in out or "residual_delta" not in out:
        raise ValueError("P12 losses require goal_count_lambdas and residual_delta")

    folded = score_rates_to_1x2(out["goal_count_lambdas"], max_goals=score_grid_max_goals)
    p_score = folded["p_score_1x2"].clamp_min(EPS)
    p_score = p_score / p_score.sum(dim=-1, keepdim=True).clamp_min(EPS)
    if prior_mode == "score":
        p_prior = p_score.detach()
    else:
        if "anchor_logits" not in out:
            raise ValueError("anchor_logits are required for blend prior mode")
        p_anchor = F.softmax(out["anchor_logits"], dim=-1)
        p_prior = geometric_blend_prior(p_anchor.detach(), p_score.detach(), alpha=blend_alpha)

    p_prior = p_prior.clamp_min(EPS)
    p_prior = p_prior / p_prior.sum(dim=-1, keepdim=True).clamp_min(EPS)
    final_logits = torch.log(p_prior) + out["residual_delta"]
    p_final = F.softmax(final_logits, dim=-1)
    l_final = F.cross_entropy(final_logits, y_1x2)
    l_count = poisson_count_nll(out["goal_count_lambdas"], y_home_goals, y_away_goals)
    l_residual = residual_centered_l2(out["residual_delta"])
    if float(anchor_ce_weight) > 0:
        if "anchor_logits" not in out:
            raise ValueError("anchor_logits are required when anchor_ce_weight > 0")
        l_anchor = F.cross_entropy(out["anchor_logits"], y_1x2)
    else:
        l_anchor = final_logits.new_tensor(0.0)

    loss = (
        l_final
        + float(count_loss_weight) * l_count
        + float(residual_l2_weight) * l_residual
        + float(anchor_ce_weight) * l_anchor
    )
    return {
        "loss": loss,
        "L_final": l_final,
        "L_count": l_count,
        "L_delta_centered_l2": l_residual,
        "L_anchor": l_anchor,
        "p_final": p_final,
        "p_prior": p_prior,
        "p_score_1x2": p_score,
        "final_logits": final_logits,
        "tail_mass": folded["tail_mass"],
        "score_probs": folded["score_probs"],
        "home_pmf": folded["home_pmf"],
        "away_pmf": folded["away_pmf"],
    }


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


def _draw_metric_bundle(prefix: str, probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probs = probs.detach().cpu().float().clamp_min(EPS)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    labels = labels.detach().cpu().long()
    true_draw = labels == 1
    top2 = torch.topk(probs, k=2, dim=-1).indices
    margin = probs.max(dim=-1).values - probs[:, 1]
    out = {
        f"{prefix}_logloss": logloss_from_probs(probs, labels),
        f"{prefix}_draw_nll": float((-torch.log(probs[true_draw, 1].clamp_min(EPS))).mean().item()) if int(true_draw.sum().item()) else 0.0,
        f"{prefix}_draw_top2": float((top2[true_draw] == 1).any(dim=-1).float().mean().item()) if int(true_draw.sum().item()) else 0.0,
        f"{prefix}_mean_p_draw_true_draw": float(probs[true_draw, 1].mean().item()) if int(true_draw.sum().item()) else 0.0,
        f"{prefix}_draw_margin_to_top": float(margin.mean().item()) if probs.numel() else 0.0,
    }
    if prefix in {"anchor", "blend_prior", "p_prior"}:
        out[f"{prefix}_ece"] = _multiclass_ece(probs, labels, 10)
    return out


def _classwise_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probs = probs.detach().cpu().float().clamp_min(EPS)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    labels = labels.detach().cpu().long()
    preds = probs.argmax(dim=-1)
    out: dict[str, Any] = {}
    for idx, name in [(0, "home"), (1, "draw"), (2, "away")]:
        mask = labels == idx
        pred_mask = preds == idx
        tp = mask & pred_mask
        out[f"{name}_nll"] = float((-torch.log(probs[mask, idx].clamp_min(EPS))).mean().item()) if int(mask.sum().item()) else 0.0
        out[f"{name}_recall"] = float(tp.sum().item() / mask.sum().item()) if int(mask.sum().item()) else 0.0
        out[f"{name}_precision"] = float(tp.sum().item() / pred_mask.sum().item()) if int(pred_mask.sum().item()) else 0.0
    return out


def compute_prior_residual_metrics(preds: dict[str, torch.Tensor], labels: torch.Tensor) -> dict[str, Any]:
    p_final = preds["p_final"].detach().cpu().float().clamp_min(EPS)
    p_final = p_final / p_final.sum(dim=-1, keepdim=True).clamp_min(EPS)
    p_prior = preds["p_prior"].detach().cpu().float().clamp_min(EPS)
    p_prior = p_prior / p_prior.sum(dim=-1, keepdim=True).clamp_min(EPS)
    residual = preds["residual_delta"].detach().cpu().float()
    metrics = _draw_metric_bundle("p_prior", p_prior, labels)
    metrics.update(
        {
            "final_minus_prior_logloss": logloss_from_probs(p_final, labels) - metrics["p_prior_logloss"],
            "final_vs_prior_draw_corr": _pearson(p_final[:, 1], p_prior[:, 1]),
            "final_vs_prior_draw_mae": float((p_final[:, 1] - p_prior[:, 1]).abs().mean().item()),
            "residual_centered_l2": float(residual_centered_l2(residual).item()),
            "residual_abs_mean": float(residual.abs().mean().item()),
            "residual_abs_p95": float(torch.quantile(residual.abs().reshape(-1), 0.95).item()),
        }
    )
    if "p_anchor" in preds:
        metrics.update(_draw_metric_bundle("anchor", preds["p_anchor"], labels))
        metrics.update(_draw_metric_bundle("blend_prior", p_prior, labels))
    return metrics


def train_epoch(
    model: P12ScorePriorResidualITransformer,
    data: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    prior_mode: str,
    blend_alpha: float,
    count_loss_weight: float,
    residual_l2_weight: float,
    anchor_ce_weight: float,
    score_grid_max_goals: int,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "L_final": 0.0, "L_count": 0.0, "L_delta_centered_l2": 0.0, "L_anchor": 0.0}
    n_batches = 0
    n_correct = 0
    n_total = 0
    perm = torch.randperm(len(data["X"]))
    for start in range(0, len(perm), batch_size):
        idx = perm[start : start + batch_size]
        xb = data["X"][idx].to(device)
        yb = data["y_1x2"][idx].to(device)
        hgb = data["y_home_goals"][idx].to(device)
        agb = data["y_away_goals"][idx].to(device)
        out = model(xb)
        losses = compute_p12_losses(
            out,
            yb,
            hgb,
            agb,
            prior_mode=prior_mode,
            blend_alpha=blend_alpha,
            count_loss_weight=count_loss_weight,
            residual_l2_weight=residual_l2_weight,
            anchor_ce_weight=anchor_ce_weight,
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
def predict_outputs(
    model: P12ScorePriorResidualITransformer,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
    prior_mode: str,
    blend_alpha: float,
    score_grid_max_goals: int,
) -> dict[str, torch.Tensor]:
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {
        "p_final": [],
        "p_prior": [],
        "p_score_1x2": [],
        "goal_count_lambdas": [],
        "tail_mass": [],
        "score_probs": [],
        "home_pmf": [],
        "away_pmf": [],
        "residual_delta": [],
    }
    anchor_chunks: list[torch.Tensor] = []
    for start in range(0, len(data["X"]), batch_size):
        xb = data["X"][start : start + batch_size].to(device)
        yb = data["y_1x2"][start : start + batch_size].to(device)
        hgb = data["y_home_goals"][start : start + batch_size].to(device)
        agb = data["y_away_goals"][start : start + batch_size].to(device)
        out = model(xb)
        losses = compute_p12_losses(
            out,
            yb,
            hgb,
            agb,
            prior_mode=prior_mode,
            blend_alpha=blend_alpha,
            count_loss_weight=0.0,
            residual_l2_weight=0.0,
            anchor_ce_weight=0.0,
            score_grid_max_goals=score_grid_max_goals,
        )
        for key in chunks:
            if key in losses:
                chunks[key].append(losses[key].cpu())
            elif key in out:
                chunks[key].append(out[key].cpu())
        if "anchor_logits" in out:
            anchor_chunks.append(F.softmax(out["anchor_logits"], dim=-1).cpu())
    preds = {key: torch.cat(value, dim=0) for key, value in chunks.items() if value}
    if anchor_chunks:
        preds["p_anchor"] = torch.cat(anchor_chunks, dim=0)
    return preds


@torch.no_grad()
def evaluate(
    model: P12ScorePriorResidualITransformer,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
    prior_mode: str,
    blend_alpha: float,
    score_grid_max_goals: int,
    constant_count_nll: float,
    constant_prior_ll: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    preds = predict_outputs(model, data, batch_size, device, prior_mode, blend_alpha, score_grid_max_goals)
    labels = data["y_1x2"]
    p_final = preds["p_final"].clamp_min(EPS)
    val_metrics = compute_draw_auxiliary_metrics(p_final, labels)
    val_metrics.update(
        {
            "accuracy": float((p_final.argmax(dim=-1) == labels).float().mean().item()),
            "logloss": logloss_from_probs(p_final, labels),
            "brier": _brier_from_probs(p_final, labels, 3),
            "ece": _multiclass_ece(p_final, labels, 10),
        }
    )
    val_metrics.update(_draw_metric_bundle("final", p_final, labels))
    val_metrics["draw_top2"] = val_metrics.get("draw_top2_recall", val_metrics.get("final_draw_top2", 0.0))
    val_metrics["mean_draw_margin_to_top"] = val_metrics.get("final_draw_margin_to_top", 0.0)
    val_metrics.update(_classwise_metrics(p_final, labels))

    folded = {
        "p_score_1x2": preds["p_score_1x2"],
        "tail_mass": preds["tail_mass"],
        "score_probs": preds["score_probs"],
        "home_pmf": preds["home_pmf"],
        "away_pmf": preds["away_pmf"],
    }
    goal_count_metrics = compute_score_count_metrics(
        preds["goal_count_lambdas"],
        folded,
        data["y_home_goals"],
        data["y_away_goals"],
        constant_count_nll,
    )
    score_derived = compute_score_derived_1x2_metrics(preds["p_score_1x2"], p_final, labels, constant_prior_ll)
    score_derived = {
        ("p_score_derived" + key[len("score_derived") :]) if key.startswith("score_derived") else key: value
        for key, value in score_derived.items()
    }
    prior_residual = compute_prior_residual_metrics(preds, labels)
    return val_metrics, goal_count_metrics, {**score_derived, **prior_residual}, preds


def write_prior_predictions_csv(path: Path, data: dict[str, Any], preds: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["match_id", "y_true", "prior_home", "prior_draw", "prior_away", "final_home", "final_draw", "final_away"],
        )
        writer.writeheader()
        for idx, match_id in enumerate(data["match_ids"]):
            writer.writerow(
                {
                    "match_id": match_id,
                    "y_true": int(data["y_1x2"][idx].item()),
                    "prior_home": float(preds["p_prior"][idx, 0].item()),
                    "prior_draw": float(preds["p_prior"][idx, 1].item()),
                    "prior_away": float(preds["p_prior"][idx, 2].item()),
                    "final_home": float(preds["p_final"][idx, 0].item()),
                    "final_draw": float(preds["p_final"][idx, 1].item()),
                    "final_away": float(preds["p_final"][idx, 2].item()),
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train P12 explicit score-prior residual probes")
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
    parser.add_argument("--prior-mode", default="score", choices=PRIOR_MODES)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument("--count-loss-weight", type=float, default=0.10)
    parser.add_argument("--residual-l2-weight", type=float, default=0.0)
    parser.add_argument("--anchor-ce-weight", type=float, default=0.0)
    parser.add_argument("--score-grid-max-goals", type=int, default=10)
    parser.add_argument("--scaling", default="robust", choices=SCALING_CHOICES)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--variant", default="manual")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def prepare_p12_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing = [
        Path(out_dir) / "report.json",
        Path(out_dir) / "best_model.pth",
        Path(out_dir) / "val_predictions.csv",
        Path(out_dir) / "prior_residual_metrics.json",
        Path(out_dir) / "score_label_audit.json",
    ]
    if not allow_overwrite and any(path.exists() for path in existing):
        raise FileExistsError(f"Refusing to overwrite existing P12 outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    if "test" in Path(args.train_ids).name.lower() or "test" in Path(args.val_ids).name.lower():
        raise SystemExit("P12 refuses any test split path")
    set_deterministic_seed(args.seed)
    prepare_p12_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feature_names = p11_selected_feature_names(args.feature_groups)
    market_type_ids = p11_selected_feature_market_type_ids(args.feature_groups)
    train_rows = load_rows_for_ids(args.data, load_split_ids(args.train_ids))
    val_rows = load_rows_for_ids(args.data, load_split_ids(args.val_ids))
    if args.max_samples > 0:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: min(args.max_samples, len(val_rows))]
    train_label_audit = score_label_audit(train_rows)
    val_label_audit = score_label_audit(val_rows)
    write_json(Path(args.out_dir) / "score_label_audit.json", {"train": train_label_audit, "val": val_label_audit})
    if not train_label_audit["passes_min_coverage"] or not val_label_audit["passes_min_coverage"]:
        blocked = {
            "phase": "P12 explicit score-prior residual probe",
            "variant": args.variant,
            "verdict": "P12_BLOCKED_BY_LABEL_AUDIT",
            "score_label_audit": {"train": train_label_audit, "val": val_label_audit},
            "hard_fail_reasons": ["P12_BLOCKED_BY_LABEL_AUDIT"],
        }
        write_json(Path(args.out_dir) / "report.json", blocked)
        raise SystemExit("P12_BLOCKED_BY_LABEL_AUDIT")

    train_data = build_p11_dataset(train_rows, args.feature_groups)
    val_data = build_p11_dataset(val_rows, args.feature_groups)
    scaler = fit_p4_feature_scaler(train_data["X"], feature_names, args.scaling)
    scaler_path = save_feature_scaler(scaler, args.out_dir)
    train_data["X"] = apply_p4_feature_scaler(train_data["X"], scaler)
    val_data["X"] = apply_p4_feature_scaler(val_data["X"], scaler)
    anchor_only = compute_anchor_only_metrics(val_data["p_euro_anchor"], val_data["y_1x2"])
    constant_count_nll = constant_mean_rate_count_nll(
        train_data["y_home_goals"], train_data["y_away_goals"], val_data["y_home_goals"], val_data["y_away_goals"]
    )
    constant_prior_ll = constant_class_prior_logloss(train_data["y_1x2"], val_data["y_1x2"])

    model = P12ScorePriorResidualITransformer(
        n_features=len(feature_names),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        market_type_ids=market_type_ids,
        enable_anchor_head=args.prior_mode == "blend" or args.anchor_ce_weight > 0,
    ).to(device)
    model.initialize_goal_count_from_means(float(train_data["y_home_goals"].mean().item()), float(train_data["y_away_goals"].mean().item()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Rows: train={len(train_data['X'])} val={len(val_data['X'])}")
    print(f"Model params: {n_params:,}, prior_mode={args.prior_mode}, blend_alpha={args.blend_alpha}")

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
            args.prior_mode,
            args.blend_alpha,
            args.count_loss_weight,
            args.residual_l2_weight,
            args.anchor_ce_weight,
            args.score_grid_max_goals,
        )
        val_metrics, goal_count_metrics, prior_residual_metrics, _ = evaluate(
            model,
            val_data,
            args.batch_size * 2,
            device,
            args.prior_mode,
            args.blend_alpha,
            args.score_grid_max_goals,
            constant_count_nll,
            constant_prior_ll,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "val_logloss": val_metrics["logloss"],
            "val_ece": val_metrics["ece"],
            "val_draw_class_nll": val_metrics["draw_class_nll"],
            "val_draw_top2": val_metrics["draw_top2"],
            "val_mean_p_draw_on_true_draw": val_metrics["mean_p_draw_on_true_draw"],
            "val_count_nll": goal_count_metrics.get("val_count_nll", 0.0),
            "p_prior_logloss": prior_residual_metrics.get("p_prior_logloss", 0.0),
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
            f"draw_top2={val_metrics['draw_top2']:.4f} "
            f"prior_ll={prior_residual_metrics.get('p_prior_logloss', 0.0):.4f}"
            + (" *" if is_best else "")
        )

    ckpt_path = Path(args.out_dir) / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_val, final_goal_count, final_prior_residual, preds = evaluate(
        model,
        val_data,
        args.batch_size * 2,
        device,
        args.prior_mode,
        args.blend_alpha,
        args.score_grid_max_goals,
        constant_count_nll,
        constant_prior_ll,
    )
    write_val_predictions_csv(str(Path(args.out_dir) / "val_predictions.csv"), val_data["match_ids"], val_data["y_1x2"], preds["p_final"])
    write_prior_predictions_csv(Path(args.out_dir) / "val_prior_predictions.csv", val_data, preds)
    write_json(Path(args.out_dir) / "val_metrics.json", final_val)
    write_json(Path(args.out_dir) / "classwise_metrics.json", _classwise_metrics(preds["p_final"], val_data["y_1x2"]))
    write_json(Path(args.out_dir) / "draw_rank_metrics.json", _draw_metric_bundle("final", preds["p_final"], val_data["y_1x2"]))
    write_json(Path(args.out_dir) / "draw_margin_metrics.json", {"mean_draw_margin_to_top": final_val["mean_draw_margin_to_top"]})
    write_json(Path(args.out_dir) / "calibration_metrics.json", {"ece": final_val["ece"], "classwise_ece_draw": final_val["classwise_ece_draw"]})
    write_json(Path(args.out_dir) / "goal_count_metrics.json", final_goal_count)
    write_json(Path(args.out_dir) / "score_derived_1x2_metrics.json", {k: v for k, v in final_prior_residual.items() if k.startswith("p_score_derived") or k == "constant_class_prior_1x2_logloss"})
    write_json(Path(args.out_dir) / "prior_residual_metrics.json", final_prior_residual)

    data_summary = {
        "train_samples": int(len(train_data["X"])),
        "val_samples": int(len(val_data["X"])),
        "test_ids_used": False,
        "validation_fit_used": False,
        "posthoc_val_fit_used": False,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "score_label_audit": {"train": train_label_audit, "val": val_label_audit},
    }
    manifest = {
        "phase": "P12",
        "variant": args.variant,
        "seed": args.seed,
        "prior_mode": args.prior_mode,
        "blend_alpha": float(args.blend_alpha),
        "count_loss_weight": float(args.count_loss_weight),
        "residual_l2_weight": float(args.residual_l2_weight),
        "anchor_ce_weight": float(args.anchor_ce_weight),
        "warmup_epochs": 0,
        "p6_default_behavior_changed": False,
        "large_score_grid_head_added": False,
        "draw_specific_loss_added": False,
        "validation_posthoc_fit_used": False,
    }
    write_json(Path(args.out_dir) / "training_manifest.json", manifest)
    report = {
        "phase": "P12 explicit score-prior residual probe",
        "run_mode": "smoke" if args.max_samples else "formal",
        "model": "P12ScorePriorResidualITransformer",
        "config": vars(args),
        "data": data_summary,
        "training_manifest": manifest,
        "score_baselines": {
            "constant_train_mean_rate_count_nll": constant_count_nll,
            "constant_class_prior_1x2_logloss": constant_prior_ll,
        },
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "anchor_only_baseline": anchor_only,
        "train_metrics": history[-1] if history else {},
        "val_metrics": final_val,
        "goal_count_metrics": final_goal_count,
        "score_derived_1x2_metrics": {k: v for k, v in final_prior_residual.items() if k.startswith("p_score_derived") or k == "constant_class_prior_1x2_logloss"},
        "prior_residual_metrics": final_prior_residual,
        "history": history,
        "warnings": [],
        "params": n_params,
        "scaler": {"path": scaler_path, **scaler},
        "run_path": args.out_dir,
        "artifacts": {
            "best_model": str(ckpt_path),
            "val_predictions": str(Path(args.out_dir) / "val_predictions.csv"),
            "val_prior_predictions": str(Path(args.out_dir) / "val_prior_predictions.csv"),
            "score_label_audit": str(Path(args.out_dir) / "score_label_audit.json"),
        },
    }
    write_json(Path(args.out_dir) / "report.json", report)
    (Path(args.out_dir) / "report.md").write_text(
        "\n".join(
            [
                "# P12 Run Report",
                "",
                f"- variant: `{args.variant}`",
                f"- seed: `{args.seed}`",
                f"- val_logloss: `{final_val['logloss']}`",
                f"- draw_top2: `{final_val['draw_top2']}`",
                f"- p_prior_logloss: `{final_prior_residual.get('p_prior_logloss')}`",
                f"- final_minus_prior_logloss: `{final_prior_residual.get('final_minus_prior_logloss')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Report saved to {Path(args.out_dir) / 'report.json'}")
    print(f"Best val_logloss: {best_val_logloss:.6f}")


if __name__ == "__main__":
    main()
