"""Train the P6 residual Patch-iTransformer objective probe.

This script reads train/val ids only, never accepts test ids, and fits all
normalization parameters on the training split only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer_v2 import FEATURE_DEFS
from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
from eval.odds_metrics import logloss_from_probs
from tools.p4_goal_diff_utils import compute_anchor_only_metrics, compute_p4_losses, fold_goal_diff_probs
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
    write_val_goal_diff_predictions_csv,
    write_val_predictions_csv,
)


FEATURE_MARKET_TYPES = {name: market for name, market in FEATURE_DEFS}


def p6_selected_feature_names(feature_groups: str) -> list[str]:
    """Return the P4-compatible feature list supported by P6 phase one."""
    return selected_feature_names(feature_groups)


def p6_selected_feature_market_type_ids(feature_groups: str) -> list[int]:
    names = p6_selected_feature_names(feature_groups)
    return [int(FEATURE_MARKET_TYPES[name]) for name in names]


def prepare_p6_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing_outputs = [
        Path(out_dir) / "report.json",
        Path(out_dir) / "best_model.pth",
        Path(out_dir) / "val_predictions.csv",
        Path(out_dir) / "val_goal_diff_predictions.csv",
        Path(out_dir) / "scaler.json",
    ]
    if not allow_overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(f"Refusing to overwrite existing P6 outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train P6 residual Patch-iTransformer objective probe")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--feature-groups", default="euro", help="P6 feature groups: euro or euro,asian")
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
    parser.add_argument("--diff-loss-weight", type=float, default=0.30)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--delta-l2-weight", type=float, default=0.01)
    parser.add_argument("--lambda-draw-bce", type=float, default=0.0)
    parser.add_argument("--lambda-final-draw-bce", type=float, default=0.0)
    parser.add_argument("--enable-draw-logit-coupling", action="store_true")
    parser.add_argument("--draw-coupling-scale", type=float, default=0.0)
    parser.add_argument("--scaling", default="robust", choices=SCALING_CHOICES)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def apply_draw_logit_coupling(
    delta_logits: torch.Tensor,
    draw_aux_logit: torch.Tensor | None = None,
    enable_draw_logit_coupling: bool = False,
    draw_coupling_scale: float = 0.0,
) -> torch.Tensor:
    """Bound a draw-specific auxiliary logit and add it to the final draw residual."""
    if not enable_draw_logit_coupling or float(draw_coupling_scale) == 0.0:
        return delta_logits
    if draw_aux_logit is None:
        raise ValueError("draw_aux_logit is required when draw logit coupling is enabled")
    adjusted = delta_logits.clone()
    adjusted[:, 1] = adjusted[:, 1] + float(draw_coupling_scale) * torch.tanh(draw_aux_logit)
    return adjusted


def compute_p6_losses(
    out: dict[str, torch.Tensor],
    p_euro_anchor: torch.Tensor,
    y_1x2: torch.Tensor,
    y_goal_diff: torch.Tensor,
    diff_loss_weight: float,
    consistency_loss_weight: float,
    delta_l2_weight: float,
    lambda_draw_bce: float = 0.0,
    lambda_final_draw_bce: float = 0.0,
    enable_draw_logit_coupling: bool = False,
    draw_coupling_scale: float = 0.0,
) -> dict[str, torch.Tensor]:
    delta_logits = apply_draw_logit_coupling(
        out["delta_logits"],
        out.get("draw_aux_logit"),
        enable_draw_logit_coupling=enable_draw_logit_coupling,
        draw_coupling_scale=draw_coupling_scale,
    )
    losses = compute_p4_losses(
        delta_logits,
        out["goal_diff_logits"],
        p_euro_anchor,
        y_1x2,
        y_goal_diff,
        diff_loss_weight=diff_loss_weight,
        consistency_loss_weight=consistency_loss_weight,
        delta_l2_weight=delta_l2_weight,
    )
    if float(lambda_draw_bce) > 0:
        if "draw_aux_logit" not in out:
            raise ValueError("draw_aux_logit is required when lambda_draw_bce > 0")
        draw_target = (y_1x2 == 1).float()
        l_draw_aux = F.binary_cross_entropy_with_logits(out["draw_aux_logit"], draw_target)
        losses["loss"] = losses["loss"] + float(lambda_draw_bce) * l_draw_aux
        losses["L_draw_aux"] = l_draw_aux
    else:
        losses["L_draw_aux"] = losses["loss"].new_tensor(0.0)
    if float(lambda_final_draw_bce) > 0:
        draw_target = (y_1x2 == 1).float()
        l_final_draw = F.binary_cross_entropy(losses["p_final"][:, 1].clamp(1e-8, 1.0 - 1e-8), draw_target)
        losses["loss"] = losses["loss"] + float(lambda_final_draw_bce) * l_final_draw
        losses["L_final_draw"] = l_final_draw
    else:
        losses["L_final_draw"] = losses["loss"].new_tensor(0.0)
    losses["delta_logits_after_coupling"] = delta_logits
    return losses


def compute_draw_auxiliary_metrics(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> dict[str, Any]:
    probs = probs.detach().cpu().float().clamp(1e-8, 1.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    labels = labels.detach().cpu().long()
    preds = probs.argmax(dim=-1)
    true_draw = labels == 1
    pred_draw = preds == 1
    n_true_draw = int(true_draw.sum().item())
    n_pred_draw = int(pred_draw.sum().item())

    if n_true_draw:
        draw_recall = float((pred_draw & true_draw).sum().item() / n_true_draw)
        draw_class_nll = float((-torch.log(probs[true_draw, 1].clamp_min(1e-8))).mean().item())
        mean_p_draw_on_true_draw = float(probs[true_draw, 1].mean().item())
        top2 = torch.topk(probs, k=2, dim=-1).indices
        draw_top2_recall = float((top2[true_draw] == 1).any(dim=-1).float().mean().item())
    else:
        draw_recall = 0.0
        draw_class_nll = 0.0
        mean_p_draw_on_true_draw = 0.0
        draw_top2_recall = 0.0

    non_draw = ~true_draw
    mean_p_draw_on_non_draw = float(probs[non_draw, 1].mean().item()) if int(non_draw.sum().item()) else 0.0
    draw_precision = float((pred_draw & true_draw).sum().item() / n_pred_draw) if n_pred_draw else 0.0
    draw_prob = probs[:, 1]
    draw_target = true_draw.float()
    classwise_ece_draw = _binary_ece(draw_prob, draw_target, n_bins=n_bins)

    return {
        "draw_class_nll": draw_class_nll,
        "draw_precision": draw_precision,
        "draw_recall": draw_recall,
        "draw_top2_recall": draw_top2_recall,
        "argmax_draw_count": n_pred_draw,
        "mean_p_draw": float(draw_prob.mean().item()),
        "mean_p_draw_on_true_draw": mean_p_draw_on_true_draw,
        "mean_p_draw_on_non_draw": mean_p_draw_on_non_draw,
        "classwise_ece_draw": classwise_ece_draw,
    }


def _binary_ece(confidence: torch.Tensor, target: torch.Tensor, n_bins: int = 10) -> float:
    confidence = confidence.detach().cpu().float().clamp(0.0, 1.0)
    target = target.detach().cpu().float()
    if confidence.numel() == 0:
        return 0.0
    ece = 0.0
    for i in range(int(n_bins)):
        low = i / n_bins
        high = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if int(mask.sum().item()) == 0:
            continue
        bin_conf = float(confidence[mask].mean().item())
        bin_acc = float(target[mask].mean().item())
        ece += float(mask.float().mean().item()) * abs(bin_acc - bin_conf)
    return float(ece)


def train_epoch(
    model: P6ResidualPatchITransformer,
    data: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    diff_loss_weight: float,
    consistency_loss_weight: float,
    delta_l2_weight: float,
    lambda_draw_bce: float,
    lambda_final_draw_bce: float,
    enable_draw_logit_coupling: bool,
    draw_coupling_scale: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "L_1x2": 0.0,
        "L_diff": 0.0,
        "L_consistency": 0.0,
        "L_delta": 0.0,
        "L_draw_aux": 0.0,
        "L_final_draw": 0.0,
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
        out = model(xb)
        losses = compute_p6_losses(
            out,
            ab,
            yb,
            ygb,
            diff_loss_weight=diff_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            delta_l2_weight=delta_l2_weight,
            lambda_draw_bce=lambda_draw_bce,
            lambda_final_draw_bce=lambda_final_draw_bce,
            enable_draw_logit_coupling=enable_draw_logit_coupling,
            draw_coupling_scale=draw_coupling_scale,
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
    model: P6ResidualPatchITransformer,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
    enable_draw_logit_coupling: bool = False,
    draw_coupling_scale: float = 0.0,
) -> dict[str, torch.Tensor]:
    model.eval()
    p_final, p_from_diff, q_diff, delta_logits_list = [], [], [], []
    for start in range(0, len(data["X"]), batch_size):
        xb = data["X"][start : start + batch_size].to(device)
        ab = data["p_euro_anchor"][start : start + batch_size].to(device)
        out = model(xb)
        delta_logits = apply_draw_logit_coupling(
            out["delta_logits"],
            out.get("draw_aux_logit"),
            enable_draw_logit_coupling=enable_draw_logit_coupling,
            draw_coupling_scale=draw_coupling_scale,
        )
        final_logits = torch.log(ab.clamp_min(1e-8)) + delta_logits
        p_final.append(torch.softmax(final_logits, dim=-1).cpu())
        q = torch.softmax(out["goal_diff_logits"], dim=-1)
        q_diff.append(q.cpu())
        p_from_diff.append(fold_goal_diff_probs(q).cpu())
        delta_logits_list.append(delta_logits.cpu())
    return {
        "p_final": torch.cat(p_final, dim=0),
        "p_from_diff": torch.cat(p_from_diff, dim=0),
        "q_diff": torch.cat(q_diff, dim=0),
        "delta_logits": torch.cat(delta_logits_list, dim=0),
    }


@torch.no_grad()
def evaluate(
    model: P6ResidualPatchITransformer,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
    enable_draw_logit_coupling: bool = False,
    draw_coupling_scale: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    preds = predict_outputs(
        model,
        data,
        batch_size,
        device,
        enable_draw_logit_coupling=enable_draw_logit_coupling,
        draw_coupling_scale=draw_coupling_scale,
    )
    labels = data["y_1x2"]
    goal_labels = data["y_goal_diff"]
    p_final = preds["p_final"].clamp_min(1e-8)
    q = preds["q_diff"].clamp_min(1e-8)
    p_from_diff = preds["p_from_diff"].clamp_min(1e-8)

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
    zero_recall = (
        float((true_zero & pred_zero).sum().item() / true_zero.sum().item())
        if int(true_zero.sum().item()) > 0
        else 0.0
    )
    final_vs_diff_kl = (p_from_diff * (p_from_diff.log() - p_final.log())).sum(dim=-1).mean().item()
    goal_diff_metrics = {
        "diff_nll": logloss_from_probs(q, goal_labels),
        "diff_acc": float((q.argmax(dim=-1) == goal_labels).float().mean().item()),
        "bucket_acc": float((q.argmax(dim=-1) == goal_labels).float().mean().item()),
        "zero_recall": zero_recall,
        "p_from_diff_logloss": logloss_from_probs(p_from_diff, labels),
        "final_vs_diff_kl": float(final_vs_diff_kl),
    }
    return val_metrics, goal_diff_metrics, preds


def _brier_from_probs(probs: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    one_hot = F.one_hot(labels.detach().cpu().long(), num_classes=n_classes).float()
    return float(((probs.detach().cpu().float() - one_hot) ** 2).sum(dim=-1).mean().item())


def _multiclass_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().long()
    confidence, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    return _binary_ece(confidence, correct, n_bins=n_bins)


def build_p6_report_payload(
    config: dict[str, Any],
    data: dict[str, Any],
    best_epoch: int,
    best_val_logloss: float,
    val_metrics: dict[str, Any],
    goal_diff_metrics: dict[str, Any],
    anchor_only_baseline: dict[str, Any],
    warnings: list[str],
    train_metrics: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "phase": "P6 residual Patch-iTransformer objective probe",
        "run_mode": config.get("run_mode") or ("smoke" if config.get("max_samples") else "formal"),
        "model": "P6ResidualPatchITransformer",
        "config": config,
        "data": data,
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "anchor_only_baseline": anchor_only_baseline,
        "train_metrics": train_metrics or {},
        "val_metrics": val_metrics,
        "goal_diff_metrics": goal_diff_metrics,
        "history": history or [],
        "warnings": warnings,
    }
    payload.update(extra)
    return payload


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    feature_names = p6_selected_feature_names(args.feature_groups)
    market_type_ids = p6_selected_feature_market_type_ids(args.feature_groups)
    set_deterministic_seed(args.seed)
    prepare_p6_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
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
        raise ValueError("P6 selected feature names drifted from P4 dataset construction")

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
        enable_draw_aux_head=args.lambda_draw_bce > 0 or args.enable_draw_logit_coupling,
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
            args.diff_loss_weight,
            args.consistency_loss_weight,
            args.delta_l2_weight,
            args.lambda_draw_bce,
            args.lambda_final_draw_bce,
            args.enable_draw_logit_coupling,
            args.draw_coupling_scale,
        )
        val_metrics, goal_diff_metrics, _ = evaluate(
            model,
            val_data,
            args.batch_size * 2,
            device,
            enable_draw_logit_coupling=args.enable_draw_logit_coupling,
            draw_coupling_scale=args.draw_coupling_scale,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_L_1x2": train_metrics["L_1x2"],
            "train_L_diff": train_metrics["L_diff"],
            "train_L_consistency": train_metrics["L_consistency"],
            "train_L_delta": train_metrics["L_delta"],
            "train_L_draw_aux": train_metrics["L_draw_aux"],
            "train_L_final_draw": train_metrics["L_final_draw"],
            "train_acc": train_metrics["accuracy"],
            "val_logloss": val_metrics["logloss"],
            "val_brier": val_metrics["brier"],
            "val_ece": val_metrics["ece"],
            "val_accuracy": val_metrics["accuracy"],
            "val_argmax_draw_count": val_metrics["argmax_draw_count"],
            "val_draw_class_nll": val_metrics["draw_class_nll"],
            "val_draw_precision": val_metrics["draw_precision"],
            "val_draw_recall": val_metrics["draw_recall"],
            "val_draw_top2_recall": val_metrics["draw_top2_recall"],
            "val_classwise_ece_draw": val_metrics["classwise_ece_draw"],
            "val_mean_p_draw": val_metrics["mean_p_draw"],
            "val_mean_p_draw_on_true_draw": val_metrics["mean_p_draw_on_true_draw"],
            "val_mean_p_draw_on_non_draw": val_metrics["mean_p_draw_on_non_draw"],
            "val_goal_diff_nll": goal_diff_metrics["diff_nll"],
            "val_goal_diff_acc": goal_diff_metrics["diff_acc"],
            "val_goal_diff_zero_recall": goal_diff_metrics["zero_recall"],
            "val_p_from_diff_logloss": goal_diff_metrics["p_from_diff_logloss"],
            "val_final_vs_diff_kl": goal_diff_metrics["final_vs_diff_kl"],
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
            "enable_draw_aux_head": args.lambda_draw_bce > 0 or args.enable_draw_logit_coupling,
        },
        Path(args.out_dir) / "best_model.pth",
            )
        print(
            f"Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
            f"val_logloss={val_metrics['logloss']:.4f} "
            f"diff_acc={goal_diff_metrics['diff_acc']:.4f} "
            f"draw_recall={val_metrics['draw_recall']:.4f} "
            f"draw_precision={val_metrics['draw_precision']:.4f}"
            + (" *" if is_best else "")
        )

    ckpt_path = Path(args.out_dir) / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_val, final_goal_diff, preds = evaluate(
        model,
        val_data,
        args.batch_size * 2,
        device,
        enable_draw_logit_coupling=args.enable_draw_logit_coupling,
        draw_coupling_scale=args.draw_coupling_scale,
    )
    write_val_predictions_csv(
        str(Path(args.out_dir) / "val_predictions.csv"),
        val_data["match_ids"],
        val_data["y_1x2"],
        preds["p_final"],
    )
    write_val_goal_diff_predictions_csv(
        str(Path(args.out_dir) / "val_goal_diff_predictions.csv"),
        val_data["match_ids"],
        val_data["y_goal_diff"],
        preds["q_diff"],
        preds["p_from_diff"],
    )

    warnings: list[str] = []
    if best_val_logloss > anchor_only["anchor_only_val_logloss"] + 0.0003:
        warnings.append("P6 residual is worse than anchor-only by more than 0.0003")

    data_summary = {
        "train_samples": int(len(train_data["X"])),
        "val_samples": int(len(val_data["X"])),
        "test_ids_used": False,
        "validation_fit_used": False,
        "posthoc_val_fit_used": False,
        "checkpoint_selection": "best_val_logloss",
        "checkpoint_selection_warning": True,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "train_skipped_rows": int(train_data["skipped_rows"]),
        "val_skipped_rows": int(val_data["skipped_rows"]),
        "anchor_fallback_count": int(val_data["anchor_fallback_count"]),
        "negative_time_valid_euro_count": int(val_data["negative_time_valid_euro_count"]),
    }
    report = build_p6_report_payload(
        config=vars(args),
        data=data_summary,
        best_epoch=best_epoch,
        best_val_logloss=best_val_logloss,
        anchor_only_baseline=anchor_only,
        train_metrics=history[-1] if history else {},
        val_metrics=final_val,
        goal_diff_metrics=final_goal_diff,
        history=history,
        warnings=warnings,
        params=n_params,
        scaler={"path": scaler_path, **scaler},
        run_path=args.out_dir,
        artifacts={
            "best_model": str(ckpt_path),
            "val_predictions": str(Path(args.out_dir) / "val_predictions.csv"),
            "val_goal_diff_predictions": str(Path(args.out_dir) / "val_goal_diff_predictions.csv"),
        },
    )
    report_path = Path(args.out_dir) / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report saved to {report_path}")
    print(f"Best val_logloss: {best_val_logloss:.6f}")
    print(f"Anchor-only val_logloss: {anchor_only['anchor_only_val_logloss']:.6f}")


if __name__ == "__main__":
    main()
