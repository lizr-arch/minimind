"""Train P9 market-prior anchored residual variants.

P9 preserves the P6 Patch-iTransformer architecture and data path. It only adds
fixed draw-preservation regularizers around the existing market close prior.
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
from tools.p6_train_residual_patch_itransformer import (
    _brier_from_probs,
    _multiclass_ece,
    compute_draw_auxiliary_metrics,
)
from eval.odds_metrics import logloss_from_probs


FEATURE_MARKET_TYPES = {name: market for name, market in FEATURE_DEFS}


def p9_selected_feature_names(feature_groups: str) -> list[str]:
    return selected_feature_names(feature_groups)


def p9_selected_feature_market_type_ids(feature_groups: str) -> list[int]:
    return [int(FEATURE_MARKET_TYPES[name]) for name in p9_selected_feature_names(feature_groups)]


def prepare_p9_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing_outputs = [
        Path(out_dir) / "report.json",
        Path(out_dir) / "best_model.pth",
        Path(out_dir) / "val_predictions.csv",
        Path(out_dir) / "val_goal_diff_predictions.csv",
        Path(out_dir) / "scaler.json",
    ]
    if not allow_overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(f"Refusing to overwrite existing P9 outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def compute_p9_losses(
    out: dict[str, torch.Tensor],
    p_market_anchor: torch.Tensor,
    y_1x2: torch.Tensor,
    y_goal_diff: torch.Tensor,
    diff_loss_weight: float,
    consistency_loss_weight: float,
    delta_l2_weight: float,
    draw_delta_l2_weight: float = 0.0,
    anchor_kl_weight: float = 0.0,
    true_draw_floor_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    losses = compute_p4_losses(
        out["delta_logits"],
        out["goal_diff_logits"],
        p_market_anchor,
        y_1x2,
        y_goal_diff,
        diff_loss_weight=diff_loss_weight,
        consistency_loss_weight=consistency_loss_weight,
        delta_l2_weight=delta_l2_weight,
    )
    p_final = losses["p_final"].clamp_min(1e-8)
    anchor = p_market_anchor.clamp_min(1e-8)

    if float(draw_delta_l2_weight) > 0:
        l_draw_delta = out["delta_logits"][:, 1].pow(2).mean()
        losses["loss"] = losses["loss"] + float(draw_delta_l2_weight) * l_draw_delta
    else:
        l_draw_delta = losses["loss"].new_tensor(0.0)

    if float(anchor_kl_weight) > 0:
        l_anchor_kl = (anchor * (anchor.log() - p_final.log())).sum(dim=-1).mean()
        losses["loss"] = losses["loss"] + float(anchor_kl_weight) * l_anchor_kl
    else:
        l_anchor_kl = losses["loss"].new_tensor(0.0)

    if float(true_draw_floor_weight) > 0:
        true_draw = y_1x2 == 1
        if int(true_draw.sum().item()) > 0:
            l_true_draw_floor = torch.relu(anchor[true_draw, 1] - p_final[true_draw, 1]).mean()
        else:
            l_true_draw_floor = losses["loss"].new_tensor(0.0)
        losses["loss"] = losses["loss"] + float(true_draw_floor_weight) * l_true_draw_floor
    else:
        l_true_draw_floor = losses["loss"].new_tensor(0.0)

    losses["L_draw_delta"] = l_draw_delta
    losses["L_anchor_kl"] = l_anchor_kl
    losses["L_true_draw_floor"] = l_true_draw_floor
    return losses


def train_epoch(
    model: P6ResidualPatchITransformer,
    data: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    diff_loss_weight: float,
    consistency_loss_weight: float,
    delta_l2_weight: float,
    draw_delta_l2_weight: float,
    anchor_kl_weight: float,
    true_draw_floor_weight: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "L_1x2": 0.0,
        "L_diff": 0.0,
        "L_consistency": 0.0,
        "L_delta": 0.0,
        "L_draw_delta": 0.0,
        "L_anchor_kl": 0.0,
        "L_true_draw_floor": 0.0,
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
        losses = compute_p9_losses(
            out,
            ab,
            yb,
            ygb,
            diff_loss_weight=diff_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            delta_l2_weight=delta_l2_weight,
            draw_delta_l2_weight=draw_delta_l2_weight,
            anchor_kl_weight=anchor_kl_weight,
            true_draw_floor_weight=true_draw_floor_weight,
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
        final_logits = torch.log(ab.clamp_min(1e-8)) + out["delta_logits"]
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
def evaluate(model: P6ResidualPatchITransformer, data: dict[str, Any], batch_size: int, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    preds = predict_outputs(model, data, batch_size, device)
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
    draw_ranks = (torch.argsort(p_final, dim=-1, descending=True) == 1).nonzero(as_tuple=False)[:, 1] + 1
    draw_margin = p_final.max(dim=-1).values - p_final[:, 1]
    val_metrics.update(
        {
            "draw_rank1_count": int((draw_ranks == 1).sum().item()),
            "draw_rank2_count": int((draw_ranks == 2).sum().item()),
            "draw_rank3_count": int((draw_ranks == 3).sum().item()),
            "mean_draw_margin_to_top": float(draw_margin.mean().item()),
        }
    )
    true_zero = goal_labels == 3
    pred_zero = q.argmax(dim=-1) == 3
    zero_recall = float((true_zero & pred_zero).sum().item() / true_zero.sum().item()) if int(true_zero.sum().item()) else 0.0
    goal_diff_metrics = {
        "diff_nll": logloss_from_probs(q, goal_labels),
        "diff_acc": float((q.argmax(dim=-1) == goal_labels).float().mean().item()),
        "zero_recall": zero_recall,
        "p_from_diff_logloss": logloss_from_probs(p_from_diff, labels),
        "final_vs_diff_kl": float((p_from_diff * (p_from_diff.log() - p_final.log())).sum(dim=-1).mean().item()),
    }
    return val_metrics, goal_diff_metrics, preds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train P9 market-prior anchored residual variants")
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
    parser.add_argument("--draw-delta-l2-weight", type=float, default=0.0)
    parser.add_argument("--anchor-kl-weight", type=float, default=0.0)
    parser.add_argument("--true-draw-floor-weight", type=float, default=0.0)
    parser.add_argument("--scaling", default="robust", choices=SCALING_CHOICES)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--variant", default="manual")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if "test" in Path(args.train_ids).name.lower() or "test" in Path(args.val_ids).name.lower():
        raise SystemExit("P9 refuses any test split path")
    feature_names = p9_selected_feature_names(args.feature_groups)
    market_type_ids = p9_selected_feature_market_type_ids(args.feature_groups)
    set_deterministic_seed(args.seed)
    prepare_p9_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_rows = load_rows_for_ids(args.data, load_split_ids(args.train_ids))
    val_rows = load_rows_for_ids(args.data, load_split_ids(args.val_ids))
    if args.max_samples > 0:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: min(args.max_samples, len(val_rows))]
    print(f"Rows: train={len(train_rows)} val={len(val_rows)}")
    train_data = build_p4_dataset(train_rows, args.feature_groups)
    val_data = build_p4_dataset(val_rows, args.feature_groups)
    if train_data["feature_names"] != feature_names or val_data["feature_names"] != feature_names:
        raise ValueError("P9 selected feature names drifted from dataset construction")
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
            args.draw_delta_l2_weight,
            args.anchor_kl_weight,
            args.true_draw_floor_weight,
        )
        val_metrics, goal_diff_metrics, _ = evaluate(model, val_data, args.batch_size * 2, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "val_logloss": val_metrics["logloss"],
            "val_ece": val_metrics["ece"],
            "val_accuracy": val_metrics["accuracy"],
            "val_draw_class_nll": val_metrics["draw_class_nll"],
            "val_draw_recall": val_metrics["draw_recall"],
            "val_draw_precision": val_metrics["draw_precision"],
            "val_draw_top2_recall": val_metrics["draw_top2_recall"],
            "val_mean_p_draw_on_true_draw": val_metrics["mean_p_draw_on_true_draw"],
            "val_mean_draw_margin_to_top": val_metrics["mean_draw_margin_to_top"],
            "val_goal_diff_nll": goal_diff_metrics["diff_nll"],
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
    final_val, final_goal_diff, preds = evaluate(model, val_data, args.batch_size * 2, device)
    write_val_predictions_csv(str(Path(args.out_dir) / "val_predictions.csv"), val_data["match_ids"], val_data["y_1x2"], preds["p_final"])
    write_val_goal_diff_predictions_csv(
        str(Path(args.out_dir) / "val_goal_diff_predictions.csv"),
        val_data["match_ids"],
        val_data["y_goal_diff"],
        preds["q_diff"],
        preds["p_from_diff"],
    )
    data_summary = {
        "train_samples": int(len(train_data["X"])),
        "val_samples": int(len(val_data["X"])),
        "test_ids_used": False,
        "validation_fit_used": False,
        "posthoc_val_fit_used": False,
        "anchor_type": "market_close_de_vig",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "train_skipped_rows": int(train_data["skipped_rows"]),
        "val_skipped_rows": int(val_data["skipped_rows"]),
        "anchor_fallback_count": int(val_data["anchor_fallback_count"]),
        "negative_time_valid_euro_count": int(val_data["negative_time_valid_euro_count"]),
    }
    report = {
        "phase": "P9 Market-Prior Anchored Residual Model",
        "run_mode": "smoke" if args.max_samples else "formal",
        "model": "P6ResidualPatchITransformer",
        "config": vars(args),
        "data": data_summary,
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "anchor_only_baseline": anchor_only,
        "train_metrics": history[-1] if history else {},
        "val_metrics": final_val,
        "goal_diff_metrics": final_goal_diff,
        "history": history,
        "warnings": [],
        "params": n_params,
        "scaler": {"path": scaler_path, **scaler},
        "run_path": args.out_dir,
        "artifacts": {
            "best_model": str(ckpt_path),
            "val_predictions": str(Path(args.out_dir) / "val_predictions.csv"),
            "val_goal_diff_predictions": str(Path(args.out_dir) / "val_goal_diff_predictions.csv"),
        },
    }
    report_path = Path(args.out_dir) / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report saved to {report_path}")
    print(f"Best val_logloss: {best_val_logloss:.6f}")
    print(f"Anchor-only val_logloss: {anchor_only['anchor_only_val_logloss']:.6f}")


if __name__ == "__main__":
    main()
