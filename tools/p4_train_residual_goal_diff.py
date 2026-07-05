"""Train the P4 residual goal-diff objective probe.

This script reads train/val ids only, never test ids. It builds selected P4
features from the existing P3b bucketizer and writes run-scoped artifacts.
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
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_metrics import logloss_from_probs
from model.odds_patch_itransformer_v2 import FEATURE_DEFS
from model.p4_residual_goal_diff import P4ResidualGoalDiffModel
from tools.p3b_train_patch_itransformer import bucketize_sample_v2_for_training
from tools.p4_goal_diff_utils import (
    compute_1x2_metrics,
    compute_anchor_only_metrics,
    compute_p4_losses,
    extract_euro_anchor_probs,
    fold_goal_diff_probs,
    goal_diff_to_bucket,
)


EURO_MAP = {"home": 0, "draw": 1, "away": 2}
FEATURE_INDEX = {name: idx for idx, (name, _) in enumerate(FEATURE_DEFS)}
P4_FEATURES = {
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
    "time": ["time_log", "bucket_valid_count"],
}
SCALING_CHOICES = ("none", "standard", "robust")
SCALER_EXCLUDED_FEATURES = {"has_euro", "has_asian", "bucket_valid_count"}
SCALING_EPS = 1e-6


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing_outputs = [
        Path(out_dir) / "report.json",
        Path(out_dir) / "best_model.pth",
        Path(out_dir) / "val_predictions.csv",
        Path(out_dir) / "val_goal_diff_predictions.csv",
    ]
    if not allow_overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(f"Refusing to overwrite existing P4 outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def load_split_ids(path: str) -> set[str]:
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mid = line.strip()
            if mid:
                ids.add(mid)
    return ids


def load_rows_for_ids(jsonl_path: str, match_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row.get("match_id")) in match_ids:
                rows.append(row)
    return rows


def selected_feature_names(feature_groups: str) -> list[str]:
    selected = {part.strip().lower() for part in feature_groups.split(",") if part.strip()}
    if selected not in ({"euro"}, {"euro", "asian"}):
        raise ValueError("P4.1 feature_groups must be 'euro' or 'euro,asian'")
    names = list(P4_FEATURES["euro"])
    if "asian" in selected:
        names.extend(P4_FEATURES["asian"])
    names.extend(P4_FEATURES["time"])
    return names


def build_p4_dataset(rows: list[dict[str, Any]], feature_groups: str) -> dict[str, Any]:
    names = selected_feature_names(feature_groups)
    indices = [FEATURE_INDEX[name] for name in names]
    X, y_1x2, y_goal_diff, anchors, match_ids = [], [], [], [], []
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

        X.append(xb_full[:, indices, :])
        y_1x2.append(EURO_MAP[result])
        y_goal_diff.append(goal_diff_to_bucket(home_goals - away_goals))
        anchors.append(anchor)
        match_ids.append(str(row.get("match_id", row_idx)))
        fallback_count += int(anchor_diag["used_fallback"])
        negative_time_valid_euro_count += int(anchor_diag["negative_time_valid_euro_count"])

    if not X:
        raise ValueError("No P4 labelled rows available")
    return {
        "X": torch.stack(X),
        "y_1x2": torch.tensor(y_1x2, dtype=torch.long),
        "y_goal_diff": torch.tensor(y_goal_diff, dtype=torch.long),
        "p_euro_anchor": torch.stack(anchors),
        "match_ids": match_ids,
        "feature_names": names,
        "anchor_fallback_count": fallback_count,
        "negative_time_valid_euro_count": negative_time_valid_euro_count,
        "skipped_rows": skipped,
    }


def fit_p4_feature_scaler(x_train: torch.Tensor, feature_names: list[str], scaling: str) -> dict[str, Any]:
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
            if not math.isfinite(scale) or scale <= SCALING_EPS:
                scale = _safe_scale(values)
                fallback = "std_or_unit"
            stats = {"index": idx, "center": center, "scale": scale, "q25": q25, "q75": q75, "fallback": fallback}
        scaler["scaled_features"].append(name)
        scaler["features"][name] = stats
    return scaler


def apply_p4_feature_scaler(x: torch.Tensor, scaler: dict[str, Any]) -> torch.Tensor:
    if scaler.get("scaling") == "none":
        return x
    for stats in scaler.get("features", {}).values():
        idx = int(stats["index"])
        x[..., idx, :] = (x[..., idx, :] - float(stats["center"])) / float(stats["scale"])
    return x


def save_feature_scaler(scaler: dict[str, Any], out_dir: str) -> str:
    path = Path(out_dir) / "scaler.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)
    return str(path)


def _tensor_float(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def _safe_scale(values: torch.Tensor) -> float:
    scale = _tensor_float(values.std(unbiased=False))
    return scale if math.isfinite(scale) and scale > SCALING_EPS else 1.0


def train_epoch(
    model: P4ResidualGoalDiffModel,
    data: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    diff_loss_weight: float,
    consistency_loss_weight: float,
    delta_l2_weight: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "L_1x2": 0.0, "L_diff": 0.0, "L_consistency": 0.0, "L_delta": 0.0}
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
        losses = compute_p4_losses(
            out["delta_logits"],
            out["goal_diff_logits"],
            ab,
            yb,
            ygb,
            diff_loss_weight=diff_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            delta_l2_weight=delta_l2_weight,
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size_actual = int(len(idx))
        for key in totals:
            totals[key] += float(losses[key].detach().cpu().item())
        n_correct += int((losses["p_final"].argmax(dim=-1) == yb).sum().item())
        n_total += batch_size_actual
        n_batches += 1

    metrics = {key: value / max(n_batches, 1) for key, value in totals.items()}
    metrics["accuracy"] = n_correct / max(n_total, 1)
    return metrics


@torch.no_grad()
def predict_outputs(
    model: P4ResidualGoalDiffModel,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
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
def evaluate(
    model: P4ResidualGoalDiffModel,
    data: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    preds = predict_outputs(model, data, batch_size, device)
    labels = data["y_1x2"]
    goal_labels = data["y_goal_diff"]
    val_metrics = compute_1x2_metrics(preds["p_final"], labels)
    q = preds["q_diff"].clamp_min(1e-8)
    p_final = preds["p_final"].clamp_min(1e-8)
    p_from_diff = preds["p_from_diff"].clamp_min(1e-8)
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


def write_val_predictions_csv(path: str, match_ids: list[str], labels: torch.Tensor, probs: torch.Tensor) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["match_id", "y_true", "p_home", "p_draw", "p_away", "pred_class", "correct"],
        )
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
                }
            )


def write_val_goal_diff_predictions_csv(
    path: str,
    match_ids: list[str],
    labels: torch.Tensor,
    q_diff: torch.Tensor,
    p_from_diff: torch.Tensor,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["match_id", "y_goal_diff", "pred_bucket", "correct_bucket"]
    fields += [f"p_bucket_{idx}" for idx in range(7)]
    fields += ["p_home_from_diff", "p_draw_from_diff", "p_away_from_diff"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, q in enumerate(q_diff):
            pred = int(q.argmax().item())
            y_true = int(labels[idx].item())
            row = {
                "match_id": match_ids[idx] if idx < len(match_ids) else str(idx),
                "y_goal_diff": y_true,
                "pred_bucket": pred,
                "correct_bucket": int(pred == y_true),
                "p_home_from_diff": float(p_from_diff[idx, 0].item()),
                "p_draw_from_diff": float(p_from_diff[idx, 1].item()),
                "p_away_from_diff": float(p_from_diff[idx, 2].item()),
            }
            for bucket in range(7):
                row[f"p_bucket_{bucket}"] = float(q[bucket].item())
            writer.writerow(row)


def build_report_payload(
    config: dict[str, Any],
    data: dict[str, Any],
    best_epoch: int,
    best_val_logloss: float,
    anchor_only_baseline: dict[str, Any],
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    goal_diff_metrics: dict[str, Any],
    history: list[dict[str, Any]],
    warnings: list[str],
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "phase": "P4 residual goal-diff objective probe",
        "model": "P4ResidualGoalDiffModel",
        "config": config,
        "data": data,
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "anchor_only_baseline": anchor_only_baseline,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "goal_diff_metrics": goal_diff_metrics,
        "history": history,
        "warnings": warnings,
    }
    payload.update(extra)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P4 residual goal-diff objective probe")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--feature-groups", default="euro", help="P4.1 feature groups: euro or euro,asian")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--diff-loss-weight", type=float, default=0.30)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--delta-l2-weight", type=float, default=0.01)
    parser.add_argument("--scaling", default="robust", choices=SCALING_CHOICES)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    prepare_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
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
    feature_names = train_data["feature_names"]
    scaler = fit_p4_feature_scaler(train_data["X"], feature_names, args.scaling)
    scaler_path = save_feature_scaler(scaler, args.out_dir)
    train_data["X"] = apply_p4_feature_scaler(train_data["X"], scaler)
    val_data["X"] = apply_p4_feature_scaler(val_data["X"], scaler)

    anchor_only = compute_anchor_only_metrics(val_data["p_euro_anchor"], val_data["y_1x2"])
    model = P4ResidualGoalDiffModel(
        n_features=len(feature_names),
        d_model=args.d_model,
        dropout=args.dropout,
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
        )
        val_metrics, goal_diff_metrics, _ = evaluate(model, val_data, args.batch_size * 2, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_L_1x2": train_metrics["L_1x2"],
            "train_L_diff": train_metrics["L_diff"],
            "train_L_consistency": train_metrics["L_consistency"],
            "train_L_delta": train_metrics["L_delta"],
            "train_acc": train_metrics["accuracy"],
            "val_logloss": val_metrics["logloss"],
            "val_brier": val_metrics["brier"],
            "val_ece": val_metrics["ece"],
            "val_accuracy": val_metrics["accuracy"],
            "val_argmax_draw_count": val_metrics["argmax_draw_count"],
            "val_draw_recall": val_metrics["draw_recall"],
            "val_mean_p_draw": val_metrics["mean_p_draw"],
            "val_mean_p_draw_on_true_draw": val_metrics["mean_p_draw_on_true_draw"],
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
                    "scaler": scaler,
                    "best_epoch": best_epoch,
                    "best_val_logloss": best_val_logloss,
                },
                Path(args.out_dir) / "best_model.pth",
            )
        print(
            f"Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
            f"val_logloss={val_metrics['logloss']:.4f} "
            f"diff_acc={goal_diff_metrics['diff_acc']:.4f} "
            f"draw_recall={val_metrics['draw_recall']:.4f}"
            + (" *" if is_best else "")
        )

    ckpt_path = Path(args.out_dir) / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_val, final_goal_diff, preds = evaluate(model, val_data, args.batch_size * 2, device)
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
        warnings.append("P4 residual is worse than anchor-only by more than 0.0003")

    data_summary = {
        "train_samples": int(len(train_data["X"])),
        "val_samples": int(len(val_data["X"])),
        "test_ids_used": False,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "train_skipped_rows": int(train_data["skipped_rows"]),
        "val_skipped_rows": int(val_data["skipped_rows"]),
        "anchor_fallback_count": int(val_data["anchor_fallback_count"]),
        "negative_time_valid_euro_count": int(val_data["negative_time_valid_euro_count"]),
    }
    report = build_report_payload(
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
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")
    print(f"Best val_logloss: {best_val_logloss:.6f}")
    print(f"Anchor-only val_logloss: {anchor_only['anchor_only_val_logloss']:.6f}")


if __name__ == "__main__":
    main()
