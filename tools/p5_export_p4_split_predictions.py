"""Export P4 train/val split predictions with final, anchor, and diff probabilities.

This P5.2a source exporter deliberately refuses test splits. It reconstructs the
P4 dataset from the run report/config and checkpoint instead of relying on the
run-scoped validation CSVs, because existing P4 artifacts do not include train
predictions or anchor columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p4_residual_goal_diff import P4ResidualGoalDiffModel
from tools.p4_train_residual_goal_diff import (
    apply_p4_feature_scaler,
    build_p4_dataset,
    load_rows_for_ids,
    load_split_ids,
    predict_outputs,
    selected_feature_names,
)


EXPORT_FIELDS = [
    "match_id",
    "y_true",
    "y_goal_diff",
    "p_home",
    "p_draw",
    "p_away",
    "p_anchor_home",
    "p_anchor_draw",
    "p_anchor_away",
    "p_home_from_diff",
    "p_draw_from_diff",
    "p_away_from_diff",
    "pred_class",
    "correct",
    "pred_diff_bucket",
    "correct_diff_bucket",
]


@dataclass(frozen=True)
class ExportPlan:
    run_dir: Path
    split: str
    out_csv: Path
    report_path: Path
    checkpoint_path: Path
    scaler_path: Path
    data_path: Path
    ids_path: Path
    feature_groups: str
    d_model: int
    dropout: float
    batch_size: int
    test_ids_used: bool


def build_export_plan(run_dir: str | Path, split: str, out_csv: str | Path, no_test: bool) -> ExportPlan:
    """Validate requested split and resolve P4 run config into an export plan."""
    if not no_test:
        raise ValueError("--no-test is required for P5 split prediction export")
    split = split.lower().strip()
    if split == "test":
        raise ValueError("test split export is forbidden")
    if split not in {"train", "val"}:
        raise ValueError("split must be one of: train, val")

    run_dir = Path(run_dir)
    report_path = run_dir / "report.json"
    checkpoint_path = run_dir / "best_model.pth"
    scaler_path = run_dir / "scaler.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report.json: {report_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_model.pth: {checkpoint_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler.json: {scaler_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report.get("config", {})
    required = ["data", "train_ids", "val_ids", "feature_groups", "d_model", "dropout", "batch_size"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"report config missing required fields: {missing}")

    ids_key = "train_ids" if split == "train" else "val_ids"
    return ExportPlan(
        run_dir=run_dir,
        split=split,
        out_csv=Path(out_csv),
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,
        data_path=_resolve_report_path(run_dir, config["data"]),
        ids_path=_resolve_report_path(run_dir, config[ids_key]),
        feature_groups=str(config["feature_groups"]),
        d_model=int(config["d_model"]),
        dropout=float(config["dropout"]),
        batch_size=int(config["batch_size"]),
        test_ids_used=False,
    )


def export_split_predictions(plan: ExportPlan, device: str = "cpu") -> list[dict[str, Any]]:
    """Run P4 checkpoint inference for the requested train/val split and write CSV."""
    scaler = json.loads(plan.scaler_path.read_text(encoding="utf-8"))
    match_ids = load_split_ids(str(plan.ids_path))
    rows = load_rows_for_ids(str(plan.data_path), match_ids)
    data = build_p4_dataset(rows, plan.feature_groups)
    data["X"] = apply_p4_feature_scaler(data["X"], scaler)

    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = P4ResidualGoalDiffModel(
        n_features=len(selected_feature_names(plan.feature_groups)),
        d_model=plan.d_model,
        dropout=plan.dropout,
    ).to(torch_device)
    checkpoint = torch.load(plan.checkpoint_path, map_location=torch_device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    preds = predict_outputs(model, data, plan.batch_size, torch_device)
    output_rows = build_prediction_rows(
        match_ids=data["match_ids"],
        y_true=data["y_1x2"],
        y_goal_diff=data["y_goal_diff"],
        p_anchor=data["p_euro_anchor"],
        preds=preds,
    )
    write_prediction_rows(plan.out_csv, output_rows)
    return output_rows


def build_prediction_rows(
    match_ids: list[str],
    y_true: torch.Tensor,
    y_goal_diff: torch.Tensor,
    p_anchor: torch.Tensor,
    preds: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Build CSV-ready prediction rows from P4 tensors."""
    p_final = preds["p_final"].detach().cpu().float()
    p_from_diff = preds["p_from_diff"].detach().cpu().float()
    q_diff = preds["q_diff"].detach().cpu().float()
    labels = y_true.detach().cpu().long()
    goal_labels = y_goal_diff.detach().cpu().long()
    anchors = p_anchor.detach().cpu().float()
    rows = []
    for idx, match_id in enumerate(match_ids):
        pred_class = int(torch.argmax(p_final[idx]).item())
        pred_diff_bucket = int(torch.argmax(q_diff[idx]).item())
        row = {
            "match_id": str(match_id),
            "y_true": int(labels[idx].item()),
            "y_goal_diff": int(goal_labels[idx].item()),
            "p_home": float(p_final[idx, 0].item()),
            "p_draw": float(p_final[idx, 1].item()),
            "p_away": float(p_final[idx, 2].item()),
            "p_anchor_home": float(anchors[idx, 0].item()),
            "p_anchor_draw": float(anchors[idx, 1].item()),
            "p_anchor_away": float(anchors[idx, 2].item()),
            "p_home_from_diff": float(p_from_diff[idx, 0].item()),
            "p_draw_from_diff": float(p_from_diff[idx, 1].item()),
            "p_away_from_diff": float(p_from_diff[idx, 2].item()),
            "pred_class": pred_class,
            "correct": int(pred_class == int(labels[idx].item())),
            "pred_diff_bucket": pred_diff_bucket,
            "correct_diff_bucket": int(pred_diff_bucket == int(goal_labels[idx].item())),
        }
        rows.append({field: row[field] for field in EXPORT_FIELDS})
    return rows


def write_prediction_rows(out_csv: str | Path, rows: list[dict[str, Any]]) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_report_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return run_dir / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export P4 split predictions for P5.2a")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()

    plan = build_export_plan(args.run_dir, args.split, args.out_csv, no_test=args.no_test)
    rows = export_split_predictions(plan, device=args.device)
    print(f"Exported {len(rows)} {plan.split} rows to {plan.out_csv}")
    print("test_ids_used=false")


if __name__ == "__main__":
    main()
