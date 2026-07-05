"""Export P18 AH-head predictions for P19 rolling folds.

This is an evaluation-only helper. A rolling fold trains once on the fold train
window, selects a checkpoint without using forward rows, then uses this script to
export selection and forward predictions from the frozen checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p14_train_market_replication import apply_feature_scaler, write_market_replication_csv, write_val_predictions_csv
from tools.p18_ah_fixture_strategy import load_ah_checkpoint_model
from tools.p18_train_ah_cover_aux import build_p18_dataset, evaluate, write_ah_cover_predictions_csv
from tools.p19_rolling_backtest import assert_no_test_paths
from tools.p4_train_residual_goal_diff import load_rows_for_ids, load_split_ids


def prepare_export_output_dir(out_dir: str | Path, allow_overwrite: bool = False) -> None:
    path = Path(out_dir)
    outputs = [
        path / "val_predictions.csv",
        path / "val_market_replication.csv",
        path / "val_ah_cover_predictions.csv",
        path / "export_report.json",
        path / "export_report.md",
    ]
    if not allow_overwrite and any(item.exists() for item in outputs):
        raise FileExistsError(f"Refusing to overwrite existing P19 export outputs in {path}")
    path.mkdir(parents=True, exist_ok=True)


def build_export_report_payload(
    split_name: str,
    checkpoint_dir: str,
    ids_path: str,
    row_count: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "P19",
        "task": "export_ah_predictions_for_rolling_fold",
        "split_name": split_name,
        "checkpoint_dir": checkpoint_dir,
        "ids_path": ids_path,
        "row_count": int(row_count),
        "metrics": metrics,
        "input_policy": {
            "no_test_split_loaded": True,
            "no_training": True,
            "checkpoint_already_frozen": True,
        },
    }


def write_report_md(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics", {})
    lines = [
        "# P19 AH Prediction Export",
        "",
        f"- split: `{payload.get('split_name')}`",
        f"- checkpoint: `{payload.get('checkpoint_dir')}`",
        f"- ids: `{payload.get('ids_path')}`",
        f"- rows: `{payload.get('row_count')}`",
        f"- logloss: `{metrics.get('logloss')}`",
        f"- ah_cover_acc: `{metrics.get('ah_cover_acc')}`",
        f"- ah_direct_unit_mae: `{metrics.get('ah_direct_unit_mae')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    assert_no_test_paths([args.data, args.ids, args.checkpoint_dir, args.out_dir])
    prepare_export_output_dir(args.out_dir, args.allow_overwrite)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, info, scaler = load_ah_checkpoint_model(Path(args.checkpoint_dir), device)
    feature_groups = info.get("config", {}).get("feature_groups", "euro,asian,ou")
    rows = load_rows_for_ids(args.data, load_split_ids(args.ids))
    data = build_p18_dataset(rows, feature_groups)
    if list(data["feature_names"]) != list(info["feature_names"]):
        raise ValueError(f"Feature names mismatch for {args.checkpoint_dir}")
    data["X"] = apply_feature_scaler(data["X"], scaler)
    val_metrics, market_metrics, goal_diff_metrics, ah_metrics, preds = evaluate(model, data, args.batch_size, device)
    out_dir = Path(args.out_dir)
    write_val_predictions_csv(out_dir / "val_predictions.csv", data["match_ids"], data["y_1x2"], preds["p_final"])
    write_market_replication_csv(out_dir / "val_market_replication.csv", data, preds)
    write_ah_cover_predictions_csv(out_dir / "val_ah_cover_predictions.csv", data, preds)
    metrics = {
        **{key: _json_safe(value) for key, value in val_metrics.items()},
        **{key: _json_safe(value) for key, value in market_metrics.items()},
        **{key: _json_safe(value) for key, value in goal_diff_metrics.items()},
        **{key: _json_safe(value) for key, value in ah_metrics.items()},
    }
    payload = build_export_report_payload(args.split_name, args.checkpoint_dir, args.ids, len(data["match_ids"]), metrics)
    (out_dir / "export_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report_md(out_dir / "export_report.md", payload)
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export P18 AH predictions for a P19 split")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-name", default="forward")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    payload = export_predictions(build_arg_parser().parse_args())
    print(
        "P19 export: "
        f"split={payload['split_name']} rows={payload['row_count']} "
        f"logloss={payload['metrics'].get('logloss')} "
        f"ah_acc={payload['metrics'].get('ah_cover_acc')}"
    )


if __name__ == "__main__":
    main()
