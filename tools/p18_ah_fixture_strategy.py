"""P18.3 Asian-handicap fixture strategy from trained AH-cover heads.

This is an inference/strategy layer, not a new training phase. It loads P18 AH
auxiliary checkpoints, converts the 5-class cover probabilities into expected
upper-side settlement units, and applies fixed confidence thresholds to produce
upper/lower/hold decisions for pre-match prediction exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
from tools.p14_train_market_replication import apply_feature_scaler, p14_selected_feature_market_type_ids, p14_selected_feature_names
from tools.p17_predict_fixture import assert_no_test_paths, build_fixture_dataset, load_prediction_rows


AH_COVER_LABELS = ["upper_full_win", "upper_half_win", "push", "upper_half_loss", "upper_full_loss"]
AH_UNIT_VALUES = [1.0, 0.5, 0.0, -0.5, -1.0]
DEFAULT_THRESHOLDS = [0.20, 0.25]
EPS = 1e-12


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: Iterable[Any]) -> float:
    vals = [_safe_float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def expected_upper_units_from_probs(probs: Iterable[Any]) -> float:
    values = [_safe_float(value) for value in probs]
    if len(values) != len(AH_UNIT_VALUES):
        raise ValueError(f"Expected {len(AH_UNIT_VALUES)} AH cover probabilities, got {len(values)}")
    total = sum(max(value, 0.0) for value in values)
    if total <= EPS:
        values = [1.0 / len(AH_UNIT_VALUES)] * len(AH_UNIT_VALUES)
    else:
        values = [max(value, 0.0) / total for value in values]
    return float(sum(prob * unit for prob, unit in zip(values, AH_UNIT_VALUES)))


def ah_side_from_expected_units(expected_units: float, threshold: float) -> str:
    if expected_units >= float(threshold):
        return "upper"
    if expected_units <= -float(threshold):
        return "lower"
    return "hold"


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _checkpoint_enables_ah_head(config: dict[str, Any], state_dict: dict[str, Any]) -> bool:
    if any(str(key).startswith("ah_cover_head.") for key in state_dict):
        return True
    return any(
        _safe_float(config.get(key)) > 0.0
        for key in ("ah_cover_loss_weight", "ah_unit_loss_weight", "ah_side_loss_weight")
    )


def _checkpoint_enables_ah_unit_head(config: dict[str, Any], state_dict: dict[str, Any]) -> bool:
    if any(str(key).startswith("ah_unit_head.") for key in state_dict):
        return True
    return _safe_float(config.get("ah_direct_unit_loss_weight")) > 0.0


def load_ah_checkpoint_model(run_dir: Path, device: torch.device) -> tuple[P6ResidualPatchITransformer, dict[str, Any], dict[str, Any]]:
    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {}) or {}
    state_dict = checkpoint["model_state_dict"]
    feature_groups = config.get("feature_groups", "euro,asian,ou")
    feature_names = list(checkpoint.get("feature_names") or p14_selected_feature_names(feature_groups))
    market_type_ids = list(checkpoint.get("market_type_ids") or p14_selected_feature_market_type_ids(feature_groups))
    enable_draw_risk = any(str(key).startswith("draw_risk_head.") for key in state_dict)
    enable_ah_cover = _checkpoint_enables_ah_head(config, state_dict)
    enable_ah_unit = _checkpoint_enables_ah_unit_head(config, state_dict)
    model = P6ResidualPatchITransformer(
        n_features=len(feature_names),
        d_model=int(config.get("d_model", 64)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        d_ff=int(config.get("d_ff", 128)),
        dropout=float(config.get("dropout", 0.10)),
        market_type_ids=market_type_ids,
        enable_draw_risk_head=enable_draw_risk,
        enable_ah_cover_head=enable_ah_cover,
        enable_ah_unit_head=enable_ah_unit,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    scaler_path = Path(config.get("scaler_path") or run_dir / "scaler.json")
    if not scaler_path.exists():
        scaler_path = run_dir / "scaler.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    return model, {"checkpoint": checkpoint, "config": config, "feature_names": feature_names}, scaler


@torch.no_grad()
def predict_ah_strategy_rows(
    model: P6ResidualPatchITransformer,
    dataset: dict[str, Any],
    scaler: dict[str, Any],
    seed: int,
    variant: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    x = apply_feature_scaler(dataset["X"].clone(), scaler).to(device)
    out = model(x)
    if "ah_cover_logits" not in out:
        raise ValueError("P18.3 requires checkpoints with ah_cover_logits")
    probs = F.softmax(out["ah_cover_logits"], dim=-1).detach().cpu()
    direct_units = out.get("ah_unit_pred")
    if direct_units is not None:
        direct_units = direct_units.detach().cpu().float().clamp(-1.0, 1.0)
    rows = []
    for idx, bookmaker in enumerate(dataset["bookmakers"]):
        class_probs = [float(probs[idx, cls_idx].item()) for cls_idx in range(len(AH_COVER_LABELS))]
        expected_units = float(direct_units[idx].item()) if direct_units is not None else expected_upper_units_from_probs(class_probs)
        row = {
            "variant": variant,
            "seed": int(seed),
            "match_id": dataset["match_ids"][idx],
            "bookmaker": bookmaker,
            "expected_upper_units": expected_units,
            "abs_expected_upper_units": abs(expected_units),
            "raw_side": "upper" if expected_units > 0.0 else ("lower" if expected_units < 0.0 else "neutral"),
            "expected_units_source": "direct_head" if direct_units is not None else "cover_probs",
        }
        for cls_idx, name in enumerate(AH_COVER_LABELS):
            row[f"p_{name}"] = class_probs[cls_idx]
        row.update(dataset["latest_snapshots"][idx])
        rows.append(row)
    return rows


def summarize_ah_strategy_rows(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    mean_expected = _mean(row.get("expected_upper_units") for row in rows)
    mean_abs = _mean(abs(_safe_float(row.get("expected_upper_units"))) for row in rows)
    mean_line = _mean(row.get("asian_line") for row in rows)
    decisions = []
    for threshold in thresholds:
        decision = ah_side_from_expected_units(mean_expected, threshold)
        decisions.append(
            {
                "threshold": float(threshold),
                "decision": decision,
                "passes_threshold": decision != "hold",
                "mean_expected_upper_units": mean_expected,
                "mean_abs_expected_units": mean_abs,
            }
        )
    return {
        "n_rows": len(rows),
        "mean_expected_upper_units": mean_expected,
        "mean_abs_expected_units": mean_abs,
        "mean_asian_line": mean_line,
        "mean_upper_water": _mean(row.get("upper_water") for row in rows),
        "mean_lower_water": _mean(row.get("lower_water") for row in rows),
        "threshold_decisions": decisions,
    }


def build_ah_strategy_payload(
    input_rows: list[dict[str, Any]],
    strategy_rows: list[dict[str, Any]],
    thresholds: list[float],
    variant: str,
) -> dict[str, Any]:
    first = input_rows[0]
    fixture = {
        "match_id": first.get("match_id"),
        "source_match_id": first.get("source_match_id"),
        "home_team": first.get("home_team"),
        "away_team": first.get("away_team"),
        "kickoff_time_utc": first.get("kickoff_time_utc"),
        "label_status": first.get("label_status"),
        "bookmaker_count": len(input_rows),
    }
    summary = summarize_ah_strategy_rows(strategy_rows, thresholds)
    summary["upper_side_label"] = f"{fixture.get('home_team')} AH upper side"
    summary["lower_side_label"] = f"{fixture.get('away_team')} AH lower side"
    return {
        "phase": "P18.3",
        "task": "pre_match_ah_strategy",
        "variant": variant,
        "fixture": fixture,
        "input_policy": {
            "no_test_split_loaded": True,
            "no_label_fabricated": True,
            "not_betting_advice": True,
            "strategy_layer_only_no_new_training": True,
        },
        "strategy": summary,
        "rows": strategy_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report_md(path: Path, payload: dict[str, Any]) -> None:
    fixture = payload["fixture"]
    strategy = payload["strategy"]
    lines = [
        "# P18.3 AH Fixture Strategy",
        "",
        f"- fixture: {fixture.get('home_team')} vs {fixture.get('away_team')}",
        f"- match_id: `{fixture.get('match_id')}`",
        f"- variant: `{payload.get('variant')}`",
        f"- mean_asian_line: `{strategy['mean_asian_line']:.6f}`",
        f"- expected_upper_units: `{strategy['mean_expected_upper_units']:.6f}`",
        f"- abs_expected_upper_units: `{strategy['mean_abs_expected_units']:.6f}`",
        "",
        "## Decisions",
    ]
    for item in strategy["threshold_decisions"]:
        lines.append(
            f"- threshold {item['threshold']:.3f}: `{item['decision']}` "
            f"(pass={item['passes_threshold']})"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "- This is pre-match inference on unlabeled prediction rows.",
            "- It is a strategy signal, not real-money betting advice.",
            "- `upper` means the home/upper AH side in the exported Asian line; `lower` means the opposite AH side.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P18.3 AH fixture strategy")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-root", default="runs/p18_ah_cover_aux_matrix")
    parser.add_argument("--run-prefix", default="p18_ahw_010_seed")
    parser.add_argument("--variant", default="p18_ahw_010")
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P18.3 AH fixture strategy without --no-test")
    assert_no_test_paths([args.input, args.out_dir])
    out_dir = Path(args.out_dir)
    if not args.allow_overwrite and ((out_dir / "ah_strategy_report.json").exists() or (out_dir / "ah_strategy_report.md").exists()):
        raise SystemExit(f"Refusing to overwrite existing P18.3 AH strategy outputs in {out_dir}")
    input_rows = load_prediction_rows(Path(args.input))
    seeds = _parse_csv_ints(args.seeds)
    thresholds = _parse_csv_floats(args.thresholds)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, Any]] = []
    checkpoint_root = Path(args.checkpoint_root)
    for seed in seeds:
        run_dir = checkpoint_root / f"{args.run_prefix}{seed}"
        model, info, scaler = load_ah_checkpoint_model(run_dir, device)
        dataset = build_fixture_dataset(input_rows, list(info["feature_names"]))
        all_rows.extend(predict_ah_strategy_rows(model, dataset, scaler, seed, args.variant, device))
    payload = build_ah_strategy_payload(input_rows, all_rows, thresholds, args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "ah_strategy_by_seed.csv", all_rows)
    (out_dir / "ah_strategy_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report_md(out_dir / "ah_strategy_report.md", payload)
    print(
        "P18.3 AH strategy: "
        f"expected_upper_units={payload['strategy']['mean_expected_upper_units']:.6f}, "
        f"decisions={[(item['threshold'], item['decision']) for item in payload['strategy']['threshold_decisions']]}"
    )
    print(f"Report saved to {out_dir / 'ah_strategy_report.md'}")


if __name__ == "__main__":
    main()
