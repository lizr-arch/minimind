"""P18.0 Asian-handicap cover diagnostics for frozen mainline checkpoints.

This phase does not train a new model. It uses existing P14/P16-style
checkpoints, derives coarse cover probabilities from the goal-diff head, and
compares those directional calls to canonical Asian-handicap settlement on the
validation split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.asian_handicap import settle_asian_5class
from tools.p14_train_market_replication import apply_feature_scaler, build_p14_dataset, predict_outputs
from tools.p17_predict_fixture import load_checkpoint_model
from tools.p4_train_residual_goal_diff import load_rows_for_ids, load_split_ids


GOAL_DIFF_BUCKETS = (
    ("away_by_3plus", -3),
    ("away_by_2", -2),
    ("away_by_1", -1),
    ("draw", 0),
    ("home_by_1", 1),
    ("home_by_2", 2),
    ("home_by_3plus", 3),
)
AH_UNIT_MAP = {
    "upper_full_win": 1.0,
    "upper_half_win": 0.5,
    "push": 0.0,
    "upper_half_loss": -0.5,
    "upper_full_loss": -1.0,
}
P18_SLICES = ("all", "favorite_le_0_50", "favorite_0_75_to_1_50", "favorite_ge_1_75", "flat")
EPS = 1e-12


class P18ProtocolError(RuntimeError):
    pass


def _contains_test_path(path: str | Path) -> bool:
    parts = [part.lower() for part in Path(path).parts]
    return any(part == "test" or part.startswith("test_") or part.startswith("test-") for part in parts)


def assert_no_test_paths(paths: Iterable[str | Path]) -> None:
    for path in paths:
        if path and _contains_test_path(path):
            raise P18ProtocolError(f"P18 refuses test split or test artifact path: {path}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def _is_present_asian(event: dict[str, Any]) -> bool:
    if "asian_source" in event:
        return event.get("asian_source") in {"raw_update", "forward_fill"}
    if "has_asian" in event:
        return bool(event.get("has_asian"))
    upper = _safe_float(event.get("upper_water"))
    lower = _safe_float(event.get("lower_water"))
    return upper > 0.0 and lower > 0.0


def latest_pre_kickoff_asian_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    timeline = row.get("raw_timeline", []) or row.get("odds_timeline", []) or []
    valid: list[tuple[float, dict[str, Any]]] = []
    negative_time_valid_asian_count = 0
    for event in timeline:
        minutes = _safe_float(event.get("minutes_before_kickoff"), default=-1.0)
        line = _safe_float(event.get("asian_line"), default=float("nan"))
        upper = _safe_float(event.get("upper_water"), default=0.0)
        lower = _safe_float(event.get("lower_water"), default=0.0)
        if not math.isfinite(line) or upper <= 0.0 or lower <= 0.0 or not _is_present_asian(event):
            continue
        snapshot = {
            "minutes_before_kickoff": minutes,
            "asian_line": line,
            "upper_water": upper,
            "lower_water": lower,
        }
        if minutes < 0:
            negative_time_valid_asian_count += 1
            continue
        valid.append((minutes, snapshot))
    if not valid:
        return {
            "valid_asian": False,
            "minutes_before_kickoff": None,
            "asian_line": None,
            "upper_water": None,
            "lower_water": None,
            "negative_time_valid_asian_count": negative_time_valid_asian_count,
        }
    _, snapshot = min(valid, key=lambda item: item[0])
    snapshot["valid_asian"] = True
    snapshot["negative_time_valid_asian_count"] = negative_time_valid_asian_count
    return snapshot


def settle_upper_units(home_goals: int, away_goals: int, asian_line: float) -> float:
    result = settle_asian_5class(int(home_goals), int(away_goals), float(asian_line))
    return AH_UNIT_MAP[result]


def _unit_to_bucket(unit: float) -> str:
    if unit >= 1.0:
        return "full_win"
    if unit > 0.0:
        return "half_win"
    if unit == 0.0:
        return "push"
    if unit <= -1.0:
        return "full_loss"
    return "half_loss"


def expected_upper_units_from_goal_diff_probs(q_goal_diff: dict[str, float] | list[float] | tuple[float, ...], asian_line: float) -> dict[str, Any]:
    if isinstance(q_goal_diff, dict):
        probs = [_safe_float(q_goal_diff.get(name)) for name, _ in GOAL_DIFF_BUCKETS]
    else:
        probs = [_safe_float(value) for value in q_goal_diff]
    total = sum(max(value, 0.0) for value in probs)
    if total <= EPS:
        probs = [1.0 / len(GOAL_DIFF_BUCKETS)] * len(GOAL_DIFF_BUCKETS)
    else:
        probs = [max(value, 0.0) / total for value in probs]
    units_by_bucket = []
    bucket_probs = {"upper_full_win_prob": 0.0, "upper_half_win_prob": 0.0, "push_prob": 0.0, "upper_half_loss_prob": 0.0, "upper_full_loss_prob": 0.0}
    for (name, margin), prob in zip(GOAL_DIFF_BUCKETS, probs):
        unit = settle_upper_units(margin, 0, asian_line)
        units_by_bucket.append({"bucket": name, "representative_margin": margin, "prob": prob, "upper_units": unit})
        bucket = _unit_to_bucket(unit)
        if bucket == "full_win":
            bucket_probs["upper_full_win_prob"] += prob
        elif bucket == "half_win":
            bucket_probs["upper_half_win_prob"] += prob
        elif bucket == "push":
            bucket_probs["push_prob"] += prob
        elif bucket == "half_loss":
            bucket_probs["upper_half_loss_prob"] += prob
        else:
            bucket_probs["upper_full_loss_prob"] += prob
    upper_win_prob = bucket_probs["upper_full_win_prob"] + bucket_probs["upper_half_win_prob"]
    upper_loss_prob = bucket_probs["upper_half_loss_prob"] + bucket_probs["upper_full_loss_prob"]
    expected_units = sum(item["prob"] * item["upper_units"] for item in units_by_bucket)
    return {
        "expected_upper_units": float(expected_units),
        "upper_win_prob": float(upper_win_prob),
        "upper_loss_prob": float(upper_loss_prob),
        **{key: float(value) for key, value in bucket_probs.items()},
        "coarse_bucket_warning": abs(float(asian_line)) >= 3.0,
        "units_by_bucket": units_by_bucket,
    }


def _slice_for_line(asian_line: float) -> str:
    abs_line = abs(float(asian_line))
    if abs_line <= 1e-9:
        return "flat"
    if abs_line <= 0.50:
        return "favorite_le_0_50"
    if abs_line <= 1.50:
        return "favorite_0_75_to_1_50"
    return "favorite_ge_1_75"


def _model_side(expected_units: float) -> str:
    if expected_units > 1e-9:
        return "upper"
    if expected_units < -1e-9:
        return "lower"
    return "neutral"


def summarize_ah_rows(rows: list[dict[str, Any]], seed: int | str, slice_name: str) -> dict[str, Any]:
    selected = [row for row in rows if slice_name == "all" or row.get("slice") == slice_name]
    n = len(selected)
    actual_units = [_safe_float(row.get("actual_upper_units")) for row in selected]
    expected_units = [_safe_float(row.get("expected_upper_units")) for row in selected]
    model_side_units = []
    direction_hits = []
    upper_picks = 0
    lower_picks = 0
    neutral_picks = 0
    for row in selected:
        side = row.get("predicted_side")
        actual = _safe_float(row.get("actual_upper_units"))
        if side == "upper":
            model_side_units.append(actual)
            upper_picks += 1
        elif side == "lower":
            model_side_units.append(-actual)
            lower_picks += 1
        else:
            model_side_units.append(0.0)
            neutral_picks += 1
        if actual > 0 and side == "upper":
            direction_hits.append(1.0)
        elif actual < 0 and side == "lower":
            direction_hits.append(1.0)
        elif actual != 0 and side in {"upper", "lower"}:
            direction_hits.append(0.0)
    return {
        "seed": seed,
        "slice": slice_name,
        "n": n,
        "mean_line": _mean([_safe_float(row.get("asian_line")) for row in selected]),
        "mean_actual_upper_units": _mean(actual_units),
        "mean_expected_upper_units": _mean(expected_units),
        "upper_baseline_avg_units": _mean(actual_units),
        "lower_baseline_avg_units": -_mean(actual_units),
        "model_side_avg_units": _mean(model_side_units),
        "direction_accuracy_ex_push": _mean(direction_hits),
        "upper_pick_rate": upper_picks / n if n else 0.0,
        "lower_pick_rate": lower_picks / n if n else 0.0,
        "neutral_pick_rate": neutral_picks / n if n else 0.0,
        "actual_upper_positive_rate": sum(1 for value in actual_units if value > 0) / n if n else 0.0,
        "actual_push_rate": sum(1 for value in actual_units if value == 0) / n if n else 0.0,
        "coarse_bucket_warning_count": sum(1 for row in selected if row.get("coarse_bucket_warning")),
    }


def aggregate_seed_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["slice"]), []).append(row)
    aggregate = []
    for slice_name, group in sorted(grouped.items()):
        item = {
            "slice": slice_name,
            "seed_count": len(group),
        }
        for key in (
            "n",
            "mean_line",
            "mean_actual_upper_units",
            "mean_expected_upper_units",
            "upper_baseline_avg_units",
            "lower_baseline_avg_units",
            "model_side_avg_units",
            "direction_accuracy_ex_push",
            "upper_pick_rate",
            "lower_pick_rate",
            "actual_upper_positive_rate",
            "actual_push_rate",
            "coarse_bucket_warning_count",
        ):
            values = [_safe_float(row.get(key)) for row in group]
            item[f"mean_{key}"] = _mean(values)
            if key in {"model_side_avg_units", "direction_accuracy_ex_push"}:
                item[f"stdev_{key}"] = _stdev(values)
        aggregate.append(item)
    return aggregate


def build_report_payload(by_seed: list[dict[str, Any]], aggregate: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    all_row = next((row for row in aggregate if row.get("slice") == "all"), {})
    verdict = "P18_AH_DIAGNOSTIC_READY" if int(all_row.get("seed_count", 0) or 0) >= 1 and float(all_row.get("mean_n", 0.0) or 0.0) > 0 else "P18_AH_DIAGNOSTIC_INSUFFICIENT"
    return {
        "phase": "P18.0",
        "metadata": metadata,
        "input_policy": {
            "no_test_split_loaded": True,
            "no_new_model_family": True,
            "diagnostic_only": True,
            "goal_diff_bucket_note": "AH expected units use 7 coarse goal-diff buckets; >=3 and <=-3 are representative buckets.",
        },
        "summary_by_seed": by_seed,
        "summary_aggregate": aggregate,
        "verdict": verdict,
        "next_actions": [
            "If current goal-diff-derived AH direction is weak, add a small AH auxiliary head in P18.1.",
            "Keep P14/P16 1X2 market-replication metrics as the mainline guardrail.",
            "Evaluate AH slices separately: flat/small favorite, mid favorite, and big favorite.",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid_labelled_context(row: dict[str, Any]) -> bool:
    label = row.get("label", {}) or {}
    return "home_goals" in label and "away_goals" in label and label.get("euro_result") in {"home", "draw", "away"}


def build_source_context_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    context = []
    skipped = 0
    label_idx = 0
    for row_idx, row in enumerate(rows):
        if not _valid_labelled_context(row):
            skipped += 1
            continue
        label = row.get("label", {}) or {}
        snapshot = latest_pre_kickoff_asian_snapshot(row)
        if not snapshot.get("valid_asian"):
            label_idx += 1
            skipped += 1
            continue
        home_goals = int(label["home_goals"])
        away_goals = int(label["away_goals"])
        line = float(snapshot["asian_line"])
        actual = settle_upper_units(home_goals, away_goals, line)
        context.append(
            {
                "row_idx": row_idx,
                "label_idx": label_idx,
                "match_id": str(row.get("match_id", row_idx)),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "asian_line": line,
                "upper_water": snapshot.get("upper_water"),
                "lower_water": snapshot.get("lower_water"),
                "asian_minutes_before_kickoff": snapshot.get("minutes_before_kickoff"),
                "negative_time_valid_asian_count": snapshot.get("negative_time_valid_asian_count", 0),
                "actual_upper_units": actual,
                "actual_upper_result": _unit_to_bucket(actual),
                "slice": _slice_for_line(line),
            }
        )
        label_idx += 1
    return context


def _q_row_dict(q_values: list[float]) -> dict[str, float]:
    return {name: float(value) for (name, _), value in zip(GOAL_DIFF_BUCKETS, q_values)}


def run_seed_diagnostics(
    run_dir: Path,
    source_rows: list[dict[str, Any]],
    seed: int,
    device: torch.device,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    model, info, scaler = load_checkpoint_model(run_dir, device)
    feature_groups = info.get("config", {}).get("feature_groups", "euro,asian,ou")
    data = build_p14_dataset(source_rows, feature_groups)
    if list(data["feature_names"]) != list(info["feature_names"]):
        raise ValueError(f"Feature names mismatch for {run_dir}")
    data["X"] = apply_feature_scaler(data["X"], scaler)
    preds = predict_outputs(model, data, batch_size, device)
    q = F.softmax(preds["q_diff"], dim=-1).detach().cpu()
    p_final = preds["p_final"].detach().cpu()
    context = build_source_context_rows(source_rows)
    if len(data["match_ids"]) != q.shape[0]:
        raise ValueError(f"P18 prediction tensor mismatch for {run_dir}: data={len(data['match_ids'])}, q={q.shape[0]}")
    rows = []
    for ctx in context:
        idx = int(ctx["label_idx"])
        if idx < 0 or idx >= q.shape[0]:
            raise ValueError(f"P18 context label index out of range for {run_dir}: {idx} / {q.shape[0]}")
        q_dict = _q_row_dict(q[idx].tolist())
        expected = expected_upper_units_from_goal_diff_probs(q_dict, float(ctx["asian_line"]))
        side = _model_side(float(expected["expected_upper_units"]))
        row = {
            "seed": seed,
            **ctx,
            "p_home": float(p_final[idx, 0].item()),
            "p_draw": float(p_final[idx, 1].item()),
            "p_away": float(p_final[idx, 2].item()),
            **q_dict,
            "expected_upper_units": expected["expected_upper_units"],
            "upper_win_prob": expected["upper_win_prob"],
            "upper_loss_prob": expected["upper_loss_prob"],
            "upper_full_win_prob": expected["upper_full_win_prob"],
            "upper_half_win_prob": expected["upper_half_win_prob"],
            "push_prob": expected["push_prob"],
            "upper_half_loss_prob": expected["upper_half_loss_prob"],
            "upper_full_loss_prob": expected["upper_full_loss_prob"],
            "predicted_side": side,
            "model_side_actual_units": float(ctx["actual_upper_units"]) if side == "upper" else (-float(ctx["actual_upper_units"]) if side == "lower" else 0.0),
            "coarse_bucket_warning": bool(expected["coarse_bucket_warning"]),
        }
        rows.append(row)
    return rows


def write_report_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# P18.0 AH Cover Diagnostics", "", f"Verdict: `{payload.get('verdict')}`", ""]
    lines.append("## Aggregate")
    for row in payload.get("summary_aggregate", []):
        lines.append(
            f"- {row.get('slice')}: n={row.get('mean_n')}, "
            f"model_units={float(row.get('mean_model_side_avg_units', 0.0)):.6f}, "
            f"direction_acc={float(row.get('mean_direction_accuracy_ex_push', 0.0)):.6f}, "
            f"upper_pick={float(row.get('mean_upper_pick_rate', 0.0)):.6f}, "
            f"upper_base={float(row.get('mean_upper_baseline_avg_units', 0.0)):.6f}, "
            f"lower_base={float(row.get('mean_lower_baseline_avg_units', 0.0)):.6f}"
        )
    lines.extend(["", "## Guardrails"])
    for key, value in payload.get("input_policy", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Actions"])
    for action in payload.get("next_actions", []):
        lines.append(f"- {action}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P18.0 AH cover diagnostics")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--checkpoint-root", default="runs/p14_market_replication")
    parser.add_argument("--run-prefix", default="p14_market_kl_030_balanced_seed")
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--out-dir", default="runs/p18_ah_cover_diagnostics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P18.0 without --no-test")
    assert_no_test_paths([args.data, args.val_ids, args.out_dir])
    out_dir = Path(args.out_dir)
    if not args.allow_overwrite and ((out_dir / "p18_report.json").exists() or (out_dir / "p18_report.md").exists()):
        raise SystemExit(f"Refusing to overwrite existing P18 outputs in {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    val_ids = load_split_ids(args.val_ids)
    source_rows = load_rows_for_ids(args.data, val_ids)
    seeds = _parse_int_list(args.seeds)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    all_detail_rows: list[dict[str, Any]] = []
    by_seed: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = Path(args.checkpoint_root) / f"{args.run_prefix}{seed}"
        detail_rows = run_seed_diagnostics(run_dir, source_rows, seed, device)
        all_detail_rows.extend(detail_rows)
        for slice_name in P18_SLICES:
            by_seed.append(summarize_ah_rows(detail_rows, seed=seed, slice_name=slice_name))
    aggregate = aggregate_seed_summaries(by_seed)
    metadata = {
        "phase": "P18.0",
        "data": args.data,
        "val_ids": args.val_ids,
        "checkpoint_root": args.checkpoint_root,
        "run_prefix": args.run_prefix,
        "seeds": seeds,
        "run_count": len(seeds),
        "source_val_rows": len(source_rows),
        "diagnostic_rows": len(all_detail_rows),
    }
    payload = build_report_payload(by_seed, aggregate, metadata)
    write_csv(out_dir / "p18_ah_cover_detail.csv", all_detail_rows)
    write_csv(out_dir / "p18_ah_cover_summary_by_seed.csv", by_seed)
    write_csv(out_dir / "p18_ah_cover_summary.csv", aggregate)
    (out_dir / "p18_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report_md(out_dir / "p18_report.md", payload)
    print(f"P18.0 verdict: {payload['verdict']}")
    print(f"Report saved to {out_dir / 'p18_report.md'}")


if __name__ == "__main__":
    main()
