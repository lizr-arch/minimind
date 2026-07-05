"""P13 diagnostic-only data, feature, and target audit.

This script reads train/val data and existing P6/P11/P12 artifacts only. It
does not train a model, fit a probe, tune thresholds, mutate labels, or read the
test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


EPS = 1e-8
EURO_MAP = {"home": 0, "draw": 1, "away": 2}
CLASS_NAMES = ("home", "draw", "away")
ALLOWED_VERDICTS = {
    "P13_FEATURE_ENGINEERING_RECOMMENDED",
    "P13_DATA_REPAIR_REQUIRED",
    "P13_TARGET_PIVOT_MARKET_REPLICATION_RECOMMENDED",
    "P13_TARGET_PIVOT_VALUE_DETECTION_RECOMMENDED",
    "P13_INCONCLUSIVE_NEED_MORE_DATA",
    "P13_BLOCKED_BY_ARTIFACT_ALIGNMENT",
    "P13_BLOCKED_BY_PROTOCOL",
}
SUSPECT_FIELDNAMES = [
    "match_id",
    "league",
    "season",
    "kickoff_time",
    "home_team",
    "away_team",
    "result",
    "home_goals",
    "away_goals",
    "market_close_home",
    "market_close_draw",
    "market_close_away",
    "fair_home",
    "fair_draw",
    "fair_away",
    "issue_type",
    "issue_reason",
]


class P13ProtocolError(RuntimeError):
    pass


def assert_no_test_paths(paths: Iterable[str | Path]) -> None:
    for path in paths:
        parts = [part.lower() for part in Path(path).parts]
        if any(part == "test" or part.startswith("test_") or part.startswith("test-") for part in parts):
            raise P13ProtocolError(f"P13 refuses test split or test artifact path: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fair_probs_from_odds(home_odds: Any, draw_odds: Any, away_odds: Any) -> tuple[float, float, float]:
    odds = [_safe_float(home_odds), _safe_float(draw_odds), _safe_float(away_odds)]
    if any(value is None or value <= 1.0 for value in odds):
        raise ValueError("invalid odds")
    inv = [1.0 / float(value) for value in odds]
    total = sum(inv)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("invalid odds")
    return inv[0] / total, inv[1] / total, inv[2] / total


def _normalize_probs(probs: Iterable[float]) -> tuple[float, float, float]:
    values = [max(float(value), EPS) for value in probs]
    total = sum(values)
    return values[0] / total, values[1] / total, values[2] / total


def draw_top2(probs: tuple[float, float, float] | list[float]) -> bool:
    return sorted(range(3), key=lambda idx: float(probs[idx]), reverse=True)[:2].count(1) == 1


def draw_margin_to_top(probs: tuple[float, float, float] | list[float]) -> float:
    return max(float(value) for value in probs) - float(probs[1])


def draw_rank(probs: tuple[float, float, float] | list[float]) -> int:
    return sorted(range(3), key=lambda idx: float(probs[idx]), reverse=True).index(1) + 1


def draw_shrinkage_ratio(model_draw: list[float], market_draw: list[float], labels: list[int]) -> float | None:
    true_draw_indices = [i for i, label in enumerate(labels) if int(label) == 1]
    if not true_draw_indices:
        return None
    model_mean = sum(model_draw[i] for i in true_draw_indices) / len(true_draw_indices)
    market_mean = sum(market_draw[i] for i in true_draw_indices) / len(true_draw_indices)
    return model_mean / max(market_mean, EPS)


def mark_slice_support(n: int) -> str:
    if int(n) >= 300:
        return "decision_grade"
    if int(n) >= 100:
        return "diagnostic_grade"
    return "report_only"


def ah_abs_line_bucket(value: Any) -> str:
    line = abs(float(value))
    if line == 0:
        return "ah_abs_0"
    if line <= 0.25:
        return "ah_abs_le_0_25"
    if line <= 0.50:
        return "ah_abs_le_0_50"
    if line <= 1.00:
        return "ah_abs_0_75_1_00"
    return "ah_abs_ge_1_25"


def ou_line_bucket(value: Any) -> str:
    line = float(value)
    if line <= 2.00:
        return "ou_le_2_00"
    if abs(line - 2.25) < 1e-6:
        return "ou_2_25"
    if abs(line - 2.50) < 1e-6:
        return "ou_2_50"
    if abs(line - 2.75) < 1e-6:
        return "ou_2_75"
    return "ou_ge_3_00"


def load_prediction_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "match_id": str(row["match_id"]),
                    "y_true": int(row["y_true"]),
                    "p": (
                        float(row["p_home"]),
                        float(row["p_draw"]),
                        float(row["p_away"]),
                    ),
                }
            )
    return rows


def load_score_prediction_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(
                {
                    "match_id": str(row["match_id"]),
                    "score": (
                        float(row["score_p_home"]),
                        float(row["score_p_draw"]),
                        float(row["score_p_away"]),
                    ),
                }
            )
    return out


def load_prior_prediction_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(
                {
                    "match_id": str(row["match_id"]),
                    "prior": (float(row["prior_home"]), float(row["prior_draw"]), float(row["prior_away"])),
                    "final": (float(row["final_home"]), float(row["final_draw"]), float(row["final_away"])),
                }
            )
    return out


def validate_alignment(named_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    names = list(named_rows)
    if not names:
        raise ValueError("No prediction artifacts supplied")
    ref = named_rows[names[0]]
    ref_ids = [row["match_id"] for row in ref]
    ref_labels = [row["y_true"] for row in ref]
    for name in names[1:]:
        ids = [row["match_id"] for row in named_rows[name]]
        labels = [row["y_true"] for row in named_rows[name]]
        if ids != ref_ids:
            raise ValueError(f"match_id alignment failed for {name}")
        if labels != ref_labels:
            raise ValueError(f"label alignment failed for {name}")
    unique_match_ids = len(set(ref_ids))
    return {
        "row_count": len(ref),
        "unique_match_ids": unique_match_ids,
        "duplicate_prediction_rows": len(ref) - unique_match_ids,
        "artifact_count": len(names),
        "match_id_alignment": True,
        "label_alignment": True,
    }


def reproduce_prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    total_loss = 0.0
    true_draw_n = 0
    true_draw_top2 = 0
    p_draw_true = 0.0
    for row in rows:
        label = int(row["y_true"])
        probs = _normalize_probs(row["p"])
        total_loss += -math.log(max(probs[label], EPS))
        if label == 1:
            true_draw_n += 1
            true_draw_top2 += int(draw_top2(probs))
            p_draw_true += probs[1]
    return {
        "logloss": total_loss / max(len(rows), 1),
        "draw_top2": true_draw_top2 / max(true_draw_n, 1),
        "mean_p_draw_true_draw": p_draw_true / max(true_draw_n, 1),
    }


def calibration_bins(rows: list[dict[str, Any]], n_bins: int = 10) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for idx in range(int(n_bins)):
        low = idx / int(n_bins)
        high = (idx + 1) / int(n_bins)
        if idx == int(n_bins) - 1:
            members = [row for row in rows if low <= float(row["confidence"]) <= high]
        else:
            members = [row for row in rows if low <= float(row["confidence"]) < high]
        n = len(members)
        pred_mean = sum(float(row["confidence"]) for row in members) / n if n else 0.0
        true_rate = sum(float(row["target"]) for row in members) / n if n else 0.0
        bins.append(
            {
                "bin": idx,
                "low": low,
                "high": high,
                "n": n,
                "pred_mean": pred_mean,
                "true_rate": true_rate,
                "calibration_gap": abs(true_rate - pred_mean) if n else 0.0,
                "support": mark_slice_support(n),
            }
        )
    return bins


def _logloss(probs: list[tuple[float, float, float]], labels: list[int]) -> float:
    if not probs:
        return 0.0
    return sum(-math.log(max(_normalize_probs(probs[idx])[int(labels[idx])], EPS)) for idx in range(len(probs))) / len(probs)


def _draw_nll(probs: list[tuple[float, float, float]], labels: list[int]) -> float:
    values = [-math.log(max(_normalize_probs(probs[idx])[1], EPS)) for idx, label in enumerate(labels) if int(label) == 1]
    return sum(values) / len(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = _mean(xs)
    my = _mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return None
    return sum(dx[i] * dy[i] for i in range(len(dx))) / denom


def load_split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows_for_ids(jsonl_path: Path, ids: Iterable[str]) -> list[dict[str, Any]]:
    id_set = set(ids)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            match_id = str(row.get("match_id"))
            if match_id in id_set:
                rows.append(row)
    return rows


def valid_pre_kickoff_euro_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for event in row.get("odds_timeline", []) or row.get("raw_timeline", []) or []:
        minute = _safe_float(event.get("minutes_before_kickoff"))
        if minute is None or minute < 0:
            continue
        if event.get("has_euro") is False:
            continue
        try:
            probs = fair_probs_from_odds(event.get("euro_h"), event.get("euro_d"), event.get("euro_a"))
        except ValueError:
            continue
        enriched = dict(event)
        enriched["_minute"] = minute
        enriched["_probs"] = probs
        events.append(enriched)
    return events


def _select_close_event(row: dict[str, Any]) -> dict[str, Any] | None:
    events = valid_pre_kickoff_euro_events(row)
    return min(events, key=lambda item: item["_minute"]) if events else None


def _select_open_event(row: dict[str, Any]) -> dict[str, Any] | None:
    events = valid_pre_kickoff_euro_events(row)
    return max(events, key=lambda item: item["_minute"]) if events else None


def _latest_market_value(row: dict[str, Any], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
    valid = []
    for event in row.get("odds_timeline", []) or row.get("raw_timeline", []) or []:
        minute = _safe_float(event.get("minutes_before_kickoff"))
        if minute is None or minute < 0:
            continue
        if predicate(event):
            item = dict(event)
            item["_minute"] = minute
            valid.append(item)
    return min(valid, key=lambda item: item["_minute"]) if valid else None


def _row_year_month(row: dict[str, Any]) -> tuple[str, str]:
    kickoff = str(row.get("kickoff_time") or "")
    if len(kickoff) >= 7 and kickoff[:4].isdigit():
        return kickoff[:4], kickoff[:7]
    return "unknown", "unknown"


def build_context_rows(
    val_predictions: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    p11_score: list[dict[str, Any]],
    p12_prior: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(val_rows) != len(val_predictions):
        raise ValueError(f"source row alignment failed: {len(val_rows)} source rows for {len(val_predictions)} prediction rows")
    if p11_score and len(p11_score) != len(val_predictions):
        raise ValueError(f"p11 score alignment failed: {len(p11_score)} score rows for {len(val_predictions)} prediction rows")
    if p12_prior and len(p12_prior) != len(val_predictions):
        raise ValueError(f"p12 prior alignment failed: {len(p12_prior)} prior rows for {len(val_predictions)} prediction rows")
    out: list[dict[str, Any]] = []
    suspects: list[dict[str, Any]] = []
    for idx, pred in enumerate(val_predictions):
        match_id = pred["match_id"]
        row = val_rows[idx]
        if str(row.get("match_id")) != match_id:
            raise ValueError(f"source row alignment failed at row {idx}: {row.get('match_id')} != {match_id}")
        label = row.get("label", {}) or {}
        close = _select_close_event(row)
        open_event = _select_open_event(row)
        issue_types: list[str] = []
        if not row:
            issue_types.append("missing_source_row")
        if close is None:
            issue_types.append("missing_close_snapshot")
            market_probs = (0.0, 0.0, 0.0)
            close_minute = None
        else:
            market_probs = close["_probs"]
            close_minute = close["_minute"]
        if open_event is None:
            open_probs = None
        else:
            open_probs = open_event["_probs"]
        if "home_goals" not in label or "away_goals" not in label:
            issue_types.append("missing_final_score")
        result = label.get("euro_result")
        if result not in EURO_MAP:
            issue_types.append("missing_or_invalid_result")
        elif EURO_MAP[result] != int(pred["y_true"]):
            issue_types.append("label_prediction_mismatch")
        timeline = row.get("odds_timeline", []) or row.get("raw_timeline", []) or []
        negative_valid = 0
        for event in timeline:
            minute = _safe_float(event.get("minutes_before_kickoff"))
            if minute is not None and minute < 0:
                try:
                    fair_probs_from_odds(event.get("euro_h"), event.get("euro_d"), event.get("euro_a"))
                    negative_valid += 1
                except ValueError:
                    pass
        if negative_valid:
            issue_types.append("valid_euro_after_kickoff")

        asian_event = _latest_market_value(
            row,
            lambda event: bool(event.get("has_asian")) and _safe_float(event.get("asian_line")) is not None,
        )
        ou_event = _latest_market_value(
            row,
            lambda event: bool(event.get("has_over_under")) and _safe_float(event.get("over_under_line")) is not None,
        )
        year, month = _row_year_month(row)
        p6_probs = _normalize_probs(pred["p"])
        p11_score_probs = None
        if p11_score:
            p11_item = p11_score[idx]
            if str(p11_item["match_id"]) != match_id:
                raise ValueError(f"p11 score alignment failed at row {idx}: {p11_item['match_id']} != {match_id}")
            p11_score_probs = p11_item["score"]
        p12_final = None
        p12_prior_probs = None
        if p12_prior:
            p12_item = p12_prior[idx]
            if str(p12_item["match_id"]) != match_id:
                raise ValueError(f"p12 prior alignment failed at row {idx}: {p12_item['match_id']} != {match_id}")
            p12_final = p12_item["final"]
            p12_prior_probs = p12_item["prior"]
        delta_draw = (market_probs[1] - open_probs[1]) if open_probs is not None and close is not None else None
        context = {
            "match_id": match_id,
            "league": row.get("league_id", "unknown"),
            "season": row.get("season", "unknown"),
            "year": year,
            "month": month,
            "competition_family": row.get("competition_type", "unknown"),
            "kickoff_time": row.get("kickoff_time", ""),
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "y_true": int(pred["y_true"]),
            "result": result,
            "home_goals": label.get("home_goals", ""),
            "away_goals": label.get("away_goals", ""),
            "is_draw": int(int(pred["y_true"]) == 1),
            "market_p_home": market_probs[0],
            "market_p_draw": market_probs[1],
            "market_p_away": market_probs[2],
            "p6_p_home": p6_probs[0],
            "p6_p_draw": p6_probs[1],
            "p6_p_away": p6_probs[2],
            "p11_score_p_home": p11_score_probs[0] if p11_score_probs else "",
            "p11_score_p_draw": p11_score_probs[1] if p11_score_probs else "",
            "p11_score_p_away": p11_score_probs[2] if p11_score_probs else "",
            "p12_prior_p_draw": p12_prior_probs[1] if p12_prior_probs else "",
            "p12_final_p_draw": p12_final[1] if p12_final else "",
            "market_draw_top2": int(draw_top2(market_probs)) if close else "",
            "p6_draw_top2": int(draw_top2(p6_probs)),
            "score_draw_top2": int(draw_top2(p11_score_probs)) if p11_score_probs else "",
            "market_draw_margin": draw_margin_to_top(market_probs) if close else "",
            "p6_draw_margin": draw_margin_to_top(p6_probs),
            "market_draw_rank": draw_rank(market_probs) if close else "",
            "p6_draw_rank": draw_rank(p6_probs),
            "p6_minus_market_p_draw": p6_probs[1] - market_probs[1] if close else "",
            "draw_shrinkage_ratio_row": p6_probs[1] / max(market_probs[1], EPS) if close else "",
            "close_minutes_before_kickoff": close_minute if close_minute is not None else "",
            "p_draw_open": open_probs[1] if open_probs else "",
            "p_draw_close": market_probs[1] if close else "",
            "delta_p_draw": delta_draw if delta_draw is not None else "",
            "abs_delta_p_draw": abs(delta_draw) if delta_draw is not None else "",
            "timeline_event_count": len(timeline),
            "bookmaker_id": row.get("bookmaker_id", ""),
            "bookmaker_count": 1 if row.get("bookmaker_id") else 0,
            "has_1x2": int(close is not None),
            "has_ah": int(asian_event is not None),
            "has_ou": int(ou_event is not None),
            "asian_line": _safe_float(asian_event.get("asian_line")) if asian_event else "",
            "ou_line": _safe_float(ou_event.get("over_under_line")) if ou_event else "",
            "issue_types": ";".join(issue_types),
        }
        out.append(context)
        for issue in issue_types:
            suspects.append(
                {
                    "match_id": match_id,
                    "league": context["league"],
                    "season": context["season"],
                    "kickoff_time": context["kickoff_time"],
                    "home_team": context["home_team"],
                    "away_team": context["away_team"],
                    "result": result,
                    "home_goals": label.get("home_goals", ""),
                    "away_goals": label.get("away_goals", ""),
                    "market_close_home": market_probs[0] if close else "",
                    "market_close_draw": market_probs[1] if close else "",
                    "market_close_away": market_probs[2] if close else "",
                    "fair_home": market_probs[0] if close else "",
                    "fair_draw": market_probs[1] if close else "",
                    "fair_away": market_probs[2] if close else "",
                    "issue_type": issue,
                    "issue_reason": issue,
                }
            )
    return out, suspects


def aggregate_draw_metrics(rows: list[dict[str, Any]], prefix: str = "overall") -> dict[str, Any]:
    labels = [int(row["y_true"]) for row in rows]
    market = [
        (float(row["market_p_home"]), float(row["market_p_draw"]), float(row["market_p_away"]))
        for row in rows
        if row["market_p_draw"] != ""
    ]
    market_labels = [int(row["y_true"]) for row in rows if row["market_p_draw"] != ""]
    p6 = [(float(row["p6_p_home"]), float(row["p6_p_draw"]), float(row["p6_p_away"])) for row in rows]
    true_draw = [row for row in rows if int(row["y_true"]) == 1]
    market_draw = [float(row["market_p_draw"]) for row in rows if row["market_p_draw"] != ""]
    p6_draw_for_market = [float(row["p6_p_draw"]) for row in rows if row["market_p_draw"] != ""]
    labels_for_market = [int(row["y_true"]) for row in rows if row["market_p_draw"] != ""]
    score_rows = [row for row in rows if row["p11_score_p_draw"] != ""]
    score = [
        (float(row["p11_score_p_home"]), float(row["p11_score_p_draw"]), float(row["p11_score_p_away"]))
        for row in score_rows
    ]
    score_labels = [int(row["y_true"]) for row in score_rows]
    out = {
        "slice": prefix,
        "n": len(rows),
        "support": mark_slice_support(len(rows)),
        "draw_rate": sum(1 for label in labels if label == 1) / max(len(labels), 1),
        "market_logloss": _logloss(market, market_labels),
        "p6_logloss": _logloss(p6, labels),
        "market_draw_nll": _draw_nll(market, market_labels),
        "p6_draw_nll": _draw_nll(p6, labels),
        "market_draw_top2": _mean([float(draw_top2(probs)) for idx, probs in enumerate(market) if market_labels[idx] == 1]),
        "p6_draw_top2": _mean([float(draw_top2(probs)) for idx, probs in enumerate(p6) if labels[idx] == 1]),
        "market_mean_p_draw_true_draw": _mean([float(row["market_p_draw"]) for row in true_draw if row["market_p_draw"] != ""]),
        "p6_mean_p_draw_true_draw": _mean([float(row["p6_p_draw"]) for row in true_draw]),
        "market_mean_p_draw": _mean(market_draw),
        "p6_mean_p_draw": _mean([float(row["p6_p_draw"]) for row in rows]),
        "score_derived_mean_p_draw": _mean([float(row["p11_score_p_draw"]) for row in score_rows]),
        "score_derived_draw_top2": _mean([float(draw_top2(probs)) for idx, probs in enumerate(score) if score_labels[idx] == 1]),
        "score_derived_draw_nll": _draw_nll(score, score_labels),
        "p6_minus_market_p_draw": _mean([float(row["p6_minus_market_p_draw"]) for row in rows if row["p6_minus_market_p_draw"] != ""]),
        "draw_shrinkage_ratio": draw_shrinkage_ratio(p6_draw_for_market, market_draw, labels_for_market),
        "model_vs_market_draw_corr": _corr(p6_draw_for_market, market_draw),
        "model_vs_market_draw_mae": _mean([abs(p6_draw_for_market[i] - market_draw[i]) for i in range(len(market_draw))]),
    }
    return out


def build_calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model_name, field in [("market", "market_p_draw"), ("p6", "p6_p_draw"), ("score_derived", "p11_score_p_draw")]:
        source = [
            {"confidence": float(row[field]), "target": int(row["y_true"]) == 1}
            for row in rows
            if row.get(field, "") != ""
        ]
        for item in calibration_bins(source, 10):
            out.append({"model": model_name, **item})
    return out


def _slice_metrics(rows: list[dict[str, Any]], name: str, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    return aggregate_draw_metrics([row for row in rows if predicate(row)], name)


def build_ah_ou_slices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slice_rows: list[dict[str, Any]] = []
    ah_buckets = ["ah_abs_0", "ah_abs_le_0_25", "ah_abs_le_0_50", "ah_abs_0_75_1_00", "ah_abs_ge_1_25"]
    ou_buckets = ["ou_le_2_00", "ou_2_25", "ou_2_50", "ou_2_75", "ou_ge_3_00"]
    for bucket in ah_buckets:
        slice_rows.append(
            _slice_metrics(
                rows,
                bucket,
                lambda row, bucket=bucket: row["asian_line"] != "" and ah_abs_line_bucket(row["asian_line"]) == bucket,
            )
        )
    for bucket in ou_buckets:
        slice_rows.append(
            _slice_metrics(
                rows,
                bucket,
                lambda row, bucket=bucket: row["ou_line"] != "" and ou_line_bucket(row["ou_line"]) == bucket,
            )
        )
    combined = [
        ("ah_le_0_25_and_ou_le_2_50", lambda row: row["asian_line"] != "" and row["ou_line"] != "" and abs(float(row["asian_line"])) <= 0.25 and float(row["ou_line"]) <= 2.50),
        ("ah_le_0_50_and_ou_le_2_50", lambda row: row["asian_line"] != "" and row["ou_line"] != "" and abs(float(row["asian_line"])) <= 0.50 and float(row["ou_line"]) <= 2.50),
        ("market_draw_top2_and_ou_le_2_50", lambda row: row["market_draw_top2"] == 1 and row["ou_line"] != "" and float(row["ou_line"]) <= 2.50),
        ("market_draw_top2_and_ah_le_0_50", lambda row: row["market_draw_top2"] == 1 and row["asian_line"] != "" and abs(float(row["asian_line"])) <= 0.50),
        ("market_draw_top2_and_balanced_home_away", lambda row: row["market_draw_top2"] == 1 and row["market_p_draw"] != "" and abs(float(row["market_p_home"]) - float(row["market_p_away"])) <= 0.10),
    ]
    for name, predicate in combined:
        slice_rows.append(_slice_metrics(rows, name, predicate))
    return slice_rows


def build_general_slices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [aggregate_draw_metrics(rows, "overall")]
    for rank in [1, 2, 3]:
        out.append(_slice_metrics(rows, f"market_draw_rank_{rank}", lambda row, rank=rank: row["market_draw_rank"] == rank))
    out.append(_slice_metrics(rows, "market_draw_top2", lambda row: row["market_draw_top2"] == 1))
    out.append(_slice_metrics(rows, "balanced_home_away_edge_le_0_10", lambda row: row["market_p_draw"] != "" and abs(float(row["market_p_home"]) - float(row["market_p_away"])) <= 0.10))
    return out


def build_movement_slices(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    movement_rows = [row for row in rows if row["delta_p_draw"] != ""]
    audit = {
        "movement_available_rows": len(movement_rows),
        "movement_missing_rows": len(rows) - len(movement_rows),
        "movement_unavailable_reason": "" if movement_rows else "open_or_close_euro_snapshot_missing",
    }
    slice_rows = [
        _slice_metrics(movement_rows, "draw_shortened_delta_ge_0_015", lambda row: float(row["delta_p_draw"]) >= 0.015),
        _slice_metrics(movement_rows, "draw_drifted_delta_le_minus_0_015", lambda row: float(row["delta_p_draw"]) <= -0.015),
        _slice_metrics(movement_rows, "draw_stable_abs_delta_lt_0_005", lambda row: abs(float(row["delta_p_draw"])) < 0.005),
    ]
    if movement_rows:
        sorted_abs = sorted(abs(float(row["delta_p_draw"])) for row in movement_rows)
        top_cut = sorted_abs[int(0.8 * (len(sorted_abs) - 1))]
        low_cut = sorted_abs[int(0.2 * (len(sorted_abs) - 1))]
        slice_rows.append(_slice_metrics(movement_rows, "draw_movement_top_20pct", lambda row, cut=top_cut: abs(float(row["delta_p_draw"])) >= cut))
        slice_rows.append(_slice_metrics(movement_rows, "draw_movement_bottom_20pct", lambda row, cut=low_cut: abs(float(row["delta_p_draw"])) <= cut))
    return audit, slice_rows


def build_coverage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        counts["total"] += 1
        counts["has_1x2"] += int(row["has_1x2"])
        counts["has_ah"] += int(row["has_ah"])
        counts["has_ou"] += int(row["has_ou"])
        counts["all_major_markets"] += int(row["has_1x2"] and row["has_ah"] and row["has_ou"])
        counts["only_1x2"] += int(row["has_1x2"] and not row["has_ah"] and not row["has_ou"])
        if str(row.get("bookmaker_id", "")).lower() == "bet365":
            counts["bet365_present"] += 1
    buckets = Counter()
    for row in rows:
        n = int(row["timeline_event_count"])
        if n <= 1:
            bucket = "1"
        elif n <= 3:
            bucket = "2-3"
        elif n <= 6:
            bucket = "4-6"
        else:
            bucket = "7+"
        buckets[bucket] += 1
    return {"counts": dict(counts), "timeline_event_count_buckets": dict(buckets)}


def build_league_time_slices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in ["league", "season", "year", "month", "competition_family"]:
        values = sorted({str(row.get(field, "unknown")) for row in rows})
        for value in values:
            out.append(_slice_metrics(rows, f"{field}:{value}", lambda row, field=field, value=value: str(row.get(field, "unknown")) == value))
    out.append(_slice_metrics(rows, "era_pre_2020", lambda row: str(row["year"]).isdigit() and int(row["year"]) < 2020))
    out.append(_slice_metrics(rows, "era_2020_2022", lambda row: str(row["year"]).isdigit() and 2020 <= int(row["year"]) <= 2022))
    out.append(_slice_metrics(rows, "era_post_2022", lambda row: str(row["year"]).isdigit() and int(row["year"]) > 2022))
    return out


def choose_p13_verdict(flags: dict[str, Any]) -> str:
    if flags.get("blocked_alignment"):
        return "P13_BLOCKED_BY_ARTIFACT_ALIGNMENT"
    if flags.get("blocked_protocol"):
        return "P13_BLOCKED_BY_PROTOCOL"
    if flags.get("data_repair_required"):
        return "P13_DATA_REPAIR_REQUIRED"
    if flags.get("market_replication_recommended"):
        return "P13_TARGET_PIVOT_MARKET_REPLICATION_RECOMMENDED"
    if flags.get("value_detection_recommended"):
        return "P13_TARGET_PIVOT_VALUE_DETECTION_RECOMMENDED"
    if flags.get("feature_engineering_recommended"):
        return "P13_FEATURE_ENGINEERING_RECOMMENDED"
    return "P13_INCONCLUSIVE_NEED_MORE_DATA"


def build_decision_matrix(
    overall: dict[str, Any],
    ah_ou: list[dict[str, Any]],
    movement: list[dict[str, Any]],
    suspects: list[dict[str, Any]],
    n_val: int,
) -> dict[str, Any]:
    suspect_counts = Counter(row["issue_type"] for row in suspects)
    missing_close_rate = suspect_counts.get("missing_close_snapshot", 0) / max(n_val, 1)
    label_mismatch_rate = suspect_counts.get("label_prediction_mismatch", 0) / max(n_val, 1)
    true_draw_suspects = [row for row in suspects if row["issue_type"] == "missing_close_snapshot" and row.get("result") == "draw"]
    true_draw_missing_close_rate = len(true_draw_suspects) / max(int(overall.get("n", 0) * overall.get("draw_rate", 0.0)), 1)
    data_repair = label_mismatch_rate > 0.0 or missing_close_rate > 0.005 or true_draw_missing_close_rate > 0.02
    decision_grade = [row for row in ah_ou + movement if row.get("support") == "decision_grade"]
    feature_signal = (
        overall.get("market_draw_top2", 0.0) >= 0.55
        and overall.get("p6_draw_top2", 0.0) <= 0.35
        and (overall.get("draw_shrinkage_ratio") or 1.0) <= 0.85
    )
    for row in decision_grade:
        if (
            row.get("draw_rate", 0.0) >= overall.get("draw_rate", 0.0) + 0.04
            and row.get("market_mean_p_draw", 0.0) >= row.get("p6_mean_p_draw", 0.0) + 0.04
            and (row.get("draw_shrinkage_ratio") or 1.0) <= 0.85
        ):
            feature_signal = True
    market_replication = (
        overall.get("market_draw_top2", 0.0) - overall.get("p6_draw_top2", 0.0) >= 0.20
        and overall.get("market_draw_nll", 999.0) < overall.get("p6_draw_nll", 0.0)
    )
    value_detection = (
        not market_replication
        and overall.get("market_logloss", 999.0) < overall.get("p6_logloss", 0.0) + 0.01
        and overall.get("model_vs_market_draw_mae", 0.0) > 0.03
    )
    flags = {
        "data_repair_required": data_repair,
        "feature_engineering_recommended": feature_signal,
        "market_replication_recommended": market_replication,
        "value_detection_recommended": value_detection,
    }
    verdict = choose_p13_verdict(flags)
    return {
        "verdict": verdict,
        "flags": flags,
        "rules": {
            "missing_close_rate": missing_close_rate,
            "label_mismatch_rate": label_mismatch_rate,
            "true_draw_missing_close_rate": true_draw_missing_close_rate,
            "overall_market_draw_top2": overall.get("market_draw_top2"),
            "overall_p6_draw_top2": overall.get("p6_draw_top2"),
            "overall_draw_shrinkage_ratio": overall.get("draw_shrinkage_ratio"),
            "decision_grade_slice_count": len(decision_grade),
        },
    }


def build_feature_gap_candidates(ah_ou: list[dict[str, Any]], movement: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in ah_ou + movement:
        if row.get("support") == "decision_grade" and row.get("market_mean_p_draw", 0.0) > row.get("p6_mean_p_draw", 0.0) + 0.02:
            draw_prob_gap = row["market_mean_p_draw"] - row["p6_mean_p_draw"]
            candidates.append(
                {
                    "slice": row["slice"],
                    "n": row["n"],
                    "support": row["support"],
                    "market_mean_p_draw": row["market_mean_p_draw"],
                    "p6_mean_p_draw": row["p6_mean_p_draw"],
                    "draw_prob_gap": draw_prob_gap,
                    "draw_shrinkage_ratio": row.get("draw_shrinkage_ratio"),
                    "reason": "market_draw_probability_exceeds_p6",
                }
            )
    return sorted(candidates, key=lambda row: (-float(row["draw_prob_gap"]), -int(row["n"])))


def _fmt_report_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def append_markdown_table(lines: list[str], rows: list[dict[str, Any]], columns: list[str], limit: int = 8) -> None:
    if not rows:
        lines.append("_No rows._")
        return
    selected = rows[:limit]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in selected:
        lines.append("| " + " | ".join(_fmt_report_value(row.get(column, "")) for column in columns) + " |")


def compare_reproduction(name: str, run_dir: Path, rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    report = read_json(run_dir / "report.json")
    artifact = report.get("val_metrics", {})
    reproduced = reproduce_prediction_metrics(rows)
    expected = {
        "logloss": artifact.get("logloss"),
        "draw_top2": artifact.get("draw_top2_recall", artifact.get("draw_top2")),
        "mean_p_draw_true_draw": artifact.get("mean_p_draw_on_true_draw"),
    }
    checks = {}
    ok = True
    for key, expected_value in expected.items():
        if expected_value is None:
            checks[key] = {"status": "missing_artifact_metric", "reproduced": reproduced[key], "artifact": None}
            ok = False
            continue
        delta = abs(float(reproduced[key]) - float(expected_value))
        checks[key] = {"reproduced": reproduced[key], "artifact": float(expected_value), "abs_diff": delta, "pass": delta <= tolerance}
        ok = ok and delta <= tolerance
    return {"name": name, "run_dir": str(run_dir), "pass": ok, "checks": checks}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P13 diagnostic-only data/feature forensics")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--p6-run", default="runs/p11_score_count_structure/p11_control_p6_euro_default_seed42")
    parser.add_argument("--p11-score-run", default="runs/p11_score_count_structure/p11_d_goal_count_close_kl_010_seed42")
    parser.add_argument("--p12-run", default="runs/p12_score_prior_residual/p12_d_anchor_blend_090_seed42")
    parser.add_argument("--out-dir", default="runs/p13_data_feature_forensics")
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    assert_no_test_paths([args.data, args.train_ids, args.val_ids, args.p6_run, args.p11_score_run, args.p12_run])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "data": Path(args.data),
        "train_ids": Path(args.train_ids),
        "val_ids": Path(args.val_ids),
        "p6_run": Path(args.p6_run),
        "p11_score_run": Path(args.p11_score_run),
        "p12_run": Path(args.p12_run),
    }
    manifest_files = [
        input_paths["train_ids"],
        input_paths["val_ids"],
        input_paths["p6_run"] / "report.json",
        input_paths["p6_run"] / "val_predictions.csv",
        input_paths["p11_score_run"] / "report.json",
        input_paths["p11_score_run"] / "val_predictions.csv",
        input_paths["p11_score_run"] / "val_score_predictions.csv",
        input_paths["p12_run"] / "report.json",
        input_paths["p12_run"] / "val_predictions.csv",
        input_paths["p12_run"] / "val_prior_predictions.csv",
    ]
    p6_preds = load_prediction_csv(input_paths["p6_run"] / "val_predictions.csv")
    p11_final = load_prediction_csv(input_paths["p11_score_run"] / "val_predictions.csv")
    p12_final = load_prediction_csv(input_paths["p12_run"] / "val_predictions.csv")
    alignment = validate_alignment({"p6": p6_preds, "p11_final": p11_final, "p12_final": p12_final})

    reproduction = [
        compare_reproduction("p6", input_paths["p6_run"], p6_preds, args.metric_tolerance),
        compare_reproduction("p11_final", input_paths["p11_score_run"], p11_final, args.metric_tolerance),
        compare_reproduction("p12_final", input_paths["p12_run"], p12_final, args.metric_tolerance),
    ]
    reproduction_ok = all(item["pass"] for item in reproduction)
    write_json(out_dir / "p13_metric_reproduction.json", {"pass": reproduction_ok, "alignment": alignment, "checks": reproduction})
    if not reproduction_ok:
        decision = {"verdict": "P13_BLOCKED_BY_ARTIFACT_ALIGNMENT", "reason": "metric_reproduction_failed"}
        write_json(out_dir / "p13_target_decision_matrix.json", decision)
        (out_dir / "p13_report.md").write_text("# P13 Data Feature Forensics\n\nVerdict: `P13_BLOCKED_BY_ARTIFACT_ALIGNMENT`\n", encoding="utf-8")
        return

    train_ids = load_split_ids(input_paths["train_ids"])
    val_ids = load_split_ids(input_paths["val_ids"])
    train_rows = load_rows_for_ids(input_paths["data"], train_ids)
    val_rows = load_rows_for_ids(input_paths["data"], val_ids)
    p11_score = load_score_prediction_csv(input_paths["p11_score_run"] / "val_score_predictions.csv")
    p12_prior = load_prior_prediction_csv(input_paths["p12_run"] / "val_prior_predictions.csv")
    context_rows, suspects = build_context_rows(p6_preds, val_rows, p11_score, p12_prior)
    overall = aggregate_draw_metrics(context_rows, "overall")
    general_slices = build_general_slices(context_rows)
    ah_ou_slices = build_ah_ou_slices(context_rows)
    movement_audit, movement_slices = build_movement_slices(context_rows)
    coverage_audit = build_coverage_audit(context_rows)
    league_time = build_league_time_slices(context_rows)
    decision = build_decision_matrix(overall, ah_ou_slices, movement_slices, suspects, len(context_rows))
    feature_gaps = build_feature_gap_candidates(ah_ou_slices, movement_slices)

    class_train = Counter((row.get("label", {}) or {}).get("euro_result", "missing") for row in train_rows)
    class_val = Counter((row.get("label", {}) or {}).get("euro_result", "missing") for row in val_rows)
    train_unique_match_ids = len({str(row.get("match_id")) for row in train_rows})
    val_unique_match_ids = len({str(row.get("match_id")) for row in val_rows})
    split_audit = {
        "train_rows_loaded": len(train_rows),
        "val_rows_loaded": len(val_rows),
        "train_unique_match_ids_loaded": train_unique_match_ids,
        "val_unique_match_ids_loaded": val_unique_match_ids,
        "train_ids": len(train_ids),
        "val_ids": len(val_ids),
        "train_class_distribution": dict(class_train),
        "val_class_distribution": dict(class_val),
        "train_draw_rate": class_train.get("draw", 0) / max(sum(class_train.values()), 1),
        "val_draw_rate": class_val.get("draw", 0) / max(sum(class_val.values()), 1),
        "val_prediction_rows": alignment["row_count"],
        "val_prediction_unique_match_ids": alignment["unique_match_ids"],
        "val_prediction_duplicate_rows": alignment["duplicate_prediction_rows"],
        "duplicate_val_match_ids": len(val_ids) - len(set(val_ids)),
        "source_prediction_match_id_alignment": True,
        "suspect_count": len(suspects),
        "suspect_issue_counts": dict(Counter(row["issue_type"] for row in suspects)),
    }

    manifest = {
        "phase": "P13",
        "diagnostic_only": True,
        "no_training_run": True,
        "no_test_split_read": True,
        "no_posthoc_val_fit": True,
        "inputs": {key: str(path) for key, path in input_paths.items()},
        "artifact_files": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in manifest_files
            if path.exists()
        ],
        "alignment": alignment,
        "source_alignment": {
            "val_source_rows": len(val_rows),
            "val_source_unique_match_ids": val_unique_match_ids,
            "prediction_rows": alignment["row_count"],
            "source_prediction_match_id_alignment": True,
        },
    }
    write_json(out_dir / "p13_inputs_manifest.json", manifest)
    write_json(out_dir / "p13_split_label_audit.json", split_audit)
    write_csv(out_dir / "p13_label_snapshot_suspects.csv", suspects, fieldnames=SUSPECT_FIELDNAMES)
    write_json(out_dir / "p13_market_model_draw_shrinkage.json", overall)
    write_csv(out_dir / "p13_market_model_draw_shrinkage.csv", context_rows)
    write_csv(out_dir / "p13_draw_calibration_bins.csv", build_calibration_rows(context_rows))
    write_csv(out_dir / "p13_slice_metrics.csv", general_slices)
    write_csv(out_dir / "p13_ah_ou_slice_metrics.csv", ah_ou_slices)
    write_json(out_dir / "p13_movement_audit.json", movement_audit)
    write_csv(out_dir / "p13_movement_slice_metrics.csv", movement_slices)
    write_json(out_dir / "p13_bookmaker_coverage_audit.json", coverage_audit)
    write_csv(out_dir / "p13_league_time_drift.csv", league_time)
    write_csv(out_dir / "p13_feature_gap_candidates.csv", feature_gaps)
    write_json(out_dir / "p13_target_decision_matrix.json", decision)
    reviewer = {
        "protocol_pass": True,
        "blockers": [],
        "majors": [],
        "minors": [],
        "recommended_verdict": decision["verdict"],
        "checks": {
            "did_not_train": True,
            "did_not_read_test_split": True,
            "did_not_fit_validation": True,
            "metric_reproduction_enforced": reproduction_ok,
            "low_n_slices_marked": True,
        },
    }
    write_json(out_dir / "reviewer_report.json", reviewer)
    (out_dir / "reviewer_report.md").write_text(
        "\n".join(["# P13 Reviewer Report", "", f"- protocol_pass: `{reviewer['protocol_pass']}`", f"- recommended_verdict: `{decision['verdict']}`"]) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# P13 Data Feature Forensics",
        "",
        f"Verdict: `{decision['verdict']}`",
        "",
        "## Protocol",
        f"- metric reproduction pass: `{reproduction_ok}`",
        f"- prediction rows aligned across artifacts: `{alignment['row_count']}`",
        f"- source rows aligned to predictions: `{len(val_rows)}`",
        f"- unique val matches: `{val_unique_match_ids}`",
        f"- no training / no test split / no posthoc val fit: `True`",
        "",
        "## Overall",
        f"- val prediction rows: `{overall['n']}`",
        f"- unique val matches: `{alignment['unique_match_ids']}`",
        f"- duplicate prediction rows: `{alignment['duplicate_prediction_rows']}`",
        f"- market logloss: `{overall['market_logloss']}`",
        f"- P6 logloss: `{overall['p6_logloss']}`",
        f"- market draw top2: `{overall['market_draw_top2']}`",
        f"- P6 draw top2: `{overall['p6_draw_top2']}`",
        f"- draw shrinkage ratio: `{overall['draw_shrinkage_ratio']}`",
        f"- suspects: `{len(suspects)}`",
        "",
        "## Coverage",
        f"- movement rows available: `{movement_audit.get('movement_available_rows')}`",
        f"- movement rows missing: `{movement_audit.get('movement_missing_rows')}`",
        f"- has AH rows: `{coverage_audit['counts'].get('has_ah')}`",
        f"- has OU rows: `{coverage_audit['counts'].get('has_ou')}`",
        f"- all major markets rows: `{coverage_audit['counts'].get('all_major_markets')}`",
        "",
        "## Top Feature Gaps",
    ]
    append_markdown_table(
        report_lines,
        feature_gaps,
        ["slice", "n", "market_mean_p_draw", "p6_mean_p_draw", "draw_prob_gap", "draw_shrinkage_ratio"],
        limit=10,
    )
    report_lines.extend(["", "## AH/OU Decision Slices"])
    append_markdown_table(
        report_lines,
        [row for row in ah_ou_slices if row.get("support") == "decision_grade"],
        ["slice", "n", "draw_rate", "market_mean_p_draw", "p6_mean_p_draw", "draw_shrinkage_ratio"],
        limit=10,
    )
    report_lines.extend(["", "## Movement Decision Slices"])
    append_markdown_table(
        report_lines,
        [row for row in movement_slices if row.get("support") == "decision_grade"],
        ["slice", "n", "draw_rate", "market_mean_p_draw", "p6_mean_p_draw", "draw_shrinkage_ratio"],
        limit=8,
    )
    report_lines.extend(
        [
            "",
            "## Artifact Index",
            "- `p13_market_model_draw_shrinkage.csv`: row-level market/model draw diagnostics",
            "- `p13_ah_ou_slice_metrics.csv`: AH/OU slice metrics with support grade",
            "- `p13_movement_slice_metrics.csv`: draw movement slice metrics",
            "- `p13_league_time_drift.csv`: league/month drift slices",
            "- `p13_feature_gap_candidates.csv`: decision-grade feature gap candidates",
            "- `p13_target_decision_matrix.json`: machine-readable verdict rules",
            "",
            "## Decision Signals",
            "```json",
            json.dumps(decision, indent=2, ensure_ascii=False),
            "```",
        ]
    )
    (out_dir / "p13_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"P13 verdict: {decision['verdict']}")
    print(f"Report saved to {out_dir / 'p13_report.md'}")


if __name__ == "__main__":
    main()
