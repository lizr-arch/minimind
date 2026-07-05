"""P8 frozen draw-signal forensics and market-prior audit.

P8 is diagnostic-only. It does not modify P6, does not use the test split, and
does not promote any probe to mainline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_metrics import ece_from_probs, logloss_from_probs


EURO_MAP = {"home": 0, "draw": 1, "away": 2}
PROB_COLS = ("p_home", "p_draw", "p_away")
FIXED_P8_SEEDS = [42, 123, 2025]
FIXED_P8_PROBES: dict[str, dict[str, Any]] = {
    "probe_0_train_class_prior": {"kind": "deterministic"},
    "probe_1_market_close_de_vig": {"kind": "deterministic", "snapshot": "close"},
    "probe_2_market_open_de_vig": {"kind": "deterministic", "snapshot": "open"},
    "probe_3_train_only_multinomial_logistic_market_features": {"kind": "logistic"},
    "probe_4_train_only_binary_draw_then_conditional_home_away_logistic": {"kind": "factorized_logistic"},
    "probe_5_train_only_tiny_tabular_mlp_market_features": {"kind": "tiny_mlp"},
}
MARKET_FEATURES = [
    "q_home_open",
    "q_draw_open",
    "q_away_open",
    "q_home_close",
    "q_draw_close",
    "q_away_close",
    "q_draw_movement",
    "q_home_movement",
    "q_away_movement",
    "favorite_strength",
    "home_away_edge",
    "odds_entropy",
    "overround_open",
    "overround_close",
    "event_count",
    "earliest_event_minutes",
    "latest_event_minutes",
    "time_span_minutes",
]


class P8ProtocolError(RuntimeError):
    pass


def devig_1x2(home_odds: Any, draw_odds: Any, away_odds: Any) -> tuple[tuple[float, float, float], float]:
    try:
        odds = [float(home_odds), float(draw_odds), float(away_odds)]
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid odds") from exc
    if any((not math.isfinite(value)) or value <= 1.0 for value in odds):
        raise ValueError("invalid odds")
    inv = [1.0 / value for value in odds]
    overround = sum(inv)
    if overround <= 0 or not math.isfinite(overround):
        raise ValueError("invalid odds")
    return (inv[0] / overround, inv[1] / overround, inv[2] / overround), overround


def valid_pre_kickoff_euro_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for event in row.get("odds_timeline", []) or row.get("raw_timeline", []) or []:
        try:
            minute = float(event.get("minutes_before_kickoff"))
            probs, overround = devig_1x2(event.get("euro_h"), event.get("euro_d"), event.get("euro_a"))
        except (TypeError, ValueError):
            continue
        if minute < 0:
            continue
        if event.get("has_euro") is False:
            continue
        enriched = dict(event)
        enriched["_minute"] = minute
        enriched["_probs"] = probs
        enriched["_overround"] = overround
        events.append(enriched)
    return events


def _all_minutes(row: dict[str, Any]) -> list[float]:
    minutes = []
    for event in row.get("odds_timeline", []) or row.get("raw_timeline", []) or []:
        try:
            minute = float(event.get("minutes_before_kickoff"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(minute):
            minutes.append(minute)
    return minutes


def _select_snapshot(events: list[dict[str, Any]], snapshot: str) -> dict[str, Any] | None:
    if not events:
        return None
    if snapshot in {"open", "earliest"}:
        return max(events, key=lambda item: item["_minute"])
    if snapshot in {"close", "latest"}:
        return min(events, key=lambda item: item["_minute"])
    raise ValueError(f"unknown snapshot: {snapshot}")


def _entropy(probs: tuple[float, float, float]) -> float:
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def build_market_prior_row(row: dict[str, Any]) -> dict[str, Any]:
    match_id = str(row.get("match_id", ""))
    result = (row.get("label") or {}).get("euro_result")
    y_true = EURO_MAP.get(result, -1)
    events = valid_pre_kickoff_euro_events(row)
    minutes = _all_minutes(row)
    out: dict[str, Any] = {
        "match_id": match_id,
        "league": str(row.get("league_id") or "unknown"),
        "season": str(row.get("season") or "unknown"),
        "y_true": y_true,
        "event_count": len(row.get("odds_timeline", []) or row.get("raw_timeline", []) or []),
        "valid_pre_kickoff_euro_count": len(events),
        "earliest_event_minutes": max(minutes) if minutes else "",
        "latest_event_minutes": min(minutes) if minutes else "",
        "time_span_minutes": (max(minutes) - min(minutes)) if minutes else "",
    }
    for name, selector in [
        ("open", "open"),
        ("close", "close"),
        ("latest_pre_kickoff", "latest"),
        ("earliest_available", "earliest"),
    ]:
        event = _select_snapshot(events, selector)
        if event is None:
            out[f"{name}_missing_reason"] = "no_valid_pre_kickoff_euro"
            for key in ("home", "draw", "away"):
                out[f"q_{key}_{name}"] = ""
            out[f"overround_{name}"] = ""
            continue
        probs = event["_probs"]
        out[f"{name}_missing_reason"] = ""
        out[f"q_home_{name}"] = probs[0]
        out[f"q_draw_{name}"] = probs[1]
        out[f"q_away_{name}"] = probs[2]
        out[f"overround_{name}"] = event["_overround"]

    # Friendly aliases used by probe features and tests.
    for key in ("home", "draw", "away"):
        out[f"q_{key}_open"] = out.get(f"q_{key}_open", "")
        out[f"q_{key}_close"] = out.get(f"q_{key}_close", "")
    out["open_missing_reason"] = out.get("open_missing_reason", "")
    out["close_missing_reason"] = out.get("close_missing_reason", "")
    close = _snapshot_probs(out, "close")
    open_ = _snapshot_probs(out, "open")
    if close is None:
        favorite_strength = home_away_edge = odds_entropy = ""
    else:
        favorite_strength = abs(close[0] - close[2])
        home_away_edge = close[0] - close[2]
        odds_entropy = _entropy(close)
    out["favorite_strength"] = favorite_strength
    out["home_away_edge"] = home_away_edge
    out["odds_entropy"] = odds_entropy
    if close is not None and open_ is not None:
        out["q_home_movement"] = close[0] - open_[0]
        out["q_draw_movement"] = close[1] - open_[1]
        out["q_away_movement"] = close[2] - open_[2]
    else:
        out["q_home_movement"] = out["q_draw_movement"] = out["q_away_movement"] = ""
    out["overround_open"] = out.get("overround_open", "")
    out["overround_close"] = out.get("overround_close", "")
    return out


def _snapshot_probs(row: dict[str, Any], snapshot: str) -> tuple[float, float, float] | None:
    values = [row.get(f"q_{key}_{snapshot}") for key in ("home", "draw", "away")]
    if any(value == "" or value is None for value in values):
        return None
    try:
        probs = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) for value in probs):
        return None
    return probs  # type: ignore[return-value]


def draw_rank_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probs = probs.detach().cpu().float().clamp_min(1e-8)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    labels = labels.detach().cpu().long()
    preds = probs.argmax(dim=-1)
    true_draw = labels == 1
    pred_draw = preds == 1
    draw_ranks = (torch.argsort(probs, dim=-1, descending=True) == 1).nonzero(as_tuple=False)[:, 1] + 1
    draw_margin = probs.max(dim=-1).values - probs[:, 1]
    n_true_draw = int(true_draw.sum().item())
    top2 = torch.topk(probs, k=2, dim=-1).indices
    return {
        "draw_recall_argmax": float((pred_draw & true_draw).sum().item() / n_true_draw) if n_true_draw else 0.0,
        "draw_precision_argmax": float((pred_draw & true_draw).sum().item() / pred_draw.sum().item())
        if int(pred_draw.sum().item())
        else 0.0,
        "draw_top2": float((top2[true_draw] == 1).any(dim=-1).float().mean().item()) if n_true_draw else 0.0,
        "draw_rank1_count": int((draw_ranks == 1).sum().item()),
        "draw_rank2_count": int((draw_ranks == 2).sum().item()),
        "draw_rank3_count": int((draw_ranks == 3).sum().item()),
        "mean_draw_margin_to_top": float(draw_margin.mean().item()) if probs.numel() else 0.0,
        "mean_p_draw_all": float(probs[:, 1].mean().item()) if probs.numel() else 0.0,
        "mean_p_draw_true_draw": float(probs[true_draw, 1].mean().item()) if n_true_draw else 0.0,
        "draw_brier": float(((probs[:, 1] - true_draw.float()) ** 2).mean().item()) if probs.numel() else 0.0,
    }


def prediction_metrics(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    labels = torch.tensor([int(row["y_true"]) for row in rows], dtype=torch.long)
    probs = torch.tensor([[float(row["p_home"]), float(row["p_draw"]), float(row["p_away"])] for row in rows])
    probs = probs.clamp_min(1e-8)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    preds = probs.argmax(dim=-1)
    out = {
        "val_logloss": logloss_from_probs(probs, labels),
        "ece": ece_from_probs(probs, labels, 3)["ece"],
        "accuracy": float((preds == labels).float().mean().item()),
    }
    out.update(draw_rank_metrics(probs, labels))
    for idx, name in [(0, "home"), (1, "draw"), (2, "away")]:
        mask = labels == idx
        out[f"classwise_nll_{name}"] = float((-torch.log(probs[mask, idx])).mean().item()) if int(mask.sum()) else 0.0
        out[f"{name}_recall"] = float((preds[mask] == idx).float().mean().item()) if int(mask.sum()) else 0.0
    if prefix:
        return {f"{prefix}_{key}": value for key, value in out.items()}
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def _market_prediction_rows(market_rows: list[dict[str, Any]], snapshot: str) -> list[dict[str, Any]]:
    out = []
    for row in market_rows:
        probs = _snapshot_probs(row, snapshot)
        if probs is None or int(row.get("y_true", -1)) not in (0, 1, 2):
            continue
        out.append(
            {
                "match_id": row["match_id"],
                "y_true": int(row["y_true"]),
                "p_home": probs[0],
                "p_draw": probs[1],
                "p_away": probs[2],
            }
        )
    return out


def build_p6_vs_market_audit(
    market_rows: list[dict[str, Any]],
    p6_predictions: list[dict[str, Any]],
    snapshot: str = "close",
) -> dict[str, Any]:
    market_preds = _market_prediction_rows(market_rows, snapshot)
    p6_by_id = {str(row["match_id"]): row for row in p6_predictions}
    paired_market = []
    paired_p6 = []
    for market in market_preds:
        p6 = p6_by_id.get(str(market["match_id"]))
        if p6 is None:
            continue
        paired_market.append(market)
        paired_p6.append(
            {
                "match_id": p6["match_id"],
                "y_true": int(p6["y_true"]),
                "p_home": float(p6["p_home"]),
                "p_draw": float(p6["p_draw"]),
                "p_away": float(p6["p_away"]),
            }
        )
    market_metrics = prediction_metrics(paired_market, prefix="market") if paired_market else {}
    p6_metrics = prediction_metrics(paired_p6, prefix="p6") if paired_p6 else {}
    market_draw = [float(row["p_draw"]) for row in paired_market]
    p6_draw = [float(row["p_draw"]) for row in paired_p6]
    true_draw_mask = [int(row["y_true"]) == 1 for row in paired_market]
    market_true = [p for p, flag in zip(market_draw, true_draw_mask) if flag]
    p6_true = [p for p, flag in zip(p6_draw, true_draw_mask) if flag]
    overall = {
        **market_metrics,
        **p6_metrics,
        "paired_count": len(paired_market),
        "p6_minus_market_p_draw": _mean_or_zero([a - b for a, b in zip(p6_draw, market_draw)]),
        "mean_p6_minus_market_p_draw_all": _mean_or_zero([a - b for a, b in zip(p6_draw, market_draw)]),
        "mean_p6_minus_market_p_draw_true_draw": _mean_or_zero([a - b for a, b in zip(p6_true, market_true)]),
        "p6_vs_market_draw_corr": _pearson(p6_draw, market_draw),
        "p6_market_p_draw_corr": _pearson(p6_draw, market_draw),
        "p6_vs_market_draw_mae": _mean_or_zero([abs(a - b) for a, b in zip(p6_draw, market_draw)]),
        "p6_market_p_draw_mae": _mean_or_zero([abs(a - b) for a, b in zip(p6_draw, market_draw)]),
        "p6_draw_shrinkage_ratio": (_mean_or_zero(p6_true) / _mean_or_zero(market_true))
        if _mean_or_zero(market_true) > 0
        else 0.0,
        "true_draw_underpricing_by_model": _mean_or_zero([1.0 - p for p in p6_true]),
        "true_draw_underpricing_by_market": _mean_or_zero([1.0 - p for p in market_true]),
    }
    if "market_val_logloss" in overall:
        overall["market_logloss"] = overall["market_val_logloss"]
    if "market_classwise_nll_draw" in overall:
        overall["market_draw_nll"] = overall["market_classwise_nll_draw"]
    if "market_ece" in overall:
        overall["market_ece"] = overall["market_ece"]
    if "market_mean_p_draw_true_draw" in overall:
        overall["market_mean_p_draw_true_draw"] = overall["market_mean_p_draw_true_draw"]
    if "market_mean_draw_margin_to_top" in overall:
        overall["market_mean_draw_margin_to_top"] = overall["market_mean_draw_margin_to_top"]
    return {"phase": "P8 P6-vs-market draw audit", "snapshot": snapshot, "overall": overall}


def _mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_label_split_audit(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids = [str(row.get("match_id")) for row in train_rows]
    val_ids = [str(row.get("match_id")) for row in val_rows]
    duplicates = sorted(set(train_ids) & set(val_ids))
    hard_fail_reasons: list[str] = []
    if duplicates:
        hard_fail_reasons.append("duplicate_match_ids")

    def label_counts(rows: list[dict[str, Any]]) -> Counter[int]:
        counts: Counter[int] = Counter()
        for row in rows:
            result = (row.get("label") or {}).get("euro_result")
            y = EURO_MAP.get(result, -1)
            counts[y] += 1
            if y == -1:
                hard_fail_reasons.append(f"invalid_label:{row.get('match_id')}")
        return counts

    train_counts = label_counts(train_rows)
    val_counts = label_counts(val_rows)
    total_train = max(sum(count for key, count in train_counts.items() if key >= 0), 1)
    total_val = max(sum(count for key, count in val_counts.items() if key >= 0), 1)
    after_kickoff = 0
    invalid_odds = 0
    latest_after_kickoff = 0
    for row in train_rows + val_rows:
        minutes = _all_minutes(row)
        if minutes and min(minutes) < 0:
            latest_after_kickoff += 1
        for event in row.get("odds_timeline", []) or row.get("raw_timeline", []) or []:
            try:
                minute = float(event.get("minutes_before_kickoff"))
                devig_1x2(event.get("euro_h"), event.get("euro_d"), event.get("euro_a"))
            except (TypeError, ValueError):
                invalid_odds += 1
                continue
            if minute < 0:
                after_kickoff += 1
    if after_kickoff:
        hard_fail_reasons.append("after_kickoff_events")
    if invalid_odds / max(sum(len(row.get("odds_timeline", []) or row.get("raw_timeline", []) or []) for row in train_rows + val_rows), 1) > 0.001:
        hard_fail_reasons.append("invalid_odds_rate_gt_0.1pct")
    train_draw_rate = train_counts[1] / total_train
    val_draw_rate = val_counts[1] / total_val
    if abs(train_draw_rate - val_draw_rate) >= 0.030:
        hard_fail_reasons.append("train_val_draw_rate_shift")
    return {
        "phase": "P8 label split audit",
        "train_samples": total_train,
        "val_samples": total_val,
        "train_draw_rate": train_draw_rate,
        "val_draw_rate": val_draw_rate,
        "train_home_rate": train_counts[0] / total_train,
        "val_home_rate": val_counts[0] / total_val,
        "train_away_rate": train_counts[2] / total_train,
        "val_away_rate": val_counts[2] / total_val,
        "class_mapping_integrity": all(key in (0, 1, 2) for key in train_counts | val_counts),
        "duplicate_match_ids": duplicates,
        "missing_result_labels": train_counts[-1] + val_counts[-1],
        "invalid_odds_values": invalid_odds,
        "odds_after_kickoff_count": after_kickoff,
        "latest_event_after_kickoff_count": latest_after_kickoff,
        "hard_fail_reasons": sorted(set(hard_fail_reasons)),
    }


def parse_registered_probes(raw: str) -> list[str]:
    probes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in probes if item not in FIXED_P8_PROBES]
    if unknown:
        raise ValueError(f"Unregistered P8 probe(s): {unknown}")
    if len(set(probes)) != len(probes):
        raise ValueError("Duplicate P8 probes are not allowed")
    return probes


def parse_registered_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [item for item in seeds if item not in FIXED_P8_SEEDS]
    if unknown:
        raise ValueError(f"Unregistered P8 seed(s): {unknown}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate P8 seeds are not allowed")
    return seeds


def validate_probe_protocol(
    train_ids: str,
    val_ids: str,
    probes: list[str],
    seeds: list[int],
    promote_mainline: bool = False,
) -> dict[str, Any]:
    if "test" in Path(train_ids).name.lower() or "test" in Path(val_ids).name.lower():
        raise P8ProtocolError("P8 refuses test split paths")
    hard_fail_reasons = []
    if probes != list(FIXED_P8_PROBES):
        hard_fail_reasons.append("probe_matrix_mutation")
    if seeds != FIXED_P8_SEEDS:
        hard_fail_reasons.append("seed_matrix_mutation")
    if promote_mainline:
        hard_fail_reasons.append("probe_promoted_to_mainline")
    return {
        "phase": "P8",
        "test_split_loaded": False,
        "official_val_used_for_fitting": False,
        "new_candidates_added_after_val": False,
        "p6_default_behavior_changed": False,
        "probe_promoted_to_mainline": bool(promote_mainline),
        "market_snapshot_silent_fallback": False,
        "missing_odds_silent_imputation": False,
        "probes": probes,
        "seeds": seeds,
        "hard_fail_reasons": hard_fail_reasons,
    }


def load_split_ids(path: str | Path) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_rows_for_ids(jsonl_path: str | Path, match_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row.get("match_id")) in match_ids:
                rows.append(row)
    return rows


def load_predictions_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return [
            {
                "match_id": row["match_id"],
                "y_true": int(float(row["y_true"])),
                "p_home": float(row["p_home"]),
                "p_draw": float(row["p_draw"]),
                "p_away": float(row["p_away"]),
            }
            for row in csv.DictReader(f)
        ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_market_prior_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_market_prior_row(row) for row in rows]


def _class_prior_predictions(train_market: list[dict[str, Any]], val_market: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(int(row["y_true"]) for row in train_market if int(row.get("y_true", -1)) in (0, 1, 2))
    total = max(sum(counts.values()), 1)
    probs = [counts[idx] / total for idx in range(3)]
    return [
        {"match_id": row["match_id"], "y_true": int(row["y_true"]), "p_home": probs[0], "p_draw": probs[1], "p_away": probs[2]}
        for row in val_market
        if int(row.get("y_true", -1)) in (0, 1, 2)
    ]


def _market_snapshot_predictions(val_market: list[dict[str, Any]], snapshot: str) -> list[dict[str, Any]]:
    return _market_prediction_rows(val_market, snapshot)


def _feature_tensor(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, list[str]]:
    values = []
    kept = []
    for row in rows:
        vec = []
        ok = True
        for name in MARKET_FEATURES:
            value = row.get(name)
            if value == "" or value is None:
                ok = False
                break
            vec.append(float(value))
        if ok and int(row.get("y_true", -1)) in (0, 1, 2):
            values.append(vec)
            kept.append(str(row["match_id"]))
    if not values:
        return torch.empty(0, len(MARKET_FEATURES)), kept
    return torch.tensor(values, dtype=torch.float32), kept


def _labels_for_ids(rows: list[dict[str, Any]], ids: list[str]) -> torch.Tensor:
    by_id = {str(row["match_id"]): int(row["y_true"]) for row in rows}
    return torch.tensor([by_id[mid] for mid in ids], dtype=torch.long)


def _standardize_train_only(train_x: torch.Tensor, val_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if train_x.numel() == 0:
        return train_x, val_x
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std


def _train_linear_probe(
    train_market: list[dict[str, Any]],
    val_market: list[dict[str, Any]],
    seed: int,
    factorized: bool = False,
    tiny_mlp: bool = False,
    epochs: int = 120,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    train_x, train_ids = _feature_tensor(train_market)
    val_x, val_ids = _feature_tensor(val_market)
    if train_x.numel() == 0 or val_x.numel() == 0:
        return []
    train_x, val_x = _standardize_train_only(train_x, val_x)
    y = _labels_for_ids(train_market, train_ids)
    if tiny_mlp:
        model = torch.nn.Sequential(torch.nn.Linear(train_x.shape[1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    elif factorized:
        model = torch.nn.Linear(train_x.shape[1], 2)
    else:
        model = torch.nn.Linear(train_x.shape[1], 3)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(train_x)
        if factorized:
            draw_logit = logits[:, 0]
            ha_logit = logits[:, 1]
            p_draw = torch.sigmoid(draw_logit)
            p_home = (1 - p_draw) * torch.sigmoid(ha_logit)
            p_away = (1 - p_draw) * (1 - torch.sigmoid(ha_logit))
            probs = torch.stack([p_home, p_draw, p_away], dim=-1).clamp_min(1e-8)
            loss = F.nll_loss(probs.log(), y)
        else:
            loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(val_x)
        if factorized:
            p_draw = torch.sigmoid(logits[:, 0])
            p_home = (1 - p_draw) * torch.sigmoid(logits[:, 1])
            p_away = (1 - p_draw) * (1 - torch.sigmoid(logits[:, 1]))
            probs = torch.stack([p_home, p_draw, p_away], dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
    labels = _labels_for_ids(val_market, val_ids)
    return [
        {
            "match_id": mid,
            "y_true": int(labels[idx].item()),
            "p_home": float(probs[idx, 0].item()),
            "p_draw": float(probs[idx, 1].item()),
            "p_away": float(probs[idx, 2].item()),
        }
        for idx, mid in enumerate(val_ids)
    ]


def run_probe(probe_id: str, train_market: list[dict[str, Any]], val_market: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if probe_id == "probe_0_train_class_prior":
        return _class_prior_predictions(train_market, val_market)
    if probe_id == "probe_1_market_close_de_vig":
        return _market_snapshot_predictions(val_market, "close")
    if probe_id == "probe_2_market_open_de_vig":
        return _market_snapshot_predictions(val_market, "open")
    if probe_id == "probe_3_train_only_multinomial_logistic_market_features":
        return _train_linear_probe(train_market, val_market, seed, factorized=False)
    if probe_id == "probe_4_train_only_binary_draw_then_conditional_home_away_logistic":
        return _train_linear_probe(train_market, val_market, seed, factorized=True)
    if probe_id == "probe_5_train_only_tiny_tabular_mlp_market_features":
        return _train_linear_probe(train_market, val_market, seed, tiny_mlp=True)
    raise ValueError(f"Unregistered P8 probe: {probe_id}")


def run_probe_matrix(
    train_market: list[dict[str, Any]],
    val_market: list[dict[str, Any]],
    probes: list[str],
    seeds: list[int],
    out_root: Path,
) -> dict[str, Any]:
    pred_dir = out_root / "probe_predictions"
    rows = []
    for probe_id in probes:
        probe_kind = FIXED_P8_PROBES[probe_id]["kind"]
        effective_seeds = seeds if probe_kind != "deterministic" else [seeds[0]]
        for seed in effective_seeds:
            preds = run_probe(probe_id, train_market, val_market, seed)
            pred_path = pred_dir / f"{probe_id}_seed{seed}.csv"
            write_csv(pred_path, preds, ["match_id", "y_true", "p_home", "p_draw", "p_away"])
            metrics = prediction_metrics(preds) if preds else {}
            row = {"probe_id": probe_id, "seed": seed, "n_predictions": len(preds), **metrics}
            rows.append(row)
    write_csv(out_root / "probe_results_by_seed.csv", rows)
    summary: dict[str, Any] = {"phase": "P8 probe matrix", "probes": []}
    for probe_id in probes:
        probe_rows = [row for row in rows if row["probe_id"] == probe_id]
        if not probe_rows:
            summary["probes"].append({"probe_id": probe_id, "status": "missing"})
            continue
        metric_keys = [key for key in probe_rows[0] if key not in {"probe_id", "seed", "n_predictions"}]
        item = {
            "probe_id": probe_id,
            "seed_count": len(probe_rows),
            "mean_n_predictions": _mean_or_zero([float(row["n_predictions"]) for row in probe_rows]),
        }
        for key in metric_keys:
            item[f"mean_{key}"] = _mean_or_zero([float(row[key]) for row in probe_rows if row.get(key) is not None])
        summary["probes"].append(item)
    write_json(out_root / "probe_results_summary.json", summary)
    lines = ["# P8 Probe Report", ""]
    for item in summary["probes"]:
        lines.append(
            f"- {item['probe_id']}: seeds={item.get('seed_count', 0)}, "
            f"logloss={item.get('mean_val_logloss', 0.0):.6f}, "
            f"draw_top2={item.get('mean_draw_top2', 0.0):.6f}, "
            f"mean_p_draw_true={item.get('mean_mean_p_draw_true_draw', 0.0):.6f}"
        )
    (out_root / "probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def compute_p8_decision_gates(
    p6_vs_market: dict[str, Any],
    label_audit: dict[str, Any],
    probe_summary: dict[str, Any],
) -> dict[str, Any]:
    overall = p6_vs_market.get("overall", {})
    verdict: list[str] = []
    recommendations: list[str] = []
    recommended_next_phase = "P9 Needs Manual Review"

    if label_audit.get("hard_fail_reasons"):
        verdict.append("P8_HARD_STOP_DATA_OR_SPLIT_REPAIR")
        recommendations.append("Stop model work until label/split/data hard failures are repaired.")
        recommended_next_phase = "P9 Data/Split Repair"
        return {
            "verdict": verdict,
            "recommendations": recommendations,
            "recommended_next_phase": recommended_next_phase,
        }

    market_draw_recall = float(overall.get("market_draw_recall_argmax", 0.0) or 0.0)
    market_rank1 = int(overall.get("market_draw_rank1_count", 0) or 0)
    if market_draw_recall < 0.010 and market_rank1 <= 5:
        verdict.append("P8_STOP_DRAW_ARGMAX_PRIMARY_GATE")
        recommendations.append("Use draw NLL/top2/p_draw_true/margin/calibration as primary draw gates; draw argmax is rare even for market priors.")

    p6_minus_market_true = float(overall.get("mean_p6_minus_market_p_draw_true_draw", 0.0) or 0.0)
    shrinkage = float(overall.get("p6_draw_shrinkage_ratio", 1.0) or 1.0)
    p6_top2 = float(overall.get("p6_draw_top2", 0.0) or 0.0)
    market_top2 = float(overall.get("market_draw_top2", 0.0) or 0.0)
    if p6_minus_market_true <= -0.030 or shrinkage <= 0.85 or (market_top2 - p6_top2) >= 0.050:
        verdict.append("P8_P6_SUPPRESSES_DRAW_BELOW_MARKET")
        recommendations.append("Next model should predict residuals around de-vig market priors instead of raw 1X2 probabilities.")
        recommended_next_phase = "P9 Market-Prior Anchored Residual Model"

    p6_logloss = float(overall.get("p6_val_logloss", 0.0) or 0.0)
    p6_mean_draw = float(overall.get("p6_mean_p_draw_true_draw", 0.0) or 0.0)
    p6_margin = float(overall.get("p6_mean_draw_margin_to_top", 0.0) or 0.0)
    for probe in probe_summary.get("probes", []):
        if (
            float(probe.get("mean_val_logloss", 999.0) or 999.0) <= p6_logloss + 0.002
            and float(probe.get("mean_draw_top2", 0.0) or 0.0) >= p6_top2 + 0.050
            and float(probe.get("mean_mean_p_draw_true_draw", 0.0) or 0.0) >= p6_mean_draw + 0.020
            and float(probe.get("mean_mean_draw_margin_to_top", 999.0) or 999.0) <= p6_margin - 0.030
        ):
            verdict.append("P8_SIMPLE_MARKET_PROBE_RECOVERS_DRAW_SIGNAL")
            recommendations.append(f"{probe.get('probe_id')} recovers draw behavior from simple market features.")
            recommended_next_phase = "P9 Market-Prior Anchored Residual Model"
            break

    all_fail = True
    considered = [probe for probe in probe_summary.get("probes", []) if probe.get("probe_id") != "probe_0_train_class_prior"]
    for probe in considered:
        if not (
            float(probe.get("mean_draw_top2", 0.0) or 0.0) <= 0.320
            and float(probe.get("mean_mean_p_draw_true_draw", 0.0) or 0.0) <= 0.205
            and float(probe.get("mean_classwise_nll_draw", 999.0) or 999.0) >= 1.630
        ):
            all_fail = False
            break
    if considered and all_fail:
        verdict.append("P8_ALL_MARKET_AND_PROBES_FAIL_DRAW_SIGNAL")
        recommendations.append("Do not continue draw-specific 1X2 objectives; inspect score structure.")
        if recommended_next_phase == "P9 Needs Manual Review":
            recommended_next_phase = "P9 Score-Structure Probe"

    if not verdict:
        verdict.append("P8_NO_HARD_DIRECTIONAL_SIGNAL")
        recommendations.append("Review diagnostics before opening any new model phase.")
    return {
        "verdict": sorted(set(verdict)),
        "recommendations": recommendations,
        "recommended_next_phase": recommended_next_phase,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P8 frozen draw signal forensics")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--p6-predictions", default="runs/p6_residual_patch_itransformer/p6_euro_default_seed42/val_predictions.csv")
    parser.add_argument("--out-root", default="runs/p8_draw_signal_forensics")
    parser.add_argument("--diagnostics-dir", default="diagnostics")
    parser.add_argument("--probes", default=",".join(FIXED_P8_PROBES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FIXED_P8_SEEDS))
    parser.add_argument("--max-samples", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    probes = parse_registered_probes(args.probes)
    seeds = parse_registered_seeds(args.seeds)
    protocol = validate_probe_protocol(args.train_ids, args.val_ids, probes, seeds)
    out_root = Path(args.out_root)
    diagnostics_dir = Path(args.diagnostics_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "protocol_audit.json", protocol)
    if protocol["hard_fail_reasons"]:
        raise SystemExit(f"P8 protocol failed: {protocol['hard_fail_reasons']}")

    train_rows = load_rows_for_ids(args.data, load_split_ids(args.train_ids))
    val_rows = load_rows_for_ids(args.data, load_split_ids(args.val_ids))
    if args.max_samples > 0:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: min(args.max_samples, len(val_rows))]
    train_market = build_market_prior_rows(train_rows)
    val_market = build_market_prior_rows(val_rows)
    all_market = train_market + val_market
    market_fields = sorted({key for row in all_market for key in row})
    write_csv(diagnostics_dir / "p8_market_priors.csv", all_market, market_fields)
    write_json(diagnostics_dir / "p8_market_priors_schema.json", {"fields": market_fields})
    (diagnostics_dir / "p8_market_prior_summary.md").write_text(
        f"# P8 Market Priors\n\n- train_rows: `{len(train_market)}`\n- val_rows: `{len(val_market)}`\n",
        encoding="utf-8",
    )
    p6_predictions = load_predictions_csv(args.p6_predictions)
    p6_vs_market = build_p6_vs_market_audit(val_market, p6_predictions, snapshot="close")
    write_json(diagnostics_dir / "p8_p6_vs_market_draw_audit.json", p6_vs_market)
    write_csv(diagnostics_dir / "p8_p6_vs_market_draw_audit.csv", [p6_vs_market["overall"]])
    (diagnostics_dir / "p8_p6_vs_market_draw_audit.md").write_text(
        "# P8 P6-vs-Market Draw Audit\n\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in p6_vs_market["overall"].items())
        + "\n",
        encoding="utf-8",
    )
    label_audit = build_label_split_audit(train_rows, val_rows)
    write_json(diagnostics_dir / "p8_label_split_audit.json", label_audit)
    write_csv(diagnostics_dir / "p8_label_split_audit.csv", [label_audit])
    (diagnostics_dir / "p8_label_split_audit.md").write_text(
        "# P8 Label Split Audit\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in label_audit.items()) + "\n",
        encoding="utf-8",
    )
    probe_summary = run_probe_matrix(train_market, val_market, probes, seeds, out_root)
    decision_gates = compute_p8_decision_gates(p6_vs_market, label_audit, probe_summary)
    final_report = {
        "phase": "P8 Frozen Draw Signal Forensics / Market-Prior Audit",
        "protocol_audit": protocol,
        "p6_vs_market": p6_vs_market,
        "label_split_audit": label_audit,
        "probe_summary": probe_summary,
        "decision_gates": decision_gates,
    }
    write_json(out_root / "p8_final_report.json", final_report)
    print(f"P8 final report: {out_root / 'p8_final_report.json'}")
    print(f"P8 protocol hard_fail_reasons: {protocol['hard_fail_reasons']}")


if __name__ == "__main__":
    main()
