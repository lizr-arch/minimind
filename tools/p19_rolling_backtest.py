"""P19 rolling-backtest protocol helpers for AH strategy validation.

P19 is deliberately conservative: it does not introduce a new model family and
does not read test artifacts. The module provides reusable audit, fold-building,
and fixture-collapsed scoring utilities that can be used before any expensive
rolling retraining run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.asian_handicap import get_upper_lower_goals, settle_asian_5class
from model.odds_patch_itransformer_v2 import FEATURE_DEFS
from tools.p14_train_market_replication import p14_selected_feature_names
from tools.p18_ah_cover_diagnostics import latest_pre_kickoff_asian_snapshot
from tools.p4_train_residual_goal_diff import load_rows_for_ids, load_split_ids


FEATURE_BLACKLIST_TERMS = (
    "goal",
    "score",
    "result",
    "cover",
    "settle",
    "settlement",
    "payoff",
    "unit",
    "label",
    "winner",
    "home_goals",
    "away_goals",
    "goal_diff",
)
AH_LABEL_NORMALIZATION = {
    "upper_full_win": "upper_full_win",
    "full_win": "upper_full_win",
    "upper_half_win": "upper_half_win",
    "half_win": "upper_half_win",
    "push": "push",
    "upper_half_loss": "upper_half_loss",
    "half_loss": "upper_half_loss",
    "upper_full_loss": "upper_full_loss",
    "full_loss": "upper_full_loss",
}
DEFAULT_CANDIDATE_REGISTRY = {
    "phase": "P19",
    "task": "rolling_retrain_backtest_protocol",
    "baseline": "P18.0_goal_diff_directional_baseline",
    "cover_variants": ["p18_ahw_003"],
    "direct_variants": ["p18d_direct030"],
    "agreement_modes": ["both"],
    "thresholds": [0.15, 0.20, 0.25, 0.30],
    "locked_strategy": {
        "cover_variant": "p18_ahw_003",
        "direct_variant": "p18d_direct030",
        "mode": "both",
        "threshold": 0.25,
    },
    "policy": {
        "no_test_set": True,
        "no_new_model_family": True,
        "strategy_selection_inside_fold_only": True,
    },
}


class P19ProtocolError(RuntimeError):
    pass


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "selected"}
    return bool(value)


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(statistics.fmean(vals)) if vals else 0.0


def _parse_dt(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return datetime.min
    raw = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _contains_test_path(path: str | Path) -> bool:
    parts = [part.lower() for part in Path(path).parts]
    return any(part == "test" or part.startswith("test_") or part.startswith("test-") for part in parts)


def assert_no_test_paths(paths: Iterable[str | Path]) -> None:
    for path in paths:
        if path and _contains_test_path(path):
            raise P19ProtocolError(f"P19 refuses test split or test artifact path: {path}")


def scan_feature_blacklist(
    feature_names: Iterable[str],
    blacklist_terms: Iterable[str] = FEATURE_BLACKLIST_TERMS,
) -> dict[str, Any]:
    names = [str(feature) for feature in feature_names]
    terms = [term.lower() for term in blacklist_terms]
    hits = []
    for feature in names:
        lower = feature.lower()
        for term in terms:
            if term in lower:
                hits.append({"feature": feature, "term": term})
                break
    return {
        "status": "FAIL" if hits else "PASS",
        "checked_feature_count": len(names),
        "hits": hits,
        "blacklist_terms": terms,
    }


def normalize_ah_result(value: Any) -> str | None:
    return AH_LABEL_NORMALIZATION.get(str(value or "").strip().lower())


def _settle_row_result(row: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    label = row.get("label", {}) or {}
    snapshot = latest_pre_kickoff_asian_snapshot(row)
    diag = {
        "valid_asian": bool(snapshot.get("valid_asian")),
        "negative_time_valid_asian_count": int(snapshot.get("negative_time_valid_asian_count") or 0),
    }
    if not snapshot.get("valid_asian"):
        return None, diag
    try:
        home_goals = int(label["home_goals"])
        away_goals = int(label["away_goals"])
        asian_line = float(snapshot["asian_line"])
    except (KeyError, TypeError, ValueError):
        diag["missing_label"] = True
        return None, diag
    upper_side = str(label.get("upper_side") or "home").lower()
    try:
        upper_goals, lower_goals = get_upper_lower_goals(home_goals, away_goals, upper_side)
    except ValueError:
        diag["unsupported_upper_side"] = upper_side
        return None, diag
    return settle_asian_5class(upper_goals, lower_goals, asian_line), diag


def audit_ah_settlement(rows: list[dict[str, Any]], sample_limit: int = 20) -> dict[str, Any]:
    checked = 0
    skipped = 0
    negative_time_valid_asian_count = 0
    mismatches = []
    missing_label_result = 0
    unsupported_upper_side_count = 0
    for row in rows:
        computed, diag = _settle_row_result(row)
        negative_time_valid_asian_count += int(diag.get("negative_time_valid_asian_count") or 0)
        if "unsupported_upper_side" in diag:
            unsupported_upper_side_count += 1
        if computed is None:
            skipped += 1
            continue
        expected = normalize_ah_result((row.get("label", {}) or {}).get("asian_result"))
        if expected is None:
            missing_label_result += 1
            skipped += 1
            continue
        checked += 1
        if computed != expected:
            mismatches.append(
                {
                    "match_id": row.get("match_id"),
                    "bookmaker_id": row.get("bookmaker_id"),
                    "computed": computed,
                    "label": expected,
                    "raw_label": (row.get("label", {}) or {}).get("asian_result"),
                }
            )
    status = "PASS" if not mismatches and unsupported_upper_side_count == 0 else "FAIL"
    return {
        "status": status,
        "checked_rows": checked,
        "skipped_rows": skipped,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:sample_limit],
        "missing_label_result_count": missing_label_result,
        "unsupported_upper_side_count": unsupported_upper_side_count,
        "negative_time_valid_asian_count": negative_time_valid_asian_count,
    }


def audit_duplicate_fixtures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("match_id"))].append(row)
    rows_per_match = [len(group) for group in grouped.values()]
    multi_bookmaker = 0
    for group in grouped.values():
        if len({str(row.get("bookmaker_id") or "") for row in group}) > 1:
            multi_bookmaker += 1
    return {
        "status": "PASS",
        "row_count": len(rows),
        "unique_fixture_count": len(grouped),
        "max_rows_per_fixture": max(rows_per_match, default=0),
        "mean_rows_per_fixture": _mean(rows_per_match),
        "multi_bookmaker_fixture_count": multi_bookmaker,
    }


def _group_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("match_id"))].append(row)
    out = []
    for match_id, group in grouped.items():
        kickoff = min((_parse_dt(row.get("kickoff_time")) for row in group), default=datetime.min)
        out.append({"match_id": match_id, "kickoff": kickoff, "rows": group})
    out.sort(key=lambda item: (item["kickoff"], item["match_id"]))
    return out


def _advance_after_embargo(groups: list[dict[str, Any]], start_idx: int, previous_time: datetime, embargo_days: int) -> int:
    if int(embargo_days) <= 0 or previous_time == datetime.min:
        return start_idx
    cutoff = previous_time + timedelta(days=int(embargo_days))
    idx = start_idx
    while idx < len(groups) and groups[idx]["kickoff"] <= cutoff:
        idx += 1
    return idx


def _window_meta(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {"start": None, "end": None, "matches": 0, "rows": 0, "match_ids": []}
    return {
        "start": groups[0]["kickoff"].isoformat() if groups[0]["kickoff"] != datetime.min else "",
        "end": groups[-1]["kickoff"].isoformat() if groups[-1]["kickoff"] != datetime.min else "",
        "matches": len(groups),
        "rows": sum(len(item["rows"]) for item in groups),
        "match_ids": [item["match_id"] for item in groups],
    }


def validate_fold_no_fixture_overlap(fold: dict[str, Any]) -> None:
    train = set(fold.get("train_match_ids") or [])
    selection = set(fold.get("selection_match_ids") or [])
    forward = set(fold.get("forward_match_ids") or [])
    overlaps = {
        "train_selection": sorted(train & selection),
        "train_forward": sorted(train & forward),
        "selection_forward": sorted(selection & forward),
    }
    if any(overlaps.values()):
        raise P19ProtocolError(f"P19 fold has fixture overlap: {fold.get('fold_id')} {overlaps}")


def build_rolling_folds(
    rows: list[dict[str, Any]],
    n_folds: int = 5,
    min_train_matches: int = 500,
    selection_matches: int = 100,
    forward_matches: int = 100,
    embargo_days: int = 3,
) -> list[dict[str, Any]]:
    groups = _group_matches(rows)
    if min_train_matches <= 0 or selection_matches <= 0 or forward_matches <= 0:
        raise ValueError("min_train_matches, selection_matches and forward_matches must be positive")
    folds = []
    for fold_idx in range(int(n_folds)):
        train_end = int(min_train_matches) + fold_idx * int(forward_matches)
        if train_end >= len(groups):
            break
        selection_start = _advance_after_embargo(groups, train_end, groups[train_end - 1]["kickoff"], embargo_days)
        selection_end = selection_start + int(selection_matches)
        if selection_end > len(groups):
            break
        forward_start = _advance_after_embargo(groups, selection_end, groups[selection_end - 1]["kickoff"], embargo_days)
        forward_end = forward_start + int(forward_matches)
        if forward_end > len(groups):
            break
        train_meta = _window_meta(groups[:train_end])
        selection_meta = _window_meta(groups[selection_start:selection_end])
        forward_meta = _window_meta(groups[forward_start:forward_end])
        fold = {
            "fold_id": f"fold_{fold_idx + 1:02d}",
            "embargo_days": int(embargo_days),
            "train_start": train_meta["start"],
            "train_end": train_meta["end"],
            "selection_start": selection_meta["start"],
            "selection_end": selection_meta["end"],
            "forward_start": forward_meta["start"],
            "forward_end": forward_meta["end"],
            "train_matches": train_meta["matches"],
            "selection_matches": selection_meta["matches"],
            "forward_matches": forward_meta["matches"],
            "train_rows": train_meta["rows"],
            "selection_rows": selection_meta["rows"],
            "forward_rows": forward_meta["rows"],
            "train_match_ids": train_meta["match_ids"],
            "selection_match_ids": selection_meta["match_ids"],
            "forward_match_ids": forward_meta["match_ids"],
        }
        validate_fold_no_fixture_overlap(fold)
        folds.append(fold)
    return folds


def fixture_collapsed_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if _safe_bool(row.get("selected"))]
    row_payoffs = [_safe_float(row.get("model_side_payoff")) for row in selected]
    row_hits = [_safe_float(row.get("direction_hit")) for row in selected if row.get("direction_hit") not in (None, "")]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row.get("match_id"))].append(row)
    fixture_payoffs = []
    fixture_hits = []
    for group in grouped.values():
        fixture_payoffs.append(_mean(_safe_float(row.get("model_side_payoff")) for row in group))
        hits = [_safe_float(row.get("direction_hit")) for row in group if row.get("direction_hit") not in (None, "")]
        if hits:
            fixture_hits.append(_mean(hits))
    return {
        "selected_rows": len(selected),
        "selected_unique_fixtures": len(grouped),
        "row_avg_payoff": _mean(row_payoffs),
        "row_direction_accuracy_ex_push": _mean(row_hits),
        "fixture_avg_payoff": _mean(fixture_payoffs),
        "fixture_total_units": float(sum(fixture_payoffs)),
        "fixture_direction_accuracy_ex_push": _mean(fixture_hits),
    }


def max_drawdown(payoffs: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for payoff in payoffs:
        equity += float(payoff)
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return float(worst)


def build_p19_verdict_payload(
    audit_status: str,
    fold_scores: list[dict[str, Any]],
    min_fixture_payoff: float = 0.05,
    min_positive_fold_rate: float = 0.70,
    min_unique_fixtures_per_fold: int = 80,
) -> dict[str, Any]:
    positive = [row for row in fold_scores if _safe_float(row.get("fixture_avg_payoff")) > 0.0]
    positive_rate = len(positive) / len(fold_scores) if fold_scores else 0.0
    aggregate_payoff = _mean(_safe_float(row.get("fixture_avg_payoff")) for row in fold_scores)
    latest = fold_scores[-1] if fold_scores else {}
    insufficient = [
        row
        for row in fold_scores
        if int(_safe_float(row.get("selected_unique_fixtures"))) < int(min_unique_fixtures_per_fold)
    ]
    if audit_status != "PASS":
        verdict = "P19_ROLLING_BACKTEST_BLOCKED_AUDIT_FAIL"
    elif not fold_scores:
        verdict = "P19_ROLLING_BACKTEST_NO_FOLDS"
    elif insufficient:
        verdict = "P19_ROLLING_BACKTEST_INSUFFICIENT_FIXTURES"
    elif _safe_float(latest.get("fixture_avg_payoff")) < float(min_fixture_payoff):
        verdict = "P19_ROLLING_BACKTEST_FAIL_LATEST_DECAY"
    elif aggregate_payoff <= 0.0:
        verdict = "P19_ROLLING_BACKTEST_FAIL_NEGATIVE_AGGREGATE"
    elif positive_rate < float(min_positive_fold_rate):
        verdict = "P19_ROLLING_BACKTEST_FAIL_UNSTABLE_FOLDS"
    else:
        verdict = "P19_ROLLING_BACKTEST_PASS_CANDIDATE_EDGE"
    return {
        "phase": "P19",
        "task": "rolling_retrain_backtest_verdict",
        "audit_status": audit_status,
        "fold_count": len(fold_scores),
        "positive_fold_rate": positive_rate,
        "aggregate_fixture_avg_payoff": aggregate_payoff,
        "latest_fold": latest,
        "insufficient_fixture_folds": insufficient,
        "rules": {
            "min_fixture_payoff": float(min_fixture_payoff),
            "min_positive_fold_rate": float(min_positive_fold_rate),
            "min_unique_fixtures_per_fold": int(min_unique_fixtures_per_fold),
        },
        "verdict": verdict,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_fold_split_files(out_dir: str | Path, folds: list[dict[str, Any]]) -> None:
    base = Path(out_dir)
    for fold in folds:
        fold_dir = base / str(fold["fold_id"])
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "selection", "forward"):
            ids = [str(match_id) for match_id in fold.get(f"{split_name}_match_ids", [])]
            (fold_dir / f"{split_name}_match_ids.txt").write_text(
                "".join(f"{match_id}\n" for match_id in ids),
                encoding="utf-8",
            )


def write_audit_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# P19 Protocol Audit",
        "",
        f"Status: `{payload.get('status')}`",
        "",
        f"- rows: `{payload.get('row_count')}`",
        f"- fixtures: `{payload.get('duplicate_fixtures', {}).get('unique_fixture_count')}`",
        f"- feature blacklist: `{payload.get('feature_blacklist', {}).get('status')}`",
        f"- AH settlement: `{payload.get('ah_settlement', {}).get('status')}`",
        f"- AH settlement mismatches: `{payload.get('ah_settlement', {}).get('mismatch_count')}`",
        f"- negative-time valid AH events: `{payload.get('ah_settlement', {}).get('negative_time_valid_asian_count')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_audit_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_names = p14_selected_feature_names("euro,asian,ou")
    feature_blacklist = scan_feature_blacklist(feature_names)
    feature_defs_blacklist = scan_feature_blacklist([name for name, _ in FEATURE_DEFS])
    ah_settlement = audit_ah_settlement(rows)
    duplicate_fixtures = audit_duplicate_fixtures(rows)
    status = "PASS"
    if feature_blacklist["status"] != "PASS" or feature_defs_blacklist["status"] != "PASS" or ah_settlement["status"] != "PASS":
        status = "FAIL"
    return {
        "phase": "P19",
        "task": "protocol_preflight_audit",
        "status": status,
        "row_count": len(rows),
        "feature_blacklist": feature_blacklist,
        "feature_defs_blacklist": feature_defs_blacklist,
        "ah_settlement": ah_settlement,
        "duplicate_fixtures": duplicate_fixtures,
        "input_policy": {"no_test_split_loaded": True, "no_model_training": True},
    }


def load_rows_for_split_paths(data_path: str, split_paths: list[str]) -> list[dict[str, Any]]:
    assert_no_test_paths([data_path, *split_paths])
    ids: set[str] = set()
    for path in split_paths:
        ids.update(load_split_ids(path))
    return load_rows_for_ids(data_path, ids)


def run_audit(args: argparse.Namespace) -> None:
    rows = load_rows_for_split_paths(args.data, _parse_csv_strings(args.ids))
    payload = build_audit_payload(rows)
    out_dir = Path(args.out_dir)
    write_json(out_dir / "p19_protocol_audit.json", payload)
    write_audit_md(out_dir / "p19_protocol_audit.md", payload)
    print(f"P19 audit status={payload['status']} rows={payload['row_count']} report={out_dir / 'p19_protocol_audit.md'}")


def run_build_folds(args: argparse.Namespace) -> None:
    rows = load_rows_for_split_paths(args.data, _parse_csv_strings(args.ids))
    folds = build_rolling_folds(
        rows,
        n_folds=args.n_folds,
        min_train_matches=args.min_train_matches,
        selection_matches=args.selection_matches,
        forward_matches=args.forward_matches,
        embargo_days=args.embargo_days,
    )
    payload = {
        "phase": "P19",
        "task": "rolling_fold_manifest",
        "fold_count": len(folds),
        "folds": folds,
        "input_policy": {"no_test_split_loaded": True, "split_by_fixture_time": True},
    }
    out_dir = Path(args.out_dir)
    write_json(out_dir / "fold_manifest.json", payload)
    write_fold_split_files(out_dir, folds)
    print(f"P19 folds={len(folds)} report={out_dir / 'fold_manifest.json'}")


def run_score_detail(args: argparse.Namespace) -> None:
    assert_no_test_paths([args.detail_csv, args.out_dir])
    with Path(args.detail_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    score = fixture_collapsed_score(rows)
    payload = build_p19_verdict_payload(
        audit_status=args.audit_status,
        fold_scores=[{"fold_id": "existing_detail", **score}],
        min_unique_fixtures_per_fold=args.min_unique_fixtures,
    )
    out_dir = Path(args.out_dir)
    write_json(out_dir / "p19_existing_detail_score.json", {"score": score, "verdict": payload})
    print(f"P19 existing detail fixture_payoff={score['fixture_avg_payoff']:.6f} verdict={payload['verdict']}")


def _parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P19 rolling backtest protocol utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Run no-test feature/settlement/duplicate audits")
    audit.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    audit.add_argument("--ids", default="data/odds_real/splits_v6/train_match_ids.txt,data/odds_real/splits_v6/val_match_ids.txt")
    audit.add_argument("--out-dir", default="runs/p19_rolling_backtest/audits")

    folds = sub.add_parser("build-folds", help="Build chronological rolling fold manifest")
    folds.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    folds.add_argument("--ids", default="data/odds_real/splits_v6/train_match_ids.txt,data/odds_real/splits_v6/val_match_ids.txt")
    folds.add_argument("--n-folds", type=int, default=5)
    folds.add_argument("--min-train-matches", type=int, default=6000)
    folds.add_argument("--selection-matches", type=int, default=600)
    folds.add_argument("--forward-matches", type=int, default=600)
    folds.add_argument("--embargo-days", type=int, default=3)
    folds.add_argument("--out-dir", default="runs/p19_rolling_backtest/folds")

    score = sub.add_parser("score-detail", help="Fixture-collapse an existing selected-detail CSV")
    score.add_argument("--detail-csv", required=True)
    score.add_argument("--audit-status", default="PASS")
    score.add_argument("--min-unique-fixtures", type=int, default=80)
    score.add_argument("--out-dir", default="runs/p19_rolling_backtest/scores")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "audit":
        run_audit(args)
    elif args.command == "build-folds":
        run_build_folds(args)
    elif args.command == "score-detail":
        run_score_detail(args)


if __name__ == "__main__":
    main()
