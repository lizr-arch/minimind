"""P18.8 forward-selection check for AH agreement strategy.

This is a conservative no-test validation: choose an AH agreement gate using
earlier validation rows only, then score the chosen gate on later validation
rows. It helps distinguish a genuinely stable strategy from a gate that only
looked good after full-validation inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_agreement_strategy import (
    DEFAULT_COVER_VARIANTS,
    DEFAULT_DIRECT_VARIANTS,
    DEFAULT_MODES,
    agreement_stats,
)
from tools.p18_ah_robustness_slices import build_agreement_detail_rows, load_contexts_with_metadata
from tools.p18_ah_threshold_sweep import load_ah_head_predictions


DEFAULT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25]


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def assign_forward_periods(rows: list[dict[str, Any]], train_fraction: float = 0.5) -> dict[int, str]:
    valid = [
        (int(row["label_idx"]), str(row.get("kickoff_time") or ""))
        for row in rows
        if "label_idx" in row and str(row.get("kickoff_time") or "")
    ]
    valid.sort(key=lambda item: item[1])
    cut = int(len(valid) * float(train_fraction))
    period_by_idx: dict[int, str] = {}
    for pos, (label_idx, _) in enumerate(valid):
        period_by_idx[label_idx] = "selection" if pos < cut else "forward"
    return period_by_idx


def score_candidate_rows(rows: list[dict[str, Any]], min_selected: int = 50) -> dict[str, Any]:
    selected = [row for row in rows if bool(row.get("selected"))]
    hits = [_safe_float(row.get("direction_hit")) for row in selected if row.get("direction_hit") is not None]
    selected_n = len(selected)
    return {
        "total_n": len(rows),
        "selected_n": selected_n,
        "coverage": selected_n / len(rows) if rows else 0.0,
        "model_side_avg_units": _mean([_safe_float(row.get("model_side_units")) for row in selected]),
        "model_side_avg_payoff": _mean([_safe_float(row.get("model_side_payoff")) for row in selected]),
        "direction_accuracy_ex_push": _mean(hits),
        "eligible": selected_n >= int(min_selected),
    }


def split_score_candidate_rows(rows: list[dict[str, Any]], min_selected: int) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_rows = [row for row in rows if row.get("period") == "selection"]
    forward_rows = [row for row in rows if row.get("period") == "forward"]
    return score_candidate_rows(selection_rows, min_selected), score_candidate_rows(forward_rows, min_selected)


def select_best_candidate(candidate_rows: list[dict[str, Any]], min_selected: int = 50) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    scored = []
    for candidate_id, rows in grouped.items():
        selection, forward = split_score_candidate_rows(rows, min_selected)
        if not selection["eligible"]:
            continue
        first = rows[0]
        scored.append(
            {
                "candidate_id": candidate_id,
                "cover_variant": first.get("cover_variant"),
                "direct_variant": first.get("direct_variant"),
                "mode": first.get("mode"),
                "threshold": _safe_float(first.get("threshold")),
                "selection_score": selection,
                "forward_score": forward,
            }
        )
    if not scored:
        return {}
    return max(
        scored,
        key=lambda row: (
            float(row["selection_score"]["model_side_avg_payoff"]),
            float(row["selection_score"]["direction_accuracy_ex_push"]),
            int(row["selection_score"]["selected_n"]),
        ),
    )


def build_forward_selection_payload(
    candidate_rows: list[dict[str, Any]],
    selected_candidate: dict[str, Any],
    min_selected: int = 50,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not selected_candidate:
        return {
            "phase": "P18.8",
            "task": "forward_selection_agreement_gate",
            "metadata": metadata or {},
            "selected_candidate": {},
            "selection_score": {},
            "forward_score": {},
            "verdict": "P18_FORWARD_SELECTION_NO_ELIGIBLE_CANDIDATE",
        }
    selection = selected_candidate["selection_score"]
    forward = selected_candidate["forward_score"]
    if forward["eligible"] and forward["model_side_avg_payoff"] > 0.0 and forward["direction_accuracy_ex_push"] >= 0.50:
        verdict = "P18_FORWARD_SELECTION_PASS"
    elif forward["selected_n"] > 0:
        verdict = "P18_FORWARD_SELECTION_DECAY"
    else:
        verdict = "P18_FORWARD_SELECTION_INSUFFICIENT_FORWARD"
    return {
        "phase": "P18.8",
        "task": "forward_selection_agreement_gate",
        "metadata": metadata or {},
        "input_policy": {
            "no_test_split_loaded": True,
            "select_on_early_validation_score_on_late_validation": True,
            "not_betting_advice": True,
        },
        "selected_candidate": {
            "candidate_id": selected_candidate["candidate_id"],
            "cover_variant": selected_candidate.get("cover_variant"),
            "direct_variant": selected_candidate.get("direct_variant"),
            "mode": selected_candidate.get("mode"),
            "threshold": selected_candidate.get("threshold"),
        },
        "selection_score": selection,
        "forward_score": forward,
        "min_selected": int(min_selected),
        "verdict": verdict,
    }


def candidate_id(cover_variant: str, direct_variant: str, mode: str, threshold: float) -> str:
    return f"{cover_variant}__{direct_variant}__{mode}__t{float(threshold):.3f}"


def build_candidate_detail_rows(
    contexts: list[dict[str, Any]],
    cover_by_seed: dict[int, list[tuple[float, ...]]],
    direct_by_seed: dict[int, list[tuple[float, ...]]],
    cover_variant: str,
    direct_variant: str,
    threshold: float,
    mode: str,
    period_by_idx: dict[int, str],
) -> list[dict[str, Any]]:
    cid = candidate_id(cover_variant, direct_variant, mode, threshold)
    rows = build_agreement_detail_rows(cover_by_seed, direct_by_seed, contexts, threshold, mode)
    for row in rows:
        row["candidate_id"] = cid
        row["cover_variant"] = cover_variant
        row["direct_variant"] = direct_variant
        row["period"] = period_by_idx.get(int(row["label_idx"]), "unknown")
    return rows


def summarize_candidates(candidate_rows: list[dict[str, Any]], min_selected: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    summaries = []
    for cid, rows in sorted(grouped.items()):
        selection, forward = split_score_candidate_rows(rows, min_selected)
        first = rows[0]
        summaries.append(
            {
                "candidate_id": cid,
                "cover_variant": first.get("cover_variant"),
                "direct_variant": first.get("direct_variant"),
                "mode": first.get("mode"),
                "threshold": _safe_float(first.get("threshold")),
                "selection_selected_n": selection["selected_n"],
                "selection_payoff": selection["model_side_avg_payoff"],
                "selection_dir_acc": selection["direction_accuracy_ex_push"],
                "selection_eligible": selection["eligible"],
                "forward_selected_n": forward["selected_n"],
                "forward_payoff": forward["model_side_avg_payoff"],
                "forward_dir_acc": forward["direction_accuracy_ex_push"],
                "forward_eligible": forward["eligible"],
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report_md(path: Path, payload: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    selected = payload.get("selected_candidate", {})
    lines = [
        "# P18.8 AH Agreement Forward Selection",
        "",
        f"Verdict: `{payload.get('verdict')}`",
        "",
        "## Selected Candidate",
        f"- candidate: `{selected.get('candidate_id')}`",
        f"- selection payoff: `{payload.get('selection_score', {}).get('model_side_avg_payoff')}`",
        f"- forward payoff: `{payload.get('forward_score', {}).get('model_side_avg_payoff')}`",
        f"- forward selected: `{payload.get('forward_score', {}).get('selected_n')}`",
        "",
        "## Top Forward Candidates",
    ]
    for row in sorted(summaries, key=lambda item: float(item["forward_payoff"]), reverse=True)[:12]:
        lines.append(
            f"- {row['candidate_id']}: selection={float(row['selection_payoff']):.6f}, "
            f"forward={float(row['forward_payoff']):.6f}, "
            f"forward_n={row['forward_selected_n']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P18.8 forward-selection AH agreement validation")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--cover-root", default="runs/p18_ah_cover_aux_matrix")
    parser.add_argument("--direct-root", default="runs/p18_ah_direct_aux_matrix")
    parser.add_argument("--cover-variants", default=",".join(DEFAULT_COVER_VARIANTS))
    parser.add_argument("--direct-variants", default=",".join(DEFAULT_DIRECT_VARIANTS))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--min-selected", type=int, default=50)
    parser.add_argument("--out-dir", default="runs/p18_ah_forward_selection")
    parser.add_argument("--no-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P18.8 forward selection without --no-test")
    seeds = _parse_csv_ints(args.seeds)
    cover_variants = _parse_csv_strings(args.cover_variants)
    direct_variants = _parse_csv_strings(args.direct_variants)
    thresholds = _parse_csv_floats(args.thresholds)
    modes = _parse_csv_strings(args.modes)
    contexts = load_contexts_with_metadata(args.data, args.val_ids)
    context_by_idx = {int(row["label_idx"]): row for row in contexts}
    period_by_idx = assign_forward_periods(contexts, args.train_fraction)
    all_rows: list[dict[str, Any]] = []
    cover_cache: dict[str, dict[int, list[tuple[float, ...]]]] = {}
    direct_cache: dict[str, dict[int, list[tuple[float, ...]]]] = {}
    for cover_variant in cover_variants:
        cover_cache[cover_variant] = load_ah_head_predictions(Path(args.cover_root), cover_variant, seeds, context_by_idx)
    for direct_variant in direct_variants:
        direct_cache[direct_variant] = load_ah_head_predictions(Path(args.direct_root), direct_variant, seeds, context_by_idx)
    for cover_variant in cover_variants:
        for direct_variant in direct_variants:
            for mode in modes:
                for threshold in thresholds:
                    all_rows.extend(
                        build_candidate_detail_rows(
                            contexts,
                            cover_cache[cover_variant],
                            direct_cache[direct_variant],
                            cover_variant,
                            direct_variant,
                            threshold,
                            mode,
                            period_by_idx,
                        )
                    )
    summaries = summarize_candidates(all_rows, args.min_selected)
    selected = select_best_candidate(all_rows, args.min_selected)
    metadata = {
        "cover_variants": cover_variants,
        "direct_variants": direct_variants,
        "modes": modes,
        "thresholds": thresholds,
        "seeds": seeds,
        "train_fraction": float(args.train_fraction),
        "min_selected": int(args.min_selected),
    }
    payload = build_forward_selection_payload(all_rows, selected, args.min_selected, metadata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "p18_forward_candidate_summary.csv", summaries)
    (out_dir / "p18_forward_selection_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report_md(out_dir / "p18_forward_selection_report.md", payload, summaries)
    print(
        "P18.8 forward selection: "
        f"verdict={payload['verdict']} "
        f"candidate={payload.get('selected_candidate', {}).get('candidate_id')} "
        f"forward_payoff={payload.get('forward_score', {}).get('model_side_avg_payoff')}"
    )
    print(f"Report saved to {out_dir / 'p18_forward_selection_report.md'}")


if __name__ == "__main__":
    main()
