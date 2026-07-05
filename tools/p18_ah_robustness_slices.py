"""P18.7 robustness slices for the agreement-gated AH strategy."""

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

from tools.p18_ah_agreement_strategy import agreement_decision
from tools.p18_ah_cover_diagnostics import build_source_context_rows
from tools.p18_ah_threshold_sweep import load_ah_head_predictions, payoff_for_side
from tools.p4_train_residual_goal_diff import load_rows_for_ids, load_split_ids


DEFAULT_SLICE_KEYS = ["overall", "seed", "line_slice", "bookmaker_id", "league_id", "season", "time_quartile"]


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def assign_time_quartiles(rows: list[dict[str, Any]], n_buckets: int = 4) -> None:
    valid = [(idx, str(row.get("kickoff_time") or "")) for idx, row in enumerate(rows) if str(row.get("kickoff_time") or "")]
    valid.sort(key=lambda item: item[1])
    for row in rows:
        row["time_quartile"] = "unknown"
    total = len(valid)
    for pos, (idx, _) in enumerate(valid):
        bucket = min(int(pos * int(n_buckets) / max(total, 1)) + 1, int(n_buckets))
        rows[idx]["time_quartile"] = f"Q{bucket}"


def build_agreement_detail_rows(
    cover_by_seed: dict[int, list[tuple[float, ...]]],
    direct_by_seed: dict[int, list[tuple[float, ...]]],
    contexts: list[dict[str, Any]],
    threshold: float,
    mode: str,
) -> list[dict[str, Any]]:
    detail_rows = []
    contexts_by_idx = {int(row["label_idx"]): row for row in contexts if "label_idx" in row}
    for seed in sorted(set(cover_by_seed) & set(direct_by_seed)):
        cover_rows = cover_by_seed[seed]
        direct_rows = direct_by_seed[seed]
        for idx, (cover, direct) in enumerate(zip(cover_rows, direct_rows)):
            ctx = contexts_by_idx.get(idx, {})
            cover_expected = float(cover[0])
            direct_expected = float(direct[0])
            actual = float(cover[1])
            upper_water = float(cover[2]) if len(cover) >= 3 else 1.0
            lower_water = float(cover[3]) if len(cover) >= 4 else 1.0
            decision = agreement_decision(cover_expected, direct_expected, threshold, mode)
            selected = decision != "hold"
            units = actual if decision == "upper" else (-actual if decision == "lower" else 0.0)
            payoff = payoff_for_side(actual, decision, upper_water, lower_water) if selected else 0.0
            if not selected or actual == 0:
                hit = None
            elif actual > 0 and decision == "upper":
                hit = 1.0
            elif actual < 0 and decision == "lower":
                hit = 1.0
            else:
                hit = 0.0
            detail_rows.append(
                {
                    "seed": seed,
                    "label_idx": idx,
                    "match_id": ctx.get("match_id", ""),
                    "bookmaker_id": ctx.get("bookmaker_id", "unknown"),
                    "league_id": ctx.get("league_id", "unknown"),
                    "season": ctx.get("season", "unknown"),
                    "competition_type": ctx.get("competition_type", "unknown"),
                    "kickoff_time": ctx.get("kickoff_time", ""),
                    "line_slice": ctx.get("line_slice", ctx.get("slice", "unknown")),
                    "asian_line": ctx.get("asian_line", ""),
                    "actual_upper_units": actual,
                    "cover_expected_upper_units": cover_expected,
                    "direct_expected_upper_units": direct_expected,
                    "combined_expected_upper_units": (cover_expected + direct_expected) / 2.0,
                    "threshold": float(threshold),
                    "mode": mode,
                    "decision": decision,
                    "selected": selected,
                    "model_side_units": units,
                    "model_side_payoff": payoff,
                    "direction_hit": hit,
                }
            )
    assign_time_quartiles(detail_rows)
    return detail_rows


def summarize_by_slice(rows: list[dict[str, Any]], slice_key: str, min_selected: int = 50) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if slice_key == "overall":
        grouped["overall"] = rows
    else:
        for row in rows:
            grouped.setdefault(str(row.get(slice_key, "unknown") or "unknown"), []).append(row)
    summaries = []
    for value, group in sorted(grouped.items(), key=lambda item: item[0]):
        selected = [row for row in group if bool(row.get("selected"))]
        hits = [_safe_float(row.get("direction_hit")) for row in selected if row.get("direction_hit") is not None]
        payoff = _mean([_safe_float(row.get("model_side_payoff")) for row in selected])
        dir_acc = _mean(hits)
        if len(selected) < int(min_selected):
            status = "INSUFFICIENT"
        elif payoff > 0.0 and dir_acc >= 0.50:
            status = "PASS"
        else:
            status = "WEAK_SLICE"
        summaries.append(
            {
                "slice_key": slice_key,
                "slice_value": value,
                "total_n": len(group),
                "selected_n": len(selected),
                "coverage": len(selected) / len(group) if group else 0.0,
                "model_side_avg_units": _mean([_safe_float(row.get("model_side_units")) for row in selected]),
                "model_side_avg_payoff": payoff,
                "direction_accuracy_ex_push": dir_acc,
                "upper_pick_rate": sum(1 for row in selected if row.get("decision") == "upper") / len(selected) if selected else 0.0,
                "lower_pick_rate": sum(1 for row in selected if row.get("decision") == "lower") / len(selected) if selected else 0.0,
                "status": status,
            }
        )
    return summaries


def load_contexts_with_metadata(data_path: str, val_ids_path: str) -> list[dict[str, Any]]:
    source_rows = load_rows_for_ids(data_path, load_split_ids(val_ids_path))
    contexts = build_source_context_rows(source_rows)
    enriched = []
    for ctx in contexts:
        row = source_rows[int(ctx["row_idx"])]
        item = dict(ctx)
        item.update(
            {
                "line_slice": ctx.get("slice", "unknown"),
                "bookmaker_id": row.get("bookmaker_id") or row.get("bookmaker") or "unknown",
                "league_id": row.get("league_id") or row.get("league") or row.get("competition_type") or "unknown",
                "season": row.get("season") or "unknown",
                "competition_type": row.get("competition_type") or "unknown",
                "kickoff_time": row.get("kickoff_time") or "",
            }
        )
        enriched.append(item)
    return enriched


def build_payload(
    detail_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    min_selected: int,
) -> dict[str, Any]:
    overall = next((row for row in summary_rows if row["slice_key"] == "overall" and row["slice_value"] == "overall"), {})
    weak = [
        row
        for row in summary_rows
        if row["status"] == "WEAK_SLICE" and int(row["selected_n"]) >= int(min_selected)
    ]
    if weak:
        verdict = "P18_AGREEMENT_HAS_SLICE_RISK"
    elif overall and overall.get("status") == "PASS":
        verdict = "P18_AGREEMENT_ROBUST_SLICE_CHECK_PASS"
    else:
        verdict = "P18_AGREEMENT_ROBUSTNESS_INSUFFICIENT"
    return {
        "phase": "P18.7",
        "task": "agreement_gated_ah_robustness_slices",
        "metadata": metadata,
        "input_policy": {
            "no_test_split_loaded": True,
            "strategy_validation_only_no_new_training": True,
            "not_betting_advice": True,
        },
        "overall": overall,
        "weak_slices": weak,
        "summary_rows": summary_rows,
        "detail_count": len(detail_rows),
        "verdict": verdict,
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


def write_report_md(path: Path, payload: dict[str, Any]) -> None:
    overall = payload.get("overall", {})
    lines = [
        "# P18.7 AH Agreement Robustness Slices",
        "",
        f"Verdict: `{payload.get('verdict')}`",
        "",
        "## Overall",
        (
            f"- selected={overall.get('selected_n')} / total={overall.get('total_n')}, "
            f"coverage={float(overall.get('coverage', 0.0)):.3f}, "
            f"payoff={float(overall.get('model_side_avg_payoff', 0.0)):.6f}, "
            f"dir_acc={float(overall.get('direction_accuracy_ex_push', 0.0)):.6f}"
        ),
        "",
        "## Weak Slices",
    ]
    weak = payload.get("weak_slices", [])
    if not weak:
        lines.append("- none above min_selected")
    for row in weak[:20]:
        lines.append(
            f"- {row['slice_key']}={row['slice_value']}: selected={row['selected_n']}, "
            f"payoff={float(row['model_side_avg_payoff']):.6f}, "
            f"dir_acc={float(row['direction_accuracy_ex_push']):.6f}"
        )
    lines.extend(["", "## Top Slice Summaries"])
    for row in sorted(payload.get("summary_rows", []), key=lambda item: (item["slice_key"], -float(item["selected_n"])))[:40]:
        lines.append(
            f"- {row['slice_key']}={row['slice_value']}: selected={row['selected_n']}, "
            f"coverage={float(row['coverage']):.3f}, payoff={float(row['model_side_avg_payoff']):.6f}, "
            f"status={row['status']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P18.7 AH agreement robustness slices")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--cover-root", default="runs/p18_ah_cover_aux_matrix")
    parser.add_argument("--direct-root", default="runs/p18_ah_direct_aux_matrix")
    parser.add_argument("--cover-variant", default="p18_ahw_003")
    parser.add_argument("--direct-variant", default="p18d_direct030")
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--mode", default="both", choices=("any", "both"))
    parser.add_argument("--slice-keys", default=",".join(DEFAULT_SLICE_KEYS))
    parser.add_argument("--min-selected", type=int, default=50)
    parser.add_argument("--out-dir", default="runs/p18_ah_robustness_slices")
    parser.add_argument("--no-test", action="store_true")
    return parser


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P18.7 robustness without --no-test")
    seeds = _parse_csv_ints(args.seeds)
    contexts = load_contexts_with_metadata(args.data, args.val_ids)
    context_by_idx = {int(row["label_idx"]): row for row in contexts}
    cover_by_seed = load_ah_head_predictions(Path(args.cover_root), args.cover_variant, seeds, context_by_idx)
    direct_by_seed = load_ah_head_predictions(Path(args.direct_root), args.direct_variant, seeds, context_by_idx)
    detail_rows = build_agreement_detail_rows(cover_by_seed, direct_by_seed, contexts, args.threshold, args.mode)
    summary_rows = []
    for key in _parse_csv_strings(args.slice_keys):
        summary_rows.extend(summarize_by_slice(detail_rows, key, args.min_selected))
    metadata = {
        "cover_variant": args.cover_variant,
        "direct_variant": args.direct_variant,
        "seeds": seeds,
        "threshold": float(args.threshold),
        "mode": args.mode,
        "min_selected": int(args.min_selected),
        "slice_keys": _parse_csv_strings(args.slice_keys),
    }
    payload = build_payload(detail_rows, summary_rows, metadata, args.min_selected)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "p18_ah_robustness_detail.csv", detail_rows)
    write_csv(out_dir / "p18_ah_robustness_summary.csv", summary_rows)
    (out_dir / "p18_ah_robustness_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report_md(out_dir / "p18_ah_robustness_report.md", payload)
    print(
        "P18.7 robustness: "
        f"verdict={payload['verdict']} "
        f"overall_payoff={float(payload['overall'].get('model_side_avg_payoff', 0.0)):.6f}"
    )
    print(f"Report saved to {out_dir / 'p18_ah_robustness_report.md'}")


if __name__ == "__main__":
    main()
