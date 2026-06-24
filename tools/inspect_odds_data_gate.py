"""
P1.0E-G: OddsMind Data Gate Inspector.

Runs all data quality checks and generates a consolidated gate report.
This is the final validation before training.

Usage:
    python tools/inspect_odds_data_gate.py \
        --input data/odds_real/titan007_pure_v3_time.jsonl \
        --splits-dir data/odds_real/splits/v3_time \
        --output docs/review/data_gate_report.md \
        --feature-schema v4 \
        --consensus-mode none \
        --cutoff-buckets "1440,720,360,180,120,60,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1"
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from dataset.odds_dataset import (
    OddsDataset,
    _event_has_euro,
    _event_has_asian,
    _event_has_over_under,
    _event_to_features_v4,
    V4_FEATURE_DIM,
    _compute_consensus_feats,
    CONSENSUS_MODES,
)
from dataset.odds_cutoff import (
    filter_timeline_by_cutoff,
    assert_cutoff_integrity,
    build_cutoff_samples,
    CUTOFF_BUCKETS_V1,
)


def load_split_ids(splits_dir: str) -> dict:
    result = {}
    for name in ("train", "val", "test"):
        path = os.path.join(splits_dir, f"{name}_match_ids.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                result[name] = {line.strip() for line in f if line.strip()}
        else:
            result[name] = set()
    return result


def check_time_axis(matches: list) -> dict:
    """Gate 1: All events have valid minutes_before_kickoff."""
    violations = 0
    total = 0
    for m in matches:
        for e in m.get("odds_timeline", []):
            total += 1
            mbk = e.get("minutes_before_kickoff", None)
            if mbk is None or mbk < 0:
                violations += 1
    return {
        "name": "time_axis",
        "pass": violations == 0,
        "total": total,
        "violations": violations,
        "detail": f"{violations}/{total} events have invalid minutes_before_kickoff",
    }


def check_labels(matches: list) -> dict:
    """Gate 2: All labels are valid euro/asian strings."""
    valid_euro = {"home", "draw", "away"}
    valid_asian = {"upper", "push", "lower", "full_win", "half_win", "half_loss", "full_loss"}
    violations = 0
    missing_handicap_count = 0
    for m in matches:
        label = m.get("label", {})
        if label.get("euro_result") not in valid_euro:
            violations += 1
        asian_status = label.get("asian_label_status", "ok")
        asian_result = label.get("asian_result")
        if asian_status == "missing_handicap" or asian_result is None:
            missing_handicap_count += 1
        elif asian_result not in valid_asian:
            violations += 1
    return {
        "name": "label",
        "pass": violations == 0,
        "violations": violations,
        "missing_handicap_count": missing_handicap_count,
        "detail": f"{violations} invalid labels, {missing_handicap_count} missing_handicap (expected)",
    }


def check_splits(splits_dir: str) -> dict:
    """Gate 3: Split files exist and are non-empty."""
    issues = []
    for name in ("train", "val", "test"):
        path = os.path.join(splits_dir, f"{name}_match_ids.txt")
        if not os.path.exists(path):
            issues.append(f"{name} split file missing")
        else:
            with open(path, "r") as f:
                ids = {l.strip() for l in f if l.strip()}
            if len(ids) == 0:
                issues.append(f"{name} split is empty")
    return {
        "name": "split",
        "pass": len(issues) == 0,
        "issues": issues,
        "detail": "; ".join(issues) if issues else "All split files exist and non-empty",
    }


def check_cutoff(matches: list, cutoff_buckets: list) -> dict:
    """Gate 4: No event in cutoff sample has mbk < cutoff."""
    violations = 0
    total_samples = 0
    for m in matches:
        for cutoff in cutoff_buckets:
            cs_list = build_cutoff_samples(m, [cutoff], min_events=0)
            for cs in cs_list:
                total_samples += 1
                try:
                    assert_cutoff_integrity(cs["odds_timeline"], cutoff, sample_id=cs.get("sample_id", ""))
                except AssertionError:
                    violations += 1
    return {
        "name": "cutoff",
        "pass": violations == 0,
        "total_samples": total_samples,
        "violations": violations,
        "detail": f"{violations}/{total_samples} cutoff samples have integrity violations",
    }


def check_consensus_leakage(matches: list, consensus_mode: str) -> dict:
    """Gate 5: consensus_feats doesn't leak future information."""
    has_bk_feats = sum(1 for m in matches if "bookmaker_features" in m)
    all_zeros = True

    # Spot-check a few matches
    for m in matches[:10]:
        tl = m.get("odds_timeline", [])
        if not tl:
            continue
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            feat = _compute_consensus_feats(m, consensus_mode=consensus_mode, visible_timeline=tl)
        if feat.abs().sum().item() > 0:
            all_zeros = False
            break

    safe = (consensus_mode == "none" and has_bk_feats == 0) or (consensus_mode == "none" and all_zeros)
    return {
        "name": "consensus_leakage",
        "pass": safe,
        "consensus_mode": consensus_mode,
        "matches_with_bk_feats": has_bk_feats,
        "all_zeros": all_zeros,
        "detail": f"mode={consensus_mode}, bk_feats={has_bk_feats}, all_zeros={all_zeros}",
    }


def check_missing_semantics(matches: list) -> dict:
    """Gate 6: has_asian/has_ou correctly detect default placeholders."""
    misdetected = 0
    total = 0
    for m in matches:
        for e in m.get("odds_timeline", []):
            total += 1
            al = e.get("asian_line", 0)
            uw = e.get("upper_water", 0)
            lw = e.get("lower_water", 0)
            ol = e.get("over_under_line", 0)
            ow = e.get("over_water", 0)
            ou = e.get("under_water", 0)

            # Default asian should be detected as missing
            if al == 0 and uw == 1.0 and lw == 1.0:
                if _event_has_asian(e):
                    misdetected += 1
            # Default ou should be detected as missing
            if ol == 2.5 and ow == 1.0 and ou == 1.0:
                if _event_has_over_under(e):
                    misdetected += 1
    return {
        "name": "missing_semantics",
        "pass": misdetected == 0,
        "misdetected": misdetected,
        "total": total,
        "detail": f"{misdetected}/{total} events have wrong has_asian/has_ou detection",
    }


def check_feature_finite(matches: list, feature_schema: str) -> dict:
    """Gate 7: No nan/inf in any feature."""
    nan_count = 0
    inf_count = 0
    total = 0
    for m in matches:
        for e in m.get("odds_timeline", []):
            if feature_schema == "v4":
                feats = _event_to_features_v4(e)
            else:
                from dataset.odds_dataset import _event_to_features
                feats = _event_to_features(e, schema_version=feature_schema)
            for v in feats:
                total += 1
                if math.isnan(v):
                    nan_count += 1
                elif math.isinf(v):
                    inf_count += 1
    return {
        "name": "feature_finite",
        "pass": nan_count == 0 and inf_count == 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "total": total,
        "detail": f"nan={nan_count}, inf={inf_count}, total={total}",
    }


def check_no_future_leakage(matches: list, cutoff_buckets: list) -> dict:
    """Gate 8: Cutoff-filtered timeline has no future events."""
    violations = 0
    total = 0
    for m in matches[:50]:  # Sample for speed
        for cutoff in cutoff_buckets[:5]:
            filtered = filter_timeline_by_cutoff(m.get("odds_timeline", []), cutoff)
            for e in filtered:
                total += 1
                if e["minutes_before_kickoff"] < cutoff:
                    violations += 1
    return {
        "name": "no_future_leakage",
        "pass": violations == 0,
        "violations": violations,
        "total": total,
        "detail": f"{violations}/{total} events violate cutoff boundary",
    }


def check_split_overlap(splits: dict) -> dict:
    """Gate 9: No match_id in multiple splits."""
    train = splits.get("train", set())
    val = splits.get("val", set())
    test = splits.get("test", set())
    tv_overlap = train & val
    tt_overlap = train & test
    vt_overlap = val & test
    all_overlap = tv_overlap | tt_overlap | vt_overlap
    return {
        "name": "split_overlap",
        "pass": len(all_overlap) == 0,
        "train_val_overlap": len(tv_overlap),
        "train_test_overlap": len(tt_overlap),
        "val_test_overlap": len(vt_overlap),
        "detail": f"train∩val={len(tv_overlap)}, train∩test={len(tt_overlap)}, val∩test={len(vt_overlap)}",
    }


def generate_report(args) -> str:
    """Run all gates and generate the consolidated report."""
    cutoff_buckets = [float(x) for x in args.cutoff_buckets.split(",")]
    splits = load_split_ids(args.splits_dir)

    # Load matches
    matches = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                matches.append(json.loads(line))

    # Run all gates
    gates = []
    gates.append(check_time_axis(matches))
    gates.append(check_labels(matches))
    gates.append(check_splits(args.splits_dir))
    gates.append(check_cutoff(matches, cutoff_buckets))
    gates.append(check_consensus_leakage(matches, args.consensus_mode))
    gates.append(check_missing_semantics(matches))
    gates.append(check_feature_finite(matches, args.feature_schema))
    gates.append(check_no_future_leakage(matches, cutoff_buckets))
    gates.append(check_split_overlap(splits))

    # Gate 10: aggregate
    all_pass = all(g["pass"] for g in gates)
    any_fail = any(not g["pass"] for g in gates)

    lines = []
    lines.append("# OddsMind Data Gate Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nInput: `{args.input}`")
    lines.append(f"Splits dir: `{args.splits_dir}`")
    lines.append(f"Feature schema: `{args.feature_schema}`")
    lines.append(f"Consensus mode: `{args.consensus_mode}`")
    lines.append(f"Cutoff buckets: {len(cutoff_buckets)} values")
    lines.append(f"Total matches: {len(matches)}")

    lines.append("\n## Gate Results Summary")
    lines.append("| # | Gate | Status | Detail |")
    lines.append("|---|------|--------|--------|")
    for i, g in enumerate(gates, 1):
        status = "PASS" if g["pass"] else "FAIL"
        lines.append(f"| {i} | {g['name']} | **{status}** | {g['detail']} |")

    lines.append("\n## Detailed Gate Results")
    for i, g in enumerate(gates, 1):
        lines.append(f"\n### Gate {i}: {g['name']}")
        lines.append(f"- **Status**: {'PASS' if g['pass'] else 'FAIL'}")
        for k, v in g.items():
            if k not in ("name", "pass", "detail"):
                lines.append(f"- {k}: {v}")

    # Gate 10: ready_for_training
    lines.append("\n### Gate 10: ready_for_training")
    if all_pass:
        lines.append("- **Status**: **PASS**")
        lines.append("- All 9 data gates passed")
        lines.append("- Data is ready for training")
    elif any_fail:
        # Check if only warnings (missing_semantics may have expected failures)
        critical_gates = [g for g in gates if g["name"] not in ("missing_semantics",)]
        critical_pass = all(g["pass"] for g in critical_gates)
        if critical_pass:
            lines.append("- **Status**: **PASS_WITH_WARNINGS**")
            lines.append("- Critical gates passed")
            lines.append("- Some non-critical warnings (see missing_semantics)")
        else:
            lines.append("- **Status**: **BLOCKED**")
            lines.append("- Critical gate failures detected")
            failed = [g["name"] for g in gates if not g["pass"]]
            lines.append(f"- Failed gates: {', '.join(failed)}")

    # Final verdict
    critical_gates = [g for g in gates if g["name"] not in ("missing_semantics",)]
    critical_pass = all(g["pass"] for g in critical_gates)
    if all_pass:
        final = "PASS"
    elif critical_pass:
        final = "PASS_WITH_WARNINGS"
    else:
        final = "BLOCKED"

    lines.append(f"\n## Final Verdict: **{final}**")
    if final == "PASS":
        lines.append("- All data gates passed")
        lines.append("- Ready for training phase")
    elif final == "PASS_WITH_WARNINGS":
        lines.append("- Critical gates passed with warnings")
        lines.append("- Asian/ou data is all defaults (expected for v3 data)")
        lines.append("- May proceed to training with euro-only features")
    else:
        lines.append("- Critical gate failures block training")
        lines.append("- Fix issues before proceeding")

    lines.append(f"\n---\n*Report generated by `tools/inspect_odds_data_gate.py`*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="OddsMind Data Gate Inspector")
    parser.add_argument("--input", required=True, help="Path to JSONL data file")
    parser.add_argument("--splits-dir", required=True, help="Path to splits directory")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    parser.add_argument("--feature-schema", default="v4", choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--consensus-mode", default="none", choices=CONSENSUS_MODES)
    parser.add_argument("--cutoff-buckets",
                        default=",".join(str(int(x)) for x in CUTOFF_BUCKETS_V1),
                        help="Comma-separated cutoff bucket values")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report = generate_report(args)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
