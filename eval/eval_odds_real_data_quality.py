"""
OddsMind Real Data Quality Audit (P0.9)

Audits an OddsMind JSONL file for schema correctness, odds validity,
label distribution, timeline statistics, and split readiness.

Usage:
    python eval/eval_odds_real_data_quality.py \
        --data data/odds_imported/xxx.jsonl \
        --out-json runs/quality_audit.json
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.odds_import_schema import validate_imported_match
from dataset.odds_split import parse_kickoff_time


def audit_jsonl(jsonl_path: str, asian_label_mode: str = "5class") -> dict:
    """
    Audit a JSONL file and return a quality report dict.
    """
    report = {
        "num_matches": 0,
        "schema_errors": 0,
        "invalid_odds": 0,
        "invalid_labels": 0,
        "warnings": [],
        "timeline": {
            "lengths": [],
            "single_event_matches": 0,
            "multi_event_matches": 0,
            "closing_only_ratio": 0.0,
            "cutoff_available": {"90": 0, "60": 0, "30": 0, "0": 0},
        },
        "label_distribution": {
            "euro_result": {},
            "asian_result": {},
        },
        "time_range": {"min": "", "max": ""},
        "match_ids": [],
    }

    matches = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                report["schema_errors"] += 1
                continue
            matches.append(m)

    report["num_matches"] = len(matches)
    if not matches:
        report["warnings"].append("No valid matches found")
        return report

    # Schema validation
    for m in matches:
        errs = validate_imported_match(m)
        if errs:
            report["schema_errors"] += len(errs)
            for e in errs[:3]:  # Sample
                if e not in report["warnings"]:
                    report["warnings"].append(e)

    # Timeline stats
    lengths = []
    for m in matches:
        tl = m.get("odds_timeline", [])
        lengths.append(len(tl))
        if len(tl) == 1:
            report["timeline"]["single_event_matches"] += 1
        else:
            report["timeline"]["multi_event_matches"] += 1

        # Cutoff availability
        mbks = [e["minutes_before_kickoff"] for e in tl if "minutes_before_kickoff" in e]
        for cutoff in [90, 60, 30, 0]:
            if any(mbk >= cutoff for mbk in mbks):
                report["timeline"]["cutoff_available"][str(cutoff)] += 1

    report["timeline"]["lengths"] = lengths
    if lengths:
        report["timeline"]["avg_len"] = sum(lengths) / len(lengths)
        report["timeline"]["min_len"] = min(lengths)
        report["timeline"]["max_len"] = max(lengths)
        report["timeline"]["closing_only_ratio"] = report["timeline"]["single_event_matches"] / len(matches)

    # Label distribution
    euro_counter = Counter()
    asian_counter = Counter()
    for m in matches:
        label = m.get("label", {})
        euro_counter[label.get("euro_result", "unknown")] += 1
        asian_counter[label.get("asian_result", "unknown")] += 1
    report["label_distribution"]["euro_result"] = dict(euro_counter)
    report["label_distribution"]["asian_result"] = dict(asian_counter)

    # Time range
    kicks = []
    for m in matches:
        try:
            kicks.append(parse_kickoff_time(m["kickoff_time"]))
        except Exception:
            pass
    if kicks:
        report["time_range"]["min"] = min(kicks).isoformat()
        report["time_range"]["max"] = max(kicks).isoformat()
        report["time_range"]["span_days"] = (max(kicks) - min(kicks)).days

    # Odds validity checks
    for m in matches:
        for e in m.get("odds_timeline", []):
            for k in ["euro_h", "euro_d", "euro_a"]:
                v = e.get(k, 0)
                if v <= 0 or v > 100:
                    report["invalid_odds"] += 1
                    break

    # Warnings
    if report["timeline"]["single_event_matches"] == len(matches):
        report["warnings"].append("All matches are closing-only (single event) — no time-series value")
    if report["timeline"]["cutoff_available"].get("90", 0) == 0:
        report["warnings"].append("No matches have events at T-90+ (time-series training limited)")

    report["match_ids"] = [m["match_id"] for m in matches[:5]]
    return report


def main():
    parser = argparse.ArgumentParser(description="OddsMind Real Data Quality Audit")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    report = audit_jsonl(args.data)

    output = json.dumps(report, indent=2, default=str)
    print(output)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(output + "\n")


if __name__ == "__main__":
    main()
