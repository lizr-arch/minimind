"""Leak and artifact audit for accepted P6 formal runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PROB_COLS = ("p_home", "p_draw", "p_away")
SUSPICIOUS_FEATURE_TOKENS = (
    "label",
    "result",
    "score",
    "goal",
    "winner",
    "home_score",
    "away_score",
    "y_true",
)


def audit_p6_runs(
    runs_root: str | Path,
    expected_seeds: list[int],
    train_ids_path: str | Path,
    val_ids_path: str | Path,
    expected_train_rows: int,
    expected_val_rows: int,
    run_prefix: str = "p6_euro_default_seed",
) -> dict[str, Any]:
    runs_root = Path(runs_root)
    train_ids = _load_ids(train_ids_path)
    val_ids = _load_ids(val_ids_path)
    overlap = sorted(set(train_ids) & set(val_ids))
    runs = []
    verdict: list[str] = []
    duplicate_match_id_count = 0
    total_rows = 0

    if overlap:
        verdict.append("P6_LEAK_FAIL_TRAIN_VAL_OVERLAP")

    for seed in expected_seeds:
        run_dir = runs_root / f"{run_prefix}{seed}"
        if not run_dir.exists():
            verdict.append("P6_LEAK_FAIL_MISSING_SEED")
            runs.append({"seed": seed, "run_dir": str(run_dir), "status": "missing"})
            continue
        report_path = run_dir / "report.json"
        preds_path = run_dir / "val_predictions.csv"
        if not report_path.exists() or not preds_path.exists():
            verdict.append("P6_LEAK_FAIL_MISSING_ARTIFACT")
            runs.append({"seed": seed, "run_dir": str(run_dir), "status": "missing_artifact"})
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        data = report.get("data", {})
        run_verdict = []
        if data.get("test_ids_used") is not False:
            run_verdict.append("P6_LEAK_FAIL_TEST_USED")
        if data.get("validation_fit_used") is not False:
            run_verdict.append("P6_LEAK_FAIL_VALIDATION_FIT_FLAG")
        if data.get("posthoc_val_fit_used", False) is not False:
            run_verdict.append("P6_LEAK_FAIL_POSTHOC_VAL_FIT_FLAG")
        if data.get("train_samples") != expected_train_rows:
            run_verdict.append("P6_LEAK_FAIL_TRAIN_SAMPLE_COUNT")
        if data.get("val_samples") != expected_val_rows:
            run_verdict.append("P6_LEAK_FAIL_VAL_SAMPLE_COUNT")
        if data.get("anchor_fallback_count") != 0:
            run_verdict.append("P6_LEAK_FAIL_ANCHOR_FALLBACK")
        if data.get("negative_time_valid_euro_count") != 0:
            run_verdict.append("P6_LEAK_FAIL_NEGATIVE_TIME_EURO")
        bad_features = _suspicious_features(data.get("feature_names", []))
        if bad_features:
            run_verdict.append("P6_LEAK_FAIL_LABEL_FEATURE")
        pred_audit = _audit_predictions(preds_path, expected_val_rows)
        duplicate_match_id_count += pred_audit["duplicate_match_id_count"]
        total_rows += pred_audit["row_count"]
        run_verdict.extend(pred_audit["verdict"])
        verdict.extend(run_verdict)
        runs.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "status": "fail" if run_verdict else "pass",
                "best_epoch": report.get("best_epoch"),
                "best_val_logloss": report.get("best_val_logloss"),
                "feature_names": data.get("feature_names", []),
                "suspicious_features": bad_features,
                "prediction_audit": pred_audit,
                "verdict": run_verdict,
                "checkpoint_selection_warning": (
                    "best_val_logloss checkpoint is selected on official val; treated as existing project convention, "
                    "not as post-hoc hyperparameter fitting in this audit"
                ),
            }
        )

    verdict = sorted(set(verdict))
    if not verdict:
        verdict.append("P6_LEAK_AUDIT_PASS")
    return {
        "phase": "P6 leak audit",
        "status": "fail" if any(item.startswith("P6_LEAK_FAIL") for item in verdict) else "pass",
        "runs_root": str(runs_root),
        "run_prefix": run_prefix,
        "expected_seeds": expected_seeds,
        "expected_train_rows": expected_train_rows,
        "expected_val_rows": expected_val_rows,
        "train_id_count": len(train_ids),
        "val_id_count": len(val_ids),
        "train_val_overlap_count": len(overlap),
        "train_val_overlap_examples": overlap[:20],
        "total_prediction_rows": total_rows,
        "duplicate_match_id_count": duplicate_match_id_count,
        "duplicate_match_id_is_expected": duplicate_match_id_count > 0,
        "runs": runs,
        "verdict": verdict,
    }


def write_markdown_report(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# P6 Leak Audit",
        "",
        "## Summary",
        "",
        f"- status: `{payload.get('status')}`",
        f"- train_id_count: `{payload.get('train_id_count')}`",
        f"- val_id_count: `{payload.get('val_id_count')}`",
        f"- train_val_overlap_count: `{payload.get('train_val_overlap_count')}`",
        f"- total_prediction_rows: `{payload.get('total_prediction_rows')}`",
        f"- duplicate_match_id_count: `{payload.get('duplicate_match_id_count')}`",
        f"- duplicate_match_id_is_expected: `{payload.get('duplicate_match_id_is_expected')}`",
        "",
        "## Runs",
        "",
        "| Seed | Status | Best epoch | Best val logloss | Verdict |",
        "|---:|---|---:|---:|---|",
    ]
    for run in payload.get("runs", []):
        lines.append(
            f"| {run.get('seed')} | {run.get('status')} | {run.get('best_epoch')} | "
            f"{_fmt(run.get('best_val_logloss'))} | {', '.join(run.get('verdict', [])) or 'PASS'} |"
        )
    lines.extend(["", "## Verdict", ""])
    for item in payload.get("verdict", []):
        lines.append(f"- {item}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_predictions(path: Path, expected_rows: int) -> dict[str, Any]:
    row_count = 0
    verdict = []
    ids = []
    p_true_exact_one_count = 0
    invalid_sum_count = 0
    invalid_prob_count = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            ids.append(row["match_id"])
            probs = []
            for col in PROB_COLS:
                try:
                    value = float(row[col])
                except (TypeError, ValueError):
                    invalid_prob_count += 1
                    value = math.nan
                probs.append(value)
                if not math.isfinite(value) or value < 0 or value > 1:
                    invalid_prob_count += 1
            if not math.isclose(sum(probs), 1.0, abs_tol=1e-4):
                invalid_sum_count += 1
            y_true = int(row["y_true"])
            if math.isclose(probs[y_true], 1.0, abs_tol=1e-12):
                p_true_exact_one_count += 1
    duplicate_count = sum(count - 1 for count in Counter(ids).values() if count > 1)
    if row_count != expected_rows:
        verdict.append("P6_LEAK_FAIL_VAL_PREDICTION_ROW_COUNT")
    if invalid_prob_count:
        verdict.append("P6_LEAK_FAIL_PROBABILITY_VALUE")
    if invalid_sum_count:
        verdict.append("P6_LEAK_FAIL_PROBABILITY_SUM")
    if p_true_exact_one_count:
        verdict.append("P6_LEAK_WARN_TRUE_PROB_EXACT_ONE")
    return {
        "row_count": row_count,
        "unique_match_id_count": len(set(ids)),
        "duplicate_match_id_count": duplicate_count,
        "invalid_probability_count": invalid_prob_count,
        "invalid_probability_sum_count": invalid_sum_count,
        "p_true_exact_one_count": p_true_exact_one_count,
        "verdict": verdict,
    }


def _load_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _suspicious_features(feature_names: list[str]) -> list[str]:
    bad = []
    for name in feature_names:
        lowered = str(name).lower()
        if any(token in lowered for token in SUSPICIOUS_FEATURE_TOKENS):
            bad.append(str(name))
    return bad


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P6 formal runs for leakage and artifact issues.")
    parser.add_argument("--runs-root", default="runs/p6_residual_patch_itransformer")
    parser.add_argument("--expected-seeds", default="42,123,2025")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--expected-train-rows", type=int, default=26441)
    parser.add_argument("--expected-val-rows", type=int, default=4747)
    parser.add_argument("--run-prefix", default="p6_euro_default_seed")
    parser.add_argument("--out-json", default="runs/p6_residual_patch_itransformer/p6_leak_audit.json")
    parser.add_argument("--out-md", default="runs/p6_residual_patch_itransformer/p6_leak_audit.md")
    args = parser.parse_args()

    expected_seeds = [int(seed.strip()) for seed in args.expected_seeds.split(",") if seed.strip()]
    payload = audit_p6_runs(
        runs_root=args.runs_root,
        expected_seeds=expected_seeds,
        train_ids_path=args.train_ids,
        val_ids_path=args.val_ids,
        expected_train_rows=args.expected_train_rows,
        expected_val_rows=args.expected_val_rows,
        run_prefix=args.run_prefix,
    )
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, out_md)
    print(f"P6 leak audit json: {out_json}")
    print(f"P6 leak audit markdown: {out_md}")
    print(f"Status: {payload['status']}")
    print(f"Verdict: {', '.join(payload['verdict'])}")


if __name__ == "__main__":
    main()
