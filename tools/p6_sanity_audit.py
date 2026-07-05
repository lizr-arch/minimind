"""Sanity audit for P6 vs P4 validation predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


PROB_COLS = ("p_home", "p_draw", "p_away")


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match_id = row["match_id"]
            rows.append(
                {
                    "match_id": match_id,
                    "y_true": int(row["y_true"]),
                    "p_home": float(row["p_home"]),
                    "p_draw": float(row["p_draw"]),
                    "p_away": float(row["p_away"]),
                }
            )
    return rows


def nll(row: dict[str, Any]) -> float:
    y_true = int(row["y_true"])
    prob = float(row[PROB_COLS[y_true]])
    return -math.log(max(prob, 1e-15))


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x == 0 or denom_y == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def paired_bootstrap(deltas: list[float], samples: int = 5000, seed: int = 42) -> dict[str, float]:
    if not deltas:
        raise ValueError("paired bootstrap requires at least one delta")
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(samples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low_idx = max(0, int(0.025 * samples) - 1)
    high_idx = min(samples - 1, int(0.975 * samples))
    low99_idx = max(0, int(0.005 * samples) - 1)
    high99_idx = min(samples - 1, int(0.995 * samples))
    mean_delta = sum(deltas) / n
    return {
        "samples": samples,
        "mean_delta": mean_delta,
        "ci95_low": means[low_idx],
        "ci95_high": means[high_idx],
        "ci99_low": means[low99_idx],
        "ci99_high": means[high99_idx],
        "p_candidate_better": sum(1 for value in means if value < 0) / samples,
    }


def topk_contribution_share(deltas: list[float], k: int = 20) -> dict[str, float | int | None]:
    improvements = sorted([-value for value in deltas if value < 0], reverse=True)
    regressions = sorted([value for value in deltas if value > 0], reverse=True)
    total_improvement = sum(improvements)
    total_regression = sum(regressions)
    top_improvement = sum(improvements[:k])
    top_regression = sum(regressions[:k])
    return {
        "topk": k,
        "improvement_total": total_improvement,
        "regression_total": total_regression,
        "improvement_share": top_improvement / total_improvement if total_improvement else None,
        "regression_share": top_regression / total_regression if total_regression else None,
    }


def aggregate_seed_pairs(
    baseline_root: str | Path,
    candidate_root: str | Path,
    expected_seeds: list[int],
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 42,
    baseline_prefix: str = "p4_euro_default_seed",
    candidate_prefix: str = "p6_euro_default_seed",
    baseline_label: str = "P4 euro_default",
    candidate_label: str = "P6 euro_default",
) -> dict[str, Any]:
    baseline_root = Path(baseline_root)
    candidate_root = Path(candidate_root)
    seed_pairs = []
    per_row_deltas: dict[int, list[float]] = {}
    probability_pairs = {key: ([], []) for key in PROB_COLS}
    audit = {
        "test_ids_used": False,
        "validation_fit_used": False,
        "posthoc_val_fit_used": False,
        "anchor_fallback_count": 0,
        "negative_time_valid_euro_count": 0,
        "missing_expected_seeds": [],
    }

    for seed_value in expected_seeds:
        baseline_run = _find_seed_run(baseline_root, seed_value, baseline_prefix)
        candidate_run = _find_seed_run(candidate_root, seed_value, candidate_prefix)
        if baseline_run is None or candidate_run is None:
            audit["missing_expected_seeds"].append(seed_value)
            continue
        base_preds = load_predictions(baseline_run / "val_predictions.csv")
        cand_preds = load_predictions(candidate_run / "val_predictions.csv")
        if len(base_preds) != len(cand_preds):
            raise ValueError(f"prediction row count mismatch for seed {seed_value}")
        deltas = []
        for row_idx, (base, cand) in enumerate(zip(base_preds, cand_preds)):
            if base["match_id"] != cand["match_id"]:
                raise ValueError(f"match_id mismatch for seed {seed_value} row {row_idx}")
            if base["y_true"] != cand["y_true"]:
                raise ValueError(f"y_true mismatch for seed {seed_value} row {row_idx}")
            delta = nll(cand) - nll(base)
            deltas.append(delta)
            per_row_deltas.setdefault(row_idx, []).append(delta)
            for col in PROB_COLS:
                probability_pairs[col][0].append(float(base[col]))
                probability_pairs[col][1].append(float(cand[col]))
        seed_pairs.append(
            {
                "seed": seed_value,
                "baseline_run": str(baseline_run),
                "candidate_run": str(candidate_run),
                "match_count": len(base_preds),
                "baseline_logloss": _mean([nll(row) for row in base_preds]),
                "candidate_logloss": _mean([nll(row) for row in cand_preds]),
                "mean_delta_nll": _mean(deltas),
                "fraction_matches_improved": sum(1 for value in deltas if value < 0) / len(deltas),
            }
        )
        _merge_audit(audit, baseline_run / "report.json")
        _merge_audit(audit, candidate_run / "report.json")

    match_level_deltas = [_mean(values) for _, values in sorted(per_row_deltas.items())]
    bootstrap = paired_bootstrap(match_level_deltas, samples=bootstrap_samples, seed=bootstrap_seed)
    top20 = topk_contribution_share(match_level_deltas, k=20)
    correlations = {
        col: pearson_corr(pair[0], pair[1])
        for col, pair in probability_pairs.items()
    }
    payload = {
        "phase": "P6 sanity audit",
        "baseline": baseline_label,
        "candidate": candidate_label,
        "expected_seeds": expected_seeds,
        "seed_count": len(seed_pairs),
        "match_count": len(match_level_deltas),
        "row_pairing": "row_order",
        "seed_pairs": seed_pairs,
        "mean_seed_delta_nll": _mean([item["mean_delta_nll"] for item in seed_pairs]),
        "mean_match_delta_nll": _mean(match_level_deltas),
        "fraction_matches_improved": sum(1 for value in match_level_deltas if value < 0) / len(match_level_deltas),
        "bootstrap": bootstrap,
        "top20_contribution": top20,
        "probability_correlations": correlations,
        "audit": audit,
    }
    payload["verdict"] = sanity_verdict(payload)
    return payload


def sanity_verdict(payload: dict[str, Any]) -> list[str]:
    verdict = []
    audit = payload.get("audit", {})
    bootstrap = payload.get("bootstrap", {})
    if audit.get("test_ids_used") is not False:
        verdict.append("P6_SANITY_FAIL_TEST_READ")
    if audit.get("validation_fit_used") is not False:
        verdict.append("P6_SANITY_FAIL_VAL_FIT")
    if audit.get("posthoc_val_fit_used") is not False:
        verdict.append("P6_SANITY_FAIL_POSTHOC_VAL_FIT")
    if audit.get("missing_expected_seeds"):
        verdict.append("P6_SANITY_MISSING_SEEDS")
    if verdict:
        return verdict
    if payload.get("mean_match_delta_nll", 0) < 0:
        verdict.append("P6_SANITY_MEAN_DELTA_IMPROVES")
    if bootstrap.get("ci95_high", 1) < 0 and bootstrap.get("p_candidate_better", 0) >= 0.95:
        verdict.append("P6_SANITY_BOOTSTRAP_PASS")
    elif bootstrap.get("p_candidate_better", 0) >= 0.8:
        verdict.append("P6_SANITY_BOOTSTRAP_WEAK")
    else:
        verdict.append("P6_SANITY_BOOTSTRAP_INCONCLUSIVE")
    top20_share = (payload.get("top20_contribution") or {}).get("improvement_share")
    if payload.get("match_count", 0) >= 100 and top20_share is not None:
        if top20_share <= 0.40:
            verdict.append("P6_SANITY_TOP20_NOT_CONCENTRATED")
        else:
            verdict.append("P6_SANITY_TOP20_CONCENTRATED")
    verdict.append("P6_SANITY_NO_TEST_USED")
    verdict.append("P6_SANITY_NO_VAL_FIT")
    return verdict


def write_markdown_report(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# P6 Sanity Audit",
        "",
        "## Summary",
        "",
        f"- match_count: `{payload.get('match_count')}`",
        f"- seed_count: `{payload.get('seed_count')}`",
        f"- mean_match_delta_nll: `{_fmt(payload.get('mean_match_delta_nll'), 10)}`",
        f"- fraction_matches_improved: `{_fmt(payload.get('fraction_matches_improved'), 6)}`",
        "",
        "## Bootstrap",
        "",
        f"- p_candidate_better: `{_fmt(payload.get('bootstrap', {}).get('p_candidate_better'), 6)}`",
        f"- ci95_low: `{_fmt(payload.get('bootstrap', {}).get('ci95_low'), 10)}`",
        f"- ci95_high: `{_fmt(payload.get('bootstrap', {}).get('ci95_high'), 10)}`",
        f"- ci99_low: `{_fmt(payload.get('bootstrap', {}).get('ci99_low'), 10)}`",
        f"- ci99_high: `{_fmt(payload.get('bootstrap', {}).get('ci99_high'), 10)}`",
        "",
        "## Top-20 Contribution",
        "",
        f"- improvement_share: `{_fmt(payload.get('top20_contribution', {}).get('improvement_share'), 6)}`",
        f"- regression_share: `{_fmt(payload.get('top20_contribution', {}).get('regression_share'), 6)}`",
        "",
        "## Probability Correlations",
        "",
    ]
    for key, value in payload.get("probability_correlations", {}).items():
        lines.append(f"- {key}: `{_fmt(value, 6)}`")
    lines.extend(
        [
            "",
            "## Seed Pairs",
            "",
            "| Seed | P4 logloss | P6 logloss | Delta | Match improved |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload.get("seed_pairs", []):
        lines.append(
            f"| {item.get('seed')} | {_fmt(item.get('baseline_logloss'))} | "
            f"{_fmt(item.get('candidate_logloss'))} | {_fmt(item.get('mean_delta_nll'))} | "
            f"{_fmt(item.get('fraction_matches_improved'), 6)} |"
        )
    lines.extend(["", "## Audit", ""])
    for key, value in payload.get("audit", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Verdict", ""])
    for item in payload.get("verdict", []):
        lines.append(f"- {item}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_seed_run(root: Path, seed_value: int, prefix: str) -> Path | None:
    matches = sorted(root.glob(f"{prefix}{seed_value}"))
    for match in matches:
        if (match / "val_predictions.csv").exists() and (match / "report.json").exists():
            return match
    return None


def _merge_audit(audit: dict[str, Any], report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    data = report.get("data", {})
    audit["test_ids_used"] = audit["test_ids_used"] or data.get("test_ids_used") is not False
    audit["validation_fit_used"] = audit["validation_fit_used"] or data.get("validation_fit_used", False) is not False
    audit["posthoc_val_fit_used"] = audit["posthoc_val_fit_used"] or data.get("posthoc_val_fit_used", False) is not False
    audit["anchor_fallback_count"] += int(data.get("anchor_fallback_count", 0) or 0)
    audit["negative_time_valid_euro_count"] += int(data.get("negative_time_valid_euro_count", 0) or 0)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty list")
    return float(sum(values) / len(values))


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P6 vs P4 sanity audit on validation predictions.")
    parser.add_argument("--baseline-root", default="runs/p4_residual_goal_diff/p4_1_runs")
    parser.add_argument("--candidate-root", default="runs/p6_residual_patch_itransformer")
    parser.add_argument("--baseline-prefix", default="p4_euro_default_seed")
    parser.add_argument("--candidate-prefix", default="p6_euro_default_seed")
    parser.add_argument("--baseline-label", default="P4 euro_default")
    parser.add_argument("--candidate-label", default="P6 euro_default")
    parser.add_argument("--expected-seeds", default="42,123,2025")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--out-json", default="runs/p6_residual_patch_itransformer/p6_sanity_audit.json")
    parser.add_argument("--out-md", default="runs/p6_residual_patch_itransformer/p6_sanity_audit.md")
    args = parser.parse_args()

    expected_seeds = [int(seed.strip()) for seed in args.expected_seeds.split(",") if seed.strip()]
    payload = aggregate_seed_pairs(
        baseline_root=args.baseline_root,
        candidate_root=args.candidate_root,
        expected_seeds=expected_seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        baseline_prefix=args.baseline_prefix,
        candidate_prefix=args.candidate_prefix,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, out_md)
    print(f"P6 sanity audit json: {out_json}")
    print(f"P6 sanity audit markdown: {out_md}")
    print(f"Mean match delta NLL: {payload['mean_match_delta_nll']:.10f}")
    print(f"Bootstrap p_candidate_better: {payload['bootstrap']['p_candidate_better']:.6f}")
    print(f"Verdict: {', '.join(payload['verdict'])}")


if __name__ == "__main__":
    main()
