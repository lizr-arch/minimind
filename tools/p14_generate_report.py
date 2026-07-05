"""Generate P14 market-replication summary reports from completed runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_runs(run_dirs: list[Path], verdict: str) -> dict[str, Any]:
    rows = []
    for run_dir in run_dirs:
        report = read_json(run_dir / "report.json")
        rows.append(
            {
                "run": run_dir.name,
                "seed": int(report["config"]["seed"]),
                "selected_epoch": int(report["best_epoch"]),
                "best_logloss_epoch": int(report["best_logloss_epoch"]),
                "selected_val_logloss": float(report["val_metrics"]["logloss"]),
                "best_logloss": float(report["best_logloss"]),
                "draw_top2": float(report["val_metrics"]["draw_top2_recall"]),
                "mean_p_draw_true_draw": float(report["val_metrics"]["mean_p_draw_on_true_draw"]),
                "mean_p_draw": float(report["val_metrics"]["mean_p_draw"]),
                "market_kl": float(report["market_replication_metrics"]["market_kl"]),
                "market_draw_mae": float(report["market_replication_metrics"]["market_draw_mae"]),
                "model_minus_market_p_draw": float(report["market_replication_metrics"]["model_minus_market_p_draw"]),
                "market_draw_top2": float(report["market_replication_metrics"]["market_draw_top2"]),
                "anchor_logloss": float(report["anchor_only_baseline"]["anchor_only_val_logloss"]),
                "draw_risk_acc": report["goal_diff_metrics"].get("draw_risk_acc"),
                "draw_risk_nll": report["goal_diff_metrics"].get("draw_risk_nll"),
            }
        )
    metric_keys = [
        "selected_val_logloss",
        "best_logloss",
        "draw_top2",
        "mean_p_draw_true_draw",
        "mean_p_draw",
        "market_kl",
        "market_draw_mae",
        "model_minus_market_p_draw",
    ]
    if any(row.get("draw_risk_acc") is not None for row in rows):
        metric_keys.extend(["draw_risk_acc", "draw_risk_nll"])
    summary = {
        key: {
            "mean": statistics.mean(row[key] for row in rows if row.get(key) is not None),
            "stdev": statistics.pstdev(row[key] for row in rows if row.get(key) is not None),
        }
        for key in metric_keys
    }
    baselines = {
        "p6_seed42_logloss": 0.9374789231561449,
        "p6_seed42_draw_top2": 0.3423913043478261,
        "p6_seed42_mean_p_draw_true_draw": 0.19780794765118842,
        "p13_market_logloss": rows[0]["anchor_logloss"] if rows else None,
        "p13_market_draw_top2": rows[0]["market_draw_top2"] if rows else None,
    }
    return {"runs": rows, "summary": summary, "baselines": baselines, "verdict": verdict}


def write_markdown(path: Path, payload: dict[str, Any], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        "## Baselines",
    ]
    for key, value in payload["baselines"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## 3-Seed Summary",
            "| metric | mean | stdev |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, stats in payload["summary"].items():
        lines.append(f"| {key} | {stats['mean']:.6f} | {stats['stdev']:.6f} |")
    lines.extend(
        [
            "",
            "## Runs",
            "| run | seed | selected_epoch | selected_val_logloss | draw_top2 | mean_p_draw_true_draw | market_kl | market_draw_mae | draw_risk_acc |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["runs"]:
        lines.append(
            "| {run} | {seed} | {selected_epoch} | {selected_val_logloss:.6f} | {draw_top2:.6f} | "
            "{mean_p_draw_true_draw:.6f} | {market_kl:.6f} | {market_draw_mae:.6f} | {draw_risk_acc} |".format(
                **{**row, "draw_risk_acc": "" if row.get("draw_risk_acc") is None else f"{row['draw_risk_acc']:.6f}"}
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Balanced checkpoint selection trades a tiny amount of label logloss for materially better market/draw structure.",
            "- This is still market replication, not value detection. It should be followed by selection ablation and calibration checks before any value target.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate P14 market-replication summary reports")
    parser.add_argument("--runs-root", default="runs/p14_market_replication")
    parser.add_argument(
        "--run-names",
        default="p14_market_kl_030_balanced_seed42,p14_market_kl_030_balanced_seed123,p14_market_kl_030_balanced_seed2025",
    )
    parser.add_argument("--out-json", default="runs/p14_market_replication/p14_market_replication_summary.json")
    parser.add_argument("--out-md", default="runs/p14_market_replication/p14_market_replication_summary.md")
    parser.add_argument("--title", default="P14 Market Replication Summary")
    parser.add_argument("--verdict", default="P14_MARKET_REPLICATION_PROMISING_NEEDS_SELECTION_ABLATION")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.runs_root)
    run_dirs = [root / name.strip() for name in args.run_names.split(",") if name.strip()]
    payload = summarize_runs(run_dirs, args.verdict)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_md, payload, args.title)
    print(f"Summary saved to {out_json}")
    print(f"Markdown saved to {out_md}")


if __name__ == "__main__":
    main()
