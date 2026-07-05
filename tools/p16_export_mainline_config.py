"""Export the promoted P16/P14 market-replication mainline config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAINLINE_VERDICT = "P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE"


def build_mainline_config(summary: dict[str, Any], source_run_root: str) -> dict[str, Any]:
    if summary.get("verdict") != MAINLINE_VERDICT:
        raise ValueError(f"P16 summary is not promoted: {summary.get('verdict')}")
    promoted = [
        row
        for row in summary.get("candidate_summaries", [])
        if row.get("decision", {}).get("verdict") == MAINLINE_VERDICT
    ]
    if len(promoted) != 1:
        raise ValueError(f"Expected exactly one promoted P16 candidate, found {len(promoted)}")
    candidate = promoted[0]
    if candidate.get("variant") != "p16_b_mktkl_100x" or candidate.get("selection_rule") != "old_balanced":
        raise ValueError(f"Unexpected promoted mainline candidate: {candidate.get('variant')} / {candidate.get('selection_rule')}")
    control = summary.get("control_summary", {})
    return {
        "schema_version": 1,
        "mainline_id": "p16_b_mktkl_100x_old_balanced",
        "status": "promoted",
        "source_phase": "P16",
        "source_run_root": source_run_root,
        "source_verdict": summary.get("verdict"),
        "model_family": "P6ResidualPatchITransformer",
        "train_script": "tools/p14_train_market_replication.py",
        "train_args": {
            "data": "data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl",
            "train_ids": "data/odds_real/splits_v6/train_match_ids.txt",
            "val_ids": "data/odds_real/splits_v6/val_match_ids.txt",
            "feature_groups": "euro,asian,ou",
            "epochs": 20,
            "batch_size": 128,
            "lr": 0.0003,
            "weight_decay": 0.01,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
            "dropout": 0.10,
            "label_loss_weight": 1.0,
            "market_loss_weight": 0.30,
            "checkpoint_selection": "balanced",
            "balanced_logloss_ceiling": 0.9375,
            "diff_loss_weight": 0.30,
            "consistency_loss_weight": 0.10,
            "delta_l2_weight": 0.01,
            "draw_risk_loss_weight": 0.0,
            "scaling": "robust",
        },
        "selection_rule": "old_balanced",
        "calibration_policy": {
            "default": "identity_no_fit",
            "val_fitted_calibration_promotable": False,
            "note": "Val-fitted calibration remains diagnostic only.",
        },
        "promotion_evidence": {
            "control": {
                "variant": control.get("variant"),
                "mean_val_logloss": control.get("mean_val_logloss"),
                "mean_ece": control.get("mean_ece"),
                "draw_top2": control.get("draw_top2"),
                "mean_p_draw_true_draw": control.get("mean_p_draw_true_draw"),
                "market_draw_mae": control.get("market_draw_mae"),
            },
            "metrics": {
                "variant": candidate.get("variant"),
                "selection_rule": candidate.get("selection_rule"),
                "mean_val_logloss": candidate.get("mean_val_logloss"),
                "mean_ece": candidate.get("mean_ece"),
                "draw_top2": candidate.get("draw_top2"),
                "mean_p_draw_true_draw": candidate.get("mean_p_draw_true_draw"),
                "market_draw_mae": candidate.get("market_draw_mae"),
                "model_minus_market_p_draw": candidate.get("model_minus_market_p_draw"),
                "slice_gate_pass": candidate.get("slice_gate_pass"),
                "draw_top2_stdev": candidate.get("draw_top2_stdev"),
                "mean_p_draw_true_draw_stdev": candidate.get("mean_p_draw_true_draw_stdev"),
            },
            "failed_gates": candidate.get("decision", {}).get("failed_gates", []),
        },
        "artifacts": {
            "p16_report": f"{source_run_root}/p16_report.md",
            "p16_results_summary": f"{source_run_root}/p16_results_summary.json",
            "p16_slice_stability": f"{source_run_root}/p16_slice_stability.csv",
            "p16_calibration_diagnostic": f"{source_run_root}/p16_calibration_diagnostic.json",
        },
        "next_phase_policy": {
            "p17_value_detection": "design_only_until_requested",
            "p15_draw_risk_aux": "diagnostic_only_not_mainline",
        },
    }


def write_mainline_artifacts(config: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(config), encoding="utf-8")


def _render_markdown(config: dict[str, Any]) -> str:
    metrics = config["promotion_evidence"]["metrics"]
    control = config["promotion_evidence"]["control"]
    args = config["train_args"]
    lines = [
        "# OddsMind Mainline Status",
        "",
        f"Verdict: `{config['source_verdict']}`",
        "",
        "## Current Mainline",
        f"- mainline_id: `{config['mainline_id']}`",
        f"- model_family: `{config['model_family']}`",
        f"- train_script: `{config['train_script']}`",
        f"- selection_rule: `{config['selection_rule']}`",
        f"- market_loss_weight: `{args['market_loss_weight']}`",
        f"- checkpoint_selection: `{args['checkpoint_selection']}`",
        f"- balanced_logloss_ceiling: `{args['balanced_logloss_ceiling']}`",
        "",
        "## Promotion Evidence",
        f"- control_logloss: `{control.get('mean_val_logloss')}`",
        f"- mainline_logloss: `{metrics.get('mean_val_logloss')}`",
        f"- control_draw_top2: `{control.get('draw_top2')}`",
        f"- mainline_draw_top2: `{metrics.get('draw_top2')}`",
        f"- control_true_draw_p: `{control.get('mean_p_draw_true_draw')}`",
        f"- mainline_true_draw_p: `{metrics.get('mean_p_draw_true_draw')}`",
        f"- mainline_market_draw_mae: `{metrics.get('market_draw_mae')}`",
        f"- slice_gate_pass: `{metrics.get('slice_gate_pass')}`",
        "",
        "## Guardrails",
        "- val-fitted calibration is diagnostic only and is not promotable.",
        "- P15 draw-risk auxiliary remains diagnostic-only.",
        "- P17 value detection is design-only until explicitly requested.",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export OddsMind promoted P16 mainline config")
    parser.add_argument("--summary-json", default="runs/p16_market_replication_selection/p16_results_summary.json")
    parser.add_argument("--source-run-root", default="runs/p16_market_replication_selection")
    parser.add_argument("--out-json", default="configs/oddsmind_mainline.json")
    parser.add_argument("--out-md", default="docs/oddsmind_mainline_status.md")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    config = build_mainline_config(summary, args.source_run_root)
    write_mainline_artifacts(config, Path(args.out_json), Path(args.out_md))
    print(f"Mainline config written to {args.out_json}")
    print(f"Mainline status written to {args.out_md}")


if __name__ == "__main__":
    main()
