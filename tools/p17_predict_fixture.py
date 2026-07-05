"""Run frozen OddsMind mainline-style checkpoints on a pre-match fixture export.

This is a single-fixture inference utility. It accepts prediction_export_v1
rows with no labels, loads existing P14/P16-style checkpoints, and writes
model probabilities plus market-anchor edges. It never reads test splits and
never fabricates labels for future matches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
from tools.p14_train_market_replication import apply_feature_scaler, p14_selected_feature_market_type_ids, p14_selected_feature_names
from tools.p3b_train_patch_itransformer import bucketize_sample_v2_for_training
from tools.p4_goal_diff_utils import EPS, extract_euro_anchor_probs, fold_goal_diff_probs
from tools.p4_train_residual_goal_diff import FEATURE_INDEX


OUTCOMES = ("home", "draw", "away")
PROB_KEYS = ("p_home", "p_draw", "p_away")
MARKET_KEYS = ("market_home", "market_draw", "market_away")


class P17FixtureProtocolError(RuntimeError):
    pass


def _contains_test_path(path: str | Path) -> bool:
    parts = [part.lower() for part in Path(path).parts]
    return any(part == "test" or part.startswith("test_") or part.startswith("test-") for part in parts)


def assert_no_test_paths(paths: Iterable[str | Path]) -> None:
    for path in paths:
        if path and _contains_test_path(path):
            raise P17FixtureProtocolError(f"P17 fixture inference refuses test split or test artifact path: {path}")


def assert_prediction_rows_are_pre_match(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise P17FixtureProtocolError("No prediction rows supplied")
    for row in rows:
        if row.get("label") is not None or row.get("label_status") != "not_available_pre_match":
            raise P17FixtureProtocolError("P17 fixture inference requires pre-match prediction rows without labels")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _normalise(values: Iterable[Any]) -> list[float]:
    vals = [max(_safe_float(value), 0.0) for value in values]
    total = sum(vals)
    if total <= EPS:
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return [value / total for value in vals]


def _latest_event(row: dict[str, Any]) -> dict[str, Any]:
    timeline = row.get("raw_timeline", []) or row.get("odds_timeline", []) or []
    valid = [event for event in timeline if _safe_float(event.get("minutes_before_kickoff"), default=-1.0) >= 0.0]
    if not valid:
        return {}
    return min(valid, key=lambda event: _safe_float(event.get("minutes_before_kickoff"), default=1e12))


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    assert_no_test_paths([path])
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("bookmaker_samples"), list):
            rows = list(obj["bookmaker_samples"])
        elif isinstance(obj, list):
            rows = obj
        else:
            rows = [obj]
    else:
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    assert_prediction_rows_are_pre_match(rows)
    return rows


def build_fixture_dataset(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    assert_prediction_rows_are_pre_match(rows)
    indices = [FEATURE_INDEX[name] for name in feature_names]
    x_rows: list[torch.Tensor] = []
    p_market: list[torch.Tensor] = []
    match_ids: list[str] = []
    bookmakers: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    latest_snapshots: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        timeline = row.get("raw_timeline", []) or row.get("odds_timeline", []) or []
        if not timeline:
            raise P17FixtureProtocolError(f"Prediction row has no timeline: {row.get('match_id', idx)}")
        xb_full = bucketize_sample_v2_for_training(timeline, clean_missing_markets=True)
        anchor, diag = extract_euro_anchor_probs(row)
        latest = _latest_event(row)
        x_rows.append(xb_full[:, indices, :])
        p_market.append(anchor)
        match_ids.append(str(row.get("match_id", idx)))
        bookmakers.append(str(row.get("bookmaker_id", row.get("bookmaker", idx))))
        diagnostics.append(diag)
        latest_snapshots.append(
            {
                "minutes_before_kickoff": _safe_float(latest.get("minutes_before_kickoff")),
                "euro_h": _safe_float(latest.get("euro_h")),
                "euro_d": _safe_float(latest.get("euro_d")),
                "euro_a": _safe_float(latest.get("euro_a")),
                "asian_line": _safe_float(latest.get("asian_line")),
                "upper_water": _safe_float(latest.get("upper_water")),
                "lower_water": _safe_float(latest.get("lower_water")),
                "over_under_line": _safe_float(latest.get("over_under_line")),
                "over_water": _safe_float(latest.get("over_water")),
                "under_water": _safe_float(latest.get("under_water")),
            }
        )
    return {
        "X": torch.stack(x_rows),
        "p_market": torch.stack(p_market),
        "match_ids": match_ids,
        "bookmakers": bookmakers,
        "anchor_diagnostics": diagnostics,
        "latest_snapshots": latest_snapshots,
        "feature_names": feature_names,
    }


def load_checkpoint_model(run_dir: Path, device: torch.device) -> tuple[P6ResidualPatchITransformer, dict[str, Any], dict[str, Any]]:
    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    feature_names = list(checkpoint.get("feature_names") or p14_selected_feature_names(config.get("feature_groups", "euro,asian,ou")))
    market_type_ids = list(checkpoint.get("market_type_ids") or p14_selected_feature_market_type_ids(config.get("feature_groups", "euro,asian,ou")))
    model = P6ResidualPatchITransformer(
        n_features=len(feature_names),
        d_model=int(config.get("d_model", 64)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        d_ff=int(config.get("d_ff", 128)),
        dropout=float(config.get("dropout", 0.10)),
        market_type_ids=market_type_ids,
        enable_draw_risk_head=float(config.get("draw_risk_loss_weight", 0.0) or 0.0) > 0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler_path = Path(config.get("scaler_path") or run_dir / "scaler.json")
    if not scaler_path.exists():
        scaler_path = run_dir / "scaler.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    return model, {"checkpoint": checkpoint, "config": config, "feature_names": feature_names}, scaler


@torch.no_grad()
def predict_fixture_rows(
    model: P6ResidualPatchITransformer,
    dataset: dict[str, Any],
    scaler: dict[str, Any],
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    x = apply_feature_scaler(dataset["X"].clone(), scaler).to(device)
    p_market = dataset["p_market"].to(device)
    out = model(x)
    final_logits = torch.log(p_market.clamp_min(EPS)) + out["delta_logits"]
    p_final = F.softmax(final_logits, dim=-1).detach().cpu()
    q_diff = F.softmax(out["goal_diff_logits"], dim=-1)
    p_from_diff = fold_goal_diff_probs(q_diff).detach().cpu()
    rows = []
    for idx, bookmaker in enumerate(dataset["bookmakers"]):
        probs = _normalise(p_final[idx].tolist())
        market = _normalise(dataset["p_market"][idx].tolist())
        diff_probs = _normalise(p_from_diff[idx].tolist())
        edges = [probs[i] - market[i] for i in range(3)]
        pred_idx = max(range(3), key=lambda item: probs[item])
        edge_idx = max(range(3), key=lambda item: edges[item])
        row = {
            "seed": int(seed),
            "match_id": dataset["match_ids"][idx],
            "bookmaker": bookmaker,
            "p_home": probs[0],
            "p_draw": probs[1],
            "p_away": probs[2],
            "market_home": market[0],
            "market_draw": market[1],
            "market_away": market[2],
            "edge_home": edges[0],
            "edge_draw": edges[1],
            "edge_away": edges[2],
            "p_from_diff_home": diff_probs[0],
            "p_from_diff_draw": diff_probs[1],
            "p_from_diff_away": diff_probs[2],
            "top_model_outcome": OUTCOMES[pred_idx],
            "top_edge_outcome": OUTCOMES[edge_idx],
        }
        row.update(dataset["latest_snapshots"][idx])
        rows.append(row)
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_safe_float(row.get(key)) for row in rows]
    return sum(vals) / len(vals) if vals else 0.0


def _summarize_prediction_rows(rows: list[dict[str, Any]], bookmaker: str | None = None) -> dict[str, Any]:
    probs = [_mean(rows, key) for key in PROB_KEYS]
    market = [_mean(rows, key) for key in MARKET_KEYS]
    probs = _normalise(probs)
    market = _normalise(market)
    edges = [probs[i] - market[i] for i in range(3)]
    pred_idx = max(range(3), key=lambda item: probs[item])
    edge_idx = max(range(3), key=lambda item: edges[item])
    payload = {
        "p_home": probs[0],
        "p_draw": probs[1],
        "p_away": probs[2],
        "market_home": market[0],
        "market_draw": market[1],
        "market_away": market[2],
        "edge_home": edges[0],
        "edge_draw": edges[1],
        "edge_away": edges[2],
        "top_model_outcome": OUTCOMES[pred_idx],
        "top_edge_outcome": OUTCOMES[edge_idx],
        "n_rows": len(rows),
    }
    if bookmaker is not None:
        payload["bookmaker"] = bookmaker
    return payload


def build_ensemble_report_payload(
    input_rows: list[dict[str, Any]],
    seed_outputs: list[dict[str, Any]],
    run_source: str,
    checkpoint_note: str,
) -> dict[str, Any]:
    assert_prediction_rows_are_pre_match(input_rows)
    flat_rows = [row for seed_payload in seed_outputs for row in seed_payload.get("rows", [])]
    by_bookmaker: dict[str, list[dict[str, Any]]] = {}
    for row in flat_rows:
        by_bookmaker.setdefault(str(row.get("bookmaker", "")), []).append(row)
    bookmaker_ensembles = [
        _summarize_prediction_rows(rows, bookmaker=bookmaker)
        for bookmaker, rows in sorted(by_bookmaker.items(), key=lambda item: item[0])
    ]
    first = input_rows[0]
    fixture = {
        "match_id": first.get("match_id"),
        "source_match_id": first.get("source_match_id"),
        "home_team": first.get("home_team"),
        "away_team": first.get("away_team"),
        "kickoff_time_utc": first.get("kickoff_time_utc"),
        "label_status": first.get("label_status"),
        "bookmaker_count": len(input_rows),
    }
    return {
        "phase": "P17.1",
        "task": "pre_match_fixture_prediction",
        "fixture": fixture,
        "run_source": run_source,
        "checkpoint_note": checkpoint_note,
        "input_policy": {
            "no_test_split_loaded": True,
            "no_label_fabricated": True,
            "not_betting_advice": True,
        },
        "seeds": [item.get("seed") for item in seed_outputs],
        "ensemble": _summarize_prediction_rows(flat_rows),
        "bookmaker_ensembles": bookmaker_ensembles,
        "seed_outputs": seed_outputs,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report_md(path: Path, payload: dict[str, Any]) -> None:
    fixture = payload["fixture"]
    ensemble = payload["ensemble"]
    lines = [
        "# P17.1 Fixture Prediction",
        "",
        f"- fixture: {fixture.get('home_team')} vs {fixture.get('away_team')}",
        f"- match_id: `{fixture.get('match_id')}`",
        f"- kickoff_time_utc: `{fixture.get('kickoff_time_utc')}`",
        f"- checkpoint_note: `{payload.get('checkpoint_note')}`",
        f"- run_source: `{payload.get('run_source')}`",
        "",
        "## Ensemble",
        f"- model 1X2: home={ensemble['p_home']:.6f}, draw={ensemble['p_draw']:.6f}, away={ensemble['p_away']:.6f}",
        f"- market anchor: home={ensemble['market_home']:.6f}, draw={ensemble['market_draw']:.6f}, away={ensemble['market_away']:.6f}",
        f"- edge: home={ensemble['edge_home']:.6f}, draw={ensemble['edge_draw']:.6f}, away={ensemble['edge_away']:.6f}",
        f"- top_model_outcome: `{ensemble['top_model_outcome']}`",
        f"- top_edge_outcome: `{ensemble['top_edge_outcome']}`",
        "",
        "## Bookmakers",
    ]
    for row in payload.get("bookmaker_ensembles", []):
        lines.append(
            f"- {row['bookmaker']}: model {row['p_home']:.6f}/{row['p_draw']:.6f}/{row['p_away']:.6f}; "
            f"market {row['market_home']:.6f}/{row['market_draw']:.6f}/{row['market_away']:.6f}; "
            f"edge {row['edge_home']:.6f}/{row['edge_draw']:.6f}/{row['edge_away']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "- This is pre-match inference on unlabeled prediction rows.",
            "- It is not a real-money betting recommendation.",
            "- The market anchor is no-vig Euro probability from the nearest valid pre-kickoff event.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P17.1 pre-match fixture inference")
    parser.add_argument("--input", required=True, help="prediction_export_v1 JSONL or multi-bookmaker JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-root", default="runs/p14_market_replication")
    parser.add_argument("--run-prefix", default="p14_market_kl_030_balanced_seed")
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P17.1 fixture inference without --no-test")
    assert_no_test_paths([args.input, args.out_dir])
    out_dir = Path(args.out_dir)
    if not args.allow_overwrite and ((out_dir / "fixture_prediction_report.json").exists() or (out_dir / "fixture_prediction_report.md").exists()):
        raise SystemExit(f"Refusing to overwrite existing fixture prediction outputs in {out_dir}")
    rows = load_prediction_rows(Path(args.input))
    seeds = _parse_ints(args.seeds)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    seed_outputs = []
    all_rows = []
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_note = "p14_market_kl_030_balanced checkpoints; exact pre-match inference, no labels"
    for seed in seeds:
        run_dir = checkpoint_root / f"{args.run_prefix}{seed}"
        model, info, scaler = load_checkpoint_model(run_dir, device)
        feature_names = list(info["feature_names"])
        dataset = build_fixture_dataset(rows, feature_names)
        pred_rows = predict_fixture_rows(model, dataset, scaler, seed, device)
        seed_outputs.append({"seed": seed, "run_dir": str(run_dir), "rows": pred_rows})
        all_rows.extend(pred_rows)

    payload = build_ensemble_report_payload(rows, seed_outputs, run_source=str(checkpoint_root), checkpoint_note=checkpoint_note)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "fixture_prediction_by_seed.csv", all_rows)
    write_csv(out_dir / "fixture_prediction_by_bookmaker.csv", payload["bookmaker_ensembles"])
    (out_dir / "fixture_prediction_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report_md(out_dir / "fixture_prediction_report.md", payload)
    print(
        "P17.1 fixture ensemble: "
        f"home={payload['ensemble']['p_home']:.6f}, "
        f"draw={payload['ensemble']['p_draw']:.6f}, "
        f"away={payload['ensemble']['p_away']:.6f}"
    )
    print(f"Report saved to {out_dir / 'fixture_prediction_report.md'}")


if __name__ == "__main__":
    main()
