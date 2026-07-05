"""Prediction-distribution diagnostics for P3.1."""

import argparse
import csv
import json
import math
import os
from pathlib import Path


CLASS_NAMES = ["home", "draw", "away"]
PROB_FIELDS = ["p_home", "p_draw", "p_away"]
EPS = 1e-12


def _as_float(value) -> float:
    return float(value)


def _as_int(value) -> int:
    return int(float(value))


def read_prediction_csv(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "match_id": row.get("match_id", ""),
                    "y_true": _as_int(row["y_true"]),
                    "p_home": _as_float(row["p_home"]),
                    "p_draw": _as_float(row["p_draw"]),
                    "p_away": _as_float(row["p_away"]),
                    "pred_class": _as_int(row["pred_class"]),
                    "correct": _as_int(row.get("correct", 0)),
                }
            )
    return rows


def _distribution(values: list[int]) -> dict:
    return {name: sum(1 for value in values if value == idx) for idx, name in enumerate(CLASS_NAMES)}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _ece_binary(confidences: list[float], outcomes: list[int], n_bins: int = 10) -> tuple[float, list[dict]]:
    bins = []
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        selected = [
            idx for idx, confidence in enumerate(confidences)
            if (confidence >= lo and (confidence < hi or (i == n_bins - 1 and confidence <= hi)))
        ]
        if not selected:
            bins.append({"bin": i, "low": lo, "high": hi, "count": 0, "mean_confidence": None, "empirical_rate": None})
            continue
        mean_conf = sum(confidences[idx] for idx in selected) / len(selected)
        empirical = sum(outcomes[idx] for idx in selected) / len(selected)
        ece += (len(selected) / total) * abs(mean_conf - empirical)
        bins.append(
            {
                "bin": i,
                "low": _round(lo),
                "high": _round(hi),
                "count": len(selected),
                "mean_confidence": _round(mean_conf),
                "empirical_rate": _round(empirical),
            }
        )
    return _round(ece), bins


def _ece_multiclass(rows: list[dict], n_bins: int = 10) -> float:
    confidences = []
    outcomes = []
    for row in rows:
        probs = [row[field] for field in PROB_FIELDS]
        pred = max(range(3), key=lambda idx: probs[idx])
        confidences.append(probs[pred])
        outcomes.append(1 if pred == row["y_true"] else 0)
    ece, _ = _ece_binary(confidences, outcomes, n_bins=n_bins)
    return ece


def compute_prediction_diagnostics(model_name: str, rows: list[dict]) -> dict:
    y_true = [_as_int(row["y_true"]) for row in rows]
    pred = [_as_int(row["pred_class"]) for row in rows]
    probs = [[_as_float(row[field]) for field in PROB_FIELDS] for row in rows]

    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for true_label, pred_label in zip(y_true, pred):
        confusion[true_label][pred_label] += 1

    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        tp = confusion[idx][idx]
        fp = sum(confusion[other][idx] for other in range(3) if other != idx)
        fn = sum(confusion[idx][other] for other in range(3) if other != idx)
        support = sum(confusion[idx])
        class_losses = [-math.log(max(prob[idx], EPS)) for true, prob in zip(y_true, probs) if true == idx]
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        per_class[name] = {
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(_safe_div(2 * precision * recall, precision + recall)),
            "support": support,
            "logloss": _round(sum(class_losses) / len(class_losses)) if class_losses else None,
        }

    mean_probs = {
        name: _round(sum(prob[idx] for prob in probs) / len(probs))
        for idx, name in enumerate(CLASS_NAMES)
    }
    mean_prob_by_true_class = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        selected = [prob for true, prob in zip(y_true, probs) if true == idx]
        if selected:
            mean_prob_by_true_class[class_name] = {
                prob_name: _round(sum(prob[pidx] for prob in selected) / len(selected))
                for pidx, prob_name in enumerate(PROB_FIELDS)
            }
        else:
            mean_prob_by_true_class[class_name] = {prob_name: None for prob_name in PROB_FIELDS}

    true_draw_indices = [idx for idx, label in enumerate(y_true) if label == 1]
    non_draw_indices = [idx for idx, label in enumerate(y_true) if label != 1]
    mean_p_draw_on_true_draw = (
        sum(probs[idx][1] for idx in true_draw_indices) / len(true_draw_indices)
        if true_draw_indices else 0.0
    )
    mean_p_draw_on_non_draw = (
        sum(probs[idx][1] for idx in non_draw_indices) / len(non_draw_indices)
        if non_draw_indices else 0.0
    )
    draw_ece, draw_bins = _ece_binary([prob[1] for prob in probs], [1 if label == 1 else 0 for label in y_true])
    per_class_ece = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class_ece[name], _ = _ece_binary([prob[idx] for prob in probs], [1 if label == idx else 0 for label in y_true])

    predicted_distribution = _distribution(pred)
    draw_argmax_count = predicted_distribution["draw"]
    draw_recall = per_class["draw"]["recall"]

    return {
        "model": model_name,
        "n_samples": len(rows),
        "true_class_distribution": _distribution(y_true),
        "predicted_class_distribution": predicted_distribution,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "per_class_logloss": {name: per_class[name]["logloss"] for name in CLASS_NAMES},
        "mean_probabilities": mean_probs,
        "mean_predicted_probability_by_true_class": mean_prob_by_true_class,
        "mean_p_draw": mean_probs["draw"],
        "mean_p_draw_on_true_draw": _round(mean_p_draw_on_true_draw),
        "mean_p_draw_on_non_draw": _round(mean_p_draw_on_non_draw),
        "draw_argmax_count": draw_argmax_count,
        "draw_recall": draw_recall,
        "draw_calibration_bins": draw_bins,
        "ece": _ece_multiclass(rows),
        "ece_per_class": per_class_ece,
        "draw_argmax_collapse": draw_argmax_count == 0,
        "draw_prob_underconfident": mean_p_draw_on_true_draw < 0.15,
    }


def _model_name_from_path(path: str | Path) -> str:
    name = Path(path).stem
    for suffix in ("_val_predictions", "_predictions"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _write_markdown(payload: dict, out_md: str | Path) -> None:
    lines = [
        "# P3 Prediction Diagnostics",
        "",
        "| Model | argmax_home | argmax_draw | argmax_away | draw_recall | mean_p_draw | mean_p_draw_true_draw | logloss_home | logloss_draw | logloss_away | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, diag in payload["models"].items():
        pred_dist = diag["predicted_class_distribution"]
        losses = diag["per_class_logloss"]
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    str(pred_dist["home"]),
                    str(pred_dist["draw"]),
                    str(pred_dist["away"]),
                    _fmt(diag["draw_recall"]),
                    _fmt(diag["mean_p_draw"]),
                    _fmt(diag["mean_p_draw_on_true_draw"]),
                    _fmt(losses["home"]),
                    _fmt(losses["draw"]),
                    _fmt(losses["away"]),
                    _fmt(diag["ece"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Verdict"])
    for item in payload["verdict"]:
        lines.append(f"- `{item}`")
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(pred_paths: list[str]) -> dict:
    models = {}
    verdict = set()
    for path in pred_paths:
        model_name = _model_name_from_path(path)
        diag = compute_prediction_diagnostics(model_name, read_prediction_csv(path))
        models[model_name] = diag
        if diag["draw_argmax_collapse"]:
            verdict.add("DRAW_ARGMAX_COLLAPSE")
        if diag["draw_prob_underconfident"]:
            verdict.add("DRAW_PROB_UNDERCONFIDENT")

    if any(name.startswith("raw") and diag.get("draw_argmax_collapse") for name, diag in models.items()):
        verdict.add("DRAW_COLLAPSE_NOT_TRANSFORMER_SPECIFIC")
    if "p3a" in models and "p3b" in models:
        if models["p3b"]["mean_p_draw_on_true_draw"] < models["p3a"]["mean_p_draw_on_true_draw"]:
            verdict.add("P3B_DRAW_CALIBRATION_WORSE_THAN_P3A")
    return {"models": models, "verdict": sorted(verdict)}


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 prediction diagnostics")
    parser.add_argument("--pred", action="append", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    payload = build_payload(args.pred)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_markdown(payload, args.out_md)
    print(f"Prediction diagnostics saved to {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
