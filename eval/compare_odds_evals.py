"""
OddsMind Eval Comparison Script (P0.5B)

Compares multiple evaluation JSON files (deterministic baselines,
tabular baselines, OddsMind Transformer model) side by side.

Usage:
    python eval/compare_odds_evals.py \
        --eval-json tabular=runs/tabular_eval.json \
        --eval-json deterministic=runs/baseline_eval.json \
        --out-json runs/comparison.json
"""

import argparse
import json
import os
import sys


def load_eval(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(data: dict, key_path: list) -> dict:
    """Walk a nested dict path to extract accuracy/logloss/brier."""
    d = data
    for k in key_path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return {}
    if isinstance(d, dict):
        return {
            "accuracy": d.get("accuracy"),
            "logloss": d.get("logloss"),
            "brier": d.get("brier"),
        }
    return {}


def main():
    parser = argparse.ArgumentParser(description="Compare OddsMind Evaluations")
    parser.add_argument("--eval-json", type=str, action="append", default=[],
                        help="Label=path pairs, e.g. 'tabular=runs/tabular_eval.json'")
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    evals = {}
    for entry in args.eval_json:
        if "=" in entry:
            label, path = entry.split("=", 1)
        else:
            label = os.path.basename(entry).replace(".json", "")
            path = entry
        evals[label] = load_eval(path)

    comparison = {"models": {}}

    for label, data in evals.items():
        entry = {}

        # For deterministic baseline format: data["baselines"]["euro_implied_prob"]
        if "baselines" in data:
            for bl_name in data["baselines"]:
                bl = data["baselines"][bl_name]
                if "accuracy" in bl:
                    entry[f"{bl_name}_euro_acc"] = bl.get("accuracy")
                    entry[f"{bl_name}_euro_logloss"] = bl.get("logloss")
                    entry[f"{bl_name}_brier"] = bl.get("brier")
        else:
            # Standard eval format: data["euro"]["accuracy"]
            entry["euro_accuracy"] = extract_metrics(data, ["euro"]).get("accuracy")
            entry["euro_logloss"] = extract_metrics(data, ["euro"]).get("logloss")
            entry["euro_brier"] = extract_metrics(data, ["euro"]).get("brier")
            entry["asian_accuracy"] = extract_metrics(data, ["asian"]).get("accuracy")
            entry["asian_logloss"] = extract_metrics(data, ["asian"]).get("logloss")
            entry["asian_brier"] = extract_metrics(data, ["asian"]).get("brier")
            entry["num_samples"] = data.get("num_samples")

        comparison["models"][label] = entry

    output = json.dumps(comparison, indent=2, default=str)
    print(output)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(output + "\n")


if __name__ == "__main__":
    main()
