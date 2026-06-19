"""
OddsMind Real Dataset Dry Run Orchestrator (P0.9)

Runs the full pipeline: import → quality audit → split → baselines → training → eval → comparison.

Usage (with fixture — smoke test):
    python tools/odds_real_dataset_dry_run.py \
        --input data/odds_fixtures/raw/football_data_sample.csv \
        --preset football_data_b365 --league-id EPL --bookmaker-id B365 \
        --upper-side home --limit 20 --out-dir runs/p0_9_smoke

Usage (with real data — requires user to provide path):
    python tools/odds_real_dataset_dry_run.py \
        --input path/to/real.csv --is-real \
        --preset football_data_b365 --league-id EPL --bookmaker-id B365 \
        --upper-side home --limit 500 --out-dir runs/p0_9_real
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run(cmd: list, label: str) -> bool:
    print(f"\n--- {label} ---")
    print(f"  {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[-300:]}")
        return False
    print(f"  OK")
    if r.stdout.strip():
        # Print first few lines
        lines = r.stdout.strip().split("\n")[:8]
        for l in lines:
            print(f"  | {l[:120]}")
    return True


def main():
    parser = argparse.ArgumentParser(description="OddsMind Real Dataset Dry Run")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--is-real", action="store_true",
                        help="Mark as real data (triggers actual write + train)")
    parser.add_argument("--preset", type=str, default="football_data_b365")
    parser.add_argument("--league-id", type=str, default="EPL")
    parser.add_argument("--season", type=str, default="")
    parser.add_argument("--bookmaker-id", type=str, default="B365")
    parser.add_argument("--upper-side", type=str, default="home")
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--default-minutes", type=float, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="runs/oddsmind_p0_9_dry_run")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    jsonl_path = os.path.join(args.out_dir, "imported.jsonl")
    import_report_path = os.path.join(args.out_dir, "import_report.json")
    split_dir = os.path.join(args.out_dir, "splits")
    quality_path = os.path.join(args.out_dir, "quality_audit.json")
    det_baseline_path = os.path.join(args.out_dir, "deterministic_baseline_eval.json")
    tabular_dir = os.path.join(args.out_dir, "tabular_logistic")
    tabular_eval_path = os.path.join(args.out_dir, "tabular_eval.json")
    oddsmind_dir = os.path.join(args.out_dir, "oddsmind_supervised")
    oddsmind_eval_path = os.path.join(args.out_dir, "oddsmind_eval.json")
    comparison_path = os.path.join(args.out_dir, "comparison.json")

    # ── Step 1: Import ──────────────────────────────────────────────
    ok = run([
        sys.executable, "tools/odds_import_csv.py",
        "--input", args.input, "--preset", args.preset,
        "--league-id", args.league_id, "--season", args.season,
        "--bookmaker-id", args.bookmaker_id,
        "--upper-side", args.upper_side,
        "--asian-label-mode", args.asian_label_mode,
        "--default-minutes-before-kickoff", str(args.default_minutes),
        "--limit", str(args.limit),
        "--write", "--out", jsonl_path, "--report", import_report_path,
    ], "Import CSV")
    if not ok:
        print("\nImport failed — stopping.")
        sys.exit(1)

    # ── Step 2: Quality audit ───────────────────────────────────────
    run([
        sys.executable, "eval/eval_odds_real_data_quality.py",
        "--data", jsonl_path, "--out-json", quality_path,
    ], "Quality audit")

    # ── Step 3: Build splits ────────────────────────────────────────
    run([
        sys.executable, "tools/odds_build_splits.py",
        "--data", jsonl_path, "--out-dir", split_dir,
        "--train-ratio", "0.7", "--val-ratio", "0.15", "--test-ratio", "0.15",
    ], "Build splits")

    # ── Step 4: Deterministic baseline ──────────────────────────────
    test_ids = os.path.join(split_dir, "test_match_ids.txt")
    run([
        sys.executable, "eval/eval_odds_baselines.py",
        "--data", jsonl_path,
        "--split-match-ids", test_ids,
        "--cutoffs", "0", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", args.asian_label_mode,
        "--out-json", det_baseline_path,
    ], "Deterministic baseline")

    # ── Step 5: Tabular baseline train + eval ───────────────────────
    train_ids = os.path.join(split_dir, "train_match_ids.txt")
    val_ids = os.path.join(split_dir, "val_match_ids.txt")
    run([
        sys.executable, "trainer/train_odds_tabular_baseline.py",
        "--data", jsonl_path, "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--model-type", "logistic",
        "--device", "cpu", "--cutoffs", "0", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", args.asian_label_mode,
        "--train-match-ids", train_ids,
        "--val-match-ids", val_ids, "--eval-every-epoch",
        "--out-dir", tabular_dir,
    ], "Tabular train")
    run([
        sys.executable, "eval/eval_odds_tabular_baseline.py",
        "--data", jsonl_path,
        "--model", os.path.join(tabular_dir, "odds_tabular_logistic.pth"),
        "--model-type", "logistic",
        "--split-match-ids", test_ids,
        "--cutoffs", "0", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", args.asian_label_mode,
        "--device", "cpu", "--out-json", tabular_eval_path,
    ], "Tabular eval")

    # ── Step 6: OddsMind supervised train + eval ────────────────────
    run([
        sys.executable, "trainer/train_odds_supervised.py",
        "--data", jsonl_path, "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--hidden-size", str(args.hidden_size),
        "--num-layers", str(args.num_layers),
        "--num-heads", str(args.num_heads),
        "--device", "cpu", "--cutoffs", "0", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", args.asian_label_mode,
        "--train-match-ids", train_ids,
        "--val-match-ids", val_ids, "--eval-every-epoch",
        "--transformer-backend", "odds_native",
        "--out-dir", oddsmind_dir,
    ], "OddsMind train")
    run([
        sys.executable, "eval/eval_odds_model.py",
        "--data", jsonl_path,
        "--model", os.path.join(oddsmind_dir, "oddsmind_smoke.pth"),
        "--split-match-ids", test_ids,
        "--cutoffs", "0", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", args.asian_label_mode,
        "--hidden-size", str(args.hidden_size),
        "--num-layers", str(args.num_layers),
        "--num-heads", str(args.num_heads),
        "--device", "cpu", "--transformer-backend", "odds_native",
        "--out-json", oddsmind_eval_path,
    ], "OddsMind eval")

    # ── Step 7: Comparison ──────────────────────────────────────────
    run([
        sys.executable, "eval/compare_odds_evals.py",
        "--eval-json", f"deterministic={det_baseline_path}",
        "--eval-json", f"tabular={tabular_eval_path}",
        "--eval-json", f"oddsmind={oddsmind_eval_path}",
        "--out-json", comparison_path,
    ], "Comparison")

    print(f"\n=== Dry run complete ===")
    print(f"  Output: {args.out_dir}")
    for f in os.listdir(args.out_dir):
        print(f"    {f}")


if __name__ == "__main__":
    main()
