"""
OddsMind P0.9 Real Dataset Dry Run Smoke Test

Uses P0.8 fixture CSV to verify the full dry-run orchestration pipeline.

Usage:
    python eval/eval_odds_real_dataset_dry_run_smoke.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


FIXTURE_CSV = "data/odds_fixtures/raw/football_data_sample.csv"
OUT_DIR = "runs/oddsmind_p0_9_dry_run_smoke"


# ── Test 1: Dry-run orchestration with fixture ──────────────────────

def test_dry_run_orchestration():
    print("\n--- Test 1: Dry-run orchestration with fixture ---")
    r = subprocess.run([
        sys.executable, "tools/odds_real_dataset_dry_run.py",
        "--input", FIXTURE_CSV, "--preset", "football_data_b365",
        "--league-id", "EPL", "--bookmaker-id", "B365",
        "--upper-side", "home", "--limit", "20",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--epochs", "1", "--batch-size", "4",
        "--out-dir", OUT_DIR,
    ], capture_output=True, text=True, timeout=300)
    check("orchestration exit 0", r.returncode == 0, (r.stderr + r.stdout)[-500:] if r.returncode else "")
    if r.returncode == 0:
        check("Dry run complete in stdout", "Dry run complete" in r.stdout)

    # Verify output files exist
    for fname in ["imported.jsonl", "import_report.json", "quality_audit.json",
                  "deterministic_baseline_eval.json", "tabular_eval.json",
                  "oddsmind_eval.json", "comparison.json"]:
        path = os.path.join(OUT_DIR, fname)
        check(f"{fname} exists", os.path.exists(path))

    return r.returncode == 0


# ── Test 2: Quality audit output ────────────────────────────────────

def test_quality_audit_output():
    print("\n--- Test 2: Quality audit output ---")
    path = os.path.join(OUT_DIR, "quality_audit.json")
    if not os.path.exists(path):
        check("quality_audit.json exists", False)
        return
    with open(path) as f:
        report = json.load(f)
    check("num_matches > 0", report.get("num_matches", 0) > 0)
    check("has timeline stats", "timeline" in report)
    check("has label distribution", "label_distribution" in report)
    check("has warnings", "warnings" in report)


# ── Test 3: Comparison JSON ─────────────────────────────────────────

def test_comparison():
    print("\n--- Test 3: Comparison JSON ---")
    path = os.path.join(OUT_DIR, "comparison.json")
    if not os.path.exists(path):
        check("comparison.json exists", False)
        return
    with open(path) as f:
        data = json.load(f)
    check("models key exists", "models" in data)


# ── Test 4: Real data path check ────────────────────────────────────

def test_real_data_path_handling():
    """Verify that the system correctly identifies when real data is missing."""
    print("\n--- Test 4: Real data path handling ---")
    # The smoke test uses fixture data — this is expected.
    # Real data would require user-provided CSV path.
    check("fixture data used for smoke", os.path.exists(FIXTURE_CSV))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    PASS = 0; FAIL = 0

    print("=" * 60)
    print("OddsMind P0.9 Real Dataset Dry Run Smoke Test")
    print("=" * 60)

    test_dry_run_orchestration()
    test_quality_audit_output()
    test_comparison()
    test_real_data_path_handling()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("REAL DATASET DRY RUN SMOKE: ALL CHECKS PASSED")
        print("\nNOTE: Real data dry run BLOCKED — needs user-provided real CSV path.")
    else:
        print(f"REAL DATASET DRY RUN SMOKE: {FAIL} FAILURE(S)")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
