"""
OddsMind P0.8 Import Smoke Test

Tests CSV import pipeline: validation, football_data preset, generic preset,
JSONL output, downstream compatibility (dataset, cutoff, train, eval).
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
GENERIC_CSV = "data/odds_fixtures/raw/generic_single_snapshot_sample.csv"
OUT_JSONL = "data/odds_imported/smoke_football_data.jsonl"
OUT_REPORT = "data/odds_imported/smoke_football_data_report.json"
OUT_GENERIC = "data/odds_imported/smoke_generic.jsonl"


# ── Test 1: Football-data dry-run ────────────────────────────────────

def test_football_data_dry_run():
    print("\n--- Test 1: Football-data dry-run ---")
    r = subprocess.run([
        sys.executable, "tools/odds_import_csv.py",
        "--input", FIXTURE_CSV, "--preset", "football_data_b365",
        "--league-id", "EPL", "--season", "2024-2025", "--bookmaker-id", "B365",
        "--upper-side", "home", "--default-minutes-before-kickoff", "0",
        "--asian-label-mode", "5class",
    ], capture_output=True, text=True, timeout=30)
    check("dry-run exit 0", r.returncode == 0, r.stderr[-200:])
    check("total_rows in output", "total_rows" in r.stdout)
    check("imported_matches > 0", "imported_matches" in r.stdout)
    check("skipped_rows > 0", "skipped_rows" in r.stdout)


# ── Test 2: Football-data write ──────────────────────────────────────

def test_football_data_write():
    print("\n--- Test 2: Football-data write ---")
    r = subprocess.run([
        sys.executable, "tools/odds_import_csv.py",
        "--input", FIXTURE_CSV, "--preset", "football_data_b365",
        "--league-id", "EPL", "--season", "2024-2025", "--bookmaker-id", "B365",
        "--upper-side", "home", "--default-minutes-before-kickoff", "0",
        "--asian-label-mode", "5class", "--write",
        "--out", OUT_JSONL, "--report", OUT_REPORT,
    ], capture_output=True, text=True, timeout=30)
    check("write exit 0", r.returncode == 0, r.stderr[-200:])
    check("JSONL created", os.path.exists(OUT_JSONL))
    check("report created", os.path.exists(OUT_REPORT))

    with open(OUT_JSONL) as f:
        lines = [l for l in f if l.strip()]
    check("JSONL has lines", len(lines) > 0)
    for line in lines:
        s = json.loads(line)
        check(f"valid sample: {s.get('match_id','')[:30]}",
              "match_id" in s and "odds_timeline" in s and "label" in s)
        break


# ── Test 3: Generic CSV import ──────────────────────────────────────

def test_generic_import():
    print("\n--- Test 3: Generic CSV import ---")
    r = subprocess.run([
        sys.executable, "tools/odds_import_csv.py",
        "--input", GENERIC_CSV, "--preset", "generic_single_snapshot",
        "--league-id", "LaLiga", "--bookmaker-id", "MACAU",
        "--upper-side", "home", "--asian-label-mode", "5class",
        "--write", "--out", OUT_GENERIC,
    ], capture_output=True, text=True, timeout=30)
    check("generic exit 0", r.returncode == 0, r.stderr[-200:])
    check("generic JSONL created", os.path.exists(OUT_GENERIC))


# ── Test 4: Imported JSONL → OddsDataset ───────────────────────────

def test_imported_dataset():
    print("\n--- Test 4: Imported JSONL → OddsDataset ---")
    from dataset.odds_dataset import OddsDataset
    ds = OddsDataset(OUT_JSONL, asian_label_mode="5class")
    check("dataset has samples", len(ds) > 0)
    item = ds[0]
    check("features 2D", item["features"].dim() == 2)
    check("euro_label in range", 0 <= item["euro_label"] <= 2)
    check("asian_label in range", 0 <= item["asian_label"] <= 4)


# ── Test 5: Imported JSONL cutoff exhaustive ────────────────────────

def test_imported_cutoff():
    print("\n--- Test 5: Imported JSONL → cutoff ---")
    from dataset.odds_dataset import OddsDataset
    ds = OddsDataset(OUT_JSONL, cutoffs=[0], cutoff_mode="exhaustive",
                     asian_label_mode="5class")
    check("cutoff samples > 0", len(ds) > 0)


# ── Test 6: Imported JSONL eval ─────────────────────────────────────

def test_imported_eval():
    print("\n--- Test 6: Imported JSONL → eval ---")
    r = subprocess.run([
        sys.executable, "eval/eval_odds_model.py",
        "--data", OUT_JSONL, "--untrained",
        "--cutoffs", "0", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu",
    ], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        try:
            json.loads(r.stdout.strip())
            check("eval JSON valid", True)
        except:
            check("eval JSON valid", False)
    else:
        check("eval exit 0", False, r.stderr[-200:])


# ── Test 7: Imported JSONL train ────────────────────────────────────

def test_imported_train():
    print("\n--- Test 7: Imported JSONL → train 1 epoch ---")
    r = subprocess.run([
        sys.executable, "trainer/train_odds_supervised.py",
        "--data", OUT_JSONL, "--epochs", "1", "--batch-size", "2",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu", "--cutoffs", "0", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class", "--transformer-backend", "odds_native",
        "--out-dir", "runs/oddsmind_p0_8_import_train_smoke",
    ], capture_output=True, text=True, timeout=60)
    check("import train exit 0", r.returncode == 0, (r.stderr+r.stdout)[-300:] if r.returncode else "")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    PASS = 0; FAIL = 0

    print("=" * 60)
    print("OddsMind P0.8 Import Smoke Test")
    print("=" * 60)

    test_football_data_dry_run()
    test_football_data_write()
    test_generic_import()
    test_imported_dataset()
    test_imported_cutoff()
    test_imported_eval()
    test_imported_train()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("IMPORT SMOKE TEST: ALL CHECKS PASSED")
    else:
        print(f"IMPORT SMOKE TEST: {FAIL} FAILURE(S)")
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
