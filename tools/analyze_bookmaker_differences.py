"""
Bookmaker Difference Analysis — H1 through H6 (sub-step 2b)

Analyses multi-bookmaker data from v6_phase1a.jsonl to test six
hypotheses about bookmaker behaviour. Results inform the baseline
bookmaker selection strategy for Phase 1a.

Hypotheses:
  H1: Different bookmakers have different margin / overround levels
  H2: Different bookmakers have different favourite-longshot bias profiles
  H3: Different bookmakers have systematic bias toward home/favourite teams
  H4: Lead-lag in price changes (requires timeline data — may be deferred)
  H5: Higher inter-bookmaker disagreement predicts higher uncertainty
  H6: Euro-Asian inconsistency predicts higher prediction error

Usage:
    python tools/analyze_bookmaker_differences.py \
        --data data/odds_real/v6_phase1a.jsonl \
        --out-json eval/reports/bookmaker_analysis.json
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_baselines_v2 import Phase1Baseline

EURO_MAP = {"home": 0, "draw": 1, "away": 2}


def load_rows(path: str) -> list:
    """Load all rows from flat Phase 1a JSONL."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_by_match(rows: list) -> dict:
    """Group rows by match_id → list of bookmaker rows."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["match_id"]].append(r)
    return dict(groups)


# ── H1: Margin differences ──────────────────────────────────────────

def analyse_h1(rows: list) -> dict:
    """
    Compute mean overround (margin) per bookmaker × league.

    margin = (1/euro_h + 1/euro_d + 1/euro_a) - 1
    """
    stats = defaultdict(lambda: {"margins": [], "count": 0})
    for r in rows:
        bk = r.get("bookmaker_id", "unknown")
        lg = r.get("league_id", "unknown")
        try:
            h, d, a = float(r["close_euro_h"]), float(r["close_euro_d"]), float(r["close_euro_a"])
            if h <= 0 or d <= 0 or a <= 0:
                continue
        except (TypeError, ValueError, KeyError):
            continue
        margin = (1.0 / h + 1.0 / d + 1.0 / a) - 1.0
        key = f"{bk}|{lg}"
        stats[key]["margins"].append(margin)
        stats[key]["count"] += 1

    result = {}
    for key, val in stats.items():
        ms = val["margins"]
        if len(ms) < 5:
            continue
        mean_m = sum(ms) / len(ms)
        sorted_ms = sorted(ms)
        result[key] = {
            "bookmaker": key.split("|")[0],
            "league": key.split("|")[1],
            "count": val["count"],
            "mean_margin": round(mean_m, 6),
            "median_margin": round(sorted_ms[len(ms) // 2], 6),
            "p10_margin": round(sorted_ms[len(ms) // 10], 6),
            "p90_margin": round(sorted_ms[int(len(ms) * 0.9)], 6),
        }

    # Per-bookmaker summary (across all leagues)
    bk_summary = defaultdict(lambda: {"margins": []})
    for key, val in stats.items():
        bk = key.split("|")[0]
        bk_summary[bk]["margins"].extend(val["margins"])
    bk_result = {}
    for bk, val in bk_summary.items():
        ms = val["margins"]
        bk_result[bk] = {
            "total_count": len(ms),
            "mean_margin": round(sum(ms) / max(len(ms), 1), 6),
        }
    result["_by_bookmaker"] = bk_result
    return result


# ── H2: Favourite-Longshot Bias ─────────────────────────────────────

def analyse_h2(rows: list) -> dict:
    """
    For each bookmaker, bucket implied probabilities and compare
    average implied probability against observed frequency.

    Positive bias = implied > actual → bookmaker overcharges for
    this bucket (underestimates true probability).

    Buckets: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    """
    baseline = Phase1Baseline()
    # Collect per-bookmaker: list of (implied_prob, actual_result)
    bk_data = defaultdict(lambda: {"home": [], "draw": [], "away": []})

    for r in rows:
        bk = r.get("bookmaker_id", "unknown")
        try:
            q_h, q_d, q_a = baseline.compute_euro_prior(
                float(r["close_euro_h"]),
                float(r["close_euro_d"]),
                float(r["close_euro_a"]),
            )
        except (TypeError, ValueError, KeyError):
            continue
        label = r["label"]["euro_result"]
        actual_h = 1.0 if label == "home" else 0.0
        actual_d = 1.0 if label == "draw" else 0.0
        actual_a = 1.0 if label == "away" else 0.0

        bk_data[bk]["home"].append((q_h, actual_h))
        bk_data[bk]["draw"].append((q_d, actual_d))
        bk_data[bk]["away"].append((q_a, actual_a))

    bucket_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    result = {}

    for bk, outcomes in bk_data.items():
        bk_result = {}
        for outcome_name in ["home", "draw", "away"]:
            pairs = outcomes[outcome_name]
            buckets_out = {}
            for i in range(len(bucket_edges) - 1):
                lo, hi = bucket_edges[i], bucket_edges[i + 1]
                bucket_pairs = [
                    (imp, act)
                    for imp, act in pairs
                    if lo <= imp < hi or (hi == 1.0 and imp == 1.0)
                ]
                if len(bucket_pairs) < 10:
                    continue
                avg_imp = sum(p[0] for p in bucket_pairs) / len(bucket_pairs)
                avg_act = sum(p[1] for p in bucket_pairs) / len(bucket_pairs)
                bias = avg_imp - avg_act
                bucket_key = f"{lo:.1f}-{hi:.1f}"
                buckets_out[bucket_key] = {
                    "count": len(bucket_pairs),
                    "avg_implied": round(avg_imp, 4),
                    "avg_actual": round(avg_act, 4),
                    "bias": round(bias, 4),
                }
            bk_result[outcome_name] = buckets_out
        result[bk] = bk_result

    return result


# ── H3: Home / Favourite Systemic Bias ──────────────────────────────

def analyse_h3(rows: list) -> dict:
    """
    By bookmaker, bucket matches by implied home-win probability
    and compare predicted vs actual home-win rate.

    Buckets: <0.3, 0.3-0.4, 0.4-0.5, 0.5-0.6, >0.6
    """
    baseline = Phase1Baseline()
    bk_data = defaultdict(list)

    for r in rows:
        bk = r.get("bookmaker_id", "unknown")
        try:
            q_h, _, _ = baseline.compute_euro_prior(
                float(r["close_euro_h"]),
                float(r["close_euro_d"]),
                float(r["close_euro_a"]),
            )
        except (TypeError, ValueError, KeyError):
            continue
        actual_h = 1.0 if r["label"]["euro_result"] == "home" else 0.0
        bk_data[bk].append((q_h, actual_h))

    bucket_edges = [0.0, 0.3, 0.4, 0.5, 0.6, 1.0]
    result = {}

    for bk, pairs in bk_data.items():
        bk_result = {}
        for i in range(len(bucket_edges) - 1):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            bucket_pairs = [
                (imp, act)
                for imp, act in pairs
                if lo <= imp < hi or (hi == 1.0 and imp == 1.0)
            ]
            if len(bucket_pairs) < 10:
                continue
            avg_imp = sum(p[0] for p in bucket_pairs) / len(bucket_pairs)
            avg_act = sum(p[1] for p in bucket_pairs) / len(bucket_pairs)
            bucket_key = f"{lo:.1f}-{hi:.1f}"
            bk_result[bucket_key] = {
                "count": len(bucket_pairs),
                "avg_implied_home": round(avg_imp, 4),
                "avg_actual_home": round(avg_act, 4),
                "bias": round(avg_imp - avg_act, 4),
            }
        result[bk] = bk_result

    return result


# ── H4: Lead-Lag ────────────────────────────────────────────────────

def analyse_h4(_rows: list) -> dict:
    """
    Lead-lag analysis requires the full timeline format (v6_all.jsonl)
    to compare the timing of price changes across bookmakers.

    The flat Phase 1a format does not contain enough temporal information
    to compute lead-lag.  Deferred to Phase 1b.
    """
    return {
        "status": "deferred",
        "reason": "Requires timeline format (v6_all.jsonl) — "
                  "flat Phase 1a data only has open/close, not "
                  "intermediate timestamps. Will revisit in Phase 1b.",
    }


# ── H5: Inter-bookmaker Disagreement vs Uncertainty ─────────────────

def analyse_h5(rows: list) -> dict:
    """
    For each match with ≥3 bookmakers, compute:
      - std of close_euro_h across bookmakers (disagreement proxy)
      - actual home-win rate for matches in that disagreement bucket
    """
    groups = group_by_match(rows)

    # Per-match disagreement
    match_disagreements = []
    for mid, bk_rows in groups.items():
        if len(bk_rows) < 3:
            continue
        home_odds = []
        for r in bk_rows:
            try:
                h = float(r["close_euro_h"])
                if h > 1.0:
                    home_odds.append(h)
            except (TypeError, ValueError, KeyError):
                continue
        if len(home_odds) < 3:
            continue
        mean_h = sum(home_odds) / len(home_odds)
        var_h = sum((x - mean_h) ** 2 for x in home_odds) / len(home_odds)
        std_h = math.sqrt(var_h)
        # Convert to implied prob disagreement
        imp_probs = [1.0 / x for x in home_odds]
        mean_imp = sum(imp_probs) / len(imp_probs)
        std_imp = math.sqrt(
            sum((x - mean_imp) ** 2 for x in imp_probs) / len(imp_probs)
        )
        actual_home = 1.0 if bk_rows[0]["label"]["euro_result"] == "home" else 0.0
        match_disagreements.append(
            {
                "match_id": mid,
                "num_bookmakers": len(home_odds),
                "mean_euro_h": round(mean_h, 4),
                "std_euro_h": round(std_h, 4),
                "std_implied_prob": round(std_imp, 6),
                "actual_home_win": actual_home,
                "avg_implied_home": round(mean_imp, 4),
            }
        )

    if len(match_disagreements) < 30:
        return {"status": "insufficient_data", "count": len(match_disagreements)}

    # Sort by disagreement and bucket
    sorted_m = sorted(match_disagreements, key=lambda x: x["std_implied_prob"])
    n = len(sorted_m)
    tercile_size = n // 3

    result = {}
    for label, indices in [
        ("low_disagreement", range(0, tercile_size)),
        ("mid_disagreement", range(tercile_size, 2 * tercile_size)),
        ("high_disagreement", range(2 * tercile_size, n)),
    ]:
        bucket = [sorted_m[i] for i in indices]
        if not bucket:
            continue
        n_b = len(bucket)
        home_rate = sum(m["actual_home_win"] for m in bucket) / n_b
        avg_imp = sum(m["avg_implied_home"] for m in bucket) / n_b
        avg_std = sum(m["std_implied_prob"] for m in bucket) / n_b
        result[label] = {
            "count": n_b,
            "avg_std_implied_prob": round(avg_std, 6),
            "avg_implied_home": round(avg_imp, 4),
            "actual_home_win_rate": round(home_rate, 4),
            "bias": round(avg_imp - home_rate, 4),
        }

    result["total_matches"] = n
    result["interpretation"] = (
        "If actual_home_win_rate varies systematically "
        "across disagreement buckets, inter-bookmaker "
        "disagreement is a useful uncertainty signal."
    )
    return result


# ── H6: Euro-Asian Inconsistency vs Prediction Error ────────────────

def analyse_h6(rows: list) -> dict:
    """
    For each bookmaker, compute |euro_p_h - asian_p_h| and bucket.
    Compare logloss in each inconsistency bucket.

    Only for line=0.0 (where asian directly implies home/away split
    independent of draw) and line=-0.5 (where asian implies p_h directly).
    """
    baseline = Phase1Baseline()
    bk_data = defaultdict(list)

    for r in rows:
        bk = r.get("bookmaker_id", "unknown")
        line = r.get("close_asian_line")
        if line not in (0.0, -0.5):
            continue
        if not baseline._has_asian(
            line,
            r.get("close_asian_upper_water"),
            r.get("close_asian_lower_water"),
        ):
            continue
        try:
            q_h, q_d, q_a = baseline.compute_euro_prior(
                float(r["close_euro_h"]),
                float(r["close_euro_d"]),
                float(r["close_euro_a"]),
            )
        except (TypeError, ValueError, KeyError):
            continue

        p_upper = baseline.compute_asian_p_upper(
            float(r["close_asian_upper_water"]),
            float(r["close_asian_lower_water"]),
        )

        # Euro-implied p_h
        euro_p_h = q_h

        # Asian-implied p_h
        if line == -0.5:
            asian_p_h = p_upper
        else:  # line == 0.0
            r_ah = p_upper  # = p_h/(p_h+p_a)
            non_draw = 1.0 - q_d
            asian_p_h = non_draw * r_ah

        inconsistency = abs(euro_p_h - asian_p_h)
        label_idx = EURO_MAP.get(r["label"]["euro_result"], 0)

        bk_data[bk].append(
            {
                "inconsistency": inconsistency,
                "euro_p_h": euro_p_h,
                "asian_p_h": asian_p_h,
                "label": label_idx,
                "line": line,
            }
        )

    result = {}
    for bk, items in bk_data.items():
        if len(items) < 30:
            continue
        # Bucket by inconsistency
        sorted_items = sorted(items, key=lambda x: x["inconsistency"])
        n = len(sorted_items)
        tercile_size = n // 3

        bk_result = {}
        for label_text, indices in [
            ("low_inconsistency", range(0, tercile_size)),
            ("mid_inconsistency", range(tercile_size, 2 * tercile_size)),
            ("high_inconsistency", range(2 * tercile_size, n)),
        ]:
            bucket = [sorted_items[i] for i in indices]
            if not bucket:
                continue
            n_b = len(bucket)
            avg_inc = sum(m["inconsistency"] for m in bucket) / n_b

            # Logloss per sample: -log(p_correct)
            logloss_vals = []
            accuracy_vals = []
            for m_item in bucket:
                # Recompute the full 1X2 prob from euro prior for logloss
                try:
                    # Use the stored label to find the correct match row
                    # This is a rough estimate — full logloss needs all 3 probs
                    pass
                except Exception:
                    continue

            # Simple proxy: use euro_p_h vs actual home rate
            home_actuals = []
            home_implied = []
            for m_item in bucket:
                home_implied.append(m_item["euro_p_h"])
                home_actuals.append(1.0 if m_item["label"] == 0 else 0.0)

            home_rate = sum(home_actuals) / max(len(home_actuals), 1)
            avg_imp = sum(home_implied) / max(len(home_implied), 1)
            # logloss proxy: cross-entropy for home win
            eps = 1e-12
            ll_vals = []
            for imp, act in zip(home_implied, home_actuals):
                p = imp if act == 1.0 else (1.0 - imp)
                p = max(min(p, 1.0 - eps), eps)
                ll_vals.append(-math.log(p))
            avg_ll = sum(ll_vals) / max(len(ll_vals), 1)

            bk_result[label_text] = {
                "count": n_b,
                "avg_inconsistency": round(avg_inc, 6),
                "avg_implied_home": round(avg_imp, 4),
                "actual_home_rate": round(home_rate, 4),
                "avg_logloss_proxy": round(avg_ll, 4),
            }
        result[bk] = bk_result

    if not result:
        return {"status": "insufficient_data"}

    result["interpretation"] = (
        "If avg_logloss_proxy increases with inconsistency bucket, "
        "euro-asian disagreement is a useful signal for prediction "
        "difficulty.  avg_logloss_proxy is home-win-only cross-entropy."
    )
    return result


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bookmaker Difference Analysis (H1-H6)"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    rows = load_rows(args.data)
    print(f"  Total rows: {len(rows)}")
    bookmakers = sorted(set(r.get("bookmaker_id", "?") for r in rows))
    print(f"  Bookmakers: {bookmakers}")

    report = {
        "total_rows": len(rows),
        "bookmakers": bookmakers,
    }

    # H1
    print("Running H1: Margin differences ...")
    report["H1_margin"] = analyse_h1(rows)

    # H2
    print("Running H2: Favourite-Longshot Bias ...")
    report["H2_favourite_longshot_bias"] = analyse_h2(rows)

    # H3
    print("Running H3: Home/Favourite systemic bias ...")
    report["H3_home_bias"] = analyse_h3(rows)

    # H4
    print("Running H4: Lead-Lag (deferred) ...")
    report["H4_lead_lag"] = analyse_h4(rows)

    # H5
    print("Running H5: Inter-bookmaker disagreement ...")
    report["H5_disagreement_vs_uncertainty"] = analyse_h5(rows)

    # H6
    print("Running H6: Euro-Asian inconsistency vs error ...")
    report["H6_euro_asian_inconsistency"] = analyse_h6(rows)

    output = json.dumps(report, indent=2, default=str)
    print("\n=== H1 Summary (mean margin by bookmaker) ===")
    h1_bk = report["H1_margin"].get("_by_bookmaker", {})
    for bk, val in sorted(h1_bk.items(), key=lambda x: x[1]["mean_margin"]):
        print(f"  {bk}: margin={val['mean_margin']:.4f}  (n={val['total_count']})")

    print("\n=== H5 Summary (disagreement vs home-win rate) ===")
    h5 = report["H5_disagreement_vs_uncertainty"]
    if "total_matches" in h5:
        for k in ["low_disagreement", "mid_disagreement", "high_disagreement"]:
            b = h5.get(k, {})
            if b:
                print(
                    f"  {k}: std={b['avg_std_implied_prob']:.4f}  "
                    f"implied={b['avg_implied_home']:.3f}  "
                    f"actual={b['actual_home_win_rate']:.3f}  "
                    f"bias={b['bias']:.4f}"
                )
    else:
        print(f"  {h5.get('status', 'N/A')}")

    print("\n=== H6 Summary (inconsistency vs logloss) ===")
    h6 = report["H6_euro_asian_inconsistency"]
    for bk, bk_data in sorted(h6.items()):
        if bk.startswith("_") or not isinstance(bk_data, dict):
            continue
        print(f"  {bk}:")
        for k in ["low_inconsistency", "mid_inconsistency", "high_inconsistency"]:
            b = bk_data.get(k, {})
            if b:
                print(
                    f"    {k}: inc={b['avg_inconsistency']:.4f}  "
                    f"home_rate={b['actual_home_rate']:.3f}  "
                    f"logloss={b['avg_logloss_proxy']:.4f}"
                )

    if args.out_json:
        out_dir = os.path.dirname(args.out_json) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
