"""
P2.3 OddsMind Dual-path Inference CLI — single-match and JSONL batch inference.

Input contract: JSON with match_id, kickoff_time, odds_timeline, cutoff_minutes,
optional league_id, bookmaker_id. Missing market fields allowed.

Output contract (P2.3 dual-path):
  - known league: OddsMind primary + market baseline + KNN explanation
  - unknown league: market baseline primary + KNN explanation + OddsMind experimental

Usage:
    python infer/predict_match.py --match match.json --config-dir runs/p2_0_deployment_candidate

    python infer/predict_match.py --batch matches.jsonl --config-dir runs/p2_0_deployment_candidate --out results.jsonl
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import (
    _event_to_features, _event_to_missing_mask, _event_has_euro,
    _event_has_asian, _event_has_over_under, compute_market_availability,
    FEATURE_KEYS, BOOKMAKER_MAP,
)
from dataset.odds_cutoff import filter_timeline_by_cutoff, assert_cutoff_integrity
from eval.odds_baselines import close_no_vig_euro, euro_probs_to_tensor
from infer.knn_retrieval import load_index, retrieve_similar, compute_explanation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model loading ─────────────────────────────────────────────────────

def load_model(ckpt_path, config_dir=None):
    """Load OddsMindModel from checkpoint with default config."""
    cfg = OddsMindConfig(
        hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
        asian_num_classes=5, dropout=0.1, pooling_mode="mean",
        feature_schema_version="v3", transformer_backend="odds_native",
    )
    T = 1.09

    if config_dir:
        ckpt_path = os.path.join(config_dir, "model_candidate.pth")
        cal_path = os.path.join(config_dir, "calibration.json")
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                T = json.load(f).get("temperature", 1.09)

    model = OddsMindModel(cfg).to(DEVICE)
    ckp = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model.load_state_dict(ckp, strict=False)
    model.eval()
    return model, T


# ── Feature building ──────────────────────────────────────────────────

def build_features(timeline, cutoff_minutes, max_seq_len=256):
    """Build (features, missing_mask) from an odds timeline."""
    filtered = filter_timeline_by_cutoff(timeline, cutoff_minutes)
    if filtered:
        assert_cutoff_integrity(filtered, cutoff_minutes, sample_id="inference")

    # Sort descending by minutes_before_kickoff (earliest first)
    sorted_tl = sorted(filtered, key=lambda e: e["minutes_before_kickoff"], reverse=True)
    if len(sorted_tl) > max_seq_len:
        sorted_tl = sorted_tl[-max_seq_len:]

    if len(sorted_tl) == 0:
        fdim = len(FEATURE_KEYS) + 3  # v3 has 13
        return torch.zeros(1, 0, fdim), torch.zeros(1, 0, fdim), torch.zeros(1, 0, dtype=torch.bool)

    feats = torch.tensor([_event_to_features(e, schema_version="v3") for e in sorted_tl],
                          dtype=torch.float32).unsqueeze(0)  # [1, T, 13]
    mmask = torch.tensor([_event_to_missing_mask(e) for e in sorted_tl],
                          dtype=torch.float32).unsqueeze(0)
    amask = torch.ones(1, len(sorted_tl), dtype=torch.bool)
    return feats, mmask, amask


def build_warnings(timeline, cutoff_minutes):
    """Generate warnings dict for inference output."""
    availability = compute_market_availability(timeline)
    warnings = []
    if not availability["has_euro"]: warnings.append("missing_euro")
    if not availability["has_asian"]: warnings.append("missing_asian")
    if not availability["has_over_under"]: warnings.append("missing_over_under")

    filtered = [e for e in timeline if e.get("minutes_before_kickoff", 0) >= cutoff_minutes]
    if len(filtered) == 0: warnings.append("insufficient_timeline")
    if len(filtered) < 3: warnings.append("sparse_timeline")

    return warnings


# ── Policy loading (P2.5C gate-based) ─────────────────────────────────

def load_policy(config_dir):
    """Load gate results and registry from config directory."""
    gate = {}
    registry = {"default_prediction_policy": {"primary_source": "market_baseline"}}
    if config_dir:
        gate_path = os.path.join(config_dir, "..", "p2_5c_registry", "per_league_gate_results.json")
        reg_path = os.path.join(config_dir, "..", "p2_5c_registry", "global_model_registry.json")
        # Also try relative to cwd
        for gp in [gate_path,
                    os.path.join("runs", "p2_5c_registry", "per_league_gate_results.json")]:
            if os.path.exists(gp):
                with open(gp, encoding="utf-8") as f:
                    gate = json.load(f)
                break
        for rp in [reg_path,
                    os.path.join("runs", "p2_5c_registry", "global_model_registry.json")]:
            if os.path.exists(rp):
                with open(rp, encoding="utf-8") as f:
                    registry = json.load(f)
                break
    return gate, registry


def get_route(league_id, gate_results):
    """Determine routing based on per-league gate results."""
    if not league_id or league_id not in gate_results:
        return "market_baseline_primary_path", False
    entry = gate_results[league_id]
    if entry.get("primary_gate_pass", False):
        return "global_model_primary_path", True
    return "market_baseline_primary_path", False


# ── Prediction (P2.5C gate-based) ──────────────────────────────────────

def predict_match(match: dict, model, T: float, gate_results: dict, registry: dict,
                  retrieval_index: list) -> dict:
    """Gate-based inference contract (P2.5C)."""
    match_id = match.get("match_id", "unknown")
    kickoff = match.get("kickoff_time", "")
    league_id = match.get("league_id", "")
    bookmaker_id = match.get("bookmaker_id", "")
    cutoff = match.get("cutoff_minutes", 0)
    timeline = match.get("odds_timeline", [])

    route, gate_pass = get_route(league_id, gate_results)
    K = 100

    # ── Build features ──
    feats, mmask, amask = build_features(timeline, cutoff)
    if feats.shape[1] == 0:
        return {
            "match_id": match_id, "cutoff_minutes": cutoff, "route": route,
            "error": "insufficient_timeline",
            "warnings": build_warnings(timeline, cutoff),
        }

    # ── Market baseline (always primary fallback) ──
    baseline_probs = torch.full((3,), 1.0/3.0)
    filtered = [e for e in timeline if e.get("minutes_before_kickoff", 0) >= cutoff]
    if filtered:
        try:
            baseline_probs = euro_probs_to_tensor(close_no_vig_euro(filtered))
        except (ValueError, KeyError):
            pass

    market_baseline = {
        "home": round(float(baseline_probs[0]), 4),
        "draw": round(float(baseline_probs[1]), 4),
        "away": round(float(baseline_probs[2]), 4),
    }

    # ── OddsMind model (always computed as neural view) ──
    with torch.no_grad():
        out = model(feats.to(DEVICE), attention_mask=amask.to(DEVICE), missing_mask=mmask.to(DEVICE))
    euro_logits = out["euro_logits"].cpu()[0]
    raw_probs = F.softmax(euro_logits, dim=-1)
    cal_probs = F.softmax(euro_logits / T, dim=-1)

    # ── KNN retrieval ──
    knn_explanation = {}
    try:
        if retrieval_index:
            similar = retrieve_similar(match, retrieval_index, K=K)
            knn_explanation = compute_explanation(similar)
    except Exception:
        knn_explanation = {"support_level": "insufficient", "historical_distribution": {},
                           "top_matches": [], "k": 0}

    # ── Build output (P2.5C gate-based) ──
    warnings = build_warnings(timeline, cutoff)

    model_id = registry.get("active_neural_model", {}).get("model_id", "global_odds_only_base_v1")

    if gate_pass:
        primary_source = "oddsmind_calibrated"
        primary_probs = {"home": round(float(cal_probs[0]), 4),
                          "draw": round(float(cal_probs[1]), 4),
                          "away": round(float(cal_probs[2]), 4)}
        pred_class = int(torch.argmax(cal_probs).item())
        oddsmind_primary_safe = True
        oddsmind_warnings = []
    else:
        primary_source = "market_baseline"
        primary_probs = market_baseline
        pred_class = int(torch.argmax(baseline_probs).item())
        oddsmind_primary_safe = False
        oddsmind_warnings = ["neural_model_view_only", "primary_gate_failed"]
        if not league_id or league_id not in gate_results:
            warnings.append("unknown_or_unvalidated_domain")
        if league_id in gate_results:
            warnings.append("global_model_not_primary_safe_for_domain")
        warnings.append("market_baseline_primary")

    class_names = ["home", "draw", "away"]

    return {
        "match_id": match_id,
        "kickoff_time": kickoff,
        "league_id": league_id,
        "cutoff_minutes": cutoff,
        "timeline_events": len(filtered),
        "route": route,
        "gate_pass": gate_pass,
        "primary_prediction": {
            "source": primary_source,
            "class": class_names[pred_class],
            "home": primary_probs["home"],
            "draw": primary_probs["draw"],
            "away": primary_probs["away"],
        },
        "market_baseline": market_baseline,
        "oddsmind_view": {
            "enabled": True,
            "model_id": model_id,
            "primary_safe": oddsmind_primary_safe,
            "raw_probs": {
                "home": round(float(raw_probs[0]), 4),
                "draw": round(float(raw_probs[1]), 4),
                "away": round(float(raw_probs[2]), 4),
            },
            "calibrated_probs": {
                "home": round(float(cal_probs[0]), 4),
                "draw": round(float(cal_probs[1]), 4),
                "away": round(float(cal_probs[2]), 4),
            },
            "temperature": T,
            "warnings": oddsmind_warnings,
        },
        "similar_matches": {
            "k": knn_explanation.get("k", 0),
            "support_level": knn_explanation.get("support_level", "insufficient"),
            "historical_distribution": knn_explanation.get("historical_distribution", {}),
            "top_matches": knn_explanation.get("top_matches", []),
        },
        "warnings": warnings,
    }


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OddsMind P2.3 Dual-path Prediction CLI")
    parser.add_argument("--match", type=str, default="", help="Single match JSON file")
    parser.add_argument("--batch", type=str, default="", help="JSONL batch input")
    parser.add_argument("--ckpt", type=str, default="runs/p2_0_deployment_candidate/model_candidate.pth")
    parser.add_argument("--config-dir", type=str, default="", help="Deployment config dir")
    parser.add_argument("--temperature", type=float, default=1.09)
    parser.add_argument("--out", type=str, default="", help="Output JSONL file")
    parser.add_argument("--human", action="store_true", help="Human-readable summary")
    parser.add_argument("--no-retrieval", action="store_true", help="Disable KNN retrieval")
    args = parser.parse_args()

    config_dir = args.config_dir
    ckpt_path = args.ckpt
    if config_dir:
        ckpt_path = os.path.join(config_dir, "model_candidate.pth")

    model, T = load_model(ckpt_path, config_dir=config_dir)
    if args.temperature != 1.09: T = args.temperature

    gate_results, registry = load_policy(config_dir)

    retrieval_index = None
    if not args.no_retrieval:
        idx_path = os.path.join(config_dir, "retrieval_index.json") if config_dir else ""
        if idx_path and os.path.exists(idx_path):
            retrieval_index = load_index(idx_path)
            print(f"Model loaded. T={T:.3f}. Gate entries: {len(gate_results)}. "
                  f"Retrieval index: {len(retrieval_index)} entries.", file=sys.stderr)
        else:
            print(f"Model loaded. T={T:.3f}. Gate entries: {len(gate_results)}. "
                  f"No retrieval index.", file=sys.stderr)
    else:
        print(f"Model loaded. T={T:.3f}. Retrieval disabled.", file=sys.stderr)

    if args.match:
        with open(args.match) as f: match = json.load(f)
        result = predict_match(match, model, T, gate_results, registry, retrieval_index)
        print(json.dumps(result, indent=2, default=str))
        if args.human:
            _print_human_v2(result)

    elif args.batch:
        results = []
        with open(args.batch) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                match = json.loads(line)
                result = predict_match(match, model, T, gate_results, registry, retrieval_index)
                results.append(result)

        out_path = args.out or args.batch.replace(".jsonl", "_predicted.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"Predicted {len(results)} matches -> {out_path}", file=sys.stderr)


def _print_human_v2(result):
    if "error" in result:
        print(f"\n  {result['match_id']}: ERROR — {result['error']}")
        return
    p = result["primary_prediction"]
    o = result["oddsmind_view"]
    m = result["market_baseline"]
    s = result["similar_matches"]
    print(f"\n  {result['match_id']} (@ cutoff={result['cutoff_minutes']}min, "
          f"route={result['route']})")
    print(f"    Primary  [{p['source']}]: {p['home']:.3f} {p['draw']:.3f} {p['away']:.3f}")
    if o["enabled"]:
        tag = "" if o["primary_safe"] else " [EXPERIMENTAL]"
        print(f"    OddsMind{tag}: {o['calibrated_probs']['home']:.3f} "
              f"{o['calibrated_probs']['draw']:.3f} {o['calibrated_probs']['away']:.3f}")
    print(f"    Market:   {m['home']:.3f} {m['draw']:.3f} {m['away']:.3f}")
    if s.get("support_level") != "insufficient":
        h = s["historical_distribution"]
        print(f"    KNN (K={s['k']}, {s['support_level']}): "
              f"H={h.get('home',0):.3f} D={h.get('draw',0):.3f} A={h.get('away',0):.3f}")
    if result["warnings"]:
        print(f"    Warnings: {result['warnings']}")


if __name__ == "__main__":
    main()
