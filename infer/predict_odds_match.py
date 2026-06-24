"""
OddsMind P1.11 Inference Script — full score output integration.

Usage:
    python infer/predict_odds_match.py \
        --model runs/p1_10r_a2_10ep/oddsmind_with_score.pth \
        --input data/odds_fixtures/sample_odds_matches.jsonl \
        --hidden-size 512 --num-layers 16 --num-heads 8 --device cuda

Output: JSON with euro_probs, asian_probs, score_prediction,
        score_derived_1x2, top_k_scorelines, consistency_check.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import EURO_MAP, ASIAN_MAP

EURO_REV = {v: k for k, v in EURO_MAP.items()}
ASIAN_REV_3 = {v: k for k, v in ASIAN_MAP.items()}
# P0.3 5-class
from dataset.odds_dataset import ASIAN_MAP_5CLASS
ASIAN_REV_5 = {v: k for k, v in ASIAN_MAP_5CLASS.items()}


def load_model(model_path: str, config: OddsMindConfig, device: str) -> OddsMindModel:
    model = OddsMindModel(config).to(device)
    ckp = torch.load(model_path, map_location=device)
    if isinstance(ckp, dict) and "model_state_dict" in ckp:
        state_dict = ckp["model_state_dict"]
    else:
        state_dict = ckp
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_match(input_path: str) -> dict:
    """Load a single match from JSONL (first line) or single JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Try as JSONL — take first non-empty line
    for line in text.split("\n"):
        line = line.strip()
        if line and line.startswith("{"):
            return json.loads(line)

    # Fallback: single JSON object
    if text.startswith("{"):
        return json.loads(text)

    raise ValueError(f"Cannot parse input: {input_path}")


def predict(model: OddsMindModel, match: dict, device: str,
           max_seq_len: int = 64, asian_label_mode: str = "3class",
           score_loss_type: str = "poisson", top_k: int = 5,
           calibration_config: str = "") -> dict:
    """Run inference on a single match dict with full P1.11 score outputs."""
    from dataset.odds_dataset import _event_to_features
    from model.score_utils import independent_poisson_score_grid, top_k_scorelines, consistency_check, disagreement_policy
    from dataset.odds_dataset import compute_market_availability

    timeline = match.get("odds_timeline", [])
    cutoff = match.get("cutoff_minutes", 0)

    # Filter and truncate
    filtered = [e for e in timeline if e["minutes_before_kickoff"] >= cutoff]
    if len(filtered) > max_seq_len:
        filtered = filtered[-max_seq_len:]

    # Build tensor [1, seq_len, feature_dim]
    schema_ver = model.config.feature_schema_version
    feats = [_event_to_features(e, schema_version=schema_ver) for e in filtered]
    features = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    attention_mask = torch.ones(1, len(filtered), dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(features, attention_mask=attention_mask,
                    score_loss_type=score_loss_type)

    euro_probs = torch.softmax(out["euro_logits"], dim=-1)[0].cpu()
    asian_probs = torch.softmax(out["asian_logits"], dim=-1)[0].cpu()
    score_preds = out.get("score_preds")
    if score_preds is not None:
        score_preds = score_preds[0].cpu()

    euro_pred_idx = int(torch.argmax(euro_probs).item())
    asian_pred_idx = int(torch.argmax(asian_probs).item())

    # ── Euro probs dict ──
    euro_dict = {
        "home": round(euro_probs[0].item(), 4),
        "draw": round(euro_probs[1].item(), 4),
        "away": round(euro_probs[2].item(), 4),
    }

    # ── Asian probs dict ──
    if asian_label_mode == "5class":
        asian_rev = ASIAN_REV_5
        asian_probs_dict = {
            "upper_full_win": round(asian_probs[0].item(), 4),
            "upper_half_win": round(asian_probs[1].item(), 4),
            "push": round(asian_probs[2].item(), 4),
            "upper_half_loss": round(asian_probs[3].item(), 4),
            "upper_full_loss": round(asian_probs[4].item(), 4),
        }
    else:
        asian_rev = ASIAN_REV_3
        asian_probs_dict = {
            "upper": round(asian_probs[0].item(), 4),
            "push": round(asian_probs[1].item(), 4),
            "lower": round(asian_probs[2].item(), 4),
        }

    result = {
        "euro_probs": euro_dict,
        "asian_probs": asian_probs_dict,
        "euro_prediction": EURO_REV[euro_pred_idx],
        "asian_prediction": asian_rev[asian_pred_idx],
        "market_availability": compute_market_availability(timeline),
    }

    # ── P1.11: Full score output ──
    if score_preds is not None:
        lambda_home = float(score_preds[0].item())
        lambda_away = float(score_preds[1].item())

        result["score_prediction"] = {
            "home_goals": round(lambda_home, 2),
            "away_goals": round(lambda_away, 2),
        }

        # Score-derived 1X2 probabilities
        bp = independent_poisson_score_grid(lambda_home, lambda_away)
        result["score_derived_1x2"] = {
            "home_win_prob": bp["home_win_prob"],
            "draw_prob": bp["draw_prob"],
            "away_win_prob": bp["away_win_prob"],
            "expected_total_goals": bp["expected_total_goals"],
            "expected_goal_diff": bp["expected_goal_diff"],
        }

        # Top-k scorelines
        result["top_k_scorelines"] = top_k_scorelines(
            lambda_home, lambda_away, k=top_k
        )

        # Euro Head vs Score-derived consistency check
        cc = consistency_check(euro_dict, bp)
        result["consistency_check"] = cc

        # P1.12: Disagreement policy
        result["disagreement_policy"] = disagreement_policy(
            euro_dict, bp, cc["js_distance"], cc["winner_agreement"]
        )

        # P1.12: Calibrated outputs (if calibration config provided)
        if calibration_config:
            from model.score_calibration import ScoreDistributionCalibrator
            calibrator = ScoreDistributionCalibrator.load(calibration_config)
            cal = calibrator.calibrate(lambda_home, lambda_away)

            result["score_calibrated_1x2"] = {
                "home_win_prob": cal["home_win_prob"],
                "draw_prob": cal["draw_prob"],
                "away_win_prob": cal["away_win_prob"],
                "expected_total_goals": cal["expected_total_goals"],
                "expected_goal_diff": cal["expected_goal_diff"],
            }
            result["calibrated_top_k_scorelines"] = top_k_scorelines(
                lambda_home, lambda_away, k=top_k
            )  # Using calibrated grid would need re-ranking
            # Re-rank from calibrated grid
            g = cal['score_matrix']
            G = g.shape[0]
            cal_scorelines = []
            for i in range(G):
                for j in range(G):
                    cal_scorelines.append({"score": f"{i}-{j}", "home": i, "away": j,
                                           "prob": round(g[i, j].item(), 6)})
            cal_scorelines.sort(key=lambda x: x['prob'], reverse=True)
            result["calibrated_top_k_scorelines"] = cal_scorelines[:top_k]

    return result


def main():
    parser = argparse.ArgumentParser(description="OddsMind Single-Match Inference")
    parser.add_argument("--model", type=str, default="out_odds/oddsmind_smoke.pth",
                        help="Path to trained checkpoint")
    parser.add_argument("--input", type=str, default="data/odds_fixtures/sample_odds_matches.jsonl",
                        help="Path to JSON/JSONL match file")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--untrained", action="store_true",
                        help="Use a randomly initialized model (scaffold test)")
    # P0.3 asian label mode
    parser.add_argument("--asian-label-mode", type=str, default="3class",
                        choices=["3class", "5class"],
                        help="Asian handicap label granularity")
    parser.add_argument("--transformer-backend", type=str, default="odds_native",
                        choices=["odds_native", "minimind"])
    # P1.11 score output
    parser.add_argument("--score-loss-type", type=str, default="poisson",
                        choices=["mse", "poisson"],
                        help="Score pred activation: poisson=exp, mse=softplus")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top scorelines to output")
    # P1.12 calibration
    parser.add_argument("--calibration-config", type=str, default="",
                        help="Path to P1.12 calibration config JSON")
    args = parser.parse_args()

    asian_num_classes = 5 if args.asian_label_mode == "5class" else 3
    config = OddsMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        asian_num_classes=asian_num_classes,
        transformer_backend=args.transformer_backend,
    )

    device = args.device
    if args.untrained:
        print("[scaffold mode] Using untrained random model", file=sys.stderr)
        model = OddsMindModel(config).to(device)
        model.eval()
    else:
        print(f"Loading model from {args.model}", file=sys.stderr)
        # P1.11: auto-detect score head version from checkpoint
        ckp = torch.load(args.model, map_location='cpu', weights_only=True)
        if isinstance(ckp, dict) and "model_state_dict" in ckp:
            state_dict = ckp["model_state_dict"]
        else:
            state_dict = ckp
        has_old_score = any('score_head.head.' in k for k in state_dict.keys())
        has_new_score = any('score_head.trunk.' in k for k in state_dict.keys())
        if has_old_score and not has_new_score:
            config.score_head_version = "v1"
        elif has_new_score:
            config.score_head_version = "v2"
        print(f"Detected score_head_version={config.score_head_version}", file=sys.stderr)

        model = OddsMindModel(config).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    match = load_match(args.input)
    result = predict(model, match, device, max_seq_len=args.max_seq_len,
                     asian_label_mode=args.asian_label_mode,
                     score_loss_type=args.score_loss_type,
                     top_k=args.top_k,
                     calibration_config=args.calibration_config)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
