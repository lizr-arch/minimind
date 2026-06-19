"""
OddsMind P0.1 Inference Script

Reads a single JSON match file or the first line of a JSONL file,
runs the model, and outputs predicted probabilities.

Usage:
    python infer/predict_odds_match.py \
        --model out_odds/oddsmind_smoke.pth \
        --input data/odds_fixtures/sample_odds_matches.jsonl \
        --device cpu

Output: JSON object with euro_probs, asian_probs, predictions.
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
ASIAN_REV = {v: k for k, v in ASIAN_MAP.items()}


def load_model(model_path: str, config: OddsMindConfig, device: str) -> OddsMindModel:
    model = OddsMindModel(config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
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


def predict(model: OddsMindModel, match: dict, device: str, max_seq_len: int = 64) -> dict:
    """Run inference on a single match dict."""
    from dataset.odds_dataset import _event_to_features

    timeline = match.get("odds_timeline", [])
    cutoff = match.get("cutoff_minutes", 0)

    # Filter and truncate
    filtered = [e for e in timeline if e["minutes_before_kickoff"] >= cutoff]
    if len(filtered) > max_seq_len:
        filtered = filtered[-max_seq_len:]

    # Build tensor [1, seq_len, feature_dim]
    feats = [_event_to_features(e) for e in filtered]
    features = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    attention_mask = torch.ones(1, len(filtered), dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(features, attention_mask=attention_mask)

    euro_probs = torch.softmax(out["euro_logits"], dim=-1)[0].cpu()
    asian_probs = torch.softmax(out["asian_logits"], dim=-1)[0].cpu()

    euro_pred_idx = int(torch.argmax(euro_probs).item())
    asian_pred_idx = int(torch.argmax(asian_probs).item())

    return {
        "euro_probs": {
            "home": round(euro_probs[0].item(), 4),
            "draw": round(euro_probs[1].item(), 4),
            "away": round(euro_probs[2].item(), 4),
        },
        "asian_probs": {
            "upper": round(asian_probs[0].item(), 4),
            "push": round(asian_probs[1].item(), 4),
            "lower": round(asian_probs[2].item(), 4),
        },
        "euro_prediction": EURO_REV[euro_pred_idx],
        "asian_prediction": ASIAN_REV[asian_pred_idx],
    }


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
    args = parser.parse_args()

    config = OddsMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
    )

    device = args.device
    if args.untrained:
        print("[scaffold mode] Using untrained random model", file=sys.stderr)
        model = OddsMindModel(config).to(device)
        model.eval()
    else:
        print(f"Loading model from {args.model}", file=sys.stderr)
        model = load_model(args.model, config, device)

    match = load_match(args.input)
    result = predict(model, match, device, max_seq_len=args.max_seq_len)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
