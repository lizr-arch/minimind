"""
OddsMind Model Evaluation Script (P0.4)

Loads a trained (or untrained) model, runs evaluation on a split,
and outputs a JSON summary with per-cutoff breakdown.

Usage:
    python eval/eval_odds_model.py \
        --data data/odds_fixtures/sample_odds_matches_5class.jsonl \
        --model runs/oddsmind_p0_3_5class_smoke/oddsmind_smoke.pth \
        --split-match-ids data/odds_fixtures/splits_p0_4/test_match_ids.txt \
        --cutoffs 90,60,30 --cutoff-mode exhaustive --asian-label-mode 5class \
        --hidden-size 64 --num-layers 2 --num-heads 4 --device cpu
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from eval.odds_metrics import (
    accuracy_from_logits,
    logloss_from_logits,
    brier_from_logits,
    class_counts,
    prediction_counts,
    ece_from_logits,
)


def evaluate_model(
    model: OddsMindModel,
    loader: DataLoader,
    device: str,
    asian_num_classes: int,
    score_loss_type: str = "poisson",
) -> dict:
    """
    Run evaluation over a DataLoader and return metrics dict.
    """
    model.eval()

    all_euro_logits = []
    all_asian_logits = []
    all_euro_labels = []
    all_asian_labels = []
    all_score_preds = []
    all_score_labels = []
    all_league_ids = []
    all_match_ids = []
    all_market_euro = []
    all_market_asian = []
    all_market_ou = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            euro_labels = batch["euro_labels"].to(device)
            asian_labels = batch["asian_labels"].to(device)
            score_labels = batch.get("score_labels")

            out = model(features, attention_mask=attention_mask,
                        score_loss_type=score_loss_type)

            all_euro_logits.append(out["euro_logits"].cpu())
            all_asian_logits.append(out["asian_logits"].cpu())
            all_euro_labels.append(euro_labels.cpu())
            all_asian_labels.append(asian_labels.cpu())
            if "score_preds" in out and score_labels is not None:
                all_score_preds.append(out["score_preds"].cpu())
                all_score_labels.append(score_labels.cpu())

            # P1.16: collect metadata for grouped metrics
            league_ids = batch.get("league_ids", [""] * features.shape[0])
            all_league_ids.extend(league_ids)

        # P1.16: compute per-sample market availability from batch
            # Check if the first (oldest) event in each sample has euro/asian/ou
            for i in range(features.shape[0]):
                fv = features[i][attention_mask[i]]  # [valid_T, F]
                if fv.shape[0] > 0:
                    # v2/v3 features: idx 10=has_euro, 11=has_asian, 12=has_ou
                    has_eur = bool((fv[0, 10] > 0.5).item()) if fv.shape[1] > 10 else bool((fv[0, 1] > 1.0).item())
                    has_asn = bool((fv[0, 11] > 0.5).item()) if fv.shape[1] > 10 else True
                    has_ou  = bool((fv[0, 12] > 0.5).item()) if fv.shape[1] > 10 else False
                else:
                    has_eur, has_asn, has_ou = False, False, False
                all_market_euro.append(has_eur)
                all_market_asian.append(has_asn)
                all_market_ou.append(has_ou)

    euro_logits = torch.cat(all_euro_logits, dim=0)
    asian_logits = torch.cat(all_asian_logits, dim=0)
    euro_labels = torch.cat(all_euro_labels, dim=0)
    asian_labels = torch.cat(all_asian_labels, dim=0)

    result = {
        "num_samples": int(euro_logits.shape[0]),
        "euro": {
            "accuracy": accuracy_from_logits(euro_logits, euro_labels),
            "logloss": logloss_from_logits(euro_logits, euro_labels),
            "brier": brier_from_logits(euro_logits, euro_labels, 3),
            "label_counts": class_counts(euro_labels, 3),
            "prediction_counts": prediction_counts(euro_logits, 3),
            "ece": ece_from_logits(euro_logits, euro_labels, 3),
        },
        "asian": {
            "accuracy": accuracy_from_logits(asian_logits, asian_labels),
            "logloss": logloss_from_logits(asian_logits, asian_labels),
            "brier": brier_from_logits(asian_logits, asian_labels, asian_num_classes),
            "label_counts": class_counts(asian_labels, asian_num_classes),
            "prediction_counts": prediction_counts(asian_logits, asian_num_classes),
            "ece": ece_from_logits(asian_logits, asian_labels, asian_num_classes),
        },
    }

    # P1.16: Per-league metrics
    if all_league_ids and any(lid for lid in all_league_ids):
        result["by_league"] = _compute_by_league(
            euro_logits, asian_logits, euro_labels, asian_labels,
            all_league_ids, asian_num_classes,
        )

    # P1.16: Per-market-availability metrics
    if all_market_euro:
        result["by_market"] = _compute_by_market(
            euro_logits, asian_logits, euro_labels, asian_labels,
            all_market_euro, all_market_asian, all_market_ou, asian_num_classes,
        )

    if all_score_preds:
        sp = torch.cat(all_score_preds, dim=0)
        sl = torch.cat(all_score_labels, dim=0)
        home_mae = (sp[:,0] - sl[:,0]).abs().mean().item()
        away_mae = (sp[:,1] - sl[:,1]).abs().mean().item()
        total_mae = ((sp.sum(-1) - sl.sum(-1)).abs().mean().item())
        diff_mae = ((sp[:,0]-sp[:,1]) - (sl[:,0]-sl[:,1])).abs().mean().item()
        exact = ((sp.round() == sl).all(-1)).float().mean().item()
        collapse = (sp.std(0) < 0.05).any().item()

        # P1.10: result-from-score accuracy
        # home win: h>a, draw: h==a, away win: h<a
        pred_result = torch.where(
            sp[:,0] > sp[:,1], 0,
            torch.where(sp[:,0] < sp[:,1], 2, 1)
        )
        true_result = torch.where(
            sl[:,0] > sl[:,1], 0,
            torch.where(sl[:,0] < sl[:,1], 2, 1)
        )
        result_from_score = (pred_result == true_result).float().mean().item()

        result["score"] = {
            "home_goal_mae": home_mae,
            "away_goal_mae": away_mae,
            "total_goal_mae": total_mae,
            "goal_diff_mae": diff_mae,
            "exact_score_accuracy": exact,
            "result_from_score_accuracy": result_from_score,
            "avg_pred_home_goals": sp[:,0].mean().item(),
            "avg_true_home_goals": sl[:,0].mean().item(),
            "avg_pred_away_goals": sp[:,1].mean().item(),
            "avg_true_away_goals": sl[:,1].mean().item(),
            "score_prediction_collapse_flag": collapse,
        }

        # P1.11: Score calibration report
        pred_total = sp.sum(-1)
        true_total = sl.sum(-1)
        pred_diff = sp[:,0] - sp[:,1]
        true_diff = sl[:,0] - sl[:,1]

        # Std comparison
        calib = {
            "pred_total_std": round(pred_total.std().item(), 4),
            "true_total_std": round(true_total.std().item(), 4),
            "pred_diff_std": round(pred_diff.std().item(), 4),
            "true_diff_std": round(true_diff.std().item(), 4),
        }

        # Per-bin total-goal MAE (bins: 0,1,2,3,4,5+)
        bins = [0, 1, 2, 3, 4, 5]
        bin_mae = {}
        for i, lo in enumerate(bins):
            if i < len(bins) - 1:
                hi = bins[i + 1]
                mask = (true_total >= lo) & (true_total < hi)
                label = f"{lo}-{hi - 1}"
            else:
                mask = true_total >= lo
                label = f"{lo}+"
            if mask.sum() > 0:
                mae_bin = (pred_total[mask] - true_total[mask]).abs().mean().item()
                bin_mae[f"total_{label}_mae"] = round(mae_bin, 4)
                bin_mae[f"total_{label}_count"] = int(mask.sum().item())
        calib["per_bin_total_goal_mae"] = bin_mae

        result["score_calibration"] = calib

    return result


# ── P1.16: Grouped metric helpers ──────────────────────────────────────

def _compute_by_league(
    euro_logits: torch.Tensor,
    asian_logits: torch.Tensor,
    euro_labels: torch.Tensor,
    asian_labels: torch.Tensor,
    league_ids: list,
    asian_num_classes: int,
) -> dict:
    """Compute per-league metrics."""
    by_league = {}
    for league in sorted(set(league_ids)):
        if not league:
            continue
        idx = torch.tensor([i for i, lid in enumerate(league_ids) if lid == league])
        if idx.numel() == 0:
            continue
        eu_l = euro_logits[idx]
        as_l = asian_logits[idx]
        eu_lbl = euro_labels[idx]
        as_lbl = asian_labels[idx]
        by_league[league] = {
            "count": int(idx.numel()),
            "euro_accuracy": accuracy_from_logits(eu_l, eu_lbl),
            "euro_logloss": logloss_from_logits(eu_l, eu_lbl),
            "euro_brier": brier_from_logits(eu_l, eu_lbl, 3),
            "asian_accuracy": accuracy_from_logits(as_l, as_lbl),
            "asian_logloss": logloss_from_logits(as_l, as_lbl),
        }
    return by_league


def _compute_by_market(
    euro_logits: torch.Tensor,
    asian_logits: torch.Tensor,
    euro_labels: torch.Tensor,
    asian_labels: torch.Tensor,
    has_euro: list,
    has_asian: list,
    has_ou: list,
    asian_num_classes: int,
) -> dict:
    """Compute metrics segmented by market availability."""
    he = torch.tensor(has_euro, dtype=torch.bool)
    ha = torch.tensor(has_asian, dtype=torch.bool)
    ho = torch.tensor(has_ou, dtype=torch.bool)

    def _segment(mask, label):
        if mask.sum() == 0:
            return {"count": 0}
        return {
            "count": int(mask.sum().item()),
            "euro_accuracy": accuracy_from_logits(euro_logits[mask], euro_labels[mask]),
            "euro_logloss": logloss_from_logits(euro_logits[mask], euro_labels[mask]),
            "asian_accuracy": accuracy_from_logits(asian_logits[mask], asian_labels[mask]),
            "asian_logloss": logloss_from_logits(asian_logits[mask], asian_labels[mask]),
        }

    return {
        "euro_only": _segment(he & ~ha & ~ho, "euro only"),
        "euro_asian": _segment(he & ha & ~ho, "euro + asian"),
        "all_three": _segment(he & ha & ho, "all three markets"),
        "has_euro": _segment(he, "has euro"),
        "has_asian": _segment(ha, "has asian"),
        "has_over_under": _segment(ho, "has over/under"),
    }


def evaluate_by_cutoff(
    model: OddsMindModel,
    dataset: OddsDataset,
    collator: OddsCollator,
    device: str,
    asian_num_classes: int,
    cutoffs: list,
    score_loss_type: str = "poisson",
) -> dict:
    """Evaluate per cutoff by building separate datasets for each cutoff."""
    by_cutoff = {}
    for cutoff in cutoffs:
        ds_cutoff = OddsDataset(
            dataset._jsonl_path,
            max_seq_len=dataset.max_seq_len,
            cutoff_minutes=cutoff,
            cutoff_mode="none",
            asian_label_mode=dataset.asian_label_mode,
            allowed_match_ids=dataset._allowed_ids,
            feature_schema_version=dataset.feature_schema_version,
        )
        loader = DataLoader(ds_cutoff, batch_size=8, shuffle=False, collate_fn=collator)
        if len(ds_cutoff) == 0:
            by_cutoff[str(int(cutoff))] = {"num_samples": 0}
            continue
        result = evaluate_model(model, loader, device, asian_num_classes, score_loss_type)
        by_cutoff[str(int(cutoff))] = {
            "num_samples": result["num_samples"],
            "euro_accuracy": result["euro"]["accuracy"],
            "euro_logloss": result["euro"]["logloss"],
            "euro_ece": result["euro"]["ece"]["ece"],
            "asian_accuracy": result["asian"]["accuracy"],
            "asian_logloss": result["asian"]["logloss"],
            "asian_ece": result["asian"]["ece"]["ece"],
        }
    return by_cutoff


def main():
    parser = argparse.ArgumentParser(description="OddsMind Model Evaluation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--split-match-ids", type=str, default="")
    parser.add_argument("--cutoffs", type=str, default="")
    parser.add_argument("--cutoff-mode", type=str, default="exhaustive")
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--transformer-backend", type=str, default="odds_native",
                        choices=["odds_native", "minimind"])
    parser.add_argument("--score-loss-type", type=str, default="poisson",
                        choices=["mse", "poisson"],
                        help="Score pred activation: poisson=exp, mse=softplus")
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    device = args.device
    asian_num_classes = 5 if args.asian_label_mode == "5class" else 3

    # Load model
    config = OddsMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        asian_num_classes=asian_num_classes,
        transformer_backend=args.transformer_backend,
    )

    if args.untrained or not args.model:
        model = OddsMindModel(config).to(device)
        model.eval()
    else:
        ckp = torch.load(args.model, map_location=device)
        # Handle transfer checkpoint format (wrapped in dict with model_state_dict)
        if isinstance(ckp, dict) and "model_state_dict" in ckp:
            state_dict = ckp["model_state_dict"]
        else:
            state_dict = ckp

        # P1.10/P1.13A: auto-detect score head version from checkpoint keys
        has_old_score = any('score_head.head.' in k for k in state_dict.keys())
        has_v2_score = any('score_head.home_head.' in k for k in state_dict.keys())
        has_grid_score = any('score_head._grid_idx' in k for k in state_dict.keys())
        if has_old_score and not has_v2_score and not has_grid_score:
            config.score_head_version = "v1"
        elif has_grid_score:
            config.score_head_version = "grid"
        elif has_v2_score:
            config.score_head_version = "v2"
        # else: keep default (v2)

        model = OddsMindModel(config).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    # Load match IDs if split specified
    allowed_ids = None
    if args.split_match_ids:
        allowed_ids = load_match_ids_from_file(args.split_match_ids)

    cutoffs = None
    if args.cutoffs:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    # Dataset for the split
    ds = OddsDataset(
        args.data,
        max_seq_len=args.max_seq_len,
        cutoffs=cutoffs,
        cutoff_mode=args.cutoff_mode,
        asian_label_mode=args.asian_label_mode,
        allowed_match_ids=allowed_ids,
        feature_schema_version="v2",  # P1.15B
    )

    collator = OddsCollator()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    result = evaluate_model(model, loader, device, asian_num_classes, args.score_loss_type)

    # By-cutoff breakdown
    if cutoffs and args.cutoff_mode == "exhaustive":
        result["by_cutoff"] = evaluate_by_cutoff(
            model, ds, collator, device, asian_num_classes, cutoffs, args.score_loss_type
        )

    # Count unique matches
    unique_matches = len(allowed_ids) if allowed_ids else ds.num_raw_matches
    result["num_matches"] = unique_matches

    print(json.dumps(result, indent=2, default=str))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
