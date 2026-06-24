"""
P1.12: Score Distribution Calibration — parameter search on VALIDATION set only.

Usage:
    # Grid search on val
    python eval/eval_p1_12_score_calibration.py \
        --data data/odds_real/v5_b365.jsonl \
        --model runs/p1_10r_a2_10ep/oddsmind_with_score.pth \
        --split-ids data/odds_real/splits_v5/val_match_ids.txt \
        --hidden-size 512 --num-layers 16 --num-heads 8 --device cuda \
        --out-dir data/reports/p1_12

    # Test evaluation with best config
    python eval/eval_p1_12_score_calibration.py \
        --data data/odds_real/v5_b365.jsonl \
        --model runs/p1_10r_a2_10ep/oddsmind_with_score.pth \
        --split-ids data/odds_real/splits_v5/test_match_ids.txt \
        --calibration-config data/reports/p1_12/best_calibration_config.json \
        --hidden-size 512 --num-layers 16 --num-heads 8 --device cuda \
        --out-dir data/reports/p1_12 --eval-only
"""
import argparse, json, os, sys, itertools, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from model.score_calibration import ScoreDistributionCalibrator
from model.score_utils import independent_poisson_score_grid, top_k_scorelines
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file


def gather_predictions(model, loader, device, score_loss_type="poisson"):
    """Collect lambda predictions and true scores from val/test set."""
    model.eval()
    all_lambdas = []
    all_true = []
    all_euro_probs = []

    with torch.no_grad():
        for batch in loader:
            f = batch['features'].to(device)
            m = batch['attention_mask'].to(device)
            sl = batch.get('score_labels')
            if sl is None:
                continue

            out = model(f, attention_mask=m, score_loss_type=score_loss_type)
            sp = out['score_preds'].cpu()  # [B, 2] — lambda predictions
            all_lambdas.append(sp)
            all_true.append(sl.cpu())
            all_euro_probs.append(torch.softmax(out['euro_logits'], dim=-1).cpu())

    lambdas = torch.cat(all_lambdas, dim=0)  # [N, 2]
    true_scores = torch.cat(all_true, dim=0)  # [N, 2]
    euro_probs = torch.cat(all_euro_probs, dim=0)  # [N, 3]
    return lambdas, true_scores, euro_probs


def compute_metrics(lambdas, true_scores, euro_probs, calibrator=None):
    """
    Compute score metrics with optional calibration.

    Returns dict of metrics.
    """
    N = lambdas.shape[0]
    metrics = {"num_samples": N}

    # Uncalibrated lambda predictions
    pred_home = lambdas[:, 0]  # [N]
    pred_away = lambdas[:, 1]  # [N]
    true_home = true_scores[:, 0]
    true_away = true_scores[:, 1]

    # Compute calibrated score grids if calibrator provided
    if calibrator:
        cal_home_win = torch.zeros(N)
        cal_draw = torch.zeros(N)
        cal_away_win = torch.zeros(N)
        cal_total_expected = torch.zeros(N)
        cal_diff_expected = torch.zeros(N)

        for n in range(N):
            bp = calibrator.calibrate(float(lambdas[n, 0]), float(lambdas[n, 1]))
            cal_home_win[n] = bp['home_win_prob']
            cal_draw[n] = bp['draw_prob']
            cal_away_win[n] = bp['away_win_prob']
            cal_total_expected[n] = bp['expected_total_goals']
            cal_diff_expected[n] = bp['expected_goal_diff']

        # Calibrated result from score
        cal_pred_result = torch.where(
            cal_home_win > cal_away_win, 0,
            torch.where(cal_home_win < cal_away_win, 2, 1)
        )
    else:
        cal_home_win = None
        cal_draw = None
        cal_away_win = None

    # Basic uncalibrated metrics
    pred_total = pred_home + pred_away
    true_total = true_home + true_away
    pred_diff = pred_home - pred_away
    true_diff = true_home - true_away

    metrics["uncalibrated"] = {
        "home_goal_mae": (pred_home - true_home).abs().mean().item(),
        "away_goal_mae": (pred_away - true_away).abs().mean().item(),
        "total_goal_mae": (pred_total - true_total).abs().mean().item(),
        "goal_diff_mae": (pred_diff - true_diff).abs().mean().item(),
        "pred_total_std": pred_total.std().item(),
        "true_total_std": true_total.std().item(),
        "total_std_gap": true_total.std().item() - pred_total.std().item(),
        "pred_diff_std": pred_diff.std().item(),
        "true_diff_std": true_diff.std().item(),
        "diff_std_gap": true_diff.std().item() - pred_diff.std().item(),
        "avg_pred_total": pred_total.mean().item(),
        "avg_true_total": true_total.mean().item(),
    }

    # Result from score (uncalibrated)
    pred_result = torch.where(pred_home > pred_away, 0, torch.where(pred_home < pred_away, 2, 1))
    true_result = torch.where(true_home > true_away, 0, torch.where(true_home < true_away, 2, 1))
    metrics["uncalibrated"]["result_from_score_accuracy"] = (pred_result == true_result).float().mean().item()

    # Exact score (uncalibrated) — round lambda
    metrics["uncalibrated"]["exact_score_accuracy"] = ((pred_home.round() == true_home) & (pred_away.round() == true_away)).float().mean().item()

    # Top-k hit rate (uncalibrated)
    for k in [1, 3, 5]:
        hits = 0
        for n in range(N):
            tk = top_k_scorelines(float(lambdas[n, 0]), float(lambdas[n, 1]), k=k)
            true_score_str = f"{int(true_home[n].item())}-{int(true_away[n].item())}"
            if any(s['score'] == true_score_str for s in tk):
                hits += 1
        metrics["uncalibrated"][f"top{k}_scoreline_hit_rate"] = hits / N

    # Per-bin MAE (uncalibrated)
    bins_mae = {}
    for lo, hi, label in [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 4, "3"), (4, 5, "4"), (5, 99, "5+")]:
        mask = (true_total >= lo) & (true_total < hi)
        if mask.sum() > 0:
            bins_mae[f"total_{label}_mae"] = (pred_total[mask] - true_total[mask]).abs().mean().item()
            bins_mae[f"total_{label}_count"] = int(mask.sum().item())
    metrics["uncalibrated"]["per_bin"] = bins_mae

    # Euro-vs-Score consistency (uncalibrated)
    from model.score_utils import consistency_check
    js_list = []
    agree_count = 0
    for n in range(min(N, 500)):  # sample for speed
        ep = {"home": float(euro_probs[n, 0]), "draw": float(euro_probs[n, 1]), "away": float(euro_probs[n, 2])}
        bp = independent_poisson_score_grid(float(lambdas[n, 0]), float(lambdas[n, 1]))
        cc = consistency_check(ep, bp)
        js_list.append(cc['js_distance'])
        if cc['winner_agreement']:
            agree_count += 1
    metrics["uncalibrated"]["js_divergence_mean"] = float(sum(js_list) / len(js_list)) if js_list else 0.0
    metrics["uncalibrated"]["winner_agreement_rate"] = agree_count / len(js_list) if js_list else 0.0

    # Calibrated metrics
    if calibrator and cal_home_win is not None:
        cal_result = torch.where(cal_home_win > cal_away_win, 0, torch.where(cal_home_win < cal_away_win, 2, 1))
        metrics["calibrated"] = {
            "result_from_score_accuracy": (cal_result == true_result).float().mean().item(),
            "avg_pred_total": cal_total_expected.mean().item(),
            "avg_true_total": true_total.mean().item(),
            "total_std_gap": true_total.std().item() - cal_total_expected.std().item(),
        }
        # Top-k with calibration
        for k in [1, 3, 5]:
            hits = 0
            for n in range(N):
                bp = calibrator.calibrate(float(lambdas[n, 0]), float(lambdas[n, 1]))
                # Re-rank from calibrated grid
                g = bp['score_matrix']
                G = g.shape[0]
                scorelines = []
                for i in range(G):
                    for j in range(G):
                        scorelines.append({"score": f"{i}-{j}", "prob": g[i, j].item()})
                scorelines.sort(key=lambda x: x['prob'], reverse=True)
                true_score_str = f"{int(true_home[n].item())}-{int(true_away[n].item())}"
                if any(s['score'] == true_score_str for s in scorelines[:k]):
                    hits += 1
            metrics["calibrated"][f"top{k}_scoreline_hit_rate"] = hits / N

        # JS divergence (calibrated)
        js_list_cal = []
        for n in range(min(N, 500)):
            ep = {"home": float(euro_probs[n, 0]), "draw": float(euro_probs[n, 1]), "away": float(euro_probs[n, 2])}
            bp = calibrator.calibrate(float(lambdas[n, 0]), float(lambdas[n, 1]))
            cc = consistency_check(ep, bp)
            js_list_cal.append(cc['js_distance'])
        metrics["calibrated"]["js_divergence_mean"] = float(sum(js_list_cal) / len(js_list_cal)) if js_list_cal else 0.0

        # Deltas
        metrics["delta"] = {}
        for key in metrics["calibrated"]:
            if key in metrics["uncalibrated"] and isinstance(metrics["calibrated"][key], (int, float)):
                metrics["delta"][key] = metrics["calibrated"][key] - metrics["uncalibrated"][key]

    return metrics


def grid_search(lambdas, true_scores, euro_probs):
    """Grid search calibration params on val set. Optimized for speed."""
    N = lambdas.shape[0]
    pred_home = lambdas[:, 0]
    pred_away = lambdas[:, 1]
    true_home = true_scores[:, 0]
    true_away = true_scores[:, 1]
    pred_total = pred_home + pred_away
    true_total = true_home + true_away
    true_result = torch.where(true_home > true_away, 0, torch.where(true_home < true_away, 2, 1))

    base_total_std = pred_total.std().item()
    true_total_std = true_total.std().item()
    base_total_std_gap = true_total_std - base_total_std

    # Uncalibrated result accuracy
    pred_result = torch.where(pred_home > pred_away, 0, torch.where(pred_home < pred_away, 2, 1))
    base_result = (pred_result == true_result).float().mean().item()

    total_temps = [1.0, 1.2, 1.5, 2.0]
    lambda_scales = [0.9, 1.0, 1.1]
    dc_rhos = [-0.1, 0.0, 0.1]

    best_config = None
    best_score = float('-inf')
    all_results = []

    for tt, ls, dc in itertools.product(total_temps, lambda_scales, dc_rhos):
        cal = ScoreDistributionCalibrator(
            total_temperature=tt, lambda_scale=ls, dc_rho=dc
        )

        # Fast: only compute result_from_score and total_std for each sample
        cal_total_expected = torch.zeros(N)
        cal_home_win = torch.zeros(N)
        cal_away_win = torch.zeros(N)

        for n in range(N):
            bp = cal.calibrate(float(lambdas[n, 0]), float(lambdas[n, 1]))
            cal_home_win[n] = bp['home_win_prob']
            cal_away_win[n] = bp['away_win_prob']
            cal_total_expected[n] = bp['expected_total_goals']

        cal_result = torch.where(cal_home_win > cal_away_win, 0, torch.where(cal_home_win < cal_away_win, 2, 1))
        cal_result_acc = (cal_result == true_result).float().mean().item()
        cal_total_std = cal_total_expected.std().item()
        cal_std_gap = true_total_std - cal_total_std

        # Score: reward lower std_gap, reward higher/better result accuracy
        std_gap_reduction = base_total_std_gap - cal_std_gap
        result_change = cal_result_acc - base_result
        combined = std_gap_reduction * 0.7 + result_change * 0.3

        all_results.append({
            "config": cal.to_dict(),
            "score": round(combined, 6),
            "std_gap_reduction": round(std_gap_reduction, 6),
            "result_change": round(result_change, 6),
            "uncal_std_gap": round(base_total_std_gap, 4),
            "cal_std_gap": round(cal_std_gap, 4),
            "uncal_result_acc": round(base_result, 4),
            "cal_result_acc": round(cal_result_acc, 4),
        })

        if combined > best_score:
            best_score = combined
            best_config = cal.to_dict()

    return best_config, all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split-ids", required=True)
    parser.add_argument("--calibration-config", default="", help="Path to pre-fitted config (eval mode)")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate with given config, no grid search")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default="data/reports/p1_12")
    parser.add_argument("--sample", type=int, default=0, help="Limit samples for fast grid search")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # Load model
    cfg = OddsMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads, asian_num_classes=3,
        score_head_version='v2'
    )
    model = OddsMindModel(cfg).to(device)
    ckp = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(ckp, strict=False)
    model.eval()

    # Load data
    ids = load_match_ids_from_file(args.split_ids)
    ds = OddsDataset(args.data, cutoffs=[0], cutoff_mode='exhaustive',
                     asian_label_mode='3class', allowed_match_ids=ids)
    if args.sample > 0 and len(ds) > args.sample:
        ds.samples = ds.samples[:args.sample]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=OddsCollator())
    print(f"Split: {len(ds)} samples")

    # Gather predictions
    lambdas, true_scores, euro_probs = gather_predictions(model, loader, device)
    print(f"Gathered {lambdas.shape[0]} predictions")

    if args.eval_only and args.calibration_config:
        # Test set evaluation with pre-fitted config
        calibrator = ScoreDistributionCalibrator.load(args.calibration_config)
        print(f"Loaded calibration config: {calibrator.to_dict()}")
        metrics = compute_metrics(lambdas, true_scores, euro_probs, calibrator=calibrator)
    elif args.calibration_config:
        # Val mode with pre-fitted config
        calibrator = ScoreDistributionCalibrator.load(args.calibration_config)
        metrics = compute_metrics(lambdas, true_scores, euro_probs, calibrator=calibrator)
    else:
        # Grid search on val
        print("Running grid search...")
        best_config, all_results = grid_search(lambdas, true_scores, euro_probs)

        # Save all results
        with open(os.path.join(args.out_dir, "calibration_search_val.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        # Save best config
        best_path = os.path.join(args.out_dir, "best_calibration_config.json")
        with open(best_path, "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"Best config saved to {best_path}")
        print(f"Best config: {json.dumps(best_config, indent=2)}")

        calibrator = ScoreDistributionCalibrator.from_dict(best_config)
        metrics = compute_metrics(lambdas, true_scores, euro_probs, calibrator=calibrator)

    # Print key comparison
    uc = metrics.get('uncalibrated', {})
    cal = metrics.get('calibrated', {})
    delta = metrics.get('delta', {})
    print(f"\n{'Metric':<35} {'Uncalibrated':>12} {'Calibrated':>12} {'Delta':>12}")
    print("-" * 71)
    for key in ['total_std_gap', 'diff_std_gap', 'result_from_score_accuracy', 'exact_score_accuracy',
                'top1_scoreline_hit_rate', 'top3_scoreline_hit_rate', 'top5_scoreline_hit_rate',
                'js_divergence_mean']:
        uv = uc.get(key, float('nan'))
        cv = cal.get(key, float('nan'))
        dv = delta.get(key, float('nan'))
        print(f"{key:<35} {uv:>12.4f} {cv:>12.4f} {dv:>+12.4f}")

    # Save full metrics
    out_name = "calibrated_test_metrics.json" if args.eval_only else "calibrated_val_metrics.json"
    with open(os.path.join(args.out_dir, out_name), "w") as f:
        json.dump(metrics, f, indent=2, default=str)


if __name__ == "__main__":
    main()
