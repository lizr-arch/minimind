"""P3.3 market-specific MLP and late-fusion diagnostic.

This script does not read test ids. It splits the official train rows into a
base-training subset and a calibration subset, trains market-specific MLPs on
base-training rows, fits a tiny stacking head on calibration predictions, and
evaluates everything on the official validation split.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_metrics import accuracy_from_probs, brier_from_probs, ece_from_probs, logloss_from_probs
from model.odds_patch_itransformer_v2 import extract_cross_market_feature
from tools.p3_save_raw_pooled_mlp_baseline import load_rows_for_ids, load_split_ids


EURO_MAP = {"home": 0, "draw": 1, "away": 2}
MARKETS = ("euro", "asian", "ou")
MARKET_FEATURES = {
    "euro": [
        "euro_h",
        "euro_d",
        "euro_a",
        "imp_h",
        "imp_d",
        "imp_a",
        "euro_margin",
        "time_log",
        "has_euro",
    ],
    "asian": [
        "asian_line",
        "neg_asian_line",
        "upper_water",
        "lower_water",
        "asian_upper_implied",
        "asian_lower_implied",
        "asian_implied_gap",
        "asian_margin",
        "asian_home_strength_proxy",
        "has_asian",
    ],
    "ou": [
        "over_under_line",
        "over_water",
        "under_water",
        "over_implied",
        "under_implied",
        "ou_implied_gap",
        "ou_margin",
        "ou_goal_proxy",
        "has_ou",
    ],
}
EPS = 1e-8


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing = [Path(out_dir) / "market_fusion_report.json", Path(out_dir) / "market_fusion_report.md"]
    if not allow_overwrite and any(path.exists() for path in existing):
        raise FileExistsError(f"Refusing to overwrite existing market fusion outputs in {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def split_train_calibration_rows(
    rows: list[dict],
    seed: int,
    calibration_fraction: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    keyed = sorted(rows, key=lambda row: str(row.get("match_id", "")))
    rng = random.Random(seed)
    rng.shuffle(keyed)
    n_calib = max(1, int(round(len(keyed) * calibration_fraction)))
    n_calib = min(n_calib, len(keyed) - 1)
    return keyed[n_calib:], keyed[:n_calib]


def _timeline(row: dict) -> list[dict]:
    return row.get("raw_timeline", []) or row.get("odds_timeline", [])


def _clean_market_value(event: dict, feature_name: str, market: str) -> float:
    if market == "asian" and extract_cross_market_feature(event, "has_asian") <= 0:
        return 0.0
    if market == "ou" and extract_cross_market_feature(event, "has_ou") <= 0:
        return 0.0
    return extract_cross_market_feature(event, feature_name)


def _event_market_features(event: dict, market: str) -> list[float]:
    if market == "euro":
        return [_clean_market_value(event, name, market) for name in MARKET_FEATURES["euro"]]
    if market == "asian":
        has_asian = _clean_market_value(event, "has_asian", market)
        if has_asian <= 0:
            return [0.0 for _ in MARKET_FEATURES["asian"]]
        line = _clean_market_value(event, "asian_line", market)
        upper_imp = _clean_market_value(event, "asian_upper_implied", market)
        lower_imp = _clean_market_value(event, "asian_lower_implied", market)
        return [
            line,
            -line,
            _clean_market_value(event, "upper_water", market),
            _clean_market_value(event, "lower_water", market),
            upper_imp,
            lower_imp,
            upper_imp - lower_imp,
            _clean_market_value(event, "asian_margin", market),
            -line + (upper_imp - lower_imp),
            has_asian,
        ]
    if market == "ou":
        has_ou = _clean_market_value(event, "has_ou", market)
        if has_ou <= 0:
            return [0.0 for _ in MARKET_FEATURES["ou"]]
        line = _clean_market_value(event, "over_under_line", market)
        over_imp = _clean_market_value(event, "over_implied", market)
        under_imp = _clean_market_value(event, "under_implied", market)
        return [
            line,
            _clean_market_value(event, "over_water", market),
            _clean_market_value(event, "under_water", market),
            over_imp,
            under_imp,
            over_imp - under_imp,
            _clean_market_value(event, "ou_margin", market),
            line + (over_imp - under_imp),
            has_ou,
        ]
    raise ValueError(f"Unknown market: {market}")


def build_market_features(row: dict, market: str) -> torch.Tensor:
    events = _timeline(row)
    width = len(MARKET_FEATURES[market])
    if not events:
        return torch.zeros(width, dtype=torch.float32)
    values = []
    for event in events:
        feats = _event_market_features(event, market)
        if all(math.isfinite(value) for value in feats):
            values.append(feats)
    if not values:
        return torch.zeros(width, dtype=torch.float32)
    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor.mean(dim=0)


def build_dataset(rows: list[dict], market: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    X, y, match_ids = [], [], []
    for idx, row in enumerate(rows):
        label = row.get("label", {}).get("euro_result")
        if label not in EURO_MAP:
            continue
        X.append(build_market_features(row, market))
        y.append(EURO_MAP[label])
        match_ids.append(str(row.get("match_id", idx)))
    if not X:
        raise ValueError(f"No labelled rows for market {market}")
    return torch.stack(X), torch.tensor(y, dtype=torch.long), match_ids


def fit_standardizer(x_train: torch.Tensor) -> dict:
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False)
    std = torch.where(std > EPS, std, torch.ones_like(std))
    return {"mean": mean, "std": std}


def apply_standardizer(x: torch.Tensor, scaler: dict) -> torch.Tensor:
    return (x - scaler["mean"]) / scaler["std"]


class MarketMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict_probs(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        logits = model(x[start : start + batch_size].to(device))
        chunks.append(F.softmax(logits, dim=-1).clamp(1e-9, 1 - 1e-9).cpu())
    return torch.cat(chunks)


def evaluate_probs(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    return {
        "accuracy": accuracy_from_probs(probs, labels),
        "logloss": logloss_from_probs(probs, labels),
        "brier": brier_from_probs(probs, labels, 3),
        "ece": ece_from_probs(probs, labels, 3)["ece"],
        "argmax_draw_count": int((probs.argmax(dim=-1) == 1).sum().item()),
        "draw_recall": _draw_recall(probs, labels),
    }


def _draw_recall(probs: torch.Tensor, labels: torch.Tensor) -> float:
    draw = labels == 1
    if not torch.any(draw):
        return 0.0
    preds = probs.argmax(dim=-1)
    return float(((preds == 1) & draw).sum().item() / draw.sum().item())


def train_market_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[MarketMLP, dict, dict]:
    set_deterministic_seed(seed)
    scaler = fit_standardizer(x_train)
    x_train_s = apply_standardizer(x_train, scaler)
    x_eval_s = apply_standardizer(x_eval, scaler)
    model = MarketMLP(input_dim=x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_logloss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train_s))
        for start in range(0, len(x_train_s), batch_size):
            idx = perm[start : start + batch_size]
            logits = model(x_train_s[idx].to(device))
            loss = F.cross_entropy(logits, y_train[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        eval_probs = predict_probs(model, x_eval_s, batch_size * 2, device)
        eval_loss = logloss_from_probs(eval_probs, y_eval)
        if eval_loss < best_logloss:
            best_logloss = eval_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, {"best_epoch": best_epoch, "best_calibration_logloss": best_logloss}


def make_fusion_inputs(probs_by_market: dict[str, torch.Tensor], market_order: list[str]) -> torch.Tensor:
    return torch.cat([probs_by_market[name] for name in market_order], dim=1)


class FusionHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_fusion_head(
    x_calib: torch.Tensor,
    y_calib: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    device: torch.device,
    epochs: int,
    lr: float,
) -> tuple[FusionHead, dict]:
    set_deterministic_seed(seed)
    model = FusionHead(input_dim=x_calib.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_state = None
    best_calib_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        logits = model(x_calib.to(device))
        loss = F.cross_entropy(logits, y_calib.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probs = F.softmax(model(x_calib.to(device)), dim=-1).clamp(1e-9, 1 - 1e-9).cpu()
        calib_loss = logloss_from_probs(probs, y_calib)
        if calib_loss < best_calib_loss:
            best_calib_loss = calib_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_probs = predict_probs(model, x_val, len(x_val), device)
    return model, {"best_epoch": best_epoch, "best_calibration_logloss": best_calib_loss, "val_metrics": evaluate_probs(val_probs, y_val)}


def average_blend(probs_by_market: dict[str, torch.Tensor], market_order: list[str]) -> torch.Tensor:
    stacked = torch.stack([probs_by_market[name] for name in market_order], dim=0)
    return stacked.mean(dim=0).clamp(1e-9, 1 - 1e-9)


def run_one_seed(args, seed: int, train_rows: list[dict], val_rows: list[dict], device: torch.device) -> dict:
    base_rows, calib_rows = split_train_calibration_rows(train_rows, seed=seed, calibration_fraction=args.calibration_fraction)
    calib_probs_by_market = {}
    val_probs_by_market = {}
    market_rows = []

    for market in MARKETS:
        x_base, y_base, _ = build_dataset(base_rows, market)
        x_calib, y_calib, _ = build_dataset(calib_rows, market)
        x_val, y_val, _ = build_dataset(val_rows, market)
        model, scaler, train_info = train_market_model(
            x_base,
            y_base,
            x_calib,
            y_calib,
            seed=seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        x_calib_s = apply_standardizer(x_calib, scaler)
        x_val_s = apply_standardizer(x_val, scaler)
        calib_probs = predict_probs(model, x_calib_s, args.batch_size * 2, device)
        val_probs = predict_probs(model, x_val_s, args.batch_size * 2, device)
        calib_probs_by_market[market] = calib_probs
        val_probs_by_market[market] = val_probs
        market_rows.append(
            {
                "variant": market,
                "seed": seed,
                "metrics": evaluate_probs(val_probs, y_val),
                "train_info": train_info,
                "feature_names": MARKET_FEATURES[market],
            }
        )

    market_order = list(MARKETS)
    x_fusion_calib = make_fusion_inputs(calib_probs_by_market, market_order)
    x_fusion_val = make_fusion_inputs(val_probs_by_market, market_order)
    _, fusion_info = train_fusion_head(
        x_fusion_calib,
        y_calib,
        x_fusion_val,
        y_val,
        seed=seed,
        device=device,
        epochs=args.fusion_epochs,
        lr=args.fusion_lr,
    )
    avg_probs = average_blend(val_probs_by_market, market_order)
    return {
        "seed": seed,
        "base_train_rows": len(base_rows),
        "calibration_rows": len(calib_rows),
        "val_rows": len(val_rows),
        "market_results": market_rows,
        "average_blend": {"variant": "average_blend", "seed": seed, "metrics": evaluate_probs(avg_probs, y_val)},
        "stacking_fusion": {"variant": "stacking_fusion", "seed": seed, **fusion_info},
    }


def summarize_runs(runs: list[dict]) -> dict:
    by_variant: dict[str, list[dict]] = {}
    for run in runs:
        for row in run["market_results"]:
            by_variant.setdefault(row["variant"], []).append(row["metrics"])
        by_variant.setdefault("average_blend", []).append(run["average_blend"]["metrics"])
        by_variant.setdefault("stacking_fusion", []).append(run["stacking_fusion"]["val_metrics"])

    summary_rows = []
    for variant, metrics_rows in sorted(by_variant.items()):
        losses = [row["logloss"] for row in metrics_rows]
        briers = [row["brier"] for row in metrics_rows]
        eces = [row["ece"] for row in metrics_rows]
        accs = [row["accuracy"] for row in metrics_rows]
        draws = [row["draw_recall"] for row in metrics_rows]
        summary_rows.append(
            {
                "variant": variant,
                "mean_val_logloss": float(np.mean(losses)),
                "std_val_logloss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                "best_val_logloss": float(np.min(losses)),
                "mean_brier": float(np.mean(briers)),
                "mean_ECE": float(np.mean(eces)),
                "mean_val_acc": float(np.mean(accs)),
                "mean_draw_recall": float(np.mean(draws)),
            }
        )
    best = min(summary_rows, key=lambda row: row["mean_val_logloss"]) if summary_rows else None
    euro = next((row for row in summary_rows if row["variant"] == "euro"), None)
    fusion = next((row for row in summary_rows if row["variant"] == "stacking_fusion"), None)
    verdict = []
    if best:
        if best["variant"] == "euro":
            verdict.append("EURO_ONLY_STILL_BEST")
        elif best["variant"] in {"stacking_fusion", "average_blend"}:
            verdict.append("MARKET_FUSION_IMPROVES_VAL_LOGLOSS")
        else:
            verdict.append("NON_EURO_MARKET_HAS_STANDALONE_SIGNAL")
    if euro and fusion and fusion["mean_val_logloss"] >= euro["mean_val_logloss"] - 0.0001:
        verdict.append("FUSION_NO_GAIN_OVER_EURO")
    return {
        "variants": summary_rows,
        "best_variant": best["variant"] if best else None,
        "verdict": verdict,
    }


def build_report_payload(runs: list[dict], summary: dict, inputs: dict, verdict: list[str]) -> dict:
    next_actions = []
    if "FUSION_NO_GAIN_OVER_EURO" in verdict:
        next_actions.append("Do not merge late fusion yet; inspect Asian/OU feature definitions and market coverage.")
    if "MARKET_FUSION_IMPROVES_VAL_LOGLOSS" in verdict:
        next_actions.append("Promote market-specific fusion to a P3.4 controlled experiment with out-of-fold stacking.")
    if "NON_EURO_MARKET_HAS_STANDALONE_SIGNAL" in verdict:
        next_actions.append("Keep the strongest non-Euro branch as an auxiliary signal, not a direct raw concatenation.")
    next_actions.append("For production combining, learn fusion weights from train-only out-of-fold predictions, not from official val.")
    return {
        "inputs": inputs,
        "runs": runs,
        "summary": summary,
        "verdict": verdict,
        "next_actions": next_actions[:5],
    }


def _fmt(value) -> str:
    return f"{float(value):.4f}"


def write_report(out_dir: str, payload: dict) -> None:
    out = Path(out_dir)
    json_path = out / "market_fusion_report.json"
    md_path = out / "market_fusion_report.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = [
        "# P3.3 Market Fusion Diagnostic",
        "",
        "## Summary",
        "",
        "| Variant | mean_val_logloss | std | best | mean_Brier | mean_ECE | mean_val_acc | draw_recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"].get("variants", []):
        lines.append(
            f"| {row['variant']} | {_fmt(row['mean_val_logloss'])} | {_fmt(row['std_val_logloss'])} | "
            f"{_fmt(row['best_val_logloss'])} | {_fmt(row['mean_brier'])} | {_fmt(row['mean_ECE'])} | "
            f"{_fmt(row['mean_val_acc'])} | {_fmt(row['mean_draw_recall'])} |"
        )
    lines.extend(["", "## Verdict"])
    lines.extend(f"- `{item}`" for item in payload.get("verdict", []))
    lines.extend(["", "## Next Actions"])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload.get("next_actions", []), start=1))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="P3.3 market-specific fusion diagnostic")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--out-dir", default="runs/p3_3_market_fusion")
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fusion-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fusion-lr", type=float, default=0.03)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    prepare_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
    started = time.time()
    train_ids = load_split_ids(args.train_ids)
    val_ids = load_split_ids(args.val_ids)
    train_rows = load_rows_for_ids(args.data, train_ids)
    val_rows = load_rows_for_ids(args.data, val_ids)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    runs = []
    for seed in seeds:
        print(f"Running seed {seed} on {device}...")
        runs.append(run_one_seed(args, seed, train_rows, val_rows, device))

    summary = summarize_runs(runs)
    verdict = list(summary.get("verdict", []))
    payload = build_report_payload(
        runs=runs,
        summary=summary,
        inputs={
            "data": args.data,
            "train_ids": args.train_ids,
            "val_ids": args.val_ids,
            "test_ids_used": False,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "seeds": seeds,
            "calibration_fraction": args.calibration_fraction,
            "runtime_seconds": time.time() - started,
        },
        verdict=verdict,
    )
    write_report(args.out_dir, payload)
    print(f"Report json path: {Path(args.out_dir) / 'market_fusion_report.json'}")
    print(f"Report markdown path: {Path(args.out_dir) / 'market_fusion_report.md'}")
    print(f"Best variant: {summary.get('best_variant')}")
    print(f"Verdict: {', '.join(verdict)}")


if __name__ == "__main__":
    main()
