"""Run the fixed P6/P6.1 training matrix without touching the test split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FIXED_VARIANTS = {
    "euro_default_draw_aux_0005": {"lambda_draw_bce": 0.005},
    "euro_default_draw_aux_001": {"lambda_draw_bce": 0.010},
    "euro_default_draw_aux_002": {"lambda_draw_bce": 0.020},
    "euro_default_final_draw_bce_005": {"lambda_final_draw_bce": 0.005},
    "euro_default_final_draw_bce_010": {"lambda_final_draw_bce": 0.010},
    "euro_default_draw_logit_coupled_weak": {
        "lambda_final_draw_bce": 0.005,
        "enable_draw_logit_coupling": True,
        "draw_coupling_scale": 0.25,
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed P6 draw auxiliary matrix")
    parser.add_argument("--mode", default="official_3seed", choices=["official_3seed"])
    parser.add_argument("--variants", required=True)
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--out-root", default="runs/p6_1_draw_aux")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run without --no-test")

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    out_root = Path(args.out_root)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        if variant not in FIXED_VARIANTS:
            raise SystemExit(f"Unknown fixed P6/P6.2 variant: {variant}")
        variant_config = FIXED_VARIANTS[variant]
        for seed in seeds:
            run_dir = out_root / f"p6_{variant}_seed{seed}"
            existing_outputs = [run_dir / "report.json", run_dir / "best_model.pth"]
            if args.skip_existing and any(path.exists() for path in existing_outputs):
                print(f"SKIP existing {run_dir}", flush=True)
                continue
            if any(path.exists() for path in existing_outputs):
                raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")

            cmd = [
                sys.executable,
                "tools/p6_train_residual_patch_itransformer.py",
                "--data",
                args.data,
                "--train-ids",
                args.train_ids,
                "--val-ids",
                args.val_ids,
                "--feature-groups",
                "euro",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--d-model",
                "96",
                "--n-heads",
                "4",
                "--n-layers",
                "2",
                "--d-ff",
                "192",
                "--dropout",
                "0.10",
                "--diff-loss-weight",
                "0.30",
                "--consistency-loss-weight",
                "0.10",
                "--delta-l2-weight",
                "0.01",
                "--scaling",
                "robust",
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--out-dir",
                str(run_dir),
            ]
            log_path = log_dir / f"p6_{variant}_seed{seed}.log"
            if variant_config.get("lambda_draw_bce"):
                cmd.extend(["--lambda-draw-bce", str(variant_config["lambda_draw_bce"])])
            if variant_config.get("lambda_final_draw_bce"):
                cmd.extend(["--lambda-final-draw-bce", str(variant_config["lambda_final_draw_bce"])])
            if variant_config.get("enable_draw_logit_coupling"):
                cmd.append("--enable-draw-logit-coupling")
            if variant_config.get("draw_coupling_scale"):
                cmd.extend(["--draw-coupling-scale", str(variant_config["draw_coupling_scale"])])
            print(f"RUN {variant} seed={seed} config={variant_config} -> {run_dir}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                print(f"FAILED {variant} seed={seed}; see {log_path}", flush=True)
                raise SystemExit(result.returncode)
            print(f"DONE {variant} seed={seed}; log={log_path}", flush=True)


if __name__ == "__main__":
    main()
