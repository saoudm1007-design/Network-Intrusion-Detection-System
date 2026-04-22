"""CLI entrypoint: train | evaluate | predict."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.predict import predict_csv
from src.train import run_training
from src.utils import get_logger, load_config, load_json

log = get_logger(__name__)


def cmd_train(args) -> None:
    cfg = load_config(args.config)
    run_training(cfg, quick=args.quick)


def cmd_evaluate(args) -> None:
    cfg = load_config(args.config)
    metrics_path = Path(cfg["paths"]["artifacts_dir"]) / "metrics.json"
    if not metrics_path.exists():
        log.error(f"{metrics_path} not found — run `python main.py train` first.")
        return
    metrics = load_json(metrics_path)
    log.info("=== Test-split macro-F1 by model ===")
    for name, m in sorted(metrics.items(), key=lambda kv: -kv[1]["macro_f1"]):
        log.info(f"  {name:25s}  {m['macro_f1']:.4f}")


def cmd_predict(args) -> None:
    cfg = load_config(args.config)
    predict_csv(cfg, args.input, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(prog="nids")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train all models end-to-end")
    p_train.add_argument("--quick", action="store_true",
                         help="Fewer Optuna trials + MLP/AE epochs (fast sanity check)")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Print metrics from the last training run")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Run inference on a CSV")
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--output", required=True)
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
