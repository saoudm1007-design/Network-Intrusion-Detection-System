"""Orchestrate training of all models and save artifacts + figures + metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data_loader import load_dataset
from .evaluate import evaluate_classifier, save_metrics
from .models.autoencoder import ae_predict_binary, train_autoencoder
from .models.cnn1d import cnn1d_predict, cnn1d_predict_proba, train_cnn1d
from .models.isolation_forest import build_iforest, iforest_predict_binary
from .models.mlp import mlp_predict, mlp_predict_proba, train_mlp
from .models.random_forest import build_rf
from .models.xgboost_model import build_default_xgboost, tune_xgboost
from .preprocessing import build_splits, clean_frame, save_artifacts
from .utils import get_logger, save_pickle, set_seed

log = get_logger(__name__)


def run_training(cfg: dict, quick: bool = False) -> dict:
    set_seed(cfg["seed"])
    art = Path(cfg["paths"]["artifacts_dir"])
    fig = Path(cfg["paths"]["figures_dir"])
    art.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg)
    df = clean_frame(df)
    splits, scaler, label_enc = build_splits(df, cfg)
    save_artifacts(scaler, label_enc, splits.feature_names, cfg)

    class_names = splits.class_names
    n_classes = len(class_names)
    benign_idx = class_names.index("Benign")
    binary_names = ["Benign", "Attack"]

    metrics: dict[str, dict] = {}

    # ---------- Random Forest (multi + binary) ----------
    log.info("Training Random Forest (multi-class)...")
    rf_multi = build_rf(cfg, cfg["seed"])
    rf_multi.fit(splits.X_train, splits.y_train_multi)
    save_pickle(rf_multi, art / "rf_multi.pkl")
    y_pred = rf_multi.predict(splits.X_test)
    y_proba = rf_multi.predict_proba(splits.X_test)
    metrics["rf_multi"] = evaluate_classifier(
        "rf_multi", splits.y_test_multi, y_pred, class_names, fig,
        y_proba=y_proba,
        feature_importances=rf_multi.feature_importances_,
        feature_names=splits.feature_names,
    )

    log.info("Training Random Forest (binary)...")
    rf_bin = build_rf(cfg, cfg["seed"])
    rf_bin.fit(splits.X_train, splits.y_train_bin)
    save_pickle(rf_bin, art / "rf_binary.pkl")
    y_pred = rf_bin.predict(splits.X_test)
    y_proba = rf_bin.predict_proba(splits.X_test)
    metrics["rf_binary"] = evaluate_classifier(
        "rf_binary", splits.y_test_bin, y_pred, binary_names, fig, y_proba=y_proba,
    )

    # ---------- XGBoost (multi + binary, tuned with Optuna) ----------
    n_trials = 5 if quick else cfg["models"]["xgboost"]["optuna_trials"]
    log.info(f"Tuning XGBoost (multi-class) with {n_trials} trials...")
    xgb_multi = tune_xgboost(splits.X_train, splits.y_train_multi,
                             splits.X_val, splits.y_val_multi,
                             multi=True, n_trials=n_trials, seed=cfg["seed"])
    xgb_multi.save_model(str(art / "xgb_multi.json"))
    y_pred = xgb_multi.predict(splits.X_test)
    y_proba = xgb_multi.predict_proba(splits.X_test)
    metrics["xgb_multi"] = evaluate_classifier(
        "xgb_multi", splits.y_test_multi, y_pred, class_names, fig,
        y_proba=y_proba,
        feature_importances=xgb_multi.feature_importances_,
        feature_names=splits.feature_names,
    )

    log.info(f"Tuning XGBoost (binary) with {n_trials} trials...")
    xgb_bin = tune_xgboost(splits.X_train, splits.y_train_bin,
                           splits.X_val, splits.y_val_bin,
                           multi=False, n_trials=n_trials, seed=cfg["seed"])
    xgb_bin.save_model(str(art / "xgb_binary.json"))
    y_pred = xgb_bin.predict(splits.X_test)
    y_proba = xgb_bin.predict_proba(splits.X_test)
    metrics["xgb_binary"] = evaluate_classifier(
        "xgb_binary", splits.y_test_bin, y_pred, binary_names, fig, y_proba=y_proba,
    )

    # ---------- MLP (multi-class) ----------
    if quick:
        cfg["models"]["mlp"]["epochs"] = min(cfg["models"]["mlp"]["epochs"], 10)
    log.info("Training MLP (multi-class)...")
    mlp = train_mlp(splits.X_train, splits.y_train_multi,
                    splits.X_val, splits.y_val_multi,
                    cfg=cfg, seed=cfg["seed"], n_classes=n_classes)
    torch.save({"state_dict": mlp.state_dict(),
                "in_dim": splits.X_train.shape[1],
                "hidden_dims": cfg["models"]["mlp"]["hidden_dims"],
                "n_classes": n_classes,
                "dropout": cfg["models"]["mlp"]["dropout"]},
               art / "mlp_multi.pt")
    y_pred = mlp_predict(mlp, splits.X_test)
    y_proba = mlp_predict_proba(mlp, splits.X_test)
    metrics["mlp_multi"] = evaluate_classifier(
        "mlp_multi", splits.y_test_multi, y_pred, class_names, fig, y_proba=y_proba,
    )

    # ---------- CNN-1D (multi-class) ----------
    if quick:
        cfg["models"]["cnn1d"]["epochs"] = min(cfg["models"]["cnn1d"]["epochs"], 10)
    log.info("Training CNN-1D (multi-class)...")
    cnn = train_cnn1d(splits.X_train, splits.y_train_multi,
                      splits.X_val, splits.y_val_multi,
                      cfg=cfg, seed=cfg["seed"], n_classes=n_classes)
    torch.save({"state_dict": cnn.state_dict(),
                "in_dim": splits.X_train.shape[1],
                "n_classes": n_classes,
                "dropout": cfg["models"]["cnn1d"]["dropout"]},
               art / "cnn1d_multi.pt")
    y_pred = cnn1d_predict(cnn, splits.X_test)
    y_proba = cnn1d_predict_proba(cnn, splits.X_test)
    metrics["cnn1d_multi"] = evaluate_classifier(
        "cnn1d_multi", splits.y_test_multi, y_pred, class_names, fig, y_proba=y_proba,
    )

    # ---------- Autoencoder (benign-only, binary scoring) ----------
    if quick:
        cfg["models"]["autoencoder"]["epochs"] = min(cfg["models"]["autoencoder"]["epochs"], 10)
    benign_mask = splits.y_train_bin == 0
    log.info(f"Training Autoencoder on {benign_mask.sum():,} benign flows...")
    ae, threshold = train_autoencoder(splits.X_train[benign_mask], cfg, cfg["seed"])
    torch.save({"state_dict": ae.state_dict(),
                "in_dim": splits.X_train.shape[1],
                "hidden_dims": cfg["models"]["autoencoder"]["hidden_dims"],
                "threshold": threshold},
               art / "autoencoder.pt")
    y_pred = ae_predict_binary(ae, splits.X_test, threshold)
    metrics["autoencoder_binary"] = evaluate_classifier(
        "autoencoder_binary", splits.y_test_bin, y_pred, binary_names, fig,
    )

    # ---------- Isolation Forest (unsupervised binary) ----------
    attack_ratio = min(0.5, max(0.01, splits.original_attack_ratio))
    log.info(f"Training IsolationForest (contamination={attack_ratio:.4f})...")
    iso = build_iforest(cfg, cfg["seed"], attack_ratio)
    iso.fit(splits.X_train)
    save_pickle(iso, art / "iforest.pkl")
    y_pred = iforest_predict_binary(iso, splits.X_test)
    metrics["iforest_binary"] = evaluate_classifier(
        "iforest_binary", splits.y_test_bin, y_pred, binary_names, fig,
    )

    # ---------- Pick best multi-class model as the "final" for predict/dashboard ----------
    candidates = {k: v["macro_f1"] for k, v in metrics.items() if k.endswith("_multi")}
    best_name = max(candidates, key=candidates.get)
    log.info(f"Best multi-class model: {best_name}  (macro-F1={candidates[best_name]:.4f})")
    from .utils import save_json
    save_json({"best_model": best_name, "macro_f1": candidates[best_name]},
              art / "best_model.json")

    save_metrics(metrics, art / "metrics.json")
    return metrics
