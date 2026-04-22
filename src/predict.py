"""Load the best-trained model and run inference on new CSV data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from .models.autoencoder import Autoencoder, ae_anomaly_score
from .models.cnn1d import CNN1D, cnn1d_predict, cnn1d_predict_proba
from .models.mlp import MLP, mlp_predict, mlp_predict_proba
from .utils import get_logger, load_json, load_pickle

log = get_logger(__name__)


def _load_mlp(path: Path) -> MLP:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(blob["in_dim"], blob["hidden_dims"], blob["n_classes"], blob["dropout"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def _load_autoencoder(path: Path) -> tuple[Autoencoder, float]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = Autoencoder(blob["in_dim"], blob["hidden_dims"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, float(blob["threshold"])


def predict_csv(cfg: dict, input_path: str, output_path: str,
                model_override: str | None = None) -> pd.DataFrame:
    art = Path(cfg["paths"]["artifacts_dir"])
    scaler = load_pickle(art / "scaler.pkl")
    label_enc = load_pickle(art / "label_encoder.pkl")
    feature_names = load_json(art / "selected_features.json")["features"]
    best = model_override or load_json(art / "best_model.json")["best_model"]
    log.info(f"Using model: {best}")

    df = pd.read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]
    has_label = "Label" in df.columns
    if has_label:
        keep_labels = df["Label"].values
        df = df.drop(columns=["Label"])

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required features: {missing[:5]}...")
    X = df[feature_names].values
    Xs = scaler.transform(X)

    # Load multi-class predictor
    if best == "rf_multi":
        model = load_pickle(art / "rf_multi.pkl")
        pred_multi = model.predict(Xs)
        proba_multi = model.predict_proba(Xs)
    elif best == "xgb_multi":
        import xgboost as xgb
        from scipy.special import softmax
        booster = xgb.Booster()
        booster.load_model(str(art / "xgb_multi.json"))
        dmat = xgb.DMatrix(Xs, feature_names=feature_names)
        raw_pred = booster.predict(dmat)
        proba_multi = raw_pred if raw_pred.ndim == 2 else softmax(raw_pred.reshape(len(Xs), -1), axis=1)
        pred_multi = proba_multi.argmax(axis=1)
    elif best == "mlp_multi":
        model = _load_mlp(art / "mlp_multi.pt")
        pred_multi = mlp_predict(model, Xs)
        proba_multi = mlp_predict_proba(model, Xs)
    elif best == "cnn1d_multi":
        blob = torch.load(art / "cnn1d_multi.pt", map_location="cpu", weights_only=False)
        model = CNN1D(blob["in_dim"], blob["n_classes"], blob["dropout"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        pred_multi = cnn1d_predict(model, Xs)
        proba_multi = cnn1d_predict_proba(model, Xs)
    else:
        raise ValueError(f"Unknown best model: {best}")

    pred_labels = label_enc.inverse_transform(pred_multi)
    pred_confidence = proba_multi.max(axis=1)
    pred_binary = (pred_labels != "Benign").astype(int)

    # Autoencoder anomaly score (secondary signal)
    ae_score = None
    ae_path = art / "autoencoder.pt"
    if ae_path.exists():
        ae_model, ae_threshold = _load_autoencoder(ae_path)
        ae_score = ae_anomaly_score(ae_model, Xs)

    out = pd.DataFrame({
        "predicted_class": pred_labels,
        "is_attack": pred_binary,
        "confidence": pred_confidence,
    })
    if ae_score is not None:
        out["anomaly_score"] = ae_score
    if has_label:
        out.insert(0, "true_label", keep_labels[: len(out)])

    out.to_csv(output_path, index=False)
    log.info(f"Wrote {len(out):,} predictions to {output_path}")
    return out
