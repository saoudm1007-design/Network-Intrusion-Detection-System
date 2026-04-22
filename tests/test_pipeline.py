"""Smoke test: exercise the preprocessing + RF training path on the short CSV."""
from __future__ import annotations

from sklearn.metrics import f1_score

from src.data_loader import load_dataset
from src.models.random_forest import build_rf
from src.preprocessing import build_splits, clean_frame
from src.utils import load_config, set_seed


def test_rf_short_csv():
    cfg = load_config("config.yaml")
    assert not cfg["data"]["use_full_dataset"], "Smoke test expects short CSV"
    set_seed(cfg["seed"])

    df = load_dataset(cfg)
    df = clean_frame(df)
    splits, _, _ = build_splits(df, cfg)

    rf = build_rf(cfg, cfg["seed"])
    rf.fit(splits.X_train, splits.y_train_multi)
    pred = rf.predict(splits.X_test)
    macro = f1_score(splits.y_test_multi, pred, average="macro")
    assert macro >= 0.60, f"Macro-F1 too low on short CSV: {macro:.3f}"
