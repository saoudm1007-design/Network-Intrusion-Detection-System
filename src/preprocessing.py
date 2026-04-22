"""Clean, label, split, scale, and SMOTE the CIC-IDS2017 data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .utils import get_logger, save_pickle

log = get_logger(__name__)

# Map all raw CIC-IDS2017 sub-labels (and the short-CSV labels) to 8 project classes.
LABEL_MAP = {
    # short CSV labels
    "Normal Traffic": "Benign",
    "DDoS": "DDoS",
    "DoS": "DoS",
    "Port Scanning": "PortScan",
    "Bots": "Bot",
    "Brute Force": "BruteForce",
    "Web Attacks": "WebAttack",
    # full dataset labels
    "BENIGN": "Benign",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "Web Attack \xef\xbf\xbd Brute Force": "WebAttack",
    "Web Attack \xef\xbf\xbd XSS": "WebAttack",
    "Web Attack \xef\xbf\xbd Sql Injection": "WebAttack",
}


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train_multi: np.ndarray
    y_val_multi: np.ndarray
    y_test_multi: np.ndarray
    y_train_bin: np.ndarray
    y_val_bin: np.ndarray
    y_test_bin: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    original_attack_ratio: float  # pre-SMOTE attack fraction, for IsolationForest


def _normalize_label(raw: str) -> str:
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    for prefix, target in (("Web Attack", "WebAttack"), ("DoS", "DoS")):
        if str(raw).startswith(prefix):
            return target
    return str(raw)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning frame")
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    log.info(f"Dropped {before - len(df):,} rows with NaN/inf")
    before = len(df)
    df.drop_duplicates(inplace=True)
    log.info(f"Dropped {before - len(df):,} duplicate rows")
    return df


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_class"] = df["Label"].map(_normalize_label)
    df["label_binary"] = (df["label_class"] != "Benign").astype(int)
    return df


def drop_constant_columns(X: pd.DataFrame) -> pd.DataFrame:
    nunique = X.nunique()
    drop_cols = nunique[nunique <= 1].index.tolist()
    if drop_cols:
        log.info(f"Dropping {len(drop_cols)} constant cols: {drop_cols}")
        X = X.drop(columns=drop_cols)
    return X


def drop_correlated(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop_cols = [c for c in upper.columns if (upper[c] > threshold).any()]
    if drop_cols:
        log.info(f"Dropping {len(drop_cols)} highly correlated cols (>{threshold})")
        X = X.drop(columns=drop_cols)
    return X


def build_splits(
    df: pd.DataFrame,
    cfg: dict,
    apply_smote: bool = True,
) -> tuple[Splits, StandardScaler, LabelEncoder]:
    seed = cfg["seed"]
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]

    df = add_label_columns(df)

    # Optional: subsample Benign to keep memory under control on full dataset
    max_benign = cfg["data"].get("max_benign_samples")
    if max_benign is not None:
        benign_mask = df["label_class"] == "Benign"
        if benign_mask.sum() > max_benign:
            log.info(f"Subsampling Benign from {benign_mask.sum():,} to {max_benign:,}")
            benign_keep = df[benign_mask].sample(n=max_benign, random_state=seed)
            df = pd.concat([benign_keep, df[~benign_mask]], ignore_index=True)

    y_multi_raw = df["label_class"].values
    y_bin = df["label_binary"].values
    X = df.drop(columns=["Label", "label_class", "label_binary"])

    X = drop_constant_columns(X)
    X = drop_correlated(X, cfg["data"]["correlation_threshold"])
    feature_names = X.columns.tolist()

    label_enc = LabelEncoder().fit(y_multi_raw)
    y_multi = label_enc.transform(y_multi_raw)
    class_names = label_enc.classes_.tolist()

    # train vs temp
    X_train, X_tmp, ym_train, ym_tmp, yb_train, yb_tmp = train_test_split(
        X.values, y_multi, y_bin,
        test_size=test_size + val_size,
        random_state=seed,
        stratify=y_multi,
    )
    # val vs test
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, ym_val, ym_test, yb_val, yb_test = train_test_split(
        X_tmp, ym_tmp, yb_tmp,
        test_size=1 - rel_val,
        random_state=seed,
        stratify=ym_tmp,
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    original_attack_ratio = float(np.mean(yb_train))

    if apply_smote and cfg["smote"]["enabled"]:
        counts = np.bincount(ym_train)
        benign_cls = int(label_enc.transform(["Benign"])[0])
        target = int(cfg["smote"].get("target_per_minority", 50000))
        smallest = int(counts.min())
        k = min(cfg["smote"]["k_neighbors"], max(1, smallest - 1))
        # Build dict: every non-benign class whose count < target is raised to target
        # (but not above benign count to keep balance sane). Classes already >= target are left alone.
        strategy = {}
        for cls, cnt in enumerate(counts):
            if cls == benign_cls:
                continue
            if cnt < target:
                strategy[int(cls)] = min(target, int(counts[benign_cls]))
        if strategy and smallest > 1:
            log.info(f"Applying SMOTE (k_neighbors={k}); target counts: {strategy}")
            sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
            X_train, ym_train = sm.fit_resample(X_train, ym_train)
            yb_train = (ym_train != benign_cls).astype(int)
        else:
            log.info("Skipping SMOTE — all minority classes already meet target or smallest class too rare")

    splits = Splits(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train_multi=ym_train, y_val_multi=ym_val, y_test_multi=ym_test,
        y_train_bin=yb_train, y_val_bin=yb_val, y_test_bin=yb_test,
        feature_names=feature_names, class_names=class_names,
        original_attack_ratio=original_attack_ratio,
    )
    log.info(f"Splits — train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")
    log.info(f"Classes: {class_names}")
    return splits, scaler, label_enc


def save_artifacts(scaler: StandardScaler, label_enc: LabelEncoder,
                   feature_names: list[str], cfg: dict) -> None:
    art = Path(cfg["paths"]["artifacts_dir"])
    save_pickle(scaler, art / "scaler.pkl")
    save_pickle(label_enc, art / "label_encoder.pkl")
    from .utils import save_json
    save_json({"features": feature_names}, art / "selected_features.json")
    log.info(f"Saved scaler, label encoder, and feature list to {art}")
