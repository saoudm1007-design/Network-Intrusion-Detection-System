import numpy as np
from sklearn.ensemble import IsolationForest


def build_iforest(cfg: dict, seed: int, contamination: float) -> IsolationForest:
    return IsolationForest(
        n_estimators=cfg["models"]["isolation_forest"]["n_estimators"],
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )


def iforest_predict_binary(model: IsolationForest, X) -> np.ndarray:
    """Map IsolationForest predictions (-1=anomaly, 1=inlier) to 1=attack, 0=benign."""
    raw = model.predict(X)
    return (raw == -1).astype(int)
