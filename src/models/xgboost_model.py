"""XGBoost with Optuna TPE search. Binary + multi-class variants."""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from ..utils import get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y])


def tune_xgboost(X_train, y_train, X_val, y_val, *, multi: bool,
                 n_trials: int, seed: int) -> XGBClassifier:
    n_classes = len(np.unique(y_train)) if multi else 2
    objective_name = "multi:softprob" if multi else "binary:logistic"
    sw_train = _sample_weights(y_train) if multi else None

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        kwargs = dict(
            objective=objective_name,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            eval_metric="mlogloss" if multi else "logloss",
            **params,
        )
        if multi:
            kwargs["num_class"] = n_classes
        model = XGBClassifier(**kwargs)
        fit_kwargs = {}
        if multi:
            fit_kwargs["sample_weight"] = sw_train
        else:
            pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
            model.set_params(scale_pos_weight=pos)
        model.fit(X_train, y_train, **fit_kwargs)
        pred = model.predict(X_val)
        return f1_score(y_val, pred, average="macro")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"Best XGBoost trial macro-F1={study.best_value:.4f}, params={study.best_params}")

    best_params = study.best_params
    kwargs = dict(
        objective=objective_name,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="mlogloss" if multi else "logloss",
        **best_params,
    )
    if multi:
        kwargs["num_class"] = n_classes
    best = XGBClassifier(**kwargs)
    fit_kwargs = {}
    if multi:
        fit_kwargs["sample_weight"] = sw_train
    else:
        pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        best.set_params(scale_pos_weight=pos)
    best.fit(X_train, y_train, **fit_kwargs)
    return best


def build_default_xgboost(cfg: dict, seed: int, multi: bool, n_classes: int) -> XGBClassifier:
    p = cfg["models"]["xgboost"]["default"]
    kwargs = dict(
        objective="multi:softprob" if multi else "binary:logistic",
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="mlogloss" if multi else "logloss",
    )
    if multi:
        kwargs["num_class"] = n_classes
    return XGBClassifier(**kwargs)
