from sklearn.ensemble import RandomForestClassifier


def build_rf(cfg: dict, seed: int) -> RandomForestClassifier:
    p = cfg["models"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        class_weight=p["class_weight"],
        n_jobs=p["n_jobs"],
        random_state=seed,
    )
