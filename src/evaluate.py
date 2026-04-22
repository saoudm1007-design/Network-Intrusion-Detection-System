"""Produce classification reports, confusion matrices, ROC, PR, feature-importance plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, roc_curve, auc,
)

from .utils import get_logger, save_json

log = get_logger(__name__)


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_confusion_matrix(y_true, y_pred, class_names, out_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    title = "Confusion Matrix"
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        title = "Confusion Matrix (normalized)"
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) * 0.8)))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_multiclass(y_true, y_proba, class_names, out_path):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        yt = (y_true == i).astype(int)
        if yt.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yt, y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (one-vs-rest)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_multiclass(y_true, y_proba, class_names, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        yt = (y_true == i).astype(int)
        if yt.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(yt, y_proba[:, i])
        ax.plot(rec, prec, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances, feature_names, out_path, top_k: int = 15):
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=df, x="importance", y="feature", palette="mako", ax=ax)
    ax.set_title(f"Top {top_k} Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_classifier(name: str, y_true, y_pred, class_names, figures_dir,
                        y_proba=None, feature_importances=None, feature_names=None):
    figures_dir = _ensure_dir(figures_dir)
    macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0, output_dict=True)
    log.info(f"=== {name} ===  macro-F1={macro:.4f}")
    log.info("\n" + classification_report(y_true, y_pred, target_names=class_names,
                                          zero_division=0))

    plot_confusion_matrix(y_true, y_pred, class_names,
                          figures_dir / f"cm_{name}.png", normalize=False)
    plot_confusion_matrix(y_true, y_pred, class_names,
                          figures_dir / f"cm_{name}_norm.png", normalize=True)

    if y_proba is not None and len(class_names) > 1 and y_proba.ndim == 2:
        plot_roc_multiclass(y_true, y_proba, class_names,
                            figures_dir / f"roc_{name}.png")
        plot_pr_multiclass(y_true, y_proba, class_names,
                           figures_dir / f"pr_{name}.png")

    if feature_importances is not None and feature_names is not None:
        plot_feature_importance(feature_importances, feature_names,
                                figures_dir / f"featimp_{name}.png")

    return {"macro_f1": float(macro), "report": report}


def save_metrics(all_metrics: dict, path: str | Path) -> None:
    save_json(all_metrics, path)
