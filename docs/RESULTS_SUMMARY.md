# CyberShield NIDS — Results Summary

## Final Model Leaderboard

| Rank | Model | Type | Macro-F1 | Precision | Recall |
|---|---|---|---|---|---|
| #1 | **XGBoost (Multi-class)** | Ensemble | **0.9845** | 0.973 | 0.998 |
| #2 | Random Forest (Multi-class) | Ensemble | 0.9775 | 0.966 | 0.991 |
| #3 | XGBoost (Binary) | Ensemble | 0.9991 | 0.999 | 0.999 |
| #4 | Random Forest (Binary) | Ensemble | 0.9988 | 0.999 | 0.999 |
| #5 | 1D-CNN | Deep Learning | 0.8400 | 0.793 | 0.985 |
| #6 | MLP Neural Network | Deep Learning | 0.8296 | 0.779 | 0.989 |
| #7 | Autoencoder | Anomaly Detection | 0.7480 | 0.841 | 0.751 |
| #8 | Isolation Forest | Unsupervised | 0.5328 | 0.534 | 0.534 |

## Production Model: XGBoost (Multi-class) — Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Test Samples | Severity |
|---|---|---|---|---|---|
| Benign | 1.000 | 0.998 | 0.999 | 75,001 | Safe |
| DDoS | 1.000 | 1.000 | 1.000 | 19,202 | Critical |
| DoS | 1.000 | 1.000 | 1.000 | 29,062 | Critical |
| BruteForce | 1.000 | 0.999 | 1.000 | 1,373 | High |
| PortScan | 0.997 | 0.999 | 0.998 | 13,604 | Medium |
| WebAttack | 0.985 | 0.997 | 0.991 | 321 | High |
| Bot | 0.833 | 0.990 | 0.905 | 292 | High |

## Key Findings

### 1. Ensemble Models Dominate on Tabular Data
XGBoost (0.9845) and Random Forest (0.9775) significantly outperform deep learning models (MLP: 0.8296, CNN-1D: 0.8400). This is consistent with established ML literature — gradient-boosted trees excel on structured features.

### 2. SMOTE Rescues Rare Classes
Without SMOTE, Bot class (~0.1% of data) was effectively invisible to models. After SMOTE (50k synthetic samples), Bot detection reached F1=0.905 — a dramatic improvement.

### 3. Deep Learning Models Trade Precision for Recall
MLP and CNN-1D achieve >0.98 recall (catch almost all attacks) but lower precision (more false alarms). This is useful in high-security environments where missing an attack is worse than a false alarm.

### 4. Autoencoder Detects Without Labels
The Autoencoder achieves 0.748 Macro F1 without ever seeing attack labels during training. It learns "what normal looks like" and flags deviations. This proves viability for zero-day attack detection.

### 5. Near-Zero False Positive Rate
XGBoost misclassifies only 0.2% of benign traffic as attacks. In a SOC processing 1 million flows per day, that's ~2,000 false alerts — manageable with the severity system (most are Medium/Low).

## Generated Figures

All figures are auto-generated during training and saved to `reports/figures/`:

| Figure | Description |
|---|---|
| `cm_xgb_multi.png` | Confusion matrix (raw counts) — XGBoost multi-class |
| `cm_xgb_multi_norm.png` | Confusion matrix (normalized) — XGBoost multi-class |
| `roc_xgb_multi.png` | ROC curves (one-vs-rest) — XGBoost multi-class |
| `pr_xgb_multi.png` | Precision-Recall curves — XGBoost multi-class |
| `featimp_xgb_multi.png` | Top 15 feature importance — XGBoost |
| `featimp_rf_multi.png` | Top 15 feature importance — Random Forest |
| `cm_*.png` | Confusion matrices for all models |
| `roc_*.png` | ROC curves for all models |
| `pr_*.png` | PR curves for all models |

## Training Configuration

| Parameter | Value |
|---|---|
| Dataset | CIC-IDS2017 full (~2.5M flows) |
| Benign cap | 500,000 |
| SMOTE target | 50,000 per minority class |
| Train/Val/Test split | 70% / 15% / 15% (stratified) |
| Random seed | 42 |
| XGBoost Optuna trials | 15 |
| MLP/CNN epochs | 50 (early stopping, patience=5) |
| AE epochs | 50 |
| Total training time | ~40 minutes (CPU) |
