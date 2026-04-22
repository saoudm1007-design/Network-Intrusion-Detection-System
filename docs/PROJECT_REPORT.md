# CyberShield NIDS — Project Report

## 1. Abstract

This project presents a Machine Learning-based Network Intrusion Detection System (NIDS) trained on the CIC-IDS2017 dataset containing approximately 2.5 million labeled network flows. We implement and compare six models spanning three paradigms: ensemble learning (Random Forest, XGBoost), deep learning (MLP, 1D-CNN, Autoencoder), and unsupervised anomaly detection (Isolation Forest). To address severe class imbalance — where attack traffic represents less than 17% of flows and rare classes like Bot and WebAttack constitute under 0.1% — we employ a combination of SMOTE oversampling and class-weighted loss functions. All models are optimized for Macro F1-Score to ensure equitable performance across all attack categories. Our best model, an Optuna-tuned XGBoost classifier, achieves a Macro F1-Score of **0.9845** on the held-out test set, with per-class F1 scores exceeding 0.90 for all seven traffic categories. The system is deployed as an interactive Streamlit dashboard enabling SOC analysts to upload network traffic, receive real-time threat classifications, and export PDF reports.

## 2. Introduction

### 2.1 Problem Statement

Modern Security Operations Centers (SOCs) face an overwhelming volume of network traffic logs. Traditional signature-based intrusion detection systems fail to detect zero-day attacks and sophisticated anomalies that deviate from known patterns. Machine learning offers a data-driven alternative that can learn complex patterns from historical network flow data and generalize to unseen traffic.

### 2.2 Objectives

1. Build a multi-class classifier that distinguishes Benign traffic from six attack types: DDoS, DoS, PortScan, Bot, BruteForce, and WebAttack
2. Handle severe class imbalance inherent in real-world network data
3. Compare ensemble, deep learning, and unsupervised approaches
4. Deploy the system as a web-based threat detection dashboard
5. Optimize for Macro F1-Score to ensure detection of rare attack classes

### 2.3 Related Work

The CIC-IDS2017 dataset has been widely used for benchmarking NIDS solutions. Prior work by Sharafaldin et al. (2018) established baseline results using ensemble methods. Recent studies have explored deep learning approaches including CNNs and autoencoders for anomaly-based detection. Our work combines multiple paradigms and adds a production deployment layer missing from most academic benchmarks.

## 3. Dataset

### 3.1 CIC-IDS2017 Overview

| Property | Value |
|---|---|
| Total flows | 2,520,751 |
| Features | 52 numeric packet flow statistics |
| Classes | 7 (Benign + 6 attack types) |
| Capture period | 5 days (Monday–Friday) |
| Source | Canadian Institute for Cybersecurity |

### 3.2 Class Distribution (Raw)

| Class | Count | Percentage |
|---|---|---|
| Benign | ~2,094,896 | 83.1% |
| DoS | ~231,073 | 9.2% |
| PortScan | ~158,930 | 6.3% |
| DDoS | ~128,027 | 5.1% |
| BruteForce | ~13,835 | 0.5% |
| WebAttack | ~2,180 | 0.1% |
| Bot | ~1,966 | 0.1% |

The dataset exhibits extreme class imbalance — Benign traffic outnumbers the rarest attack class (Bot) by a factor of over 1,000.

### 3.3 Features

The 52 features capture packet-level and flow-level statistics:

- **Packet size statistics:** Forward/backward packet lengths (min, max, mean, std)
- **Flow timing:** Duration, inter-arrival times (IAT) for forward/backward directions
- **Packet counts:** Total forward/backward packets, packets per second
- **TCP flags:** FIN, PSH, ACK flag counts
- **Window sizes:** Initial TCP window bytes (forward/backward)
- **Activity patterns:** Active/idle time means and extremes

## 4. Methodology

### 4.1 Data Preprocessing Pipeline

```
Raw CSV (2.5M rows)
  → Strip whitespace from column names
  → Replace inf/-inf with NaN, drop affected rows
  → Remove duplicate flows (161 removed)
  → Subsample Benign to 500,000 (memory constraint)
  → Drop 16 features with >0.95 Pearson correlation
  → Stratified split: 70% train / 15% val / 15% test
  → StandardScaler fit on training set only
  → SMOTE on training set (Bot, BruteForce, WebAttack → 50k each)
  → Final training set: 788,716 flows
```

**Key design decisions:**

1. **Benign subsampling (500k):** The full 2.1M benign flows exceeded available RAM (11 GB) during model training. Subsampling preserves the distribution while enabling training.

2. **Correlation-based feature elimination:** 16 features with >0.95 correlation to another feature were dropped (e.g., `Total Length of Fwd Packets` ≈ `Subflow Fwd Bytes`). This reduces multicollinearity without losing discriminative power.

3. **SMOTE + class weighting (dual approach):** SMOTE addresses the sample-count deficit for rare classes. Class weighting adjusts the loss function so misclassifying a rare attack costs more. Using both provides robustness — SMOTE helps models see diverse minority examples, while class weighting prevents overfitting on synthetic boundaries.

4. **Scaler fit on training only:** The StandardScaler is fit exclusively on training data to prevent data leakage from validation/test sets.

### 4.2 Models

#### 4.2.1 Random Forest (Baseline)

- 150 decision trees with max_depth=25
- `class_weight='balanced'`
- Serves as the baseline, matching the lab solution approach

#### 4.2.2 XGBoost (Production Model)

- Gradient-boosted trees optimized via Optuna (15-trial TPE search)
- Tuned parameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda
- Sample weights computed inversely proportional to class frequency
- Selected as production model based on highest validation Macro F1

#### 4.2.3 MLP Neural Network

- Architecture: Input → 256 → 128 → 64 → 7 (output)
- Activation: ReLU with Dropout (p=0.3)
- Optimizer: Adam (lr=0.001)
- Early stopping on validation Macro F1 (patience=5)
- Class-weighted CrossEntropyLoss

#### 4.2.4 1D-CNN

- Architecture: Conv1d(1→64, k=3) → ReLU → BN → Conv1d(64→128, k=3) → ReLU → BN → AdaptiveAvgPool → FC(128→64) → FC(64→7)
- Treats the 36-feature vector as a 1D signal
- Same training protocol as MLP (early stopping, class-weighted loss)

#### 4.2.5 Autoencoder (Anomaly Detection)

- Symmetric architecture: Input → 64 → 32 → 16 → 32 → 64 → Input
- **Trained on Benign flows only** — learns to reconstruct normal traffic
- Anomaly score = mean squared reconstruction error per flow
- Threshold set at the 99th percentile of training reconstruction errors
- Flows exceeding the threshold are flagged as attacks
- Designed to detect zero-day attacks that supervised models have never seen

#### 4.2.6 Isolation Forest (Unsupervised Baseline)

- 100 estimators, contamination set to the observed attack ratio
- No labels used during training
- Provides an unsupervised baseline for comparison

### 4.3 Evaluation Metrics

- **Primary:** Macro F1-Score (averages F1 equally across all 7 classes)
- **Supporting:** Per-class Precision, Recall, F1-Score, Confusion Matrix, ROC Curves (one-vs-rest), Precision-Recall Curves, FPR/FNR

**Why Macro F1, not Accuracy?** A model that always predicts "Benign" achieves ~83% accuracy but catches zero attacks. Macro F1 forces equitable performance across all classes, including rare ones like Bot (292 test samples) and WebAttack (321 test samples).

## 5. Results

### 5.1 Model Comparison

| Model | Type | Macro-F1 | Precision | Recall |
|---|---|---|---|---|
| **XGBoost** | Ensemble | **0.9845** | 0.973 | 0.998 |
| Random Forest | Ensemble | 0.9775 | 0.966 | 0.991 |
| 1D-CNN | Deep Learning | 0.8400 | 0.793 | 0.985 |
| MLP | Deep Learning | 0.8296 | 0.779 | 0.989 |
| Autoencoder | Deep Learning | 0.7480 | 0.841 | 0.751 |
| Isolation Forest | Unsupervised | 0.5328 | 0.534 | 0.534 |

### 5.2 XGBoost Per-Class Performance (Production Model)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Benign | 1.000 | 0.998 | 0.999 | 75,001 |
| Bot | 0.833 | 0.990 | 0.905 | 292 |
| BruteForce | 1.000 | 0.999 | 1.000 | 1,373 |
| DDoS | 1.000 | 1.000 | 1.000 | 19,202 |
| DoS | 1.000 | 1.000 | 1.000 | 29,062 |
| PortScan | 0.997 | 0.999 | 0.998 | 13,604 |
| WebAttack | 0.985 | 0.997 | 0.991 | 321 |

**Key observations:**
- XGBoost achieves near-perfect F1 (≥0.99) on 5 of 7 classes
- Bot is the hardest class (F1=0.905) due to its extremely low representation, but SMOTE brought it well above baseline
- WebAttack detection (F1=0.991) is remarkably strong despite only 321 test samples
- The FPR for Benign traffic is 0.2%, meaning very few false alarms for SOC analysts

### 5.3 Analysis

**Ensemble vs Deep Learning:** XGBoost and Random Forest significantly outperform the neural network models on this tabular dataset. This is consistent with established benchmarks showing gradient-boosted trees dominate on structured/tabular data. The deep learning models (MLP, CNN-1D) achieve strong recall (>0.98) but lower precision, generating more false positives.

**Supervised vs Unsupervised:** The Autoencoder achieves a respectable 0.748 Macro F1 without any attack labels during training. This demonstrates viability for zero-day detection. However, the supervised models' advantage is clear when known attack patterns exist in the training data.

**Effect of SMOTE:** Without SMOTE, the Bot class had only ~1,400 training samples (0.2% of training data). SMOTE raised this to 50,000, enabling XGBoost to achieve 0.905 F1 on Bot — a class that was previously nearly invisible to the model.

## 6. Deployment

### 6.1 Web Dashboard (Streamlit)

The system is deployed as a 7-page Streamlit web application:

1. **Home:** Project overview, model leaderboard, key metrics
2. **Upload & Analyze:** CSV upload with progress animation, one-click demo data loading
3. **Threat Report:** Filterable results table with severity levels (Critical/High/Medium/Safe), PDF export
4. **Analytics:** Traffic composition charts, attack severity timeline, anomaly score distributions
5. **Model Comparison:** Interactive model switcher with per-model metrics, confusion matrices, ROC/PR curves
6. **Feature Analysis:** Feature importance rankings, correlation analysis, feature categorization
7. **About:** Architecture diagram, technology stack

### 6.2 Key Dashboard Features

- **Live model switching:** Analysts can switch between XGBoost, RF, MLP, and CNN-1D to compare predictions on the same data
- **PDF export:** One-click threat report generation for incident documentation
- **Dark mode:** Toggle between light and dark themes
- **Severity classification:** Attacks are categorized as Critical (DDoS, DoS), High (Bot, BruteForce, WebAttack), or Medium (PortScan)

### 6.3 CLI Interface

```bash
python main.py train     # Train all models
python main.py evaluate  # Print metrics leaderboard
python main.py predict   # Score new CSV data
```

## 7. Reproducibility

- All random seeds fixed at 42
- Dependencies pinned in `requirements.txt`
- Single `config.yaml` for all hyperparameters and paths
- Smoke test (`pytest tests/test_pipeline.py`) validates pipeline on short CSV
- Pre-trained models saved in `artifacts/` for instant dashboard deployment

## 8. Conclusion

We built a complete NIDS pipeline from raw data to deployed dashboard, achieving 0.9845 Macro F1-Score with an Optuna-tuned XGBoost classifier. The combination of SMOTE oversampling and class-weighted losses proved effective at handling extreme class imbalance, enabling detection of rare attack classes (Bot: F1=0.905, WebAttack: F1=0.991). The Autoencoder anomaly detector provides a complementary zero-day detection capability without requiring labeled attack data.

### 8.1 Future Work

- **Transformer-based models:** Apply attention mechanisms to network flow sequences
- **Online learning:** Incremental model updates as new traffic arrives
- **Explainability:** SHAP values for per-prediction explanations
- **Real-time ingestion:** Connect to live packet capture (pcap) sources
- **Federated learning:** Train across distributed SOCs without sharing raw traffic data

## 9. Technology Stack

| Category | Technologies |
|---|---|
| ML/DL | scikit-learn, XGBoost, PyTorch, Optuna, imbalanced-learn |
| Data | pandas, NumPy, pyarrow |
| Visualization | Matplotlib, Seaborn, Streamlit |
| Testing | pytest |
| Config | PyYAML |
