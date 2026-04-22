# CyberShield NIDS — Methodology Document

## 1. Data Pipeline

### 1.1 Data Loading
- Dataset: CIC-IDS2017 (~2.5M flows, 52 features, 7 classes)
- Loaded from compressed ZIP containing daily CSVs
- Column names stripped of whitespace (known CIC-IDS2017 issue)

### 1.2 Data Cleaning
| Step | Action | Rows Affected |
|---|---|---|
| Infinite values | Replace inf/-inf with NaN | 0 |
| Missing values | Drop rows with NaN | 0 |
| Duplicates | Drop exact duplicate flows | 161 |
| Constant features | Drop zero-variance columns | 0 |
| Correlated features | Drop features with >0.95 Pearson correlation | 16 columns removed |

### 1.3 Label Mapping
The raw CIC-IDS2017 has 15 sub-labels. We consolidate them into 7 project classes:

| Raw Labels | Project Class |
|---|---|
| BENIGN, Normal Traffic | Benign |
| DDoS | DDoS |
| DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed | DoS |
| PortScan | PortScan |
| Bot | Bot |
| FTP-Patator, SSH-Patator | BruteForce |
| Web Attack - Brute Force, Web Attack - XSS, Web Attack - Sql Injection | WebAttack |

### 1.4 Class Imbalance Strategy

**Problem:** Benign traffic is 83% of all flows. Bot and WebAttack are under 0.1%.

**Solution: Two-pronged approach**

1. **SMOTE (Synthetic Minority Oversampling Technique)**
   - Applied ONLY on the training split (never on validation/test)
   - Rare classes (Bot, BruteForce, WebAttack) oversampled to 50,000 each
   - Uses k_neighbors=5 to generate synthetic samples between existing minority samples
   - Benign subsampled to 500,000 to control memory usage

2. **Class-weighted loss functions**
   - Random Forest / MLP: `class_weight='balanced'` (sklearn) / weighted CrossEntropyLoss (PyTorch)
   - XGBoost: `sample_weight` inversely proportional to class frequency
   - Provides a second layer of protection against majority-class bias

**Justification:** SMOTE alone can overfit on synthetic sample boundaries. Class weighting alone doesn't provide enough minority representation for complex models. Combining both gives robustness.

### 1.5 Data Splitting
| Split | Size | Purpose |
|---|---|---|
| Training | 788,716 (after SMOTE) | Model training |
| Validation | 138,854 | Hyperparameter tuning, early stopping |
| Test | 138,855 | Final evaluation (never seen during training/tuning) |

- Stratified splitting preserves class proportions
- Random seed fixed at 42 for reproducibility
- StandardScaler fit on training data only (prevents data leakage)

## 2. Model Architectures

### 2.1 Random Forest
```
150 Decision Trees (max_depth=25)
  → Each tree votes on the class
  → Majority vote = final prediction
  → class_weight='balanced'
```
**Why:** Strong baseline for tabular data. Resistant to overfitting. Provides feature importance rankings.

### 2.2 XGBoost (Production Model)
```
Gradient-Boosted Trees
  → Each tree corrects errors of the previous
  → Optuna tunes: n_estimators, max_depth, learning_rate, 
    subsample, colsample_bytree, min_child_weight, reg_lambda
  → 15 trials with TPE sampler
  → Sample weights for class imbalance
```
**Why:** Consistently top performer on tabular data. Optuna tuning finds optimal hyperparameters automatically.

### 2.3 MLP (Multi-Layer Perceptron)
```
Input (36 features)
  → Linear(256) → ReLU → Dropout(0.3)
  → Linear(128) → ReLU → Dropout(0.3)
  → Linear(64)  → ReLU → Dropout(0.3)
  → Linear(7)   → Softmax
```
**Why:** Satisfies the deep learning requirement. Can learn non-linear decision boundaries that trees might miss.

### 2.4 1D-CNN
```
Input (36 features) → reshape to (1, 36)
  → Conv1d(1→64, kernel=3, padding=1) → ReLU → BatchNorm
  → Conv1d(64→128, kernel=3, padding=1) → ReLU → BatchNorm
  → AdaptiveAvgPool1d(1) → Flatten
  → Linear(128→64) → ReLU → Dropout(0.3)
  → Linear(64→7)
```
**Why:** Treats features as a 1D sequence and learns local patterns via sliding filters. Common architecture in NIDS literature.

### 2.5 Autoencoder (Anomaly Detection)
```
Encoder: Input(36) → 64 → 32 → 16
Decoder: 16 → 32 → 64 → Output(36)
  → Trained ONLY on Benign flows
  → Anomaly score = reconstruction error (MSE)
  → Threshold = 99th percentile of training errors
```
**Why:** Detects attacks WITHOUT labeled attack data. If the autoencoder can't reconstruct a flow well (high MSE), it looks abnormal → likely an attack. Critical for zero-day detection.

### 2.6 Isolation Forest
```
100 Random Trees
  → Each tree randomly splits features
  → Anomalies require fewer splits to isolate
  → contamination = observed attack ratio
```
**Why:** Fully unsupervised baseline. No labels needed. Provides comparison point for the autoencoder.

## 3. Evaluation Protocol

### 3.1 Metrics
- **Macro F1-Score** (primary): Averages F1 across all 7 classes equally. A model ignoring rare classes gets penalized.
- **Per-class Precision/Recall/F1**: Identifies which specific attacks the model struggles with.
- **Confusion Matrix**: Shows what gets confused with what.
- **ROC Curves (one-vs-rest)**: Measures class separability at various thresholds.
- **Precision-Recall Curves**: More informative than ROC for imbalanced classes.

### 3.2 Model Selection
1. Train all models on the training set
2. Evaluate on the validation set during training (for early stopping / Optuna)
3. Final evaluation on the held-out test set (never used for any training decisions)
4. Select the model with the highest test Macro F1 as the production model

### 3.3 No Data Leakage Verification
- StandardScaler fit on training set only
- SMOTE applied to training set only
- Optuna uses validation set (never test set)
- Test set predictions generated once, after all training is complete

## 4. Hyperparameter Search (Optuna)

### 4.1 Search Space (XGBoost)
| Parameter | Range |
|---|---|
| n_estimators | 150 – 600 (step 50) |
| max_depth | 4 – 12 |
| learning_rate | 0.02 – 0.3 (log scale) |
| subsample | 0.7 – 1.0 |
| colsample_bytree | 0.7 – 1.0 |
| min_child_weight | 1 – 8 |
| reg_lambda | 0.001 – 10.0 (log scale) |

### 4.2 Search Strategy
- **Sampler:** TPE (Tree-structured Parzen Estimator)
- **Trials:** 15
- **Objective:** Maximize Macro F1-Score on validation set
- **Seed:** 42

## 5. Feature Importance

Top features by XGBoost importance (consistent with Random Forest rankings):
1. Destination Port
2. Init_Win_bytes_forward
3. Bwd Packet Length Std
4. Flow IAT Max
5. Fwd IAT Max
6. Bwd Packet Length Mean
7. Flow Duration
8. Packet Length Std
9. Fwd Packet Length Mean
10. Bwd IAT Mean

These features align with cybersecurity domain knowledge — attack traffic typically exhibits abnormal packet sizes, unusual timing patterns, and targets specific destination ports.
