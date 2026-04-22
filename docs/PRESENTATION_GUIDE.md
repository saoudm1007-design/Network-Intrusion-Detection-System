# CyberShield NIDS — Presentation Guide

## 1. Project Overview

We built a **Machine Learning-based Network Intrusion Detection System (NIDS)** that analyzes network flow data and classifies traffic as **Benign** or one of **6 attack types**: DDoS, DoS, PortScan, Bot, BruteForce, and WebAttack.

**Key stats:**
- **0.9845 Macro F1-Score** on 139k test flows (production model: XGBoost)
- **8 models** trained: 2 ensemble, 3 deep learning, 1 unsupervised, 2 binary variants
- **2.5 million flows** from the CIC-IDS2017 dataset
- **7 classes** detected with per-class F1 ≥ 0.90 for all classes
- **6 attack types** with severity levels: Critical, High, Medium
- **7-page Streamlit dashboard** with live model switching, PDF export, dark mode
- **Zero data leakage** — scaler, SMOTE, and tuning never touch the test set

---

## 2. Core AI/ML Concepts Used

### 2.1 Ensemble Learning (Random Forest + XGBoost)

**Random Forest** builds 150 independent decision trees, each trained on a random subset of data and features. The final prediction is the majority vote across all trees.

**XGBoost (Extreme Gradient Boosting)** builds trees sequentially — each new tree specifically corrects the errors of the previous ones. This iterative refinement is why it outperforms Random Forest.

**How we implement it:**
- Random Forest: 150 trees, max_depth=25, class_weight='balanced'
- XGBoost: hyperparameters tuned by Optuna (15-trial TPE search) — the optimizer automatically finds the best combination of learning rate, tree depth, regularization, etc.

**Why both?** Random Forest is the robust baseline (matches the lab). XGBoost is the competition weapon — it wins because gradient boosting focuses on hard-to-classify samples (like rare attack classes).

### 2.2 Deep Learning (MLP + 1D-CNN)

**MLP (Multi-Layer Perceptron):** A feedforward neural network with 3 hidden layers (256→128→64 neurons). Each layer applies a linear transformation, ReLU activation (keeps positive values, zeros negative), and Dropout (randomly disables 30% of neurons during training to prevent overfitting).

**1D-CNN (1D Convolutional Neural Network):** Treats the 36-feature vector as a 1D signal and slides convolutional filters across it. Two Conv1d layers (1→64 channels, 64→128 channels) with kernel size 3 learn local patterns between adjacent features. BatchNorm stabilizes training. AdaptiveAvgPool compresses to a fixed-size vector before the final dense layers.

**How we implement it:**
- Both trained in PyTorch with class-weighted CrossEntropyLoss
- Adam optimizer with early stopping (patience=5 epochs on validation Macro F1)
- Both output 7-class softmax probabilities

**Why both?** The proposal requires "MLP, CNN-1D, or Autoencoder" — we implemented all three to demonstrate breadth.

### 2.3 Anomaly Detection (Autoencoder + Isolation Forest)

**Autoencoder:** A neural network trained to reconstruct its input. The key insight: we train it **only on Benign traffic**. It learns "what normal looks like." When an attack flow is fed in, the reconstruction is poor (high error) — flagging it as anomalous.

```
Normal flow  → Autoencoder → Good reconstruction  → Low error  → Benign
Attack flow  → Autoencoder → Bad reconstruction   → High error → Attack!
```

The threshold is set at the 99th percentile of training reconstruction errors.

**Isolation Forest:** A tree-based anomaly detector that works by randomly splitting data. Anomalies (attacks) are "easy to isolate" — they need fewer random splits. Normal flows need many splits because they cluster together. No labels needed.

**Why include unsupervised methods?** They can detect **zero-day attacks** — attack types the supervised models have never seen during training.

### 2.4 Class Imbalance Handling (SMOTE + Class Weighting)

**The problem:** 83% of traffic is Benign. Bot is 0.1%. A model that always says "Benign" gets 83% accuracy but catches zero attacks.

**SMOTE (Synthetic Minority Oversampling Technique):** Creates synthetic samples for rare classes by interpolating between existing minority samples. We oversample Bot, BruteForce, and WebAttack to 50,000 each — applied ONLY on the training set (never val/test).

**Class weighting:** Adjusts the loss function so misclassifying a Bot costs 1000x more than misclassifying a Benign flow. Implemented via `class_weight='balanced'` (sklearn) and weighted CrossEntropyLoss (PyTorch).

**Why both?** SMOTE gives the model diverse minority examples to learn from. Class weighting ensures the loss function prioritizes rare classes. Combining both provides robustness — SMOTE alone can overfit on synthetic boundaries; weights alone don't provide enough sample diversity.

### 2.5 Hyperparameter Optimization (Optuna)

**Optuna** uses TPE (Tree-structured Parzen Estimator) to efficiently search the hyperparameter space. Instead of trying every combination (grid search), it builds a probabilistic model of which regions produce good results and focuses the search there.

**Our search space (XGBoost):**
- n_estimators: 150–600
- max_depth: 4–12
- learning_rate: 0.02–0.3
- subsample: 0.7–1.0
- colsample_bytree: 0.7–1.0
- min_child_weight: 1–8
- reg_lambda: 0.001–10.0

**15 trials** — each trains a full XGBoost model and evaluates on the validation set. The best trial's parameters become the production model.

### 2.6 Evaluation: Macro F1-Score

**Why not accuracy?** A model that always predicts "Benign" gets 83% accuracy but catches zero attacks. Useless in a SOC.

**Macro F1** averages the F1-score across all 7 classes **equally**:
- Bot (292 test samples) counts the same as Benign (75,000 test samples)
- Forces the model to actually learn rare attacks
- F1 itself balances Precision (are the alerts real?) and Recall (did we catch all attacks?)

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     CyberShield NIDS                             │
│                                                                  │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐ │
│  │   Data    │──▶│Processing │──▶│  Model    │──▶│  Serving  │ │
│  │   Layer   │   │  Layer    │   │  Layer    │   │  Layer    │ │
│  └───────────┘   └───────────┘   └───────────┘   └───────────┘ │
│                                                                  │
│  CSV/ZIP          Clean           6 Models        Dashboard      │
│  2.5M flows       SMOTE           Optuna          CLI            │
│  52 features      Scale           Evaluate        PDF Export     │
│  7 classes        Split           Select best     Model Switch   │
└──────────────────────────────────────────────────────────────────┘
```

### How it works step by step:

1. **Data Loading:** CIC-IDS2017 ZIP (8 daily CSVs) → merged into 2.5M-row DataFrame
2. **Cleaning:** Remove inf/NaN, drop 161 duplicates, strip column whitespace
3. **Label Mapping:** 15 raw sub-labels → 7 project classes (e.g., "DoS Hulk" + "DoS slowloris" → "DoS")
4. **Benign Subsampling:** 2.1M benign → 500k (memory constraint)
5. **Feature Selection:** Drop 16 features with >0.95 correlation → 36 remaining
6. **Stratified Split:** 70/15/15 train/val/test (seed=42)
7. **Scaling:** StandardScaler fit on training data only
8. **SMOTE:** Oversample Bot/BruteForce/WebAttack to 50k each on training set
9. **Training:** 6 models trained sequentially (RF → XGBoost → MLP → CNN-1D → AE → IsoForest)
10. **Evaluation:** Macro F1 + confusion matrices + ROC + PR curves for every model
11. **Selection:** Best multi-class model auto-selected (XGBoost: 0.9845)
12. **Serving:** Dashboard loads saved models and scores uploaded CSVs in real-time

### Data Leakage Prevention

```
  ✓ Scaler fit on TRAINING data only
  ✓ SMOTE applied to TRAINING set only
  ✓ Optuna tunes on VALIDATION set only
  ✓ Test set used ONCE for final evaluation
  ✓ Random seed fixed at 42 for reproducibility
```

---

## 4. The 6 Models

| # | Model | Type | Macro-F1 | Purpose |
|---|---|---|---|---|
| 1 | **XGBoost** | Ensemble | **0.9845** | Production model — best overall |
| 2 | **Random Forest** | Ensemble | 0.9775 | Baseline (matches lab) |
| 3 | **1D-CNN** | Deep Learning | 0.8400 | Convolutional approach to tabular data |
| 4 | **MLP** | Deep Learning | 0.8296 | Classic feedforward neural network |
| 5 | **Autoencoder** | Deep Learning | 0.7480 | Zero-day anomaly detection (benign-only training) |
| 6 | **Isolation Forest** | Unsupervised | 0.5328 | Unsupervised baseline |

### XGBoost Per-Class Performance (Production Model)

| Class | Precision | Recall | F1-Score | Samples | Severity |
|---|---|---|---|---|---|
| Benign | 1.000 | 0.998 | 0.999 | 75,001 | Safe |
| DDoS | 1.000 | 1.000 | 1.000 | 19,202 | Critical |
| DoS | 1.000 | 1.000 | 1.000 | 29,062 | Critical |
| BruteForce | 1.000 | 0.999 | 1.000 | 1,373 | High |
| PortScan | 0.997 | 0.999 | 0.998 | 13,604 | Medium |
| WebAttack | 0.985 | 0.997 | 0.991 | 321 | High |
| Bot | 0.833 | 0.990 | 0.905 | 292 | High |

**Key takeaway:** Near-perfect detection (≥0.99 F1) on 5 of 7 classes. Even the hardest class (Bot, only 292 test samples) achieves 0.905 F1 thanks to SMOTE.

---

## 5. Web Dashboard (Streamlit)

### 7 Pages

| Page | What it shows |
|---|---|
| **Home** | Project overview, 4 key metric cards, model leaderboard, attack type descriptions with severity, dataset summary |
| **Upload & Analyze** | One-click demo buttons (49 or 10k flows), CSV uploader, progress bar animation, post-upload stats |
| **Threat Report** | Filterable table (by class, severity, confidence), color-coded rows (red/orange/yellow/green), PDF export, CSV downloads |
| **Analytics** | Donut chart, attack breakdown bars, severity timeline (SOC monitor), confidence histograms, anomaly score distribution, top-10 anomalous flows |
| **Model Comparison** | Full leaderboard, Macro F1 bar chart with threshold lines, per-model detail viewer (classification report, confusion matrix, ROC, PR, feature importance) |
| **Feature Analysis** | 36 features listed, XGBoost + RF importance charts, feature categories (packet stats, timing, flags, etc.) |
| **About** | Architecture diagram, tech stack, methodology |

### Dashboard Features

| Feature | Description |
|---|---|
| **Model Selector** | Sidebar dropdown — switch between XGBoost, RF, MLP, CNN-1D live. Reload data to compare predictions. |
| **Dark Mode** | Toggle in sidebar — full dark theme across the app |
| **PDF Export** | One-click threat report: executive summary, severity breakdown, top 20 threats table |
| **Demo Buttons** | "Load sample (49 flows)" and "Load 10k flows" — no file browsing needed |
| **Progress Bar** | Animated: "Loading model → Scaling features → Running inference → Done!" |
| **Severity System** | Critical (DDoS, DoS), High (Bot, BruteForce, WebAttack), Medium (PortScan), Safe (Benign) |
| **Severity Timeline** | Scatter plot of flows by index, color-coded by severity — simulates real-time SOC monitor |

---

## 6. Dataset: CIC-IDS2017

| Property | Value |
|---|---|
| Source | Canadian Institute for Cybersecurity |
| Total flows | 2,520,751 |
| Features | 52 numeric (packet sizes, timing, flags, headers) |
| Classes | 7 (Benign + 6 attack types) |
| Capture period | 5 days (Monday–Friday) |
| Used features (after selection) | 36 (16 dropped for >0.95 correlation) |

### Class Distribution

| Class | Raw Count | % of Total | After SMOTE (train) |
|---|---|---|---|
| Benign | ~2,094,896 | 83.1% | 500,000 (subsampled) |
| DoS | ~231,073 | 9.2% | unchanged |
| PortScan | ~158,930 | 6.3% | unchanged |
| DDoS | ~128,027 | 5.1% | unchanged |
| BruteForce | ~13,835 | 0.5% | 50,000 (SMOTE) |
| WebAttack | ~2,180 | 0.1% | 50,000 (SMOTE) |
| Bot | ~1,966 | 0.1% | 50,000 (SMOTE) |

---

## 7. Code Structure

```
cyber-ai-nids/
├── main.py                  # CLI: train | evaluate | predict
├── config.yaml              # All hyperparameters + paths
├── requirements.txt         # Pinned dependencies
├── src/
│   ├── data_loader.py       # Load short CSV or full ZIP
│   ├── preprocessing.py     # Clean, label, split, scale, SMOTE
│   ├── train.py             # Orchestrates all 6 models
│   ├── evaluate.py          # Metrics, confusion matrices, ROC, PR
│   ├── predict.py           # Inference on new CSVs
│   ├── utils.py             # Seed, logging, save/load
│   └── models/
│       ├── random_forest.py # sklearn RF
│       ├── xgboost_model.py # XGBoost + Optuna tuning
│       ├── mlp.py           # PyTorch MLP (256→128→64)
│       ├── cnn1d.py         # PyTorch 1D-CNN
│       ├── autoencoder.py   # PyTorch AE (benign-only)
│       └── isolation_forest.py
├── dashboard/
│   └── app.py               # Streamlit (7 pages, ~600 lines)
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory Data Analysis
├── artifacts/               # Saved models + scaler + encoder
├── reports/figures/         # 30 auto-generated plots
├── tests/
│   └── test_pipeline.py     # Smoke test
└── docs/                    # 6 documentation files
```

### Key Design Decisions

- **Single config.yaml:** All hyperparameters, paths, and settings in one file. Change one value, retrain, everything updates.
- **Modular models:** Each model is its own file with build/train/predict functions. Easy to add a 7th model.
- **Auto-generated figures:** evaluate.py produces all plots during training. Paper and dashboard always show current results.
- **Model override in predict:** Dashboard can switch models without retraining — `model_override` parameter in predict_csv().
- **SMOTE after split:** Prevents synthetic samples from leaking test-set information.

---

## 8. Results Analysis

### Why Ensemble > Deep Learning on This Data?

XGBoost (0.9845) and RF (0.9775) beat MLP (0.8296) and CNN-1D (0.8400) because:
- CIC-IDS2017 features are **tabular** (structured numeric data), not images or sequences
- Tree-based models handle tabular data natively — they split on feature thresholds
- Neural networks need to learn these splits from scratch, and 36 features isn't enough "structure" for CNNs to find meaningful patterns
- This is consistent with published ML benchmarks: gradient-boosted trees dominate on tabular data

### Why Autoencoder Works for Zero-Day Detection

- Trained **only on Benign** flows — never sees any attack patterns
- Learns a compressed representation of "normal" traffic (36→64→32→16→32→64→36)
- Any flow that can't be well-reconstructed is flagged as anomalous
- Achieves 0.748 Macro F1 **without any attack labels**
- In practice: deploy alongside XGBoost. XGBoost catches known attacks; AE catches novel ones.

### Effect of SMOTE on Rare Classes

| Class | Without SMOTE (train samples) | With SMOTE | Test F1 |
|---|---|---|---|
| Bot | ~1,400 (0.2%) | 50,000 | 0.905 |
| WebAttack | ~1,500 (0.2%) | 50,000 | 0.991 |
| BruteForce | ~9,700 (1.4%) | 50,000 | 1.000 |

Without SMOTE, Bot would be nearly invisible during training. SMOTE made the difference between ~0.3 and 0.905 F1.

---

## 9. Live Demo Script

### Step 1: Home Page (30 sec)
"This is our NIDS — it scores network traffic in real-time. We trained 8 models on 2.5 million flows. The production model is XGBoost with 98.45% Macro F1."

### Step 2: Load Demo Data (30 sec)
Go to **Upload & Analyze** → Click **"Load short dataset (10k flows)"**
"We're loading 10,000 real network flows from CIC-IDS2017. Watch the progress bar — it scales features and runs XGBoost inference in a few seconds."

### Step 3: Threat Report (1 min)
Go to **Threat Report** → Filter to **"Critical"** severity only
"These are the DDoS and DoS attacks detected. Each row shows the predicted class, severity level, and model confidence. We can export this as a PDF for incident documentation."
Click **"Export Threat Report PDF"** → show the download.

### Step 4: Analytics (1 min)
Go to **Analytics**
"The donut chart shows traffic composition. The severity timeline below simulates what a SOC analyst would see — red dots are critical threats, green is normal. Notice the cluster of attacks around flow index 4000–5000."

### Step 5: Switch Models (1 min)
In the sidebar, change model from **XGBoost** to **MLP Neural Network** → Go back to **Upload & Analyze** → Click **"Load short dataset"** again
"Now we're using the MLP instead of XGBoost. Notice the threat count changes — MLP generates more false positives because it has lower precision. This is why we selected XGBoost as the production model."

### Step 6: Model Comparison (1 min)
Go to **Model Comparison** → Select **xgb_multi** from the detail viewer
"Here's the full leaderboard. XGBoost leads at 0.9845. The confusion matrix shows near-perfect classification — the only confusion is between Bot and Benign, which is the hardest class. The ROC curves show AUC near 1.0 for every class."

### Step 7: Feature Analysis (30 sec)
Go to **Feature Analysis**
"These are the top features driving detection. Destination Port is #1 — attacks target specific ports. Init_Win_bytes_forward is #2 — attack traffic has unusual TCP window sizes. These align with cybersecurity domain knowledge."

### Total demo time: ~5–6 minutes

---

## 10. Anticipated Questions & Answers

**Q: Why XGBoost over a deep learning model?**
A: On tabular data like CIC-IDS2017, gradient-boosted trees consistently outperform neural networks. This is well-documented in ML literature (Grinsztajn et al., 2022). Our results confirm it: XGBoost 0.9845 vs MLP 0.8296. We still implemented MLP, CNN-1D, and Autoencoder to satisfy the deep learning requirement and demonstrate breadth.

**Q: Is SMOTE cheating? You're creating fake data.**
A: No. SMOTE creates synthetic samples by interpolating between real minority samples — it doesn't fabricate from nothing. Critically, we apply SMOTE **only on the training set**, never on validation or test. The test set metrics reflect real-world performance. SMOTE is an established technique published in the Journal of Artificial Intelligence Research (Chawla et al., 2002).

**Q: How do you prevent data leakage?**
A: Three safeguards: (1) StandardScaler is fit on training data only — val/test are transformed with training statistics. (2) SMOTE is applied to training data only — no synthetic val/test samples. (3) Optuna tunes on the validation set — the test set is touched once, at the very end, for final evaluation.

**Q: Why subsample Benign to 500k instead of using all 2.1M?**
A: Memory constraint. Training XGBoost and RF on 2.1M rows with SMOTE blowing up the training set to 10M+ rows exceeded our 11GB RAM. Subsampling to 500k preserves the class distribution while keeping training feasible. The 500k benign samples still vastly outnumber any attack class.

**Q: What's the Autoencoder good for if XGBoost is better?**
A: XGBoost is better on **known** attack types. The Autoencoder can detect attacks it has **never seen** — zero-day attacks. Since it only learns "normal," anything unusual gets flagged. In production, you'd run both: XGBoost for known threats, AE for novel ones.

**Q: Why Macro F1 instead of Accuracy?**
A: With 83% Benign traffic, a model that always says "Benign" gets 83% accuracy but catches zero attacks. Macro F1 treats all 7 classes equally — Bot (0.1% of data) matters as much as Benign (83%). This is the standard metric for imbalanced classification in cybersecurity.

**Q: Can the system work on real-time traffic?**
A: The current system works on CSV uploads (batch mode). For real-time, you'd add a packet capture layer (e.g., CICFlowMeter) that converts raw packets into flow statistics, then pipe the features into our predict function. The model inference takes <1 second per batch.

**Q: How long does training take?**
A: ~40 minutes on CPU (11GB RAM machine) for all 6 models on the full 2.5M-row dataset. Quick mode (short CSV, fewer Optuna trials) takes ~60 seconds.

**Q: What would you improve with more time?**
A: (1) Transformer-based model for temporal flow patterns, (2) Online learning for model updates as new attacks emerge, (3) SHAP explanations for per-prediction interpretability, (4) Real-time packet capture integration, (5) Federated learning across distributed SOCs.

**Q: Why did CNN-1D not beat the MLP?**
A: On tabular data, adjacent features don't have the spatial relationship that CNNs exploit in images. Feature position in CIC-IDS2017 is arbitrary — "Fwd Packet Length" next to "Bwd Packet Length" isn't a meaningful spatial signal. CNN-1D slightly outperformed MLP (0.84 vs 0.83) by finding some local correlations, but neither matches tree-based models on this data type.

**Q: What's the False Positive Rate?**
A: 0.2% — XGBoost misclassifies only 0.2% of benign traffic as attacks. In a SOC processing 1 million flows per day, that's ~2,000 false alerts. With our severity system (Critical/High/Medium), analysts can prioritize the highest-confidence threats first.

---

## 11. Key Terminology Quick Reference

| Term | Definition |
|---|---|
| NIDS | Network Intrusion Detection System — monitors network traffic for malicious activity |
| SOC | Security Operations Center — team that monitors and responds to security events |
| CIC-IDS2017 | Benchmark dataset with 2.5M labeled network flows from the Canadian Institute for Cybersecurity |
| Macro F1 | Metric that averages F1-score equally across all classes, regardless of class size |
| SMOTE | Synthetic Minority Oversampling Technique — creates synthetic samples for rare classes |
| XGBoost | Extreme Gradient Boosting — builds trees sequentially, each correcting prior errors |
| Random Forest | Ensemble of independent decision trees that vote on the prediction |
| MLP | Multi-Layer Perceptron — feedforward neural network with fully connected layers |
| CNN-1D | 1D Convolutional Neural Network — slides filters across feature vectors |
| Autoencoder | Neural network that compresses and reconstructs input; high reconstruction error = anomaly |
| Isolation Forest | Unsupervised anomaly detector; anomalies are easier to isolate with random splits |
| Optuna | Hyperparameter optimization framework using TPE (Tree-structured Parzen Estimator) |
| Stratified Split | Data splitting that preserves class proportions in train/val/test sets |
| StandardScaler | Normalizes features to mean=0, standard deviation=1 |
| DDoS | Distributed Denial of Service — flooding a target from multiple sources |
| DoS | Denial of Service — exhausting a target's resources |
| PortScan | Probing a target to discover open ports and running services |
| BruteForce | Systematically guessing credentials (e.g., FTP/SSH passwords) |
| Bot | Compromised machine communicating with attacker's command-and-control server |
| WebAttack | Exploiting web applications (SQL injection, XSS, brute force login) |
| FPR | False Positive Rate — percentage of benign traffic incorrectly flagged as attacks |
| ROC Curve | Receiver Operating Characteristic — plots true positive rate vs false positive rate |
| PR Curve | Precision-Recall Curve — more informative than ROC for imbalanced data |
| TPE | Tree-structured Parzen Estimator — Bayesian optimization algorithm used by Optuna |
| Data Leakage | When test/validation information accidentally influences training, inflating metrics |
| Zero-Day | An attack exploiting an unknown vulnerability — not in any training data |
| Class Imbalance | When some classes have far more samples than others (e.g., 83% Benign vs 0.1% Bot) |

---

## 12. Technology Stack

| Category | Technologies |
|---|---|
| **Ensemble ML** | scikit-learn (Random Forest, Isolation Forest), XGBoost |
| **Deep Learning** | PyTorch (MLP, 1D-CNN, Autoencoder) |
| **Optimization** | Optuna (TPE hyperparameter search) |
| **Imbalance** | imbalanced-learn (SMOTE) |
| **Data** | pandas, NumPy, pyarrow |
| **Visualization** | Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **PDF Export** | fpdf2 |
| **Config** | PyYAML |
| **Testing** | pytest |
| **Serialization** | joblib (sklearn models), torch.save (PyTorch models), JSON (XGBoost) |
