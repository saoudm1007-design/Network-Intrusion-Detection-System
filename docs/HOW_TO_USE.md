# CyberShield NIDS — How to Use

## Overview

CyberShield NIDS has two interfaces:
1. **CLI (Command Line)** — for training, evaluating, and batch predictions
2. **Web Dashboard (Streamlit)** — for interactive analysis, threat reports, and visual exploration

---

## CLI Usage

### Training Models

```bash
# Full training (all 6 models on full dataset, ~40 min)
python main.py train --config config.yaml

# Quick training (fewer trials, fewer epochs, ~60 sec)
python main.py train --config config.yaml --quick
```

**What happens during training:**
1. Dataset is loaded and cleaned (inf/NaN removed, duplicates dropped)
2. Benign flows are subsampled to 500k (memory control)
3. Data is split 70/15/15 (train/val/test), stratified by class
4. Features are scaled with StandardScaler (fit on train only)
5. SMOTE is applied to rare attack classes on the training set
6. All 6 models are trained and evaluated
7. Best model is auto-selected by Macro F1
8. Models, scaler, encoder, and figures are saved to `artifacts/` and `reports/figures/`

### Viewing Results

```bash
python main.py evaluate --config config.yaml
```

Prints the Macro F1 leaderboard for all trained models:
```
=== Test-split macro-F1 by model ===
  xgb_binary                 0.9991
  rf_binary                  0.9988
  xgb_multi                  0.9845
  rf_multi                   0.9775
  cnn1d_multi                0.8400
  mlp_multi                  0.8296
  autoencoder_binary         0.7480
  iforest_binary             0.5328
```

### Predicting on New Data

```bash
python main.py predict --input your_traffic.csv --output predictions.csv
```

**Input:** A CSV file with the same columns as CIC-IDS2017 (52 numeric features). The `Label` column is optional — if present, it will be preserved in the output for comparison.

**Output:** A CSV with these columns:

| Column | Description |
|---|---|
| `true_label` | Original label from input (if provided) |
| `predicted_class` | Model's prediction: Benign, DDoS, DoS, PortScan, Bot, BruteForce, or WebAttack |
| `is_attack` | 1 if attack, 0 if benign |
| `confidence` | Model's confidence in the prediction (0.0 – 1.0) |
| `anomaly_score` | Autoencoder reconstruction error (higher = more unusual) |

### Running Tests

```bash
pytest tests/test_pipeline.py -v
```

Trains Random Forest on the short CSV (10k rows) and verifies Macro F1 >= 0.60 in under 60 seconds.

### Changing Configuration

All settings are in `config.yaml`:

```yaml
# Switch between short and full dataset
data:
  use_full_dataset: true     # false = 10k rows, true = 2.5M rows

# Adjust SMOTE
smote:
  enabled: true
  target_per_minority: 50000  # how many samples per rare class

# Tune model parameters
models:
  xgboost:
    optuna_trials: 15         # more trials = better tuning, longer training
  mlp:
    epochs: 50
    patience: 5               # early stopping patience
```

---

## Web Dashboard Usage

### Launching

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### Page-by-Page Guide

#### 1. Home

The landing page shows:
- **Key metrics** — number of models trained, best Macro F1, production model name, features used
- **Attack types** — descriptions of all 6 attack categories with severity levels
- **Model leaderboard** — all 8 models ranked by Macro F1
- **Dataset summary** — training size, test size, imbalance strategy

#### 2. Upload & Analyze

**Option A: One-click demo data**
- Click **"Load sample (49 flows)"** for a quick test
- Click **"Load short dataset (10k flows)"** for a more comprehensive demo

**Option B: Upload your own CSV**
- Click "Browse files" and select a CIC-IDS2017-formatted CSV
- The system shows a progress bar as it scales features and runs inference
- After scoring, you'll see summary metrics: total flows, benign count, attacks detected, average confidence

**After loading data**, navigate to the other pages to explore results.

#### 3. Threat Report

- **Filters** — narrow results by attack class, severity level (Critical/High/Medium/Safe), or minimum confidence score
- **Severity counts** — colored metrics showing how many Critical, High, Medium, and Safe flows
- **Results table** — every flow with its predicted class, severity, confidence, and anomaly score. Rows are color-coded:
  - Red = Critical (DDoS, DoS)
  - Orange = High (Bot, BruteForce, WebAttack)
  - Yellow = Medium (PortScan)
  - Green = Safe (Benign)
- **Downloads:**
  - "Download full CSV" — all predictions
  - "Download attacks only" — only malicious flows
  - "Export Threat Report PDF" — formatted PDF report with executive summary, severity breakdown, and top 20 threats

#### 4. Analytics

Visual analysis of the uploaded traffic:
- **Traffic Composition** — donut chart showing predicted class distribution
- **Attack Type Breakdown** — horizontal bar chart of attack counts
- **Attack Severity Timeline** — scatter plot showing each flow by index, color-coded by severity. Simulates a real-time SOC monitor view
- **Confidence Distribution** — histogram of model confidence scores per class
- **Anomaly Score Distribution** — histogram comparing autoencoder reconstruction errors for benign vs attack flows
- **Severity Summary** — table with flow counts, percentages, and average confidence per severity level
- **Top 10 Most Anomalous Flows** — ranked by autoencoder reconstruction error

#### 5. Model Comparison

- **Leaderboard** — all 8 models ranked with type and Macro F1
- **Macro F1 Bar Chart** — visual comparison with "Excellent" (0.95) and "Good" (0.80) threshold lines
- **Model Detail Viewer** — select any model from the dropdown to see:
  - Per-class classification report (Precision, Recall, F1, Support)
  - Normalized confusion matrix
  - ROC curves (one-vs-rest)
  - Precision-Recall curves
  - Feature importance chart (RF and XGBoost only)
  - Raw confusion matrix

#### 6. Feature Analysis

- **Selected Features** — all 36 features used by the models, listed with index numbers
- **Feature Importance (XGBoost)** — bar chart of top 15 most important features from the production model
- **Feature Importance (Random Forest)** — bar chart for comparison
- **Correlation Heatmap** — feature-to-feature correlations (if EDA notebook was run)
- **Feature Categories** — features grouped by type: Packet Length Stats, Flow Timing, Packet Counts, TCP Flags, Header/Window, Activity/Idle

#### 7. About

- Project description and methodology summary
- Architecture diagram showing the data flow pipeline
- Technology stack breakdown (ML/DL, Data, Visualization)
- Team member section

### Sidebar Features

The left sidebar includes:

- **Navigation** — click any page to navigate
- **Dark Mode Toggle** — switch between light and dark themes
- **Model Selector** — dropdown to switch between XGBoost, Random Forest, MLP, and 1D-CNN. When you load data, it uses the selected model for inference. Switch models and reload data to compare predictions live.
- **Quick Stats** — active model name, Macro F1, loaded flow count, threats found

### Tips for Demo Day

1. **Start on the Home page** — shows your project overview and model leaderboard
2. **Click "Load short dataset (10k flows)"** on the Upload page — scores 10k flows in seconds
3. **Show the Threat Report** — filter to "Critical" severity only to highlight DDoS/DoS detections
4. **Export a PDF** — click "Export Threat Report PDF" to show the report generation feature
5. **Go to Analytics** — the severity timeline gives a "SOC monitor" feel
6. **Model Comparison** — select different models in the detail viewer to show confusion matrices and ROC curves
7. **Switch models** — use the sidebar dropdown to change from XGBoost to MLP, reload data, and show how results differ
8. **Toggle dark mode** — shows UI polish and effort

---

## Common Workflows

### "I want to test on my own network data"
1. Export your network flows to a CSV with CIC-IDS2017-compatible columns
2. Upload it on the dashboard or run `python main.py predict --input your_data.csv --output results.csv`
3. Review the threat report

### "I want to retrain with different settings"
1. Edit `config.yaml` (e.g., change `optuna_trials`, `max_benign_samples`, `hidden_dims`)
2. Run `python main.py train --config config.yaml`
3. Restart the dashboard to load the new models

### "I want to add a new model"
1. Create `src/models/your_model.py` with a train function and predict function
2. Add it to `src/train.py` (follow the pattern of existing models)
3. Add it to `src/predict.py` if it should be selectable for inference
4. Run training

### "I want to deploy on a server"
1. Provision a t3.medium (4 GB) or t3.large (8 GB) AWS instance
2. Clone the repo, install dependencies
3. Copy `artifacts/` (pre-trained models) to avoid retraining on the server
4. Run `streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0`
5. Open port 8501 in the security group
