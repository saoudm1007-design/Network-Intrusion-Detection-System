# CyberShield NIDS — Installation & Setup Guide

## Prerequisites

- Python 3.10 or 3.11
- pip (package manager)
- 8 GB RAM minimum (for full dataset training)
- ~500 MB disk space (dataset + models)

## Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd cyber-ai-nids
```

## Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| numpy, pandas | Data manipulation |
| scikit-learn | Random Forest, Isolation Forest, preprocessing |
| xgboost | Gradient boosting classifier |
| imbalanced-learn | SMOTE oversampling |
| torch | MLP, 1D-CNN, Autoencoder (PyTorch) |
| optuna | Hyperparameter tuning |
| streamlit | Web dashboard |
| matplotlib, seaborn | Visualization |
| fpdf2 | PDF report generation |
| joblib | Model serialization |
| pyyaml | Configuration loading |
| pyarrow | Parquet file support |
| pytest | Testing |

## Step 4: Prepare Dataset

### Option A: Short CSV (10k rows — quick testing)
Already included at `data/raw/cicids2017.csv`. No action needed.

### Option B: Full Dataset (2.5M rows — production training)
1. Place `cicids2017_full.zip` into `data/raw/`
2. Edit `config.yaml` and set `data.use_full_dataset: true`

## Step 5: Train Models

### Quick training (short CSV, ~60 seconds)
```bash
python main.py train --config config.yaml --quick
```

### Full training (full dataset, ~40 minutes)
```bash
python main.py train --config config.yaml
```

This will:
- Load and clean the dataset
- Apply SMOTE and feature scaling
- Train all 6 models (RF, XGBoost, MLP, CNN-1D, Autoencoder, IsolationForest)
- Generate evaluation figures in `reports/figures/`
- Save all artifacts to `artifacts/`

## Step 6: Verify

```bash
# Check model scores
python main.py evaluate --config config.yaml

# Run smoke test
pytest tests/test_pipeline.py -v
```

## Step 7: Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Step 8: Predict on New Data

```bash
python main.py predict --input <your_csv> --output predictions.csv
```

The input CSV must have the same feature columns as CIC-IDS2017 (52 numeric columns).

## Troubleshooting

| Issue | Solution |
|---|---|
| Out of memory during training | Reduce `max_benign_samples` in config.yaml (default: 500000) |
| SMOTE fails | Reduce `target_per_minority` in config.yaml |
| Port 8501 in use | `fuser -k 8501/tcp` then relaunch |
| Dashboard won't load | Run `streamlit run dashboard/app.py` directly from terminal |
| Missing features error on predict | Ensure your CSV has the CIC-IDS2017 column schema |

## AWS Deployment (Optional)

For deploying both the NIDS dashboard and other projects:

| Instance | RAM | Cost/month | Recommendation |
|---|---|---|---|
| t3.small (2 GB) | 2 GB | ~$15 | Demo only, tight |
| t3.medium (4 GB) | 4 GB | ~$30 | NIDS dashboard only |
| t3.large (8 GB) | 8 GB | ~$60 | Multiple projects |

```bash
# On the server:
pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```
