# 🔍 CyberShield NIDS

ML-based Network Intrusion Detection System — XGBoost, MLP, CNN, Autoencoder + Isolation Forest on CIC-IDS2017. 98.45% Macro F1-Score. Streamlit dashboard with real-time classification and PDF threat reports.

🟢 **Live Demo:** [nids.saoud.site](https://nids.saoud.site)

---

## Results

| Rank | Model | Type | Macro-F1 |
|------|-------|------|----------|
| #1 | **XGBoost** | Ensemble | **0.9845** |
| #2 | Random Forest | Ensemble | 0.9775 |
| #3 | 1D-CNN | Deep Learning | 0.8400 |
| #4 | MLP Neural Network | Deep Learning | 0.8296 |
| #5 | Autoencoder | Anomaly Detection | 0.7480 |
| #6 | Isolation Forest | Unsupervised | 0.5328 |

### XGBoost Per-Class Performance

| Class | F1-Score | Samples | Severity |
|-------|----------|---------|----------|
| Benign | 0.999 | 75,001 | Safe |
| DDoS | 1.000 | 19,202 | Critical |
| DoS | 1.000 | 29,062 | Critical |
| BruteForce | 1.000 | 1,373 | High |
| PortScan | 0.998 | 13,604 | Medium |
| WebAttack | 0.991 | 321 | High |
| Bot | 0.905 | 292 | High |

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py train --config config.yaml       # Train all 6 models
python main.py evaluate --config config.yaml    # Print leaderboard
python main.py predict --input <csv> --output results.csv  # Predict
streamlit run dashboard/app.py                  # Launch dashboard
```

---

## Features

- **6 ML models** compared — Random Forest, XGBoost, MLP, 1D-CNN, Autoencoder, Isolation Forest
- **CIC-IDS2017 dataset** — ~2.5M network flows, 7 traffic classes
- **SMOTE oversampling** — rescues rare attack classes (Bot: 0.1% → F1=0.905)
- **Optuna hyperparameter tuning** — 15-trial TPE search for XGBoost
- **Streamlit dashboard** — 7 pages: upload, threat report, analytics, model comparison, feature analysis
- **PDF threat report export** — one-click downloadable reports
- **Model switcher** — compare predictions across all 6 models live

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | Project overview, model leaderboard, dataset summary |
| **Upload & Analyze** | CSV upload or one-click demo data loading |
| **Threat Report** | Filterable attack table with severity, PDF export |
| **Analytics** | Traffic composition, attack breakdown, severity timeline |
| **Model Comparison** | All 6 models compared with metrics and plots |
| **Feature Analysis** | Feature importance, correlation analysis |
| **About** | Architecture diagram, methodology, tech stack |

---

## Dataset

- **Source:** CIC-IDS2017 (Canadian Institute for Cybersecurity)
- **Size:** ~2.5 million network flows, 52 numeric features
- **Classes:** Benign, DDoS, DoS, PortScan, Bot, BruteForce, WebAttack
- **Preprocessing:** Benign subsampled to 500k, SMOTE for rare classes, StandardScaler
- **Split:** 70% train / 15% validation / 15% test (stratified)

---

## Methodology

1. **Data Engineering** — Clean inf/NaN, drop duplicates, remove correlated features (>0.95)
2. **Imbalance Handling** — SMOTE oversampling + class-weighted loss functions
3. **Models** — 2 ensemble, 3 deep learning, 1 unsupervised
4. **Tuning** — Optuna TPE search (15 trials) for XGBoost
5. **Metric** — Macro F1-Score (treats rare attack classes equally)
6. **Deployment** — Streamlit dashboard + PDF export

---

## Project Structure

```
├── main.py                  # CLI: train | evaluate | predict
├── config.yaml              # Hyperparameters and paths
├── requirements.txt         # Dependencies
├── src/
│   ├── data_loader.py       # Dataset loading
│   ├── preprocessing.py     # Clean, split, scale, SMOTE
│   ├── train.py             # Train all 6 models
│   ├── evaluate.py          # Metrics, confusion matrices, ROC
│   ├── predict.py           # Inference on new data
│   └── models/              # RF, XGBoost, MLP, CNN, AE, IF
├── dashboard/
│   └── app.py               # Streamlit dashboard (7 pages)
├── artifacts/               # Saved models, scalers, encoders
├── data/                    # Raw + processed datasets
├── reports/                 # Auto-generated plots + paper
├── tests/                   # Pipeline smoke test
└── docs/                    # 9 documentation files
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and data pipeline |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | ML methodology, SMOTE, tuning strategy |
| [RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md) | All model results and key findings |
| [INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md) | Setup instructions |
| [HOW_TO_USE.md](docs/HOW_TO_USE.md) | Dashboard and CLI usage guide |
| [PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md) | Presentation script |
| [PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Full research paper |

---

## Tech Stack

- **ML:** scikit-learn, XGBoost, PyTorch, imbalanced-learn (SMOTE), Optuna
- **Dashboard:** Streamlit
- **Data:** pandas, numpy, pyarrow
- **Visualization:** matplotlib, seaborn

---

## License

Course project — AI & Cyber Security, Spring 2026.
