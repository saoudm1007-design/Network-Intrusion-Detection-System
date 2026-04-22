# Cyber-AI NIDS Project — Full-Grade Plan

## Context

This is the Spring 2026 **AI & Cyber Security** course project: build a Machine-Learning-based Network Intrusion Detection System (NIDS) on **CIC-IDS2017** and compete against other teams on a hidden test set. Total grade is 100 points across Data Pipeline (20), Methodology (20), Reproducibility (15), Website (15), Paper (15), and Performance Rank (15), with up to +2% overall bonus for a leaderboard top-3 finish.

The professor's `lab_04` notebook solves a simplified binary version of the problem with a single Random Forest. Our project must go further on every axis: full dataset, multi-class (plus binary), SMOTE + class weighting, ensemble + deep-learning models, tuned for **Macro F1**, wrapped in a Streamlit dashboard, documented in a 4-page IEEE paper, and packaged as a reproducible Git repo.

### User-confirmed decisions
- **Dataset:** Full `cicids2017_full.zip` (~2.8M flows). Prototype on short `cicids2017.csv` (10k rows) for pipeline iteration.
- **Target:** **Both models** — binary (Benign vs Attack) and multi-class (Benign/DDoS/DoS/PortScan/Bot/BruteForce/WebAttack). Dashboard shows binary verdict + attack-type prediction.
- **Imbalance:** **SMOTE + class weighting** combined. SMOTE on train only (post-split) for rare classes; `class_weight='balanced'` / `scale_pos_weight` on models.
- **Advanced models (beyond RF/XGBoost baseline):** **XGBoost (tuned) + MLP + Autoencoder**. Skip 1D-CNN.
- **Dashboard:** **Streamlit**.

---

## Repository layout

```
cyber-ai-nids/
├── README.md                    # setup, run, demo instructions
├── requirements.txt
├── main.py                      # entrypoint: train | evaluate | predict
├── config.yaml                  # hyperparams, paths, random seed
├── data/
│   ├── raw/                     # cicids2017_full.zip (gitignored)
│   └── processed/               # cleaned parquet (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # load + merge CIC-IDS2017 CSVs
│   ├── preprocessing.py         # clean, encode, split, scale, SMOTE
│   ├── eda.py                   # plots + class stats (called from notebook)
│   ├── feature_selection.py     # correlation filter + RF importance
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── mlp.py               # PyTorch MLP
│   │   ├── autoencoder.py       # PyTorch AE for anomaly scoring
│   │   └── isolation_forest.py
│   ├── train.py                 # orchestrates training all models
│   ├── evaluate.py              # macro F1, confusion matrix, ROC, PR
│   └── utils.py                 # seed, logging, model (de)serialization
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_comparison.ipynb
├── artifacts/                   # saved models + scalers + encoders
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── rf_binary.pkl
│   ├── rf_multi.pkl
│   ├── xgb_binary.json
│   ├── xgb_multi.json
│   ├── mlp_multi.pt
│   └── autoencoder.pt
├── reports/
│   ├── figures/                 # confusion matrices, ROC, feature imp.
│   └── paper/                   # IEEE LaTeX source + PDF
├── dashboard/
│   └── app.py                   # Streamlit dashboard
└── tests/
    └── test_pipeline.py         # smoke test on short CSV
```

---

## Phase 1 — Data Engineering (Proposal: Data Pipeline 20 pts)

**Files:** `src/data_loader.py`, `src/preprocessing.py`, `src/eda.py`, `src/feature_selection.py`, `notebooks/01_eda.ipynb`, `notebooks/02_preprocessing.ipynb`

1. **Load & merge full dataset.** Unzip `cicids2017_full.zip`; concatenate the eight daily CSVs into one DataFrame. Strip whitespace from column names (a known CIC-IDS2017 gotcha).
2. **Clean** — reusing the pattern from `lab_04_solution.ipynb` cell 8:
   - Replace `inf`/`-inf` with `NaN` via `df.replace([np.inf, -np.inf], np.nan)`.
   - Drop or median-impute NaN rows (< 0.1% of full data).
   - Drop duplicate flows.
   - Drop constant / zero-variance columns.
3. **Label harmonization.** Map the raw 15 sub-labels to the proposal's 7 classes (`Benign`, `DDoS`, `DoS`, `PortScan`, `Bot`, `BruteForce`, `WebAttack`, `Infiltration`). Create both `label_multi` and `label_binary` (Benign=0, else=1) columns — same pattern as solution cell 8 but generalized.
4. **EDA notebook.** Class distribution, feature correlation heatmap, top-15 distributions by class. Save figures to `reports/figures/`.
5. **Feature selection.**
   - Drop features with > 0.95 correlation to another feature.
   - Rank remaining with a quick `RandomForestClassifier.feature_importances_` on a stratified 100k sample; keep top-30.
   - Document the final feature list in the paper.
6. **Split before SMOTE.** Stratified 70/15/15 train/val/test with `random_state=42` (mirrors solution cell 8 but 3-way). Scale with `StandardScaler` fit on train only.
7. **Imbalance handling.** Apply `imblearn.SMOTE(sampling_strategy='not majority', k_neighbors=5)` **on the training split only** for the multi-class model. Additionally pass `class_weight='balanced'` (RF/MLP) or `scale_pos_weight`/`sample_weight` (XGBoost). Justify the combined choice in the paper (SMOTE covers Bot/WebAttack which have < 1% support; class weights protect against SMOTE overfitting on synthetic boundaries).

**Deliverable checks:** EDA notebook renders cleanly; `preprocessing.py` is importable and produces deterministic splits given the seed.

---

## Phase 2 — Model Development (Proposal: Methodology 20 pts)

**Files:** `src/models/*.py`, `src/train.py`, `src/evaluate.py`, `notebooks/03_model_comparison.ipynb`

Train **both binary and multi-class variants** of each model. Macro F1 on the held-out test split is the selection metric.

| Model | Purpose | Library | Key hyperparameters |
|---|---|---|---|
| **Random Forest** | Baseline (matches lab) | scikit-learn | `n_estimators=200, class_weight='balanced', n_jobs=-1` |
| **XGBoost** | Competition champion | xgboost | tuned via Optuna (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, `scale_pos_weight` for binary / `sample_weight` for multi) |
| **MLP** | Deep-learning requirement | PyTorch | 3 hidden layers (256→128→64), ReLU, dropout 0.3, Adam, early stopping on val macro-F1 |
| **Autoencoder** | Zero-day anomaly detector | PyTorch | symmetric 4-layer encoder/decoder, trained on BENIGN only, threshold = 99th percentile of train reconstruction error |
| **Isolation Forest** | Unsupervised baseline | scikit-learn | `contamination=attack_ratio` (from solution cell 16) |

**Hyperparameter tuning.** Use `optuna` with 30-trial TPE search on XGBoost and MLP; fix RF/IsoForest at sensible defaults. Log every trial to `reports/optuna_study.db` for the paper.

**Evaluation (`evaluate.py`).** For each model output:
- `classification_report` (macro + per-class precision/recall/F1).
- Confusion matrix PNG (both normalized and raw).
- Per-class ROC curves + one-vs-rest AUC for multi-class.
- Precision-Recall curves (critical for imbalanced data).
- FPR / FNR at the chosen threshold.
- Feature-importance bar chart for RF and XGBoost.

**Final production model:** whichever of `{RF, XGBoost, MLP}` scores highest macro-F1 on the validation split. Retrain on train+val before predicting the test set.

---

## Phase 3 — Reproducibility (Proposal: 15 pts)

**Files:** `README.md`, `requirements.txt`, `main.py`, `config.yaml`, `tests/test_pipeline.py`

- Pin all versions in `requirements.txt` (python 3.11, numpy, pandas, scikit-learn, xgboost, imbalanced-learn, torch, streamlit, optuna, matplotlib, seaborn, joblib, pyyaml).
- `main.py` CLI with `train|evaluate|predict` subcommands (argparse).
- Single `config.yaml` holds seed (42), paths, hyperparameters.
- `README.md`: clone → `pip install -r requirements.txt` → `python main.py train --config config.yaml` → `streamlit run dashboard/app.py`.
- Smoke test in `tests/test_pipeline.py`: runs full pipeline on the 10k short CSV in < 60s to prove end-to-end reproducibility.
- Commit artifacts to `artifacts/` so the dashboard runs without retraining (use Git LFS if > 100 MB).
- **Both teammates commit** (proposal rule 6.3). Split commits fairly — e.g., one owns `src/models/*` + paper, the other owns `src/preprocessing.py` + dashboard.

---

## Phase 4 — Dashboard (Proposal: Website 15 pts)

**File:** `dashboard/app.py` (Streamlit)

Pages:
1. **Upload** — CSV uploader that expects the CIC-IDS2017 feature schema; auto-validates columns.
2. **Threat Report** — table of flows colored by predicted class, with binary verdict + multi-class attack type + confidence score. Download-as-CSV button.
3. **Summary** — counts per attack class, top-10 most anomalous flows (by autoencoder reconstruction error), pie chart of traffic composition.
4. **Model Info** — displays the loaded model name, training Macro F1, test confusion matrix, and feature importance plot (loaded from `reports/figures/`).

Load models once via `@st.cache_resource`. Accept the same scaler + label encoder saved in Phase 1. Include two example CSVs (one clean, one with attacks) in `dashboard/examples/` for demo day.

---

## Phase 5 — Paper (Proposal: 15 pts, 4 pages IEEE)

**File:** `reports/paper/paper.tex` (IEEEtran class)

Sections:
1. **Abstract** — problem, approach (multi-class NIDS with SMOTE + tuned XGBoost vs MLP vs AE), headline macro-F1.
2. **Introduction & Related Work** — SOC alert fatigue, prior CIC-IDS2017 benchmarks.
3. **Methodology** — preprocessing pipeline, justification of SMOTE + class weighting, model architectures, hyperparameter search protocol.
4. **Results** — table of macro-F1 per model, confusion matrices, ROC + PR curves, feature importance, FPR/FNR trade-off.
5. **Conclusion & Future Work** — CNN-1D, transformer flow models, online learning.

Generate all figures from `reports/figures/` — produced during Phase 2 `evaluate.py` so they always match the current model.

---

## Phase 6 — Competition Submission

- Freeze the highest-val-F1 model + scaler + encoder.
- Run `python main.py predict --input <hidden_test.csv> --output submission.csv`.
- Sanity-check: class distribution of predictions roughly matches training distribution; no NaNs.
- Confirm **no leakage**: test CSV was never seen by scaler.fit, SMOTE, or tuning.

---

## Critical files to create / reuse

| Path | Origin |
|---|---|
| `src/preprocessing.py` | Generalize `lab_04_solution.ipynb` cell 8 (binary→both labels, add SMOTE, 3-way split) |
| `src/models/random_forest.py` | Adapt cell 10 (`RandomForestClassifier(class_weight='balanced')`) |
| `src/models/isolation_forest.py` | Adapt cell 16 (`IsolationForest(contamination=attack_ratio)`) |
| `src/evaluate.py` | Generalize cells 12 & 14 (confusion matrix + feature importance) |
| `src/models/xgboost_model.py` | New — Optuna-tuned |
| `src/models/mlp.py` | New — PyTorch |
| `src/models/autoencoder.py` | New — PyTorch, trained on benign-only |
| `dashboard/app.py` | New — Streamlit |

---

## Verification

1. **Smoke test:** `pytest tests/test_pipeline.py` — trains RF on the short CSV, predicts, asserts macro-F1 ≥ 0.80 in < 60 s.
2. **Full pipeline:** `python main.py train --config config.yaml` completes on the full dataset and writes all five models + figures.
3. **Metric sanity:** `python main.py evaluate` prints per-model macro-F1; XGBoost should beat RF by ≥ 2 points; all models ≥ 0.90 macro-F1 on the test split (prior CIC-IDS2017 benchmarks).
4. **Dashboard demo:** `streamlit run dashboard/app.py`; upload `dashboard/examples/attack_sample.csv`; threat report highlights malicious flows with the correct attack types.
5. **Reproducibility:** fresh clone in a new venv → `pip install -r requirements.txt` → `python main.py predict --input …` → identical output (seed-pinned).
6. **Paper:** 4 pages, IEEE format, all figures regenerated from `reports/figures/`.

---

## Weekly timeline (aligned with proposal §7)

| Week | Output |
|---|---|
| 3 | Proposal read ✓, repo scaffold, short-CSV EDA |
| 4 | Full-dataset load/clean + SMOTE + feature selection done (checkpoint) |
| 5 | All five models trained & tuned; `evaluate.py` generating figures |
| 6 | Streamlit dashboard working with cached models |
| 7-12 | Paper draft, Optuna retuning, dashboard polish, reproducibility audit |
| 13 | Final submission + demo day |
