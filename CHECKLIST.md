# Cyber-AI NIDS — Execution Checklist

Tick items as you finish them. Each phase maps to a grading component.

---

## Phase 0 — Setup (before any coding)

- [ ] Create Git repo `cyber-ai-nids` and push empty scaffold to GitHub
- [ ] Invite teammate as collaborator
- [ ] Create `README.md`, `requirements.txt`, `.gitignore`, `config.yaml`
- [ ] Create folder structure: `src/`, `src/models/`, `data/raw/`, `data/processed/`, `notebooks/`, `artifacts/`, `reports/figures/`, `reports/paper/`, `dashboard/`, `tests/`
- [ ] Add `data/raw/` and `data/processed/` to `.gitignore`
- [ ] Set up Python 3.11 virtualenv; install: `numpy pandas scikit-learn xgboost imbalanced-learn torch streamlit optuna matplotlib seaborn joblib pyyaml pytest`
- [ ] Freeze versions → `pip freeze > requirements.txt`
- [ ] Commit the scaffold (both teammates should make at least one commit)

---

## Phase 1 — Data Engineering *(Data Pipeline 20 pts)*

### Loading
- [ ] Copy `cicids2017_full.zip` into `data/raw/`
- [ ] Write `src/data_loader.py` — unzip, concat 8 daily CSVs, strip whitespace from column names
- [ ] Write `src/utils.py` — `set_seed(42)`, logging helpers, save/load pickle

### Cleaning
- [ ] In `src/preprocessing.py`: replace `inf`/`-inf` with `NaN`
- [ ] Drop or median-impute NaN rows
- [ ] Drop exact duplicate flows
- [ ] Drop constant / zero-variance columns
- [ ] Save cleaned DataFrame to `data/processed/cleaned.parquet`

### Labels
- [ ] Build label map: 15 raw labels → 7 classes (Benign, DDoS, DoS, PortScan, Bot, BruteForce, WebAttack, Infiltration)
- [ ] Create `label_multi` column (LabelEncoder) and save encoder to `artifacts/label_encoder.pkl`
- [ ] Create `label_binary` column (0=Benign, 1=Attack)

### EDA
- [ ] Create `notebooks/01_eda.ipynb`
- [ ] Plot class distribution bar chart → `reports/figures/class_dist.png`
- [ ] Plot correlation heatmap → `reports/figures/corr_heatmap.png`
- [ ] Plot top-15 feature distributions per class → `reports/figures/feature_dists.png`
- [ ] Write 2–3 bullet observations in markdown cells

### Feature selection
- [ ] Write `src/feature_selection.py`
- [ ] Drop features with > 0.95 correlation to another feature
- [ ] Train quick RF on 100k stratified sample; keep top-30 features by importance
- [ ] Save final feature list to `artifacts/selected_features.json`

### Split + scale + SMOTE
- [ ] Stratified 70/15/15 train/val/test split (`random_state=42`)
- [ ] Fit `StandardScaler` on train only → save to `artifacts/scaler.pkl`
- [ ] Apply `SMOTE(sampling_strategy='not majority', k_neighbors=5)` on train split only (multi-class model)
- [ ] Save processed splits to `data/processed/{X,y}_{train,val,test}.parquet`

### Phase 1 verification
- [ ] `preprocessing.py` is importable as a module
- [ ] Re-running the pipeline with seed=42 produces byte-identical outputs
- [ ] EDA notebook renders top-to-bottom without errors

---

## Phase 2 — Model Development *(Methodology 20 pts)*

### Baseline models
- [ ] `src/models/random_forest.py` — `RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1)` — train binary + multi
- [ ] `src/models/isolation_forest.py` — `IsolationForest(contamination=attack_ratio)` unsupervised baseline

### Advanced models
- [ ] `src/models/xgboost_model.py` — XGBoost with Optuna 30-trial TPE search (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, scale_pos_weight)
- [ ] `src/models/mlp.py` — PyTorch MLP (256→128→64, ReLU, dropout 0.3, Adam, early stopping on val macro-F1)
- [ ] `src/models/autoencoder.py` — symmetric PyTorch AE, trained on **benign only**, threshold = 99th percentile of train reconstruction error

### Orchestration
- [ ] `src/train.py` — loops over all models, saves to `artifacts/`
- [ ] Log Optuna trials to `reports/optuna_study.db`
- [ ] Save model metadata (training macro-F1, timestamp, config) to `artifacts/metadata.json`

### Evaluation
- [ ] `src/evaluate.py` outputs for each model:
  - [ ] `classification_report` (macro + per-class)
  - [ ] Confusion matrix PNG (raw + normalized) → `reports/figures/cm_<model>_<binary|multi>.png`
  - [ ] ROC curves (one-vs-rest for multi-class) → `reports/figures/roc_<model>.png`
  - [ ] Precision-Recall curves → `reports/figures/pr_<model>.png`
  - [ ] FPR / FNR at chosen threshold
  - [ ] Feature-importance bar chart (RF + XGBoost) → `reports/figures/featimp_<model>.png`
- [ ] `notebooks/03_model_comparison.ipynb` — side-by-side metric table

### Final model selection
- [ ] Pick highest val-macro-F1 model
- [ ] Retrain on train+val
- [ ] Save as `artifacts/final_model.{pkl|pt|json}`

### Phase 2 verification
- [ ] All 5 models train without error
- [ ] XGBoost beats RF by ≥ 2 macro-F1 points
- [ ] All supervised models reach ≥ 0.90 macro-F1 on test split

---

## Phase 3 — Reproducibility *(15 pts)*

- [ ] Pin all versions in `requirements.txt`
- [ ] Write `main.py` with `train | evaluate | predict` argparse subcommands
- [ ] `python main.py train --config config.yaml` runs end-to-end
- [ ] `python main.py predict --input <csv> --output submission.csv` works on any CIC-IDS2017-shaped CSV
- [ ] `config.yaml` holds seed, paths, hyperparameters (single source of truth)
- [ ] Write `tests/test_pipeline.py` — smoke test on 10k short CSV, asserts macro-F1 ≥ 0.80 in < 60 s
- [ ] `pytest tests/` passes
- [ ] `README.md` has: project description, setup, train/predict/demo commands, team members
- [ ] Commit trained artifacts (use Git LFS if any file > 100 MB)
- [ ] Both teammates have ≥ 10 commits visible on GitHub
- [ ] Fresh-clone test: create a new venv, `pip install -r requirements.txt`, run `main.py predict` — works first try

---

## Phase 4 — Streamlit Dashboard *(Website 15 pts)*

- [ ] `dashboard/app.py` — multi-page Streamlit app
- [ ] Load models/scaler/encoder once via `@st.cache_resource`
- [ ] **Page 1 — Upload:** CSV uploader + schema validator (checks expected columns)
- [ ] **Page 2 — Threat Report:** table per flow with binary verdict + attack type + confidence; row color-coded by class; CSV download button
- [ ] **Page 3 — Summary:** attack-class counts, top-10 anomalous flows by AE score, traffic composition pie chart
- [ ] **Page 4 — Model Info:** loaded model name, training macro-F1, test confusion matrix image, feature importance image
- [ ] Ship `dashboard/examples/benign_sample.csv` and `dashboard/examples/attack_sample.csv`
- [ ] Handles malformed CSV uploads gracefully (friendly error message)
- [ ] `streamlit run dashboard/app.py` works on a fresh clone

---

## Phase 5 — IEEE Paper *(15 pts, 4 pages)*

- [ ] Set up `reports/paper/paper.tex` using `IEEEtran` class
- [ ] **Abstract** — 150–200 words, includes headline macro-F1
- [ ] **Introduction + Related Work** — SOC alert fatigue motivation, CIC-IDS2017 prior benchmarks, 3–5 citations
- [ ] **Methodology** — preprocessing pipeline diagram, SMOTE+class-weight justification, model architectures, Optuna protocol
- [ ] **Results**
  - [ ] Table of macro-F1 per model (binary + multi)
  - [ ] Best-model confusion matrix
  - [ ] Per-class ROC + PR curves
  - [ ] Feature-importance bar chart
  - [ ] FPR/FNR trade-off discussion
- [ ] **Conclusion + Future Work** — CNN-1D, transformer flow models, online learning
- [ ] All figures imported from `reports/figures/` (auto-regenerated)
- [ ] Compiles to 4 pages exactly
- [ ] Commit `paper.pdf` alongside `.tex`

---

## Phase 6 — Competition Submission

- [ ] Freeze `final_model`, `scaler.pkl`, `label_encoder.pkl`, `selected_features.json`
- [ ] Run `python main.py predict --input hidden_test.csv --output submission.csv`
- [ ] Sanity checks:
  - [ ] No NaNs in `submission.csv`
  - [ ] Predicted class distribution roughly matches training distribution
  - [ ] Row count matches input
- [ ] **Data-leakage audit:** confirm scaler.fit, SMOTE, and Optuna tuning never saw the hidden test data
- [ ] Tag the final commit: `git tag v1.0-submission`
- [ ] Submit: GitHub repo link + `paper.pdf` + live demo URL/video

---

## Final pre-submission sweep

- [ ] `README.md` quickstart works on a fresh WSL machine
- [ ] All figures in paper match the figures in `reports/figures/`
- [ ] Smoke test passes
- [ ] Dashboard runs locally with example CSVs
- [ ] No hardcoded absolute paths anywhere in the codebase
- [ ] Both teammates' names in `README.md` and paper
- [ ] Both teammates' commits visible in `git log`

---

## Grade tracker

| Component | Target | Status |
|---|---|---|
| Data Pipeline (20) | 20/20 | ☐ |
| Methodology (20) | 20/20 | ☐ |
| Reproducibility (15) | 15/15 | ☐ |
| Website (15) | 15/15 | ☐ |
| Paper (15) | 15/15 | ☐ |
| Performance Rank (15) | 15/15 + bonus | ☐ |
| **Total** | **100 + bonus** | ☐ |
