# CyberShield NIDS — Architecture Document

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CyberShield NIDS                                  │
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   │
│  │  Data Layer  │──▶│  Processing │──▶│ Model Layer │──▶│  Serving     │   │
│  │             │   │  Layer      │   │             │   │  Layer       │   │
│  │ - CSV/ZIP   │   │ - Clean     │   │ - Train     │   │ - Dashboard  │   │
│  │ - Raw data  │   │ - SMOTE     │   │ - Evaluate  │   │ - CLI        │   │
│  │             │   │ - Scale     │   │ - Select    │   │ - PDF Export │   │
│  └─────────────┘   └─────────────┘   └─────────────┘   └──────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Data Flow Architecture

### 2.1 Training Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                │
│                                                                          │
│  cicids2017_full.zip                                                     │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │  Load &  │──▶│  Clean   │──▶│  Label   │──▶│Subsample │             │
│  │  Merge   │   │  Data    │   │  Map     │   │ Benign   │             │
│  │ 8 CSVs   │   │ inf/NaN  │   │ 15 → 7   │   │ → 500k   │             │
│  │ 2.5M rows│   │ dupes    │   │ classes  │   │          │             │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘             │
│                                                      │                   │
│                                                      ▼                   │
│                              ┌──────────────────────────────────┐        │
│                              │      Stratified Split (seed=42)  │        │
│                              │   70% train / 15% val / 15% test │        │
│                              └──────┬──────────┬──────────┬─────┘        │
│                                     │          │          │              │
│                                     ▼          ▼          ▼              │
│                               ┌─────────┐ ┌────────┐ ┌────────┐        │
│                               │  Train  │ │  Val   │ │  Test  │        │
│                               │ 538k    │ │ 139k   │ │ 139k   │        │
│                               └────┬────┘ └────────┘ └────────┘        │
│                                    │                                     │
│                        ┌───────────┴───────────┐                        │
│                        ▼                       ▼                        │
│                  ┌──────────┐           ┌──────────┐                    │
│                  │  Scaler  │           │  SMOTE   │                    │
│                  │  fit()   │           │  50k per │                    │
│                  │ train    │           │  minority│                    │
│                  │ only     │           │  class   │                    │
│                  └────┬─────┘           └────┬─────┘                    │
│                       │                      │                          │
│                       ▼                      ▼                          │
│                  ┌─────────────────────────────────┐                    │
│                  │    Final Training Set: 789k     │                    │
│                  │    (scaled + oversampled)        │                    │
│                  └──────────────┬──────────────────┘                    │
│                                │                                        │
│                                ▼                                        │
│             ┌──────────────────────────────────────┐                    │
│             │         MODEL TRAINING                │                    │
│             │                                      │                    │
│             │  ┌────┐ ┌─────┐ ┌─────┐ ┌─────┐    │                    │
│             │  │ RF │ │ XGB │ │ MLP │ │CNN1D│    │                    │
│             │  └──┬─┘ └──┬──┘ └──┬──┘ └──┬──┘    │                    │
│             │     │      │       │       │        │                    │
│             │  ┌──┴──────┴───────┴───────┴──┐     │                    │
│             │  │    Evaluate on Test Set     │     │                    │
│             │  │    (Macro F1 comparison)    │     │                    │
│             │  └─────────────┬───────────────┘     │                    │
│             │                │                      │                    │
│             │  ┌─────┐  ┌───┴───┐                  │                    │
│             │  │ AE  │  │ Best  │                  │                    │
│             │  │(ben)│  │ Model │                  │                    │
│             │  └──┬──┘  └───┬───┘                  │                    │
│             │     │         │                      │                    │
│             │  ┌──┴──┐  ┌──┴───┐                  │                    │
│             │  │IsoF │  │ Save │                  │                    │
│             │  └─────┘  │ .pkl │                  │                    │
│             │           │ .pt  │                  │                    │
│             │           │ .json│                  │                    │
│             │           └──────┘                  │                    │
│             └──────────────────────────────────────┘                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Inference Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                         │
│                                                                  │
│  New CSV Upload                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  Validate│──▶│  Clean   │──▶│  Scale   │──▶│   Predict    │ │
│  │  Columns │   │  inf/NaN │   │  (saved  │   │   (selected  │ │
│  │          │   │          │   │  scaler) │   │    model)    │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬───────┘ │
│                                                       │         │
│                                              ┌────────┴───────┐ │
│                                              │                │ │
│                                              ▼                ▼ │
│                                        ┌──────────┐   ┌───────┐│
│                                        │Multi-cls │   │  AE   ││
│                                        │Predict   │   │Anomaly││
│                                        │Class +   │   │Score  ││
│                                        │Confidence│   │       ││
│                                        └────┬─────┘   └───┬───┘│
│                                             │             │    │
│                                             ▼             ▼    │
│                                    ┌────────────────────────┐  │
│                                    │     Output CSV         │  │
│                                    │  predicted_class       │  │
│                                    │  is_attack             │  │
│                                    │  confidence            │  │
│                                    │  anomaly_score         │  │
│                                    └────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 3. Model Architectures

### 3.1 Random Forest

```
                    ┌──────────┐
                    │ Input    │
                    │ 36 feat  │
                    └────┬─────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
     │ Tree 1  │   │ Tree 2  │...│Tree 150 │
     │depth≤25 │   │depth≤25 │   │depth≤25 │
     └────┬────┘   └────┬────┘   └────┬────┘
          │              │              │
          └──────────────┼──────────────┘
                         │
                    ┌────┴─────┐
                    │ Majority │
                    │   Vote   │
                    └────┬─────┘
                         │
                    ┌────┴─────┐
                    │ 7 Classes│
                    └──────────┘

  Config: n_estimators=150, max_depth=25, class_weight='balanced'
```

### 3.2 XGBoost

```
                    ┌──────────┐
                    │ Input    │
                    │ 36 feat  │
                    └────┬─────┘
                         │
                    ┌────┴─────┐
                    │ Tree 1   │──── predicts residuals
                    └────┬─────┘
                         │
                    ┌────┴─────┐
                    │ Tree 2   │──── corrects Tree 1's errors
                    └────┬─────┘
                         │
                        ...
                         │
                    ┌────┴─────┐
                    │ Tree N   │──── final correction
                    └────┬─────┘
                         │
                    ┌────┴─────┐
                    │   Sum    │
                    │ all tree │
                    │ outputs  │
                    └────┬─────┘
                         │
                    ┌────┴─────┐
                    │ Softmax  │
                    │ 7 Classes│
                    └──────────┘

  Tuned by Optuna (15 trials):
    n_estimators:     150-600
    max_depth:        4-12
    learning_rate:    0.02-0.3
    subsample:        0.7-1.0
    colsample_bytree: 0.7-1.0
    min_child_weight: 1-8
    reg_lambda:       0.001-10.0
```

### 3.3 MLP (Multi-Layer Perceptron)

```
  ┌──────────────────────────────────────────────┐
  │              MLP Architecture                 │
  │                                              │
  │  Input Layer        36 neurons (features)    │
  │       │                                      │
  │       ▼                                      │
  │  ┌─────────┐                                 │
  │  │Dense 256│─── ReLU ─── Dropout(0.3)       │
  │  └────┬────┘                                 │
  │       ▼                                      │
  │  ┌─────────┐                                 │
  │  │Dense 128│─── ReLU ─── Dropout(0.3)       │
  │  └────┬────┘                                 │
  │       ▼                                      │
  │  ┌─────────┐                                 │
  │  │Dense 64 │─── ReLU ─── Dropout(0.3)       │
  │  └────┬────┘                                 │
  │       ▼                                      │
  │  ┌─────────┐                                 │
  │  │Dense 7  │─── Softmax                     │
  │  └─────────┘                                 │
  │                                              │
  │  Loss: CrossEntropy (class-weighted)         │
  │  Optimizer: Adam (lr=0.001)                  │
  │  Early stopping: patience=5 on val macro-F1  │
  └──────────────────────────────────────────────┘
```

### 3.4 1D-CNN

```
  ┌──────────────────────────────────────────────┐
  │            1D-CNN Architecture                │
  │                                              │
  │  Input: 36 features                          │
  │       │                                      │
  │       ▼  reshape to (1, 36)                  │
  │                                              │
  │  ┌───────────────────────────────┐           │
  │  │ Conv1d(1→64, kernel=3, pad=1)│           │
  │  │ ReLU                          │           │
  │  │ BatchNorm(64)                 │           │
  │  └──────────────┬────────────────┘           │
  │                 ▼                             │
  │  ┌───────────────────────────────┐           │
  │  │Conv1d(64→128, kernel=3, pad=1)│          │
  │  │ ReLU                          │           │
  │  │ BatchNorm(128)                │           │
  │  └──────────────┬────────────────┘           │
  │                 ▼                             │
  │  ┌───────────────────────────────┐           │
  │  │ AdaptiveAvgPool1d(1)          │           │
  │  │ output: (128, 1) → flatten   │           │
  │  └──────────────┬────────────────┘           │
  │                 ▼                             │
  │  ┌─────────┐                                 │
  │  │Dense 64 │─── ReLU ─── Dropout(0.3)       │
  │  └────┬────┘                                 │
  │       ▼                                      │
  │  ┌─────────┐                                 │
  │  │Dense 7  │─── Softmax                     │
  │  └─────────┘                                 │
  │                                              │
  │  The 1D convolutions slide filters across    │
  │  the feature vector, learning local patterns │
  │  between adjacent features.                  │
  └──────────────────────────────────────────────┘
```

### 3.5 Autoencoder (Anomaly Detection)

```
  ┌──────────────────────────────────────────────────────────────┐
  │                 Autoencoder Architecture                      │
  │                                                              │
  │        ENCODER                          DECODER              │
  │                                                              │
  │  Input: 36 features                                          │
  │       │                                          │           │
  │       ▼                                          ▼           │
  │  ┌─────────┐                              ┌─────────┐       │
  │  │Dense 64 │── ReLU                       │Dense 64 │── out │
  │  └────┬────┘                              └────┬────┘       │
  │       ▼                                        ▲             │
  │  ┌─────────┐                              ┌────┴────┐       │
  │  │Dense 32 │── ReLU                       │Dense 32 │──ReLU │
  │  └────┬────┘                              └────┬────┘       │
  │       ▼                                        ▲             │
  │  ┌─────────┐                              ┌────┴────┐       │
  │  │Dense 16 │── ReLU ─────────────────────▶│Dense 16 │──ReLU │
  │  └─────────┘     (bottleneck)             └─────────┘       │
  │                                                              │
  │  Training: BENIGN flows only                                 │
  │  Loss: MSE(input, reconstruction)                            │
  │  Anomaly score = MSE per flow                                │
  │  Threshold = 99th percentile of training MSE                 │
  │                                                              │
  │  ┌─────────────────────────────────────────────┐             │
  │  │ If reconstruction_error > threshold:        │             │
  │  │    → Flow is ANOMALOUS (likely attack)      │             │
  │  │ Else:                                       │             │
  │  │    → Flow is NORMAL (benign)                │             │
  │  └─────────────────────────────────────────────┘             │
  └──────────────────────────────────────────────────────────────┘
```

### 3.6 Isolation Forest

```
  ┌──────────────────────────────────────────────┐
  │         Isolation Forest Architecture         │
  │                                              │
  │  Input: 36 features (NO labels)              │
  │       │                                      │
  │       ▼                                      │
  │  ┌──────────────────────────────────┐        │
  │  │  100 Isolation Trees             │        │
  │  │                                  │        │
  │  │  Each tree:                      │        │
  │  │  1. Pick random feature          │        │
  │  │  2. Pick random split value      │        │
  │  │  3. Partition data               │        │
  │  │  4. Repeat until isolated        │        │
  │  │                                  │        │
  │  │  Anomalies = fewer splits needed │        │
  │  │  Normal    = more splits needed  │        │
  │  └──────────────┬───────────────────┘        │
  │                 │                             │
  │                 ▼                             │
  │  ┌──────────────────────────────────┐        │
  │  │  Anomaly Score                   │        │
  │  │  = avg path length across trees  │        │
  │  │                                  │        │
  │  │  Short path → anomaly (attack)   │        │
  │  │  Long path  → normal (benign)    │        │
  │  └──────────────────────────────────┘        │
  │                                              │
  │  contamination = observed attack ratio       │
  └──────────────────────────────────────────────┘
```

## 4. Dashboard Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                      SIDEBAR                             │    │
│  │  Navigation (7 pages)                                    │    │
│  │  Dark Mode Toggle                                        │    │
│  │  Model Selector (XGB / RF / MLP / CNN1D)                │    │
│  │  Quick Stats (model, F1, flows, threats)                 │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                     MAIN AREA                            │    │
│  │                                                          │    │
│  │  Page 1: Home ─────── Overview + Leaderboard             │    │
│  │  Page 2: Upload ───── CSV upload + Demo buttons          │    │
│  │  Page 3: Threats ──── Filtered table + PDF export        │    │
│  │  Page 4: Analytics ── Charts + Timeline + Anomaly        │    │
│  │  Page 5: Models ───── Comparison + Detail viewer         │    │
│  │  Page 6: Features ─── Importance + Correlation           │    │
│  │  Page 7: About ────── Architecture + Tech stack          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                   DATA LAYER                             │    │
│  │                                                          │    │
│  │  @st.cache_resource:                                     │    │
│  │    config.yaml ─── loaded once                           │    │
│  │    best_model.json ─── model selection                   │    │
│  │    metrics.json ─── all model scores                     │    │
│  │    selected_features.json ─── feature list               │    │
│  │                                                          │    │
│  │  st.session_state:                                       │    │
│  │    preds ─── current prediction DataFrame                │    │
│  │    dark_mode ─── theme toggle                            │    │
│  │    active_model ─── selected model key                   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  ARTIFACT LAYER                          │    │
│  │                                                          │    │
│  │  artifacts/                                              │    │
│  │    scaler.pkl ──────── StandardScaler                    │    │
│  │    label_encoder.pkl ─ LabelEncoder                      │    │
│  │    rf_multi.pkl ────── Random Forest model               │    │
│  │    xgb_multi.json ──── XGBoost model                    │    │
│  │    mlp_multi.pt ────── MLP weights                      │    │
│  │    cnn1d_multi.pt ──── CNN1D weights                    │    │
│  │    autoencoder.pt ──── AE weights + threshold           │    │
│  │    iforest.pkl ─────── Isolation Forest model            │    │
│  │                                                          │    │
│  │  reports/figures/                                        │    │
│  │    cm_*.png ────────── Confusion matrices                │    │
│  │    roc_*.png ───────── ROC curves                       │    │
│  │    pr_*.png ────────── Precision-Recall curves           │    │
│  │    featimp_*.png ───── Feature importance charts         │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 5. File Architecture

```
cyber-ai-nids/
│
├── main.py                          # CLI entrypoint
│     └── argparse: train | evaluate | predict
│           │
│           ├── train  ──▶ src/train.py ──▶ runs all models
│           ├── evaluate ──▶ loads artifacts/metrics.json
│           └── predict ──▶ src/predict.py ──▶ inference
│
├── config.yaml                      # Single source of truth
│     ├── seed: 42
│     ├── paths (raw, processed, artifacts, figures)
│     ├── data (split sizes, correlation threshold, benign cap)
│     ├── smote (target per minority, k_neighbors)
│     └── models (RF, XGB, MLP, CNN1D, AE, IsoForest params)
│
├── src/
│   ├── data_loader.py               # load_short_csv() / load_full_zip()
│   │     └── called by train.py
│   │
│   ├── preprocessing.py             # clean → label → split → scale → SMOTE
│   │     ├── clean_frame()
│   │     ├── add_label_columns()
│   │     ├── drop_correlated()
│   │     ├── build_splits() ──▶ returns Splits dataclass
│   │     └── save_artifacts()
│   │
│   ├── train.py                     # Orchestrator
│   │     ├── loads data via data_loader
│   │     ├── preprocesses via preprocessing
│   │     ├── trains: RF → XGB → MLP → CNN1D → AE → IsoForest
│   │     ├── evaluates each via evaluate.py
│   │     ├── picks best model
│   │     └── saves everything to artifacts/
│   │
│   ├── evaluate.py                  # Metrics + Figures
│   │     ├── classification_report()
│   │     ├── plot_confusion_matrix()
│   │     ├── plot_roc_multiclass()
│   │     ├── plot_pr_multiclass()
│   │     └── plot_feature_importance()
│   │
│   ├── predict.py                   # Inference
│   │     ├── loads saved model + scaler + encoder
│   │     ├── accepts model_override for dashboard switching
│   │     ├── runs multi-class prediction
│   │     ├── adds autoencoder anomaly scores
│   │     └── outputs CSV
│   │
│   ├── utils.py                     # Shared utilities
│   │     ├── load_config(), set_seed()
│   │     ├── get_logger()
│   │     └── save/load pickle/json
│   │
│   └── models/
│       ├── random_forest.py         # build_rf()
│       ├── xgboost_model.py         # tune_xgboost() + build_default_xgboost()
│       ├── mlp.py                   # MLP class + train_mlp() + mlp_predict()
│       ├── cnn1d.py                 # CNN1D class + train_cnn1d() + cnn1d_predict()
│       ├── autoencoder.py           # Autoencoder class + train + ae_predict_binary()
│       └── isolation_forest.py      # build_iforest() + iforest_predict_binary()
│
├── dashboard/
│   └── app.py                       # Streamlit (7 pages + sidebar)
│         ├── page_home()
│         ├── page_upload()          # demo buttons + file upload + progress bar
│         ├── page_threat_report()   # filters + table + PDF export
│         ├── page_analytics()       # charts + timeline + anomaly
│         ├── page_model_comparison()# leaderboard + detail viewer
│         ├── page_feature_analysis()# importance + categories
│         └── page_about()           # architecture + tech stack
│
├── tests/
│   └── test_pipeline.py             # pytest smoke test
│
└── docs/
    ├── PROJECT_REPORT.md
    ├── METHODOLOGY.md
    ├── RESULTS_SUMMARY.md
    ├── HOW_TO_USE.md
    ├── INSTALLATION_GUIDE.md
    └── ARCHITECTURE.md              # (this document)
```

## 6. Deployment Architecture

### 6.1 Local Development

```
  ┌────────────────────────────┐
  │       Developer Machine    │
  │                            │
  │  Terminal 1:               │
  │    python main.py train    │
  │                            │
  │  Terminal 2:               │
  │    streamlit run app.py    │
  │         │                  │
  │         ▼                  │
  │    localhost:8501           │
  │         │                  │
  │         ▼                  │
  │    Browser (Chrome)        │
  └────────────────────────────┘
```

### 6.2 AWS Production

```
  ┌─────────────┐         ┌──────────────────────────┐
  │   User      │         │   AWS t3.large (8GB)     │
  │   Browser   │────────▶│                          │
  │             │  HTTPS  │   ┌──────────────────┐   │
  └─────────────┘    :8501│   │  Streamlit       │   │
                          │   │  dashboard/app.py│   │
                          │   └────────┬─────────┘   │
                          │            │              │
                          │            ▼              │
                          │   ┌──────────────────┐   │
                          │   │  artifacts/       │   │
                          │   │  - xgb_multi.json│   │
                          │   │  - scaler.pkl    │   │
                          │   │  - encoder.pkl   │   │
                          │   │  - autoencoder.pt│   │
                          │   └──────────────────┘   │
                          │                          │
                          │   RAM usage: ~1.5 GB     │
                          └──────────────────────────┘
```

## 7. Security Considerations

```
  ┌──────────────────────────────────────────────┐
  │           Data Leakage Prevention             │
  │                                              │
  │  ✓ Scaler fit on TRAINING data only          │
  │  ✓ SMOTE applied to TRAINING set only        │
  │  ✓ Optuna tunes on VALIDATION set only       │
  │  ✓ Test set used ONCE for final evaluation   │
  │  ✓ No future data used for predictions       │
  │  ✓ Random seed fixed at 42 for all splits    │
  │                                              │
  │           Input Validation                    │
  │                                              │
  │  ✓ CSV column schema validated on upload     │
  │  ✓ inf/NaN values cleaned before inference   │
  │  ✓ Missing features raise clear error        │
  │  ✓ No arbitrary code execution from CSV      │
  └──────────────────────────────────────────────┘
```
