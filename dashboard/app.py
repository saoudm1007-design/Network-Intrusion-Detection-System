"""Streamlit dashboard for the CIC-IDS2017 NIDS — Professional Edition."""
from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import predict_csv  # noqa: E402
from src.utils import load_config, load_json, load_pickle  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CFG_PATH = ROOT / "config.yaml"
FIG_DIR = ROOT / "reports" / "figures"

# ── Dark mode state ─────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

dark = st.session_state["dark_mode"]

# ── Custom CSS ───────────────────────────────────────────────────────────────
if dark:
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #e0e0e0; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a12 0%, #12121f 100%); }
        [data-testid="stSidebar"] * { color: #c0c0c0 !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] { background-color: #ffffff !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] * { color: #000000 !important; background-color: transparent !important; }
        [data-testid="stSidebar"] [data-baseweb="popover"] * { background-color: #ffffff !important; color: #000000 !important; }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div { background-color: #ffffff !important; }
        [data-testid="stMetric"] { background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 10px; padding: 15px; }
        [data-testid="stMetric"] label { color: #a0a0c0 !important; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; }
        .main-header { font-size: 2.2rem; font-weight: 700; color: #e0e0f0; margin-bottom: 0.5rem; }
        .sub-header { font-size: 1.1rem; color: #8888aa; margin-bottom: 2rem; }
        .severity-critical { background: #dc3545; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .severity-high { background: #fd7e14; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .severity-medium { background: #ffc107; color: #333; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .severity-safe { background: #28a745; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp li { color: #e0e0e0 !important; }
        .stApp .stMarkdown, .stApp .stText { color: #e0e0e0 !important; }
        [data-baseweb="menu"] * { color: #000000 !important; background-color: #ffffff !important; }
        [data-baseweb="listbox"] * { color: #000000 !important; }
        [data-testid="stToolbar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }
        [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] { background-color: #ffffff !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] * { color: #000000 !important; background-color: transparent !important; }
        [data-testid="stSidebar"] [data-baseweb="popover"] * { background-color: #ffffff !important; color: #000000 !important; }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div { background-color: #ffffff !important; }
        [data-testid="stMetric"] { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.5rem; }
        .sub-header { font-size: 1.1rem; color: #6c757d; margin-bottom: 2rem; }
        .severity-critical { background: #dc3545; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .severity-high { background: #fd7e14; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .severity-medium { background: #ffc107; color: #333; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .severity-safe { background: #28a745; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────
SEVERITY_MAP = {
    "DDoS": "Critical", "DoS": "Critical",
    "Bot": "High", "BruteForce": "High", "WebAttack": "High",
    "PortScan": "Medium", "Benign": "Safe",
}
SEVERITY_COLORS = {"Critical": "#dc3545", "High": "#fd7e14", "Medium": "#ffc107", "Safe": "#28a745"}
ATTACK_DESCRIPTIONS = {
    "DDoS": "Distributed Denial of Service — overwhelming the target with traffic from multiple sources.",
    "DoS": "Denial of Service — flooding a target to exhaust resources and deny legitimate access.",
    "Bot": "Botnet activity — compromised host communicating with command-and-control infrastructure.",
    "BruteForce": "Brute Force — systematic attempt to guess credentials (FTP/SSH).",
    "WebAttack": "Web Application Attack — SQL injection, XSS, or brute-force against web services.",
    "PortScan": "Port Scanning — reconnaissance to discover open ports and running services.",
    "Benign": "Normal, legitimate network traffic.",
}
MODEL_DISPLAY = {
    "rf_multi": "Random Forest (Multi-class)", "rf_binary": "Random Forest (Binary)",
    "xgb_multi": "XGBoost (Multi-class)", "xgb_binary": "XGBoost (Binary)",
    "mlp_multi": "MLP Neural Network", "cnn1d_multi": "1D-CNN",
    "autoencoder_binary": "Autoencoder (Anomaly)", "iforest_binary": "Isolation Forest",
}
MODEL_TYPE = {
    "rf_multi": "Ensemble", "rf_binary": "Ensemble",
    "xgb_multi": "Ensemble", "xgb_binary": "Ensemble",
    "mlp_multi": "Deep Learning", "cnn1d_multi": "Deep Learning",
    "autoencoder_binary": "Deep Learning", "iforest_binary": "Unsupervised",
}

# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def _cfg() -> dict:
    return load_config(CFG_PATH)

@st.cache_resource
def _best_model() -> dict:
    art = Path(_cfg()["paths"]["artifacts_dir"])
    p = art / "best_model.json"
    return load_json(p) if p.exists() else {}

@st.cache_resource
def _metrics() -> dict:
    art = Path(_cfg()["paths"]["artifacts_dir"])
    p = art / "metrics.json"
    return load_json(p) if p.exists() else {}

@st.cache_resource
def _feature_names() -> list[str]:
    art = Path(_cfg()["paths"]["artifacts_dir"])
    p = art / "selected_features.json"
    return load_json(p)["features"] if p.exists() else []

def _row_color(cls: str) -> str:
    if cls == "Benign":
        return "background-color: rgba(40, 167, 69, 0.08)"
    sev = SEVERITY_MAP.get(cls, "Medium")
    colors = {"Critical": "rgba(220, 53, 69, 0.12)", "High": "rgba(253, 126, 20, 0.10)", "Medium": "rgba(255, 193, 7, 0.10)"}
    return f"background-color: {colors.get(sev, 'rgba(255,193,7,0.10)')}"

MULTI_MODELS = {
    "xgb_multi": "XGBoost (Multi-class)",
    "rf_multi": "Random Forest (Multi-class)",
    "mlp_multi": "MLP Neural Network",
    "cnn1d_multi": "1D-CNN",
}

def _selected_model() -> str:
    return st.session_state.get("active_model", "xgb_multi")

def _load_demo(path: Path) -> None:
    """Load a demo CSV and run predictions."""
    out = ROOT / "artifacts" / "_predictions.csv"
    df = predict_csv(_cfg(), str(path), str(out), model_override=_selected_model())
    st.session_state["preds"] = df

# ── PDF Generator ────────────────────────────────────────────────────────────
def _generate_pdf(df: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "CyberShield NIDS - Threat Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Summary
    total = len(df)
    attacks = df[df["is_attack"] == 1]
    benign = df[df["is_attack"] == 0]
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Total flows analyzed: {total:,}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Benign flows: {len(benign):,} ({len(benign)/max(1,total)*100:.1f}%)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Attacks detected: {len(attacks):,} ({len(attacks)/max(1,total)*100:.1f}%)", new_x="LMARGIN", new_y="NEXT")
    if "confidence" in df.columns:
        pdf.cell(0, 7, f"Average confidence: {df['confidence'].mean():.1%}", new_x="LMARGIN", new_y="NEXT")
    best = _best_model()
    if best:
        pdf.cell(0, 7, f"Model used: {best.get('best_model', 'N/A')} (Macro-F1: {best.get('macro_f1', 0):.4f})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Severity breakdown
    df_sev = df.copy()
    df_sev["severity"] = df_sev["predicted_class"].map(SEVERITY_MAP)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Severity Breakdown", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for sev in ["Critical", "High", "Medium", "Safe"]:
        cnt = len(df_sev[df_sev["severity"] == sev])
        if cnt > 0:
            types = ", ".join(sorted(df_sev[df_sev["severity"] == sev]["predicted_class"].unique()))
            pdf.cell(0, 7, f"  {sev}: {cnt:,} flows ({types})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Attack class counts
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Attack Type Distribution", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for cls in sorted(df["predicted_class"].unique()):
        cnt = len(df[df["predicted_class"] == cls])
        pdf.cell(0, 7, f"  {cls}: {cnt:,}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Top threats table
    if len(attacks) > 0:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Top 20 Detected Threats", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 9)
        col_w = [10, 30, 20, 25, 25]
        headers = ["#", "Class", "Severity", "Confidence", "Anomaly Score"]
        for w, h in zip(col_w, headers):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        top = attacks.head(20)
        for i, (_, row) in enumerate(top.iterrows()):
            pdf.cell(col_w[0], 6, str(i + 1), border=1)
            pdf.cell(col_w[1], 6, str(row.get("predicted_class", "")), border=1)
            pdf.cell(col_w[2], 6, SEVERITY_MAP.get(str(row.get("predicted_class", "")), ""), border=1)
            pdf.cell(col_w[3], 6, f"{row.get('confidence', 0):.4f}" if "confidence" in row.index else "", border=1)
            pdf.cell(col_w[4], 6, f"{row.get('anomaly_score', 0):.4f}" if "anomaly_score" in row.index else "", border=1)
            pdf.ln()

    return pdf.output()

# ── PAGES ────────────────────────────────────────────────────────────────────

def page_home():
    st.markdown('<div class="main-header">🛡️ CyberShield NIDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning-based Network Intrusion Detection System — CIC-IDS2017</div>', unsafe_allow_html=True)

    best = _best_model()
    metrics = _metrics()
    features = _feature_names()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Models Trained", len(metrics))
    with c2:
        st.metric("Best Macro-F1", f"{best.get('macro_f1', 0):.4f}" if best else "N/A")
    with c3:
        st.metric("Production Model", best.get("best_model", "N/A").replace("_", " ").title())
    with c4:
        st.metric("Features Used", len(features))

    st.divider()

    left, right = st.columns([2, 1])
    with left:
        st.subheader("About This System")
        st.markdown("""
        This NIDS analyzes network flow data from the **CIC-IDS2017** dataset and classifies
        traffic into **7 categories**: Benign (normal) and 6 attack types.

        **How it works:**
        1. Upload a CSV of network flow statistics
        2. The system scales features and runs inference through the best-trained model
        3. Each flow gets a predicted class, confidence score, and anomaly score
        4. Results are presented as an actionable Threat Report

        **Attack types detected:**
        """)
        for attack, desc in ATTACK_DESCRIPTIONS.items():
            if attack != "Benign":
                sev = SEVERITY_MAP[attack]
                st.markdown(f"- **{attack}** ({sev}) — {desc}")

    with right:
        st.subheader("Model Leaderboard")
        if metrics:
            rows = []
            for k, v in sorted(metrics.items(), key=lambda x: -x[1]["macro_f1"]):
                rows.append({
                    "Model": MODEL_DISPLAY.get(k, k),
                    "Type": MODEL_TYPE.get(k, "—"),
                    "Macro-F1": f"{v['macro_f1']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Dataset Summary")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Dataset:** CIC-IDS2017")
        st.markdown("**Total flows:** ~2.5 million")
    with d2:
        st.markdown("**Training set:** ~789k flows (after SMOTE)")
        st.markdown("**Test set:** ~139k flows")
    with d3:
        st.markdown("**Imbalance handling:** SMOTE + class weighting")
        st.markdown("**Optimization metric:** Macro F1-Score")


def page_upload():
    st.markdown('<div class="main-header">📤 Upload & Analyze</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CIC-IDS2017-formatted CSV to detect network intrusions</div>', unsafe_allow_html=True)

    # ── Feature #5: Sample data buttons ──
    st.subheader("Quick Demo")
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    sample_path = ROOT / "dashboard" / "examples" / "sample.csv"
    short_path = ROOT / "data" / "raw" / "cicids2017.csv"

    with demo_col1:
        if st.button("🚀 Load sample (49 flows)", use_container_width=True):
            if sample_path.exists():
                with st.spinner("Loading sample data..."):
                    _load_demo(sample_path)
                st.success("Sample data loaded! Navigate to Threat Report or Analytics.")
            else:
                st.error("Sample file not found.")
    with demo_col2:
        if st.button("📊 Load short dataset (10k flows)", use_container_width=True):
            if short_path.exists():
                with st.spinner("Loading 10k flows..."):
                    _load_demo(short_path)
                st.success("10k flows loaded! Navigate to Threat Report or Analytics.")
            else:
                st.error("Short dataset not found.")
    with demo_col3:
        st.markdown("Or upload your own CSV below ↓")

    st.divider()

    # File uploader
    up = st.file_uploader("Select a network flow CSV file", type="csv",
                           help="The CSV must contain the same feature columns as CIC-IDS2017")

    if up is None and st.session_state.get("preds") is not None:
        df = st.session_state["preds"]
        st.info(f"Currently loaded: **{len(df):,} flows** ({(df['is_attack']==1).sum():,} attacks). Navigate to other pages to explore.")
        return
    elif up is None:
        st.markdown("---")
        st.subheader("Expected CSV Format")
        st.markdown("The uploaded file should contain these network flow features:")
        features = _feature_names()
        if features:
            cols = st.columns(4)
            for i, feat in enumerate(features):
                cols[i % 4].markdown(f"- `{feat}`")
        return

    tmp = ROOT / "artifacts" / "_uploaded.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(up.read())

    # ── Feature #2: Progress bar while scoring ──
    progress = st.progress(0, text="Preparing analysis...")
    progress.progress(10, text="Loading model and scaler...")
    time.sleep(0.3)
    progress.progress(30, text="Scaling features...")
    time.sleep(0.2)
    progress.progress(50, text="Running model inference...")

    out = ROOT / "artifacts" / "_predictions.csv"
    try:
        df = predict_csv(_cfg(), str(tmp), str(out), model_override=_selected_model())
    except Exception as e:
        progress.empty()
        st.error(f"Prediction failed: {e}")
        return

    progress.progress(80, text="Generating anomaly scores...")
    time.sleep(0.2)
    progress.progress(95, text="Building results...")
    time.sleep(0.2)
    progress.progress(100, text=f"Done — {len(df):,} flows analyzed!")
    time.sleep(0.5)
    progress.empty()

    attacks = df[df["is_attack"] == 1]
    benign = df[df["is_attack"] == 0]

    st.success(f"Analysis complete — **{len(df):,} flows** scored, **{len(attacks):,} threats** detected.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Flows", f"{len(df):,}")
    with c2:
        st.metric("Benign", f"{len(benign):,}", delta=f"{len(benign)/max(1,len(df))*100:.1f}%")
    with c3:
        st.metric("Attacks Detected", f"{len(attacks):,}",
                   delta=f"{len(attacks)/max(1,len(df))*100:.1f}%", delta_color="inverse")
    with c4:
        avg_conf = df["confidence"].mean() if "confidence" in df.columns else 0
        st.metric("Avg Confidence", f"{avg_conf:.1%}")

    st.session_state["preds"] = df
    st.info("Navigate to **Threat Report**, **Analytics**, or **Summary** to explore the results.")


def page_threat_report():
    st.markdown('<div class="main-header">🚨 Threat Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed per-flow classification results with severity levels</div>', unsafe_allow_html=True)

    df = st.session_state.get("preds")
    if df is None:
        st.warning("No data loaded. Go to **Upload & Analyze** first.")
        return

    # Filters
    st.subheader("Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        show_only = st.multiselect("Filter by class", options=sorted(df["predicted_class"].unique()),
                                    default=sorted(df["predicted_class"].unique()))
    with f2:
        severity_filter = st.multiselect("Filter by severity",
                                          options=["Critical", "High", "Medium", "Safe"],
                                          default=["Critical", "High", "Medium", "Safe"])
    with f3:
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)

    filtered = df.copy()
    filtered["severity"] = filtered["predicted_class"].map(SEVERITY_MAP)
    filtered = filtered[filtered["predicted_class"].isin(show_only)]
    filtered = filtered[filtered["severity"].isin(severity_filter)]
    if "confidence" in filtered.columns:
        filtered = filtered[filtered["confidence"] >= min_conf]

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    sev_counts = filtered["severity"].value_counts()
    with c1:
        st.metric("Showing", f"{len(filtered):,} / {len(df):,}")
    with c2:
        st.metric("🔴 Critical", sev_counts.get("Critical", 0))
    with c3:
        st.metric("🟠 High", sev_counts.get("High", 0))
    with c4:
        st.metric("🟡 Medium", sev_counts.get("Medium", 0))
    with c5:
        st.metric("🟢 Safe", sev_counts.get("Safe", 0))

    display_cols = ["predicted_class", "severity", "is_attack", "confidence"]
    if "anomaly_score" in filtered.columns:
        display_cols.append("anomaly_score")
    if "true_label" in filtered.columns:
        display_cols.insert(0, "true_label")

    display = filtered[display_cols].copy()

    def _style_row(row):
        color = _row_color(row["predicted_class"])
        return [color] * len(row)

    st.dataframe(
        display.style.apply(_style_row, axis=1).format(
            {c: "{:.4f}" for c in ["confidence", "anomaly_score"] if c in display.columns}
        ),
        width="stretch", height=500,
    )

    # Downloads
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button("📥 Download full CSV",
                           df.to_csv(index=False).encode(),
                           "nids_predictions_full.csv", "text/csv")
    with col_dl2:
        attacks_only = df[df["is_attack"] == 1]
        st.download_button("📥 Download attacks only",
                           attacks_only.to_csv(index=False).encode(),
                           "nids_attacks_only.csv", "text/csv")
    # ── Feature #1: PDF export ──
    with col_dl3:
        pdf_bytes = _generate_pdf(df)
        st.download_button("📄 Export Threat Report PDF",
                           bytes(pdf_bytes),
                           "CyberShield_Threat_Report.pdf",
                           "application/pdf")


def page_analytics():
    st.markdown('<div class="main-header">📊 Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visual analysis of the uploaded network traffic</div>', unsafe_allow_html=True)

    df = st.session_state.get("preds")
    if df is None:
        st.warning("No data loaded. Go to **Upload & Analyze** first.")
        return

    # Row 1: Distribution charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Traffic Composition")
        counts = df["predicted_class"].value_counts()
        colors = [SEVERITY_COLORS.get(SEVERITY_MAP.get(c, "Safe"), "#6c757d") for c in counts.index]
        fig, ax = plt.subplots(figsize=(6, 6))
        threshold = 3.0
        total = counts.values.sum()
        pcts = counts.values / total * 100
        labels_display = [n if p >= threshold else "" for n, p in zip(counts.index, pcts)]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=labels_display,
            autopct=lambda p: f"{p:.1f}%" if p >= threshold else "",
            colors=colors, startangle=90, pctdistance=0.82,
            labeldistance=1.12,
        )
        for t in texts:
            t.set_fontsize(8)
        for t in autotexts:
            t.set_fontsize(8)
        centre = plt.Circle((0, 0), 0.60, fc="white")
        ax.add_artist(centre)
        small = [(n, p) for n, p in zip(counts.index, pcts) if p < threshold and p > 0]
        if small:
            legend_labels = [f"{n} ({p:.1f}%)" for n, p in small]
            ax.legend(legend_labels, loc="lower left", fontsize=7, framealpha=0.8)
        ax.set_title("Predicted Class Distribution", fontweight="bold", pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Attack Type Breakdown")
        attacks = df[df["predicted_class"] != "Benign"]
        if len(attacks) > 0:
            att_counts = attacks["predicted_class"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            bars = ax.barh(att_counts.index, att_counts.values,
                           color=[SEVERITY_COLORS.get(SEVERITY_MAP.get(c, "Medium"), "#ffc107") for c in att_counts.index])
            ax.set_xlabel("Number of Flows")
            ax.set_title("Attack Types Detected", fontweight="bold")
            for bar, val in zip(bars, att_counts.values):
                ax.text(bar.get_width() + max(att_counts.values) * 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:,}", va="center", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No attacks detected in this upload.")

    st.divider()

    # ── Feature #3: Attack severity timeline ──
    st.subheader("Attack Severity Timeline")
    st.caption("Each dot is a flow, plotted by row index. Red = attack, green = benign. Simulates a real-time SOC monitor.")
    df_plot = df.copy()
    df_plot["flow_index"] = range(len(df_plot))
    df_plot["severity"] = df_plot["predicted_class"].map(SEVERITY_MAP)
    sev_order = {"Critical": 4, "High": 3, "Medium": 2, "Safe": 1}
    df_plot["sev_num"] = df_plot["severity"].map(sev_order)
    df_plot["color"] = df_plot["severity"].map(SEVERITY_COLORS)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    for sev_name, sev_val in sev_order.items():
        subset = df_plot[df_plot["severity"] == sev_name]
        if len(subset) > 0:
            ax.scatter(subset["flow_index"], subset["sev_num"],
                       c=SEVERITY_COLORS[sev_name], s=8, alpha=0.6, label=sev_name)
    ax.set_yticks(list(sev_order.values()))
    ax.set_yticklabels(list(sev_order.keys()))
    ax.set_xlabel("Flow Index (time-ordered)")
    ax.set_ylabel("Severity")
    ax.set_title("Threat Severity Over Flow Sequence", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Row 2: Confidence & Anomaly distributions
    col3, col4 = st.columns(2)
    with col3:
        if "confidence" in df.columns:
            st.subheader("Confidence Score Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            for cls in df["predicted_class"].unique():
                subset = df[df["predicted_class"] == cls]["confidence"]
                ax.hist(subset, bins=30, alpha=0.5, label=cls)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Flow Count")
            ax.set_title("Model Confidence by Predicted Class", fontweight="bold")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col4:
        if "anomaly_score" in df.columns:
            st.subheader("Anomaly Score Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            benign_scores = df[df["is_attack"] == 0]["anomaly_score"]
            attack_scores = df[df["is_attack"] == 1]["anomaly_score"]
            if len(benign_scores) > 0:
                ax.hist(benign_scores, bins=50, alpha=0.5, label="Benign", color="#28a745")
            if len(attack_scores) > 0:
                ax.hist(attack_scores, bins=50, alpha=0.5, label="Attack", color="#dc3545")
            ax.set_xlabel("Reconstruction Error")
            ax.set_ylabel("Flow Count")
            ax.set_title("Autoencoder Anomaly Scores", fontweight="bold")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    # Severity summary table
    st.subheader("Severity Summary")
    df_sev = df.copy()
    df_sev["severity"] = df_sev["predicted_class"].map(SEVERITY_MAP)
    summary_rows = []
    for sev in ["Critical", "High", "Medium", "Safe"]:
        subset = df_sev[df_sev["severity"] == sev]
        if len(subset) > 0:
            summary_rows.append({
                "Severity": sev,
                "Flow Count": len(subset),
                "% of Total": f"{len(subset)/len(df)*100:.1f}%",
                "Avg Confidence": f"{subset['confidence'].mean():.3f}" if "confidence" in subset.columns else "—",
                "Attack Types": ", ".join(sorted(subset["predicted_class"].unique())),
            })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    if "anomaly_score" in df.columns:
        st.divider()
        st.subheader("🔍 Top 10 Most Anomalous Flows")
        st.caption("Ranked by Autoencoder reconstruction error — higher = more unusual")
        top = df.sort_values("anomaly_score", ascending=False).head(10)
        st.dataframe(top, width="stretch", hide_index=True)


def page_model_comparison():
    st.markdown('<div class="main-header">🤖 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Side-by-side performance of all trained models</div>', unsafe_allow_html=True)

    metrics = _metrics()
    best = _best_model()
    if not metrics:
        st.warning("No trained models found. Run `python main.py train` first.")
        return

    st.subheader("Performance Leaderboard")
    rows = []
    for k, v in sorted(metrics.items(), key=lambda x: -x[1]["macro_f1"]):
        is_best = k == best.get("best_model", "")
        rows.append({
            "Rank": "", "Model": MODEL_DISPLAY.get(k, k),
            "Type": MODEL_TYPE.get(k, "—"), "Macro-F1": v["macro_f1"],
            "Production": "✅" if is_best else "",
        })
    for i, r in enumerate(rows):
        r["Rank"] = f"#{i+1}"
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()

    st.subheader("Macro-F1 Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [MODEL_DISPLAY.get(k, k) for k in sorted(metrics.keys(), key=lambda x: -metrics[x]["macro_f1"])]
    scores = [metrics[k]["macro_f1"] for k in sorted(metrics.keys(), key=lambda x: -metrics[x]["macro_f1"])]
    colors = ["#28a745" if s >= 0.95 else "#ffc107" if s >= 0.80 else "#dc3545" for s in scores]
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro F1-Score")
    ax.set_title("All Models — Macro F1 on Test Set", fontweight="bold")
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=10)
    ax.axvline(x=0.95, color="#28a745", linestyle="--", alpha=0.4, label="Excellent (0.95)")
    ax.axvline(x=0.80, color="#ffc107", linestyle="--", alpha=0.4, label="Good (0.80)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Model Detail Viewer")
    model_keys = list(metrics.keys())
    selected = st.selectbox("Select a model to inspect", model_keys,
                             format_func=lambda k: MODEL_DISPLAY.get(k, k))

    if selected:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Classification Report — {MODEL_DISPLAY.get(selected, selected)}**")
            report = metrics[selected].get("report", {})
            if report:
                report_rows = []
                for cls, vals in report.items():
                    if cls in ("accuracy", "macro avg", "weighted avg"):
                        continue
                    if isinstance(vals, dict):
                        report_rows.append({
                            "Class": cls,
                            "Precision": f"{vals.get('precision', 0):.3f}",
                            "Recall": f"{vals.get('recall', 0):.3f}",
                            "F1-Score": f"{vals.get('f1-score', 0):.3f}",
                            "Support": int(vals.get("support", 0)),
                        })
                if report_rows:
                    st.dataframe(pd.DataFrame(report_rows), width="stretch", hide_index=True)
                for avg_key in ("macro avg", "weighted avg"):
                    if avg_key in report and isinstance(report[avg_key], dict):
                        vals = report[avg_key]
                        st.caption(f"**{avg_key}**: P={vals.get('precision',0):.3f}  R={vals.get('recall',0):.3f}  F1={vals.get('f1-score',0):.3f}")
        with col2:
            cm_path = FIG_DIR / f"cm_{selected}_norm.png"
            if cm_path.exists():
                st.markdown("**Normalized Confusion Matrix**")
                st.image(str(cm_path), width="stretch")

        col3, col4 = st.columns(2)
        with col3:
            for img_prefix, label in [("roc_", "ROC Curves (One-vs-Rest)"), ("pr_", "Precision-Recall Curves")]:
                p = FIG_DIR / f"{img_prefix}{selected}.png"
                if p.exists():
                    st.markdown(f"**{label}**")
                    st.image(str(p), width="stretch")
        with col4:
            fi_path = FIG_DIR / f"featimp_{selected}.png"
            if fi_path.exists():
                st.markdown("**Feature Importance**")
                st.image(str(fi_path), width="stretch")
            cm_raw = FIG_DIR / f"cm_{selected}.png"
            if cm_raw.exists():
                st.markdown("**Raw Confusion Matrix**")
                st.image(str(cm_raw), width="stretch")


def page_feature_analysis():
    st.markdown('<div class="main-header">🔬 Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding which network features drive detection</div>', unsafe_allow_html=True)

    features = _feature_names()
    if not features:
        st.warning("No feature list found.")
        return

    st.subheader(f"Selected Features ({len(features)})")
    st.caption("After dropping highly correlated features (>0.95 Pearson), these are the network flow attributes used by all models:")
    col1, col2, col3 = st.columns(3)
    for i, feat in enumerate(features):
        [col1, col2, col3][i % 3].markdown(f"**{i+1}.** `{feat}`")

    st.divider()

    st.subheader("Feature Importance — XGBoost (Production Model)")
    fi_path = FIG_DIR / "featimp_xgb_multi.png"
    if fi_path.exists():
        st.image(str(fi_path), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance — Random Forest")
        fi_rf = FIG_DIR / "featimp_rf_multi.png"
        if fi_rf.exists():
            st.image(str(fi_rf), width="stretch")
    with col2:
        st.subheader("Correlation Heatmap")
        hm = FIG_DIR / "corr_heatmap.png"
        if hm.exists():
            st.image(str(hm), width="stretch")
        else:
            st.info("Run the EDA notebook to generate the correlation heatmap.")

    st.divider()
    st.subheader("Feature Category Breakdown")
    categories = {
        "Packet Length Stats": [f for f in features if "Packet Length" in f or "Packet Size" in f],
        "Flow Timing (IAT)": [f for f in features if "IAT" in f or "Duration" in f],
        "Packet Counts": [f for f in features if "Packets" in f and "Length" not in f],
        "TCP Flags": [f for f in features if "Flag" in f],
        "Header / Window": [f for f in features if "Header" in f or "Win" in f or "seg_size" in f],
        "Activity / Idle": [f for f in features if "Active" in f or "Idle" in f],
        "Other": [],
    }
    categorized = set()
    for cat_feats in categories.values():
        categorized.update(cat_feats)
    categories["Other"] = [f for f in features if f not in categorized]
    for cat, feats in categories.items():
        if feats:
            st.markdown(f"**{cat}** ({len(feats)}): {', '.join(f'`{f}`' for f in feats)}")


def page_about():
    st.markdown('<div class="main-header">ℹ️ About CyberShield NIDS</div>', unsafe_allow_html=True)

    st.subheader("Project")
    st.markdown("""
    **Course:** Artificial Intelligence and Cyber Security — Spring 2026

    **Objective:** Build a Machine Learning-based Network Intrusion Detection System (NIDS)
    that analyzes network flow data from the CIC-IDS2017 dataset and classifies traffic as
    Benign or one of six attack types.

    **Methodology:**
    - Data cleaning, feature selection, and SMOTE oversampling for class imbalance
    - Ensemble models: Random Forest, XGBoost (Optuna-tuned)
    - Deep learning models: MLP, 1D-CNN, Autoencoder (anomaly detection)
    - Unsupervised baseline: Isolation Forest
    - All models optimized for **Macro F1-Score**
    """)

    st.subheader("Architecture")
    arch = (
        "┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐\n"
        "│   Raw CSV    │───▶│   Clean &    │───▶│   Scale &    │───▶│    Model     │\n"
        "│   Upload     │    │  Preprocess  │    │  Transform   │    │  Inference   │\n"
        "└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘\n"
        "                                                                  │        \n"
        "                                                                  ▼        \n"
        "                                                          ┌──────────────┐ \n"
        "                                                          │Threat Report │ \n"
        "                                                          │+ Analytics   │ \n"
        "                                                          │+ Anomaly Scr │ \n"
        "                                                          └──────────────┘ "
    )
    st.code(arch, language=None)

    st.subheader("Technology Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **ML / DL**
        - scikit-learn
        - XGBoost + Optuna
        - PyTorch (MLP, CNN, AE)
        - imbalanced-learn (SMOTE)
        """)
    with col2:
        st.markdown("""
        **Data**
        - pandas / NumPy
        - CIC-IDS2017 (~2.5M flows)
        - StandardScaler
        - Stratified splits
        """)
    with col3:
        st.markdown("""
        **Visualization**
        - Streamlit
        - Matplotlib / Seaborn
        - Confusion matrices
        - ROC / PR curves
        """)

    st.subheader("Team")
    st.markdown("*(fill in your names here)*")


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
PAGES = {
    "🏠 Home": page_home,
    "📤 Upload & Analyze": page_upload,
    "🚨 Threat Report": page_threat_report,
    "📊 Analytics": page_analytics,
    "🤖 Model Comparison": page_model_comparison,
    "🔬 Feature Analysis": page_feature_analysis,
    "ℹ️ About": page_about,
}

st.sidebar.markdown("## 🛡️ CyberShield")
st.sidebar.markdown("---")
choice = st.sidebar.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")

# ── Feature #4: Dark mode toggle ──
dark_toggle = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"])
if dark_toggle != st.session_state["dark_mode"]:
    st.session_state["dark_mode"] = dark_toggle
    st.rerun()

st.sidebar.markdown("---")

# Model selector
metrics = _metrics()
available_multi = {k: v for k, v in MULTI_MODELS.items() if k in metrics}
if available_multi:
    st.sidebar.markdown("**Select Model**")
    current = st.session_state.get("active_model", "xgb_multi")
    if current not in available_multi:
        current = list(available_multi.keys())[0]
    selected_model = st.sidebar.selectbox(
        "Model for inference",
        options=list(available_multi.keys()),
        index=list(available_multi.keys()).index(current),
        format_func=lambda k: f"{available_multi[k]} ({metrics[k]['macro_f1']:.4f})",
        label_visibility="collapsed",
    )
    if selected_model != st.session_state.get("active_model"):
        st.session_state["active_model"] = selected_model
    active_f1 = metrics[selected_model]["macro_f1"]
    st.sidebar.markdown(f'**Active model:** <span style="color: #ffffff;">{selected_model}</span>', unsafe_allow_html=True)
    st.sidebar.markdown(f'**Macro-F1:** <span style="color: #ffffff;">{active_f1:.4f}</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")

preds = st.session_state.get("preds")
if preds is not None:
    n_attacks = (preds["is_attack"] == 1).sum()
    st.sidebar.markdown(f'**Loaded flows:** <span style="color: #ffffff;">{len(preds):,}</span>', unsafe_allow_html=True)
    st.sidebar.markdown(f'**Threats found:** <span style="color: #ffffff;">{n_attacks:,}</span>', unsafe_allow_html=True)

PAGES[choice]()
