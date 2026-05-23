"""
╔══════════════════════════════════════════════════════╗
║   AI Disaster Classification System — Dashboard      ║
║   Run: streamlit run app.py                          ║
╚══════════════════════════════════════════════════════╝
"""

import os
import json
import sqlite3
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════
CLASS_NAMES = sorted(["earthquake", "fire", "flood", "landslide", "normal", "smoke"])

CLASS_ICONS = {
    "earthquake": "🏚️",
    "fire":       "🔥",
    "flood":      "🌊",
    "landslide":  "⛰️",
    "normal":     "🌿",
    "smoke":      "💨",
}
CLASS_COLORS = {
    "earthquake": "#FF6B35",
    "fire":       "#FF3B30",
    "flood":      "#007AFF",
    "landslide":  "#8B5E3C",
    "normal":     "#22C55E",
    "smoke":      "#8E8E93",
}

# ═══════════════════════════════════════════════════════════
#  LOCAL PATHS
#  ─────────────────────────────────────────────────────────
#  After running python train.py, your project folder should
#  look like this:
#
#    your_project/
#      train.py
#      app.py
#      requirements.txt
#      dataset/
#        train/
#          earthquake/ fire/ flood/ landslide/ normal/ smoke/
#      data_splits/
#        train/ val/ test/
#      models/
#        disaster_classifier.keras   ← auto-created by train.py
#      eval/
#        confusion_matrix.png        ← auto-created by train.py
#        training_curves.png         ← auto-created by train.py
#        metrics.json                ← auto-created by train.py
#      predictions.db                ← auto-created on first run
#
#  If you moved any files, update the paths below.
# ═══════════════════════════════════════════════════════════
MODEL_PATH = os.path.join("models", "disaster_classifier.keras")
DB_PATH    = "predictions.db"
EVAL_DIR   = "eval"
IMG_SIZE   = 224     # Must match train.py — EfficientNetB0 native resolution


# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Disaster Classifier",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════
#  DATABASE  (SQLite — zero setup)
# ═══════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name  TEXT,
            pred_class  TEXT,
            confidence  REAL,
            timestamp   TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(image_name: str, pred_class: str, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (image_name, pred_class, confidence, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (
            image_name,
            pred_class,
            round(confidence, 4),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def load_predictions(filter_class: str = "All") -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    if filter_class and filter_class != "All":
        df = pd.read_sql(
            "SELECT * FROM predictions WHERE pred_class = ? ORDER BY id DESC",
            conn, params=(filter_class,),
        )
    else:
        df = pd.read_sql(
            "SELECT * FROM predictions ORDER BY id DESC", conn
        )
    conn.close()
    return df


# ═══════════════════════════════════════════════════════════
#  MODEL  (cached to avoid re-loading on each rerun)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(model, img: Image.Image):
    # Resize → float32 array → apply EfficientNet preprocessing
    # preprocess_input scales pixels to [-1, 1] — must match train.py
    arr = np.array(
        img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
        dtype=np.float32
    )
    arr = preprocess_input(arr)          # replaces manual rescale=1/255
    arr = np.expand_dims(arr, 0)         # shape: (1, 224, 224, 3)
    probs   = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return CLASS_NAMES[top_idx], float(probs[top_idx]), probs


# ═══════════════════════════════════════════════════════════
#  CSS INJECTION  (light / dark)
# ═══════════════════════════════════════════════════════════
def inject_css(dark: bool):
    if dark:
        BG, CARD, TEXT, SUB = "#0E1117", "#1B1E2B", "#F0F2F6", "#8B97A8"
        ACCENT, BORDER, MBG = "#4D8BFF", "#2C3047", "#141721"
    else:
        BG, CARD, TEXT, SUB = "#F4F6FB", "#FFFFFF", "#111827", "#6B7280"
        ACCENT, BORDER, MBG = "#2563EB", "#E5E7EB", "#EFF6FF"

    st.markdown(
        f"""
<style>
.stApp {{ background:{BG}; color:{TEXT}; }}

[data-testid="stSidebar"] {{
    background:{CARD};
    border-right:1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}

.card {{
    background:{CARD}; border:1px solid {BORDER};
    border-radius:14px; padding:20px 24px;
    margin-bottom:14px;
    box-shadow:0 1px 4px rgba(0,0,0,.06);
    transition:box-shadow .2s;
}}
.card:hover {{ box-shadow:0 6px 18px rgba(0,0,0,.10); }}

.feat-card {{
    background:{CARD}; border:1px solid {BORDER};
    border-radius:14px; padding:22px 18px;
    text-align:center;
    transition:transform .2s, box-shadow .2s;
    height:100%;
}}
.feat-card:hover {{ transform:translateY(-4px); box-shadow:0 10px 28px rgba(0,0,0,.12); }}
.feat-card .fi {{ font-size:2.2rem; margin-bottom:10px; }}
.feat-card h4 {{ color:{TEXT}; font-size:.97rem; font-weight:700; margin:0 0 6px; }}
.feat-card p  {{ color:{SUB};  font-size:.83rem; margin:0; line-height:1.55; }}

.result-card {{
    background:{CARD}; border:2px solid {ACCENT};
    border-radius:18px; padding:30px 28px;
    box-shadow:0 4px 20px {ACCENT}22;
}}
.res-class {{ font-size:2rem; font-weight:800; color:{ACCENT}; margin:6px 0 4px; }}
.res-conf  {{ color:{SUB}; font-size:.95rem; }}

.warn-card {{
    background:{"#1E1000" if dark else "#FFF7ED"};
    border:2px solid #F97316; border-radius:18px; padding:24px;
}}

.mtile {{
    background:{MBG}; border:1px solid {BORDER};
    border-radius:12px; padding:18px 14px; text-align:center;
}}
.mtile .ml {{ font-size:.73rem; color:{SUB}; text-transform:uppercase;
              letter-spacing:.05em; margin-bottom:6px; }}
.mtile .mv {{ font-size:1.65rem; font-weight:800; color:{ACCENT}; }}

.hero {{
    background:linear-gradient(135deg,{ACCENT}18 0%,{ACCENT}04 100%);
    border:1px solid {BORDER}; border-radius:20px;
    padding:52px 40px; text-align:center; margin-bottom:32px;
}}
.hero h1   {{ font-size:2.5rem; font-weight:900; color:{TEXT};
              margin-bottom:14px; line-height:1.15; }}
.hero .sub {{ font-size:1.08rem; color:{SUB}; max-width:620px;
              margin:0 auto 24px; line-height:1.65; }}

.badge {{
    display:inline-block; padding:4px 13px; border-radius:20px;
    font-size:.75rem; font-weight:700;
    background:{ACCENT}22; color:{ACCENT}; margin:3px;
}}

.sec-hdr {{
    font-size:1.2rem; font-weight:700; color:{TEXT};
    margin:24px 0 14px; padding-bottom:8px;
    border-bottom:2px solid {ACCENT}44;
}}

.pbar-wrap {{ background:{BORDER}; border-radius:6px; height:8px; margin-top:3px; }}
.pbar      {{ border-radius:6px; height:8px; }}

.stButton>button {{
    border-radius:9px; font-weight:700; padding:10px 26px;
    border:none; background:{ACCENT}; color:#fff;
    transition:opacity .2s, transform .15s;
}}
.stButton>button:hover {{ opacity:.87; transform:translateY(-1px); color:#fff; }}

::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}

.dataframe {{ font-size:.87rem !important; }}
[data-testid="stMetricValue"] {{ color:{ACCENT} !important; }}
</style>
""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
#  HELPER — metric tile HTML
# ═══════════════════════════════════════════════════════════
def mtile(label: str, value: str) -> str:
    return (
        f'<div class="mtile">'
        f'<div class="ml">{label}</div>'
        f'<div class="mv">{value}</div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════
#  PAGE — HOME
# ═══════════════════════════════════════════════════════════
def page_home():
    st.markdown(
        """
<div class="hero">
  <h1>🚨 AI Disaster Classification System</h1>
  <div class="sub">
    Deep learning-powered image analysis to automatically detect and classify
    natural disaster events — helping responders act faster and smarter.
  </div>
  <span class="badge">EfficientNetB0</span>
  <span class="badge">Transfer Learning</span>
  <span class="badge">6 Classes</span>
  <span class="badge">Real-time Inference</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-hdr">🔑 System Features</div>', unsafe_allow_html=True)

    features = [
        ("🔥", "Multi-class Detection",
         "Classifies Fire, Flood, Earthquake, Landslide, Smoke & Normal images from a single upload"),
        ("🧠", "EfficientNetB0 Model",
         "Lightweight yet powerful backbone fine-tuned with transfer learning for peak accuracy"),
        ("📊", "Live Analytics",
         "Real-time charts showing prediction trends and confidence distributions"),
        ("📁", "Prediction History",
         "Searchable, filterable log of all classifications with timestamps"),
        ("⚡", "Instant Inference",
         "Results delivered in milliseconds with per-class confidence breakdown"),
        ("🎨", "Modern UI",
         "Clean, responsive interface with full Light and Dark mode support"),
    ]

    for row in range(0, len(features), 3):
        cols = st.columns(3, gap="medium")
        for col, (icon, title, desc) in zip(cols, features[row : row + 3]):
            with col:
                st.markdown(
                    f'<div class="feat-card"><div class="fi">{icon}</div>'
                    f"<h4>{title}</h4><p>{desc}</p></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("")
    st.markdown('<div class="sec-hdr">📈 Quick Stats</div>', unsafe_allow_html=True)

    df = load_predictions()
    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        st.markdown(mtile("Total Predictions", str(len(df))), unsafe_allow_html=True)
    with c2:
        avg = f"{df['confidence'].mean()*100:.1f}%" if len(df) else "—"
        st.markdown(mtile("Avg Confidence", avg), unsafe_allow_html=True)
    with c3:
        tc = df["pred_class"].mode()[0] if len(df) else None
        v  = f"{CLASS_ICONS.get(tc, '')} {tc.title()}" if tc else "—"
        st.markdown(mtile("Top Class", v), unsafe_allow_html=True)
    with c4:
        st.markdown(
            mtile("Classes Detected",
                  f"{df['pred_class'].nunique() if len(df) else 0} / 6"),
            unsafe_allow_html=True,
        )

    st.markdown("")
    if os.path.exists(MODEL_PATH):
        st.success("✅  Trained model loaded and ready for inference.")
    else:
        st.warning(
            "⚠️  Model not found at `models/disaster_classifier.keras`\n\n"
            "**To fix:** Run `python train.py` in your VS Code terminal. "
            "Training will create the `models/` and `eval/` folders automatically."
        )


# ═══════════════════════════════════════════════════════════
#  PAGE — UPLOAD & PREDICT
# ═══════════════════════════════════════════════════════════
def page_predict(model, dark: bool):
    st.markdown("## 🔍 Upload & Predict")
    st.markdown(
        '<p style="color:#6B7280;margin-bottom:24px;">Upload a disaster image '
        "for instant AI-powered classification.</p>",
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "⚠️  Model not loaded — `models/disaster_classifier.keras` not found.\n\n"
            "**To fix:** Open a terminal in VS Code and run `python train.py`."
        )
        return

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        uploaded = st.file_uploader(
            "Drop an image or click to browse",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption=f"📷  {uploaded.name}", use_column_width=True)
            run = st.button("🚀  Classify Image", use_container_width=True)
        else:
            st.info("📤  Supported formats: JPG, JPEG, PNG")
            run = False

    with col_right:
        if uploaded and run:
            with st.spinner("Analyzing …"):
                img              = Image.open(uploaded)
                cls, conf, probs = predict_image(model, img)
                save_prediction(uploaded.name, cls, conf)

            icon  = CLASS_ICONS.get(cls, "❓")
            color = CLASS_COLORS.get(cls, "#2563EB")
            low   = conf < 0.60

            if low:
                st.markdown(
                    f"""
<div class="warn-card">
  <p style="color:#F97316;font-weight:700;margin:0 0 4px;">⚠️  Low Confidence Prediction</p>
  <p style="color:#B45309;font-size:.88rem;margin:0 0 14px;">
    The model is not confident — please verify manually.
  </p>
  <div style="font-size:2.6rem;">{icon}</div>
  <div style="font-size:1.8rem;font-weight:800;color:#F97316;">{cls.upper()}</div>
  <div style="color:#B45309;margin-top:4px;">{conf*100:.1f}% confidence</div>
</div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
<div class="result-card">
  <div style="color:#9CA3AF;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;">
    Predicted Class
  </div>
  <div style="font-size:2.8rem;margin:8px 0 2px;">{icon}</div>
  <div class="res-class" style="color:{color};">{cls.upper()}</div>
  <div class="res-conf">Confidence: <strong>{conf*100:.1f}%</strong></div>
</div>""",
                    unsafe_allow_html=True,
                )

            # ── Per-class probability bars ──
            st.markdown("")
            st.markdown("**All Class Probabilities**")
            for i, c in enumerate(CLASS_NAMES):
                p = float(probs[i])
                st.markdown(
                    f"""
<div style="margin-bottom:9px;">
  <div style="display:flex;justify-content:space-between;font-size:.87rem;margin-bottom:3px;">
    <span>{CLASS_ICONS.get(c, '')} {c.title()}</span>
    <span style="font-weight:700;">{p*100:.1f}%</span>
  </div>
  <div class="pbar-wrap">
    <div class="pbar" style="width:{p*100:.1f}%;background:{CLASS_COLORS.get(c,'#2563EB')};"></div>
  </div>
</div>""",
                    unsafe_allow_html=True,
                )

        elif not uploaded:
            st.markdown(
                """
<div class="card" style="text-align:center;padding:52px 24px;">
  <div style="font-size:3.2rem;margin-bottom:12px;">🖼️</div>
  <div style="color:#9CA3AF;font-size:.95rem;">Upload an image to see results here</div>
</div>""",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════
#  PAGE — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════
def page_analytics(dark: bool):
    st.markdown("## 📊 Analytics Dashboard")
    st.markdown(
        '<p style="color:#6B7280;margin-bottom:24px;">Insights from all predictions made by the system.</p>',
        unsafe_allow_html=True,
    )

    df = load_predictions()
    if len(df) == 0:
        st.info("📭  No predictions yet. Head to **Upload & Predict** to classify some images first.")
        return

    # Chart theme
    FIG_BG = "#1B1E2B" if dark else "#FFFFFF"
    TXT_CLR = "#F0F2F6" if dark else "#111827"
    GRID   = "#2C3047"  if dark else "#E5E7EB"
    TICK   = "#8B97A8"  if dark else "#6B7280"

    # ── Row 1 ─────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="sec-hdr">Disaster Class Distribution</div>', unsafe_allow_html=True)
        counts = df["pred_class"].value_counts().reindex(CLASS_NAMES, fill_value=0)
        colors = [CLASS_COLORS[c] for c in counts.index]

        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor(FIG_BG)
        ax.set_facecolor(FIG_BG)
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="none", width=0.58)
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .15,
                    str(v), ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=TXT_CLR)
        ax.set_xlabel("Disaster Class", color=TICK, fontsize=10)
        ax.set_ylabel("Count",          color=TICK, fontsize=10)
        ax.tick_params(colors=TICK)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_ylim(0, counts.max() * 1.18)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="sec-hdr">Confidence Score Distribution</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor(FIG_BG)
        ax.set_facecolor(FIG_BG)
        ax.hist(df["confidence"] * 100, bins=20, color="#4D8BFF", edgecolor="none", alpha=0.85)
        ax.axvline(60, color="#F97316", linestyle="--", linewidth=1.8, label="60% threshold")
        ax.set_xlabel("Confidence (%)", color=TICK, fontsize=10)
        ax.set_ylabel("Count",          color=TICK, fontsize=10)
        ax.tick_params(colors=TICK)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.legend(facecolor=FIG_BG, edgecolor=GRID, labelcolor=TXT_CLR, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Row 2 ─────────────────────────────────────────────
    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown('<div class="sec-hdr">Class-wise Average Confidence</div>', unsafe_allow_html=True)
        avg_c = (
            df.groupby("pred_class")["confidence"]
            .mean()
            .reindex(CLASS_NAMES, fill_value=0)
        )
        colors2 = [CLASS_COLORS[c] for c in avg_c.index]

        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor(FIG_BG)
        ax.set_facecolor(FIG_BG)
        bars = ax.barh(avg_c.index, avg_c.values * 100,
                       color=colors2, edgecolor="none", height=0.52)
        for b, v in zip(bars, avg_c.values):
            ax.text(b.get_width() + .8, b.get_y() + b.get_height() / 2,
                    f"{v*100:.1f}%", va="center", ha="left",
                    fontsize=10, fontweight="600", color=TXT_CLR)
        ax.set_xlabel("Avg Confidence (%)", color=TICK, fontsize=10)
        ax.set_xlim(0, 118)
        ax.tick_params(colors=TICK)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown('<div class="sec-hdr">Confusion Matrix (Eval Set)</div>', unsafe_allow_html=True)
        cm_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.info("Run `python train.py` to generate the confusion matrix.")

    # ── Training curves ───────────────────────────────────
    curve_path = os.path.join(EVAL_DIR, "training_curves.png")
    if os.path.exists(curve_path):
        st.markdown('<div class="sec-hdr">Training Curves</div>', unsafe_allow_html=True)
        st.image(curve_path, use_column_width=True)

    # ── Model metrics ─────────────────────────────────────
    mpath = os.path.join(EVAL_DIR, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            metrics = json.load(f)

        st.markdown('<div class="sec-hdr">📋 Model Evaluation Metrics</div>', unsafe_allow_html=True)
        wr = metrics.get("classification_report", {}).get("weighted avg", {})

        m1, m2, m3, m4 = st.columns(4, gap="medium")
        for col, (lbl, val) in zip(
            [m1, m2, m3, m4],
            [
                ("Test Accuracy", f"{metrics.get('test_accuracy', 0)*100:.2f}%"),
                ("Precision",     f"{wr.get('precision', 0)*100:.2f}%"),
                ("Recall",        f"{wr.get('recall', 0)*100:.2f}%"),
                ("F1-Score",      f"{wr.get('f1-score', 0)*100:.2f}%"),
            ],
        ):
            with col:
                st.markdown(mtile(lbl, val), unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="sec-hdr">Per-class Breakdown</div>', unsafe_allow_html=True)
        rows = []
        for cls in CLASS_NAMES:
            r = metrics["classification_report"].get(cls, {})
            rows.append({
                "Class":     f"{CLASS_ICONS.get(cls, '')} {cls.title()}",
                "Precision": f"{r.get('precision', 0)*100:.1f}%",
                "Recall":    f"{r.get('recall', 0)*100:.1f}%",
                "F1-Score":  f"{r.get('f1-score', 0)*100:.1f}%",
                "Support":   int(r.get("support", 0)),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
#  PAGE — PREDICTION HISTORY
# ═══════════════════════════════════════════════════════════
def page_history():
    st.markdown("## 📁 Prediction History")
    st.markdown(
        '<p style="color:#6B7280;margin-bottom:24px;">All classifications stored in the local database.</p>',
        unsafe_allow_html=True,
    )

    col_f, _, _ = st.columns([2, 2, 4])
    with col_f:
        flt = st.selectbox("Filter by Class", ["All"] + CLASS_NAMES)

    df = load_predictions(flt)
    if len(df) == 0:
        st.info("📭  No predictions found. Try uploading images in **Upload & Predict**.")
        return

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(mtile("Showing", str(len(df))), unsafe_allow_html=True)
    with c2:
        st.markdown(
            mtile("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%"),
            unsafe_allow_html=True,
        )
    with c3:
        hc = (df["confidence"] >= 0.80).sum()
        st.markdown(mtile("High Confidence (≥80%)", str(hc)), unsafe_allow_html=True)

    st.markdown("")

    disp = df[["image_name", "pred_class", "confidence", "timestamp"]].copy()
    disp.columns = ["Image Name", "Predicted Class", "Confidence", "Timestamp"]
    disp["Confidence"]       = disp["Confidence"].apply(lambda x: f"{x*100:.1f}%")
    disp["Predicted Class"]  = disp["Predicted Class"].apply(
        lambda x: f"{CLASS_ICONS.get(x, '❓')} {x.title()}"
    )

    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️  Export as CSV",
        data=df.to_csv(index=False).encode(),
        file_name="prediction_history.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════
#  PAGE — ABOUT
# ═══════════════════════════════════════════════════════════
def page_about():
    st.markdown("## ℹ️  About This System")
    st.markdown("")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="sec-hdr">📦 Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="card">
<p>Trained on a Kaggle Disaster Dataset containing aerial and ground-level images
across <strong>6 categories</strong> (5 disaster types + 1 normal class):</p>
<ul>
  <li>🔥 <strong>Fire</strong> — Wildfires, structure fires</li>
  <li>🌊 <strong>Flood</strong> — Urban and rural flooding</li>
  <li>🏚️ <strong>Earthquake</strong> — Collapsed structures, debris fields</li>
  <li>⛰️ <strong>Landslide</strong> — Mudslides, slope collapses</li>
  <li>💨 <strong>Smoke</strong> — Dense smoke plumes from fires</li>
  <li>🌿 <strong>Normal</strong> — Everyday, non-disaster scenes</li>
</ul>
<p style="margin-bottom:0;"><strong>Dataset split:</strong>
  70% Training · 15% Validation · 15% Testing</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sec-hdr">⚙️  Model Architecture</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="card">
<ul>
  <li><strong>Backbone:</strong> EfficientNetB0 (ImageNet weights)</li>
  <li><strong>Input:</strong> 224 × 224 × 3 RGB images</li>
  <li><strong>Preprocessing:</strong> <code>preprocess_input</code> — scales pixels to [−1, 1]</li>
  <li><strong>Head:</strong> GAP → BatchNorm → Dense(256, ReLU) → Dropout(0.4)
      → Dense(128, ReLU) → Dropout(0.3) → Softmax(6)</li>
  <li><strong>Training:</strong> 2-phase transfer learning<br>
      &nbsp;&nbsp;&nbsp;Phase 1 — head only (lr=1e-3)<br>
      &nbsp;&nbsp;&nbsp;Phase 2 — last 20 base layers (lr=1e-5)</li>
  <li><strong>Augmentation:</strong> Rotation · Zoom · H-Flip · Brightness</li>
  <li><strong>Class weights:</strong> Balanced (handles imbalanced data)</li>
</ul>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sec-hdr">⚠️  Limitations</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="card">
<ul style="color:#6B7280;">
  <li>Accuracy depends on training data size and quality</li>
  <li>May misclassify images outside the 6 trained classes</li>
  <li>Performance may degrade on blurry or very dark images</li>
  <li>Not designed for real-time video or live-feed scenarios</li>
  <li>Predictions below 60% confidence should be verified by a human</li>
</ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="sec-hdr">🛠️  Tech Stack</div>', unsafe_allow_html=True)
        tech = [
            ("🐍", "Python 3.9+",         "Core language"),
            ("🧠", "TensorFlow / Keras",   "Deep learning framework"),
            ("🖥️", "Streamlit",            "Interactive web dashboard"),
            ("🔢", "NumPy / Pandas",       "Numerical & data processing"),
            ("📊", "Matplotlib / Seaborn", "Data visualization"),
            ("⚙️", "Scikit-learn",         "Metrics, class weights & utilities"),
            ("🗄️", "SQLite",              "Prediction history storage"),
            ("🖼️", "Pillow",              "Image preprocessing"),
        ]
        for icon, name, desc in tech:
            st.markdown(
                f"""
<div class="card" style="padding:13px 18px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:1.4rem;">{icon}</span>
    <div>
      <div style="font-weight:700;font-size:.95rem;">{name}</div>
      <div style="color:#6B7280;font-size:.82rem;">{desc}</div>
    </div>
  </div>
</div>""",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sec-hdr">🚀  Getting Started</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="card">
<ol>
  <li>Place images in the dataset folder:<br>
      <code>dataset/train/&lt;class_name&gt;/*.jpg</code><br>
      <small>Classes: earthquake · fire · flood · landslide · normal · smoke</small>
  </li>
  <li>Install dependencies:<br>
      <code>pip install -r requirements.txt</code>
  </li>
  <li>Train the model (VS Code terminal):<br>
      <code>python train.py</code>
  </li>
  <li>Launch the dashboard:<br>
      <code>streamlit run app.py</code>
  </li>
</ol>
</div>
""",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════
#  SIDEBAR + ROUTING
# ═══════════════════════════════════════════════════════════
def main():
    init_db()

    # Session state defaults
    if "dark" not in st.session_state:
        st.session_state.dark = False
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    inject_css(st.session_state.dark)

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
<div style="text-align:center;padding:18px 0 26px;">
  <div style="font-size:2.6rem;">🚨</div>
  <div style="font-weight:800;font-size:1.1rem;">Disaster AI</div>
  <div style="font-size:.76rem;color:#6B7280;margin-top:2px;">
    Classification System
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.divider()

        nav = [
            ("🏠", "Home"),
            ("🔍", "Upload & Predict"),
            ("📊", "Analytics Dashboard"),
            ("📁", "Prediction History"),
            ("ℹ️", "About"),
        ]
        for icon, label in nav:
            active = st.session_state.page == label
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.page = label
                st.rerun()

        st.divider()

        new_dark = st.toggle("🌙  Dark Mode", value=st.session_state.dark)
        if new_dark != st.session_state.dark:
            st.session_state.dark = new_dark
            st.rerun()

        st.markdown(
            """
<div style="text-align:center;padding:18px 0 6px;
            color:#6B7280;font-size:.73rem;line-height:1.7;">
  v1.0.0 · EfficientNetB0<br>Built with ❤️ using Streamlit
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Load model ────────────────────────────────────────
    model = load_model()

    # ── Route ─────────────────────────────────────────────
    p = st.session_state.page
    if   p == "Home":                page_home()
    elif p == "Upload & Predict":    page_predict(model, st.session_state.dark)
    elif p == "Analytics Dashboard": page_analytics(st.session_state.dark)
    elif p == "Prediction History":  page_history()
    elif p == "About":               page_about()


if __name__ == "__main__":
    main()
