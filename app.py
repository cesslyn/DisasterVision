import os, json, sqlite3, datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image


#  CONFIG
CLASS_NAMES = sorted(["earthquake", "fire", "flood", "landslide", "normal"])

# Per-class palette
CLS_HEX = {
    "earthquake": "#F63E31",
    "fire":       "#FF8C42",
    "flood":      "#3B9EBF",
    "landslide":  "#9B8EA0",
    "normal":     "#52B788",
}
ACCENT = "#F63E31"

# Dark palette
DARK = dict(
    bg       = "#0b1820",
    side_bg  = "#0d1f28",
    card_bg  = "#0f2632",
    card_bdr = "#1a3d50",
    text     = "#dce8ed",
    sub      = "#5f8a9a",
    chart_bg = "#0b1820",
    grid     = "#132533",
    blob1    = "rgba(15,38,50,0.9)",
    blob2    = "rgba(246,62,49,0.08)",
)

# Light palette — clean white/slate, high contrast
LIGHT = dict(
    bg       = "#f1f5f9",
    side_bg  = "#ffffff",
    card_bg  = "#ffffff",
    card_bdr = "#e2e8f0",
    text     = "#0f172a",
    sub      = "#64748b",
    chart_bg = "#ffffff",
    grid     = "#f1f5f9",
    blob1    = "rgba(226,232,240,0.6)",
    blob2    = "rgba(246,62,49,0.03)",
)

MODEL_PATH = os.path.join("models", "disaster_classifier.keras")
DB_PATH    = "predictions.db"
EVAL_DIR   = "eval"
IMG_SIZE   = 224

PAGES = [
    ("Dashboard", "▦"),
    ("Classify",  "⬆"),
    ("History",   "◷"),
    ("About",     "ℹ"),
]

def pal():
    return DARK if st.session_state.get("dark", True) else LIGHT

def is_dark():
    return st.session_state.get("dark", True)

#  PAGE CONFIG
st.set_page_config(
    page_title="Disaster AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  DATABASE
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT, pred_class TEXT,
        confidence REAL, timestamp TEXT)""")
    conn.commit(); conn.close()

def save_prediction(name, cls, conf):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO predictions VALUES (NULL,?,?,?,?)",
        (name, cls, round(conf, 4),
         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit(); conn.close()

def load_predictions(cls_filter="All", min_conf=0.0):
    conn = sqlite3.connect(DB_PATH)
    q, params = "SELECT * FROM predictions WHERE confidence >= ?", [min_conf]
    if cls_filter != "All":
        q += " AND pred_class = ?"; params.append(cls_filter)
    q += " ORDER BY id DESC"
    df = pd.read_sql(q, conn, params=params)
    conn.close()
    return df

#  MODEL
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    return tf.keras.models.load_model(MODEL_PATH)

def run_predict(model, img):
    arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    arr = preprocess_input(np.expand_dims(arr, 0))
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

#  CSS
def inject_css():
    p   = pal()
    drk = is_dark()
    title_color = "#ffffff" if drk else p['text']

    # Sidebar shadow differs between modes
    sidebar_shadow = "2px 0 20px rgba(0,0,0,0.3)" if drk else "2px 0 16px rgba(0,0,0,0.08)"
    card_shadow    = "0 4px 24px rgba(0,0,0,0.25)" if drk else "0 2px 12px rgba(15,23,42,0.08)"
    card_shadow_hv = "0 8px 32px rgba(0,0,0,0.35)" if drk else "0 6px 24px rgba(15,23,42,0.13)"
    chart_shadow   = "0 4px 24px rgba(0,0,0,0.22)" if drk else "0 2px 12px rgba(15,23,42,0.06)"

    # ── NAV BUTTON COLORS ──────────────────────────────────────────────────────
    # Inactive icon: white in dark mode, dark slate in light mode (clearly visible on white sidebar)
    nav_icon_color   = "#dce8ed" if drk else "#1e293b"
    # Inactive button background pill: subtle in dark, light gray in light
    nav_btn_bg       = "rgba(255,255,255,0.04)" if drk else "rgba(15,23,42,0.07)"
    nav_btn_bdr      = "transparent" if drk else "#cbd5e1"
    # Hover
    nav_hover_bg     = "rgba(255,255,255,0.10)" if drk else "rgba(15,23,42,0.13)"
    nav_hover_color  = "#ffffff" if drk else "#0f172a"
    nav_hover_bdr    = "#1a3d50" if drk else "#94a3b8"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}

html, body, .stApp {{
    background: {p['bg']} !important;
    color: {p['text']} !important;
    font-family: 'Inter', sans-serif;
}}

/* ── Layered glass background ── */
.stApp::before {{
    content: '';
    position: fixed; inset: 0; z-index: -2;
    background-color: {p['bg']};
    background-image:
        radial-gradient(ellipse 80% 60% at 5% 10%,  {p['blob1']} 0%, transparent 60%),
        radial-gradient(ellipse 55% 45% at 95% 90%,  {p['blob2']} 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, {'rgba(59,158,191,0.06)' if drk else 'rgba(59,158,191,0.03)'} 0%, transparent 70%);
}}

[data-testid="stHeader"], footer {{ display: none !important; }}
.block-container {{ padding: 1.8rem 2rem !important; max-width: 100% !important; }}

/* ── Sidebar icon rail ── */
[data-testid="stSidebar"] {{
    min-width: 70px !important; max-width: 70px !important;
    background: {p['side_bg']}{'dd' if drk else ''} !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    border-right: 1px solid {p['card_bdr']} !important;
    box-shadow: {sidebar_shadow} !important;
}}
[data-testid="stSidebarContent"] {{
    padding: 16px 0 !important;
    display: flex; flex-direction: column;
    align-items: center; gap: 3px;
}}
[data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
[data-testid="stSidebar"] .stButton {{ width: 46px !important; margin: 1px auto !important; }}

/* ── Reset ALL Streamlit button states inside sidebar ── */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button:focus,
[data-testid="stSidebar"] .stButton > button:active,
[data-testid="stSidebar"] .stButton > button:focus:not(:active),
[data-testid="stSidebar"] .stButton > button:visited {{
    width: 46px !important; height: 46px !important;
    padding: 0 !important; border-radius: 13px !important;
    font-size: 17px !important;
    /* ── FIX: visible background pill + high-contrast icon color ── */
    background: {nav_btn_bg} !important;
    background-color: {nav_btn_bg} !important;
    color: {nav_icon_color} !important;
    border: 1px solid {nav_btn_bdr} !important;
    box-shadow: none !important;
    outline: none !important;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {nav_hover_bg} !important;
    background-color: {nav_hover_bg} !important;
    color: {nav_hover_color} !important;
    border-color: {nav_hover_bdr} !important;
    transform: none !important; opacity: 1 !important;
}}

/* ── Active (current page) nav button ── */
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"]:focus,
[data-testid="stSidebar"] .stButton > button[kind="primary"]:active,
[data-testid="stSidebar"] .stButton > button[kind="primary"]:focus:not(:active) {{
    background: {ACCENT}22 !important;
    background-color: {ACCENT}22 !important;
    color: {ACCENT} !important;
    border-color: {ACCENT}55 !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    background: {ACCENT}33 !important;
    background-color: {ACCENT}33 !important;
    opacity: 1 !important;
}}

/* ── Page title ── */
.pg-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem; font-weight: 800;
    color: {title_color}; letter-spacing: -0.02em;
    line-height: 1.2;
}}
.pg-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; color: {p['sub']};
    letter-spacing: 0.05em; margin-top: 2px; margin-bottom: 22px;
}}

/* ── KPI cards ── */
.kpi-row {{ display: flex; gap: 14px; margin-bottom: 18px; }}
.kpi {{
    flex: 1;
    background: {p['card_bg']}{'cc' if drk else ''};
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid {p['card_bdr']};
    border-radius: 16px;
    padding: 18px 20px 16px;
    position: relative; overflow: hidden;
    box-shadow: {card_shadow};
    transition: box-shadow 0.2s;
}}
.kpi:hover {{
    box-shadow: {card_shadow_hv};
}}
.kpi-accent {{
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 16px 16px 0 0;
}}
.kpi-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; color: {p['sub']};
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 8px;
}}
.kpi-val {{
    font-family: 'Syne', sans-serif;
    font-size: 1.85rem; font-weight: 800;
    color: {title_color}; line-height: 1;
}}
.kpi-delta {{
    font-size: 0.7rem; color: {p['sub']};
    margin-top: 5px;
}}

/* ── Chart cards ── */
.chart-card {{
    background: {p['card_bg']}{'cc' if drk else ''};
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid {p['card_bdr']};
    border-radius: 16px;
    padding: 20px 20px 14px;
    margin-bottom: 14px;
    box-shadow: {chart_shadow};
}}
.chart-title {{
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem; font-weight: 700;
    color: {title_color}; margin-bottom: 2px;
}}
.chart-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; color: {p['sub']};
    margin-bottom: 14px; letter-spacing: 0.04em;
}}

/* ── Recent activity table ── */
.act-row {{
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 9px 0;
    border-bottom: 1px solid {p['card_bdr']};
    font-size: 0.83rem;
}}
.act-row:last-child {{ border-bottom: none; }}
.act-name {{ color: {p['text']}; font-weight: 500; }}
.act-date {{ color: {p['sub']}; font-family: 'DM Mono', monospace; font-size: 0.72rem; }}
.badge {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em;
}}
.badge-hi  {{ background: #52B78820; color: #52B788; }}
.badge-lo  {{ background: {ACCENT}20; color: {ACCENT}; }}
.badge-med {{ background: #FF8C4220; color: #FF8C42; }}

/* ── Prob bar ── */
.pbar-row {{ margin-bottom: 10px; }}
.pbar-meta {{ display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:4px; color:{p['text']}; }}
.pbar-meta span:last-child {{ font-family:'DM Mono',monospace; font-weight:500; }}
.pbar-track {{ background: {p['grid']}; border-radius: 4px; height: 6px; }}
.pbar-fill  {{ border-radius: 4px; height: 6px; transition: width 0.4s; }}

/* ── Result box ── */
.result-box {{
    background: {p['card_bg']}{'cc' if drk else ''};
    backdrop-filter: blur(20px);
    border: 1px solid {p['card_bdr']};
    border-radius: 16px; padding: 28px 24px;
    box-shadow: {chart_shadow};
}}
.result-box.hi {{ border-color: {ACCENT}66; }}
.result-box.lo {{ border-color: #f9731666; }}
.result-cls {{
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    letter-spacing: -0.02em; margin: 10px 0 4px;
}}
.result-conf {{ font-family:'DM Mono',monospace; font-size:0.78rem; color:{p['sub']}; }}

/* ── Widgets ── */
label, .stSelectbox label, .stSlider label, .stFileUploader label {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important; color: {p['sub']} !important;
    text-transform: uppercase; letter-spacing: 0.1em;
}}
.stSelectbox > div > div {{
    background: {p['card_bg']}cc !important;
    border-color: {p['card_bdr']} !important;
    color: {p['text']} !important; border-radius: 10px !important;
}}
[data-testid="stFileUploader"] {{
    background: {p['card_bg']}88;
    border: 1px dashed {p['card_bdr']}; border-radius: 12px; padding: 4px;
}}
.stButton > button {{
    background: {ACCENT} !important; color: white !important;
    border: none !important; border-radius: 9px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}}
.stButton > button:hover {{ opacity: 0.86 !important; transform: none !important; }}
.stDownloadButton > button {{
    background: transparent !important;
    border: 1px solid {p['card_bdr']} !important; color: {p['text']} !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important;
}}
.stDownloadButton > button:hover {{ border-color: {ACCENT} !important; color: {ACCENT} !important; opacity: 1 !important; }}
[data-testid="stMetricValue"] {{
    color: {title_color} !important; font-family: 'Syne', sans-serif !important;
}}
[data-testid="stMetricLabel"] {{
    color: {p['sub']} !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.62rem !important;
}}
div[data-testid="stMarkdownContainer"] p {{ color: {p['text']}; }}
.stAlert {{ border-radius: 10px !important; }}
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-thumb {{ background: {p['card_bdr']}; border-radius: 2px; }}
</style>
""", unsafe_allow_html=True)


#  CHART HELPER
def mk_fig(w=5.5, h=3.2):
    p = pal()
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(p['chart_bg'])
    ax.set_facecolor(p['chart_bg'])
    ax.tick_params(colors=p['sub'], labelsize=8)
    ax.xaxis.label.set_color(p['sub'])
    ax.yaxis.label.set_color(p['sub'])
    for sp in ax.spines.values():
        sp.set_color(p['grid'])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=p['grid'], linewidth=0.6, linestyle='--')
    ax.xaxis.grid(False)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.18)
    return fig, ax, p

def save_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

#  NAV
def nav():
    with st.sidebar:
        st.markdown(f"""
        <div style="width:42px;height:42px;background:{ACCENT};border-radius:13px;
             display:flex;align-items:center;justify-content:center;
             margin:4px auto 22px;font-size:20px;line-height:1;
             box-shadow:0 4px 16px {ACCENT}66;">
          ⚠️
        </div>
        """, unsafe_allow_html=True)

        for label, icon in PAGES:
            active = st.session_state.get("page") == label
            if st.button(icon, key=f"nav_{label}", help=label,
                         type="primary" if active else "secondary"):
                st.session_state.page = label
                st.rerun()

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown("---")
        drk      = is_dark()
        tog_icon = "☀" if drk else "🌑"
        if st.button(tog_icon, key="dark_toggle",
                     help="Light mode" if drk else "Dark mode"):
            st.session_state.dark = not drk
            st.rerun()

#  PAGE: DASHBOARD
def page_dashboard():
    p   = pal()
    drk = is_dark()
    title_color = "#ffffff" if drk else p['text']

    hcol, fcol = st.columns([3, 1], gap="medium")
    with hcol:
        st.markdown('<div class="pg-title">Analytics Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="pg-sub">Real-time insights from all classifications</div>', unsafe_allow_html=True)
    with fcol:
        cls_f  = st.selectbox("Class filter", ["All"] + CLASS_NAMES, label_visibility="collapsed")

    df_all = load_predictions()
    df     = load_predictions(cls_f)

    if len(df_all) == 0:
        st.markdown(f'<div class="chart-card" style="text-align:center;padding:52px;color:{p["sub"]};">No predictions yet — go to Classify to get started.</div>', unsafe_allow_html=True)
        return

    total   = len(df)
    avg_c   = df["confidence"].mean() * 100 if total else 0
    high_c  = int((df["confidence"] >= 0.80).sum()) if total else 0
    low_c   = int((df["confidence"] < 0.60).sum())  if total else 0
    top_c   = df["pred_class"].mode()[0].title() if total else "—"
    model_acc = None
    mpath = os.path.join(EVAL_DIR, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f: metrics = json.load(f)
        model_acc = metrics.get("test_accuracy", 0) * 100

    kpi_accent_colors = [ACCENT, "#3B9EBF", "#52B788", "#FF8C42", "#9B8EA0"]
    kpi_data = [
        ("Total Classifications", f"{total:,}",        "images processed"),
        ("Avg Confidence",        f"{avg_c:.1f}%",      "across selected"),
        ("High Confidence",       f"{high_c}",          "above 80% threshold"),
        ("Low Confidence",        f"{low_c}",           "needs manual review"),
        ("Model Accuracy",        f"{model_acc:.1f}%" if model_acc else "—", "on eval test set"),
    ]
    kpi_html = ""
    for (label, val, hint), color in zip(kpi_data, kpi_accent_colors):
        kpi_html += f"""
        <div class="kpi">
          <div class="kpi-accent" style="background:{color};"></div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-delta">{hint}</div>
        </div>"""
    st.markdown(f'<div class="kpi-row">{kpi_html}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="chart-card">
          <div class="chart-title">Classification Trend</div>
          <div class="chart-sub">Daily count per disaster class</div>
        """, unsafe_allow_html=True)

        if len(df) > 1:
            df2 = df.copy()
            df2["timestamp"] = pd.to_datetime(df2["timestamp"])
            df2["date"]      = df2["timestamp"].dt.date
            ts = df2.groupby(["date","pred_class"]).size().unstack(fill_value=0).reindex(columns=CLASS_NAMES, fill_value=0)
            fig, ax, p2 = mk_fig(w=6.5, h=2.9)
            for c in CLASS_NAMES:
                if c in ts.columns and ts[c].sum() > 0:
                    ax.plot(ts.index, ts[c], label=c.title(),
                            color=CLS_HEX[c], linewidth=2, marker='o', markersize=4,
                            markerfacecolor='white', markeredgewidth=1.5)
                    ax.fill_between(ts.index, ts[c], alpha=0.08, color=CLS_HEX[c])
            ax.set_xlabel("Date", fontsize=8, fontfamily='monospace')
            ax.set_ylabel("Count", fontsize=8, fontfamily='monospace')
            plt.xticks(rotation=25, fontsize=7.5)
            ax.legend(facecolor=p2['chart_bg'], edgecolor=p2['grid'],
                      labelcolor=p2['text'], fontsize=7.5, ncol=3,
                      framealpha=0.7, loc='upper left')
            save_fig(fig)
        else:
            st.markdown(f'<div style="padding:40px;text-align:center;color:{p["sub"]};">Classify more images to see trends.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="chart-card">
          <div class="chart-title">Class Distribution</div>
          <div class="chart-sub">Share of each disaster type</div>
        """, unsafe_allow_html=True)

        counts = df["pred_class"].value_counts().reindex(CLASS_NAMES, fill_value=0)
        non_zero = counts[counts > 0]

        if non_zero.sum() > 0:
            fig, ax, p2 = mk_fig(w=3.8, h=2.9)
            ax.yaxis.grid(False); ax.xaxis.grid(False)
            wedges, texts, autotexts = ax.pie(
                non_zero.values,
                labels=None,
                colors=[CLS_HEX[c] for c in non_zero.index],
                autopct='%1.0f%%',
                pctdistance=0.78,
                startangle=90,
                wedgeprops=dict(width=0.55, edgecolor=p2['chart_bg'], linewidth=2),
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color(p2['text'])
                at.set_fontfamily('monospace')
            patches = [mpatches.Patch(color=CLS_HEX[c], label=f"{c.title()}  {counts[c]}")
                       for c in non_zero.index]
            ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5,-0.22),
                      ncol=2, fontsize=7.5, frameon=False,
                      labelcolor=p2['text'], handlelength=1)
            ax.set_aspect('equal')
            for sp in ax.spines.values(): sp.set_visible(False)
            save_fig(fig)
        else:
            st.markdown(f'<div style="padding:40px;text-align:center;color:{p["sub"]};">No data.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns([2, 3], gap="medium")

    with col3:
        st.markdown(f"""
        <div class="chart-card">
          <div class="chart-title">Avg Confidence by Class</div>
          <div class="chart-sub">Mean prediction score per category</div>
        """, unsafe_allow_html=True)

        avg_conf = df.groupby("pred_class")["confidence"].mean().reindex(CLASS_NAMES, fill_value=0)
        fig, ax, p2 = mk_fig(w=3.8, h=2.9)
        ax.yaxis.grid(False)
        bars = ax.barh(
            [c.title() for c in avg_conf.index],
            avg_conf.values * 100,
            color=[CLS_HEX[c] for c in avg_conf.index],
            edgecolor='none', height=0.52
        )
        for b, v in zip(bars, avg_conf.values):
            if v > 0:
                ax.text(b.get_width() + 0.8, b.get_y() + b.get_height()/2,
                        f"{v*100:.1f}%", va='center', ha='left',
                        fontsize=8, color=p2['text'], fontfamily='monospace')
        ax.set_xlabel("Avg Confidence (%)", fontsize=8, fontfamily='monospace')
        ax.set_xlim(0, 118)
        save_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="chart-card">
          <div class="chart-title">Recent Classifications</div>
          <div class="chart-sub">Latest predictions with confidence status</div>
        """, unsafe_allow_html=True)

        recent = df.head(8)
        if len(recent) > 0:
            act_html = ""
            for _, row in recent.iterrows():
                conf_val  = row['confidence']
                name      = row['image_name'][:28] + ("…" if len(row['image_name']) > 28 else "")
                cls_label = row['pred_class'].title()
                ts        = str(row['timestamp'])[:16]
                dot_color = CLS_HEX.get(row['pred_class'], ACCENT)
                if conf_val >= 0.80:
                    badge = '<span class="badge badge-hi">High</span>'
                elif conf_val >= 0.60:
                    badge = '<span class="badge badge-med">Medium</span>'
                else:
                    badge = '<span class="badge badge-lo">Low</span>'
                act_html += f"""
                <div class="act-row">
                  <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:0;">
                    <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};flex-shrink:0;"></div>
                    <div>
                      <div class="act-name">{cls_label}</div>
                      <div class="act-date" style="font-size:0.65rem;margin-top:1px;">{name}</div>
                    </div>
                  </div>
                  <div style="display:flex;align-items:center;gap:14px;flex-shrink:0;">
                    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{p['sub']};">{conf_val*100:.0f}%</span>
                    {badge}
                    <span class="act-date">{ts}</span>
                  </div>
                </div>"""
            st.markdown(act_html, unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="padding:32px;text-align:center;color:{p["sub"]};">No recent classifications.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Model Performance Details", expanded=False):
        c5, c6 = st.columns(2, gap="medium")
        with c5:
            cm = os.path.join(EVAL_DIR, "confusion_matrix.png")
            if os.path.exists(cm):
                st.image(cm, use_container_width=True)
        with c6:
            if os.path.exists(mpath):
                wr = metrics.get("classification_report", {}).get("weighted avg", {})
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Test Accuracy", f"{metrics.get('test_accuracy',0)*100:.2f}%")
                    st.metric("Precision",     f"{wr.get('precision',0)*100:.2f}%")
                with m2:
                    st.metric("Recall",   f"{wr.get('recall',0)*100:.2f}%")
                    st.metric("F1-Score", f"{wr.get('f1-score',0)*100:.2f}%")
                rows = [{"Class": c.title(),
                         "Precision": f"{metrics['classification_report'].get(c,{}).get('precision',0)*100:.1f}%",
                         "Recall":    f"{metrics['classification_report'].get(c,{}).get('recall',0)*100:.1f}%",
                         "F1":        f"{metrics['classification_report'].get(c,{}).get('f1-score',0)*100:.1f}%",
                         "Support":   int(metrics['classification_report'].get(c,{}).get('support',0))}
                        for c in CLASS_NAMES]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        curve = os.path.join(EVAL_DIR, "training_curves.png")
        if os.path.exists(curve):
            st.image(curve, use_container_width=True)


#  PAGE: CLASSIFY
def page_classify(model):
    p   = pal()
    drk = is_dark()
    title_color = "#ffffff" if drk else p['text']
    st.markdown('<div class="pg-title">Classify Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload a disaster image for instant AI classification</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model not found at `models/disaster_classifier.keras`.")
        return

    left, right = st.columns(2, gap="large")
    with left:
        uploaded = st.file_uploader("Drop image or click to browse", type=["jpg","jpeg","png"])
        if uploaded:
            st.image(Image.open(uploaded), caption=uploaded.name, use_container_width=True)
            run = st.button("Run Classification", use_container_width=True)
        else:
            st.markdown(f'<div class="result-box" style="text-align:center;padding:52px;color:{p["sub"]};">JPG · JPEG · PNG supported</div>', unsafe_allow_html=True)
            run = False

    with right:
        if uploaded and run:
            with st.spinner("Analyzing..."):
                cls, conf, probs = run_predict(model, Image.open(uploaded))
                save_prediction(uploaded.name, cls, conf)

            lo      = conf < 0.60
            c_color = "#f97316" if lo else CLS_HEX.get(cls, ACCENT)
            box_cls = "lo" if lo else "hi"
            warn    = f'<div style="background:#f9731618;border:1px solid #f9731644;border-radius:8px;padding:9px 14px;margin-bottom:14px;font-size:0.75rem;color:#f97316;font-family:DM Mono,monospace;letter-spacing:0.05em;">LOW CONFIDENCE — verify manually</div>' if lo else ""

            st.markdown(f"""
            <div class="result-box {box_cls}">
              {warn}
              <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:{p['sub']};text-transform:uppercase;letter-spacing:0.14em;">Predicted Class</div>
              <div class="result-cls" style="color:{c_color};">{cls.upper()}</div>
              <div class="result-conf">Confidence: <strong style="color:{title_color};">{conf*100:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.6rem;color:{p["sub"]};text-transform:uppercase;letter-spacing:0.14em;margin:20px 0 12px;">All Class Probabilities</div>', unsafe_allow_html=True)
            for i, c in enumerate(CLASS_NAMES):
                pv = float(probs[i])
                st.markdown(f"""
                <div class="pbar-row">
                  <div class="pbar-meta"><span>{c.title()}</span><span>{pv*100:.1f}%</span></div>
                  <div class="pbar-track"><div class="pbar-fill" style="width:{pv*100:.1f}%;background:{CLS_HEX.get(c,ACCENT)};"></div></div>
                </div>""", unsafe_allow_html=True)

        elif not uploaded:
            st.markdown(f'<div class="result-box" style="text-align:center;padding:52px;color:{p["sub"]};">Results will appear here</div>', unsafe_allow_html=True)


#  PAGE: HISTORY
def page_history():
    p   = pal()
    drk = is_dark()
    title_color = "#ffffff" if drk else p['text']
    st.markdown('<div class="pg-title">Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">All classifications stored in local database</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: flt     = st.selectbox("Class", ["All"] + CLASS_NAMES)
    with c2: conf_mn = st.slider("Min Confidence (%)", 0, 100, 0, 5) / 100
    with c3: sort_by = st.selectbox("Sort", ["Newest First","Oldest First","Highest Confidence","Lowest Confidence"])

    df = load_predictions(flt, conf_mn)
    if sort_by == "Oldest First":         df = df.sort_values("id")
    elif sort_by == "Highest Confidence": df = df.sort_values("confidence", ascending=False)
    elif sort_by == "Lowest Confidence":  df = df.sort_values("confidence")

    if len(df) == 0:
        st.markdown(f'<div class="chart-card" style="text-align:center;color:{p["sub"]};padding:32px;">No predictions match the current filters.</div>', unsafe_allow_html=True)
        return

    kpi_html = f"""
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-accent" style="background:{ACCENT};"></div>
        <div class="kpi-label">Showing</div><div class="kpi-val">{len(df)}</div></div>
      <div class="kpi"><div class="kpi-accent" style="background:#3B9EBF;"></div>
        <div class="kpi-label">Avg Confidence</div><div class="kpi-val">{df['confidence'].mean()*100:.1f}%</div></div>
      <div class="kpi"><div class="kpi-accent" style="background:#52B788;"></div>
        <div class="kpi-label">High Confidence</div><div class="kpi-val">{int((df['confidence']>=0.80).sum())}</div>
        <div class="kpi-delta">above 80%</div></div>
    </div>"""
    st.markdown(kpi_html, unsafe_allow_html=True)

    disp = df[["image_name","pred_class","confidence","timestamp"]].copy()
    disp.columns = ["Image","Class","Confidence","Timestamp"]
    disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{x*100:.1f}%")
    disp["Class"]      = disp["Class"].apply(str.title)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.download_button("Export CSV", df.to_csv(index=False).encode(), "history.csv", "text/csv")


#  PAGE: ABOUT
def page_about():
    p   = pal()
    drk = is_dark()
    title_color = "#ffffff" if drk else p['text']
    st.markdown('<div class="pg-title">About</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">System information and model details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    td_k = f'color:{p["sub"]};font-family:DM Mono,monospace;font-size:0.68rem;padding:7px 0;width:110px;vertical-align:top;'
    td_v = f'color:{p["text"]};padding:7px 0;font-size:0.83rem;'

    with col1:
        arch = [("BACKBONE","EfficientNetB0 — ImageNet weights"),("INPUT","224 × 224 × 3 RGB"),
                ("PREPROCESS","preprocess_input → [−1, 1]"),
                ("HEAD","GAP → BN → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Softmax(5)"),
                ("PHASE 1","Head only · lr=1e-3 · 10 epochs"),
                ("PHASE 2","Last 20 layers · lr=1e-5 · 20 epochs"),
                ("AUGMENT","Rotation · Zoom · H-Flip · Brightness"),("ACCURACY","96.40% on test set")]
        rows = "".join(f'<tr><td style="{td_k}">{k}</td><td style="{td_v}">{v}</td></tr>' for k,v in arch)
        lims = "".join(f'<li style="color:{p["sub"]};line-height:2;font-size:0.83rem;">{x}</li>' for x in [
            "Only recognizes the 5 trained disaster classes",
            "Performance degrades on blurry or dark images",
            "Not designed for real-time video or live feeds",
            "Below 60% confidence — verify manually"])
        st.markdown(f"""
        <div class="chart-card"><div class="chart-title">Model Architecture</div>
          <table style="width:100%;border-collapse:collapse;">{rows}</table></div>
        <div class="chart-card"><div class="chart-title">Limitations</div>
          <ul style="padding-left:16px;margin:0;">{lims}</ul></div>""", unsafe_allow_html=True)

    with col2:
        cls_rows = "".join(f'<tr><td style="{td_k.replace(p["sub"],ACCENT)}">{c.upper()}</td><td style="{td_v}">{d}</td></tr>' for c,d in [
            ("earthquake","Collapsed structures, debris"),("fire","Wildfires and structure fires"),
            ("flood","Urban and rural inundation"),("landslide","Mudslides, slope collapses"),
            ("normal","Non-disaster everyday scenes")])
        tech_rows = "".join(f'<tr><td style="{td_k}">{k}</td><td style="{td_v}">{v}</td></tr>' for k,v in [
            ("ML","TensorFlow / Keras"),("DASHBOARD","Streamlit"),("DATA","NumPy · Pandas"),
            ("CHARTS","Matplotlib"),("DATABASE","SQLite"),("IMAGE","Pillow")])
        st.markdown(f"""
        <div class="chart-card"><div class="chart-title">Classes</div>
          <table style="width:100%;border-collapse:collapse;">{cls_rows}</table></div>
        <div class="chart-card"><div class="chart-title">Tech Stack</div>
          <table style="width:100%;border-collapse:collapse;">{tech_rows}</table></div>""", unsafe_allow_html=True)


#  MAIN
def main():
    init_db()
    if "page" not in st.session_state: st.session_state.page = "Dashboard"
    if "dark" not in st.session_state: st.session_state.dark = True

    inject_css()
    nav()

    model = load_model()
    pg = st.session_state.page
    if   pg == "Dashboard": page_dashboard()
    elif pg == "Classify":  page_classify(model)
    elif pg == "History":   page_history()
    elif pg == "About":     page_about()

if __name__ == "__main__":
    main()