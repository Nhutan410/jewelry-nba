"""
app_instore.py
──────────────────────────────────────────────────────────────────────────────
PNJ · In-Store NBA Dashboard — Nhánh 2

Kiến trúc:
  - Đọc Excel gốc → hiển thị dữ liệu thô (Tab 1)
  - JSON cache (outputs/instore_scripts.json) lưu kết quả phân tích LLM
  - Khi Excel có khách mới chưa có trong cache → hiện nút "Cập nhật"
  - Nút cập nhật chỉ xử lý khách mới, không chạy lại toàn bộ
  - Tab 2: danh sách khách đã phân tích + script 5 bước
  - Tab 3: tìm kiếm chi tiết 1 khách

Chạy:
    streamlit run app_instore.py
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.lep_pipeline import LEPModel, get_feature_importance, DEFAULT_MODEL_DIR
from src.instore_script_engine import InstoreScriptEngine, InstoreIntentType


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH   = ROOT / "data" / "customer_data_poc_enhanced.xlsx"
CACHE_FILE  = ROOT / "outputs" / "instore_scripts.json"
MODEL_PATH  = DEFAULT_MODEL_DIR / "lep_model.pkl"

INTENT_COLORS = {
    "High Purchase": "#ef4444",
    "Exploration":   "#3b82f6",
    "Premium":       "#d4af37",
    "Low Intent":    "#6b7280",
}

INTENT_ICONS = {
    "High Purchase": "🛒",
    "Exploration":   "🔍",
    "Premium":       "👑",
    "Low Intent":    "🌱",
}

PRIORITY_COLORS = {
    "high":   "#ef4444",
    "medium": "#f97316",
    "low":    "#6b7280",
}

SCRIPT_STEPS = [
    ("opening",   "1. Opening — Chào hỏi"),
    ("khai_thac", "2. Khai thác nhu cầu"),
    ("goi_y",     "3. Gợi ý sản phẩm"),
    ("chot",      "4. Chốt đơn"),
    ("upsell",    "5. Upsell / Bán thêm"),
]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PNJ · In-Store NBA",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0f1117;
    color: #e2e8f0;
}

section[data-testid="stSidebar"] {
    background-color: #1a1d27 !important;
    border-right: 1px solid #2d3748;
}

/* Headings */
h1, h2, h3, h4 { color: #f1f5f9 !important; }

/* Divider line */
hr { border-color: #2d3748; }

/* Script step block */
.step-block {
    background: #1e2330;
    border-left: 3px solid #d4af37;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #e2e8f0;
    font-size: 14px;
    line-height: 1.6;
}
.step-label {
    font-size: 11px;
    font-weight: 700;
    color: #d4af37;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

/* Insight boxes */
.box-insight {
    background: #1e2a3a;
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    color: #93c5fd;
    font-size: 13px;
    margin: 8px 0;
}
.box-urgency {
    background: #2a1a1a;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 12px 16px;
    color: #fca5a5;
    font-size: 13px;
    font-weight: 600;
    margin: 8px 0;
}
.box-key {
    background: #2a2410;
    border: 1px solid #d4af37;
    border-radius: 8px;
    padding: 12px 16px;
    color: #fde68a;
    font-size: 13px;
    font-weight: 600;
    margin: 8px 0;
}

/* Intent badge */
.intent-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}

/* Metric row */
.kpi-card {
    background: #1a1d27;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.kpi-value { font-size: 28px; font-weight: 700; color: #d4af37; }
.kpi-label { font-size: 12px; color: #64748b; margin-top: 4px; }

/* Button overrides */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* DataFrame */
.stDataFrame { border-radius: 8px; }

/* Tab styling */
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #d4af37 !important;
    border-bottom-color: #d4af37 !important;
}

/* Selectbox / multiselect */
.stSelectbox label, .stMultiSelect label, .stTextInput label {
    color: #94a3b8 !important;
    font-size: 13px !important;
}

p, li { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# JSON CACHE LAYER
# ══════════════════════════════════════════════════════════════════════════════

def load_cache() -> Optional[dict]:
    """Load JSON cache file. Returns None if not found."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_cache(cache: dict) -> None:
    """Save cache dict to JSON file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def get_excel_customer_ids(excel_path: Path) -> list[str]:
    """Get list of customer IDs from Excel profiles sheet."""
    df = pd.read_excel(excel_path, sheet_name="profiles_enhanced")
    id_col = "c" if "c" in df.columns else "customer_id"
    return df[id_col].astype(str).tolist()


def get_cache_status(cache: Optional[dict], excel_ids: list[str]) -> dict:
    """
    Compare Excel customer IDs against cached IDs.

    Returns:
        status: "no_cache" | "outdated" | "ok"
        new_ids: list of customer IDs not yet in cache
    """
    if cache is None:
        return {"status": "no_cache", "new_ids": excel_ids}

    cached_ids = {c["customer_id"] for c in cache.get("customers", [])}
    new_ids = [cid for cid in excel_ids if cid not in cached_ids]

    if new_ids:
        return {"status": "outdated", "new_ids": new_ids}

    return {"status": "ok", "new_ids": []}


def run_pipeline_and_update_cache(
    excel_path: Path,
    new_customer_ids: Optional[list[str]],
    existing_cache: Optional[dict],
    api_key: str = "",
) -> dict:
    """
    Run LEP + InstoreScriptEngine for new customers,
    merge results with existing cache and save.

    Args:
        excel_path:        path to the Excel data file
        new_customer_ids:  list of customer IDs to process (None = all)
        existing_cache:    existing cache dict to merge into (None = start fresh)
        api_key:           OpenAI API key. If non-empty, uses GPT-4o for script
                           generation; otherwise falls back to template mode.

    Returns:
        Updated cache dict
    """
    sheets      = pd.read_excel(excel_path, sheet_name=None)
    df_profiles = sheets["profiles_enhanced"]
    df_ml       = sheets.get("ml_predictions", pd.DataFrame())

    # Rename 'c' → 'customer_id' for consistent access
    id_col = "c" if "c" in df_profiles.columns else "customer_id"
    df_profiles["customer_id"] = df_profiles[id_col].astype(str)

    # Subset to only new customers if specified
    if new_customer_ids:
        df_subset = df_profiles[df_profiles["customer_id"].isin(new_customer_ids)].copy()
        df_ml_subset = (
            df_ml[df_ml[df_ml.columns[0]].astype(str).isin(new_customer_ids)].copy()
            if not df_ml.empty else pd.DataFrame()
        )
    else:
        df_subset    = df_profiles.copy()
        df_ml_subset = df_ml.copy()

    # Train (or load) LEP model
    if MODEL_PATH.exists() and new_customer_ids:
        # Load existing model for incremental prediction
        lep = LEPModel.load(MODEL_PATH)
    else:
        lep = LEPModel(n_estimators=100)
        lep.train(df_profiles, df_ml, verbose=False)
        lep.save()

    lep_preds = lep.predict(df_subset)

    # Generate instore scripts via GPT-4o if api_key is provided, else fallback
    resolved_key = api_key.strip() or None
    engine = InstoreScriptEngine(
        api_key=resolved_key,
        use_cache=True,
    )
    instore_df = engine.generate_scripts(lep_preds, df_subset, verbose=False)

    # Convert instore_df rows → cache format
    new_entries = []
    for _, row in instore_df.iterrows():
        cid = str(row["customer_id"])
        profile_row = df_profiles[df_profiles["customer_id"] == cid]
        profile_dict = profile_row.iloc[0].to_dict() if len(profile_row) > 0 else {}

        entry = {
            "customer_id":    cid,
            "processed_at":   datetime.now().isoformat(),
            "profile": {
                "segment_rfm_tier": str(row.get("segment_rfm_tier", "")),
                "budget":           str(row.get("budget", "")),
                "style":            str(row.get("style", "")),
                "preferred_type":   str(row.get("preferred_type", "")),
                "material":         str(row.get("material", "")),
                "recency_days":     int(row.get("recency_days", 0)),
                "monetary":         float(row.get("monetary", 0)),
                "avg_discount_pct": float(row.get("avg_discount_pct", 0)),
                "web_pdp_views":    int(row.get("web_pdp_views", 0)),
                "add_to_cart":      int(row.get("add_to_cart", 0)),
                "visit_count":      int(row.get("visit_count", 0)),
                "birthday_in_days": int(row.get("birthday_in_days", 365)),
                "sig_view_ring":    int(row.get("sig_view_ring", 0)),
                "sig_view_diamond": int(row.get("sig_view_diamond", 0)),
                "sig_search_propose": int(row.get("sig_search_propose", 0)),
            },
            "lep": {
                "predicted_intent": str(row.get("lep_intent", "")),
                "confidence":       float(row.get("confidence", 0)),
                "priority":         str(row.get("priority", "low")),
            },
            "instore": {
                "instore_intent":      str(row.get("instore_intent", "")),
                "nba_strategy":        str(row.get("nba_strategy", "")),
                "psychology_trigger":  str(row.get("psychology_trigger", "")),
                "product_focus":       str(row.get("product_focus", "")),
                "product_rec_1":       str(row.get("product_rec_1", "")),
                "product_rec_2":       str(row.get("product_rec_2", "")),
                "product_rec_3":       str(row.get("product_rec_3", "")),
                "online_insight":      str(row.get("online_insight", "")),
                "urgency_signal":      str(row.get("urgency_signal", "")),
                "key_insight":         str(row.get("key_insight", "")),
                "script": {
                    "opening":   str(row.get("script_opening", "")),
                    "khai_thac": str(row.get("script_khai_thac", "")),
                    "goi_y":     str(row.get("script_goi_y", "")),
                    "chot":      str(row.get("script_chot", "")),
                    "upsell":    str(row.get("script_upsell", "")),
                },
                "script_source": str(row.get("script_source", "fallback")),
                "tokens_used":   int(row.get("tokens_used", 0)),
            },
        }
        new_entries.append(entry)

    # Merge with existing cache
    if existing_cache and existing_cache.get("customers"):
        existing_ids = {c["customer_id"] for c in existing_cache["customers"]}
        merged = existing_cache["customers"] + [
            e for e in new_entries if e["customer_id"] not in existing_ids
        ]
    else:
        merged = new_entries

    cache = {
        "version":       "2.0",
        "generated_at":  datetime.now().isoformat(),
        "excel_path":    str(excel_path),
        "total_customers": len(merged),
        "customers":     merged,
    }
    save_cache(cache)
    return cache


def cache_to_dataframe(cache: dict) -> pd.DataFrame:
    """Flatten cache customers list into a display DataFrame."""
    rows = []
    for c in cache.get("customers", []):
        p  = c.get("profile", {})
        lp = c.get("lep", {})
        ins = c.get("instore", {})
        sc  = ins.get("script", {})
        rows.append({
            "customer_id":         c["customer_id"],
            "processed_at":        c.get("processed_at", ""),
            # Profile
            "segment_rfm_tier":    p.get("segment_rfm_tier", ""),
            "budget":              p.get("budget", ""),
            "style":               p.get("style", ""),
            "preferred_type":      p.get("preferred_type", ""),
            "material":            p.get("material", ""),
            "recency_days":        p.get("recency_days", 0),
            "monetary":            p.get("monetary", 0),
            "avg_discount_pct":    p.get("avg_discount_pct", 0),
            "web_pdp_views":       p.get("web_pdp_views", 0),
            "add_to_cart":         p.get("add_to_cart", 0),
            "visit_count":         p.get("visit_count", 0),
            "birthday_in_days":    p.get("birthday_in_days", 365),
            "sig_view_ring":       p.get("sig_view_ring", 0),
            "sig_view_diamond":    p.get("sig_view_diamond", 0),
            "sig_search_propose":  p.get("sig_search_propose", 0),
            # LEP
            "lep_intent":          lp.get("predicted_intent", ""),
            "confidence":          lp.get("confidence", 0),
            "priority":            lp.get("priority", "low"),
            # Instore
            "instore_intent":      ins.get("instore_intent", ""),
            "nba_strategy":        ins.get("nba_strategy", ""),
            "psychology_trigger":  ins.get("psychology_trigger", ""),
            "product_focus":       ins.get("product_focus", ""),
            "product_rec_1":       ins.get("product_rec_1", ""),
            "product_rec_2":       ins.get("product_rec_2", ""),
            "product_rec_3":       ins.get("product_rec_3", ""),
            "online_insight":      ins.get("online_insight", ""),
            "urgency_signal":      ins.get("urgency_signal", ""),
            "key_insight":         ins.get("key_insight", ""),
            "script_source":       ins.get("script_source", ""),
            "tokens_used":         ins.get("tokens_used", 0),
            # Script
            "script_opening":      sc.get("opening", ""),
            "script_khai_thac":    sc.get("khai_thac", ""),
            "script_goi_y":        sc.get("goi_y", ""),
            "script_chot":         sc.get("chot", ""),
            "script_upsell":       sc.get("upsell", ""),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def format_vnd(val: float) -> str:
    """Format a VND number as a human-readable string."""
    if val >= 1_000_000_000:
        return f"{val / 1_000_000_000:.1f} tỷ"
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val / 1_000:.0f}K"
    return f"{val:.0f}"


def intent_badge_html(intent: str) -> str:
    """Return a colored HTML badge for an intent string."""
    color = INTENT_COLORS.get(intent, "#6b7280")
    icon  = INTENT_ICONS.get(intent, "")
    return (
        f'<span class="intent-badge" '
        f'style="background:{color}22;color:{color};border:1px solid {color}77">'
        f'{icon} {intent}</span>'
    )


def render_script_card(row: pd.Series, key_prefix: str = "card") -> None:
    """
    Render the full analysis card for one customer.
    Shows: profile info, behavioral signals, urgency, insight, script 5 bước.

    Args:
        row        : một dòng DataFrame chứa thông tin khách hàng đã phân tích.
        key_prefix : prefix cho Streamlit widget key — phải khác nhau giữa các nơi gọi
                     (để tránh StreamlitDuplicateElementKey khi cùng customer xuất hiện
                     ở nhiều tab khác nhau).
    """
    intent = row.get("instore_intent", "")
    color  = INTENT_COLORS.get(intent, "#6b7280")

    # ── Header ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(
            f"### {INTENT_ICONS.get(intent, '')} {row['customer_id']} "
            f"— {intent}",
            unsafe_allow_html=False,
        )
        st.caption(
            f"Chiến lược: **{row.get('nba_strategy', '')}** · "
            f"Tâm lý học: *{row.get('psychology_trigger', '')}* · "
            f"Script: {row.get('script_source', 'fallback')}"
        )
    with col_b:
        conf = float(row.get("confidence", 0))
        prio = str(row.get("priority", "low")).lower()
        prio_color = PRIORITY_COLORS.get(prio, "#6b7280")
        st.markdown(
            f"**LEP Confidence**  \n"
            f"<span style='font-size:24px;font-weight:700;color:{color}'>"
            f"{conf:.0%}</span>  \n"
            f"<span style='color:{prio_color};font-size:13px'>"
            f"Priority: {prio.upper()}</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Profile + Signals ─────────────────────────────────────────────────────
    col_p, col_s = st.columns(2)

    with col_p:
        st.markdown("**📋 Thông tin khách hàng**")
        profile_items = [
            ("Phân khúc",       row.get("segment_rfm_tier", "")),
            ("Ngân sách",       row.get("budget", "")),
            ("Phong cách",      row.get("style", "")),
            ("Loại trang sức",  row.get("preferred_type", "")),
            ("Chất liệu",       row.get("material", "")),
            ("Tổng chi tiêu",   format_vnd(float(row.get("monetary", 0))) + " đ"),
            ("Mua gần nhất",    f"{int(row.get('recency_days', 0))} ngày trước"),
            ("LEP Intent",      row.get("lep_intent", "")),
        ]
        for label, value in profile_items:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #1e2330'>"
                f"<span style='color:#64748b;font-size:13px'>{label}</span>"
                f"<span style='color:#e2e8f0;font-size:13px'>{value}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_s:
        st.markdown("**📡 Tín hiệu hành vi online**")
        signal_items = [
            ("🛒 Thêm vào giỏ",      int(row.get("add_to_cart", 0))),
            ("👁️  Xem trang SP",      int(row.get("web_pdp_views", 0))),
            ("💍 Xem nhẫn đính hôn", int(row.get("sig_view_ring", 0))),
            ("💎 Xem kim cương",      int(row.get("sig_view_diamond", 0))),
            ("🔍 Tìm kiếm cầu hôn",  int(row.get("sig_search_propose", 0))),
            ("🌐 Lượt ghé web/app",   int(row.get("visit_count", 0))),
            ("📅 Ngày sinh còn",      f"{int(row.get('birthday_in_days', 365))} ngày"),
        ]
        for label, value in signal_items:
            is_active = isinstance(value, int) and value > 0
            val_color = "#d4af37" if is_active else "#4b5563"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #1e2330'>"
                f"<span style='color:#64748b;font-size:13px'>{label}</span>"
                f"<span style='color:{val_color};font-size:13px;font-weight:600'>"
                f"{value}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Urgency / Insight boxes ───────────────────────────────────────────────
    urgency = str(row.get("urgency_signal", "")).strip()
    if urgency:
        st.markdown(
            f'<div class="box-urgency">⚡ {urgency}</div>',
            unsafe_allow_html=True,
        )

    online = str(row.get("online_insight", "")).strip()
    if online:
        st.markdown(
            f'<div class="box-insight">🌐 <strong>Hành vi online:</strong> {online}</div>',
            unsafe_allow_html=True,
        )

    key = str(row.get("key_insight", "")).strip()
    if key:
        st.markdown(
            f'<div class="box-key">💡 <strong>Key Insight:</strong> {key}</div>',
            unsafe_allow_html=True,
        )

    # ── Product recommendations ───────────────────────────────────────────────
    prods = [
        str(row.get(f"product_rec_{i}", "")).strip()
        for i in [1, 2, 3]
        if str(row.get(f"product_rec_{i}", "")).strip()
    ]
    if prods:
        st.markdown("**📦 Sản phẩm gợi ý cho TVV**")
        for prod in prods:
            st.markdown(f"- {prod}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sales Script 5 bước ───────────────────────────────────────────────────
    st.markdown("**🗣️ Sales Script 5 Bước**")
    script_col_map = {
        "opening":   "script_opening",
        "khai_thac": "script_khai_thac",
        "goi_y":     "script_goi_y",
        "chot":      "script_chot",
        "upsell":    "script_upsell",
    }
    for key_name, label in SCRIPT_STEPS:
        col_key = script_col_map[key_name]
        content = str(row.get(col_key, "")).strip()
        if content:
            st.markdown(
                f'<div class="step-block">'
                f'<div class="step-label">{label}</div>'
                f'{content}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Download button ────────────────────────────────────────────────────────
    cid = str(row.get("customer_id", ""))
    script_txt = "\n".join([
        f"=== SCRIPT CHO TVV — {cid} ===",
        f"Loại khách  : {row.get('instore_intent', '')}",
        f"Chiến lược  : {row.get('nba_strategy', '')}",
        f"Tâm lý học  : {row.get('psychology_trigger', '')}",
        "",
        f"Key Insight : {row.get('key_insight', '')}",
        f"Urgency     : {urgency or 'Không có'}",
        "",
        "Sản phẩm gợi ý:",
        *[f"  {i+1}. {p}" for i, p in enumerate(prods)],
        "",
        "SCRIPT 5 BƯỚC:",
        f"1. Opening   : {row.get('script_opening', '')}",
        f"2. Khai thác : {row.get('script_khai_thac', '')}",
        f"3. Gợi ý     : {row.get('script_goi_y', '')}",
        f"4. Chốt đơn  : {row.get('script_chot', '')}",
        f"5. Upsell    : {row.get('script_upsell', '')}",
    ])
    st.download_button(
        label="📥 Tải script (.txt)",
        data=script_txt,
        file_name=f"script_{cid}.txt",
        mime="text/plain",
        key=f"dl_{key_prefix}_{cid}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: Optional[pd.DataFrame]) -> dict:
    """
    Render sidebar with API key input and data filters.

    Returns dict with keys:
        api_key  (str): OpenAI API key entered by user (may be empty)
        search   (str): text search query
        intents  (list[str]): selected intent types
        priority (list[str]): selected priority levels
    """
    st.sidebar.markdown("## 💎 PNJ In-Store NBA")
    st.sidebar.caption("Nhánh 2 · Sales Script Engine")
    st.sidebar.divider()

    # ── OpenAI API Key ──────────────────────────────────────────────────────
    # Reads OPENAI_API_KEY env var as default; user can override in the field.
    st.sidebar.markdown("### 🔑 OpenAI API Key")
    env_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input(
        "API Key",
        value=env_key,
        type="password",
        placeholder="sk-...",
        help=(
            "Nhập OpenAI API Key để sinh script bằng GPT-4o.\n"
            "Nếu để trống, hệ thống dùng Fallback Template (không cần API)."
        ),
    ).strip()

    if api_key:
        st.sidebar.success("✅ GPT-4o mode (gpt-4o)")
    else:
        st.sidebar.warning("⚠️ Fallback template mode")

    st.sidebar.divider()

    # ── Filters (chỉ hiện khi đã có dữ liệu) ───────────────────────────────
    result: dict = {
        "api_key":  api_key,
        "search":   "",
        "intents":  [],
        "priority": [],
    }

    if df is None or df.empty:
        st.sidebar.info("Chưa có dữ liệu phân tích.")
        return result

    st.sidebar.markdown("### 🔍 Bộ lọc")

    result["search"] = st.sidebar.text_input(
        "Tìm khách hàng",
        placeholder="Nhập ID hoặc phân khúc...",
    )

    all_intents = sorted(df["instore_intent"].dropna().unique().tolist())
    result["intents"] = st.sidebar.multiselect(
        "In-Store Intent",
        options=all_intents,
        default=all_intents,
        format_func=lambda x: f"{INTENT_ICONS.get(x, '')} {x}",
    )

    all_prio = [p for p in ["high", "medium", "low"]
                if p in df["priority"].str.lower().values]
    result["priority"] = st.sidebar.multiselect(
        "Priority",
        options=all_prio,
        default=all_prio,
        format_func=str.upper,
    )

    st.sidebar.divider()
    # Show script source breakdown
    n_llm = (df["script_source"] == "llm").sum()
    n_fb  = len(df) - n_llm
    st.sidebar.caption(
        f"📊 {len(df)} khách đã phân tích  \n"
        f"🤖 GPT-4o: {n_llm} · 📝 Fallback: {n_fb}  \n"
        f"📁 `outputs/instore_scripts.json`"
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FILTER
# ══════════════════════════════════════════════════════════════════════════════

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filters to the analysis DataFrame."""
    out = df.copy()

    # Search by customer_id or segment
    q = filters.get("search", "").strip().lower()
    if q:
        mask = (
            out["customer_id"].astype(str).str.lower().str.contains(q, na=False)
            | out["segment_rfm_tier"].astype(str).str.lower().str.contains(q, na=False)
        )
        out = out[mask]

    # Intent filter
    intent_sel = filters.get("intents", [])
    if intent_sel:
        out = out[out["instore_intent"].isin(intent_sel)]

    # Priority filter
    prio_sel = filters.get("priority", [])
    if prio_sel:
        out = out[out["priority"].str.lower().isin(prio_sel)]

    return out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB GUIDE — HƯỚNG DẪN PHÂN LOẠI
# ══════════════════════════════════════════════════════════════════════════════

def render_tab_guide() -> None:
    """Tab hướng dẫn: giải thích cách phân loại tệp khách và chiến lược NBA."""
    st.markdown("## 📖 Hướng dẫn phân loại tệp khách hàng In-Store")
    st.caption(
        "Hệ thống tự động phân loại mỗi khách hàng vào 1 trong 4 tệp dựa trên "
        "tín hiệu hành vi online và kết quả dự đoán của mô hình Machine Learning (LEP). "
        "Mỗi tệp đi kèm một chiến lược tiếp cận và đòn tâm lý riêng."
    )

    st.divider()

    # ── Segment cards ────────────────────────────────────────────────────────
    segments = [
        {
            "icon": "👑",
            "name": "PREMIUM",
            "label": "Khách VIP / Chi tiêu cao",
            "color": "#d4af37",
            "bg": "#2a2410",
            "border": "#d4af37",
            "conditions": [
                "Thuộc phân khúc hạng cao: Gold hoặc Platinum (bất kỳ cấp nào)",
                "HOẶC tổng chi tiêu lịch sử từ 50 triệu đồng trở lên",
                "VÀ mô hình dự đoán khách đang có nhu cầu rõ ràng (mua cho bản thân, mua quà) "
                "hoặc độ tự tin dự đoán ≥ 70%",
            ],
            "strategy": "Tư vấn như VIP — Cá nhân hóa, không vội vàng",
            "psychology": "Độc quyền & khan hiếm — Khách VIP cần cảm giác được ưu tiên, được phục vụ riêng tư. "
                          "Tránh đề cập giá ngay, tập trung vào trải nghiệm và tính riêng biệt của sản phẩm.",
            "script_hint": "Mời vào khu VIP · Giới thiệu hàng giới hạn · Đề xuất dịch vụ khắc tên, hộp quà cao cấp",
        },
        {
            "icon": "🛒",
            "name": "HIGH PURCHASE",
            "label": "Khách sẵn sàng mua ngay",
            "color": "#ef4444",
            "bg": "#2a1a1a",
            "border": "#ef4444",
            "conditions": [
                "Đã bỏ ít nhất 1 sản phẩm vào giỏ hàng online nhưng chưa thanh toán",
                "HOẶC đã xem trang chi tiết sản phẩm từ 5 lần trở lên",
                "VÀ đã truy cập web/app từ 3 lần trở lên, HOẶC mô hình dự đoán với độ tự tin ≥ 75%",
            ],
            "strategy": "Chốt đơn nhanh — Giảm thiểu do dự",
            "psychology": "Sợ mất cơ hội (FOMO) — Khách đã nghiên cứu kỹ, chỉ cần một cú hích cuối. "
                          "Nhấn mạnh số lượng có hạn, ưu đãi đang còn, hoặc sản phẩm đang được nhiều người quan tâm.",
            "script_hint": "Nhắc lại sản phẩm đã xem · Tạo cảm giác khan hiếm · Đề xuất thanh toán ngay tại cửa hàng",
        },
        {
            "icon": "🔍",
            "name": "EXPLORATION",
            "label": "Khách đang tìm hiểu / khám phá",
            "color": "#3b82f6",
            "bg": "#1e2a3a",
            "border": "#3b82f6",
            "conditions": [
                "Mô hình dự đoán khách đang có ý định cầu hôn hoặc mua dịp kỷ niệm (độ tự tin ≥ 40%)",
                "HOẶC đã xem các trang nhẫn đính hôn, kim cương, hoặc tìm kiếm từ khoá liên quan đến cầu hôn",
                "HOẶC đã bỏ ít nhất 1 sản phẩm vào giỏ hàng",
                "HOẶC dịp sinh nhật / kỷ niệm sắp đến trong vòng 60 ngày",
                "HOẶC đã xem từ 2 trang sản phẩm trở lên, hoặc ghé web/app từ 2 lần trở lên",
                "HOẶC mô hình dự đoán nhu cầu tự thưởng / mua quà với độ tự tin ≥ 70%",
            ],
            "strategy": "Định hướng lựa chọn — Không ép, đồng hành",
            "psychology": "Hướng dẫn chuyên gia — Khách đang thu thập thông tin, cần người đồng hành am hiểu. "
                          "Thu hẹp lựa chọn thay vì đưa ra quá nhiều mẫu. Hỏi đúng câu để khơi gợi nhu cầu tiềm ẩn.",
            "script_hint": "Hỏi về dịp đặc biệt · Giới thiệu 2-3 mẫu phù hợp phong cách · Cho thử sản phẩm",
        },
        {
            "icon": "🌱",
            "name": "LOW INTENT",
            "label": "Khách chưa rõ nhu cầu / mới ghé thăm",
            "color": "#6b7280",
            "bg": "#1a1d27",
            "border": "#4b5563",
            "conditions": [
                "Không có sản phẩm nào trong giỏ hàng",
                "Xem trang sản phẩm tối đa 1 lần",
                "Chỉ ghé web / app tối đa 1 lần",
                "Không có tín hiệu xem nhẫn đính hôn, kim cương, hay tìm kiếm liên quan đến cầu hôn",
                "Không có dịp đặc biệt (sinh nhật, kỷ niệm) trong vòng 60 ngày tới",
                "Mô hình không đủ tự tin xác định nhu cầu (độ tự tin < 40%)",
            ],
            "strategy": "Tạo trải nghiệm — Không gây áp lực",
            "psychology": "Thiện cảm & tò mò — Mục tiêu lần này không phải chốt đơn mà là để lại ấn tượng tốt. "
                          "Tạo môi trường thoải mái, để khách tự khám phá, lưu thông tin để chăm sóc sau.",
            "script_hint": "Chào hỏi nhẹ nhàng · Giới thiệu xu hướng mới · Lưu thông tin — hẹn ghé lại",
        },
    ]

    for seg in segments:
        # Build sub-HTML trước để tránh nested f-string phức tạp
        icon = seg["icon"]
        name = seg["name"]
        label = seg["label"]
        color = seg["color"]
        bg = seg["bg"]
        border = seg["border"]
        strategy = seg["strategy"]
        psychology = seg["psychology"]
        script_hint = seg["script_hint"]
        conditions_li = "".join(f"<li>{c}</li>" for c in seg["conditions"])

        # Header card
        st.markdown(
            f'<div style="background:{bg};border:1.5px solid {border};border-radius:12px;'
            f'padding:20px 24px 8px 24px;margin-bottom:4px">'
            f'<span style="font-size:26px">{icon}</span>'
            f'&nbsp;&nbsp;<span style="font-size:17px;font-weight:700;color:{color}">{name}</span>'
            f'&nbsp;<span style="color:#94a3b8;font-size:13px">— {label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Body — dùng st.columns để layout 2 cột thay vì CSS grid
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                f'<div style="background:{bg};border-left:1.5px solid {border};'
                f'border-bottom:1.5px solid {border};border-radius:0 0 0 12px;'
                f'padding:12px 20px 20px 20px">'
                f'<div style="font-size:10px;font-weight:700;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px">'
                f'📌 Điều kiện nhận diện</div>'
                f'<ul style="margin:0;padding-left:18px;color:#cbd5e1;font-size:13px;line-height:1.85">'
                f'{conditions_li}</ul>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_right:
            st.markdown(
                f'<div style="background:{bg};border-right:1.5px solid {border};'
                f'border-bottom:1.5px solid {border};border-radius:0 0 12px 0;'
                f'padding:12px 20px 20px 20px">'
                f'<div style="font-size:10px;font-weight:700;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px">🎯 Chiến lược NBA</div>'
                f'<div style="font-size:13px;font-weight:600;color:{color};margin-bottom:10px">{strategy}</div>'
                f'<div style="font-size:10px;font-weight:700;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px">🧠 Đòn tâm lý</div>'
                f'<div style="font-size:13px;color:#cbd5e1;line-height:1.7;margin-bottom:10px">{psychology}</div>'
                f'<div style="font-size:10px;font-weight:700;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px">🗣️ Gợi ý script</div>'
                f'<div style="font-size:12px;color:#94a3b8;font-style:italic">{script_hint}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)


    # ── Urgency section ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("## ⚡ Tín hiệu Urgency — Khi nào cần nhấn mạnh ngay?")
    st.caption(
        "Urgency là những tín hiệu cho thấy khách có động lực mua hàng cao trong "
        "thời gian ngắn. TVV cần chủ động khéo léo đề cập để tăng tỷ lệ chốt đơn."
    )

    urgency_rules = [
        {
            "icon": "🎂",
            "title": "Sinh nhật cực kỳ gần (còn dưới 15 ngày)",
            "desc": "Khách có sinh nhật hoặc kỷ niệm sắp đến trong vòng 14 ngày tới. "
                    "Đây là cơ hội vàng để gợi ý mua tặng bản thân hoặc nhận quà từ người thân.",
            "action": "Chủ động chúc mừng sinh nhật sắp đến · Gợi ý tự thưởng hoặc nhắc người nhà chuẩn bị quà",
            "color": "#f97316",
        },
        {
            "icon": "📅",
            "title": "Sinh nhật / Kỷ niệm đang đến gần (còn 15–30 ngày)",
            "desc": "Dịp đặc biệt còn trong vòng 1 tháng — khách đang ở giai đoạn lý tưởng để cân nhắc mua. "
                    "Chưa quá gấp nhưng nên gieo ý tưởng sớm.",
            "action": "Nhắc nhẹ dịp sắp đến · Giới thiệu gói quà hoặc dịch vụ cá nhân hóa",
            "color": "#f59e0b",
        },
        {
            "icon": "🛒",
            "title": "Đã chọn hàng online nhưng chưa thanh toán",
            "desc": "Khách đã bỏ sản phẩm vào giỏ hàng trên web/app nhưng rời đi mà không mua. "
                    "Đây là nhóm có intent mua cao nhất — chỉ cần thêm một lý do để chốt.",
            "action": "Nhắc lại sản phẩm đã chọn · Hỗ trợ xem thử trực tiếp tại cửa hàng · Đề xuất ưu đãi thêm",
            "color": "#ef4444",
        },
        {
            "icon": "💍",
            "title": "Dấu hiệu cầu hôn / đính hôn mạnh",
            "desc": "Khách đã tìm kiếm từ khoá liên quan đến 'cầu hôn', 'nhẫn đính hôn' trên nền tảng online của PNJ. "
                    "Đây là tín hiệu cực kỳ mạnh cho thấy khách đang lên kế hoạch cho một dịp đặc biệt.",
            "action": "Hỏi thẳng về sự kiện sắp tới · Giới thiệu bộ sưu tập nhẫn đính hôn · Tư vấn dịch vụ khắc tên",
            "color": "#a855f7",
        },
    ]

    urg_cols = st.columns(2)
    for i, rule in enumerate(urgency_rules):
        with urg_cols[i % 2]:
            st.markdown(
                f"""
                <div style="
                    background:#1a1d27;
                    border-left:4px solid {rule['color']};
                    border-radius:0 10px 10px 0;
                    padding:16px 18px;
                    margin-bottom:14px;
                ">
                    <div style="font-size:22px;margin-bottom:6px">{rule['icon']}</div>
                    <div style="font-size:14px;font-weight:700;color:{rule['color']};
                                margin-bottom:6px">
                        {rule['title']}
                    </div>
                    <div style="font-size:12px;color:#94a3b8;line-height:1.6;margin-bottom:8px">
                        {rule['desc']}
                    </div>
                    <div style="font-size:11px;font-weight:700;color:#64748b;
                                text-transform:uppercase;letter-spacing:.7px;margin-bottom:3px">
                        💬 Hành động gợi ý
                    </div>
                    <div style="font-size:12px;color:#cbd5e1;font-style:italic">
                        {rule['action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DỮ LIỆU GỐC (Raw Excel)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_excel_sheets(path: str) -> dict[str, pd.DataFrame]:
    """Load all sheets from the Excel file. Cached by path string."""
    return pd.read_excel(path, sheet_name=None)


def render_tab_raw(excel_path: Path) -> None:
    """Tab 1: Display raw Excel sheets without any processing."""
    st.markdown("### 📂 Dữ liệu gốc từ Excel")
    st.caption(f"File: `{excel_path.name}` · Đây là dữ liệu thô chưa qua phân tích.")

    try:
        sheets = load_excel_sheets(str(excel_path))
    except Exception as err:
        st.error(f"Không thể đọc file Excel: {err}")
        return

    sheet_tabs = st.tabs([f"📄 {name}" for name in sheets])
    for tab, (name, df) in zip(sheet_tabs, sheets.items()):
        with tab:
            st.caption(f"{len(df)} dòng × {len(df.columns)} cột")
            st.dataframe(df, use_container_width=True, height=480)

            # Download this sheet as CSV
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                f"📥 Tải sheet '{name}' (CSV)",
                data=csv,
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"dl_sheet_{name}",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KẾT QUẢ PHÂN TÍCH
# ══════════════════════════════════════════════════════════════════════════════

def render_tab_analysis(df: pd.DataFrame, filters: dict) -> None:
    """Tab 2: Analysis results — customer list + script detail."""
    if df.empty:
        st.info("Chưa có dữ liệu phân tích. Hãy chạy pipeline ở sidebar.")
        return

    df_filtered = apply_filters(df, filters)

    # ── KPI row ───────────────────────────────────────────────────────────────
    total = len(df)
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, len(df_filtered),                                     "Khách hiển thị"),
        (c2, (df["instore_intent"] == "High Purchase").sum(),       "High Purchase 🛒"),
        (c3, (df["instore_intent"] == "Premium").sum(),             "Premium 👑"),
        (c4, (df["urgency_signal"].str.strip() != "").sum(),        "Có Urgency ⚡"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if df_filtered.empty:
        st.warning("Không tìm thấy khách nào khớp với bộ lọc.")
        return

    # ── Intent distribution chart (simple bar) ────────────────────────────────
    with st.expander("📊 Phân phối Intent", expanded=False):
        col_chart, col_info = st.columns([2, 1])

        with col_chart:
            intent_dist = (
                df_filtered["instore_intent"]
                .value_counts()
                .reset_index()
            )
            intent_dist.columns = ["intent", "count"]
            fig = go.Figure(go.Bar(
                x=[f"{INTENT_ICONS.get(i,'')} {i}" for i in intent_dist["intent"]],
                y=intent_dist["count"],
                marker_color=[INTENT_COLORS.get(i, "#6b7280") for i in intent_dist["intent"]],
                text=intent_dist["count"],
                textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=220,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown("**Avg Confidence theo Intent**")
            for intent in df_filtered["instore_intent"].unique():
                sub  = df_filtered[df_filtered["instore_intent"] == intent]
                avg  = sub["confidence"].mean()
                col  = INTENT_COLORS.get(intent, "#6b7280")
                icon = INTENT_ICONS.get(intent, "")
                st.markdown(
                    f"<div style='padding:4px 0;font-size:13px'>"
                    f"<span style='color:{col}'>{icon} {intent}</span>"
                    f"<span style='float:right;font-weight:600;color:#d4af37'>{avg:.0%}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Customer list ─────────────────────────────────────────────────────────
    st.markdown(f"**{len(df_filtered)} khách hàng** (sau lọc)")

    # Sort selector
    col_sort, col_page = st.columns([2, 1])
    with col_sort:
        sort_opt = st.selectbox(
            "Sắp xếp",
            ["Confidence ↓", "Confidence ↑", "Chi tiêu ↓", "Chi tiêu ↑",
             "Intent", "Customer ID"],
            label_visibility="collapsed",
        )
    with col_page:
        page_size = st.selectbox(
            "Số dòng",
            [5, 10, 20],
            index=1,
            label_visibility="collapsed",
        )

    sort_map = {
        "Confidence ↓":  ("confidence", False),
        "Confidence ↑":  ("confidence", True),
        "Chi tiêu ↓":    ("monetary", False),
        "Chi tiêu ↑":    ("monetary", True),
        "Intent":         ("instore_intent", True),
        "Customer ID":    ("customer_id", True),
    }
    sort_col, sort_asc = sort_map[sort_opt]
    df_sorted = df_filtered.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    # Pagination
    n_pages = max(1, (len(df_sorted) - 1) // page_size + 1)
    if n_pages > 1:
        page = st.number_input(
            f"Trang / {n_pages}", min_value=1, max_value=n_pages, value=1, step=1,
        )
    else:
        page = 1

    start = (page - 1) * page_size
    df_page = df_sorted.iloc[start : start + page_size]

    # ── Expandable customer cards ──────────────────────────────────────────────
    for _, row in df_page.iterrows():
        intent  = row.get("instore_intent", "")
        conf    = float(row.get("confidence", 0))
        urgency = str(row.get("urgency_signal", "")).strip()
        cart    = int(row.get("add_to_cart", 0))

        label_parts = [
            f"{INTENT_ICONS.get(intent, '')} {row['customer_id']}",
            f"{intent}",
            f"{conf:.0%} conf",
            row.get("segment_rfm_tier", ""),
        ]
        if urgency:
            label_parts.append("⚡")
        if cart > 0:
            label_parts.append(f"🛒×{cart}")

        with st.expander(" · ".join(label_parts), expanded=False):
            render_script_card(row, key_prefix="list")

    # ── Export buttons ─────────────────────────────────────────────────────────
    st.divider()
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        view_cols = [
            "customer_id", "segment_rfm_tier", "instore_intent", "nba_strategy",
            "confidence", "priority", "urgency_signal", "key_insight",
            "script_opening", "script_khai_thac", "script_goi_y", "script_chot", "script_upsell",
        ]
        export_df = df_filtered[[c for c in view_cols if c in df_filtered.columns]]
        csv = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Xuất CSV (kết quả lọc)",
            data=csv,
            file_name="instore_analysis.csv",
            mime="text/csv",
        )

    with col_e2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_filtered.to_excel(writer, sheet_name="full_analysis", index=False)
            # TVV quick view sheet
            tvv_cols = [
                "customer_id", "segment_rfm_tier", "budget",
                "instore_intent", "nba_strategy", "psychology_trigger", "priority",
                "urgency_signal", "key_insight",
                "product_rec_1", "product_rec_2", "product_rec_3",
                "script_opening", "script_khai_thac", "script_goi_y",
                "script_chot", "script_upsell",
            ]
            df_filtered[[c for c in tvv_cols if c in df_filtered.columns]].to_excel(
                writer, sheet_name="tvv_quick_view", index=False
            )
        st.download_button(
            "📥 Xuất Excel (đầy đủ + TVV view)",
            data=buf.getvalue(),
            file_name="instore_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TÌM KIẾM KHÁCH
# ══════════════════════════════════════════════════════════════════════════════

def render_tab_customer(df: pd.DataFrame) -> None:
    """Tab 3: Search and view one customer's full profile + script."""
    if df.empty:
        st.info("Chưa có dữ liệu phân tích.")
        return

    st.markdown("### 👤 Tra cứu khách hàng")
    st.caption("Chọn một khách để xem chi tiết hồ sơ và sales script đầy đủ.")

    # Customer selector via selectbox (searchable)
    customer_ids = sorted(df["customer_id"].astype(str).tolist())
    selected_id = st.selectbox(
        "Chọn khách hàng",
        options=customer_ids,
        format_func=lambda cid: (
            lambda row: (
                f"{cid} — {row.get('segment_rfm_tier', '')} "
                f"| {INTENT_ICONS.get(row.get('instore_intent', ''), '')} "
                f"{row.get('instore_intent', '')} "
                f"| {float(row.get('confidence', 0)):.0%}"
            )
        )(df[df["customer_id"] == cid].iloc[0].to_dict()
          if not df[df["customer_id"] == cid].empty else {}),
    )

    rows = df[df["customer_id"] == selected_id]
    if rows.empty:
        st.warning(f"Không tìm thấy khách: {selected_id}")
        return

    row = rows.iloc[0]

    st.divider()
    render_script_card(row, key_prefix="search")


# ══════════════════════════════════════════════════════════════════════════════
# CACHE STATUS BANNER
# ══════════════════════════════════════════════════════════════════════════════

def render_cache_banner(cache: Optional[dict], status: dict) -> bool:
    """
    Show a banner about the cache status.
    Returns True if the user clicked "Update" (trigger re-run).
    """
    if status["status"] == "no_cache":
        st.warning(
            "⚠️ Chưa có dữ liệu phân tích. "
            f"Cần phân tích **{len(status['new_ids'])} khách** từ Excel."
        )
        clicked = st.button(
            "🚀 Chạy phân tích lần đầu",
            type="primary",
            help="Sẽ chạy LEP + InstoreScriptEngine và lưu kết quả vào JSON cache",
        )
        return clicked

    if status["status"] == "outdated":
        n_new = len(status["new_ids"])
        st.warning(
            f"⚠️ Excel có **{n_new} khách mới** chưa có trong cache: "
            f"`{'`, `'.join(status['new_ids'][:5])}`"
            + (" ..." if n_new > 5 else "")
        )
        clicked = st.button(
            f"🔄 Cập nhật cache ({n_new} khách mới)",
            type="primary",
            help="Chỉ xử lý các khách mới, không chạy lại toàn bộ",
        )
        return clicked

    # status == "ok"
    generated_at = cache.get("generated_at", "")[:19].replace("T", " ") if cache else ""
    total        = cache.get("total_customers", 0) if cache else 0
    st.success(
        f"✅ Cache đầy đủ · {total} khách · "
        f"Cập nhật lúc: {generated_at}"
    )
    return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 💎 PNJ · In-Store NBA Dashboard")
    st.caption(
        "Nhánh 2 · In-Store Script Engine · "
        "Sales Script 5 Bước Cá Nhân Hoá cho Tư Vấn Viên"
    )
    st.divider()

    # ── Load Excel + Cache ────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        st.error(f"Không tìm thấy file dữ liệu: `{DATA_PATH}`")
        st.stop()

    excel_ids   = get_excel_customer_ids(DATA_PATH)
    cache       = load_cache()
    status      = get_cache_status(cache, excel_ids)
    df_analysis = cache_to_dataframe(cache) if cache else pd.DataFrame()

    # ── Sidebar (rendered first so api_key is available before banner) ────────
    sidebar = render_sidebar(df_analysis if not df_analysis.empty else None)
    api_key = sidebar.pop("api_key", "")   # extract key; remainder = filters
    filters = sidebar

    # ── Cache status banner ───────────────────────────────────────────────────
    should_update = render_cache_banner(cache, status)

    if should_update:
        if not api_key:
            st.info(
                "💡 Tip: Nhập **OpenAI API Key** ở sidebar để sinh script bằng GPT-4o.  \n"
                "Nếu không có key, hệ thống vẫn chạy với Fallback Template."
            )
        with st.spinner("⏳ Đang chạy pipeline... Vui lòng chờ."):
            try:
                cache = run_pipeline_and_update_cache(
                    excel_path=DATA_PATH,
                    new_customer_ids=(
                        status["new_ids"] if status["status"] == "outdated" else None
                    ),
                    existing_cache=cache,
                    api_key=api_key,
                )
                st.success("✅ Đã cập nhật cache thành công!")
                st.rerun()
            except Exception as err:
                st.error(f"❌ Lỗi khi chạy pipeline: {err}")
                st.exception(err)
                st.stop()

    # Refresh df_analysis after a successful update (st.rerun handles this,
    # but guard in case we reach here without rerun)
    if df_analysis.empty and cache:
        df_analysis = cache_to_dataframe(cache)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_raw, tab_analysis, tab_customer, tab_guide = st.tabs([
        "📂 Dữ liệu gốc",
        "🎯 Kết quả phân tích",
        "👤 Tra cứu khách",
        "📖 Hướng dẫn phân loại",
    ])

    with tab_raw:
        render_tab_raw(DATA_PATH)

    with tab_analysis:
        render_tab_analysis(df_analysis, filters)

    with tab_customer:
        render_tab_customer(df_analysis)

    with tab_guide:
        render_tab_guide()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "PNJ · In-Store NBA Engine · Nhánh 2 · "
        "LEP (RandomForest) + InstoreScriptEngine (GPT-4o / Fallback)"
    )


if __name__ == "__main__":
    main()
