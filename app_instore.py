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

# ── Load .env (nếu có) ────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv chưa cài — bỏ qua, dùng env var hệ thống

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.lep_pipeline import LEPModel, get_feature_importance, DEFAULT_MODEL_DIR
from src.instore_script_engine import (
    InstoreScriptEngine, InstoreIntentType,
    PSYCHOLOGY_TRIGGER_MAP,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH     = ROOT / "data" / "customer_data_poc_enhanced.xlsx"
CACHE_FILE    = ROOT / "outputs" / "instore_scripts.json"
MODEL_PATH    = DEFAULT_MODEL_DIR / "lep_model.pkl"
MSG_PLAN_PATH = ROOT / "outputs" / "nba_messages.json"

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

CHANNEL_LABELS = {
    "zns":    "💬 ZNS Zalo",
    "email":  "📧 Email",
    "push":   "📱 Push Notification",
    "in_app": "📲 In-App Banner",
    "store":  "🏪 Tại cửa hàng",
}

CHANNEL_COLORS = {
    "zns":    "#06b6d4",
    "email":  "#3b82f6",
    "push":   "#8b5cf6",
    "in_app": "#10b981",
    "store":  "#f59e0b",
}

TONE_LABELS = {
    "warm":      "🤗 Ấm áp",
    "urgent":    "⚡ Khẩn cấp",
    "luxurious": "👑 Sang trọng",
    "friendly":  "😊 Thân thiện",
}

SCRIPT_STEPS = [
    ("opening",   "1. Opening — Chào hỏi"),
    ("khai_thac", "2. Khai thác nhu cầu"),
    ("goi_y",     "3. Gợi ý sản phẩm"),
    ("chot",      "4. Chốt đơn"),
    ("upsell",    "5. Upsell / Bán thêm"),
]

# ── Walk-in form options ───────────────────────────────────────────────────────
_WALKIN_GENDER     = ["Nữ", "Nam"]
_WALKIN_AGE        = ["18–25 tuổi", "26–35 tuổi", "36–45 tuổi", "46+ tuổi"]
_WALKIN_STYLE      = [
    "Tối giản / Nhẹ nhàng",
    "Trẻ trung / Năng động",
    "Thanh lịch / Công sở",
    "Nổi bật / Cá tính",
]
_WALKIN_COMPANION  = [
    "Đi một mình",
    "Đi cùng bạn đời / người yêu",
    "Đi cùng bạn bè",
    "Đi cùng gia đình",
]
_WALKIN_ENGAGEMENT = [
    "Nhìn qua / Dừng xem tự nhiên",
    "Đang xem kỹ một sản phẩm",
    "Đã hỏi về sản phẩm cụ thể",
    "Đã hỏi giá / so sánh",
]
_WALKIN_PRODUCT    = ["Chưa rõ", "Nhẫn", "Bông tai", "Dây chuyền", "Vòng tay / Lắc", "Bộ trang sức"]
_WALKIN_BUDGET     = ["Chưa rõ", "Dưới 5 triệu", "5–15 triệu", "15–30 triệu", "Trên 30 triệu"]
_WALKIN_PURPOSE    = ["Chưa rõ", "Mua cho bản thân", "Mua tặng người thân", "Đang tham khảo giá"]
_WALKIN_OCCASION   = [
    "Chưa hỏi được",
    "Không có dịp cụ thể",
    "Sinh nhật (sắp đến)",
    "Kỷ niệm tình yêu / hôn nhân",
    "Cầu hôn / Đính hôn",
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

/* Outbound message card — inline styles handle per-channel theming */
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


@st.cache_data(show_spinner=False)
def load_message_plan(path: str) -> dict[str, dict]:
    """
    Load nba_messages.json → dict {customer_id: flat_msg_dict}.
    Flatten cấu trúc JSON thành dict phẳng để render_message_section dùng.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = {}
        for customer in data.get("customers", []):
            cid = str(customer.get("customer_id", ""))
            if not cid:
                continue
            ctx      = customer.get("context",  {})
            delivery = customer.get("delivery", {})
            content  = customer.get("content",  {})

            # Đảm bảo highlights là list
            raw_hl = content.get("highlights", [])
            if isinstance(raw_hl, str):
                try:
                    raw_hl = json.loads(raw_hl)
                except Exception:
                    raw_hl = [raw_hl] if raw_hl else []
            highlights = [str(h) for h in raw_hl if h]

            flat = {
                # Delivery info
                "customer_id":   cid,
                "channel":       str(delivery.get("channel",          "")),
                "priority":      str(delivery.get("priority",         "")),
                "campaign_id":   str(delivery.get("campaign_id",      "")),
                "rule_status":   str(delivery.get("rule_status",      "")),
                "product_focus": str(ctx.get("product_focus",          "")),
                # Message metadata
                "message_source": str(content.get("source",    "")),
                "tone":           str(content.get("tone",      "")),
                "tokens_used":    int(content.get("tokens_used", 0)),
                # Message content (structured)
                "llm_subject":    content.get("subject")  or "",
                "llm_greeting":   str(content.get("greeting", "") or ""),
                "llm_body":       str(content.get("body",     "") or ""),
                "llm_highlights": highlights,
                "llm_closing":    str(content.get("closing",  "") or ""),
                "llm_cta":        str(content.get("cta_text", "")),
                # Instore context (dùng để hiển thị trong detail view)
                "key_insight":        str(ctx.get("key_insight",        "")),
                "urgency_signal":     str(ctx.get("urgency_signal",     "")),
                "online_insight":     str(ctx.get("online_insight",     "")),
                "instore_intent":     str(ctx.get("instore_intent",     "")),
                "nba_strategy":       str(ctx.get("nba_strategy",       "")),
                "psychology_trigger": str(ctx.get("psychology_trigger", "")),
                "product_rec_1":      str(ctx.get("product_rec_1",      "")),
                "product_rec_2":      str(ctx.get("product_rec_2",      "")),
                "product_rec_3":      str(ctx.get("product_rec_3",      "")),
            }
            result[cid] = flat
        return result
    except Exception:
        return {}


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

    # Merge gender từ sheet profiles (profiles_enhanced không có cột gender)
    if "profiles" in sheets:
        df_gender = sheets["profiles"][["customer_id", "gender"]].copy()
        df_profiles = df_profiles.merge(df_gender, on="customer_id", how="left")
        df_profiles["gender"] = df_profiles["gender"].fillna("F")

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


def render_message_section(msg_data: dict) -> None:
    """
    Render outbound message — card thống nhất với viền bao toàn bộ nội dung.
    Mỗi channel (ZNS / Email / Push / In-App / Store) có layout riêng nhưng
    cùng chung khung card: header → body → footer.
    Toàn bộ HTML được build thành một chuỗi và render trong một st.markdown call.
    """
    # ── Trích xuất dữ liệu ───────────────────────────────────────────────────
    channel      = str(msg_data.get("channel", "")).lower()
    priority     = str(msg_data.get("priority", "")).lower()
    product_focus= str(msg_data.get("product_focus", "")).strip()
    subject_raw  = msg_data.get("llm_subject", "")
    greeting     = str(msg_data.get("llm_greeting", "") or "").strip()
    body         = str(msg_data.get("llm_body",    "") or "").strip()
    highlights   = msg_data.get("llm_highlights", [])
    closing      = str(msg_data.get("llm_closing", "") or "").strip()
    cta          = str(msg_data.get("llm_cta",     "") or "").strip()
    tone         = str(msg_data.get("tone",        "") or "").lower()
    campaign_id  = str(msg_data.get("campaign_id", ""))
    rule_status  = str(msg_data.get("rule_status", ""))
    msg_source   = str(msg_data.get("message_source", ""))

    # Đảm bảo highlights là list[str]
    if isinstance(highlights, str):
        try:
            highlights = json.loads(highlights)
        except Exception:
            highlights = [highlights] if highlights else []
    if not isinstance(highlights, list):
        highlights = []
    highlights = [str(h) for h in highlights if str(h).strip()]

    channel_label = CHANNEL_LABELS.get(channel, channel.upper())
    channel_color = CHANNEL_COLORS.get(channel, "#94a3b8")
    prio_color    = PRIORITY_COLORS.get(priority, "#6b7280")
    tone_label    = TONE_LABELS.get(tone, tone)
    subject       = str(subject_raw) if subject_raw and str(subject_raw) not in ("nan", "None", "") else ""

    # ── Meta badges ──────────────────────────────────────────────────────────
    src_html = (
        '<span style="background:#0c2218;color:#4ade80;border:1px solid #166534;'
        'border-radius:4px;font-size:10px;font-weight:700;padding:2px 8px;letter-spacing:.3px">'
        '🤖 GPT-4o</span>'
        if msg_source == "llm" else
        '<span style="background:#0c1e0c;color:#86efac;border:1px solid #14532d;'
        'border-radius:4px;font-size:10px;font-weight:700;padding:2px 8px;letter-spacing:.3px">'
        '📄 Template</span>'
    )
    rule_html = (
        f'<span style="color:#ef4444;font-size:11px;font-weight:600">🚫 {rule_status.upper()}</span>'
        if rule_status and rule_status != "allowed" else ""
    )

    # ── Channel display helpers ───────────────────────────────────────────────
    channel_icon_map = {
        "zns": "💬", "email": "📧", "push": "📱", "in_app": "📲", "store": "🏪",
    }
    channel_name_map = {
        "zns":    "ZNS Zalo",
        "email":  "Email",
        "push":   "Push Notification",
        "in_app": "In-App Banner",
        "store":  "Script Tại Cửa Hàng",
    }
    ch_icon = channel_icon_map.get(channel, "📨")
    ch_name = channel_name_map.get(channel, channel.upper())

    # ══════════════════════════════════════════════════════════════════════════
    # Build nội dung theo từng channel
    # ══════════════════════════════════════════════════════════════════════════

    content_html = ""

    # ── ZNS ──────────────────────────────────────────────────────────────────
    if channel == "zns":
        if greeting:
            content_html += (
                f'<p style="color:#e2e8f0;font-size:13.5px;line-height:1.8;margin:0 0 12px 0">'
                f'{greeting}</p>'
            )
        if body:
            content_html += (
                f'<p style="color:#cbd5e1;font-size:13px;line-height:1.8;margin:0 0 12px 0">'
                f'{body}</p>'
            )
        if highlights:
            hl_items = "".join(
                f'<div style="color:#67e8f9;font-size:13px;font-weight:600;'
                f'padding:4px 0;line-height:1.55"> {h}</div>'
                for h in highlights
            )
            content_html += (
                f'<div style="background:#050f1c;border-left:3px solid #06b6d4;'
                f'border-radius:0 6px 6px 0;padding:10px 14px;margin:4px 0 12px 0">'
                f'{hl_items}</div>'
            )
        if closing:
            closing_lines = [p.strip() for p in closing.split("\n") if p.strip()]
            closing_body  = ""
            for part in closing_lines:
                if part.startswith("*"):
                    closing_body += (
                        f'<div style="color:#64748b;font-size:12px;font-style:italic;'
                        f'padding:2px 0">{part}</div>'
                    )
                else:
                    closing_body += (
                        f'<div style="color:#94a3b8;font-size:13px;padding:2px 0">{part}</div>'
                    )
            content_html += (
                f'<div style="border-top:1px solid #1a2840;padding-top:10px;margin-top:6px">'
                f'{closing_body}</div>'
            )

    # ── Email ─────────────────────────────────────────────────────────────────
    elif channel == "email":
        if subject:
            content_html += (
                f'<div style="background:#0b1929;border-left:3px solid #3b82f6;'
                f'border-radius:0 6px 6px 0;padding:10px 14px;margin-bottom:14px">'
                f'<div style="color:#475569;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:.5px;margin-bottom:4px">Tiêu đề email</div>'
                f'<div style="color:#f1f5f9;font-size:14px;font-weight:600;line-height:1.4">'
                f'{subject}</div>'
                f'</div>'
            )
        if greeting:
            content_html += (
                f'<p style="color:#94a3b8;font-size:13px;margin:0 0 10px 0">{greeting}</p>'
            )
        if body:
            paragraphs = [
                p.strip()
                for p in body.replace("\\n\\n", "\n\n").split("\n\n")
                if p.strip()
            ]
            for para in paragraphs:
                content_html += (
                    f'<p style="color:#e2e8f0;font-size:13px;line-height:1.8;'
                    f'margin:0 0 10px 0">{para}</p>'
                )
        if highlights:
            hl_items = "".join(
                f'<div style="color:#93c5fd;font-size:13px;padding:4px 0;line-height:1.55">'
                f' {h}</div>'
                for h in highlights
            )
            content_html += (
                f'<div style="background:#091629;border-left:3px solid #3b82f6;'
                f'border-radius:0 6px 6px 0;padding:10px 14px;margin:4px 0 12px 0">'
                f'{hl_items}</div>'
            )
        if closing:
            content_html += (
                f'<div style="border-top:1px solid #1a2840;padding-top:10px;margin-top:6px;'
                f'color:#64748b;font-size:12px;font-style:italic;line-height:1.65">'
                f'{closing}</div>'
            )

    # ── Push ──────────────────────────────────────────────────────────────────
    elif channel == "push":
        notif_title = (
            f'<div style="color:#f1f5f9;font-size:13px;font-weight:700;'
            f'margin-bottom:5px;line-height:1.4">{subject}</div>'
            if subject else ""
        )
        notif_body = (
            f'<div style="color:#cbd5e1;font-size:12px;line-height:1.65">{body}</div>'
            if body else ""
        )
        content_html = (
            # Phone mockup shell
            f'<div style="display:flex;justify-content:center">'
            f'<div style="background:#100e26;border:1.5px solid #312e81;border-radius:16px;'
            f'padding:14px 16px;width:340px;box-shadow:0 4px 24px #00000066">'
            # App row
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">'
            f'<div style="width:26px;height:26px;background:linear-gradient(135deg,#d4af37,#8b6914);'
            f'border-radius:7px;display:flex;align-items:center;justify-content:center;'
            f'font-size:14px;flex-shrink:0">💎</div>'
            f'<span style="color:#94a3b8;font-size:11px;font-weight:600;flex:1">PNJ Jewelry</span>'
            f'<span style="color:#4b5563;font-size:10px">Vừa xong</span>'
            f'</div>'
            f'{notif_title}'
            f'{notif_body}'
            f'</div></div>'
        )

    # ── In-App ────────────────────────────────────────────────────────────────
    elif channel == "in_app":
        left_html = ""
        if subject:
            left_html += (
                f'<div style="font-size:15px;font-weight:700;color:#d1fae5;'
                f'line-height:1.3;margin-bottom:8px">{subject}</div>'
            )
        if body:
            left_html += (
                f'<div style="font-size:12px;color:#a7f3d0;line-height:1.7;'
                f'margin-bottom:10px">{body}</div>'
            )
        if highlights:
            hl_items = "".join(
                f'<div style="font-size:11.5px;color:#6ee7b7;padding:2px 0"> {h}</div>'
                for h in highlights[:3]
            )
            left_html += f'<div style="margin-bottom:12px">{hl_items}</div>'
        if cta:
            left_html += (
                f'<div style="background:#059669;color:#fff;font-size:12px;font-weight:700;'
                f'padding:7px 16px;border-radius:8px;display:inline-block;letter-spacing:.3px">'
                f'{cta} →</div>'
            )
        content_html = (
            f'<div style="background:linear-gradient(135deg,#0a2e1e,#051508);'
            f'border:1px solid #065f46;border-radius:10px;padding:18px 20px">'
            f'{left_html}</div>'
        )

    # ── Store ─────────────────────────────────────────────────────────────────
    elif channel == "store":
        if subject:
            content_html += (
                f'<div style="background:#100a00;border-left:3px solid #f59e0b;'
                f'border-radius:0 6px 6px 0;padding:9px 13px;margin-bottom:14px">'
                f'<div style="color:#fde68a;font-size:13px;font-weight:600">{subject}</div>'
                f'</div>'
            )
        if greeting:
            content_html += (
                f'<div style="margin-bottom:14px">'
                f'<div style="font-size:10px;color:#475569;text-transform:uppercase;'
                f'letter-spacing:.6px;margin-bottom:6px">💬 Câu mở đầu với khách</div>'
                f'<div style="background:#0d1407;border-left:3px solid #f59e0b;'
                f'border-radius:0 8px 8px 0;padding:10px 14px;color:#e2e8f0;'
                f'font-size:13px;line-height:1.7">{greeting}</div>'
                f'</div>'
            )
        if body:
            content_html += (
                f'<div style="margin-bottom:14px">'
                f'<div style="font-size:10px;color:#475569;text-transform:uppercase;'
                f'letter-spacing:.6px;margin-bottom:6px">🗣️ Script tư vấn</div>'
                f'<div style="color:#cbd5e1;font-size:13px;line-height:1.8">{body}</div>'
                f'</div>'
            )
        if highlights:
            hl_items = "".join(
                f'<div style="color:#fde68a;font-size:13px;padding:4px 0;line-height:1.5">'
                f'📦 {h}</div>'
                for h in highlights
            )
            content_html += (
                f'<div style="background:#0b0e02;border-radius:8px;padding:10px 14px;'
                f'margin-bottom:12px">{hl_items}</div>'
            )
        if closing:
            content_html += (
                f'<div style="border-top:1px solid #1e2818;padding-top:10px;margin-top:4px">'
                f'<div style="font-size:10px;color:#475569;text-transform:uppercase;'
                f'letter-spacing:.6px;margin-bottom:5px">➡️ Bước tiếp theo</div>'
                f'<div style="color:#94a3b8;font-size:12px;font-style:italic;line-height:1.65">'
                f'{closing}</div>'
                f'</div>'
            )

    # ── Generic fallback ──────────────────────────────────────────────────────
    else:
        if subject:
            content_html += (
                f'<div style="color:#94a3b8;font-size:12px;margin-bottom:8px">'
                f'<strong>Tiêu đề:</strong> {subject}</div>'
            )
        if body:
            content_html += (
                f'<div style="color:#e2e8f0;font-size:13px;line-height:1.75">{body}</div>'
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Build footer
    # ══════════════════════════════════════════════════════════════════════════
    footer_parts = []
    if cta:
        footer_parts.append(
            f'<span style="background:#0c1829;border:1px solid #3b82f6;border-radius:20px;'
            f'padding:4px 14px;color:#93c5fd;font-size:12px;font-weight:600">👆 {cta}</span>'
        )
    if tone_label:
        footer_parts.append(
            f'<span style="color:#475569;font-size:12px">Giọng điệu: '
            f'<span style="color:#a78bfa;font-weight:500">{tone_label}</span></span>'
        )
    if rule_html:
        footer_parts.append(f'<span style="margin-left:auto">{rule_html}</span>')

    footer_html = ""
    if footer_parts:
        footer_html = (
            f'<div style="padding:10px 20px 12px;background:#06080f;'
            f'border-top:1px solid #151e35">'
            f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center">'
            + "".join(footer_parts)
            + "</div></div>"
        )

    # ── Product row (header) ──────────────────────────────────────────────────
    product_chip = ""
    if product_focus:
        product_chip = (
            f'<span style="color:#475569;font-size:11px"> · </span>'
            f'<span style="background:#1c1500;border:1px solid #854d0e;border-radius:20px;'
            f'padding:2px 9px;color:#fde68a;font-size:11px;font-weight:600">'
            f'📦 {product_focus}</span>'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Assemble full card — ONE render call, viền bao trọn toàn bộ nội dung
    # ══════════════════════════════════════════════════════════════════════════
    card_html = f"""
<div style="background:#0c1020;border:1.5px solid #1e2848;border-radius:14px;
            overflow:hidden;margin:18px 0 6px 0;
            box-shadow:0 2px 20px #00000055">

  <!-- ▌Accent gradient line theo màu channel -->
  <div style="height:3px;background:linear-gradient(90deg,{channel_color}cc,{channel_color}44 60%,transparent)"></div>

  <!-- ▌HEADER -->
  <div style="padding:14px 20px 12px;border-bottom:1px solid #151e35">
    <!-- Row 1: Tiêu đề + source badge -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:9px">
      <span style="font-size:11px;font-weight:800;color:#d4af37;
                   text-transform:uppercase;letter-spacing:1.1px">
        📨 Tin Nhắn Outbound — Nhánh 2 (Insight-Driven)
      </span>
      {src_html}
    </div>
    <!-- Row 2: Priority + Channel + Campaign + Product -->
    <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">
      <span style="background:{prio_color}22;color:{prio_color};
                   border:1px solid {prio_color}55;border-radius:20px;
                   padding:2px 10px;font-size:11px;font-weight:700">
        {priority.upper()}
      </span>
      <span style="background:{channel_color}22;color:{channel_color};
                   border:1px solid {channel_color}55;border-radius:20px;
                   padding:2px 10px;font-size:11px;font-weight:600">
        {channel_label}
      </span>
      <span style="color:#3d4f6b;font-size:11px"> · </span>
      <span style="color:#3d4f6b;font-size:11px">Chiến dịch:&nbsp;
        <span style="color:#64748b;font-weight:500">{campaign_id}</span>
      </span>
      {product_chip}
    </div>
  </div>

  <!-- ▌BODY -->
  <div style="padding:16px 20px 14px">
    <!-- Channel label -->
    <div style="font-size:10px;font-weight:700;color:{channel_color};
                text-transform:uppercase;letter-spacing:.9px;margin-bottom:11px">
      {ch_icon} Nội dung {ch_name}
    </div>
    <!-- Nội dung tin nhắn -->
    <div style="background:#07091500;border:1px solid #151e35;border-radius:10px;
                padding:16px 18px">
      {content_html if content_html.strip() else '<span style="color:#3d4f6b;font-size:12px;font-style:italic">Chưa có nội dung</span>'}
    </div>
  </div>

  <!-- ▌FOOTER -->
  {footer_html}

</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)


def render_script_card(row: pd.Series, key_prefix: str = "card", msg_data: Optional[dict] = None) -> None:
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

    # ── Tin nhắn Outbound (Nhánh 1) — chỉ hiện khi được truyền vào ──────────
    if msg_data:
        render_message_section(msg_data)
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
# TAB "KHÁCH MỚI" — Walk-in Instant Script Builder
# ══════════════════════════════════════════════════════════════════════════════

def _classify_walkin_intent(obs: dict) -> tuple[InstoreIntentType, str]:
    """Phân loại instore intent từ tín hiệu quan sát tại cửa hàng."""
    age        = obs.get("age", "")
    style      = obs.get("style", "")
    budget     = obs.get("budget", "Chưa rõ")
    engagement = obs.get("engagement", "")
    purpose    = obs.get("purpose", "Chưa rõ")
    occasion   = obs.get("occasion", "Chưa hỏi được")
    companion  = obs.get("companion", "")

    is_older          = age in ["36–45 tuổi", "46+ tuổi"]
    is_premium_budget = budget == "Trên 30 triệu"
    is_premium_style  = any(k in style for k in ["Thanh lịch", "Nổi bật"])
    high_engagement   = engagement in ["Đã hỏi về sản phẩm cụ thể", "Đã hỏi giá / so sánh"]

    # PREMIUM
    if is_premium_budget and (is_premium_style or is_older):
        return InstoreIntentType.PREMIUM, "Tư vấn như VIP"
    if is_premium_budget and high_engagement:
        return InstoreIntentType.PREMIUM, "Tư vấn như VIP"

    # HIGH PURCHASE
    not_browsing = purpose not in ["Đang tham khảo giá", "Chưa rõ"]
    if high_engagement and (not_browsing or budget not in ["Chưa rõ", "Dưới 5 triệu"]):
        return InstoreIntentType.HIGH_PURCHASE, "Chốt đơn nhanh"

    # EXPLORATION
    has_occasion    = occasion not in ["Không có dịp cụ thể", "Chưa hỏi được"]
    is_couple       = any(k in companion for k in ["bạn đời", "người yêu"])
    viewing_closely = engagement in ["Đang xem kỹ một sản phẩm", "Đã hỏi về sản phẩm cụ thể"]
    has_budget      = budget != "Chưa rõ"

    if has_occasion or is_couple or (viewing_closely and has_budget):
        return InstoreIntentType.EXPLORATION, "Định hướng lựa chọn"
    if purpose in ["Mua cho bản thân", "Mua tặng người thân"]:
        return InstoreIntentType.EXPLORATION, "Định hướng lựa chọn"

    return InstoreIntentType.LOW_INTENT, "Tạo trải nghiệm"


# ── Walk-in script generation (OpenAI / fallback) ────────────────────────────

WALKIN_SYSTEM_PROMPT = """Bạn là một Tư Vấn Viên trang sức chuyên nghiệp của PNJ — không phải người viết kịch bản, mà là người đang trực tiếp đứng quầy và nói chuyện với khách.
Nhiệm vụ: Viết lời thoại cụ thể, tự nhiên cho từng bước tiếp cận khách walk-in mới tại cửa hàng.

NGUYÊN TẮC XƯNG HÔ — BẮT BUỘC TUYỆT ĐỐI:
• TVV luôn tự xưng là "em" — đây là chuẩn lịch sự trong bán hàng trang sức Việt Nam
• Khách LUÔN được gọi là "anh" (Nam) hoặc "chị" (Nữ) — TUYỆT ĐỐI không gọi khách là "em" dù khách còn rất trẻ (18–22 tuổi)
• Lý do: Gọi khách là "em" dù họ trẻ hơn mình là thiếu tôn trọng — trong trang sức cao cấp mọi khách đều là "anh/chị"
• Ví dụ đúng: "Anh thử lên tay xem nhé, em chọn mẫu này vì..." | "Chị đang hướng đến phong cách gì ạ, em tư vấn thêm cho?"
• Ví dụ SAI: "Em thấy bạn hợp với mẫu này" | "Bạn thích kiểu nào?" | "Em ơi anh tư vấn cho" | gọi khách là "bạn" hay "em"

QUY TẮC NỘI DUNG:
1. Giọng điệu: Tự nhiên như người bạn am hiểu trang sức — không cứng nhắc, không sáo rỗng, không lộ kịch bản
2. Thông tin QUAN SÁT được (giới tính, tuổi, phong cách, đến cùng ai, đang xem gì, mức độ hứng thú):
   → Đã biết rồi — KHÔNG hỏi lại, chỉ dùng để điều chỉnh cách tiếp cận và gợi ý
3. Thông tin CẦN KHAI THÁC (mục đích, dịp đặc biệt, ngân sách — nếu đánh dấu "Chưa rõ"):
   → Bước khai_thac phải dẫn dắt tự nhiên qua câu hỏi mở lồng ghép trong câu chuyện — KHÔNG hỏi thẳng, KHÔNG checklist
   → Khai thác MỤC ĐÍCH: "Anh/chị đang tìm để đeo hàng ngày hay có dịp gì đặc biệt sắp tới không ạ?"
   → Khai thác NGÂN SÁCH: "Anh/chị muốn hướng đến mẫu nhẹ nhàng tinh tế hay có chút điểm nhấn nổi bật hơn?" (đừng hỏi thẳng giá)
   → Khai thác DỊP ĐẶC BIỆT: "Nhìn là biết anh/chị đang chọn quà rồi — người nhận thích phong cách gì để em chọn đúng hơn ạ?"
   → Chỉ hỏi 1 câu mỗi lúc — hỏi xong để khách trả lời, đừng dồn nhiều câu liên tiếp
4. Nếu đã có đủ thông tin: đề xuất sản phẩm cụ thể và chốt tự nhiên, không vòng vo
5. TUYỆT ĐỐI không nhắc lịch sử mua online, CRM, app, hay hệ thống — đây là khách walk-in hoàn toàn mới
6. Mỗi bước 2–3 câu ngắn gọn, nói được ngay — không cần đọc dài
7. Trả về ĐÚNG format JSON — không thêm markdown hay giải thích ngoài JSON"""

_WALKIN_JSON_FORMAT = """{
  "intent_label": "High Purchase | Exploration | Premium | Low Intent",
  "key_insight": "1 câu tóm tắt điều TVV cần nhớ nhất về khách này",
  "opening": "...",
  "khai_thac": "...",
  "goi_y": "...",
  "chot": "...",
  "upsell": "...",
  "product_recommendations": ["gợi ý sản phẩm 1", "gợi ý sản phẩm 2", "gợi ý sản phẩm 3"]
}"""


def _build_walkin_user_prompt(obs: dict) -> str:
    """Build user prompt for walk-in customer based purely on in-store observations."""
    gender       = obs.get("gender", "Nữ")
    age          = obs.get("age", "26–35 tuổi")
    style        = obs.get("style", "Tối giản / Nhẹ nhàng")
    companion    = obs.get("companion", "Đi một mình")
    engagement   = obs.get("engagement", "Nhìn qua / Dừng xem tự nhiên")
    product_type = obs.get("product_type", "Chưa rõ")
    budget       = obs.get("budget", "Chưa rõ")
    purpose      = obs.get("purpose", "Chưa rõ")
    occasion     = obs.get("occasion", "Chưa hỏi được")

    intent, nba_strategy = _classify_walkin_intent(obs)
    psych = PSYCHOLOGY_TRIGGER_MAP.get(intent, "No Pressure")
    pron  = "anh" if gender == "Nam" else "chị"

    observable_lines = [
        f"• Giới tính: {gender}",
        f"• Độ tuổi ước tính: {age}",
        f"• Phong cách ăn mặc quan sát: {style}",
        f"• Đến cùng: {companion}",
        f"• Mức độ hứng thú: {engagement}",
    ]
    if product_type != "Chưa rõ":
        observable_lines.append(f"• Đang xem / quan tâm đến: {product_type}")

    collected_lines = []

    # Mục đích — field 1 của product selector
    purpose_opts = [
        "Mua cho bản thân", "Mua tặng bạn bè / người thân",
        "Quà cầu hôn / đính hôn", "Kỷ niệm tình yêu / hôn nhân", "Sinh nhật sắp đến",
    ]
    purpose_known = purpose != "Chưa rõ"
    if purpose_known:
        collected_lines.append(f"• [ĐÃ BIẾT] Mục đích: {purpose}")
    if occasion not in ["Chưa hỏi được", "Không có dịp cụ thể"]:
        collected_lines.append(f"• [ĐÃ BIẾT] Dịp đặc biệt: {occasion}")

    # Ngân sách — field 2 của product selector
    budget_opts = ["Dưới 5 triệu", "5–15 triệu", "15–30 triệu", "Trên 30 triệu"]
    budget_known = budget != "Chưa rõ"
    if budget_known:
        collected_lines.append(f"• [ĐÃ BIẾT] Ngân sách: {budget}")

    # Xây danh sách cần khai thác
    needed: list[tuple[str, str, str]] = []  # (field, gợi ý hỏi gián tiếp, các lựa chọn)
    if not purpose_known:
        needed.append((
            "Mục đích mua",
            f"Hỏi tự nhiên để biết {pron} mua cho bản thân hay tặng ai, có dịp đặc biệt không",
            " / ".join(purpose_opts),
        ))
    if not budget_known:
        needed.append((
            "Ngân sách",
            f"Dùng câu hỏi về phong cách / mức độ nổi bật để suy ra tầm tiền — "
            f"không hỏi thẳng số tiền",
            " / ".join(budget_opts),
        ))

    lines = [
        f"PHÂN LOẠI KHÁCH: {intent.value}",
        f"CHIẾN LƯỢC: {nba_strategy}",
        f"TÂM LÝ HỌC ÁP DỤNG: {psych}",
        "",
        "─── TVV QUAN SÁT ĐƯỢC (nhìn trực tiếp, không hỏi lại) ────────",
        *observable_lines,
    ]

    if collected_lines:
        lines += ["", "─── ĐÃ THU THẬP ĐƯỢC ─────────────────────────────────────"]
        lines += collected_lines

    if needed:
        lines += [
            "",
            "─── MỤC TIÊU KHAI THÁC TRONG BƯỚC khai_thac ─────────────",
            "Bước khai_thac phải giúp TVV tự nhiên xác định được các thông tin sau",
            "(TVV sẽ tick vào form để hệ thống gợi ý sản phẩm phù hợp):",
            "",
        ]
        for i, (field, hint, opts) in enumerate(needed, 1):
            lines += [
                f"  [{i}] {field}",
                f"      Các lựa chọn cần xác định: {opts}",
                f"      Cách dẫn dắt: {hint}",
            ]
        lines += [
            "",
            "⚠️ QUY TẮC viết bước khai_thac:",
            f"   • Chỉ đặt 1 câu hỏi đầu tiên — câu hỏi mở, lồng vào ngữ cảnh quan sát",
            f"   • Dùng chi tiết đã thấy (phong cách {style}, đang xem {product_type or 'trang sức'}, "
            f"đến cùng {companion}) để câu hỏi nghe tự nhiên, không như phỏng vấn",
            f"   • KHÔNG hỏi: 'Ngân sách bao nhiêu?' / 'Mua dịp gì?' / 'Mục đích là gì?'",
            f"   • TVV xưng 'em', gọi khách là '{pron}' — không gọi khách là 'em' hay 'bạn'",
        ]
    else:
        lines += [
            "",
            "─── ĐÃ ĐỦ THÔNG TIN ──────────────────────────────────────",
            "Bỏ qua bước khai thác — sinh script chốt đơn trực tiếp với sản phẩm cụ thể.",
        ]

    lines += [
        "",
        "─── YÊU CẦU ──────────────────────────────────────────────────",
        "Sinh Sales Script 5 bước tự nhiên, TVV có thể nói ngay tại quầy.",
        f"Khách: {gender}, {age} — TVV xưng 'em', gọi khách là '{pron}' xuyên suốt.",
        "",
        _WALKIN_JSON_FORMAT,
    ]
    return "\n".join(lines)


def _walkin_fallback_script(obs: dict, intent: InstoreIntentType, nba_strategy: str) -> dict:
    """Gender-aware, context-aware fallback template for walk-in customers."""
    gender       = obs.get("gender", "Nữ")
    companion    = obs.get("companion", "Đi một mình")
    product_type = obs.get("product_type", "Chưa rõ")
    budget       = obs.get("budget", "Chưa rõ")
    purpose      = obs.get("purpose", "Chưa rõ")
    occasion     = obs.get("occasion", "Chưa hỏi được")

    pron     = "anh" if gender == "Nam" else "chị"
    pron_cap = pron.capitalize()
    is_couple    = any(k in companion for k in ["bạn đời", "người yêu"])
    has_purpose  = purpose != "Chưa rõ"
    has_occasion = occasion not in ["Chưa hỏi được", "Không có dịp cụ thể"]
    has_budget   = budget != "Chưa rõ"
    prod_str     = product_type if product_type != "Chưa rõ" else "trang sức"

    # Khai thác: câu hỏi đầu tiên nhắm vào mục đích mua (trường quan trọng nhất cho gợi ý sản phẩm)
    # Dùng context quan sát được để câu hỏi nghe tự nhiên, không như phỏng vấn
    style        = obs.get("style", "")
    khai_parts = []

    # Ưu tiên 1 — mục đích mua (ảnh hưởng trực tiếp đến product recommendation)
    if not has_purpose and not has_occasion:
        if is_couple:
            khai_parts.append(
                f"Hai người đang xem cùng nhau — {pron} đang tìm gì đó cho hai người "
                f"hay có dịp đặc biệt nào sắp tới không ạ?"
            )
        elif product_type != "Chưa rõ":
            khai_parts.append(
                f"{pron_cap} đang xem {product_type} — đây là để đeo hàng ngày "
                f"hay {pron} đang tìm quà cho ai không ạ?"
            )
        else:
            khai_parts.append(
                f"{pron_cap} đang chọn cho bản thân hay muốn tìm quà tặng ai ạ?"
            )
    elif not has_purpose:
        if is_couple:
            khai_parts.append(
                f"Hai người đang chọn cho cả hai hay {pron} tìm quà tặng ai đó ạ?"
            )
        elif product_type != "Chưa rõ":
            khai_parts.append(
                f"{product_type} này {pron} đang chọn để tự đeo hay tặng ai ạ?"
            )
        else:
            khai_parts.append(f"{pron_cap} đang chọn cho bản thân hay muốn tặng ai ạ?")
    elif not has_occasion:
        khai_parts.append(
            f"Có dịp gì đặc biệt sắp đến không ạ — "
            f"để em chọn mẫu cho {pron} thật ý nghĩa?"
        )

    # Ưu tiên 2 — ngân sách, suy ra qua phong cách (không hỏi thẳng số tiền)
    if not has_budget:
        if any(k in style for k in ["Nổi bật", "Cá tính"]):
            khai_parts.append(
                f"Nhìn phong cách của {pron} là biết thích mẫu có điểm nhấn — "
                f"{pron_cap} muốn hướng đến mẫu tinh tế hay thêm chút điểm nhấn đặc biệt hơn ạ?"
            )
        elif any(k in style for k in ["Thanh lịch", "Công sở"]):
            khai_parts.append(
                f"{pron_cap} thích mẫu thanh lịch vừa phải hay có chút nổi bật để gây ấn tượng hơn ạ?"
            )
        else:
            khai_parts.append(
                f"{pron_cap} thích phong cách tinh tế nhẹ nhàng hay muốn nổi bật hơn một chút ạ?"
            )
    elif not khai_parts:
        khai_parts.append(
            f"Với tầm {budget}, bên em có nhiều lựa chọn rất đẹp — "
            f"để em giới thiệu mấy mẫu phù hợp nhất nhé."
        )

    khai_thac = " ".join(khai_parts) if khai_parts else (
        f"{pron_cap} thích phong cách tinh tế hay nổi bật hơn ạ?"
    )

    templates: dict[InstoreIntentType, dict] = {
        InstoreIntentType.HIGH_PURCHASE: {
            "opening": (
                f"Chào {pron}! {pron_cap} đang xem {prod_str}"
                + (" cho hai người" if is_couple else "")
                + f" đúng không ạ? Mẫu đó bên em vẫn còn — để em lấy ra cho {pron} xem thử ngay nhé."
            ),
            "khai_thac": khai_thac,
            "goi_y": (
                f"Đây là 3 mẫu {prod_str} đang được khách ưa chuộng nhất — "
                f"tuần này đã có mấy khách lấy mẫu này rồi. {pron_cap} thử lên tay xem nhé."
            ),
            "chot": (
                (f"Với tầm {budget}, mẫu này rất hợp lý — " if has_budget else "")
                + f"hàng chỉ còn số lượng có hạn thôi {pron} ạ. Để em wrap luôn nhé?"
            ),
            "upsell": (
                f"Nếu {pron} muốn nổi bật hơn một chút, bên em có phiên bản đính đá nhỏ — "
                f"nhìn sang hơn mà giá chênh không nhiều đâu ạ."
            ),
            "key_insight": "Khách hứng thú rõ — ưu tiên cho thử ngay, dùng social proof để chốt nhanh.",
        },
        InstoreIntentType.EXPLORATION: {
            "opening": (
                f"Chào {pron}! Cứ thoải mái xem {pron} nhé, có gì em hỗ trợ ngay ạ. "
                + (f"{pron_cap} đang tìm {prod_str} cho hai người hay cho cá nhân ạ?" if is_couple
                   else f"Bên em vừa về đợt {prod_str} mới — mẫu khá đẹp {pron} ạ.")
            ),
            "khai_thac": khai_thac,
            "goi_y": (
                f"Dựa vào những gì {pron} vừa chia sẻ, em chọn ra 3 mẫu phù hợp nhất — "
                f"không đưa quá nhiều để {pron} dễ quyết định hơn. {pron_cap} xem qua nhé."
            ),
            "chot": f"Trong 3 mẫu này, {pron} thấy mẫu nào hợp nhất để thử lên tay ạ?",
            "upsell": (
                f"Nếu {pron} muốn phối thêm, bên em có mẫu matching set rất hợp — "
                f"đeo cùng nhìn hoàn thiện hơn ạ."
            ),
            "key_insight": "Khách đang tìm hiểu — thu hẹp còn 3 mẫu, tránh đưa quá nhiều cùng lúc.",
        },
        InstoreIntentType.PREMIUM: {
            "opening": (
                f"Xin chào {pron}! Mời {pron} vào xem thoải mái nhé. "
                f"Bên em vừa có đợt hàng đặc biệt — một số mẫu chỉ có số lượng rất hạn chế."
            ),
            "khai_thac": (
                f"{pron_cap} đang hướng tới phong cách nào — bộ sưu tập cao cấp hay muốn thứ gì độc đáo riêng? "
                f"Để em chuẩn bị vài mẫu cho {pron} xem riêng ạ."
            ),
            "goi_y": (
                f"Đây là mẫu bên em nhập số lượng rất ít — thiết kế riêng, không đại trà. "
                f"{pron_cap} có gu nên em muốn {pron} xem trước."
            ),
            "chot": (
                f"Bên em có dịch vụ khắc tên và hộp quà riêng cho những đơn đặc biệt — "
                f"{pron} muốn em chuẩn bị không ạ?"
            ),
            "upsell": (
                f"Nếu {pron} muốn hoàn thiện bộ trang sức, bên em có phiên bản full set — "
                f"đeo cùng trông sang trọng và hoàn chỉnh hơn nhiều."
            ),
            "key_insight": "Khách cao cấp — ưu tiên trải nghiệm riêng tư, không vội, tạo cảm giác độc quyền.",
        },
        InstoreIntentType.LOW_INTENT: {
            "opening": (
                f"Chào {pron}! {pron_cap} cứ tự nhiên xem nhé — bên em vừa về mẫu mới tháng này, "
                f"có gì hay em giới thiệu thêm cho {pron} ạ."
            ),
            "khai_thac": (
                f"{pron_cap} thích phong cách trang sức như thế nào — đeo hàng ngày hay dành cho dịp đặc biệt ạ? "
                f"Em có thể gợi ý vài hướng cho {pron} tham khảo."
            ),
            "goi_y": (
                f"Để em đưa ra vài mẫu đang được khách thích nhất gần đây — "
                f"{pron} thử lên tay xem cảm giác thế nào, không nhất thiết phải quyết định ngay ạ."
            ),
            "chot": (
                f"{pron_cap} thấy mẫu nào ưng nhất? Không cần quyết định hôm nay đâu — "
                f"{pron} cứ thoải mái xem thêm ạ."
            ),
            "upsell": (
                f"Nếu hôm nay chưa quyết định, em lưu lại mẫu {pron} thích — "
                f"lần sau ghé không phải tìm lại từ đầu nhé."
            ),
            "key_insight": "Khách chưa rõ nhu cầu — tạo thiện cảm, không áp lực, mục tiêu để khách quay lại.",
        },
    }

    base = templates.get(intent, templates[InstoreIntentType.LOW_INTENT]).copy()

    if is_couple:
        recs = ["Nhẫn đôi / Couple Ring", "Dây chuyền đôi phong cách", "Bộ trang sức mini matching"]
    elif has_occasion and "Cầu hôn" in occasion:
        recs = ["Nhẫn đính hôn kim cương", "Nhẫn đôi phong cách tối giản", "Dịch vụ khắc tên + hộp quà"]
    elif has_occasion and "Sinh nhật" in occasion:
        recs = [prod_str, "Hộp quà sinh nhật cao cấp", "Dịch vụ gói quà đặc biệt"]
    elif "tặng" in purpose:
        recs = [prod_str, "Gift Set ý nghĩa", "Dịch vụ khắc tên + hộp quà"]
    else:
        recs = [
            prod_str if prod_str != "trang sức" else "Bộ sưu tập mới nhất tháng này",
            "Mẫu bestseller dễ đeo hàng ngày",
            "Phiên bản nâng cấp / đính đá",
        ]

    base.update({
        "product_recommendations": recs,
        "intent_label":       intent.value,
        "nba_strategy":       nba_strategy,
        "psychology_trigger": PSYCHOLOGY_TRIGGER_MAP.get(intent, ""),
    })
    return base


def _generate_walkin_script(obs: dict, api_key: str = "") -> tuple[dict, str]:
    """Generate walk-in script via OpenAI API or fallback template."""
    intent, nba_strategy = _classify_walkin_intent(obs)

    if api_key.strip():
        try:
            from openai import OpenAI  # noqa: PLC0415
            client      = OpenAI(api_key=api_key.strip())
            user_prompt = _build_walkin_user_prompt(obs)

            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=900,
                messages=[
                    {"role": "system", "content": WALKIN_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            parsed.setdefault("nba_strategy",       nba_strategy)
            parsed.setdefault("psychology_trigger", PSYCHOLOGY_TRIGGER_MAP.get(intent, ""))
            parsed.setdefault("intent_label",       intent.value)
            return parsed, "llm"
        except Exception as exc:
            st.warning(f"GPT-4o gặp lỗi: {exc}. Chuyển sang fallback template.")

    return _walkin_fallback_script(obs, intent, nba_strategy), "fallback"


def render_walkin_result(obs: dict, script: dict, walkin_id: str, key_prefix: str = "walkin") -> None:
    """Display walk-in script result — shows real in-store observations, no fake online signals."""
    intent = script.get("intent_label", "")
    nba    = script.get("nba_strategy", "")
    psych  = script.get("psychology_trigger", "")

    # ── Header ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"### {INTENT_ICONS.get(intent, '')} Walk-in — {intent}")
        st.caption(f"Chiến lược: **{nba}** · Tâm lý học: *{psych}*")
    with col_b:
        st.markdown(
            "<span style='color:#f97316;font-size:13px;font-weight:600'>"
            "👤 Khách Walk-in Mới</span><br>"
            "<span style='color:#64748b;font-size:12px'>Chưa có trong hệ thống</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Two-column: observed vs collected ────────────────────────────────────
    col_obs, col_col = st.columns(2)

    with col_obs:
        st.markdown("**👁️ Quan sát tại cửa hàng**")
        items_obs = [
            ("Giới tính",       obs.get("gender", "")),
            ("Độ tuổi",         obs.get("age", "")),
            ("Phong cách",      obs.get("style", "")),
            ("Đến cùng",        obs.get("companion", "")),
            ("Mức độ hứng thú", obs.get("engagement", "")),
            ("Đang xem",        obs.get("product_type", "Chưa rõ")),
        ]
        for label, value in items_obs:
            is_unknown = not value or value == "Chưa rõ"
            val_color  = "#e2e8f0" if not is_unknown else "#4b5563"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #1e2330'>"
                f"<span style='color:#64748b;font-size:13px'>{label}</span>"
                f"<span style='color:{val_color};font-size:13px'>{value or '—'}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_col:
        st.markdown("**💬 Thông tin đã thu thập**")
        items_col = [
            ("Mục đích",     obs.get("purpose",  "Chưa rõ")),
            ("Dịp đặc biệt", obs.get("occasion", "Chưa hỏi được")),
            ("Ngân sách",    obs.get("budget",   "Chưa rõ")),
        ]
        unknowns = []
        for label, value in items_col:
            is_unknown = value in ["Chưa rõ", "Chưa hỏi được"]
            if is_unknown:
                unknowns.append(label.lower())
            val_color = "#fde68a" if not is_unknown else "#6b7280"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #1e2330'>"
                f"<span style='color:#64748b;font-size:13px'>{label}</span>"
                f"<span style='color:{val_color};font-size:13px'>"
                f"{'⚠️ ' if is_unknown else ''}{value}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if unknowns:
            st.markdown(
                "<div style='background:#162016;border:1px solid #22c55e;border-radius:8px;"
                "padding:8px 12px;margin-top:8px;font-size:12px;color:#86efac'>"
                f"💡 Script đã bao gồm cách hỏi tự nhiên về: {', '.join(unknowns)}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Insight ──────────────────────────────────────────────────────────
    key = str(script.get("key_insight", "")).strip()
    if key:
        st.markdown(
            f'<div class="box-key">💡 <strong>Key Insight:</strong> {key}</div>',
            unsafe_allow_html=True,
        )

    # ── Product recommendations ───────────────────────────────────────────────
    prods = [p for p in script.get("product_recommendations", []) if p]
    if prods:
        st.markdown("**📦 Sản phẩm gợi ý**")
        for prod in prods:
            st.markdown(f"- {prod}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sales Script 5 bước ───────────────────────────────────────────────────
    # Xác định field nào còn thiếu để hiển thị cầu nối sau bước khai_thac
    _missing_fields = []
    if obs.get("purpose", "Chưa rõ") == "Chưa rõ":
        _missing_fields.append("Mục đích mua")
    if obs.get("budget", "Chưa rõ") == "Chưa rõ":
        _missing_fields.append("Ngân sách")

    st.markdown("**🗣️ Sales Script 5 Bước**")
    for key_name, label in SCRIPT_STEPS:
        content = str(script.get(key_name, "")).strip()
        if content:
            st.markdown(
                f'<div class="step-block">'
                f'<div class="step-label">{label}</div>'
                f'{content}'
                f'</div>',
                unsafe_allow_html=True,
            )
        if key_name == "khai_thac" and _missing_fields:
            fields_str = " + ".join(f"<strong>{f}</strong>" for f in _missing_fields)
            st.markdown(
                f'<div style="background:#0c1a10;border-left:3px solid #22c55e;'
                f'border-radius:0 6px 6px 0;padding:7px 12px;margin:4px 0 8px 0;'
                f'font-size:12px;color:#86efac;line-height:1.5">'
                f'→ Sau câu hỏi này, tick {fields_str} bên dưới — '
                f'hệ thống cập nhật gợi ý sản phẩm phù hợp ngay.</div>',
                unsafe_allow_html=True,
            )

    # ── Interactive product selector (chỉ dành cho Khách Mới) ───────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#0d1a0d;border:1.5px solid #22c55e;border-radius:10px;'
        'padding:12px 18px 8px 18px;margin-bottom:12px">'
        '<div style="font-size:13px;font-weight:700;color:#22c55e;'
        'text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px">'
        '🎯 Bước 2 — Tick thông tin khai thác được → Nhận gợi ý sản phẩm ngay</div>'
        '<div style="font-size:12px;color:#4ade80;line-height:1.5">'
        'Sau khi hỏi được <strong>mục đích mua</strong> và <strong>ngân sách</strong>, '
        'tick vào đây — sản phẩm phù hợp nhất cho khách này sẽ hiện ra tức thì.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    ip_c1, ip_c2, ip_c3 = st.columns(3)
    with ip_c1:
        ip_purpose = st.radio(
            "Mục đích mua",
            _IP_PURPOSE_OPTS,
            key=f"{key_prefix}_ip_purpose",
        )
    with ip_c2:
        ip_budget = st.radio(
            "Ngân sách xác nhận",
            _IP_BUDGET_OPTS,
            key=f"{key_prefix}_ip_budget",
        )
    with ip_c3:
        ip_style = st.radio(
            "Phong cách ưa thích",
            _IP_STYLE_OPTS,
            key=f"{key_prefix}_ip_style",
        )

    has_ip_selection = (
        ip_purpose != "Chưa xác định"
        or ip_budget != "Chưa rõ"
        or ip_style != "Chưa rõ"
    )

    if has_ip_selection:
        ip_recs, ip_advisory = _walkin_interactive_product_recs(
            ip_purpose, ip_budget, ip_style, obs
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📦 Sản phẩm phù hợp nhất cho khách này**")
        for prod_name, prod_desc in ip_recs:
            st.markdown(
                f'<div style="background:#0d1a0d;border:1px solid #16a34a;border-radius:8px;'
                f'padding:10px 14px;margin-bottom:8px">'
                f'<div style="color:#4ade80;font-size:13px;font-weight:600">💎 {prod_name}</div>'
                f'<div style="color:#6b7280;font-size:12px;margin-top:3px">{prod_desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="background:#162016;border:1px solid #22c55e;border-radius:8px;'
            f'padding:10px 14px;color:#86efac;font-size:13px;line-height:1.6;margin-top:4px">'
            f'💬 {ip_advisory}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────
    script_txt = "\n".join([
        f"=== SCRIPT CHO TVV — {walkin_id} ===",
        f"Loại khách  : {intent}",
        f"Chiến lược  : {nba}",
        f"Tâm lý học  : {psych}",
        "",
        "QUAN SÁT TẠI CỬA HÀNG:",
        f"  Giới tính : {obs.get('gender', '')}",
        f"  Độ tuổi   : {obs.get('age', '')}",
        f"  Phong cách: {obs.get('style', '')}",
        f"  Đến cùng  : {obs.get('companion', '')}",
        f"  Hứng thú  : {obs.get('engagement', '')}",
        f"  Đang xem  : {obs.get('product_type', '')}",
        "",
        "THÔNG TIN ĐÃ THU THẬP:",
        f"  Mục đích  : {obs.get('purpose', '')}",
        f"  Dịp       : {obs.get('occasion', '')}",
        f"  Ngân sách : {obs.get('budget', '')}",
        "",
        f"Key Insight : {key}",
        "",
        "Sản phẩm gợi ý:",
        *[f"  {i+1}. {p}" for i, p in enumerate(prods)],
        "",
        "SCRIPT 5 BƯỚC:",
        f"1. Opening   : {script.get('opening', '')}",
        f"2. Khai thác : {script.get('khai_thac', '')}",
        f"3. Gợi ý     : {script.get('goi_y', '')}",
        f"4. Chốt đơn  : {script.get('chot', '')}",
        f"5. Upsell    : {script.get('upsell', '')}",
    ])
    st.download_button(
        label="📥 Tải script (.txt)",
        data=script_txt,
        file_name=f"script_{walkin_id}.txt",
        mime="text/plain",
        key=f"dl_{key_prefix}_{walkin_id}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# WALK-IN INTERACTIVE PRODUCT SELECTOR (Chỉ dành cho Khách Mới)
# ══════════════════════════════════════════════════════════════════════════════

_IP_PURPOSE_OPTS = [
    "Chưa xác định",
    "Mua cho bản thân",
    "Mua tặng bạn bè / người thân",
    "Quà cầu hôn / đính hôn",
    "Kỷ niệm tình yêu / hôn nhân",
    "Sinh nhật sắp đến",
]
_IP_BUDGET_OPTS = ["Chưa rõ", "Dưới 5 triệu", "5–15 triệu", "15–30 triệu", "Trên 30 triệu"]
_IP_STYLE_OPTS  = ["Chưa rõ", "Tối giản / Thanh lịch", "Trẻ trung / Năng động", "Sang trọng / Cá tính"]


def _walkin_interactive_product_recs(
    purpose: str, budget: str, style: str, obs: dict,
) -> tuple[list[tuple[str, str]], str]:
    """Return ([(name, desc), ...], advisory_note) based on confirmed in-store info."""
    gender    = obs.get("gender", "Nữ")
    pron      = "anh" if gender == "Nam" else "chị"
    companion = obs.get("companion", "")
    is_couple = any(k in companion for k in ["bạn đời", "người yêu"])
    prod_type = obs.get("product_type", "trang sức")
    if prod_type == "Chưa rõ":
        prod_type = "trang sức"

    bgt_str  = budget if budget != "Chưa rõ" else "linh hoạt"
    is_high  = budget in ["15–30 triệu", "Trên 30 triệu"]
    is_mid   = budget == "5–15 triệu"
    is_minimal = "Tối giản" in style or "Thanh lịch" in style
    is_young   = "Trẻ trung" in style
    is_bold    = "Sang trọng" in style or "Cá tính" in style

    # ── Cầu hôn / Đính hôn ───────────────────────────────────────────────────
    if purpose == "Quà cầu hôn / đính hôn":
        if is_high:
            return (
                [
                    ("Nhẫn đính hôn kim cương Solitaire — PNJ Signature", "Viên tấm GIA certified, vàng 18K — biểu tượng của một dịp chỉ có một lần"),
                    ("Nhẫn đôi vàng 18K đính đá full — bộ sưu tập Romance", "Thiết kế tinh tế, phù hợp đeo cả hai"),
                    ("Dịch vụ khắc tên + hộp quà nhung cao cấp PNJ", "Hoàn thiện khoảnh khắc đặc biệt trọn vẹn"),
                ],
                f"Dịp này chỉ có một lần — {pron} xứng đáng được chọn mẫu thật đặc biệt. "
                f"Bên em có thể khắc tên và ngày cầu hôn lên nhẫn, đặt trước 2 ngày là có ngay.",
            )
        return (
            [
                ("Nhẫn đôi vàng 10K tối giản — PNJ Everyday Love", "Thiết kế tinh tế, nhiều mức giá phù hợp"),
                ("Nhẫn đính hôn đá tấm nhỏ vàng 14K", "Sang trọng vừa phải, giá trị cao hơn ngoại hình"),
                ("Gift box + khắc tên miễn phí khi mua tại cửa hàng", "Thêm ý nghĩa mà không tốn thêm chi phí"),
            ],
            f"Cầu hôn ý nghĩa không nhất thiết cần ngân sách lớn — tầm {bgt_str} "
            f"bên em vẫn có nhiều mẫu đẹp, lại được kèm dịch vụ khắc tên miễn phí.",
        )

    # ── Kỷ niệm tình yêu / hôn nhân ──────────────────────────────────────────
    if purpose == "Kỷ niệm tình yêu / hôn nhân" or (is_couple and purpose == "Chưa xác định"):
        if is_high:
            return (
                [
                    ("Dây chuyền bạch kim đính kim cương — PNJ Infinity", "Tinh tế, bền theo năm tháng"),
                    ("Bộ trang sức vàng 18K matching set — necklace + earring", "Quà trọn vẹn cho dịp kỷ niệm"),
                    ("Dịch vụ khắc ngày kỷ niệm lên trang sức", "Lưu giữ cột mốc theo cách riêng của hai người"),
                ],
                f"Kỷ niệm là cột mốc — quà nên ghi dấu được điều đó. "
                f"Bên em có thể khắc ngày tháng hoặc tên riêng lên sản phẩm, rất ý nghĩa.",
            )
        return (
            [
                ("Nhẫn đôi bạc 925 tối giản — dễ đeo hàng ngày cho cả hai", "Nhẹ nhàng mà ý nghĩa"),
                ("Dây chuyền vàng 10K sợi mỏng", "Thanh lịch, kết hợp được nhiều outfit"),
                ("Gift set bông tai + vòng tay bạc matching", "Trọn bộ quà, không cần phối thêm"),
            ],
            f"Kỷ niệm không cần hoành tráng — chỉ cần đúng ý. "
            f"Tầm {bgt_str} bên em có nhiều lựa chọn vừa đẹp vừa ý nghĩa, {pron} xem thử nhé.",
        )

    # ── Sinh nhật ─────────────────────────────────────────────────────────────
    if purpose == "Sinh nhật sắp đến":
        if is_high:
            return (
                [
                    ("Bộ trang sức vàng 18K full set — bông tai + dây chuyền", "Quà sinh nhật đáng nhớ, đủ để gây ấn tượng"),
                    ("Nhẫn vàng 18K đính đá màu theo tháng sinh", "Cá nhân hoá, độc đáo và ý nghĩa"),
                    ("Dịch vụ đóng gói quà cao cấp + thiệp viết tay PNJ", "Tạo trải nghiệm unboxing đặc biệt"),
                ],
                f"Sinh nhật là dịp người nhận sẽ nhớ mãi — để em giúp {pron} chọn quà "
                f"có thể cá nhân hóa theo sở thích, thêm phần thật đặc biệt.",
            )
        if is_mid:
            return (
                [
                    ("Bông tai vàng 10K đính đá — Birthday Collection PNJ", "Dịu dàng, phù hợp nhiều lứa tuổi"),
                    ("Dây chuyền bạc 925 mặt đá màu pastel", "Trẻ trung, hiện đại, giá hợp lý"),
                    ("Gift set vòng tay + bông tai bạc matching", "Trọn bộ quà gọn nhẹ, không cần phối thêm"),
                ],
                f"Tầm {bgt_str} có nhiều mẫu quà sinh nhật vừa đẹp vừa ý nghĩa — "
                f"bên em lấy ra cho {pron} xem thử vài mẫu nhé.",
            )
        return (
            [
                ("Dây chuyền bạc mặt đá nhỏ — PNJ Silver", "Đơn giản mà tinh tế, dễ tặng"),
                ("Bông tai bạc 925 nhỏ đeo hàng ngày", "Phù hợp mọi phong cách"),
                ("Gift card PNJ kèm bao bì quà đẹp sẵn", "Người nhận tự chọn điều mình thích"),
            ],
            f"Tầm {bgt_str} vẫn tặng được quà đẹp và ý nghĩa — "
            f"bên em có sẵn bao bì quà đẹp miễn phí, trông rất trân trọng.",
        )

    # ── Mua tặng bạn bè / người thân ─────────────────────────────────────────
    if purpose == "Mua tặng bạn bè / người thân":
        if is_high:
            return (
                [
                    ("Bộ trang sức vàng 18K đầy đủ + hộp quà sang trọng sẵn", "Ấn tượng ngay từ cái nhìn đầu tiên"),
                    ("Dây chuyền kim cương nhỏ vàng 18K", "Quà tặng không bao giờ sai, phù hợp mọi lứa tuổi"),
                    ("Dịch vụ khắc tên + gói quà nhung cao cấp miễn phí", "Nâng tầm ý nghĩa cho bất kỳ món quà nào"),
                ],
                f"Quà cao cấp cần cả nội dung lẫn hình thức — bên em có dịch vụ đóng gói "
                f"và khắc tên miễn phí để tạo ấn tượng khó quên cho người nhận.",
            )
        return (
            [
                ("Gift set bông tai + dây chuyền bạc 925 — đóng hộp sẵn", "Trọn bộ quà, không cần phối thêm"),
                ("Vòng tay bạc có charm tùy chỉnh theo ý nghĩa", "Mang câu chuyện riêng, người nhận thích lâu dài"),
                ("Gift card PNJ + bao bì quà đẹp", "Người nhận tự chọn — không bao giờ sai"),
            ],
            f"Tặng quà không cần đắt — chỉ cần đúng ý là đẹp. "
            f"Tầm {bgt_str} để em giúp {pron} chọn món quà phù hợp nhất nhé.",
        )

    # ── Mua cho bản thân ──────────────────────────────────────────────────────
    if purpose == "Mua cho bản thân":
        if is_high:
            if is_minimal:
                prods = [
                    ("Nhẫn vàng 18K đường viền mỏng — PNJ Minimal Gold", "Đeo hàng ngày, không bao giờ lỗi mốt"),
                    ("Dây chuyền vàng 18K sợi mỏng đính đá nhỏ", "Thanh lịch, phù hợp cả công sở lẫn dạo phố"),
                    ("Bộ white gold 18K — bông tai + nhẫn đơn giản", "Đẳng cấp kín đáo, không cần phô trương"),
                ]
            elif is_bold:
                prods = [
                    ("Vòng cổ vàng 18K đính đá lớn — PNJ Iconic Statement", "Điểm nhấn nổi bật cho mọi outfit"),
                    ("Nhẫn cocktail đá quý vàng 18K", "Cá tính và sang trọng"),
                    ("Bông tai drop vàng 18K — thiết kế độc đáo", "Không bị trùng với ai"),
                ]
            else:
                prods = [
                    ("Dây chuyền vàng hồng 18K — nữ tính và hiện đại", "Đang được ưa chuộng nhất hiện tại"),
                    ("Nhẫn vàng 18K stackable — tự mix theo mood", "Linh hoạt, phối được nhiều cách"),
                    ("Bộ trang sức vàng 14K phối hai màu", "Sang trọng và cá tính cùng lúc"),
                ]
            return (
                prods,
                f"Tự thưởng cho bản thân là điều {pron} hoàn toàn xứng đáng — "
                f"với tầm {bgt_str}, bên em có những mẫu đẹp mà chắc chắn {pron} sẽ đeo rất nhiều.",
            )
        if is_mid:
            if is_young:
                prods = [
                    ("Vòng tay bạc 925 charm cá nhân hoá", "Trẻ trung, tự thiết kế theo phong cách"),
                    ("Dây chuyền bạc mặt đá màu trend 2024", "Màu sắc tươi, hợp nhiều outfit"),
                    ("Bông tai bạc dài drop hiện đại", "Điểm nhấn cho look everyday"),
                ]
            elif is_bold:
                prods = [
                    ("Nhẫn statement đá lớn — cá tính và nổi bật", "Không bị trùng với ai"),
                    ("Vòng cổ nhiều tầng layering set", "Tự phối theo mood mỗi ngày"),
                    ("Bông tai cỡ lớn thiết kế bất đối xứng", "Táo bạo nhưng vẫn tinh tế"),
                ]
            else:
                prods = [
                    ("Dây chuyền vàng 10K mặt hình học thời thượng", "Phối được cả casual lẫn formal"),
                    ("Nhẫn vàng 10K đính đá nhỏ", "Đẹp ở mức giá hợp lý"),
                    ("Bông tai vàng 10K kiểu dáng trendy", "Nhỏ nhắn nhưng đủ nổi"),
                ]
            return (
                prods,
                f"Tầm {bgt_str} để tự thưởng rất hợp lý — "
                f"bên em có nhiều mẫu đang được ưa chuộng, {pron} xem thử vài mẫu nhé.",
            )
        return (
            [
                ("Dây chuyền bạc 925 mặt hình học hiện đại — PNJ Silver", "Phong cách, dễ phối đồ"),
                ("Nhẫn bạc đính đá nhỏ — stackable dễ mix", "Xu hướng đang hot, giá hợp lý"),
                ("Bông tai bạc 925 kiểu dáng đa dạng", "Đeo hàng ngày, bền đẹp"),
            ],
            f"Với tầm {bgt_str}, bên em có nhiều lựa chọn bạc 925 rất đẹp — "
            f"chất lượng chuẩn PNJ, không lo phai màu hay dị ứng.",
        )

    # ── Default fallback ──────────────────────────────────────────────────────
    if is_minimal:
        default_prods: list[tuple[str, str]] = [
            ("Dây chuyền vàng 10K sợi mỏng tối giản", "Đơn giản, thanh lịch, đeo mọi nơi"),
            ("Nhẫn bạc mỏng tối giản — stackable set", "Xu hướng minimalist đang rất hot"),
            ("Bông tai vàng nhỏ everyday", "Đeo hàng ngày không bao giờ chán"),
        ]
        default_note = f"Bên em vừa về đợt {prod_type} phong cách tối giản rất đẹp — để em lấy ra cho {pron} xem thử nhé."
    elif is_young:
        default_prods = [
            ("Vòng tay bạc 925 charm cá nhân hoá", "Trẻ trung, tự thiết kế theo ý"),
            ("Dây chuyền bạc mặt đá màu pastel", "Màu nhẹ, hợp nhiều outfit"),
            ("Bông tai bạc dài drop hiện đại", "Điểm nhấn cho look hàng ngày"),
        ]
        default_note = f"Bên em có đợt bạc 925 mẫu mới rất hợp phong cách trẻ trung — {pron} xem thử vài mẫu nhé."
    elif is_bold:
        default_prods = [
            ("Nhẫn statement đá lớn — cá tính nổi bật", "Không bị trùng với ai"),
            ("Vòng cổ nhiều tầng layering set", "Tự phối theo mood mỗi ngày"),
            ("Bông tai drop cỡ lớn thiết kế độc đáo", "Táo bạo nhưng vẫn tinh tế"),
        ]
        default_note = f"Bên em vừa có đợt hàng thiết kế độc đáo — phù hợp phong cách cá tính của {pron} đấy."
    else:
        default_prods = [
            (f"Bộ sưu tập {prod_type} mới nhất tháng này", "Mẫu mới về, chưa đại trà"),
            (f"Bestseller {prod_type} đang được chọn nhiều nhất", "Được nhiều khách tin chọn"),
            (f"Phiên bản {prod_type} nâng cấp đính đá nhỏ", "Tinh tế hơn, giá chênh không đáng kể"),
        ]
        default_note = (
            f"Bên em vừa có đợt hàng mới — để em lấy ra vài mẫu {prod_type} "
            f"phù hợp nhất với {pron} xem thử nhé. Không nhất thiết phải quyết định ngay ạ."
        )
    return default_prods, default_note


def render_tab_walkin(api_key: str = "") -> None:
    """Tab Khách Mới: form tick-chọn → sinh script tức thì cho khách walk-in."""
    st.markdown("## 👁️ Khách Mới — Script Tức Thì")
    st.caption(
        "Dành cho khách walk-in chưa có trong hệ thống. "
        "Điền những gì **quan sát được** và **hỏi được** → bấm **Sinh Script** để nhận gợi ý ngay."
    )

    st.info(
        "💡 Phần **Quan sát được ngay** (bên trái) là quan trọng nhất. "
        "Phần **Hỏi thêm nếu có thể** (bên phải) nếu chưa hỏi được thì cứ để mặc định — "
        "script sẽ tự bao gồm cách hỏi tự nhiên cho những phần đó.",
        icon=None,
    )

    st.divider()

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            '<div style="font-size:13px;font-weight:700;color:#d4af37;'
            'text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px">'
            '👁️ Quan sát được ngay</div>',
            unsafe_allow_html=True,
        )
        gender    = st.radio("Giới tính", _WALKIN_GENDER, horizontal=True, key="wk_gender")
        age       = st.radio("Độ tuổi (ước tính)", _WALKIN_AGE, key="wk_age")
        style_obs = st.radio("Phong cách ăn mặc", _WALKIN_STYLE, key="wk_style")
        companion = st.radio("Đến cùng ai", _WALKIN_COMPANION, key="wk_companion")

    with col2:
        st.markdown(
            '<div style="font-size:13px;font-weight:700;color:#3b82f6;'
            'text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px">'
            '🔍 Hành vi tại cửa hàng</div>',
            unsafe_allow_html=True,
        )
        engagement   = st.radio("Mức độ hứng thú (quan sát)", _WALKIN_ENGAGEMENT, key="wk_engagement")
        product_type = st.radio("Đang xem loại nào", _WALKIN_PRODUCT, key="wk_product")

    with col3:
        st.markdown(
            '<div style="font-size:13px;font-weight:700;color:#a855f7;'
            'text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px">'
            '💬 Hỏi thêm nếu có thể</div>',
            unsafe_allow_html=True,
        )
        budget_obs = st.radio("Ngân sách (hỏi hoặc ước tính)", _WALKIN_BUDGET, key="wk_budget")
        purpose    = st.radio("Mục đích", _WALKIN_PURPOSE, key="wk_purpose")
        occasion   = st.radio("Dịp đặc biệt", _WALKIN_OCCASION, key="wk_occasion")

    st.divider()

    btn_col, reset_col = st.columns([4, 1])
    with btn_col:
        generate = st.button(
            "✨ Sinh Script cho Khách Này",
            type="primary",
            use_container_width=True,
            key="wk_generate",
        )
    with reset_col:
        reset = st.button("🔄 Reset", use_container_width=True, key="wk_reset")

    if reset:
        for k in list(st.session_state.keys()):
            if k.startswith("wk_"):
                del st.session_state[k]
        st.rerun()

    if generate:
        obs = {
            "gender":       gender,
            "age":          age,
            "style":        style_obs,
            "companion":    companion,
            "engagement":   engagement,
            "product_type": product_type,
            "budget":       budget_obs,
            "purpose":      purpose,
            "occasion":     occasion,
        }
        with st.spinner("Đang phân tích và tạo script..."):
            script_data, source = _generate_walkin_script(obs, api_key)
        walkin_id = f"WALKIN-{datetime.now().strftime('%H%M%S')}"
        st.session_state["wk_result"] = {
            "obs":       obs,
            "script":    script_data,
            "source":    source,
            "walkin_id": walkin_id,
            "auto_updated": False,
        }

    # ── Auto-refresh gợi ý khi "Hỏi thêm" thay đổi sau lần sinh script đầu ──
    elif "wk_result" in st.session_state:
        saved_obs = st.session_state["wk_result"]["obs"]
        if (budget_obs != saved_obs.get("budget") or
                purpose != saved_obs.get("purpose") or
                occasion != saved_obs.get("occasion")):
            obs_updated = {
                **saved_obs,
                "budget":  budget_obs,
                "purpose": purpose,
                "occasion": occasion,
            }
            with st.spinner("Đang cập nhật gợi ý theo thông tin mới..."):
                script_data, source = _generate_walkin_script(obs_updated, api_key)
            st.session_state["wk_result"] = {
                "obs":          obs_updated,
                "script":       script_data,
                "source":       source,
                "walkin_id":    st.session_state["wk_result"]["walkin_id"],
                "auto_updated": True,
            }
            st.rerun()

    # ── Hiển thị kết quả (giữ nguyên khi thay đổi form) ─────────────────────
    if "wk_result" in st.session_state:
        saved    = st.session_state["wk_result"]
        obs_r    = saved["obs"]
        script_r = saved["script"]
        source_r = saved["source"]
        wid      = saved["walkin_id"]

        intent      = script_r.get("intent_label", "")
        color       = INTENT_COLORS.get(intent, "#6b7280")
        src_badge   = "🤖 GPT-4o" if source_r == "llm" else "📝 Fallback Template"
        auto_badge  = (
            '<span style="background:#7c3aed;color:#ede9fe;font-size:11px;'
            'padding:2px 8px;border-radius:10px;margin-left:8px">🔄 Đã cập nhật theo Hỏi thêm</span>'
            if saved.get("auto_updated") else ""
        )

        st.markdown(
            f'<div style="background:#1e2330;border:1.5px solid {color};border-radius:10px;'
            f'padding:12px 20px;margin-bottom:4px;display:flex;align-items:center;gap:12px">'
            f'<span style="font-size:16px;font-weight:700;color:{color}">'
            f'{INTENT_ICONS.get(intent, "")} Phân loại: {intent}</span>'
            f'<span style="color:#94a3b8;font-size:13px">'
            f'— Chiến lược: {script_r.get("nba_strategy", "")}</span>'
            f'{auto_badge}'
            f'<span style="margin-left:auto;color:#94a3b8;font-size:12px">'
            f'{src_badge}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        render_walkin_result(obs_r, script_r, wid, key_prefix="walkin")


# ══════════════════════════════════════════════════════════════════════════════
# REGENERATE OUTBOUND MESSAGES
# ══════════════════════════════════════════════════════════════════════════════

def _regenerate_outbound_messages(api_key: str) -> tuple[bool, str]:
    """
    Regenerate outbound messages for all customers using OpenAI API + instore insights.

    Loads:  instore_scripts.json (key_insight, urgency, product_recs…)
            customer_data_poc_enhanced.xlsx (profiles, ml_predictions)
    Runs:   LEP Model → NBA Engine LLM → saves outputs/nba_messages.json
    Returns: (success: bool, message: str)
    """
    try:
        from src.lep_pipeline     import LEPModel
        from src.nba_engine_llm   import NBAEngineLLM
        from src.pipeline_llm     import load_instore_cache, save_output_json

        # Load instore cache
        instore_cache = load_instore_cache(CACHE_FILE)

        # Load data
        sheets      = pd.read_excel(DATA_PATH, sheet_name=None)
        df_profiles = sheets["profiles_enhanced"]
        df_ml       = sheets.get("ml_predictions", pd.DataFrame())

        # Merge gender từ sheet profiles (profiles_enhanced không có cột gender)
        if "profiles" in sheets:
            df_gender = sheets["profiles"][["customer_id", "gender"]].copy()
            df_profiles = df_profiles.merge(df_gender, on="customer_id", how="left")
            df_profiles["gender"] = df_profiles["gender"].fillna("F")

        # Train / load LEP model
        lep = LEPModel(n_estimators=100)
        lep.train(df_profiles, df_ml, verbose=False)
        lep_preds = lep.predict(df_profiles)

        # NBA Engine LLM — use_cache=False to force re-generation
        engine = NBAEngineLLM(
            api_key=api_key.strip() or None,
            use_cache=False,
        )
        nba_result = engine.generate_actions_llm(
            lep_predictions=lep_preds,
            df_profiles=df_profiles,
            instore_cache=instore_cache,
            verbose=False,
        )

        # Save to JSON
        output_path = ROOT / "outputs" / "nba_messages.json"
        output_path.parent.mkdir(exist_ok=True)
        save_output_json(nba_result, instore_cache, output_path)

        # Clear Streamlit cache so next load picks up new file
        load_message_plan.clear()

        llm_n     = (nba_result["message_source"] == "llm").sum()
        allowed_n = (nba_result["rule_status"] == "allowed").sum()
        mode_str  = "GPT-4o" if api_key.strip() else "Fallback Template"
        return True, (
            f"✅ Đã sinh {allowed_n} tin nhắn ({llm_n} bằng {mode_str}) "
            f"cho {len(nba_result)} khách — lưu vào `outputs/nba_messages.json`"
        )
    except Exception as exc:
        return False, f"❌ Lỗi khi sinh tin nhắn: {exc}"


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

    # ── Regenerate outbound messages button ────────────────────────────────
    st.sidebar.markdown("### 📨 Sinh Tin Nhắn Outbound")
    st.sidebar.caption(
        "Dựa trên Key Insight từ phân tích In-Store để sinh tin nhắn "
        "ZNS / Email / Push / In-App / Store cho từng khách."
    )
    regen_btn = st.sidebar.button(
        "🔄 Sinh lại tin nhắn" + (" bằng GPT-4o" if api_key else " (Template)"),
        use_container_width=True,
        help=(
            "Sinh tin nhắn mới dựa trên Key Insight từ instore_scripts.json.\n"
            + ("GPT-4o sẽ tạo nội dung độc đáo cho từng khách." if api_key
               else "Thêm API Key để dùng GPT-4o thay vì template.")
        ),
        key="sidebar_regen_btn",
    )
    if regen_btn:
        with st.sidebar.status("⏳ Đang sinh tin nhắn…", expanded=True) as status_box:
            ok, msg = _regenerate_outbound_messages(api_key)
            if ok:
                status_box.update(label="✅ Hoàn thành!", state="complete", expanded=False)
                st.sidebar.success(msg)
                st.rerun()
            else:
                status_box.update(label="❌ Lỗi", state="error", expanded=True)
                st.sidebar.error(msg)

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
    st.caption("Chọn một khách để xem chi tiết hồ sơ, tin nhắn outbound và sales script đầy đủ.")

    # ── Load message plan (Nhánh 1) ───────────────────────────────────────────
    msg_plan: dict[str, dict] = {}
    if MSG_PLAN_PATH.exists():
        msg_plan = load_message_plan(str(MSG_PLAN_PATH))

    # ── Sync status summary ───────────────────────────────────────────────────
    if msg_plan:
        instore_ids = set(df["customer_id"].astype(str))
        msg_ids     = set(msg_plan.keys())
        missing_msg = instore_ids - msg_ids

        if not missing_msg:
            st.success(
                f"✅ Đồng bộ đầy đủ — cả 2 luồng (In-Store & Outbound) đều có dữ liệu "
                f"cho toàn bộ {len(instore_ids)} khách."
            )
        else:
            ids_preview = ", ".join(sorted(missing_msg)[:6])
            if len(missing_msg) > 6:
                ids_preview += " ..."
            st.warning(
                f"⚠️ **{len(missing_msg)} khách** chưa có dữ liệu tin nhắn outbound: "
                f"`{ids_preview}`  \n"
                f"Chạy `python src/pipeline_llm.py` để sinh tin nhắn dựa trên Key Insight cho các khách còn thiếu."
            )
    else:
        st.info(
            "ℹ️ Chưa có file dữ liệu tin nhắn outbound (`outputs/nba_messages.json`).  \n"
            "Chạy `python src/pipeline_llm.py` để tạo — nội dung sẽ dựa trên Key Insight từ phân tích In-Store."
        )

    st.divider()

    # ── Customer selector via selectbox (searchable) ──────────────────────────
    customer_ids = sorted(df["customer_id"].astype(str).tolist())
    selected_id = st.selectbox(
        "Chọn khách hàng",
        options=customer_ids,
        format_func=lambda cid: (
            lambda r: (
                f"{cid} — {r.get('segment_rfm_tier', '')} "
                f"| {INTENT_ICONS.get(r.get('instore_intent', ''), '')} "
                f"{r.get('instore_intent', '')} "
                f"| {float(r.get('confidence', 0)):.0%}"
                + (" | 📨" if cid in msg_plan else " | ⚠️ thiếu tin nhắn")
            )
        )(df[df["customer_id"] == cid].iloc[0].to_dict()
          if not df[df["customer_id"] == cid].empty else {}),
    )

    rows = df[df["customer_id"] == selected_id]
    if rows.empty:
        st.warning(f"Không tìm thấy khách: {selected_id}")
        return

    row     = rows.iloc[0]
    msg_row = msg_plan.get(selected_id)

    # ── Inline sync badge for selected customer ────────────────────────────────
    if msg_row:
        st.markdown(
            '<span style="background:#1e3a2a;color:#4ade80;border:1px solid #16a34a;'
            'border-radius:6px;padding:3px 12px;font-size:12px;font-weight:600">'
            '✅ Đồng bộ — có đủ dữ liệu In-Store + Tin nhắn Outbound</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="background:#2a1a1a;color:#fca5a5;border:1px solid #ef4444;'
            'border-radius:6px;padding:3px 12px;font-size:12px;font-weight:600">'
            '⚠️ Chưa có dữ liệu tin nhắn outbound cho khách này</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    render_script_card(row, key_prefix="search", msg_data=msg_row)


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
    tab_raw, tab_analysis, tab_customer, tab_walkin, tab_guide = st.tabs([
        "📂 Dữ liệu gốc",
        "🎯 Kết quả phân tích",
        "👤 Tra cứu khách",
        "👁️ Khách Mới",
        "📖 Hướng dẫn phân loại",
    ])

    with tab_raw:
        render_tab_raw(DATA_PATH)

    with tab_analysis:
        render_tab_analysis(df_analysis, filters)

    with tab_customer:
        render_tab_customer(df_analysis)

    with tab_walkin:
        render_tab_walkin(api_key)

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

# source venv/bin/activate
# streamlit run app_instore.py