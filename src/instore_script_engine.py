"""
src/instore_script_engine.py
────────────────────────────────────────────────────────────────────────────
In-Store NBA Engine — Nhánh 2 (In-store)

Nhận output LEP + df_profiles → phân loại Instore Intent Type →
sinh Sales Script 5 bước cá nhân hoá cho Tư Vấn Viên (TVV).

4 loại Instore Intent (theo framework):
  1. High Purchase   — "Sắp mua / High Purchase Signal"
  2. Exploration     — "Đang tìm hiểu / Exploration"
  3. Premium         — "Khách cao cấp / Premium"
  4. Low Intent      — "Chưa rõ nhu cầu / Low Intent"

Sales Script 5 bước:
  Opening → Khai thác nhu cầu → Gợi ý sản phẩm → Chốt đơn → Upsell

Psychology Triggers:
  High Purchase  → Scarcity + Social Proof
  Exploration    → Narrowing + Personalization
  Premium        → Exclusivity + White Glove Service
  Low Intent     → Experience + No Pressure
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import json
import time
import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

# ── OpenAI import (graceful fallback) ─────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[InStore] WARNING: openai chưa được cài. Chạy: pip install openai")
    print("[InStore] Sẽ dùng fallback template mode.")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Enums & Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

class InstoreIntentType(str, Enum):
    """4 loại intent tại cửa hàng."""
    HIGH_PURCHASE = "High Purchase"   # Sắp mua — chốt nhanh
    EXPLORATION   = "Exploration"     # Đang tìm hiểu — định hướng
    PREMIUM       = "Premium"         # Khách cao cấp — VIP service
    LOW_INTENT    = "Low Intent"      # Chưa rõ — tạo trải nghiệm


@dataclass
class InstoreCustomerProfile:
    """
    Hồ sơ đầy đủ của khách cho TVV — bao gồm tất cả thông tin CRM/LEP.
    Dùng để inject vào LLM prompt và hiển thị trực tiếp cho TVV.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    customer_id:        str
    segment_rfm_tier:   str
    budget:             str
    style:              str
    preferred_type:     str          # Nhẫn / Bông tai / ...
    material:           str          # Vàng 18K / Kim cương / ...

    # ── LEP Output ────────────────────────────────────────────────────────────
    lep_intent:         str          # engagement / anniversary / self_reward / gift
    confidence:         float
    priority:           str          # high / medium / low

    # ── Instore Classification ────────────────────────────────────────────────
    instore_intent:     InstoreIntentType
    nba_strategy:       str          # "Chốt đơn nhanh" / ...

    # ── Online Behavioral Signals ─────────────────────────────────────────────
    recency_days:       int
    monetary:           float
    avg_discount:       float
    web_pdp_views:      int          # số lần xem trang sản phẩm
    add_to_cart:        int
    wishlist:           int
    visit_count:        int          # số lần ghé site/app
    time_on_site_avg:   float        # phút TB mỗi phiên
    sig_view_ring:      int
    sig_view_diamond:   int
    sig_search_propose: int
    birthday_in_days:   int

    # ── Engagement History ────────────────────────────────────────────────────
    camp_engagement:    int
    camp_anniversary:   int
    camp_selfreward:    int

    # ── Online Insight Summary (sinh tự động) ────────────────────────────────
    online_insight:     str = ""     # mô tả hành vi online bằng tiếng Việt

    # ── Giới tính: M = Nam (anh), F = Nữ (chị) ─────────────────────────
    gender:             str = "F"

    # ── Action ───────────────────────────────────────────────────────────────
    product_focus:      str = ""
    product_recommendations: list[str] = field(default_factory=list)
    psychology_trigger: str = ""


@dataclass
class InStoreSalesScript:
    """Sales Script 5 bước + metadata đầy đủ."""
    customer_id:        str
    instore_intent:     InstoreIntentType
    nba_strategy:       str
    psychology_trigger: str

    # 5 bước
    opening:            str
    khai_thac:          str
    goi_y:              str
    chot:               str
    upsell:             str

    # metadata
    product_recommendations: list[str]
    key_insight:        str          # 1 câu tóm tắt insight quan trọng nhất
    urgency_signal:     str          # tín hiệu khẩn cấp nếu có (sinh nhật sắp đến...)
    tokens_used:        int = 0
    source:             str = "fallback"
    raw_json:           dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Instore Intent Classifier
# ══════════════════════════════════════════════════════════════════════════════

# Ngưỡng tín hiệu
_HIGH_PURCHASE_SIGNALS = {
    "add_to_cart_min": 1,
    "web_pdp_views_min": 5,
    "visit_count_min": 3,
}
_PREMIUM_TIERS = {"Platinum", "Platinum-H", "Platinum-M", "Platinum-L",
                  "Gold", "Gold-H", "Gold-M", "Gold-L"}
_PREMIUM_MONETARY = 50_000_000   # từ 50 triệu trở lên

# Ngưỡng tín hiệu sig_* tối thiểu để KHÔNG phải Low Intent
_SIG_MEANINGFUL = {
    "sig_view_ring":      1,   # xem nhẫn đính hôn ít nhất 1 lần
    "sig_view_diamond":   1,   # xem kim cương ít nhất 1 lần
    "sig_search_propose": 1,   # tìm kiếm từ khoá cầu hôn ít nhất 1 lần
    "birthday_in_days":  60,   # sinh nhật/kỷ niệm còn <= 60 ngày (tín hiệu dịp đặc biệt)
}

# Low Intent CHỈ khi TẤT CẢ tín hiệu đều ở mức rất thấp hoặc bằng 0
_LOW_INTENT_THRESHOLDS = {
    "add_to_cart_max":   0,   # không có sản phẩm nào trong giỏ
    "web_pdp_views_max": 1,   # xem trang sản phẩm <= 1 lần
    "visit_count_max":   1,   # chỉ ghé web/app tối đa 1 lần
    "confidence_max":   0.50, # model không chắc chắn về intent
}


def classify_instore_intent(
    lep_intent: str,
    confidence: float,
    profile_row: pd.Series,
) -> tuple[InstoreIntentType, str]:
    """
    Phân loại Instore Intent Type + NBA Strategy từ LEP output + profile signals.

    Luồng phân loại (ưu tiên từ trên xuống):
      1. PREMIUM   — khách high-value tier / chi tiêu cao
      2. HIGH_PURCHASE — tín hiệu mua rõ ràng (giỏ hàng / nhiều lần xem)
      3. EXPLORATION   — đang tìm hiểu: có intent LEP rõ, sig_* tích cực,
                         hoặc có hành vi online có ý nghĩa
      4. LOW_INTENT    — fallback CHỈ khi TOÀN BỘ tín hiệu đều gần bằng 0

    Returns: (InstoreIntentType, nba_strategy_str)
    """
    add_to_cart   = int(profile_row.get("add_to_cart", 0))
    web_pdp_views = int(profile_row.get("web_pdp_views", 0))
    visit_count   = int(profile_row.get("tp_web", profile_row.get("tp_app", 0)))
    monetary      = float(profile_row.get("monetary", 0))
    rfm_tier      = str(profile_row.get("segment_rfm_tier", ""))

    # Tín hiệu đặc biệt (sig_*)
    sig_view_ring     = int(profile_row.get("sig_view_ring",
                            profile_row.get("sig_view_engagement_ring", 0)))
    sig_view_diamond  = int(profile_row.get("sig_view_diamond", 0))
    sig_search_propose = int(profile_row.get("sig_search_propose", 0))
    birthday_in_days  = int(profile_row.get("birthday_in_days",
                            profile_row.get("sig_birthday_in_days", 9999)))

    # ── Premium: khách high-value bất kể intent ──────────────────────────────
    is_premium_tier     = any(t in rfm_tier for t in _PREMIUM_TIERS)
    is_premium_monetary = monetary >= _PREMIUM_MONETARY
    if (is_premium_tier or is_premium_monetary) and (
        lep_intent in ("self_reward", "gift") or confidence >= 0.70
    ):
        return InstoreIntentType.PREMIUM, "Tư vấn như VIP"

    # ── High Purchase: tín hiệu mua mạnh ─────────────────────────────────────
    has_cart    = add_to_cart >= _HIGH_PURCHASE_SIGNALS["add_to_cart_min"]
    high_views  = web_pdp_views >= _HIGH_PURCHASE_SIGNALS["web_pdp_views_min"]
    many_visits = visit_count >= _HIGH_PURCHASE_SIGNALS["visit_count_min"]
    high_conf   = confidence >= 0.75

    if (has_cart or high_views) and (many_visits or high_conf):
        return InstoreIntentType.HIGH_PURCHASE, "Chốt đơn nhanh"

    # ── Exploration: đang tìm hiểu ───────────────────────────────────────────
    # Điều kiện 1: LEP intent rõ (engagement / anniversary) + model đủ tự tin
    if lep_intent in ("engagement", "anniversary") and confidence >= 0.40:
        return InstoreIntentType.EXPLORATION, "Định hướng lựa chọn"

    # Điều kiện 2: Có tín hiệu sig_* đặc thù (xem nhẫn, xem kim cương, tìm kiếm cầu hôn)
    has_sig_ring    = sig_view_ring    >= _SIG_MEANINGFUL["sig_view_ring"]
    has_sig_diamond = sig_view_diamond >= _SIG_MEANINGFUL["sig_view_diamond"]
    has_sig_propose = sig_search_propose >= _SIG_MEANINGFUL["sig_search_propose"]
    if has_sig_ring or has_sig_diamond or has_sig_propose:
        return InstoreIntentType.EXPLORATION, "Định hướng lựa chọn"

    # Điều kiện 3: Có sản phẩm trong giỏ (bất kể số lần ghé web)
    if add_to_cart >= 1:
        return InstoreIntentType.EXPLORATION, "Định hướng lựa chọn"

    # Điều kiện 4: Dịp đặc biệt sắp tới (sinh nhật / kỷ niệm)
    if birthday_in_days <= _SIG_MEANINGFUL["birthday_in_days"]:
        return InstoreIntentType.EXPLORATION, "Định hướng lựa chọn"

    # Điều kiện 5: Hành vi browsing có ý nghĩa (xem nhiều trang SP hoặc nhiều lượt ghé)
    if web_pdp_views >= 2 or visit_count >= 2:
        return InstoreIntentType.EXPLORATION, "Định hướng lựa chọn"

    # Điều kiện 6: LEP intent self_reward / gift nhưng phải cực kỳ tự tin (vì thiếu tín hiệu khác)
    if lep_intent in ("self_reward", "gift") and confidence >= 0.70:
        return InstoreIntentType.EXPLORATION, "Định hướng lựa chọn"

    # ── Low Intent: THỰC SỰ rất ít tín hiệu ─────────────────────────────────
    # Chỉ đến đây nếu TẤT CẢ đều không thoả mãn:
    #   - add_to_cart = 0
    #   - web_pdp_views <= 1
    #   - visit_count <= 1
    #   - không có sig_* nào tích cực
    #   - không có dịp đặc biệt sắp tới
    #   - confidence < 0.40 hoặc intent không rõ
    return InstoreIntentType.LOW_INTENT, "Tạo trải nghiệm"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Online Insight Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_online_insight(profile: InstoreCustomerProfile) -> str:
    """Sinh đoạn mô tả hành vi online bằng tiếng Việt dạng ngắn gọn cho LLM."""
    parts = []

    if profile.add_to_cart > 0:
        parts.append(f"đã thêm {profile.add_to_cart} sản phẩm vào giỏ hàng chưa mua")
    if profile.web_pdp_views > 0:
        parts.append(f"xem trang sản phẩm {profile.web_pdp_views} lần")
    if profile.sig_view_ring > 0:
        parts.append(f"xem nhẫn đính hôn {profile.sig_view_ring} lần")
    if profile.sig_search_propose > 0:
        parts.append("tìm kiếm từ khoá 'cầu hôn'")
    if profile.sig_view_diamond > 0:
        parts.append(f"xem sản phẩm kim cương {profile.sig_view_diamond} lần")
    if profile.wishlist > 0:
        parts.append(f"lưu {profile.wishlist} sản phẩm vào wishlist")
    if profile.visit_count > 0:
        parts.append(f"quay lại web {profile.visit_count} lần")
    if profile.recency_days <= 7:
        parts.append(f"vừa mới ghé online dạo gần đây ({profile.recency_days} ngày trước)")
    elif profile.recency_days <= 30:
        parts.append(f"đã ghé online trong tháng qua ({profile.recency_days} ngày trước)")
    if profile.camp_engagement > 0:
        parts.append(f"đã mở {profile.camp_engagement} chiến dịch engagement")
    if profile.camp_anniversary > 0:
        parts.append(f"đã mở {profile.camp_anniversary} chiến dịch kỷ niệm")

    if not parts:
        return "Chưa có nhiều dữ liệu hành vi online."

    return "Khách " + ", ".join(parts) + "."


def build_urgency_signal(profile: InstoreCustomerProfile) -> str:
    """Phát hiện tín hiệu khẩn cấp để TVV có thể nhấn mạnh."""
    signals = []
    if 0 < profile.birthday_in_days <= 14:
        signals.append(f"🎂 Sinh nhật còn {profile.birthday_in_days} ngày!")
    elif 0 < profile.birthday_in_days <= 30:
        signals.append(f"Sinh nhật trước {profile.birthday_in_days} ngày — cơ hội tặng quà")
    if profile.add_to_cart > 0:
        signals.append("🛒 Đã chọn online nhưng chưa mua — cơ hội chốt ngay")
    if profile.sig_search_propose > 0:
        signals.append("💍 Dấu hiệu cầu hôn mạnh")
    return " | ".join(signals) if signals else ""


# ══════════════════════════════════════════════════════════════════════════════
# 4. Prompts (LLM)
# ══════════════════════════════════════════════════════════════════════════════

INSTORE_SYSTEM_PROMPT = """Bạn là chuyên gia đào tạo tư vấn bán hàng trang sức cao cấp tại Việt Nam.
Nhiệm vụ: Sinh Sales Script 5 bước CÁ NHÂN HÓA cho tư vấn viên (TVV) tại cửa hàng — 
dựa trên thông tin khách hàng cụ thể mà hệ thống CRM cung cấp.

Quy tắc bắt buộc:
1. Script phải TỰ NHIÊN như TVV thực sự nói — không cứng nhắc, không sáo rỗng
2. Xưng "em", gọi khách đúng theo giới tính được cung cấp trong thông tin: "anh" nếu Nam, "chị" nếu Nữ. TUYỆT ĐỐI không dùng "anh/chị" gộp chung hay "bạn"
3. Tích hợp thông tin cá nhân của khách vào từng bước — NHƯNG phải khéo léo, tự nhiên:
   - KHÔNG được nói thẳng: "Em thấy anh/chị đã xem..." hay "Hệ thống cho em biết chị đã..."
   - THAY VÀO ĐÓ: dùng câu hỏi gợi mở ("Anh đang tìm cho dịp đặc biệt nào không?"),
     hoặc dùng ngữ cảnh sản phẩm ("Bộ sưu tập nhẫn đính hôn bên em vừa về mẫu rất đẹp..."),
     hoặc dùng sở thích đã biết ("Biết chị thích phong cách tối giản nên em chọn riêng 3 mẫu này")
   - Khách KHÔNG được có cảm giác bị theo dõi hay bị đọc lịch sử duyệt web
4. Mỗi bước tối đa 2-3 câu — ngắn gọn, actionable
4. Áp dụng đúng tâm lý học theo loại intent: Scarcity/Social Proof (High Purchase),
   Narrowing (Exploration), Exclusivity (Premium), No Pressure (Low Intent)
5. KHÔNG dùng: "ưu đãi không thể bỏ lỡ", "cơ hội vàng", "đừng bỏ lỡ"
6. Trả về ĐÚNG format JSON — không thêm markdown hay giải thích ngoài JSON
7. QUAN TRỌNG: Dùng đúng "anh" hoặc "chị" nhất quán xuyên suốt 5 bước — không trộn lẫn
8. Trường key_insight là ghi chú NỘI BỘ cho TVV đọc — PHẢI dùng "Khách" (không dùng "anh"/"chị"), viết như nhận xét khách quan về khách hàng"""

INSTORE_INTENT_CONTEXT = {
    InstoreIntentType.HIGH_PURCHASE: """Khách có tín hiệu mua rất cao (đã xem nhiều lần / có giỏ hàng online).
NBA Strategy: CHỐT ĐƠN NHANH.
Tâm lý học: Dùng Social Proof (nhiều người chọn mẫu này) + Scarcity (hàng có hạn/mẫu giới hạn).""",

    InstoreIntentType.EXPLORATION: """Khách đang trong giai đoạn tìm hiểu, so sánh nhiều lựa chọn.
NBA Strategy: ĐỊNH HƯỚNG LỰA CHỌN — thu hẹp từ nhiều → 3 lựa chọn tốt nhất.
Tâm lý học: Personalization (TVV hiểu khách hơn khách tự hiểu) + Narrowing (giúp quyết định).""",

    InstoreIntentType.PREMIUM: """Khách cao cấp (VIP / Platinum / chi tiêu cao).
NBA Strategy: TƯ VẤN NHƯ VIP — trải nghiệm độc quyền, không vội.
Tâm lý học: Exclusivity (sản phẩm/dịch vụ riêng cho khách VIP) + White Glove (phục vụ đặc biệt).""",

    InstoreIntentType.LOW_INTENT: """Khách đang khám phá, chưa rõ nhu cầu.
NBA Strategy: TẠO TRẢI NGHIỆM — không bán mạnh, tạo thiện cảm để khách quay lại.
Tâm lý học: Experience (cho thử/cảm nhận) + No Pressure (không áp lực).""",
}

INSTORE_JSON_FORMAT = """{
  "opening": "...",
  "khai_thac": "...",
  "goi_y": "...",
  "chot": "...",
  "upsell": "...",
  "key_insight": "...",
  "product_recommendations": ["sản phẩm 1", "sản phẩm 2", "sản phẩm 3"]
}

Giải thích:
- opening: TVV chào hỏi tự nhiên khi khách bước vào — KHÔNG đề cập trực tiếp lịch sử online;
  thay vào đó gợi mở bằng sản phẩm nổi bật hoặc câu hỏi phong cách để bắt đầu cuộc trò chuyện
- khai_thac: câu hỏi khai thác nhu cầu sâu hơn
- goi_y: gợi ý sản phẩm cụ thể (dùng tên sản phẩm từ thông tin đã cho)
- chot: câu chốt đơn (tích hợp tâm lý học phù hợp)
- upsell: câu nâng hạng sản phẩm / bán thêm
- key_insight: 1 câu tóm tắt điều TVV cần nhớ nhất về khách này — dùng "Khách" (KHÔNG dùng "anh"/"chị"), đây là ghi chú nội bộ cho TVV, không phải lời nói trực tiếp với khách
- product_recommendations: 2-3 sản phẩm cụ thể nên giới thiệu"""


def _build_instore_user_prompt(profile: InstoreCustomerProfile) -> str:
    """Xây dựng prompt đầy đủ cho LLM từ hồ sơ khách."""
    intent_ctx   = INSTORE_INTENT_CONTEXT.get(profile.instore_intent, "")
    gender_label = "Nam" if str(profile.gender).upper() == "M" else "Nữ"
    pn           = "anh" if str(profile.gender).upper() == "M" else "chị"

    prompt = f"""LOẠI KHÁCH HÀNG: {profile.instore_intent.value}
NBA STRATEGY: {profile.nba_strategy}

{intent_ctx}

─── THÔNG TIN CRM CỦA KHÁCH ─────────────────────────────
Giới tính        : {gender_label} — PHẢI gọi khách là "{pn}" xuyên suốt toàn bộ script (không dùng "anh/chị" gộp, không dùng "bạn")
Phân khúc        : {profile.segment_rfm_tier}
Ngân sách        : {profile.budget}
Phong cách ưa thích: {profile.style}
Loại trang sức   : {profile.preferred_type} ({profile.material})
Tổng chi tiêu    : {profile.monetary:,.0f}đ
Mua gần nhất     : {profile.recency_days} ngày trước
Hay dùng discount: {"Có" if profile.avg_discount > 0.05 else "Không"} ({profile.avg_discount*100:.0f}%)
Sinh nhật còn    : {profile.birthday_in_days} ngày

─── HÀNH VI ONLINE (trước khi vào tiệm) ─────────────────
{profile.online_insight}

─── DỰ ĐOÁN LEP ─────────────────────────────────────────
Intent dự đoán   : {profile.lep_intent} (confidence: {profile.confidence:.0%})
Priority         : {profile.priority}

─── SẢN PHẨM TẬP TRUNG ──────────────────────────────────
{profile.product_focus}

─── TÍN HIỆU KHẨN CẤP ──────────────────────────────────
{profile.urgency_signal if profile.urgency_signal else "Không có"}

─── YÊU CẦU ─────────────────────────────────────────────
Sinh Sales Script 5 bước cho TVV dùng ngay khi khách bước vào.
Psychology trigger: {profile.psychology_trigger}

{INSTORE_JSON_FORMAT}"""
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# 5. Fallback Templates (không cần LLM)
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_SCRIPTS: dict[InstoreIntentType, dict] = {
    InstoreIntentType.HIGH_PURCHASE: {
        "opening": "Chào mừng {pn} quay lại! Bên em vừa cập nhật thêm mẫu {product_focus} mới — {pn} muốn em lấy ra xem thử không?",
        "khai_thac": "{Pn} đang tìm cho dịp đặc biệt nào vậy? Để em tư vấn thêm cho phù hợp nhé.",
        "goi_y": "Đây là mẫu {product_focus} — form rất đẹp và đeo hàng ngày cũng hợp lý. Tuần này có 3 khách đã lấy mẫu này rồi.",
        "chot": "{Pn} thử lên tay xem — cảm giác thực tế khác hẳn ảnh online. Hàng chỉ còn số lượng có hạn thôi ạ.",
        "upsell": "Nếu {pn} muốn nổi bật hơn chút, bên em có phiên bản kim cương nhỏ — nhìn sang hơn mà giá chênh không nhiều.",
        "key_insight": "Khách đã có ý định mua — ưu tiên cho thử ngay, hạn chế tư vấn dài.",
        "product_recommendations": ["{product_focus}", "Phiên bản nâng cấp có đính kim cương", "Phụ kiện đi kèm"],
    },
    InstoreIntentType.EXPLORATION: {
        "opening": "{Pn} đang tham khảo nhiều mẫu đúng không ạ? Để em giúp nhanh hơn — {pn} đang tìm cho mình hay mua tặng?",
        "khai_thac": "{Pn} thích phong cách tinh tế hay nổi bật hơn? Và ngân sách {pn} đang hướng tới khoảng bao nhiêu?",
        "goi_y": "Dựa vào phong cách {pn} vừa chia sẻ, em chọn ra 3 mẫu phù hợp nhất — {pn} xem qua nhé.",
        "chot": "Trong 3 mẫu này, {pn} thấy mẫu nào hợp nhất để thử lên tay?",
        "upsell": "Nếu {pn} muốn nổi bật hơn một chút, bên em có thể xem thêm phiên bản cao cấp hơn một chút.",
        "key_insight": "Khách đang so sánh — cần thu hẹp lựa chọn, không nên đưa ra quá nhiều mẫu cùng lúc.",
        "product_recommendations": ["{product_focus}", "Mẫu bán chạy nhất tháng", "Mẫu bestseller phong cách nhẹ nhàng"],
    },
    InstoreIntentType.PREMIUM: {
        "opening": "Chào mừng {pn}! Em đã chuẩn bị một số mẫu phù hợp với phong cách của {pn} rồi — mời {pn} vào khu VIP để xem thoải mái hơn nhé.",
        "khai_thac": "{Pn} đang hướng tới phong cách nào cho bộ sưu tập lần này? Hay để em giới thiệu những mẫu mới nhất vừa về?",
        "goi_y": "Đây là bộ {product_focus} — được làm thủ công hoàn toàn, bên em chỉ có số lượng rất hạn chế. {Pn} là khách VIP nên em ưu tiên cho xem trước.",
        "chot": "Bên em có dịch vụ khắc tên miễn phí và hộp quà cao cấp dành riêng cho khách VIP — {pn} muốn em chuẩn bị không?",
        "upsell": "Nếu {pn} muốn hoàn thiện bộ trang sức, bên em có phiên bản matching set — đeo cùng trông rất sang.",
        "key_insight": "Khách Premium — ưu tiên trải nghiệm, đừng vội chốt, tạo không gian thoải mái.",
        "product_recommendations": ["{product_focus}", "Bộ trang sức cao cấp giới hạn", "Dịch vụ khắc tên & hộp quà VIP"],
    },
    InstoreIntentType.LOW_INTENT: {
        "opening": "{Pn} cứ thoải mái xem, có gì em hỗ trợ thêm nhé! Bên em vừa về một số mẫu mới — {pn} thích phong cách nào?",
        "khai_thac": "{Pn} đang xem cho mình hay đang tìm cảm hứng ạ? Em có thể giới thiệu một vài hướng để tham khảo.",
        "goi_y": "Để em đưa ra vài mẫu đang được yêu thích nhất gần đây — {pn} thử lên tay xem cảm giác thế nào nhé.",
        "chot": "{Pn} thấy mẫu nào ưng nhất? Không cần quyết định ngay, em có thể để {pn} giữ lại xem thêm.",
        "upsell": "Nếu hôm nay chưa quyết định, em có thể lưu thông tin mẫu {pn} thích để lần sau ghé em chuẩn bị sẵn.",
        "key_insight": "Khách chưa rõ nhu cầu — tạo thiện cảm, không gây áp lực, mục tiêu là khách quay lại.",
        "product_recommendations": ["New Arrivals tháng này", "{product_focus}", "Bestseller dễ đeo hàng ngày"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 6. LLM Cache
# ══════════════════════════════════════════════════════════════════════════════

class InstoreCache:
    """File-based cache cho LLM instore script responses."""

    def __init__(self, cache_dir: str = "outputs/.instore_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, prompt: str) -> Optional[dict]:
        path = self.cache_dir / f"{self._key(prompt)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def set(self, prompt: str, response: dict) -> None:
        path = self.cache_dir / f"{self._key(prompt)}.json"
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Main Engine
# ══════════════════════════════════════════════════════════════════════════════

# Product focus mapping theo LEP intent × priority
INSTORE_PRODUCT_FOCUS: dict[str, dict[str, str]] = {
    "engagement": {
        "high":   "Bộ sưu tập Nhẫn Đính Hôn 2025 — kim cương tự nhiên",
        "medium": "Nhẫn Đôi / Couple Ring phong cách tối giản",
        "low":    "Bộ sưu tập Nhẫn mới nhất",
    },
    "anniversary": {
        "high":   "Bộ Quà Kỷ Niệm cao cấp (khắc tên, hộp quà đặc biệt)",
        "medium": "Bộ trang sức kỷ niệm — dây chuyền + bông tai",
        "low":    "Gợi ý quà tặng kỷ niệm",
    },
    "self_reward": {
        "high":   "Bộ sưu tập Premium Self-Reward — thiết kế độc đáo",
        "medium": "Trending Collection — đang hot nhất tháng",
        "low":    "New Arrivals — mẫu mới về tuần này",
    },
    "gift": {
        "high":   "Gift Finder — hộp quà tặng cao cấp, tư vấn 1-1",
        "medium": "Gift Set phổ biến — dễ chọn, ý nghĩa",
        "low":    "Gift Guide — gợi ý quà tặng theo ngân sách",
    },
}

PSYCHOLOGY_TRIGGER_MAP: dict[InstoreIntentType, str] = {
    InstoreIntentType.HIGH_PURCHASE: "Social Proof + Scarcity",
    InstoreIntentType.EXPLORATION:   "Narrowing + Personalization",
    InstoreIntentType.PREMIUM:       "Exclusivity + White Glove Service",
    InstoreIntentType.LOW_INTENT:    "Experience Creation + No Pressure",
}


class InstoreScriptEngine:
    """
    Engine sinh Sales Script In-store 5 bước cá nhân hoá.

    Flow:
      1. Merge LEP predictions + df_profiles
      2. Build InstoreCustomerProfile cho từng khách
      3. Classify Instore Intent Type
      4. Gọi LLM (hoặc fallback template) → InStoreSalesScript
      5. Trả về DataFrame đầy đủ

    Usage:
        engine = InstoreScriptEngine(api_key="sk-...", use_cache=True)
        result = engine.generate_scripts(lep_preds, df_profiles, verbose=True)
    """

    MODEL = "gpt-4o"

    def __init__(self,
                 api_key: Optional[str] = None,
                 use_cache: bool = True,
                 rate_limit_delay: float = 0.3):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.use_cache = use_cache
        self.rate_limit_delay = rate_limit_delay
        self.cache = InstoreCache() if use_cache else None

        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.mode = "llm"
            print(f"[InStore] Mode: OpenAI API ({self.MODEL})")
        else:
            self.client = None
            self.mode = "fallback"
            reason = "openai chưa cài" if not OPENAI_AVAILABLE else "không có API key"
            print(f"[InStore] Mode: Fallback template ({reason})")

        self.stats = {
            "llm_calls": 0,
            "cache_hits": 0,
            "fallback_used": 0,
            "total_tokens": 0,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_profile(self, row: pd.Series,
                       lep_intent: str,
                       confidence: float,
                       priority: str,
                       instore_intent: InstoreIntentType,
                       nba_strategy: str) -> InstoreCustomerProfile:
        """Tạo InstoreCustomerProfile từ row DataFrame."""
        product_focus = INSTORE_PRODUCT_FOCUS.get(lep_intent, {}).get(priority, "Trang sức cao cấp")
        psych_trigger = PSYCHOLOGY_TRIGGER_MAP[instore_intent]

        # Budget mapping từ segment
        rfm = str(row.get("segment_rfm_tier", ""))
        if "Platinum" in rfm:
            budget = "30 triệu trở lên"
        elif "Gold" in rfm:
            budget = "15–30 triệu"
        else:
            budget = str(row.get("budget", "5–15 triệu"))

        profile = InstoreCustomerProfile(
            customer_id        = str(row.get("customer_id", row.get("c", "unknown"))),
            gender             = str(row.get("gender", "F")),
            segment_rfm_tier   = rfm or "N/A",
            budget             = budget,
            style              = str(row.get("style", "Trẻ trung")),
            preferred_type     = str(row.get("preferred_type", "Nhẫn")),
            material           = str(row.get("material", "Vàng 18K")),
            lep_intent         = lep_intent,
            confidence         = confidence,
            priority           = priority,
            instore_intent     = instore_intent,
            nba_strategy       = nba_strategy,
            recency_days       = int(row.get("recency_days", 60)),
            monetary           = float(row.get("monetary", 0)),
            avg_discount       = float(row.get("avg_discount", 0)),
            web_pdp_views      = int(row.get("web_pdp_views", 0)),
            add_to_cart        = int(row.get("add_to_cart", 0)),
            wishlist           = int(row.get("wishlist", row.get("wishlist_count", 0))),
            visit_count        = int(row.get("tp_web", row.get("tp_app", 0))),
            time_on_site_avg   = float(row.get("time_on_site_avg", 0)),
            sig_view_ring      = int(row.get("sig_view_engagement_ring", 0)),
            sig_view_diamond   = int(row.get("sig_view_diamond", 0)),
            sig_search_propose = int(row.get("sig_search_propose", 0)),
            birthday_in_days   = int(row.get("sig_birthday_in_days", 365)),
            camp_engagement    = int(row.get("camp_engagement", 0)),
            camp_anniversary   = int(row.get("camp_anniversary", 0)),
            camp_selfreward    = int(row.get("camp_selfreward", 0)),
            product_focus      = product_focus,
            psychology_trigger = psych_trigger,
        )

        profile.online_insight    = build_online_insight(profile)
        profile.urgency_signal    = build_urgency_signal(profile)
        return profile

    def _call_llm(self, profile: InstoreCustomerProfile) -> tuple[dict, int]:
        """Gọi OpenAI API để sinh script."""
        user_prompt = _build_instore_user_prompt(profile)

        # Check cache
        if self.use_cache and self.cache:
            cached = self.cache.get(user_prompt)
            if cached:
                self.stats["cache_hits"] += 1
                return cached, 0

        time.sleep(self.rate_limit_delay)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            max_tokens=700,
            messages=[
                {"role": "system", "content": INSTORE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw_text = response.choices[0].message.content.strip()
        tokens   = response.usage.prompt_tokens + response.usage.completion_tokens

        # Strip markdown fences
        clean = raw_text
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        clean = clean.strip()

        parsed = json.loads(clean)

        if self.use_cache and self.cache:
            self.cache.set(user_prompt, parsed)

        self.stats["llm_calls"] += 1
        self.stats["total_tokens"] += tokens
        return parsed, tokens

    def _build_fallback_script(self, profile: InstoreCustomerProfile) -> tuple[dict, int]:
        """Template fallback khi không có LLM."""
        template = FALLBACK_SCRIPTS.get(profile.instore_intent,
                                        FALLBACK_SCRIPTS[InstoreIntentType.LOW_INTENT])

        pn = "anh" if str(profile.gender).upper() == "M" else "chị"
        Pn = pn.capitalize()

        # Inject product_focus và pronoun vào template
        result = {}
        for key, val in template.items():
            if isinstance(val, str):
                result[key] = (val
                               .replace("{product_focus}", profile.product_focus)
                               .replace("{pn}", pn)
                               .replace("{Pn}", Pn))
            elif isinstance(val, list):
                result[key] = [v.replace("{product_focus}", profile.product_focus)
                                .replace("{pn}", pn)
                                .replace("{Pn}", Pn) for v in val]
            else:
                result[key] = val

        self.stats["fallback_used"] += 1
        return result, 0

    def _generate_one(self, profile: InstoreCustomerProfile) -> InStoreSalesScript:
        """Sinh script cho 1 khách."""
        tokens = 0
        source = "fallback"

        try:
            if self.mode == "llm":
                raw_json, tokens = self._call_llm(profile)
                source = "llm"
            else:
                raw_json, tokens = self._build_fallback_script(profile)
                source = "fallback"
        except (json.JSONDecodeError, Exception) as e:
            print(f"[InStore] Error cho {profile.customer_id}: {e}. Dùng fallback.")
            raw_json, tokens = self._build_fallback_script(profile)
            source = "fallback_error"

        self.stats["total_tokens"] += tokens

        return InStoreSalesScript(
            customer_id          = profile.customer_id,
            instore_intent       = profile.instore_intent,
            nba_strategy         = profile.nba_strategy,
            psychology_trigger   = profile.psychology_trigger,
            opening              = raw_json.get("opening", ""),
            khai_thac            = raw_json.get("khai_thac", ""),
            goi_y                = raw_json.get("goi_y",    ""),
            chot                 = raw_json.get("chot",     ""),
            upsell               = raw_json.get("upsell",   ""),
            key_insight          = raw_json.get("key_insight", ""),
            product_recommendations = raw_json.get("product_recommendations",
                                                     profile.product_recommendations or [profile.product_focus]),
            urgency_signal       = profile.urgency_signal,
            tokens_used          = tokens,
            source               = source,
            raw_json             = raw_json,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_scripts(self,
                         lep_predictions: pd.DataFrame,
                         df_profiles: pd.DataFrame,
                         verbose: bool = True) -> pd.DataFrame:
        """
        Main method: sinh toàn bộ instore scripts.

        Args:
            lep_predictions: output LEPModel.predict()
                Cần: customer_id, predicted_intent, confidence, priority
            df_profiles: profiles_enhanced DataFrame
            verbose: in progress

        Returns:
            DataFrame với đầy đủ thông tin TVV cần (profile + intent + script 5 bước)
        """
        # Merge LEP predictions + profiles
        merged = lep_predictions.merge(
            df_profiles.rename(columns={"c": "customer_id"}),
            on="customer_id", how="left"
        )

        if verbose:
            print(f"\n[InStore] Processing {len(merged)} customers...")
            print(f"[InStore] Script mode: {self.mode.upper()}\n")

        profiles: list[InstoreCustomerProfile] = []
        scripts_map: dict[str, InStoreSalesScript] = {}

        # ── Pass 1: Build profiles & classify intent ──────────────────────────
        for _, row in merged.iterrows():
            intent     = row["predicted_intent"]
            confidence = float(row["confidence"])
            priority   = str(row.get("priority", "medium"))

            instore_intent, nba_strategy = classify_instore_intent(
                lep_intent=intent,
                confidence=confidence,
                profile_row=row,
            )

            profile = self._build_profile(
                row=row,
                lep_intent=intent,
                confidence=confidence,
                priority=priority,
                instore_intent=instore_intent,
                nba_strategy=nba_strategy,
            )
            profiles.append(profile)

        # ── Pass 2: Generate scripts ──────────────────────────────────────────
        total = len(profiles)
        for i, profile in enumerate(profiles, 1):
            if verbose:
                intent_label = profile.instore_intent.value
                print(f"  [{i:3d}/{total}] {profile.customer_id:8s} | "
                      f"{intent_label:15s} | {profile.nba_strategy}", end=" ")

            script = self._generate_one(profile)
            scripts_map[profile.customer_id] = script

            if verbose:
                icon = "🤖" if script.source == "llm" else "📄"
                print(f"{icon} | {script.key_insight[:60]}")

        # ── Pass 3: Assemble output DataFrame ────────────────────────────────
        rows = []
        for profile in profiles:
            script = scripts_map.get(profile.customer_id)
            if script is None:
                continue

            rows.append({
                # ── Identity ──────────────────────────────────────
                "customer_id":          profile.customer_id,
                "segment_rfm_tier":     profile.segment_rfm_tier,
                "budget":               profile.budget,
                "style":                profile.style,
                "preferred_type":       profile.preferred_type,
                "material":             profile.material,

                # ── LEP Output ────────────────────────────────────
                "lep_intent":           profile.lep_intent,
                "confidence":           round(profile.confidence, 3),
                "priority":             profile.priority,

                # ── Instore Classification ────────────────────────
                "instore_intent":       script.instore_intent.value,
                "nba_strategy":         script.nba_strategy,
                "psychology_trigger":   script.psychology_trigger,

                # ── CRM Signals ───────────────────────────────────
                "recency_days":         profile.recency_days,
                "monetary":             profile.monetary,
                "avg_discount_pct":     round(profile.avg_discount * 100, 1),
                "web_pdp_views":        profile.web_pdp_views,
                "add_to_cart":          profile.add_to_cart,
                "wishlist":             profile.wishlist,
                "visit_count":          profile.visit_count,
                "sig_view_ring":        profile.sig_view_ring,
                "sig_view_diamond":     profile.sig_view_diamond,
                "sig_search_propose":   profile.sig_search_propose,
                "birthday_in_days":     profile.birthday_in_days,

                # ── Insight ──────────────────────────────────────
                "online_insight":       profile.online_insight,
                "urgency_signal":       profile.urgency_signal,
                "key_insight":          script.key_insight,

                # ── Product ───────────────────────────────────────
                "product_focus":        profile.product_focus,
                "product_rec_1":        (script.product_recommendations[0]
                                         if len(script.product_recommendations) > 0 else ""),
                "product_rec_2":        (script.product_recommendations[1]
                                         if len(script.product_recommendations) > 1 else ""),
                "product_rec_3":        (script.product_recommendations[2]
                                         if len(script.product_recommendations) > 2 else ""),

                # ── Sales Script 5 bước ───────────────────────────
                "script_opening":       script.opening,
                "script_khai_thac":     script.khai_thac,
                "script_goi_y":         script.goi_y,
                "script_chot":          script.chot,
                "script_upsell":        script.upsell,

                # ── Metadata ──────────────────────────────────────
                "tokens_used":          script.tokens_used,
                "script_source":        script.source,
            })

        if verbose:
            self.print_stats()

        return pd.DataFrame(rows)

    def print_stats(self) -> None:
        """In thống kê usage."""
        print("\n[InStore] ── Usage Stats ──────────────────────")
        print(f"  API calls    : {self.stats['llm_calls']}")
        print(f"  Cache hits   : {self.stats['cache_hits']}")
        print(f"  Fallback     : {self.stats['fallback_used']}")
        print(f"  Total tokens : {self.stats['total_tokens']:,}")
        if self.stats['llm_calls'] > 0:
            est_cost = self.stats['total_tokens'] / 1_000_000 * 5.0
            print(f"  Est. cost    : ~${est_cost:.4f} USD")
        print("  ──────────────────────────────────────────────")

    def print_script_report(self, df: pd.DataFrame) -> None:
        """In report chi tiết từng script cho TVV."""
        print("\n" + "=" * 70)
        print("IN-STORE SALES SCRIPTS — TVV REPORT")
        print("=" * 70)

        for _, row in df.iterrows():
            print(f"\n{'─' * 70}")
            print(f"  👤 {row['customer_id']:8s} | {row['instore_intent']:15s} | "
                  f"{row['nba_strategy']:20s} | {row['priority'].upper()}")
            print(f"  Phân khúc: {row['segment_rfm_tier']} | "
                  f"Chi tiêu: {row['monetary']:,.0f}đ | "
                  f"Ngân sách: {row['budget']}")
            print(f"  Tâm lý học: {row['psychology_trigger']}")
            if row.get("urgency_signal"):
                print(f"  ⚡ {row['urgency_signal']}")
            print(f"  Online: {row['online_insight']}")
            print(f"  💡 Key Insight: {row['key_insight']}")
            print(f"\n  📦 Sản phẩm gợi ý:")
            for col in ["product_rec_1", "product_rec_2", "product_rec_3"]:
                if row.get(col):
                    print(f"     • {row[col]}")
            print(f"\n  🗣️  Script 5 bước:")
            print(f"  1. Opening   : {row['script_opening']}")
            print(f"  2. Khai thác : {row['script_khai_thac']}")
            print(f"  3. Gợi ý     : {row['script_goi_y']}")
            print(f"  4. Chốt      : {row['script_chot']}")
            print(f"  5. Upsell    : {row['script_upsell']}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Standalone demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.lep_pipeline import LEPModel

    print("=" * 70)
    print("IN-STORE SCRIPT ENGINE — DEMO")
    print("=" * 70)

    DATA_PATH = "data/customer_data_poc_enhanced.xlsx"
    df_p  = pd.read_excel(DATA_PATH, sheet_name="profiles_enhanced")
    df_ml = pd.read_excel(DATA_PATH, sheet_name="ml_predictions")

    lep   = LEPModel()
    lep.train(df_p, df_ml, verbose=False)
    preds = lep.predict(df_p)

    engine = InstoreScriptEngine()
    result = engine.generate_scripts(preds, df_p, verbose=True)
    engine.print_script_report(result)

    print(f"\n[Done] {len(result)} scripts generated")
    print(result[["customer_id", "instore_intent", "nba_strategy",
                  "script_opening", "key_insight"]].to_string(index=False))
