"""
src/llm_message_generator.py
────────────────────────────────────────────────────────────────────────────
LLM Message Generator — OpenAI API Integration
Sinh nội dung marketing cá nhân hóa cho từng khách hàng trang sức.

Nội dung được sinh DỰA TRÊN KEY INSIGHT từ phân tích In-Store (Nhánh 1).
Mỗi channel (ZNS / Email / Push / In-App / Store) có format và độ dài riêng.
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
from dataclasses import dataclass

# ── OpenAI import (graceful fallback nếu chưa cài) ───────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[LLM] WARNING: openai không được cài. Chạy: pip install openai")
    print("[LLM] Sẽ dùng fallback template mode.")


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CustomerContext:
    """Ngữ cảnh khách hàng — bao gồm cả thông tin instore & outbound."""
    # ── Core profile ──
    customer_id:        str
    intent:             str          # engagement / anniversary / self_reward / gift
    confidence:         float
    priority:           str          # low / medium / high
    budget:             str
    style:              str
    preferred_type:     str          # Nhẫn / Bông tai / ...
    material:           str
    recency_days:       int
    monetary:           float
    avg_discount:       float
    segment_rfm_tier:   str
    birthday_in_days:   int
    sig_view_diamond:   int
    sig_view_ring:      int
    sig_search_propose: int
    camp_engagement:    int
    camp_anniversary:   int
    camp_selfreward:    int
    channel:            str          # email / push / zns / in_app / store
    product_focus:      str
    cta:                str

    # ── Giới tính: M = Nam (anh), F = Nữ (chị) ─────────────────────────
    gender:             str = "F"

    # ── Instore analysis context (từ instore_scripts.json — Nhánh 1) ──
    key_insight:        str = ""     # ← Trường quan trọng nhất để sinh message
    urgency_signal:     str = ""     # Tín hiệu urgency từ phân tích
    online_insight:     str = ""     # Tóm tắt hành vi online
    instore_intent:     str = ""     # High Purchase / Exploration / Premium / Low Intent
    nba_strategy:       str = ""     # Chiến lược NBA
    psychology_trigger: str = ""     # Đòn tâm lý
    product_rec_1:      str = ""     # Sản phẩm gợi ý 1
    product_rec_2:      str = ""     # Sản phẩm gợi ý 2
    product_rec_3:      str = ""     # Sản phẩm gợi ý 3


@dataclass
class GeneratedMessage:
    """Kết quả message sinh từ LLM — có cấu trúc đầy đủ theo từng channel."""
    customer_id:    str
    channel:        str
    # Channel-specific structured fields
    subject:        Optional[str]    # Email: subject; Push/InApp: tiêu đề thông báo
    greeting:       Optional[str]    # ZNS/Email/Store: câu mở đầu insight-driven
    body:           str              # Nội dung chính
    highlights:     list             # Danh sách điểm nổi bật (# bullets / • bullets)
    closing:        Optional[str]    # ZNS/Email/Store: câu kết / mời tương tác
    cta_text:       str
    tone:           str              # warm / urgent / luxurious / friendly
    tokens_used:    int
    source:         str              # "llm" hoặc "fallback"
    raw_json:       dict


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — dựa trên Key Insight
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là chuyên gia marketing trang sức cao cấp PNJ tại Việt Nam.
Nhiệm vụ: Dựa trên KEY INSIGHT về hành vi và tín hiệu của khách hàng, sinh nội dung
marketing CÁ NHÂN HÓA — sâu sắc, thiết thực, KHÔNG sáo rỗng, KHÔNG chung chung giữa các khách.

Nguyên tắc bắt buộc:
1. Nội dung PHẢI phản ánh đúng Key Insight — mỗi khách một thông điệp KHÁC NHAU, không copy-paste
2. Viết tiếng Việt chuẩn, giọng ấm áp lịch sự — xưng "em", gọi khách đúng theo giới tính được cung cấp: "anh" nếu Nam, "chị" nếu Nữ. TUYỆT ĐỐI không dùng "bạn" hay "anh/chị" gộp chung
3. KHÔNG dùng: "ưu đãi không thể bỏ lỡ", "đừng bỏ lỡ", "cơ hội vàng", "flash sale"
4. Nếu có urgency (sinh nhật gần, giỏ hàng bỏ dở) → lồng ghép khéo léo, tự nhiên vào nội dung
5. Nếu insight liên quan đến cầu hôn → tiếp cận tế nhị, KHÔNG nói thẳng "bạn sắp cầu hôn"
6. TUYỆT ĐỐI không dùng ký hiệu "#" (hashtag) trong bất kỳ trường nào — không làm bullet, heading
7. ZNS highlights dùng "✦ ", email bullets dùng "▸ ", in-app dùng "✓ "
8. Trả về ĐÚNG format JSON được yêu cầu — không thêm markdown hay giải thích ngoài JSON"""


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL PROMPTS — Format riêng cho từng kênh
# ══════════════════════════════════════════════════════════════════════════════

CHANNEL_PROMPTS = {

    "zns": """Sinh tin nhắn ZNS Zalo đầy đủ — tuân theo định dạng và giới hạn nội dung Zalo OA.

FORMAT JSON bắt buộc:
{
  "greeting": "1-2 câu mở đầu — gợi nhắc insight khách tế nhị, tự nhiên (VD: về khoảnh khắc đặc biệt, về sự quan tâm đến sản phẩm)",
  "body": "2-3 câu chính — PNJ có gì dành cho dịp/nhu cầu này, cụ thể sản phẩm, không chung chung",
  "highlights": ["✦ Điểm nổi bật 1 — ngắn gọn", "✦ Điểm nổi bật 2", "✦ Điểm nổi bật 3"],
  "closing": "1-2 câu kết — đội tư vấn PNJ sẵn sàng, mời tương tác tự nhiên không ép buộc",
  "cta_text": "3-5 chữ hành động cụ thể",
  "tone": "warm | luxurious | friendly",
  "subject": null
}

Quy tắc ZNS:
- Greeting: câu hook dựa vào insight — không nói thẳng "bạn sắp cầu hôn"
- Body: 2-3 câu rõ ràng, mỗi câu 1 ý, đề cập sản phẩm phù hợp với insight
- Highlights: đúng 3 mục, bắt đầu bằng "✦ " (TUYỆT ĐỐI không dùng "#")
- Closing: 1-2 câu, gợi mở liên hệ
- Tổng toàn bộ nội dung (greeting + body + highlights + closing): 150-250 chữ""",

    "email": """Sinh email marketing đầy đủ — có cấu trúc rõ ràng như email thương hiệu cao cấp thực tế.

FORMAT JSON bắt buộc:
{
  "subject": "Tiêu đề email — dưới 60 chữ, gợi tò mò liên quan đến insight, không CAPS LOCK",
  "greeting": "Kính gửi anh/chị,",
  "body": "Đoạn 1 (2-3 câu): Opening hook — nhắc đến dịp/nhu cầu từ insight một cách tự nhiên và tinh tế.\\n\\nĐoạn 2 (2-3 câu): PNJ có gì phù hợp — sản phẩm cụ thể, thiết kế, đặc điểm nổi bật.\\n\\nĐoạn 3 (1-2 câu): Lý do nên hành động sớm — urgency nếu có, hoặc giá trị riêng.",
  "highlights": ["• Điểm nổi bật 1 — cụ thể về sản phẩm/dịch vụ PNJ", "• Điểm nổi bật 2 — lý do chọn PNJ", "• Điểm nổi bật 3 — cam kết / dịch vụ thêm"],
  "closing": "Đội ngũ tư vấn PNJ sẵn sàng hỗ trợ anh/chị chọn lựa phù hợp nhất. Liên hệ hotline 1800 545457 hoặc ghé cửa hàng PNJ gần nhất.",
  "cta_text": "3-6 chữ — rõ ràng như 'Xem bộ sưu tập' hoặc 'Đặt tư vấn ngay'",
  "tone": "warm | luxurious | friendly | urgent"
}

Quy tắc Email:
- Subject: liên quan đúng insight, tạo tò mò tự nhiên
- Body: 3 đoạn rõ ràng, cách nhau \\n\\n, tổng 120-200 chữ
- Highlights: đúng 3 bullets với thông tin cụ thể, không chung chung
- Closing: câu kết + thông tin liên hệ thực tế""",

    "push": """Sinh push notification mobile — cực ngắn, tạo tò mò tức thì, liên quan đến insight.

FORMAT JSON bắt buộc:
{
  "subject": "Tiêu đề thông báo — tối đa 50 chữ, có thể dùng 1 emoji duy nhất, liên quan insight",
  "body": "Nội dung thông báo — tối đa 80 chữ, 1-2 câu, đặt câu hỏi hoặc gợi mở liên quan đến insight",
  "highlights": [],
  "greeting": null,
  "closing": null,
  "cta_text": "2-4 chữ ngắn gọn",
  "tone": "urgent | friendly | warm"
}

Quy tắc Push:
- Subject phải liên quan đến key insight (không generic "PNJ có ưu đãi mới")
- Body: câu hỏi gợi mở hoặc thông tin liên quan đến nhu cầu cụ thể
- 1 emoji tối đa trong toàn bộ thông báo (subject hoặc body, chọn 1)
- Không quá 130 chữ tổng cộng""",

    "in_app": """Sinh in-app banner — khi khách đang browse app, cần nắm sự chú ý trong 3 giây.

FORMAT JSON bắt buộc:
{
  "subject": "Tiêu đề banner — tối đa 50 chữ, bold + emoji, liên quan trực tiếp đến insight",
  "body": "Nội dung banner — 2-3 câu, tối đa 100 chữ, nêu rõ giá trị và sản phẩm phù hợp insight",
  "highlights": ["Điểm nhanh 1 — lợi ích cụ thể", "Điểm nhanh 2 — điểm khác biệt PNJ"],
  "greeting": null,
  "closing": null,
  "cta_text": "2-4 chữ",
  "tone": "warm | luxurious | friendly"
}

Quy tắc In-App:
- Subject/Tiêu đề: bold + emoji, liên quan insight (không phải "Xem ngay!")
- Body: ngắn nhưng đủ thuyết phục để khách click — nhắc insight, nêu sản phẩm
- 2 highlights nêu điểm khác biệt cụ thể""",

    "store": """Sinh thẻ thông tin nhanh cho Tư Vấn Viên (TVV) khi khách quét loyalty card tại cửa hàng.

FORMAT JSON bắt buộc:
{
  "subject": "⚡ Insight TVV cần biết: [tóm tắt key insight ngắn gọn cho TVV]",
  "greeting": "Câu TVV mở đầu với khách — tự nhiên, không để lộ biết thông tin từ hệ thống. Xưng 'em', gọi khách đúng theo giới tính (anh/chị) được cung cấp.",
  "body": "Script ngắn cho TVV (3-4 câu): nhắc insight theo cách tự nhiên → giới thiệu sản phẩm phù hợp → tạo cảm giác được phục vụ riêng. Xưng 'em', gọi khách đúng theo giới tính được cung cấp (anh/chị) — dùng nhất quán xuyên suốt script.",
  "highlights": ["Sản phẩm ưu tiên giới thiệu: [tên SP cụ thể]", "Điểm bán hàng: [lý do phù hợp với insight]"],
  "closing": "Hành động tiếp theo TVV nên làm sau câu mở đầu",
  "cta_text": "Hành động TVV — VD: 'Dẫn khách xem khu Nhẫn Kim cương'",
  "tone": "warm"
}""",
}


# ══════════════════════════════════════════════════════════════════════════════
# INTENT CONTEXT — mô tả insight theo từng intent type
# ══════════════════════════════════════════════════════════════════════════════

INTENT_CONTEXT_PROMPTS = {
    "engagement":  "Insight: Khách có dấu hiệu chuẩn bị cho dịp cầu hôn/đính hôn (xem nhẫn, tìm kiếm). Tiếp cận tế nhị, lãng mạn — KHÔNG nói thẳng 'bạn sắp cầu hôn'. Nhắc đến khoảnh khắc đặc biệt, dịp ý nghĩa.",
    "anniversary": "Insight: Khách có xu hướng mua dịp kỷ niệm tình yêu/ngày cưới. Nhấn vào cảm xúc trân trọng — mỗi năm là một trang đẹp, quà ý nghĩa ghi dấu cột mốc.",
    "self_reward": "Insight: Khách thích tự thưởng cho bản thân. Nhấn vào sự tự tin và xứng đáng — không phụ thuộc ai, không cần dịp đặc biệt.",
    "gift":        "Insight: Khách đang tìm quà tặng cho người thân/bạn bè. Nhấn vào việc chọn đúng quà — ý nghĩa với người nhận, dịch vụ hỗ trợ chọn quà.",
}


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK TEMPLATES — Rich content, dựa trên intent + channel + insight
# ══════════════════════════════════════════════════════════════════════════════

def _pronoun(gender: str) -> str:
    """Trả về 'anh' cho Nam (M), 'chị' cho Nữ (F)."""
    return "anh" if str(gender).upper() == "M" else "chị"


def _apply_gender(d: dict, pn: str) -> dict:
    """Thay 'anh/chị' và 'Anh/chị' trong dict bằng đại từ đúng giới tính."""
    Pn = pn.capitalize()
    def _fix(v: object) -> object:
        if isinstance(v, str):
            return v.replace("anh/chị", pn).replace("Anh/chị", Pn)
        if isinstance(v, list):
            return [_fix(i) for i in v]
        return v
    return {k: _fix(v) for k, v in d.items()}


def _build_fallback(ctx: CustomerContext) -> dict:
    """
    Sinh fallback template phong phú dựa trên key_insight, urgency, channel.
    Mỗi channel có cấu trúc riêng phù hợp thực tế.
    """
    pn       = _pronoun(ctx.gender)   # "anh" hoặc "chị"
    insight  = ctx.key_insight.strip()
    urgency  = ctx.urgency_signal.strip()
    channel  = ctx.channel
    product  = ctx.product_focus or ctx.preferred_type or "trang sức"
    style    = ctx.style

    # ── Xác định ngữ cảnh từ insight ─────────────────────────────────────────
    is_proposal    = any(k in insight.lower() for k in ["cầu hôn", "đính hôn", "nhẫn"])
    is_birthday    = any(k in insight.lower() for k in ["sinh nhật", "birthday"])
    is_anniversary = any(k in insight.lower() for k in ["kỷ niệm", "anniversary"])
    is_self_reward = any(k in insight.lower() for k in ["tự thưởng", "self", "bản thân"])
    has_urgency    = bool(urgency)

    # ── Chọn hook dựa trên insight ────────────────────────────────────────────
    if is_proposal:
        hook_greeting = "Mỗi khoảnh khắc yêu thương đều xứng đáng được trân trọng bằng một món quà đặc biệt."
        hook_body = (
            f"PNJ gửi đến anh/chị những thiết kế {product} tinh tuyển, "
            "phù hợp cho dịp sinh nhật, kỷ niệm hay lời cầu hôn ý nghĩa.\n"
            "Từng chi tiết được chế tác tỉ mỉ — để khoảnh khắc quan trọng của anh/chị thêm trọn vẹn."
        )
        highlights_base = [
            "✦ Sang trọng trong thiết kế",
            "✦ Tinh tế trong từng chi tiết",
            "✦ Gửi trọn thành ý cho người thương",
        ]
        email_bullets = [
            "• Bộ sưu tập nhẫn đính hôn & nhẫn cưới — GIA certified, vàng 18K",
            "• Dịch vụ khắc tên & ngày đặc biệt lên sản phẩm miễn phí",
            "• Đội tư vấn riêng — đồng hành từng bước cho dịp trọng đại",
        ]
        subject_email = "Dành cho khoảnh khắc chỉ có một lần — PNJ có điều muốn chia sẻ"
        subject_push  = "💍 Khoảnh khắc đặc biệt xứng đáng được chuẩn bị kỹ"
        body_push     = "Bộ sưu tập nhẫn đính hôn PNJ vừa cập nhật mẫu mới — xem ngay để chuẩn bị thật hoàn hảo."
        store_subject = f"⚡ Insight TVV: Khách quan tâm {product}, có thể đang chuẩn bị dịp cầu hôn/kỷ niệm"
        store_greeting = f"Chào anh/chị! Anh/chị đang tìm {product} — bên em vừa có đợt mẫu đặc biệt rất phù hợp cho những dịp trọng đại."
        tone = "luxurious"

    elif is_birthday:
        hook_greeting = "Dịp sinh nhật thật sự ý nghĩa khi có một món quà được lựa chọn kỹ lưỡng."
        hook_body = (
            f"PNJ gửi đến anh/chị những gợi ý {product} tinh tế — "
            "vừa đẹp vừa mang ý nghĩa riêng, phù hợp để tự thưởng hay tặng người thân nhân dịp sinh nhật."
        )
        highlights_base = [
            "✦ Quà sinh nhật được cá nhân hóa theo sở thích",
            "✦ Dịch vụ khắc tên & gói quà cao cấp miễn phí",
            "✦ Giao hàng đúng hẹn — sẵn sàng cho ngày đặc biệt",
        ]
        email_bullets = [
            f"• Bộ sưu tập {product} Birthday Collection — đa dạng mức giá",
            "• Dịch vụ khắc tên & đóng hộp quà sang trọng không phụ phí",
            "• Hỗ trợ giao hàng trong ngày tại TP.HCM & Hà Nội",
        ]
        subject_email = f"Sinh nhật sắp đến — {product} phù hợp nhất cho anh/chị"
        subject_push  = "🎂 Dịp sinh nhật sắp đến — PNJ có gợi ý riêng cho anh/chị"
        body_push     = f"Bộ sưu tập {product} mới nhất đang chờ — chọn ngay để kịp cho ngày đặc biệt."
        store_subject = f"⚡ Insight TVV: Sinh nhật sắp đến — ưu tiên giới thiệu {product} và quà tặng"
        store_greeting = f"Chào anh/chị! Bên em đang có đợt {product} mới về rất đẹp — có cả dịch vụ gói quà riêng nếu anh/chị cần nhé."
        tone = "warm"

    elif is_anniversary:
        hook_greeting = "Mỗi kỷ niệm là một trang đẹp xứng đáng được ghi dấu theo cách riêng."
        hook_body = (
            f"PNJ gửi đến anh/chị những thiết kế {product} ý nghĩa — "
            "có thể khắc tên, ngày kỷ niệm để lưu giữ cột mốc đặc biệt của hai người."
        )
        highlights_base = [
            "✦ Trang sức kỷ niệm — khắc ngày tháng theo yêu cầu",
            "✦ Chứng nhận chất lượng PNJ — bền theo năm tháng",
            "✦ Tư vấn cá nhân hóa từng chi tiết cho đôi bạn",
        ]
        email_bullets = [
            f"• Bộ sưu tập {product} kỷ niệm — thiết kế đôi, tinh tế",
            "• Dịch vụ khắc tên + ngày kỷ niệm lên sản phẩm",
            "• Gift box nhung cao cấp — trọn bộ quà không cần thêm gì",
        ]
        subject_email = "Kỷ niệm xứng đáng được ghi nhớ mãi — PNJ có điều muốn gợi ý"
        subject_push  = "💕 Sắp đến ngày kỷ niệm — chuẩn bị một điều thật ý nghĩa"
        body_push     = f"Bộ {product} kỷ niệm có thể khắc tên và ngày đặc biệt — đặt trước 2 ngày là có ngay."
        store_subject = f"⚡ Insight TVV: Khách mua dịp kỷ niệm — ưu tiên {product} đôi + dịch vụ khắc tên"
        store_greeting = f"Chào anh/chị! Bên em đang có dịch vụ khắc tên miễn phí lên {product} — rất phù hợp cho dịp kỷ niệm đặc biệt ạ."
        tone = "warm"

    elif is_self_reward:
        hook_greeting = f"Đôi khi điều tốt nhất {pn} có thể làm là tự trân trọng chính mình."
        hook_body = (
            f"PNJ gửi đến anh/chị những mẫu {product} mới nhất — "
            f"phong cách {style.lower() if style else 'tinh tế'}, phù hợp đeo hàng ngày "
            "hay cho những dịp muốn tỏa sáng."
        )
        highlights_base = [
            f"✦ Bộ sưu tập {product} mới — chưa đại trà",
            "✦ Chất lượng vàng chuẩn PNJ — bảo hành toàn quốc",
            "✦ Phong cách đa dạng — tìm đúng mẫu cho cá tính riêng",
        ]
        email_bullets = [
            f"• New arrivals {product} tháng này — mẫu mới, chưa đại trà",
            "• Đa dạng phong cách từ tối giản đến nổi bật — phù hợp mọi cá tính",
            "• Bảo hành 12 tháng, đổi trả trong 7 ngày tại tất cả cửa hàng PNJ",
        ]
        subject_email = f"Tự thưởng cho bản thân — {product} mới vừa về tại PNJ"
        subject_push  = "✨ Mẫu mới vừa về — phong cách phù hợp cho anh/chị"
        body_push     = f"Bộ sưu tập {product} mới nhất của PNJ — phong cách đa dạng, xem thử để tìm mẫu ưng ý nhất."
        store_subject = f"⚡ Insight TVV: Khách thích tự thưởng — giới thiệu {product} mới + bestseller"
        store_greeting = f"Chào anh/chị! Bên em vừa về đợt {product} mới — mẫu đang được nhiều khách ưa chuộng lắm, để em lấy ra cho anh/chị xem thử nhé."
        tone = "friendly"

    else:
        # Generic fallback dựa trên product focus
        hook_greeting = f"Anh/chị đang quan tâm đến {product} — PNJ có những lựa chọn rất phù hợp dành cho anh/chị."
        hook_body = (
            f"Bộ sưu tập {product} mới nhất của PNJ được thiết kế với nhiều phong cách đa dạng — "
            "từ tối giản thanh lịch đến nổi bật cá tính, phù hợp cho mọi dịp."
        )
        highlights_base = [
            f"✦ Đa dạng mẫu {product} — nhiều phong cách lựa chọn",
            "✦ Chất lượng vàng chuẩn PNJ — cam kết bảo hành",
            "✦ Tư vấn miễn phí — đội ngũ chuyên nghiệp sẵn sàng",
        ]
        email_bullets = [
            f"• Bộ sưu tập {product} đa dạng — phù hợp nhiều phong cách",
            "• Chất lượng chuẩn PNJ — bảo hành toàn quốc",
            "• Tư vấn miễn phí tại cửa hàng hoặc online",
        ]
        subject_email = f"PNJ gửi đến anh/chị — bộ sưu tập {product} phù hợp nhất"
        subject_push  = f"💎 {product} PNJ — mẫu phù hợp với anh/chị đang chờ"
        body_push     = f"Bộ sưu tập {product} của PNJ với nhiều phong cách — xem ngay để tìm mẫu ưng ý."
        store_subject = f"⚡ Insight TVV: Khách quan tâm {product} — tư vấn đa dạng lựa chọn"
        store_greeting = f"Chào anh/chị! Bên em đang có đợt {product} rất đẹp — để em giới thiệu vài mẫu phù hợp cho anh/chị nhé."
        tone = "friendly"

    # ── Urgency lồng ghép ─────────────────────────────────────────────────────
    urgency_note = ""
    if has_urgency:
        if "sinh nhật" in urgency.lower():
            urgency_note = " Dịp sinh nhật đang đến gần — đặt trước để kịp chuẩn bị."
        elif "giỏ hàng" in urgency.lower() or "cart" in urgency.lower():
            urgency_note = " Sản phẩm anh/chị đã chọn trước đó vẫn còn — số lượng có hạn."
        elif "cầu hôn" in urgency.lower():
            urgency_note = " Thời điểm lý tưởng để chuẩn bị — đội tư vấn sẵn sàng hỗ trợ riêng."

    closing_base = (
        "Đội ngũ PNJ sẵn sàng đồng hành và tư vấn lựa chọn phù hợp nhất dành riêng cho anh/chị."
        + urgency_note
    )
    cta_zns   = "Khám phá ngay"
    cta_email = "Xem bộ sưu tập"
    cta_push  = "Mở xem ngay"
    cta_inapp = "Xem ngay"
    cta_store = "Tư vấn riêng"

    # ── Assemble theo channel ─────────────────────────────────────────────────
    if channel == "zns":
        return _apply_gender({
            "greeting":   hook_greeting,
            "body":       hook_body,
            "highlights": highlights_base,
            "closing":    closing_base + "\n* Khám phá bộ sưu tập mới nhất hoặc phản hồi tin nhắn để được hỗ trợ nhanh chóng.",
            "cta_text":   cta_zns,
            "tone":       tone,
            "subject":    None,
        }, pn)

    if channel == "email":
        body_3para = (
            f"{hook_greeting}\n\n"
            f"{hook_body}\n\n"
            + (f"Lưu ý: {urgency_note.strip()}" if urgency_note else
               f"Hãy để PNJ đồng hành cùng {pn} tìm món trang sức thật phù hợp.")
        )
        return _apply_gender({
            "subject":    subject_email,
            "greeting":   f"Kính gửi {pn},",
            "body":       body_3para,
            "highlights": email_bullets,
            "closing":    f"Đội ngũ tư vấn PNJ sẵn sàng hỗ trợ {pn} chọn lựa phù hợp nhất. "
                          "Liên hệ hotline 1800 545457 hoặc ghé cửa hàng PNJ gần nhất.",
            "cta_text":   cta_email,
            "tone":       tone,
        }, pn)

    if channel == "push":
        return _apply_gender({
            "subject":    subject_push,
            "body":       body_push + (urgency_note if urgency_note else ""),
            "highlights": [],
            "greeting":   None,
            "closing":    None,
            "cta_text":   cta_push,
            "tone":       "urgent" if has_urgency else tone,
        }, pn)

    if channel == "in_app":
        inapp_subject = f"💎 {product} mới — xem ngay" if not is_proposal else f"💍 Dành cho khoảnh khắc đặc biệt"
        inapp_body = (
            f"{hook_greeting} "
            f"PNJ có những mẫu {product} phù hợp nhất cho {pn} — "
            + (urgency_note.strip() if urgency_note else "khám phá ngay bộ sưu tập mới nhất.")
        )
        return _apply_gender({
            "subject":    inapp_subject,
            "body":       inapp_body,
            "highlights": [highlights_base[0].replace("✦ ", "✓ "), highlights_base[1].replace("✦ ", "✓ ")],
            "greeting":   None,
            "closing":    None,
            "cta_text":   cta_inapp,
            "tone":       tone,
        }, pn)

    if channel == "store":
        store_highlights = [
            f"Sản phẩm ưu tiên: {ctx.product_rec_1 or product}",
            f"Điểm bán hàng: {highlights_base[0].replace('✦ ', '')}",
        ]
        return _apply_gender({
            "subject":    store_subject,
            "greeting":   store_greeting,
            "body":       (
                f"{hook_body} "
                f"{'Lưu ý: ' + urgency_note.strip() if urgency_note else ''} "
                f"Dẫn {pn} xem mẫu phù hợp và hỏi thêm về dịp đặc biệt để tư vấn đúng hơn."
            ).strip(),
            "highlights": store_highlights,
            "closing":    f"Sau khi khai thác thêm nhu cầu, chào mời {pn} thử sản phẩm trực tiếp.",
            "cta_text":   cta_store,
            "tone":       "warm",
        }, pn)

    # Generic fallback
    return _apply_gender({
        "subject":    subject_email,
        "greeting":   hook_greeting,
        "body":       hook_body,
        "highlights": highlights_base,
        "closing":    closing_base,
        "cta_text":   cta_email,
        "tone":       tone,
    }, pn)


# ══════════════════════════════════════════════════════════════════════════════
# CACHE (tránh gọi API lặp)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleCache:
    """File-based cache đơn giản cho LLM responses."""

    def __init__(self, cache_dir: str = "outputs/.llm_cache"):
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
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2),
                        encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class LLMMessageGenerator:
    """
    Sinh marketing message cá nhân hóa dựa trên Key Insight từ instore analysis.
    Tự động fallback về rich template nếu không có API key.
    """

    MODEL = "gpt-4o"

    def __init__(self,
                 api_key: Optional[str] = None,
                 use_cache: bool = True,
                 rate_limit_delay: float = 0.3):
        self.api_key          = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.use_cache        = use_cache
        self.rate_limit_delay = rate_limit_delay
        self.cache            = SimpleCache() if use_cache else None

        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.mode   = "llm"
            print(f"[LLM] Mode: OpenAI API ({self.MODEL})")
        else:
            self.client = None
            self.mode   = "fallback"
            reason = "openai không được cài" if not OPENAI_AVAILABLE else "không có API key"
            print(f"[LLM] Mode: Fallback template ({reason})")

        self.stats = {
            "llm_calls": 0, "cache_hits": 0,
            "fallback_used": 0, "total_tokens": 0,
        }

    def _build_user_prompt(self, ctx: CustomerContext) -> str:
        """Xây dựng user prompt từ customer context — Key Insight là trung tâm."""
        channel_inst  = CHANNEL_PROMPTS.get(ctx.channel, CHANNEL_PROMPTS["zns"])
        intent_ctx    = INTENT_CONTEXT_PROMPTS.get(ctx.intent, "")

        # ── KEY INSIGHT SECTION ───────────────────────────────────────────────
        insight_section = f"""
═══ KEY INSIGHT (DỰA VÀO ĐÂY ĐỂ SINH NỘI DUNG) ═══
{ctx.key_insight if ctx.key_insight else "Không có insight cụ thể — dùng profile và intent để suy luận."}

Tín hiệu urgency: {ctx.urgency_signal if ctx.urgency_signal else "Không có"}
Hành vi online: {ctx.online_insight if ctx.online_insight else "Không có"}
Loại khách tại cửa hàng: {ctx.instore_intent if ctx.instore_intent else "Chưa xác định"}
Chiến lược NBA: {ctx.nba_strategy if ctx.nba_strategy else "Chưa xác định"}
"""

        # ── CUSTOMER PROFILE ──────────────────────────────────────────────────
        _gender_label = "Nam" if ctx.gender.upper() == "M" else "Nữ"
        _pn           = _pronoun(ctx.gender)
        profile_section = f"""
═══ THÔNG TIN KHÁCH HÀNG ═══
- Giới tính: {_gender_label} — PHẢI gọi khách là "{_pn}" xuyên suốt toàn bộ nội dung (không dùng "anh/chị" gộp chung, không dùng "bạn")
- Phân khúc: {ctx.segment_rfm_tier} | Ngân sách: {ctx.budget}
- Phong cách: {ctx.style} | Sản phẩm ưa thích: {ctx.preferred_type} ({ctx.material})
- Mua gần nhất: {ctx.recency_days} ngày trước | Tổng chi tiêu: {ctx.monetary:,.0f}đ
- Sinh nhật còn: {ctx.birthday_in_days} ngày
- Tín hiệu xem nhẫn: {ctx.sig_view_ring} | Tín hiệu xem kim cương: {ctx.sig_view_diamond}
- Tín hiệu tìm kiếm cầu hôn: {ctx.sig_search_propose}

Intent dự đoán (LEP): {ctx.intent.upper()} (confidence: {ctx.confidence:.0%})
{intent_ctx}

Sản phẩm tập trung: {ctx.product_focus}
"""

        # ── PRODUCT RECOMMENDATIONS ───────────────────────────────────────────
        recs = [r for r in [ctx.product_rec_1, ctx.product_rec_2, ctx.product_rec_3] if r]
        prod_section = ""
        if recs:
            prod_section = f"\nSản phẩm gợi ý từ phân tích instore:\n" + "\n".join(f"  - {r}" for r in recs)

        # ── CHANNEL INSTRUCTION ───────────────────────────────────────────────
        channel_section = f"""
═══ KÊNH GIAO TIẾP: {ctx.channel.upper()} ═══
{channel_inst}
"""

        return (
            "Sinh nội dung marketing cho khách hàng sau — "
            "DỰA CHÍNH VÀO KEY INSIGHT để nội dung đúng với nhu cầu thực tế.\n"
            + insight_section
            + profile_section
            + prod_section
            + channel_section
        )

    def _call_api(self, user_prompt: str) -> tuple[dict, int]:
        """Gọi OpenAI API và parse JSON response."""
        if self.use_cache and self.cache:
            cached = self.cache.get(user_prompt)
            if cached:
                self.stats["cache_hits"] += 1
                return cached, 0

        time.sleep(self.rate_limit_delay)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            max_tokens=600,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw_text = response.choices[0].message.content.strip()
        tokens   = response.usage.prompt_tokens + response.usage.completion_tokens

        # Strip markdown fences nếu có
        clean = raw_text
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        clean = clean.strip()

        parsed = json.loads(clean)

        if self.use_cache and self.cache:
            self.cache.set(user_prompt, parsed)

        self.stats["llm_calls"]    += 1
        self.stats["total_tokens"] += tokens
        return parsed, tokens

    def _parse_message(self, raw_json: dict, ctx: CustomerContext, tokens: int, source: str) -> GeneratedMessage:
        """Parse raw LLM JSON response thành GeneratedMessage chuẩn."""
        # Normalize highlights — đảm bảo là list[str]
        raw_highlights = raw_json.get("highlights", [])
        if isinstance(raw_highlights, str):
            try:
                raw_highlights = json.loads(raw_highlights)
            except Exception:
                raw_highlights = [raw_highlights] if raw_highlights else []
        highlights = [str(h) for h in raw_highlights if h]

        return GeneratedMessage(
            customer_id  = ctx.customer_id,
            channel      = ctx.channel,
            subject      = raw_json.get("subject") or None,
            greeting     = raw_json.get("greeting") or None,
            body         = str(raw_json.get("body", "")).strip(),
            highlights   = highlights,
            closing      = raw_json.get("closing") or None,
            cta_text     = str(raw_json.get("cta_text", ctx.cta)).strip(),
            tone         = str(raw_json.get("tone", "warm")),
            tokens_used  = tokens,
            source       = source,
            raw_json     = raw_json,
        )

    def generate(self, ctx: CustomerContext) -> GeneratedMessage:
        """Sinh message cho một khách hàng."""
        tokens = 0
        source = "fallback"

        try:
            if self.mode == "llm":
                prompt   = self._build_user_prompt(ctx)
                raw_json, tokens = self._call_api(prompt)
                source   = "llm"
            else:
                raw_json = _build_fallback(ctx)
                self.stats["fallback_used"] += 1
        except (json.JSONDecodeError, Exception) as exc:
            print(f"[LLM] Error cho {ctx.customer_id}: {exc}. Dùng fallback.")
            raw_json = _build_fallback(ctx)
            source   = "fallback_error"
            self.stats["fallback_used"] += 1

        self.stats["total_tokens"] += tokens
        return self._parse_message(raw_json, ctx, tokens, source)

    def generate_batch(self, contexts: list[CustomerContext],
                       verbose: bool = True) -> list[GeneratedMessage]:
        """Sinh message cho nhiều khách, có progress tracking."""
        results = []
        total   = len(contexts)

        for i, ctx in enumerate(contexts, 1):
            if verbose:
                print(f"  [{i:3d}/{total}] {ctx.customer_id} | insight: {ctx.key_insight[:40]}..." if ctx.key_insight
                      else f"  [{i:3d}/{total}] {ctx.customer_id} | {ctx.intent:12s} | {ctx.channel:8s}",
                      end=" ")

            msg = self.generate(ctx)
            results.append(msg)

            if verbose:
                icon = "🤖" if msg.source == "llm" else "📄"
                print(f"{icon} {msg.tone:10s} | {msg.body[:50]}...")

        return results

    def print_stats(self) -> None:
        """In thống kê sử dụng API."""
        print("\n[LLM] ── Usage Stats ──────────────────────")
        print(f"  API calls    : {self.stats['llm_calls']}")
        print(f"  Cache hits   : {self.stats['cache_hits']}")
        print(f"  Fallback     : {self.stats['fallback_used']}")
        print(f"  Total tokens : {self.stats['total_tokens']:,}")
        if self.stats["llm_calls"] > 0:
            est_cost = self.stats["total_tokens"] / 1_000_000 * 5.0
            print(f"  Est. cost    : ~${est_cost:.4f} USD")
        print("  ──────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: profile row + instore context → CustomerContext
# ══════════════════════════════════════════════════════════════════════════════

def row_to_context(
    row:            pd.Series,
    intent:         str,
    confidence:     float,
    priority:       str,
    channel:        str,
    product_focus:  str,
    cta:            str,
    # Instore context fields (from instore_scripts.json)
    key_insight:        str = "",
    urgency_signal:     str = "",
    online_insight:     str = "",
    instore_intent:     str = "",
    nba_strategy:       str = "",
    psychology_trigger: str = "",
    product_rec_1:      str = "",
    product_rec_2:      str = "",
    product_rec_3:      str = "",
) -> CustomerContext:
    """Chuyển đổi profile row + instore context thành CustomerContext."""
    return CustomerContext(
        customer_id        = str(row.get("c", row.get("customer_id", "unknown"))),
        gender             = str(row.get("gender", "F")),
        intent             = intent,
        confidence         = float(confidence),
        priority           = str(priority),
        budget             = str(row.get("budget", "5–15 triệu")),
        style              = str(row.get("style", "Trẻ trung")),
        preferred_type     = str(row.get("preferred_type", "Nhẫn")),
        material           = str(row.get("material", "Vàng 18K")),
        recency_days       = int(row.get("recency_days", 60)),
        monetary           = float(row.get("monetary", 0)),
        avg_discount       = float(row.get("avg_discount", 0)),
        segment_rfm_tier   = str(row.get("segment_rfm_tier", "Gold-M")),
        birthday_in_days   = int(row.get("sig_birthday_in_days", 365)),
        sig_view_diamond   = int(row.get("sig_view_diamond", 0)),
        sig_view_ring      = int(row.get("sig_view_engagement_ring", row.get("sig_view_ring", 0))),
        sig_search_propose = int(row.get("sig_search_propose", 0)),
        camp_engagement    = int(row.get("camp_engagement", 0)),
        camp_anniversary   = int(row.get("camp_anniversary", 0)),
        camp_selfreward    = int(row.get("camp_selfreward", 0)),
        channel            = channel,
        product_focus      = product_focus,
        cta                = cta,
        # Instore context
        key_insight        = key_insight,
        urgency_signal     = urgency_signal,
        online_insight     = online_insight,
        instore_intent     = instore_intent,
        nba_strategy       = nba_strategy,
        psychology_trigger = psychology_trigger,
        product_rec_1      = product_rec_1,
        product_rec_2      = product_rec_2,
        product_rec_3      = product_rec_3,
    )
