"""
src/llm_message_generator.py
────────────────────────────────────────────────────────────────────────────
LLM Message Generator — OpenAI API Integration
Sinh nội dung marketing cá nhân hóa cho từng khách hàng trang sức

Cài đặt: pip install openai
Cần:     export OPENAI_API_KEY="sk-..."
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
from dataclasses import dataclass, asdict

# ── OpenAI import (graceful fallback nếu chưa cài) ───────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[LLM] WARNING: openai không được cài. Chạy: pip install openai")
    print("[LLM] Sẽ dùng fallback template mode.")


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class CustomerContext:
    """Ngữ cảnh khách hàng để inject vào prompt."""
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


@dataclass
class GeneratedMessage:
    """Kết quả sinh từ LLM."""
    customer_id:    str
    channel:        str
    subject:        Optional[str]    # chỉ dùng cho email
    body:           str
    cta_text:       str
    tone:           str              # warm / urgent / luxurious / friendly
    tokens_used:    int
    source:         str              # "llm" hoặc "fallback"
    raw_json:       dict


# ── Prompt templates theo channel ─────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là chuyên gia marketing trang sức cao cấp tại Việt Nam.
Nhiệm vụ: sinh nội dung marketing CÁ NHÂN HÓA, ngắn gọn, tự nhiên — KHÔNG sáo rỗng.

Quy tắc bắt buộc:
1. Viết tiếng Việt thuần túy, giọng thân thiện nhưng lịch sự
2. KHÔNG dùng các cụm sáo: "ưu đãi không thể bỏ lỡ", "đừng bỏ lỡ", "cơ hội vàng"
3. Độ dài: push ≤ 60 chữ, zns ≤ 100 chữ, email subject ≤ 60 chữ, email body ≤ 180 chữ, in_app ≤ 80 chữ
4. Luôn trả lời đúng định dạng JSON được yêu cầu — không thêm markdown hay giải thích ngoài JSON
5. Tone phải phù hợp: engagement=lãng mạn tinh tế, anniversary=ấm áp trân trọng, self_reward=tự tin vui tươi, gift=chu đáo quan tâm"""

CHANNEL_PROMPTS = {
    "push": """Sinh push notification cho ứng dụng mobile.
JSON format:
{{"body": "...", "cta_text": "...", "tone": "..."}}
- body: tối đa 60 chữ, tạo tò mò nhẹ, có thể dùng emoji 1 cái
- cta_text: 2-4 chữ (VD: "Xem ngay", "Khám phá")
- tone: một trong [warm, urgent, luxurious, friendly]""",

    "zns": """Sinh tin nhắn ZNS Zalo.
JSON format:
{{"body": "...", "cta_text": "...", "tone": "..."}}
- body: tối đa 100 chữ, thân mật như nhắn bạn bè, không dùng "Kính gửi"
- cta_text: 3-5 chữ
- tone: một trong [warm, urgent, luxurious, friendly]""",

    "email": """Sinh email marketing.
JSON format:
{{"subject": "...", "body": "...", "cta_text": "...", "tone": "..."}}
- subject: tối đa 60 chữ, gợi tò mò, không dùng CAPS LOCK
- body: tối đa 180 chữ, 2-3 đoạn ngắn, có thể có 1 bullet point nếu cần
- cta_text: 3-6 chữ (VD: "Xem bộ sưu tập", "Đặt tư vấn")
- tone: một trong [warm, urgent, luxurious, friendly]""",

    "in_app": """Sinh banner/pop-up trong ứng dụng.
JSON format:
{{"body": "...", "cta_text": "...", "tone": "..."}}
- body: tối đa 80 chữ, ngắn gọn súc tích vì đang trong lúc browse
- cta_text: 2-4 chữ
- tone: một trong [warm, urgent, luxurious, friendly]""",

    "store": """Sinh script gợi ý ngắn cho nhân viên cửa hàng khi khách quét loyalty card.
JSON format:
{{"body": "...", "cta_text": "...", "tone": "..."}}
- body: tối đa 80 chữ, viết như lời nhân viên nói với khách — tự nhiên, không đọc như quảng cáo
- cta_text: hành động nhân viên nên làm (VD: "Dẫn khách xem khu Nhẫn")
- tone: warm""",
}

INTENT_CONTEXT_PROMPTS = {
    "engagement": "Khách đang có dấu hiệu sắp cầu hôn (xem nhẫn đính hôn, tìm kiếm 'cầu hôn'). Tiếp cận tinh tế, lãng mạn — KHÔNG nói thẳng 'bạn sắp cầu hôn'.",
    "anniversary": "Khách có xu hướng mua dịp kỷ niệm tình yêu/ngày cưới. Nhấn vào cảm xúc trân trọng, kỷ niệm ý nghĩa.",
    "self_reward": "Khách thích tự thưởng cho bản thân. Nhấn vào sự tự tin, xứng đáng được hưởng điều tốt đẹp — không phụ thuộc vào ai.",
    "gift": "Khách đang tìm quà tặng cho người thân. Nhấn vào việc chọn đúng quà, ý nghĩa của món quà với người nhận.",
}


# ── Cache (tránh gọi API lặp) ─────────────────────────────────────────────────
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


# ── Fallback templates (khi không có API key) ─────────────────────────────────
FALLBACK_TEMPLATES = {
    ("engagement", "push"):    {"body": "✨ Bộ sưu tập nhẫn mới vừa về — tinh tế và đặc biệt như khoảnh khắc bạn đang hướng tới", "cta_text": "Xem ngay", "tone": "luxurious"},
    ("engagement", "email"):   {"subject": "Bộ sưu tập nhẫn đính hôn mới — dành cho khoảnh khắc đặc biệt", "body": "Chúng tôi hiểu rằng có những khoảnh khắc cần được đánh dấu bằng điều gì đó thật đặc biệt.\n\nBộ sưu tập nhẫn mới nhất của chúng tôi được thiết kế để kể câu chuyện của riêng bạn — tinh xảo, lâu bền và ý nghĩa.", "cta_text": "Khám phá bộ sưu tập", "tone": "luxurious"},
    ("engagement", "zns"):     {"body": "Có những khoảnh khắc chỉ đến một lần — và xứng đáng được chuẩn bị thật kỹ 💍 Ghé xem bộ nhẫn mới nhất nhé", "cta_text": "Xem ngay", "tone": "warm"},
    ("engagement", "in_app"):  {"body": "Bộ sưu tập nhẫn mới đang chờ bạn khám phá — mỗi chiếc đều có câu chuyện riêng", "cta_text": "Xem bộ sưu tập", "tone": "luxurious"},
    ("anniversary", "push"):   {"body": "💝 Kỷ niệm của bạn sắp đến — tặng người ấy điều gì đó thật ý nghĩa năm nay", "cta_text": "Chọn quà ngay", "tone": "warm"},
    ("anniversary", "email"):  {"subject": "Kỷ niệm đặc biệt xứng đáng được ghi nhớ mãi mãi", "body": "Mỗi năm trôi qua là thêm một trang đẹp trong câu chuyện của hai bạn.\n\nChúng tôi có bộ quà kỷ niệm được thiết kế để nói lên điều bạn muốn nói — từ khắc tên đến hộp quà đặc biệt, tất cả đều có thể cá nhân hóa.", "cta_text": "Xem bộ quà kỷ niệm", "tone": "warm"},
    ("anniversary", "zns"):    {"body": "Năm nay kỷ niệm của bạn sẽ khác — một món quà được chuẩn bị thật tâm ý, khắc tên miễn phí 💕", "cta_text": "Tư vấn chọn quà", "tone": "warm"},
    ("anniversary", "in_app"): {"body": "Sắp đến ngày đặc biệt? Để chúng tôi giúp bạn chọn món quà hoàn hảo", "cta_text": "Tư vấn ngay", "tone": "warm"},
    ("self_reward", "push"):   {"body": "🌟 Bạn đã làm việc chăm chỉ — hôm nay xứng đáng có một điều gì đó thật đẹp cho mình", "cta_text": "Tự thưởng nào", "tone": "friendly"},
    ("self_reward", "email"):  {"subject": "Tự thưởng cho bản thân — vì bạn xứng đáng", "body": "Đôi khi điều tốt nhất bạn có thể làm là tự chăm sóc bản thân.\n\nBộ sưu tập Self-Reward của chúng tôi được thiết kế để bạn diện mỗi ngày — không cần dịp đặc biệt, chỉ cần bạn thích.", "cta_text": "Xem bộ sưu tập", "tone": "friendly"},
    ("self_reward", "zns"):    {"body": "Mình ơi — bao lâu rồi chưa tự thưởng gì cho bản thân chưa? 😊 Có vài món mới về, xinh lắm", "cta_text": "Xem thử đi", "tone": "friendly"},
    ("self_reward", "in_app"): {"body": "New arrivals vừa về — mấy món này mặc hàng ngày là hợp lý lắm luôn", "cta_text": "Xem ngay", "tone": "friendly"},
    ("gift", "push"):          {"body": "🎁 Sắp đến ngày quan trọng? Chúng tôi có gợi ý quà tặng phù hợp cho mọi đối tượng", "cta_text": "Xem gợi ý", "tone": "friendly"},
    ("gift", "email"):         {"subject": "Gợi ý quà tặng — chọn đúng ngay lần đầu", "body": "Chọn quà cho người thân không bao giờ dễ — nhưng chúng tôi có thể giúp.\n\nTừ phong cách, ngân sách đến sở thích, đội tư vấn sẽ giúp bạn tìm đúng món quà mà người nhận sẽ trân trọng mãi.", "cta_text": "Được tư vấn miễn phí", "tone": "friendly"},
    ("gift", "zns"):           {"body": "Tặng quà mà người nhận thật sự thích — không phải chỉ để cho có 🎁 Cho chúng tôi giúp bạn chọn nhé", "cta_text": "Tư vấn chọn quà", "tone": "friendly"},
    ("gift", "in_app"):        {"body": "Không biết tặng gì? Gift Finder của chúng tôi sẽ giúp bạn tìm đúng món trong 2 phút", "cta_text": "Thử Gift Finder", "tone": "friendly"},
}


# ── Main generator class ───────────────────────────────────────────────────────
class LLMMessageGenerator:
    """
    Sinh marketing message cá nhân hóa bằng OpenAI API.
    Tự động fallback về template nếu không có API key.
    """

    MODEL = "gpt-4o"

    def __init__(self,
                 api_key: Optional[str] = None,
                 use_cache: bool = True,
                 rate_limit_delay: float = 0.3):
        """
        Args:
            api_key: OpenAI API key. Mặc định đọc từ OPENAI_API_KEY env var.
            use_cache: Cache responses để tiết kiệm API calls.
            rate_limit_delay: Giây chờ giữa các API calls.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.use_cache = use_cache
        self.rate_limit_delay = rate_limit_delay
        self.cache = SimpleCache() if use_cache else None

        # Khởi tạo OpenAI client
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.mode = "llm"
            print(f"[LLM] Mode: OpenAI API ({self.MODEL})")
        else:
            self.client = None
            self.mode = "fallback"
            reason = "openai không được cài" if not OPENAI_AVAILABLE else "không có API key"
            print(f"[LLM] Mode: Fallback template ({reason})")

        # Stats
        self.stats = {"llm_calls": 0, "cache_hits": 0, "fallback_used": 0, "total_tokens": 0}

    def _build_user_prompt(self, ctx: CustomerContext) -> str:
        """Xây dựng user prompt từ customer context."""
        intent_ctx = INTENT_CONTEXT_PROMPTS.get(ctx.intent, "")
        channel_inst = CHANNEL_PROMPTS.get(ctx.channel, CHANNEL_PROMPTS["push"])

        # Thông tin khách hàng
        customer_info = f"""THÔNG TIN KHÁCH HÀNG:
- Phân khúc: {ctx.segment_rfm_tier} | Ngân sách: {ctx.budget}
- Phong cách: {ctx.style} | Loại ưa thích: {ctx.preferred_type} ({ctx.material})
- Lần mua gần nhất: {ctx.recency_days} ngày trước | Tổng chi: {ctx.monetary:,.0f}đ
- Hay dùng discount: {"Có" if ctx.avg_discount > 0.05 else "Không"} ({ctx.avg_discount*100:.0f}%)
- Ngày sinh nhật còn: {ctx.birthday_in_days} ngày
- Tín hiệu cầu hôn: {"Có" if ctx.sig_view_ring or ctx.sig_search_propose else "Không"}

INTENT: {ctx.intent.upper()} (confidence: {ctx.confidence:.0%})
{intent_ctx}

SẢN PHẨM TẬP TRUNG: {ctx.product_focus}
KÊNH GIAO TIẾP: {ctx.channel.upper()}

{channel_inst}"""
        return customer_info

    def _call_api(self, user_prompt: str) -> tuple[dict, int]:
        """Gọi OpenAI API và parse JSON response."""
        # Check cache
        if self.use_cache and self.cache:
            cached = self.cache.get(user_prompt)
            if cached:
                self.stats["cache_hits"] += 1
                return cached, 0

        # Rate limiting
        time.sleep(self.rate_limit_delay)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_text = response.choices[0].message.content.strip()
        tokens = response.usage.prompt_tokens + response.usage.completion_tokens

        # Parse JSON — strip markdown fences nếu có
        clean = raw_text
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        clean = clean.strip()

        parsed = json.loads(clean)

        # Cache result
        if self.use_cache and self.cache:
            self.cache.set(user_prompt, parsed)

        self.stats["llm_calls"] += 1
        self.stats["total_tokens"] += tokens
        return parsed, tokens

    def _fallback(self, ctx: CustomerContext) -> tuple[dict, int]:
        """Template fallback khi không có LLM."""
        key = (ctx.intent, ctx.channel)
        template = FALLBACK_TEMPLATES.get(key)

        if template is None:
            # Generic fallback
            template = {
                "body": f"Khám phá {ctx.product_focus} mới nhất — {ctx.cta}",
                "cta_text": ctx.cta or "Xem ngay",
                "tone": "friendly",
            }

        self.stats["fallback_used"] += 1
        return template.copy(), 0

    def generate(self, ctx: CustomerContext) -> GeneratedMessage:
        """Sinh message cho một khách hàng."""
        tokens = 0

        try:
            if self.mode == "llm":
                prompt = self._build_user_prompt(ctx)
                raw_json, tokens = self._call_api(prompt)
                source = "llm"
            else:
                raw_json, tokens = self._fallback(ctx)
                source = "fallback"
        except (json.JSONDecodeError, Exception) as e:
            print(f"[LLM] Error cho {ctx.customer_id}: {e}. Dùng fallback.")
            raw_json, tokens = self._fallback(ctx)
            source = "fallback_error"

        self.stats["total_tokens"] += tokens

        return GeneratedMessage(
            customer_id=ctx.customer_id,
            channel=ctx.channel,
            subject=raw_json.get("subject"),
            body=raw_json.get("body", ""),
            cta_text=raw_json.get("cta_text", ctx.cta),
            tone=raw_json.get("tone", "friendly"),
            tokens_used=tokens,
            source=source,
            raw_json=raw_json,
        )

    def generate_batch(self, contexts: list[CustomerContext],
                        verbose: bool = True) -> list[GeneratedMessage]:
        """Sinh message cho nhiều khách, có progress tracking."""
        results = []
        total = len(contexts)

        for i, ctx in enumerate(contexts, 1):
            if verbose:
                print(f"  [{i:3d}/{total}] {ctx.customer_id} | {ctx.intent:12s} | {ctx.channel:8s}", end=" ")

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
        if self.stats['llm_calls'] > 0:
            est_cost = self.stats['total_tokens'] / 1_000_000 * 5.0  # ~$5/1M tokens GPT-4o (blended)
            print(f"  Est. cost    : ~${est_cost:.4f} USD")
        print("  ──────────────────────────────────────────")


# ── Helper: profile row → CustomerContext ─────────────────────────────────────
def row_to_context(row: pd.Series,
                   intent: str,
                   confidence: float,
                   priority: str,
                   channel: str,
                   product_focus: str,
                   cta: str) -> CustomerContext:
    """Chuyển đổi một dòng profile DataFrame thành CustomerContext."""
    return CustomerContext(
        customer_id        = str(row.get("c", row.get("customer_id", "unknown"))),
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
        sig_view_ring      = int(row.get("sig_view_engagement_ring", 0)),
        sig_search_propose = int(row.get("sig_search_propose", 0)),
        camp_engagement    = int(row.get("camp_engagement", 0)),
        camp_anniversary   = int(row.get("camp_anniversary", 0)),
        camp_selfreward    = int(row.get("camp_selfreward", 0)),
        channel            = channel,
        product_focus      = product_focus,
        cta                = cta,
    )


# ── Standalone demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LLM MESSAGE GENERATOR — DEMO")
    print("=" * 60)

    # Demo contexts
    demo_contexts = [
        CustomerContext(
            customer_id="KH001", intent="engagement", confidence=0.81, priority="high",
            budget="5–15 triệu", style="Trẻ trung", preferred_type="Nhẫn",
            material="Vàng 18K", recency_days=118, monetary=40768070, avg_discount=0.05,
            segment_rfm_tier="Gold-H", birthday_in_days=35, sig_view_diamond=0,
            sig_view_ring=3, sig_search_propose=1, camp_engagement=6,
            camp_anniversary=2, camp_selfreward=0,
            channel="push", product_focus="Bộ sưu tập Engagement 2025",
            cta="Xem nhẫn đính hôn",
        ),
        CustomerContext(
            customer_id="KH002", intent="self_reward", confidence=0.73, priority="high",
            budget="5–15 triệu", style="Trẻ trung", preferred_type="Bông tai",
            material="Vàng 18K", recency_days=48, monetary=24861526, avg_discount=0.083,
            segment_rfm_tier="Platinum-L", birthday_in_days=39, sig_view_diamond=0,
            sig_view_ring=4, sig_search_propose=1, camp_engagement=0,
            camp_anniversary=3, camp_selfreward=5,
            channel="zns", product_focus="New Arrivals — Bông tai",
            cta="Xem bộ sưu tập",
        ),
        CustomerContext(
            customer_id="KH005", intent="anniversary", confidence=0.80, priority="high",
            budget="5–15 triệu", style="Thanh lịch", preferred_type="Bông tai",
            material="Vàng 14K", recency_days=29, monetary=33781078, avg_discount=0,
            segment_rfm_tier="Silver-L", birthday_in_days=56, sig_view_diamond=0,
            sig_view_ring=3, sig_search_propose=0, camp_engagement=1,
            camp_anniversary=3, camp_selfreward=0,
            channel="email", product_focus="Bộ quà kỷ niệm",
            cta="Xem gợi ý quà",
        ),
        CustomerContext(
            customer_id="KH009", intent="gift", confidence=0.64, priority="medium",
            budget="15–30 triệu", style="Trẻ trung", preferred_type="Lắc tay",
            material="Kim cương", recency_days=37, monetary=92862324, avg_discount=0.0375,
            segment_rfm_tier="Silver-L", birthday_in_days=56, sig_view_diamond=0,
            sig_view_ring=1, sig_search_propose=1, camp_engagement=1,
            camp_anniversary=2, camp_selfreward=0,
            channel="in_app", product_focus="Gift Guide — Lắc tay Kim cương",
            cta="Tư vấn chọn quà",
        ),
    ]

    gen = LLMMessageGenerator()

    print("\nGenerating messages...\n")
    messages = gen.generate_batch(demo_contexts, verbose=True)

    print("\n" + "=" * 60)
    print("GENERATED MESSAGES — FULL OUTPUT")
    print("=" * 60)

    for msg in messages:
        print(f"\n{'─'*50}")
        print(f"Customer : {msg.customer_id} | Channel: {msg.channel.upper()} | Source: {msg.source}")
        print(f"Tone     : {msg.tone}")
        if msg.subject:
            print(f"Subject  : {msg.subject}")
        print(f"Body     : {msg.body}")
        print(f"CTA      : [{msg.cta_text}]")
        print(f"Tokens   : {msg.tokens_used}")

    gen.print_stats()
