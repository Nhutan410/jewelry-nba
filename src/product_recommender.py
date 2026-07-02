"""Deterministic product recommendation over the production PNJ catalog.

The recommender avoids LLM ranking. It normalizes the production catalog fields
first, applies hard filters that a TVV would expect, then scores each product on
the same visible 100-point rubric used by the UI.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT / "data" / "catalog_production_enriched.json"
LEGACY_METADATA_PATH = ROOT / "data" / "metadata.json"

# Kept for backward-compatible imports in scripts/app code.
DEFAULT_METADATA_PATH = DEFAULT_CATALOG_PATH

SCORE_WEIGHTS = {
    "budget": 25,
    "category": 20,
    "occasion": 15,
    "material_stone": 15,
    "recipient_profile": 8,
    "style": 6,
    "segment_value": 7,
    "popularity": 4,
}

SCORING_RULES = [
    "Ngân sách (25): giá không vượt trần ngân sách; trong khung điểm cao nhất, thấp hơn khung vẫn được điểm tiết kiệm.",
    "Loại SP (20): khớp product_type chuẩn của catalog; nhóm gần nhau như bracelet/bangle được điểm một phần.",
    "Dịp mua (15): khớp occasion_tags hoặc tín hiệu LEP/purpose như cầu hôn, sinh nhật, quà tặng, kỷ niệm.",
    "Chất liệu/đá (15): 8 điểm chất liệu/tuổi vàng, 7 điểm đá chính/phụ; dùng field chuẩn trước, text chỉ là fallback.",
    "Tệp người thụ hưởng (8): chấm theo người thụ hưởng thật sự; gồm giới tính sản phẩm (tối đa 5đ) và nhóm tuổi adult/child (tối đa 3đ); suy từ age < 18 = child.",
    "Phân khúc (7): RFM/monetary/budget quyết định entry/mid/premium/luxury có phù hợp hay không; max 7đ khi khớp hoàn toàn.",
    "Style (6): map style khách sang tag chuẩn như minimal, youthful, elegant, luxury, bold.",
    "Bán chạy (4): ưu tiên sold/rating thực tế nếu có (có thể đạt 4đ); catalog chưa có số bán thì dùng proxy tag daily/classic/giftable — proxy tối đa 2.5đ để phân biệt với sản phẩm có real data.",
]

PRICE_BUCKETS = {
    "Dưới 5 triệu": (0, 5_000_000),
    "< 5 triệu": (0, 5_000_000),
    "<= 5 triệu": (0, 5_000_000),
    "5-15 triệu": (5_000_000, 15_000_000),
    "5–15 triệu": (5_000_000, 15_000_000),
    "15-30 triệu": (15_000_000, 30_000_000),
    "15–30 triệu": (15_000_000, 30_000_000),
    "Trên 30 triệu": (30_000_000, None),
    "30 triệu trở lên": (30_000_000, None),
}

PRODUCT_TYPE_LABELS = {
    "ring": "Nhẫn",
    "bracelet": "Vòng tay / Lắc",
    "bangle": "Vòng tay",
    "necklace": "Dây chuyền",
    "earrings": "Bông tai",
    "pendant": "Mặt dây",
    "charm": "Charm",
    "anklet": "Lắc chân",
}

PRODUCT_TYPE_ALIASES = {
    "ring": ["nhan", "nhan cuoi", "nhan dinh hon", "ring", "wedding ring", "engagement ring"],
    "bracelet": ["vong tay", "lac tay", "lac", "bracelet", "wrist"],
    "bangle": ["kieng", "vong cung", "bangle"],
    "necklace": ["day chuyen", "day co", "vong co", "necklace"],
    "earrings": ["bong tai", "hoa tai", "khuyen tai", "earring", "earrings"],
    "pendant": ["mat day", "mat day chuyen", "pendant"],
    "charm": ["charm", "hat charm"],
    "anklet": ["lac chan", "anklet"],
}

ADJACENT_PRODUCT_TYPES = {
    "bracelet": {"bangle", "anklet", "charm"},
    "bangle": {"bracelet"},
    "necklace": {"pendant", "charm"},
    "pendant": {"necklace", "charm"},
    "charm": {"bracelet", "necklace", "pendant"},
}

MATERIAL_LABELS = {
    "silver": "Bạc",
    "white_gold": "Vàng trắng",
    "yellow_gold": "Vàng vàng",
    "italian_gold": "Vàng Ý",
    "alloy": "Hợp kim",
}

STONE_LABELS = {
    "diamond": "Kim cương",
    "ecz": "ECZ / Xoàn mỹ",
    "cz": "CZ",
    "synthetic": "Đá tổng hợp",
    "pearl": "Ngọc trai",
    "sapphire": "Sapphire",
    "ruby": "Ruby",
    "topaz": "Topaz",
    "citrine": "Citrine",
    "amethyst": "Amethyst",
    "jadeite": "Cẩm thạch",
    "none": "Không gắn đá",
}

# Vietnamese-normalized aliases cho mỗi canonical stone key — dùng cho text fallback search
_STONE_TEXT_ALIASES: dict[str, set[str]] = {
    "diamond":   {"diamond", "kim cuong"},
    "ecz":       {"ecz", "xoan my"},
    "cz":        {"cz", "cubic zirconia"},
    "synthetic": {"synthetic", "da tong hop", "tong hop"},
    "pearl":     {"pearl", "ngoc trai"},
    "sapphire":  {"sapphire"},
    "ruby":      {"ruby"},
    "topaz":     {"topaz"},
    "citrine":   {"citrine"},
    "amethyst":  {"amethyst"},
    "jadeite":   {"jadeite", "cam thach"},
}

OCCASION_DISPLAY = {
    "engagement": "cầu hôn/đính hôn",
    "wedding": "cưới",
    "anniversary": "kỷ niệm",
    "birthday": "sinh nhật",
    "gift": "quà tặng",
    "self_reward": "tự thưởng",
    "daily": "đeo hằng ngày",
    "office": "công sở",
    "party": "dự tiệc",
    "special_occasion": "dịp đặc biệt",
    "date": "hẹn hò",
}

STYLE_DISPLAY = {
    "minimal": "tối giản",
    "clean": "tinh gọn",
    "modern": "hiện đại",
    "classic": "classic",
    "daily": "đeo hằng ngày",
    "elegant": "thanh lịch",
    "youthful": "trẻ trung",
    "playful": "năng động",
    "luxury": "sang trọng",
    "bold": "cá tính/nổi bật",
    "formal": "trang trọng",
    "sparkling": "lấp lánh",
    "romantic": "lãng mạn",
    "giftable": "dễ làm quà",
    "warm": "tông ấm",
    "soft": "mềm mại",
}


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def money_vnd(value: Any) -> str:
    try:
        price = int(float(value or 0))
    except (TypeError, ValueError):
        price = 0
    return f"{price:,.0f}đ".replace(",", ".") if price > 0 else "Chưa có giá"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_sold_text(value: Any) -> int:
    text = str(value or "").strip().lower().replace(",", "")
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 0


def parse_rating(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def budget_bounds(label: Any) -> tuple[float | None, float | None]:
    text = str(label or "").strip()
    if text in PRICE_BUCKETS:
        return PRICE_BUCKETS[text]
    norm = normalize_text(text)
    if not norm or norm == "chua ro":
        return None, None
    if any(mark in norm for mark in ["< 5", "<= 5", "duoi 5"]):
        return 0, 5_000_000
    if any(mark in norm for mark in ["> 30", "tren 30", "30 trieu tro len"]):
        return 30_000_000, None
    match = re.search(r"(\d+)\s*-\s*(\d+)", norm)
    if match:
        return float(match.group(1)) * 1_000_000, float(match.group(2)) * 1_000_000
    for key, bounds in PRICE_BUCKETS.items():
        if normalize_text(key) == norm:
            return bounds
    return None, None


def canonical_gender(value: Any) -> str:
    norm = normalize_text(value)
    if "unisex" in norm:
        return "unisex"
    if norm in {"m", "male", "nam", "men", "man", "boy", "con trai", "be trai"}:
        return "male"
    if norm in {"f", "female", "nu", "women", "woman", "girl", "con gai", "be gai"}:
        return "female"
    if norm in {"unisex", "both", "ca nam va nu", "nam nu"}:
        return "unisex"
    return ""


def gender_label(value: Any) -> str:
    gender = canonical_gender(value)
    return {"male": "Nam", "female": "Nữ", "unisex": "Unisex"}.get(gender, str(value or ""))


def canonical_audience(value: Any) -> str:
    norm = normalize_text(value)
    if any(token in norm for token in ["child", "kid", "tre em", "em be", "be trai", "be gai", "con trai", "con gai"]):
        return "child"
    if any(token in norm for token in ["adult", "nguoi lon", "nguoi yeu", "vo", "chong", "ban doi"]):
        return "adult"
    return ""


def audience_label(value: Any) -> str:
    audience = canonical_audience(value)
    return {"adult": "Người lớn", "child": "Trẻ em"}.get(audience, str(value or ""))


def canonical_product_type(value: Any) -> str:
    norm = normalize_text(value)
    if not norm or norm == "chua ro":
        return ""
    if norm in PRODUCT_TYPE_LABELS:
        return norm
    if "bo trang suc" in norm or "set" == norm:
        return ""
    for canonical, aliases in PRODUCT_TYPE_ALIASES.items():
        if any(alias in norm for alias in aliases):
            return canonical
    return ""


def product_type_label(value: Any) -> str:
    canonical = canonical_product_type(value) or str(value or "")
    return PRODUCT_TYPE_LABELS.get(canonical, str(value or ""))


def canonical_material(value: Any, text: str = "") -> str:
    norm = normalize_text(value)
    haystack = f"{norm} {normalize_text(text)}"
    if "silver" in norm or "pnjsilver" in haystack or re.search(r"\bbac\b", haystack):
        return "silver"
    if "white_gold" in norm or "vang trang" in haystack:
        return "white_gold"
    if "italian_gold" in norm or "vang y" in haystack:
        return "italian_gold"
    if "yellow_gold" in norm or "vang vang" in haystack or ("vang" in haystack and "trang" not in haystack):
        return "yellow_gold"
    if "alloy" in norm or "hop kim" in haystack:
        return "alloy"
    return ""


def material_label(value: Any) -> str:
    canonical = canonical_material(value)
    return MATERIAL_LABELS.get(canonical, str(value or ""))


def purity_from_text(*values: Any) -> str:
    text = normalize_text(" ".join(str(v or "") for v in values))
    if "24k" in text or "9999" in text:
        return "24k"
    if "18k" in text or "7500" in text or "75%" in text:
        return "18k"
    if "14k" in text or "5850" in text or "58,5" in text or "58.5" in text:
        return "14k"
    if "10k" in text or "4160" in text or "41,6" in text or "41.6" in text:
        return "10k"
    return ""


# Patterns cho tên sản phẩm PNJ: "Vàng 75% (18K)", "Vàng trắng 58,5% (14K)"
_NAME_MATERIAL_PATTERNS = [
    (re.compile(r"\bvang\s+trang\b"), "white_gold"),
    (re.compile(r"\bvang\s+y\b"), "italian_gold"),
    (re.compile(r"\bvang\b"), "yellow_gold"),
    (re.compile(r"\bbac\b|pnjsilver|\bsilver\b"), "silver"),
    (re.compile(r"\bhop\s+kim\b|\balloy\b"), "alloy"),
]
_NAME_PCT_PURITY = [
    (99.0, float("inf"), "24k"),
    (74.0, 76.0, "18k"),
    (57.5, 59.5, "14k"),
    (40.5, 42.5, "10k"),
]


def parse_name_material_purity(name: str) -> tuple[str, str]:
    """Extract (canonical_material, purity) từ tên sản phẩm PNJ.

    PNJ naming convention: "Loại SP Chất liệu XX% (NK) Brand Series SKU"
    Ví dụ: "Bông tai cưới Vàng 75% (18K) PNJ Trầu Cau"
           "Bông tai nam Kim cương Vàng trắng 58,5% (14K) MANCODE"
    """
    text = normalize_text(name)
    material = ""
    for pattern, mat in _NAME_MATERIAL_PATTERNS:
        if pattern.search(text):
            material = mat
            break

    # Ưu tiên parse % trực tiếp (chính xác hơn karat notation)
    purity = ""
    pct_match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
    if pct_match:
        try:
            pct = float(pct_match.group(1).replace(",", "."))
            for lo, hi, pur in _NAME_PCT_PURITY:
                if lo <= pct <= hi:
                    purity = pur
                    break
        except ValueError:
            pass
    if not purity:
        purity = purity_from_text(name)
    return material, purity


def canonical_stone(value: Any) -> str:
    norm = normalize_text(value)
    if not norm or norm in {"none", "null", "nan", "khong", "khong gan da"}:
        return "none" if norm else ""
    if "diamond" in norm or "kim cuong" in norm:
        return "diamond"
    if "ecz" in norm or "xoan my" in norm:
        return "ecz"
    if norm == "cz" or "cubic zirconia" in norm:
        return "cz"
    if "synthetic" in norm or "tong hop" in norm:
        return "synthetic"
    if "pearl" in norm or "ngoc trai" in norm:
        return "pearl"
    for key in STONE_LABELS:
        if key in norm:
            return key
    return norm


def stone_label(value: Any) -> str:
    stone = canonical_stone(value)
    return STONE_LABELS.get(stone, str(value or ""))


def _stones_from_text(text: str) -> list[str]:
    """Trích xuất danh sách đá mong muốn từ chuỗi text tự do (material preference / purpose)."""
    norm = normalize_text(text)
    stones: list[str] = []
    if "kim cuong" in norm or "diamond" in norm:
        stones.append("diamond")
    if "ngoc trai" in norm or "pearl" in norm:
        stones.append("pearl")
    if "ecz" in norm or "xoan my" in norm:
        stones.append("ecz")
    if norm == "cz" or "cubic zirconia" in norm:
        stones.append("cz")
    if "ruby" in norm:
        stones.append("ruby")
    if "sapphire" in norm:
        stones.append("sapphire")
    if "topaz" in norm:
        stones.append("topaz")
    if "citrine" in norm:
        stones.append("citrine")
    if "amethyst" in norm:
        stones.append("amethyst")
    if "cam thach" in norm or "jadeite" in norm:
        stones.append("jadeite")
    return stones


def _collect_product_text(product: dict[str, Any]) -> str:
    fields: list[str] = [
        product.get("name", ""),
        product.get("sku", ""),
        product.get("brand", ""),
        product.get("product_line", ""),
        product.get("short_description", ""),
        product.get("description", ""),
        product.get("detail_description", ""),
        product.get("search_text", ""),
        product.get("product_type", ""),
        product.get("material", ""),
        product.get("primary_stone", ""),
        product.get("secondary_stone", ""),
        " ".join(str(c) for c in product.get("categories") or []),
    ]
    features = product.get("features") or {}
    if isinstance(features, dict):
        fields.extend(str(k) for k in features.keys())
        fields.extend(str(v) for v in features.values())
    return normalize_text(" ".join(fields))


def _product_category(product: dict[str, Any], text: str = "") -> str:
    product_type = canonical_product_type(product.get("product_type"))
    if product_type:
        return product_type
    categories = product.get("categories") or []
    if isinstance(categories, list):
        for category in categories:
            product_type = canonical_product_type(category)
            if product_type:
                return product_type
    return canonical_product_type(text)


def _style_terms(value: Any) -> set[str]:
    norm = normalize_text(value)
    terms: set[str] = set()
    if not norm or norm == "chua ro":
        return terms
    if any(k in norm for k in ["toi gian", "don gian", "minimal", "nhe nhang"]):
        terms.update(["minimal", "clean", "daily", "soft"])
    if any(k in norm for k in ["tre trung", "nang dong", "youth", "playful"]):
        terms.update(["youthful", "playful", "modern"])
    if any(k in norm for k in ["thanh lich", "cong so", "elegant", "office"]):
        terms.update(["elegant", "classic", "office", "formal", "clean"])
    if any(k in norm for k in ["sang trong", "cao cap", "luxury", "premium"]):
        terms.update(["luxury", "elegant", "formal", "sparkling"])
    if any(k in norm for k in ["ca tinh", "noi bat", "doc dao", "bold"]):
        terms.update(["bold", "modern", "sparkling"])
    if any(k in norm for k in ["lang man", "romantic"]):
        terms.add("romantic")
    if not terms:
        terms.add(norm)
    return terms


def _occasion_terms_from_text(value: Any) -> set[str]:
    norm = normalize_text(value)
    terms: set[str] = set()
    if not norm or norm in {"chua ro", "khong co dip cu the", "chua hoi duoc"}:
        return terms
    if any(k in norm for k in ["cau hon", "dinh hon", "engagement", "propose"]):
        terms.add("engagement")
    if any(k in norm for k in ["cuoi", "hon nhan", "wedding", "nhan cuoi"]):
        terms.add("wedding")
    if any(k in norm for k in ["ky niem", "anniversary", "date"]):
        terms.update(["anniversary", "date"])
    if any(k in norm for k in ["sinh nhat", "birthday"]):
        terms.add("birthday")
    if any(k in norm for k in ["qua tang", "tang", "gift"]):
        terms.add("gift")
    if any(k in norm for k in ["ban than", "tu thuong", "self"]):
        terms.add("self_reward")
    if any(k in norm for k in ["hang ngay", "daily"]):
        terms.add("daily")
    if any(k in norm for k in ["du tiec", "party"]):
        terms.add("party")
    return terms


def _normalize_catalog_occasion_tags(product: dict[str, Any], text: str) -> list[str]:
    terms = set()
    for tag in product.get("occasion_tags") or []:
        norm = normalize_text(tag)
        if norm in {"general"}:
            terms.add("daily")
        elif norm:
            terms.add(norm)
    terms.update(_occasion_terms_from_text(text))
    if "special_occasion" in terms:
        terms.update(["anniversary", "party"])
    return sorted(terms)


def _normalize_catalog_style_tags(product: dict[str, Any], text: str) -> list[str]:
    terms = {normalize_text(tag) for tag in product.get("style_tags") or [] if normalize_text(tag)}
    if any(k in text for k in ["toi gian", "don gian", "minimal"]):
        terms.add("minimal")
    if any(k in text for k in ["tre trung", "nang dong"]):
        terms.update(["youthful", "playful"])
    if any(k in text for k in ["thanh lich", "tinh te"]):
        terms.update(["elegant", "clean"])
    if any(k in text for k in ["sang trong", "cao cap", "kim cuong"]):
        terms.update(["luxury", "sparkling"])
    if any(k in text for k in ["ca tinh", "doc dao", "noi bat", "pha cach"]):
        terms.add("bold")
    return sorted(terms)


def product_price(product: dict[str, Any]) -> float:
    return safe_float(product.get("net_price") or product.get("price"))


def price_range_label(price: Any) -> str:
    value = safe_float(price)
    if value <= 0:
        return "Chưa rõ"
    if value < 5_000_000:
        return "Dưới 5 triệu"
    if value <= 15_000_000:
        return "5-15 triệu"
    if value <= 30_000_000:
        return "15-30 triệu"
    return "Trên 30 triệu"


def popularity_score(product: dict[str, Any]) -> float:
    sold = parse_sold_text(product.get("sold_text") or product.get("sold"))
    rating = parse_rating(product.get("rating") or product.get("review_rating"))
    if sold or rating:
        # Có real data: sold/rating → có thể đạt tối đa ~10 trước khi bị cap ở SCORE_WEIGHTS
        sold_score = min(7.0, math.log10(max(sold, 1)) * 2.1) if sold else 0
        rating_score = min(3.0, rating / 5 * 3) if rating else 0
        return round(sold_score + rating_score, 2)

    # Proxy khi chưa có sold/rating — cố tình giới hạn max 2.5 để phân biệt rõ với real data
    tags = {normalize_text(tag) for tag in product.get("style_tags") or []}
    occasion_tags = {normalize_text(tag) for tag in product.get("occasion_tags") or []}
    score = 0.0
    if "daily" in tags or "daily" in occasion_tags:
        score += 0.9
    if "classic" in tags or "clean" in tags:
        score += 0.6
    if "giftable" in tags or "gift" in occasion_tags:
        score += 0.5
    if normalize_text(product.get("brand")) in {"pnj", "pnjsilver", "style by pnj"}:
        score += 0.3
    if normalize_text(product.get("price_tier")) in {"entry", "mid"}:
        score += 0.2
    return round(min(2.5, score), 2)


def infer_product_tags(product: dict[str, Any]) -> dict[str, Any]:
    text = _collect_product_text(product)
    product_type = _product_category(product, text)
    # Ưu tiên parse trực tiếp từ tên sản phẩm (pattern PNJ có cấu trúc rõ)
    name_material, name_purity = parse_name_material_purity(product.get("name", ""))
    material = name_material or canonical_material(product.get("material"), text)
    purity = name_purity or purity_from_text(product.get("name"), product.get("product_line"), text)
    primary_stone = canonical_stone(product.get("primary_stone") or product.get("main_stone"))
    secondary_stone = canonical_stone(product.get("secondary_stone"))

    if not primary_stone:
        if "kim cuong" in text or "diamond" in text:
            primary_stone = "diamond"
        elif "xoan my" in text or "ecz" in text:
            primary_stone = "ecz"
        elif "cz" in text:
            primary_stone = "cz"
        elif "ngoc trai" in text or "pearl" in text:
            primary_stone = "pearl"

    price = product_price(product)
    price_tier = normalize_text(product.get("price_tier"))
    if not price_tier:
        if price >= 30_000_000:
            price_tier = "luxury"
        elif price >= 15_000_000:
            price_tier = "premium"
        elif price >= 5_000_000:
            price_tier = "mid"
        else:
            price_tier = "entry"

    selling_points = []
    material_text = MATERIAL_LABELS.get(material, "")
    if material_text:
        selling_points.append(material_text + (f" {purity.upper()}" if purity else ""))
    if primary_stone and primary_stone != "none":
        selling_points.append(stone_label(primary_stone))
    style_tags = _normalize_catalog_style_tags(product, text)
    occasion_tags = _normalize_catalog_occasion_tags(product, text)
    if occasion_tags:
        selling_points.append("Phù hợp " + ", ".join(OCCASION_DISPLAY.get(o, o) for o in occasion_tags[:2]))
    if popularity_score(product) >= 4:
        selling_points.append("Có tín hiệu phổ biến trong catalog")

    return {
        "product_type": product_type,
        "product_type_label": PRODUCT_TYPE_LABELS.get(product_type, ""),
        "category": PRODUCT_TYPE_LABELS.get(product_type, ""),
        "categories": [PRODUCT_TYPE_LABELS.get(product_type, product_type)] if product_type else [],
        "price_range": price_range_label(price),
        "gender": canonical_gender(product.get("gender")),
        "audience": canonical_audience(product.get("audience")),
        "material": material,
        "material_label": material_text,
        "purity": purity,
        "primary_stone": primary_stone,
        "secondary_stone": secondary_stone,
        "style_tags": style_tags,
        "occasion_tags": occasion_tags,
        "brand_line": str(product.get("brand") or "").strip(),
        "price_tier": price_tier,
        "popularity_score": popularity_score(product),
        "selling_points": selling_points[:5],
    }


def merge_product_tags(product: dict[str, Any]) -> dict[str, Any]:
    base = dict(product)
    inferred = infer_product_tags(product)
    tags = base.get("recommendation_tags")
    base["recommendation_tags"] = {**inferred, **tags} if isinstance(tags, dict) else inferred
    return base


def _score_budget(price: float, min_budget: float | None, max_budget: float | None) -> tuple[float, str]:
    if min_budget is None and max_budget is None:
        return 10, "Chưa có ngân sách rõ, giữ điểm trung lập cho giá"
    if max_budget is not None and price > max_budget:
        return 0, "Vượt ngân sách"
    if min_budget is not None and price < min_budget:
        if min_budget <= 0:
            return 22, "Giá nằm trong ngân sách thấp"
        ratio = max(0.0, min(1.0, price / min_budget))
        return round(15 + 6 * ratio, 2), "Giá thấp hơn khung ngân sách, phù hợp nếu khách muốn tiết kiệm"
    if max_budget is None:
        return 24, "Giá đạt ngưỡng khách cao cấp"
    if min_budget in (None, 0):
        ratio = max(0.0, min(1.0, price / max_budget))
        return round(18 + 7 * ratio, 2), "Giá nằm trong ngân sách"
    midpoint = (min_budget + max_budget) / 2
    half_range = max((max_budget - min_budget) / 2, 1)
    closeness = max(0.0, 1 - abs(price - midpoint) / half_range)
    return round(20 + 5 * closeness, 2), "Giá nằm trong ngân sách"


def _category_score(requested: Any, tags: dict[str, Any], product_text: str) -> tuple[float, str]:
    desired = canonical_product_type(requested)
    actual = canonical_product_type(tags.get("product_type"))
    if not desired:
        return 8, "Chưa có loại sản phẩm rõ"
    if actual == desired:
        return SCORE_WEIGHTS["category"], f"Đúng loại sản phẩm: {PRODUCT_TYPE_LABELS.get(desired, requested)}"
    if actual in ADJACENT_PRODUCT_TYPES.get(desired, set()) or desired in ADJACENT_PRODUCT_TYPES.get(actual, set()):
        return 12, f"Nhóm sản phẩm gần với nhu cầu {PRODUCT_TYPE_LABELS.get(desired, requested)}"
    aliases = PRODUCT_TYPE_ALIASES.get(desired, [])
    if any(alias in product_text for alias in aliases):
        return 16, f"Text sản phẩm có tín hiệu gần với {PRODUCT_TYPE_LABELS.get(desired, requested)}"
    return 0, ""


def _occasion_score(prefs: dict[str, Any], tags: dict[str, Any]) -> tuple[float, str]:
    desired = {normalize_text(o) for o in prefs.get("occasion_tags") or [] if normalize_text(o)}
    actual = {normalize_text(o) for o in tags.get("occasion_tags") or [] if normalize_text(o)}
    if not desired:
        return 5, "Chưa có dịp mua rõ"
    overlap = desired & actual
    if overlap:
        text = ", ".join(OCCASION_DISPLAY.get(o, o) for o in sorted(overlap))
        return SCORE_WEIGHTS["occasion"], f"Phù hợp dịp {text}"
    if "gift" in desired and ({"birthday", "anniversary", "special_occasion", "daily"} & actual):
        return 8, "Có thể dùng làm quà dù chưa khớp dịp cụ thể"
    if "self_reward" in desired and ({"daily", "office", "party"} & actual):
        return 8, "Phù hợp nhu cầu tự thưởng/đeo thường xuyên"
    return 0, ""


def _material_stone_score(prefs: dict[str, Any], tags: dict[str, Any], product_text: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    desired_material_raw = prefs.get("material")
    desired_material = canonical_material(desired_material_raw, str(desired_material_raw))
    desired_purity = purity_from_text(desired_material_raw)
    actual_material = canonical_material(tags.get("material"), product_text)
    actual_purity = tags.get("purity") or purity_from_text(product_text)

    if desired_material:
        if desired_material == actual_material and (not desired_purity or desired_purity == actual_purity):
            score += 8
            label = MATERIAL_LABELS.get(actual_material, str(desired_material_raw))
            reasons.append(f"Chất liệu khớp {label}" + (f" {actual_purity.upper()}" if actual_purity else ""))
        elif desired_material == actual_material:
            score += 6
            reasons.append(f"Khớp nhóm chất liệu {MATERIAL_LABELS.get(actual_material, desired_material)}")
        elif (
            desired_purity
            and desired_purity == actual_purity
            and desired_material in {"white_gold", "yellow_gold", "italian_gold"}
            and actual_material in {"white_gold", "yellow_gold", "italian_gold"}
        ):
            # Đúng tuổi vàng (VD: 14K = 14K) nhưng khác màu — parity với "đúng màu, sai tuổi" (+6)
            # Khách nói "Vàng 14K" không specify màu → tuổi vàng là tiêu chí chính
            score += 6
            reasons.append(f"Đúng tuổi vàng {desired_purity.upper()}, màu vàng khác yêu cầu")
        elif desired_material in {"white_gold", "yellow_gold", "italian_gold"} and actual_material in {"white_gold", "yellow_gold", "italian_gold"}:
            score += 4
            reasons.append("Cùng nhóm vàng nhưng khác cả màu lẫn tuổi vàng")
    else:
        score += 3
        reasons.append("Chưa có chất liệu cụ thể")

    desired_stones = {canonical_stone(s) for s in prefs.get("desired_stones") or [] if canonical_stone(s)}
    actual_primary = canonical_stone(tags.get("primary_stone"))
    actual_secondary = canonical_stone(tags.get("secondary_stone"))
    if desired_stones:
        if actual_primary in desired_stones:
            score += 7
            reasons.append(f"Đá chính khớp {stone_label(actual_primary)}")
        elif actual_secondary in desired_stones:
            score += 5
            reasons.append(f"Đá phụ khớp {stone_label(actual_secondary)}")
        elif any(
            alias in product_text
            for stone in desired_stones
            for alias in _STONE_TEXT_ALIASES.get(stone, {stone})
        ):
            score += 4
            reasons.append("Text sản phẩm có tín hiệu đá khách quan tâm")
    else:
        score += 3
        reasons.append("Chưa có tín hiệu đá cụ thể")

    return min(score, SCORE_WEIGHTS["material_stone"]), reasons


def _style_match_score(customer_style: Any, product_styles: list[str], product_text: str) -> tuple[float, str]:
    desired = _style_terms(customer_style)
    if not desired:
        return 3, "Chưa có gu rõ"
    actual = {normalize_text(style) for style in product_styles if normalize_text(style)}
    overlap = desired & actual
    if overlap:
        text = ", ".join(STYLE_DISPLAY.get(o, o) for o in sorted(overlap)[:3])
        return SCORE_WEIGHTS["style"], f"Phong cách khớp {text}"
    if any(term in product_text for term in desired):
        return 6, "Text sản phẩm có tín hiệu gần với phong cách khách thích"
    return 0, ""


def _segment_score(prefs: dict[str, Any], tags: dict[str, Any]) -> tuple[float, str]:
    # Max = 7 khớp với SCORE_WEIGHTS["segment_value"].
    # Platinum = luxury tier (>=50M), Gold = premium tier (15-30M),
    # Silver/Basic = entry-mid tier.
    rfm = normalize_text(prefs.get("rfm"))
    price_tier = normalize_text(tags.get("price_tier"))
    monetary = safe_float(prefs.get("monetary"))
    premium_customer = bool(prefs.get("premium")) or monetary >= 50_000_000

    if premium_customer or "platinum" in rfm or "vip" in rfm:
        # Platinum: ngọt nhất với luxury, ổn với premium, kém khi mid/entry
        if price_tier == "luxury":
            return 7, f"Phù hợp phân khúc {prefs.get('rfm') or 'khách giá trị cao'}"
        if price_tier == "premium":
            return 6, "Sản phẩm premium phù hợp khách cao cấp"
        if price_tier == "mid":
            return 4, "Sản phẩm trung cấp — phù hợp upsell từ entry nhưng chưa đúng phân khúc Platinum"
        return 2, "Sản phẩm entry, chưa tương xứng phân khúc Platinum/VIP"

    if "gold" in rfm:
        # Gold: ngọt nhất với mid, ổn với premium, stretch với luxury, kém với entry
        if price_tier == "mid":
            return 7, "Phù hợp phân khúc Gold"
        if price_tier == "premium":
            return 6, "Sản phẩm premium — upsell tốt cho khách Gold"
        if price_tier == "luxury":
            return 4, "Vượt phân khúc Gold — chỉ phù hợp nếu khách sẵn sàng stretch"
        return 3, "Sản phẩm entry thấp hơn phân khúc Gold"

    if rfm:
        # Silver/Basic: ngọt nhất với entry, ổn với mid, kém khi premium trở lên
        if price_tier == "entry":
            return 7, f"Phù hợp phân khúc {prefs.get('rfm')}"
        if price_tier == "mid":
            return 5, f"Sản phẩm mid — upsell hợp lý cho phân khúc {prefs.get('rfm')}"
        return 3, "Sản phẩm cao hơn phân khúc hiện tại"

    return 3, "Chưa có phân khúc rõ"


def _gender_audience_score(prefs: dict[str, Any], tags: dict[str, Any]) -> tuple[float, str]:
    # Giới tính: tối đa 5 điểm | Nhóm tuổi: tối đa 3 điểm | Tổng tối đa = 8
    desired_gender = canonical_gender(prefs.get("recipient_gender") or prefs.get("gender"))
    if desired_gender == "unisex":
        desired_gender = ""
    actual_gender = canonical_gender(tags.get("gender"))
    desired_audience = canonical_audience(prefs.get("recipient_audience"))
    actual_audience = canonical_audience(tags.get("audience"))

    gender_score = 2.0  # mặc định khi chưa biết giới tính
    if desired_gender:
        if actual_gender == desired_gender:
            gender_score = 5.0
        elif actual_gender in {"", "unisex"}:
            gender_score = 3.5
        else:
            gender_score = 0.0

    audience_score = 1.5  # mặc định khi chưa biết nhóm tuổi
    if desired_audience:
        if actual_audience == desired_audience:
            audience_score = 3.0
        elif not actual_audience:
            audience_score = 1.5
        else:
            audience_score = 0.0

    score = min(SCORE_WEIGHTS["recipient_profile"], gender_score + audience_score)
    parts = []
    if desired_gender:
        parts.append(f"giới tính={gender_label(desired_gender)}")
    if desired_audience:
        parts.append(f"nhóm tuổi={audience_label(desired_audience)}")
    reason = "Khớp " + ", ".join(parts) if score >= 6.0 and parts else "Tệp người thụ hưởng chưa đủ rõ"
    return score, reason


class ProductRecommender:
    """Recommend products with hard filters and a visible 100-point score."""

    def __init__(self, catalog_path: Path | str | None = None) -> None:
        candidate = Path(catalog_path or DEFAULT_CATALOG_PATH)
        if not candidate.exists() and candidate == DEFAULT_CATALOG_PATH and LEGACY_METADATA_PATH.exists():
            candidate = LEGACY_METADATA_PATH
        self.catalog_path = candidate
        self.products = self._load_products()

    def _load_products(self) -> list[dict[str, Any]]:
        if not self.catalog_path.exists():
            return []
        with self.catalog_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for key in ["products", "items", "data", "catalog"]:
                if isinstance(raw.get(key), list):
                    raw = raw[key]
                    break
            else:
                raw = next((v for v in raw.values() if isinstance(v, list)), [])
        if not isinstance(raw, list):
            return []
        products = []
        for product in raw:
            if isinstance(product, dict):
                products.append(merge_product_tags(product))
        return products

    def recommend_for_profile(self, profile: dict[str, Any], top_n: int | None = 24) -> list[dict[str, Any]]:
        return self._recommend(self._profile_preferences(profile), top_n=top_n)

    def recommend_for_walkin(
        self,
        obs: dict[str, Any],
        purpose: str = "",
        budget: str = "",
        style: str = "",
        top_n: int | None = 24,
    ) -> list[dict[str, Any]]:
        return self._recommend(self._walkin_preferences(obs, purpose, budget, style), top_n=top_n)

    def recommend_for_recipient(
        self,
        profile: dict[str, Any],
        recipient_gender: str = "",
        recipient_audience: str = "",
        occasion: str = "",
        product_type: str = "",
        style: str = "",
        material: str = "",
        purpose: str = "",
        top_n: int | None = 24,
    ) -> list[dict[str, Any]]:
        """Recommend products using buyer's budget/RFM + explicit recipient criteria.

        Use this when the customer is buying for someone else and the TVV has
        collected information about the actual recipient/beneficiary.
        Budget and segment data come from the buyer's profile; all preference
        criteria (gender, audience, occasion, product type, style, material)
        reflect the recipient.
        """
        obs: dict[str, Any] = {
            "recipient_gender": recipient_gender or "",
            "recipient_audience": recipient_audience or "",
            "occasion": occasion or "",
            "product_type": product_type or "",
            "material": material or "",
        }
        buyer_budget = str(profile.get("budget") or "")
        rfm = str(profile.get("segment_rfm_tier") or "")
        monetary = safe_float(profile.get("monetary"))

        prefs = self._walkin_preferences(obs, purpose, buyer_budget, style)
        # Override segment fields with buyer's real profile data for accurate scoring
        prefs["rfm"] = rfm
        prefs["monetary"] = monetary
        prefs["premium"] = prefs["premium"] or "Platinum" in rfm or "VIP" in rfm or monetary >= 50_000_000
        prefs["source"] = "recipient"
        return self._recommend(prefs, top_n=top_n)

    def _profile_preferences(self, profile: dict[str, Any]) -> dict[str, Any]:
        customer_gender = canonical_gender(profile.get("gender"))
        lep_intent = normalize_text(profile.get("lep_intent") or profile.get("predicted_intent") or "")
        persona = normalize_text(profile.get("persona") or profile.get("persona_ml") or "")
        # preferred_type seeded from explicit recipient field first, then buyer history
        preferred_type = str(profile.get("recipient_product_type") or profile.get("preferred_type") or profile.get("product_focus") or "")
        rfm = str(profile.get("segment_rfm_tier") or "")

        occasions: set[str] = set()
        occasion_reasons: list[str] = []
        if "engagement" in lep_intent or safe_int(profile.get("sig_search_propose")) > 0:
            occasions.update(["engagement", "wedding"])
            occasion_reasons.append("LEP/signals cho thấy nhu cầu cầu hôn/cưới")
            preferred_type = preferred_type or "Nhẫn"
        if "anniversary" in lep_intent:
            occasions.add("anniversary")
            occasion_reasons.append("LEP intent anniversary")
        if "gift" in lep_intent or "qua tang" in persona or "gift" in persona:
            occasions.update(["gift", "birthday"])
            occasion_reasons.append("LEP/persona cho thấy mua quà tặng")
        if "self_reward" in lep_intent or "tu thuong" in persona:
            occasions.add("self_reward")
            occasion_reasons.append("LEP intent self reward")
        if safe_int(profile.get("birthday_in_days") or profile.get("sig_birthday_in_days"), 365) <= 30:
            occasions.add("birthday")
            occasion_reasons.append("Sinh nhật trong 30 ngày")

        # Extract desired stones từ material preference + behavioral signals
        desired_stones: list[str] = _stones_from_text(profile.get("material") or "")
        if safe_int(profile.get("sig_view_diamond")) > 0 and "diamond" not in desired_stones:
            desired_stones.append("diamond")

        # Mặc định: người nhận = người mua. Chỉ khi TVV biết người nhận khác
        # thì mới rẽ nhánh (gọi recommend_for_recipient với form thụ hưởng).
        explicit_recipient_gender = canonical_gender(profile.get("recipient_gender") or profile.get("beneficiary_gender"))
        recipient_gender = explicit_recipient_gender or customer_gender

        recipient_audience = canonical_audience(profile.get("recipient_audience") or profile.get("beneficiary_audience"))
        if not recipient_audience:
            age = safe_float(profile.get("age"))
            if age > 0:
                recipient_audience = "child" if age < 18 else "adult"
            else:
                recipient_audience = "adult"

        style_pref = str(profile.get("style") or "")

        return {
            "customer_gender": customer_gender,
            "recipient_gender": recipient_gender,
            "recipient_audience": recipient_audience,
            "budget": str(profile.get("budget") or ""),
            "category": preferred_type,
            "material": str(profile.get("material") or ""),
            "style": style_pref,
            "occasion_tags": sorted(occasions),
            "occasion_reasons": occasion_reasons,
            "desired_stones": desired_stones,
            "premium": "Platinum" in rfm or "VIP" in rfm or safe_float(profile.get("monetary")) >= 50_000_000,
            "rfm": rfm,
            "monetary": safe_float(profile.get("monetary")),
            "source": "profile",
        }

    def _walkin_preferences(self, obs: dict[str, Any], purpose: str, budget: str, style: str) -> dict[str, Any]:
        purpose_text = normalize_text(purpose or obs.get("purpose") or "")
        occasion_text = normalize_text(obs.get("occasion") or "")
        category = obs.get("product_type") or obs.get("recipient_product_type") or ""
        occasions: set[str] = set()
        reasons: list[str] = []

        for source in [purpose_text, occasion_text]:
            occasions.update(_occasion_terms_from_text(source))

        if "engagement" in occasions:
            category = category if category and category != "Chưa rõ" else "Nhẫn"
            reasons.append("TVV xác nhận mục đích cầu hôn/đính hôn")
        if "anniversary" in occasions:
            reasons.append("TVV xác nhận dịp kỷ niệm")
        if "birthday" in occasions:
            reasons.append("TVV xác nhận dịp sinh nhật")
        if "gift" in occasions:
            reasons.append("TVV xác nhận mua tặng")
        if "self_reward" in occasions:
            reasons.append("TVV xác nhận mua cho bản thân")

        customer_gender = canonical_gender(obs.get("gender"))
        recipient_gender = canonical_gender(obs.get("recipient_gender"))
        if not recipient_gender and "self_reward" in occasions:
            recipient_gender = customer_gender
        recipient_audience = canonical_audience(obs.get("recipient_audience") or obs.get("recipient_age_group"))
        if not recipient_audience and any(k in purpose_text for k in ["con trai", "con gai", "be trai", "be gai", "tre em"]):
            recipient_audience = "child"
        if not recipient_audience and "self_reward" in occasions:
            age = safe_float(obs.get("age"))
            recipient_audience = ("child" if age < 18 else "adult") if age > 0 else "adult"
        if not recipient_audience and (recipient_gender or "self_reward" in occasions):
            recipient_audience = "adult"

        # Extract desired stones từ material preference + context text
        context_norm = normalize_text(" ".join([str(obs.get("material") or ""), purpose_text, occasion_text]))
        desired_stones: list[str] = _stones_from_text(obs.get("material") or "")
        if "kim cuong" in context_norm or "diamond" in context_norm or "cau hon" in context_norm or "dinh hon" in context_norm:
            if "diamond" not in desired_stones:
                desired_stones.append("diamond")

        return {
            "customer_gender": customer_gender,
            "recipient_gender": recipient_gender,
            "recipient_audience": recipient_audience,
            "budget": budget or obs.get("budget") or "",
            "category": "" if category == "Chưa rõ" else category,
            "material": obs.get("material") or "",
            "style": style or obs.get("style") or "",
            "occasion_tags": sorted(occasions),
            "occasion_reasons": reasons,
            "desired_stones": desired_stones,
            "premium": (budget or obs.get("budget")) == "Trên 30 triệu",
            "rfm": "",
            "monetary": 0,
            "source": "walkin",
        }

    def _passes_hard_filters(
        self,
        product: dict[str, Any],
        prefs: dict[str, Any],
        min_budget: float | None,
        max_budget: float | None,
        *,
        enforce_category: bool,
        enforce_audience: bool,
        enforce_gender: bool,
    ) -> tuple[bool, list[str]]:
        price = product_price(product)
        if product.get("recommendation_eligible") is False:
            return False, ["Không eligible recommendation"]
        if product.get("record_class") and product.get("record_class") != "product":
            return False, ["Không phải product record"]
        if product.get("renderable_card") is False:
            return False, ["Không render được card"]
        if not product.get("name") or price <= 0:
            return False, ["Thiếu tên hoặc giá"]
        if max_budget is not None and price > max_budget:
            return False, [f"Vượt ngân sách {prefs.get('budget')}"]

        tags = product.get("recommendation_tags") or infer_product_tags(product)
        desired_category = canonical_product_type(prefs.get("category"))
        if enforce_category and desired_category and canonical_product_type(tags.get("product_type")) != desired_category:
            return False, [f"Không khớp loại {product_type_label(desired_category)}"]

        desired_audience = canonical_audience(prefs.get("recipient_audience"))
        if enforce_audience and desired_audience:
            actual_audience = canonical_audience(tags.get("audience"))
            if actual_audience and actual_audience != desired_audience:
                return False, [f"Không khớp nhóm người dùng {audience_label(desired_audience)}"]
        desired_gender = canonical_gender(prefs.get("recipient_gender") or prefs.get("gender"))
        if enforce_gender and desired_gender in {"male", "female"}:
            actual_gender = canonical_gender(tags.get("gender"))
            if actual_gender and actual_gender not in {desired_gender, "unisex"}:
                return False, [f"Không khớp giới tính người thụ hưởng {gender_label(desired_gender)}"]
        return True, []

    def _recommend(self, prefs: dict[str, Any], top_n: int | None = 24) -> list[dict[str, Any]]:
        min_budget, max_budget = budget_bounds(prefs.get("budget"))
        plans = [
            (True, True, True, "strict"),
            (False, True, True, "relaxed_category"),
            (True, False, True, "relaxed_audience"),
            (True, True, False, "relaxed_gender"),
            (False, False, False, "budget_only"),
        ]
        scored: list[dict[str, Any]] = []
        filtered_count = 0
        used_mode = "strict"

        for enforce_category, enforce_audience, enforce_gender, mode in plans:
            scored = []
            filtered_count = 0
            for product in self.products:
                passed, _ = self._passes_hard_filters(
                    product,
                    prefs,
                    min_budget,
                    max_budget,
                    enforce_category=enforce_category,
                    enforce_audience=enforce_audience,
                    enforce_gender=enforce_gender,
                )
                if not passed:
                    continue
                filtered_count += 1
                rec = self._score_product(product, prefs, min_budget, max_budget, mode)
                if rec["score"] > 0:
                    scored.append(rec)
            if scored:
                used_mode = mode
                break

        scored.sort(key=lambda item: (item["score"], item.get("popularity_raw", 0), -item.get("price", 0)), reverse=True)
        results = scored if top_n is None else scored[:top_n]
        for idx, rec in enumerate(results, 1):
            rec["rank"] = idx
            rec["filtered_count"] = filtered_count
            rec["filter_mode"] = used_mode
            if used_mode == "relaxed_category":
                rec.setdefault("evidence", []).append("Catalog không có đủ sản phẩm khớp loại SP + người thụ hưởng, đã nới loại SP")
            elif used_mode == "relaxed_audience":
                rec.setdefault("evidence", []).append("Catalog không có đủ sản phẩm khớp nhóm tuổi, đã nới điều kiện adult/child")
            elif used_mode == "relaxed_gender":
                rec.setdefault("evidence", []).append("Catalog không có đủ sản phẩm khớp giới tính người thụ hưởng, đã nới điều kiện gender")
            elif used_mode == "budget_only":
                rec.setdefault("evidence", []).append("Catalog không có match chặt, chỉ giữ điều kiện ngân sách")
        return results

    def _score_product(
        self,
        product: dict[str, Any],
        prefs: dict[str, Any],
        min_budget: float | None,
        max_budget: float | None,
        filter_mode: str,
    ) -> dict[str, Any]:
        tags = product.get("recommendation_tags") or infer_product_tags(product)
        text = _collect_product_text(product)
        price = product_price(product)
        breakdown: dict[str, float] = {}
        evidence: list[str] = []
        matched: list[str] = []

        budget_score, budget_reason = _score_budget(price, min_budget, max_budget)
        breakdown["budget"] = max(0, budget_score)
        evidence.append(budget_reason)
        if budget_score > 0:
            matched.append("budget")

        category_score, category_reason = _category_score(prefs.get("category"), tags, text)
        breakdown["category"] = category_score
        if category_reason:
            evidence.append(category_reason)
        if category_score >= SCORE_WEIGHTS["category"]:
            matched.append("category")

        occasion_score, occasion_reason = _occasion_score(prefs, tags)
        breakdown["occasion"] = occasion_score
        if occasion_reason:
            evidence.append(occasion_reason)
        if occasion_score >= SCORE_WEIGHTS["occasion"]:
            matched.append("occasion")

        material_score, material_reasons = _material_stone_score(prefs, tags, text)
        breakdown["material_stone"] = material_score
        evidence.extend(material_reasons)
        if material_score >= 12:
            matched.append("material_stone")

        style_score, style_reason = _style_match_score(prefs.get("style"), tags.get("style_tags") or [], text)
        breakdown["style"] = style_score
        if style_reason:
            evidence.append(style_reason)
        if style_score >= SCORE_WEIGHTS["style"]:
            matched.append("style")

        segment_score, segment_reason = _segment_score(prefs, tags)
        breakdown["segment_value"] = segment_score
        if segment_reason:
            evidence.append(segment_reason)
        if segment_score >= SCORE_WEIGHTS["segment_value"]:
            matched.append("segment")

        pop_raw = float(tags.get("popularity_score") or popularity_score(product))
        breakdown["popularity"] = round(min(SCORE_WEIGHTS["popularity"], pop_raw), 2)
        if breakdown["popularity"] >= 4:
            if parse_sold_text(product.get("sold_text") or product.get("sold")) or parse_rating(product.get("rating")):
                evidence.append("Có tín hiệu bán chạy/rating tốt")
            else:
                evidence.append("Catalog chưa có số bán; dùng proxy daily/classic/giftable/brand")
            matched.append("popularity")

        gender_score, gender_reason = _gender_audience_score(prefs, tags)
        breakdown["recipient_profile"] = gender_score
        if gender_reason:
            evidence.append(gender_reason)
        if gender_score >= SCORE_WEIGHTS["recipient_profile"]:
            matched.append("recipient_profile")

        score = round(min(100, sum(breakdown.values())), 2)
        product_type = tags.get("product_type") or _product_category(product, text)
        image_url = product.get("display_image_url") or product.get("image_url") or ""
        return {
            "sku": product.get("sku") or product.get("unique_id") or product.get("id") or product.get("product_id"),
            "product_id": product.get("product_id") or product.get("id"),
            "name": product.get("name", ""),
            "price": price,
            "price_text": money_vnd(price),
            "url": product.get("url", ""),
            "image_url": image_url,
            "display_image_url": image_url,
            "categories": tags.get("categories") or product.get("categories") or [],
            "product_type": product_type,
            "product_type_label": PRODUCT_TYPE_LABELS.get(product_type, product_type),
            "gender": gender_label(tags.get("gender") or product.get("gender")),
            "audience": audience_label(tags.get("audience") or product.get("audience")),
            "material": tags.get("material_label") or material_label(product.get("material")),
            "primary_stone": stone_label(tags.get("primary_stone") or product.get("primary_stone")),
            "secondary_stone": stone_label(tags.get("secondary_stone") or product.get("secondary_stone")),
            "price_tier": tags.get("price_tier") or product.get("price_tier") or "",
            "score": score,
            "score_breakdown": breakdown,
            "score_formula": "Ngân sách 25 + Loại SP 20 + Dịp 15 + Chất liệu/đá 15 + Tệp người thụ hưởng 8 + Phân khúc 7 + Style 6 + Bán chạy 4 = 100",
            "score_rules": SCORING_RULES,
            "evidence": evidence[:9],
            "matched_criteria": matched,
            "selling_points": tags.get("selling_points") or [],
            "style_tags": tags.get("style_tags") or [],
            "occasion_tags": tags.get("occasion_tags") or [],
            "premium_level": tags.get("price_tier") or "",
            "popularity_raw": pop_raw,
            "filter_mode": filter_mode,
        }


def recommendation_label(rec: dict[str, Any]) -> str:
    name = str(rec.get("name") or "").strip()
    sku = str(rec.get("sku") or "").strip()
    price = rec.get("price_text") or money_vnd(rec.get("price"))
    score = rec.get("score")
    score_part = f" · {score}/100" if score is not None else ""
    sku_part = f" ({sku})" if sku else ""
    return f"{name}{sku_part} · {price}{score_part}".strip()
