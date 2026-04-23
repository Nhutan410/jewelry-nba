"""
src/nba_engine.py
────────────────────────────────────────────────────────────────────────────
Next-Best-Action (NBA) Engine
Nhận LEP predictions (customer_id, predicted_intent, confidence)
→ tự sinh action dựa trên intent rules (không cần bảng nba_actions)
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime


# ── Intent → Action Config ────────────────────────────────────────────────────
# Toàn bộ business logic ánh xạ intent → action được định nghĩa TẠI ĐÂY.
# Không phụ thuộc vào bất kỳ sheet Excel hay file ngoài nào.

INTENT_CONFIG: dict[str, dict] = {
    "engagement": {
        "campaign_prefix": "eng",
        "product_focus": {
            "high":   "engagement_ring",
            "medium": "couple_ring",
            "low":    "ring_collection",
        },
        "cta": {
            "high":   "Tư vấn 1-1 miễn phí",
            "medium": "Xem bộ sưu tập",
            "low":    "Khám phá ngay",
        },
        "message_template": {
            "high":   "Chào [Tên], khoảnh khắc cầu hôn hoàn hảo đang chờ bạn. Nhẫn đính hôn PNJ – ưu đãi đặc biệt hôm nay 💍",
            "medium": "Khám phá bộ sưu tập nhẫn cầu hôn phù hợp với bạn tại PNJ [Tên].",
            "low":    "Bộ sưu tập nhẫn PNJ đang có nhiều mẫu mới. Xem ngay!",
        },
        "preferred_channels": ["push", "email", "zns"],
    },
    "anniversary": {
        "campaign_prefix": "anni",
        "product_focus": {
            "high":   "anniversary_set",
            "medium": "gift_set",
            "low":    "couple_jewelry",
        },
        "cta": {
            "high":   "Miễn phí khắc tên",
            "medium": "Giao nhanh 2h",
            "low":    "Xem gợi ý",
        },
        "message_template": {
            "high":   "Kỷ niệm của bạn sắp đến [Tên] – món quà ý nghĩa nhất đang chờ bạn tại PNJ 💍",
            "medium": "Tặng người ấy bộ trang sức kỷ niệm đặc biệt từ PNJ.",
            "low":    "Dịp kỷ niệm sắp đến? Khám phá gợi ý quà tặng từ PNJ.",
        },
        "preferred_channels": ["email", "zns", "push"],
    },
    "self_reward": {
        "campaign_prefix": "selfrw",
        "product_focus": {
            "high":   "premium_collection",
            "medium": "trending",
            "low":    "new_arrivals",
        },
        "cta": {
            "high":   "15% off dành riêng cho bạn",
            "medium": "Khám phá ngay",
            "low":    "Xem thêm",
        },
        "message_template": {
            "high":   "Bạn xứng đáng được tự thưởng [Tên]! Ưu đãi 15% bộ sưu tập Premium – chỉ hôm nay 💎",
            "medium": "Tự thưởng cho bản thân với trang sức đang hot nhất tuần này tại PNJ.",
            "low":    "Sản phẩm mới vừa về tại PNJ – xem ngay [Tên]!",
        },
        "preferred_channels": ["push", "in_app", "zns"],
    },
    "gift": {
        "campaign_prefix": "gift",
        "product_focus": {
            "high":   "gift_finder",
            "medium": "gift_set",
            "low":    "gift_guide",
        },
        "cta": {
            "high":   "Tư vấn chọn quà miễn phí",
            "medium": "Xem gợi ý",
            "low":    "Khám phá",
        },
        "message_template": {
            "high":   "Sắp đến ngày đặc biệt [Tên]? PNJ có hộp quà tặng + tư vấn miễn phí cho bạn 🎁",
            "medium": "Gợi ý quà tặng trang sức từ PNJ – chắc chắn người nhận sẽ thích!",
            "low":    "Tìm quà tặng ý nghĩa cho người thân yêu tại PNJ.",
        },
        "preferred_channels": ["zns", "email", "store"],
    },
}

# Priority threshold từ confidence
def _confidence_to_priority(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    elif confidence >= 0.60:
        return "medium"
    else:
        return "low"


# ── Business Rules (hard constraints) ────────────────────────────────────────
class BusinessRules:
    """Kiểm tra trước khi gửi bất kỳ action nào."""

    MIN_CONFIDENCE    = 0.45
    BIRTHDAY_WINDOW_DAYS = 30
    QUIET_HOURS = (22, 8)   # 22:00 → 08:00

    @staticmethod
    def is_quiet_hour(hour: int) -> bool:
        lo, hi = BusinessRules.QUIET_HOURS
        return hour >= lo or hour < hi

    @staticmethod
    def check(customer_row: pd.Series, channel: str, confidence: float) -> tuple[bool, str]:
        """Returns (allowed: bool, reason: str)"""
        # 1. Confidence threshold
        if confidence < BusinessRules.MIN_CONFIDENCE:
            return False, f"confidence {confidence:.2f} < threshold {BusinessRules.MIN_CONFIDENCE}"

        # 2. Opt-out: social_ads
        if customer_row.get("tp_social_ads", 0) == 0 and channel == "social_ads":
            return False, "customer not opted into social ads"

        return True, "ok"


# ── Channel Preference Learning ───────────────────────────────────────────────
class ChannelPreferenceLearner:
    """Chọn channel tốt nhất dựa trên engagement history của khách."""

    def score_channels(self, customer: pd.Series) -> dict[str, float]:
        scores = {}

        tp_e = max(customer.get("tp_email", 0), 1)
        scores["email"] = (
            customer.get("email_open", 0) * 2
            + customer.get("email_click", 0) * 5
        ) / tp_e

        tp_z = max(customer.get("tp_zns", 0), 1)
        scores["zns"] = (
            customer.get("zns_open", 0) * 2
            + customer.get("zns_click", 0) * 5
        ) / tp_z + 0.5   # ZNS baseline boost cho VN market

        tp_a = max(customer.get("tp_app", 0), 1)
        scores["push"] = customer.get("add_to_cart", 0) * 3 / tp_a

        scores["in_app"] = (
            customer.get("web_pdp_views", 0) + customer.get("add_to_cart", 0) * 2
        ) / max(customer.get("tp_web", 0) + customer.get("tp_app", 0), 1)

        scores["store"] = float(customer.get("tp_store", 0)) * 2

        max_s = max(scores.values()) if max(scores.values()) > 0 else 1
        return {k: v / max_s for k, v in scores.items()}

    def best_channel_for_intent(self, customer: pd.Series, intent: str) -> tuple[str, float]:
        """Chọn channel tốt nhất trong danh sách preferred của intent."""
        scores = self.score_channels(customer)
        preferred = INTENT_CONFIG.get(intent, {}).get("preferred_channels", list(scores.keys()))

        # Lọc theo preferred, fallback toàn bộ nếu không match
        filtered = {ch: scores.get(ch, 0) for ch in preferred}
        best = max(filtered, key=filtered.get)
        return best, round(scores.get(best, 0), 3)


# ── Personalization ────────────────────────────────────────────────────────────
def personalize_message(template: str, customer: pd.Series) -> str:
    """Template interpolation đơn giản. Prod → dùng LLM."""
    cid = str(customer.get("c", customer.get("customer_id", "bạn")))
    msg = template.replace("[Tên]", cid if len(cid) <= 10 else "bạn")

    bday = customer.get("sig_birthday_in_days", 999)
    if bday <= 7:
        msg += f" 🎂 Chỉ còn {int(bday)} ngày đến sinh nhật!"
    elif bday <= 30:
        msg += " (ưu đãi sinh nhật sắp hết hạn)"
    return msg


# ── NBA Engine chính ───────────────────────────────────────────────────────────
class NBAEngine:
    """
    Nhận LEP predictions → sinh Next-Best-Action cho từng khách.

    Input tối thiểu từ LEP:
        customer_id | predicted_intent | confidence

    Khônng cần bảng ba_actions từ ngoài — toàn bộ logic ánh xạ
    intent → action nằm trong INTENT_CONFIG ở trên.
    """

    def __init__(self):
        self.rules          = BusinessRules()
        self.channel_learner = ChannelPreferenceLearner()

    def _build_action(self, intent: str, priority: str, channel: str) -> dict:
        """Sinh action dict từ intent + priority + channel (không lookup Excel)."""
        cfg = INTENT_CONFIG.get(intent)
        if cfg is None:
            return {
                "campaign_id":    "UNKNOWN_INTENT",
                "product_focus":  "",
                "cta":            "",
                "message_template": f"Xin chào! PNJ có nhiều ưu đãi hấp dẫn dành cho bạn.",
            }
        return {
            "campaign_id":      f"{cfg['campaign_prefix']}_{priority}_{channel}",
            "product_focus":    cfg["product_focus"].get(priority, cfg["product_focus"]["low"]),
            "cta":              cfg["cta"].get(priority, cfg["cta"]["low"]),
            "message_template": cfg["message_template"].get(priority, cfg["message_template"]["low"]),
        }

    def generate_actions(self,
                         lep_predictions: pd.DataFrame,
                         df_profiles: pd.DataFrame) -> pd.DataFrame:
        """
        Main method: sinh NBA actions cho toàn bộ khách hàng.

        Args:
            lep_predictions: output của LEPModel.predict()
                Cần có: customer_id, predicted_intent, confidence
                (priority là tuỳ chọn — nếu không có sẽ tự tính từ confidence)
            df_profiles: raw profiles (để tính channel preference)

        Returns:
            DataFrame: customer_id, predicted_intent, confidence, priority,
                       channel, channel_score, campaign_id, product_focus,
                       cta, message, rule_status
        """
        # Merge predictions với profiles để lấy engagement history
        merged = lep_predictions.merge(
            df_profiles.rename(columns={"c": "customer_id"}),
            on="customer_id", how="left"
        )

        results = []
        for _, row in merged.iterrows():
            intent     = row["predicted_intent"]
            confidence = float(row["confidence"])

            # Priority: lấy từ LEP nếu có, nếu không tự tính từ confidence
            priority = str(row.get("priority", _confidence_to_priority(confidence)))
            if priority not in ("high", "medium", "low"):
                priority = _confidence_to_priority(confidence)

            # Chọn channel tốt nhất theo preference + intent
            best_channel, channel_score = self.channel_learner.best_channel_for_intent(row, intent)

            # Business rules check
            allowed, reason = self.rules.check(row, best_channel, confidence)

            # Build action từ intent rules
            action = self._build_action(intent, priority, best_channel)

            # Personalize message
            msg = personalize_message(action["message_template"], row) if allowed else "— blocked —"

            results.append({
                "customer_id":      row["customer_id"],
                "predicted_intent": intent,
                "confidence":       round(confidence, 3),
                "priority":         priority,
                "channel":          best_channel,
                "channel_score":    channel_score,
                "campaign_id":      action["campaign_id"],
                "product_focus":    action["product_focus"],
                "cta":              action["cta"],
                "message":          msg,
                "rule_status":      "allowed" if allowed else f"blocked: {reason}",
            })

        return pd.DataFrame(results)

    def get_action_summary(self, nba_df: pd.DataFrame) -> dict:
        """Tóm tắt phân phối actions."""
        total   = len(nba_df)
        allowed = (nba_df["rule_status"] == "allowed").sum()
        return {
            "total_customers": total,
            "actions_allowed": int(allowed),
            "actions_blocked": int(total - allowed),
            "by_intent":       nba_df["predicted_intent"].value_counts().to_dict(),
            "by_channel":      nba_df[nba_df["rule_status"] == "allowed"]["channel"].value_counts().to_dict(),
            "by_priority":     nba_df["priority"].value_counts().to_dict(),
        }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.lep_pipeline import LEPModel

    DATA_PATH = "data/customer_data_poc_enhanced.xlsx"
    df_p  = pd.read_excel(DATA_PATH, sheet_name="profiles_enhanced")
    df_ml = pd.read_excel(DATA_PATH, sheet_name="ml_predictions")

    # LEP
    lep   = LEPModel()
    lep.train(df_p, df_ml, verbose=False)
    preds = lep.predict(df_p)

    print("\n[LEP] Sample output:")
    print(preds[["customer_id", "predicted_intent", "confidence"]].head(4).to_string(index=False))

    # NBA — chỉ cần LEP output, không cần nba_actions sheet
    engine = NBAEngine()
    nba    = engine.generate_actions(preds, df_p)

    print("\n[NBA] Actions:")
    print(nba[["customer_id", "predicted_intent", "confidence", "priority",
               "channel", "campaign_id", "rule_status"]].to_string(index=False))

    print("\n[NBA] Summary:")
    import json
    print(json.dumps(engine.get_action_summary(nba), indent=2, ensure_ascii=False))
