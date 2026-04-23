"""
src/feedback_loop.py
────────────────────────────────────────────────────────────────────────────
Feedback Loop: xử lý interaction logs → cập nhật channel performance
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import pandas as pd
import numpy as np


# ── Reward signals ─────────────────────────────────────────────────────────────
REWARD_MAP = {
    "purchase":        10.0,
    "checkout_start":   5.0,
    "add_to_cart":      3.0,
    "pdp_view":         1.5,
    "email_click":      2.0,
    "zns_click":        2.0,
    "ad_click":         1.5,
    "app_checkout":     8.0,
    "email_open":       0.5,
    "zns_open":         0.5,
    "app_open":         0.3,
    "landing":          1.0,
    "page_view":        0.2,
    "store_visit":      2.0,
    "direct_visit":     0.3,
    "ad_impression":    0.0,
    "agent_call":       1.0,
}

NEGATIVE_EVENTS = {"unsubscribe", "opt_out", "complaint"}


class FeedbackProcessor:
    """
    Xử lý interaction log → tính reward per customer per channel
    → cập nhật channel performance table
    """

    def __init__(self):
        self.channel_perf: pd.DataFrame | None = None

    def process_interactions(self, df_interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Input: interactions sheet
        Output: customer × channel performance table
        """
        df = df_interactions.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Tính reward cho mỗi event
        df["reward"] = df["event_type"].map(REWARD_MAP).fillna(0)
        df["is_negative"] = df["event_type"].isin(NEGATIVE_EVENTS).astype(int)

        # Aggregate per customer × channel
        perf = (
            df.groupby(["customer_id", "channel"])
            .agg(
                total_reward=("reward", "sum"),
                n_interactions=("touchpoint_id", "count"),
                n_negative=("is_negative", "sum"),
                last_seen=("timestamp", "max"),
                total_cost=("cost", "sum"),
            )
            .reset_index()
        )

        # Channel score = reward / interactions (tránh div0)
        perf["channel_score"] = perf["total_reward"] / perf["n_interactions"].clip(lower=1)

        # ROI nếu có cost
        perf["roi"] = np.where(
            perf["total_cost"] > 0,
            perf["total_reward"] / perf["total_cost"],
            perf["total_reward"],  # free channel = full reward
        )

        self.channel_perf = perf
        return perf

    def get_best_channel_per_customer(self) -> pd.DataFrame:
        """Trả về channel tốt nhất cho mỗi khách."""
        assert self.channel_perf is not None, "Gọi process_interactions() trước"
        idx = self.channel_perf.groupby("customer_id")["channel_score"].idxmax()
        best = self.channel_perf.loc[idx, ["customer_id", "channel", "channel_score"]].copy()
        best.columns = ["customer_id", "best_channel", "best_channel_score"]
        return best.reset_index(drop=True)

    def get_conversion_funnel(self, df_interactions: pd.DataFrame,
                               df_purchases: pd.DataFrame) -> pd.DataFrame:
        """
        Tính conversion funnel: impressions → clicks → cart → purchase
        per channel
        """
        # Classify events
        impressions = df_interactions[
            df_interactions["event_type"].isin(["ad_impression","page_view","email_open","zns_open"])
        ].groupby("channel").size().rename("impressions")

        clicks = df_interactions[
            df_interactions["event_type"].isin(["ad_click","email_click","zns_click","landing"])
        ].groupby("channel").size().rename("clicks")

        carts = df_interactions[
            df_interactions["event_type"] == "add_to_cart"
        ].groupby("channel").size().rename("add_to_cart")

        purchases_by_channel = df_purchases.groupby("channel").size().rename("purchases")

        funnel = pd.concat([impressions, clicks, carts, purchases_by_channel], axis=1).fillna(0)
        funnel["click_rate"]   = (funnel["clicks"] / funnel["impressions"].clip(1) * 100).round(1)
        funnel["cart_rate"]    = (funnel["add_to_cart"] / funnel["clicks"].clip(1) * 100).round(1)
        funnel["convert_rate"] = (funnel["purchases"] / funnel["add_to_cart"].clip(1) * 100).round(1)

        return funnel.reset_index()

    def get_campaign_performance(self, df_interactions: pd.DataFrame) -> pd.DataFrame:
        """Performance per campaign."""
        df = df_interactions.copy()
        df["reward"] = df["event_type"].map(REWARD_MAP).fillna(0)

        camp = (
            df.groupby("campaign")
            .agg(
                touchpoints=("touchpoint_id", "count"),
                total_reward=("reward", "sum"),
                total_cost=("cost", "sum"),
                unique_customers=("customer_id", "nunique"),
            )
            .reset_index()
        )
        camp["avg_reward_per_customer"] = (camp["total_reward"] / camp["unique_customers"].clip(1)).round(2)
        camp["roi"] = np.where(
            camp["total_cost"] > 0,
            (camp["total_reward"] / camp["total_cost"]).round(2),
            camp["total_reward"].round(2),
        )
        return camp.sort_values("total_reward", ascending=False)


# ── Standalone run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = "data/customer_data_poc_enhanced.xlsx"
    df_i = pd.read_excel(DATA_PATH, sheet_name="interactions")
    df_p = pd.read_excel(DATA_PATH, sheet_name="purchases")

    fb = FeedbackProcessor()
    perf = fb.process_interactions(df_i)
    print("[Feedback] Channel performance:")
    print(perf.to_string(index=False))

    best = fb.get_best_channel_per_customer()
    print("\n[Feedback] Best channel per customer:")
    print(best.to_string(index=False))

    funnel = fb.get_conversion_funnel(df_i, df_p)
    print("\n[Feedback] Conversion funnel:")
    print(funnel.to_string(index=False))

    camp = fb.get_campaign_performance(df_i)
    print("\n[Feedback] Campaign performance:")
    print(camp.to_string(index=False))
