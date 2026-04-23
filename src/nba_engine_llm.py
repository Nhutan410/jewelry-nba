"""
src/nba_engine_llm.py
────────────────────────────────────────────────────────────────────────────
NBA Engine (LLM version)
Kế thừa NBAEngine mới (không cần df_actions) + thay thế message generation
bằng OpenAI API call thay vì template cứng.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import pandas as pd
from typing import Optional

from src.nba_engine import NBAEngine, BusinessRules, ChannelPreferenceLearner, _confidence_to_priority
from src.llm_message_generator import (
    LLMMessageGenerator, CustomerContext, GeneratedMessage, row_to_context
)


class NBAEngineLLM(NBAEngine):
    """
    NBA Engine tích hợp LLM.

    Input: LEP predictions (customer_id, predicted_intent, confidence)
    Flow:
      1. Tính priority từ confidence
      2. Chọn channel từ engagement history  
      3. Build action từ INTENT_CONFIG (không cần nba_actions sheet)
      4. Gọi LLM sinh message cá nhân hoá (hoặc fallback template nếu không có API key)
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 use_cache: bool = True):
        super().__init__()
        self.llm = LLMMessageGenerator(api_key=api_key, use_cache=use_cache)

    def generate_actions_llm(self,
                              lep_predictions: pd.DataFrame,
                              df_profiles: pd.DataFrame,
                              verbose: bool = True) -> pd.DataFrame:
        """
        End-to-end: LEP predictions → Business Rules → Channel → LLM message.

        Args:
            lep_predictions: output LEPModel.predict()
                Cần: customer_id, predicted_intent, confidence
            df_profiles: profiles_enhanced (để tính channel preference)
            verbose: in tiến trình

        Returns DataFrame: customer_id, predicted_intent, confidence, priority,
            channel, campaign_id, product_focus,
            llm_subject, llm_body, llm_cta, tone, tokens_used,
            message_source, rule_status
        """
        # Merge predictions với profiles
        merged = lep_predictions.merge(
            df_profiles.rename(columns={"c": "customer_id"}),
            on="customer_id", how="left"
        )

        if verbose:
            print(f"\n[NBA-LLM] Processing {len(merged)} customers...")
            print(f"[NBA-LLM] Message mode: {self.llm.mode.upper()}\n")

        # ── Pass 1: Business rules + Channel + Action ──────────────────────
        action_contexts: list[CustomerContext] = []
        base_rows: list[dict] = []

        for _, row in merged.iterrows():
            intent     = row["predicted_intent"]
            confidence = float(row["confidence"])

            # Priority từ LEP nếu có, nếu không tính từ confidence
            priority = str(row.get("priority", _confidence_to_priority(confidence)))
            if priority not in ("high", "medium", "low"):
                priority = _confidence_to_priority(confidence)

            # Channel tốt nhất theo intent + engagement history
            best_channel, channel_score = self.channel_learner.best_channel_for_intent(row, intent)

            # Business rules
            allowed, reason = self.rules.check(row, best_channel, confidence)
            rule_status = "allowed" if allowed else f"blocked: {reason}"

            # Build action từ INTENT_CONFIG (không lookup Excel)
            action = self._build_action(intent, priority, best_channel)

            base_rows.append({
                "customer_id":      row["customer_id"],
                "predicted_intent": intent,
                "confidence":       round(confidence, 3),
                "priority":         priority,
                "channel":          best_channel,
                "channel_score":    channel_score,
                "campaign_id":      action["campaign_id"],
                "product_focus":    action["product_focus"],
                "cta":              action["cta"],
                "rule_status":      rule_status,
                "_needs_llm":       allowed,
            })

            # Chuẩn bị context cho LLM (chỉ với allowed actions)
            if allowed:
                ctx = row_to_context(
                    row           = row,
                    intent        = intent,
                    confidence    = confidence,
                    priority      = priority,
                    channel       = best_channel,
                    product_focus = action["product_focus"],
                    cta           = action["cta"],
                )
                action_contexts.append(ctx)

        # ── Pass 2: Batch LLM generation ──────────────────────────────────
        llm_results: dict[str, GeneratedMessage] = {}

        if action_contexts:
            if verbose:
                print(f"[NBA-LLM] Generating messages for {len(action_contexts)} allowed actions:")
            messages = self.llm.generate_batch(action_contexts, verbose=verbose)
            llm_results = {m.customer_id: m for m in messages}

        # ── Pass 3: Assemble final DataFrame ──────────────────────────────
        final_rows = []
        for base in base_rows:
            cid = base["customer_id"]
            msg = llm_results.get(cid)

            final_rows.append({
                "customer_id":      cid,
                "predicted_intent": base["predicted_intent"],
                "confidence":       base["confidence"],
                "priority":         base["priority"],
                "channel":          base["channel"],
                "channel_score":    base["channel_score"],
                "campaign_id":      base["campaign_id"],
                "product_focus":    base["product_focus"],
                # LLM outputs
                "llm_subject":      msg.subject     if msg else "",
                "llm_body":         msg.body        if msg else "— blocked / no action —",
                "llm_cta":          msg.cta_text    if msg else base["cta"],
                "tone":             msg.tone        if msg else "",
                "tokens_used":      msg.tokens_used if msg else 0,
                "message_source":   msg.source      if msg else "n/a",
                "rule_status":      base["rule_status"],
            })

        result_df = pd.DataFrame(final_rows)

        if verbose:
            self.llm.print_stats()

        return result_df

    def print_message_report(self, nba_llm_df: pd.DataFrame) -> None:
        """In report đẹp cho từng message được tạo."""
        allowed = nba_llm_df[nba_llm_df["rule_status"] == "allowed"]

        print("\n" + "=" * 65)
        print("PERSONALIZED MESSAGES — LLM OUTPUT")
        print("=" * 65)

        for _, row in allowed.iterrows():
            print(f"\n{'─' * 65}")
            print(f"  {row['customer_id']:8s} | {row['predicted_intent']:12s} | "
                  f"{row['priority']:6s} | {row['channel']:8s} | "
                  f"confidence: {row['confidence']:.0%}")
            print(f"  Campaign : {row['campaign_id']}")
            print(f"  Source   : {row['message_source']} | Tone: {row['tone']}")
            if row.get("llm_subject"):
                print(f"  Subject  : {row['llm_subject']}")
            print(f"  Message  : {row['llm_body']}")
            print(f"  CTA      : [{row['llm_cta']}]")

        blocked = nba_llm_df[nba_llm_df["rule_status"] != "allowed"]
        if len(blocked) > 0:
            print(f"\n{'─' * 65}")
            print(f"  Blocked ({len(blocked)} customers):")
            for _, row in blocked.iterrows():
                print(f"  {row['customer_id']:8s} — {row['rule_status']}")


# ── Standalone run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.lep_pipeline import LEPModel

    DATA_PATH = "data/customer_data_poc_enhanced.xlsx"
    df_p  = pd.read_excel(DATA_PATH, sheet_name="profiles_enhanced")
    df_ml = pd.read_excel(DATA_PATH, sheet_name="ml_predictions")

    # LEP
    lep   = LEPModel()
    lep.train(df_p, df_ml, verbose=False)
    preds = lep.predict(df_p)

    # NBA + LLM — không cần nba_actions sheet
    engine = NBAEngineLLM()
    result = engine.generate_actions_llm(preds, df_p, verbose=True)
    engine.print_message_report(result)

    print("\n[Done] result DataFrame:")
    print(result[["customer_id", "predicted_intent", "channel",
                  "llm_body", "llm_cta", "message_source"]].to_string(index=False))
