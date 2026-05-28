"""
src/nba_engine_llm.py
────────────────────────────────────────────────────────────────────────────
NBA Engine (LLM version)
Kế thừa NBAEngine + sinh message bằng OpenAI API.

Luồng:
  1. LEP predictions → priority + channel
  2. Lookup instore_cache (từ instore_scripts.json) → lấy key_insight, urgency...
  3. Gọi LLM với đầy đủ context → rich message theo từng channel
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import pandas as pd
from typing import Optional

from src.nba_engine import NBAEngine, _confidence_to_priority
from src.llm_message_generator import (
    LLMMessageGenerator, CustomerContext, GeneratedMessage, row_to_context,
)


class NBAEngineLLM(NBAEngine):
    """
    NBA Engine tích hợp LLM.

    Điểm mới so với version cũ:
    - Nhận instore_cache (dict customer_id → instore entry từ instore_scripts.json)
    - Truyền key_insight, urgency_signal, product_rec... vào CustomerContext
    - LLM sinh message THEO insight thực tế — không generic
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 use_cache: bool = True):
        super().__init__()
        self.llm = LLMMessageGenerator(api_key=api_key, use_cache=use_cache)

    def generate_actions_llm(
        self,
        lep_predictions: pd.DataFrame,
        df_profiles:     pd.DataFrame,
        instore_cache:   Optional[dict] = None,
        verbose:         bool = True,
    ) -> pd.DataFrame:
        """
        End-to-end: LEP predictions + Instore cache → Business Rules → Channel → LLM message.

        Args:
            lep_predictions: output LEPModel.predict()  (cần: customer_id, predicted_intent, confidence)
            df_profiles:     profiles_enhanced sheet
            instore_cache:   dict { customer_id: instore_entry_dict }
                             Từ instore_scripts.json — chứa key_insight, urgency_signal, v.v.
            verbose:         In tiến trình

        Returns:
            DataFrame: customer_id, predicted_intent, confidence, priority,
                       channel, campaign_id, product_focus,
                       key_insight, urgency_signal, instore_intent,
                       llm_subject, llm_greeting, llm_body, llm_highlights_json,
                       llm_closing, llm_cta, tone, tokens_used, message_source, rule_status
        """
        instore_cache = instore_cache or {}

        # Merge predictions với profiles
        merged = lep_predictions.merge(
            df_profiles.rename(columns={"c": "customer_id"}),
            on="customer_id", how="left",
        )

        if verbose:
            print(f"\n[NBA-LLM] Processing {len(merged)} customers...")
            print(f"[NBA-LLM] Instore cache: {len(instore_cache)} entries")
            print(f"[NBA-LLM] Message mode: {self.llm.mode.upper()}\n")

        # ── Pass 1: Business rules + Channel + Build action ────────────────────
        action_contexts: list[CustomerContext] = []
        base_rows:       list[dict]            = []

        for _, row in merged.iterrows():
            intent     = row["predicted_intent"]
            confidence = float(row["confidence"])
            cid        = str(row["customer_id"])

            priority = str(row.get("priority", _confidence_to_priority(confidence)))
            if priority not in ("high", "medium", "low"):
                priority = _confidence_to_priority(confidence)

            best_channel, channel_score = self.channel_learner.best_channel_for_intent(row, intent)
            allowed, reason = self.rules.check(row, best_channel, confidence)
            rule_status = "allowed" if allowed else f"blocked: {reason}"
            action = self._build_action(intent, priority, best_channel)

            # ── Lấy instore context cho customer này ──────────────────────────
            instore = instore_cache.get(cid, {})
            ins_data = instore.get("instore", {}) if instore else {}

            key_insight        = str(ins_data.get("key_insight",         ""))
            urgency_signal     = str(ins_data.get("urgency_signal",      ""))
            online_insight     = str(ins_data.get("online_insight",      ""))
            instore_intent_val = str(ins_data.get("instore_intent",      ""))
            nba_strategy_val   = str(ins_data.get("nba_strategy",        ""))
            psych_trigger      = str(ins_data.get("psychology_trigger",  ""))
            prod_rec_1         = str(ins_data.get("product_rec_1",       ""))
            prod_rec_2         = str(ins_data.get("product_rec_2",       ""))
            prod_rec_3         = str(ins_data.get("product_rec_3",       ""))

            # product_focus: ưu tiên từ instore nếu có, fallback về action
            product_focus = str(ins_data.get("product_focus", "")) or action["product_focus"]

            base_rows.append({
                "customer_id":      cid,
                "predicted_intent": intent,
                "confidence":       round(confidence, 3),
                "priority":         priority,
                "channel":          best_channel,
                "channel_score":    channel_score,
                "campaign_id":      action["campaign_id"],
                "product_focus":    product_focus,
                "cta":              action["cta"],
                "rule_status":      rule_status,
                "_needs_llm":       allowed,
                # Instore context — sẽ ghi vào output
                "key_insight":        key_insight,
                "urgency_signal":     urgency_signal,
                "online_insight":     online_insight,
                "instore_intent":     instore_intent_val,
                "nba_strategy":       nba_strategy_val,
                "psychology_trigger": psych_trigger,
                "product_rec_1":      prod_rec_1,
                "product_rec_2":      prod_rec_2,
                "product_rec_3":      prod_rec_3,
            })

            if allowed:
                ctx = row_to_context(
                    row             = row,
                    intent          = intent,
                    confidence      = confidence,
                    priority        = priority,
                    channel         = best_channel,
                    product_focus   = product_focus,
                    cta             = action["cta"],
                    key_insight        = key_insight,
                    urgency_signal     = urgency_signal,
                    online_insight     = online_insight,
                    instore_intent     = instore_intent_val,
                    nba_strategy       = nba_strategy_val,
                    psychology_trigger = psych_trigger,
                    product_rec_1      = prod_rec_1,
                    product_rec_2      = prod_rec_2,
                    product_rec_3      = prod_rec_3,
                )
                action_contexts.append(ctx)

        # ── Pass 2: Batch LLM generation ──────────────────────────────────────
        llm_results: dict[str, GeneratedMessage] = {}

        if action_contexts:
            if verbose:
                print(f"[NBA-LLM] Generating messages for {len(action_contexts)} allowed actions:")
            messages = self.llm.generate_batch(action_contexts, verbose=verbose)
            llm_results = {m.customer_id: m for m in messages}

        # ── Pass 3: Assemble final DataFrame ──────────────────────────────────
        import json as _json

        final_rows = []
        for base in base_rows:
            cid = base["customer_id"]
            msg = llm_results.get(cid)

            # Serialize highlights list → JSON string để lưu trong DataFrame
            highlights_json = _json.dumps(msg.highlights, ensure_ascii=False) if msg else "[]"

            final_rows.append({
                "customer_id":      cid,
                "predicted_intent": base["predicted_intent"],
                "confidence":       base["confidence"],
                "priority":         base["priority"],
                "channel":          base["channel"],
                "channel_score":    base["channel_score"],
                "campaign_id":      base["campaign_id"],
                "product_focus":    base["product_focus"],
                # Instore context columns
                "key_insight":        base["key_insight"],
                "urgency_signal":     base["urgency_signal"],
                "online_insight":     base["online_insight"],
                "instore_intent":     base["instore_intent"],
                "nba_strategy":       base["nba_strategy"],
                "psychology_trigger": base["psychology_trigger"],
                "product_rec_1":      base["product_rec_1"],
                "product_rec_2":      base["product_rec_2"],
                "product_rec_3":      base["product_rec_3"],
                # Message fields
                "llm_subject":         msg.subject   if msg else "",
                "llm_greeting":        msg.greeting  if msg else "",
                "llm_body":            msg.body       if msg else "— blocked / no action —",
                "llm_highlights_json": highlights_json,
                "llm_closing":         msg.closing   if msg else "",
                "llm_cta":             msg.cta_text  if msg else base["cta"],
                "tone":                msg.tone      if msg else "",
                "tokens_used":         msg.tokens_used if msg else 0,
                "message_source":      msg.source    if msg else "n/a",
                "rule_status":         base["rule_status"],
            })

        result_df = pd.DataFrame(final_rows)

        if verbose:
            self.llm.print_stats()

        return result_df

    def print_message_report(self, nba_llm_df: pd.DataFrame) -> None:
        """In report đẹp cho từng message được tạo."""
        import json as _json

        allowed = nba_llm_df[nba_llm_df["rule_status"] == "allowed"]

        print("\n" + "=" * 70)
        print("PERSONALIZED MESSAGES — LLM OUTPUT (Insight-Driven)")
        print("=" * 70)

        for _, row in allowed.iterrows():
            print(f"\n{'─' * 70}")
            print(f"  {row['customer_id']:8s} | {row['predicted_intent']:12s} | "
                  f"{row['priority']:6s} | {row['channel']:8s} | "
                  f"confidence: {row['confidence']:.0%}")
            print(f"  Campaign : {row['campaign_id']}")
            print(f"  Source   : {row['message_source']} | Tone: {row['tone']}")

            # Key insight
            if row.get("key_insight"):
                print(f"  Insight  : {row['key_insight'][:80]}...")

            if row.get("llm_subject"):
                print(f"  Subject  : {row['llm_subject']}")
            if row.get("llm_greeting"):
                print(f"  Greeting : {row['llm_greeting'][:80]}")

            body = str(row.get("llm_body", ""))
            if body:
                # Print first 120 chars of body
                print(f"  Body     : {body[:120]}{'...' if len(body) > 120 else ''}")

            # Highlights
            try:
                hl = _json.loads(str(row.get("llm_highlights_json", "[]")))
                for h in hl[:3]:
                    print(f"  Highlight: {h}")
            except Exception:
                pass

            if row.get("llm_closing"):
                print(f"  Closing  : {str(row['llm_closing'])[:80]}")
            print(f"  CTA      : [{row['llm_cta']}]")

        blocked = nba_llm_df[nba_llm_df["rule_status"] != "allowed"]
        if len(blocked) > 0:
            print(f"\n{'─' * 70}")
            print(f"  Blocked ({len(blocked)} customers):")
            for _, row in blocked.iterrows():
                print(f"  {row['customer_id']:8s} — {row['rule_status']}")


# ── Standalone run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.lep_pipeline import LEPModel

    DATA_PATH = "data/customer_data_poc_enhanced.xlsx"
    df_p  = pd.read_excel(DATA_PATH, sheet_name="profiles_enhanced")
    df_ml = pd.read_excel(DATA_PATH, sheet_name="ml_predictions")

    lep   = LEPModel()
    lep.train(df_p, df_ml, verbose=False)
    preds = lep.predict(df_p)

    engine = NBAEngineLLM()
    result = engine.generate_actions_llm(preds, df_p, verbose=True)
    engine.print_message_report(result)

    print("\n[Done] result DataFrame:")
    print(result[["customer_id", "predicted_intent", "channel",
                  "key_insight", "llm_body", "message_source"]].to_string(index=False))
