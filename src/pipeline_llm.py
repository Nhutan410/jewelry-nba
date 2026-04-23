"""
src/pipeline_llm.py
────────────────────────────────────────────────────────────────────────────
End-to-end pipeline tích hợp LLM (OpenAI API)
LEP → NBA → LLM Message Generation → Excel Report

Cách chạy:
  export OPENAI_API_KEY="sk-..."           # optional, fallback nếu không có
  python3 src/pipeline_llm.py
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.lep_pipeline    import LEPModel, get_feature_importance
from src.nba_engine_llm  import NBAEngineLLM
from src.feedback_loop   import FeedbackProcessor

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = ROOT / "data" / "customer_data_poc_enhanced.xlsx"
OUTPUT_PATH = ROOT / "outputs" / "nba_results_llm.xlsx"
OUTPUT_PATH.parent.mkdir(exist_ok=True)


def main():
    os.chdir(ROOT)

    print("=" * 65)
    print("JEWELRY NBA — PIPELINE WITH LLM MESSAGE GENERATION")
    print("=" * 65)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        print(f"[Config] API key: {'*' * 10}{api_key[-4:]} (từ OPENAI_API_KEY)")
    else:
        print("[Config] Không có OPENAI_API_KEY → chạy fallback template mode")
        print("[Config] Để dùng OpenAI API: export OPENAI_API_KEY='sk-...'")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\n[Data] Loading: {DATA_PATH}")
    sheets      = pd.read_excel(DATA_PATH, sheet_name=None)
    df_profiles = sheets["profiles_enhanced"]
    df_inter    = sheets["interactions"]
    df_purch    = sheets["purchases"]
    df_ml       = sheets.get("ml_predictions", pd.DataFrame())
    print(f"  ✓ {len(df_profiles)} khách | {len(df_inter)} interactions | {len(df_purch)} purchases")

    # ── Step 1: LEP ────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 1 — LIFE EVENT PREDICTION")
    print("─" * 65)
    lep = LEPModel(n_estimators=100)
    lep.train(df_profiles, df_ml, verbose=True)
    lep_preds = lep.predict(df_profiles)

    # Distribution
    dist = lep_preds["predicted_intent"].value_counts()
    print("\n  Intent distribution:")
    for intent, n in dist.items():
        bar = "█" * n
        print(f"    {intent:15s} {n:3d}  {bar}")

    # Top features
    print("\n  Top 8 predictive features:")
    fi = get_feature_importance(lep, top_n=8)
    for _, r in fi.iterrows():
        bar = "▓" * int(r["importance"] * 80)
        print(f"    {r['feature']:35s} {r['importance']:.4f}  {bar}")

    # ── Step 2: NBA + LLM ──────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 2 — NEXT-BEST-ACTION + LLM PERSONALIZATION")
    print("─" * 65)

    engine     = NBAEngineLLM(api_key=api_key or None, use_cache=True)
    nba_result = engine.generate_actions_llm(lep_preds, df_profiles, verbose=True)
    engine.print_message_report(nba_result)

    # ── Step 3: Feedback ───────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 3 — FEEDBACK LOOP")
    print("─" * 65)
    fb   = FeedbackProcessor()
    fb.process_interactions(df_inter)
    best = fb.get_best_channel_per_customer()
    camp = fb.get_campaign_performance(df_inter)

    print("\n  Best channel per customer:")
    print(best.to_string(index=False))

    print("\n  Top 5 campaigns by reward:")
    print(camp.head(5)[["campaign","total_reward","roi","unique_customers"]].to_string(index=False))

    # ── Step 4: Save Excel ─────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 4 — SAVING OUTPUTS")
    print("─" * 65)

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:

        # Sheet 1: LEP predictions
        lep_preds.to_excel(writer, sheet_name="lep_predictions", index=False)

        # Sheet 2: NBA + LLM full result
        nba_result.to_excel(writer, sheet_name="nba_llm_full", index=False)

        # Sheet 3: Chỉ allowed actions với message đầy đủ
        allowed = nba_result[nba_result["rule_status"] == "allowed"].copy()
        allowed_out = allowed[[
            "customer_id", "predicted_intent", "confidence", "priority",
            "channel", "campaign_id", "product_focus",
            "llm_subject", "llm_body", "llm_cta",
            "tone", "tokens_used", "message_source"
        ]]
        allowed_out.to_excel(writer, sheet_name="messages_to_send", index=False)

        # Sheet 4: Blocked actions
        blocked = nba_result[nba_result["rule_status"] != "allowed"][[
            "customer_id", "predicted_intent", "confidence", "rule_status"
        ]]
        blocked.to_excel(writer, sheet_name="blocked_actions", index=False)

        # Sheet 5: Channel performance
        if fb.channel_perf is not None:
            fb.channel_perf.to_excel(writer, sheet_name="channel_performance", index=False)

        # Sheet 6: Campaign ROI
        camp.to_excel(writer, sheet_name="campaign_roi", index=False)

        # Sheet 7: Feature importance
        fi_all = get_feature_importance(lep, top_n=20)
        fi_all.to_excel(writer, sheet_name="feature_importance", index=False)

        # Sheet 8: Summary stats
        summary = pd.DataFrame([{
            "total_customers":   len(nba_result),
            "actions_sent":      len(allowed),
            "actions_blocked":   len(blocked),
            "llm_generated":     (nba_result["message_source"] == "llm").sum(),
            "fallback_used":     (nba_result["message_source"].str.startswith("fallback")).sum(),
            "total_tokens_used": nba_result["tokens_used"].sum(),
            "intents":           str(dist.to_dict()),
            "top_channel":       nba_result[nba_result["rule_status"]=="allowed"]["channel"].value_counts().idxmax() if len(allowed) > 0 else "n/a",
        }])
        summary.to_excel(writer, sheet_name="run_summary", index=False)

    print(f"\n  ✓ Saved → {OUTPUT_PATH}")
    print(f"  Sheets: lep_predictions | nba_llm_full | messages_to_send |")
    print(f"          blocked_actions | channel_performance | campaign_roi |")
    print(f"          feature_importance | run_summary")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Customers processed : {len(nba_result)}")
    print(f"  Messages to send    : {len(allowed)}")
    print(f"  Blocked by rules    : {len(blocked)}")
    llm_n = (nba_result["message_source"] == "llm").sum()
    fb_n  = (nba_result["message_source"].str.startswith("fallback")).sum()
    print(f"  LLM generated       : {llm_n}")
    print(f"  Fallback template   : {fb_n}")
    print(f"  Output file         : {OUTPUT_PATH}")
    print("=" * 65)

    return nba_result


if __name__ == "__main__":
    main()
