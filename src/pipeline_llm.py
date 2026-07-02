"""
src/pipeline_llm.py
────────────────────────────────────────────────────────────────────────────
End-to-end pipeline tích hợp LLM (OpenAI API)

Luồng:
  1. Load instore_scripts.json  ← key_insight, urgency, product_rec...
  2. LEP Model predict intent
  3. NBA Engine LLM → sinh message dựa trên key_insight từng khách
  4. Lưu kết quả vào outputs/nba_messages.json  (thay vì xlsx)

Cách chạy:
  export OPENAI_API_KEY="sk-..."           # optional
  python3 src/pipeline_llm.py
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Load .env (OPENAI_API_KEY, etc.) ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # python-dotenv chưa cài — dùng env var hệ thống

from src.lep_pipeline    import LEPModel, get_feature_importance
from src.nba_engine_llm  import NBAEngineLLM
from src.feedback_loop   import FeedbackProcessor

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = ROOT / "data"    / "customer_data_poc_enhanced.xlsx"
INSTORE_CACHE  = ROOT / "outputs" / "instore_scripts.json"
OUTPUT_JSON    = ROOT / "outputs" / "nba_messages.json"
OUTPUT_JSON.parent.mkdir(exist_ok=True)


def load_instore_cache(path: Path) -> dict:
    """
    Load instore_scripts.json → dict { customer_id: entry_dict }.
    Trả về dict rỗng nếu file chưa tồn tại.
    """
    if not path.exists():
        print(f"[Pipeline] Không tìm thấy instore cache: {path}")
        print("[Pipeline] Sinh message sẽ không có Key Insight — chạy instore pipeline trước để có dữ liệu tốt hơn.")
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cache = {}
        for c in data.get("customers", []):
            cid = str(c.get("customer_id", ""))
            if cid:
                cache[cid] = c
        print(f"[Pipeline] Đã load instore cache: {len(cache)} khách")
        return cache
    except Exception as exc:
        print(f"[Pipeline] Lỗi đọc instore cache: {exc}")
        return {}


def save_output_json(
    nba_df:        pd.DataFrame,
    instore_cache: dict,
    output_path:   Path,
) -> None:
    """
    Lưu kết quả message vào JSON.

    Cấu trúc mỗi customer:
      customer_id, context (insight + instore), delivery (channel...), content (message)
    """
    import json as _json

    customers = []

    for _, row in nba_df.iterrows():
        cid = str(row["customer_id"])

        # Parse highlights từ JSON string
        try:
            highlights = _json.loads(str(row.get("llm_highlights_json", "[]")))
        except Exception:
            highlights = []

        # subject: None nếu là nan / "None" / ""
        subject_raw = row.get("llm_subject")
        subject_val = None
        if subject_raw is not None and str(subject_raw) not in ("", "nan", "None"):
            subject_val = str(subject_raw)

        entry = {
            "customer_id":  cid,
            "generated_at": datetime.now().isoformat(),

            # Ngữ cảnh instore — nguồn gốc của insight
            "context": {
                "key_insight":        str(row.get("key_insight",        "")),
                "urgency_signal":     str(row.get("urgency_signal",     "")),
                "online_insight":     str(row.get("online_insight",     "")),
                "instore_intent":     str(row.get("instore_intent",     "")),
                "nba_strategy":       str(row.get("nba_strategy",       "")),
                "psychology_trigger": str(row.get("psychology_trigger", "")),
                "product_focus":      str(row.get("product_focus",      "")),
                "product_rec_1":      str(row.get("product_rec_1",      "")),
                "product_rec_2":      str(row.get("product_rec_2",      "")),
                "product_rec_3":      str(row.get("product_rec_3",      "")),
            },

            # Thông tin gửi tin
            "delivery": {
                "channel":          str(row.get("channel",          "")),
                "priority":         str(row.get("priority",         "")),
                "campaign_id":      str(row.get("campaign_id",      "")),
                "rule_status":      str(row.get("rule_status",      "")),
                "predicted_intent": str(row.get("predicted_intent", "")),
                "confidence":       float(row.get("confidence",     0)),
            },

            # Nội dung message đã sinh (structured by channel)
            "content": {
                "source":       str(row.get("message_source", "")),
                "tone":         str(row.get("tone",           "")),
                "tokens_used":  int(row.get("tokens_used",    0)),
                "subject":      subject_val,
                "greeting":     str(row.get("llm_greeting", "")) or None,
                "body":         str(row.get("llm_body",     "")),
                "highlights":   highlights,
                "closing":      str(row.get("llm_closing",  "")) or None,
                "cta_text":     str(row.get("llm_cta",      "")),
            },
        }
        customers.append(entry)

    allowed_count = sum(
        1 for c in customers
        if c["delivery"].get("rule_status") == "allowed"
    )
    blocked_count = len(customers) - allowed_count

    output = {
        "version":            "2.0",
        "generated_at":       datetime.now().isoformat(),
        "total_customers":    len(customers),
        "allowed_messages":   allowed_count,
        "blocked_messages":   blocked_count,
        "instore_cache_used": bool(instore_cache),
        "customers":          customers,
    }

    output_path.write_text(
        _json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  ✓ Saved → {output_path}")
    print(f"  Tổng: {len(customers)} | Được gửi: {allowed_count} | Bị chặn: {blocked_count}")


def main():
    os.chdir(ROOT)

    print("=" * 70)
    print("PNJ JEWELRY NBA — PIPELINE WITH LLM + INSTORE INSIGHT")
    print("=" * 70)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        print(f"[Config] API key: {'*' * 10}{api_key[-4:]} (từ OPENAI_API_KEY)")
    else:
        print("[Config] Không có OPENAI_API_KEY → chạy fallback template mode")

    # ── Step 0: Load Instore Cache ─────────────────────────────────────────────
    print(f"\n[Step 0] Loading instore cache: {INSTORE_CACHE.name}")
    instore_cache = load_instore_cache(INSTORE_CACHE)

    if instore_cache:
        # Preview sample insight
        sample_id = next(iter(instore_cache))
        sample_ins = instore_cache[sample_id].get("instore", {})
        print(f"\n  Sample customer: {sample_id}")
        print(f"  Key insight: {sample_ins.get('key_insight', 'N/A')[:80]}")
        print(f"  Urgency    : {sample_ins.get('urgency_signal', 'N/A')}")

    # ── Step 1: Load Data ──────────────────────────────────────────────────────
    print(f"\n[Step 1] Loading data: {DATA_PATH.name}")
    sheets      = pd.read_excel(DATA_PATH, sheet_name=None)
    df_profiles = sheets["profiles_enhanced"]
    df_inter    = sheets["interactions"]
    df_purch    = sheets["purchases"]
    df_ml       = sheets.get("ml_predictions", pd.DataFrame())

    # Merge gender từ sheet profiles chỉ khi profiles_enhanced chưa có cột gender
    if "profiles" in sheets and "gender" not in df_profiles.columns:
        df_gender = sheets["profiles"][["customer_id", "gender"]].copy()
        df_profiles = df_profiles.merge(df_gender, on="customer_id", how="left")
        df_profiles["gender"] = df_profiles["gender"].fillna("F")

    print(f"  ✓ {len(df_profiles)} khách | {len(df_inter)} interactions | {len(df_purch)} purchases")

    # ── Step 2: LEP ────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[Step 2] LIFE EVENT PREDICTION")
    print("─" * 70)
    lep = LEPModel(n_estimators=100)
    lep.train(df_profiles, df_ml, verbose=True)
    lep_preds = lep.predict(df_profiles)

    dist = lep_preds["predicted_intent"].value_counts()
    print("\n  Intent distribution:")
    for intent, n in dist.items():
        bar = "█" * n
        print(f"    {intent:15s} {n:3d}  {bar}")

    print("\n  Top 8 predictive features:")
    fi = get_feature_importance(lep, top_n=8)
    for _, r in fi.iterrows():
        bar = "▓" * int(r["importance"] * 80)
        print(f"    {r['feature']:35s} {r['importance']:.4f}  {bar}")

    # ── Step 3: NBA + LLM (with Instore Context) ──────────────────────────────
    print("\n" + "─" * 70)
    print("[Step 3] NEXT-BEST-ACTION + LLM (INSIGHT-DRIVEN)")
    print("─" * 70)

    engine     = NBAEngineLLM(api_key=api_key or None, use_cache=True)
    nba_result = engine.generate_actions_llm(
        lep_predictions = lep_preds,
        df_profiles     = df_profiles,
        instore_cache   = instore_cache,
        verbose         = True,
    )
    engine.print_message_report(nba_result)

    # ── Step 4: Feedback Loop ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[Step 4] FEEDBACK LOOP")
    print("─" * 70)
    fb   = FeedbackProcessor()
    fb.process_interactions(df_inter)
    best = fb.get_best_channel_per_customer()
    camp = fb.get_campaign_performance(df_inter)

    print("\n  Best channel per customer:")
    print(best.to_string(index=False))

    print("\n  Top 5 campaigns by reward:")
    print(camp.head(5)[["campaign", "total_reward", "roi", "unique_customers"]].to_string(index=False))

    # ── Step 5: Save JSON ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[Step 5] SAVING OUTPUT → JSON")
    print("─" * 70)
    save_output_json(nba_result, instore_cache, OUTPUT_JSON)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    allowed = nba_result[nba_result["rule_status"] == "allowed"]
    blocked = nba_result[nba_result["rule_status"] != "allowed"]
    llm_n   = (nba_result["message_source"] == "llm").sum()
    fb_n    = (nba_result["message_source"].str.startswith("fallback")).sum()
    insight_n = (nba_result["key_insight"].str.strip() != "").sum()

    print(f"  Customers processed : {len(nba_result)}")
    print(f"  Messages to send    : {len(allowed)}")
    print(f"  Blocked by rules    : {len(blocked)}")
    print(f"  LLM generated       : {llm_n}")
    print(f"  Fallback template   : {fb_n}")
    print(f"  With key insight    : {insight_n} / {len(nba_result)}")
    print(f"  Output file         : {OUTPUT_JSON}")
    print("=" * 70)

    return nba_result


if __name__ == "__main__":
    main()
