"""
generate_instore_scripts.py
────────────────────────────────────────────────────────────────────────────
Script sinh lại file outputs/instore_scripts.json từ Excel data.

Cách dùng:
    # Sinh toàn bộ (xoá cache cũ, chạy lại từ đầu)
    python generate_instore_scripts.py

    # Chỉ thêm khách mới chưa có trong cache
    python generate_instore_scripts.py --incremental

    # Dùng GPT-4o để tạo script (cần OPENAI_API_KEY)
    python generate_instore_scripts.py --api-key sk-...

    # Hoặc set biến môi trường rồi chạy
    export OPENAI_API_KEY=sk-...
    python generate_instore_scripts.py
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.lep_pipeline import LEPModel, DEFAULT_MODEL_DIR
from src.instore_script_engine import InstoreScriptEngine

DATA_PATH  = ROOT / "data" / "customer_data_poc_enhanced.xlsx"
CACHE_FILE = ROOT / "outputs" / "instore_scripts.json"
MODEL_PATH = DEFAULT_MODEL_DIR / "lep_model.pkl"


def load_cache() -> dict | None:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def run(incremental: bool = False, api_key: str = "") -> None:
    if not DATA_PATH.exists():
        print(f"[ERROR] Không tìm thấy file Excel: {DATA_PATH}")
        sys.exit(1)

    print(f"[1/4] Đọc dữ liệu: {DATA_PATH.name}")
    sheets      = pd.read_excel(DATA_PATH, sheet_name=None)
    df_profiles = sheets["profiles_enhanced"]
    df_ml       = sheets.get("ml_predictions", pd.DataFrame())

    if "profiles" in sheets:
        df_gender   = sheets["profiles"][["customer_id", "gender"]].copy()
        df_profiles = df_profiles.merge(df_gender, on="customer_id", how="left")
        df_profiles["gender"] = df_profiles["gender"].fillna("F")

    id_col = "c" if "c" in df_profiles.columns else "customer_id"
    df_profiles["customer_id"] = df_profiles[id_col].astype(str)

    all_ids = df_profiles["customer_id"].tolist()

    existing_cache = load_cache() if incremental else None
    if incremental and existing_cache:
        cached_ids  = {c["customer_id"] for c in existing_cache.get("customers", [])}
        new_ids     = [cid for cid in all_ids if cid not in cached_ids]
        if not new_ids:
            print("[OK] Không có khách mới. Cache đã đầy đủ.")
            return
        print(f"[2/4] Chế độ incremental — {len(new_ids)} khách mới / {len(all_ids)} tổng")
        df_subset    = df_profiles[df_profiles["customer_id"].isin(new_ids)].copy()
        df_ml_subset = (
            df_ml[df_ml[df_ml.columns[0]].astype(str).isin(new_ids)].copy()
            if not df_ml.empty else pd.DataFrame()
        )
    else:
        print(f"[2/4] Chế độ full — {len(all_ids)} khách hàng")
        df_subset    = df_profiles.copy()
        df_ml_subset = df_ml.copy()

    print("[3/4] Train / load mô hình LEP...")
    if MODEL_PATH.exists() and incremental:
        lep = LEPModel.load(MODEL_PATH)
    else:
        lep = LEPModel(n_estimators=100)
        lep.train(df_profiles, df_ml, verbose=True)
        lep.save()

    lep_preds = lep.predict(df_subset)

    resolved_key = api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip() or None
    mode = "GPT-4o" if resolved_key else "template fallback"
    print(f"[4/4] Sinh scripts ({mode}) cho {len(df_subset)} khách...")

    engine    = InstoreScriptEngine(api_key=resolved_key, use_cache=True)
    instore_df = engine.generate_scripts(lep_preds, df_subset, verbose=True)

    new_entries = []
    for _, row in instore_df.iterrows():
        cid         = str(row["customer_id"])
        profile_row = df_profiles[df_profiles["customer_id"] == cid]

        entry = {
            "customer_id":  cid,
            "processed_at": datetime.now().isoformat(),
            "profile": {
                "segment_rfm_tier":   str(row.get("segment_rfm_tier", "")),
                "budget":             str(row.get("budget", "")),
                "style":              str(row.get("style", "")),
                "preferred_type":     str(row.get("preferred_type", "")),
                "material":           str(row.get("material", "")),
                "recency_days":       int(row.get("recency_days", 0)),
                "monetary":           float(row.get("monetary", 0)),
                "avg_discount_pct":   float(row.get("avg_discount_pct", 0)),
                "web_pdp_views":      int(row.get("web_pdp_views", 0)),
                "add_to_cart":        int(row.get("add_to_cart", 0)),
                "visit_count":        int(row.get("visit_count", 0)),
                "birthday_in_days":   int(row.get("birthday_in_days", 365)),
                "sig_view_ring":      int(row.get("sig_view_ring", 0)),
                "sig_view_diamond":   int(row.get("sig_view_diamond", 0)),
                "sig_search_propose": int(row.get("sig_search_propose", 0)),
            },
            "lep": {
                "predicted_intent": str(row.get("lep_intent", "")),
                "confidence":       float(row.get("confidence", 0)),
                "priority":         str(row.get("priority", "low")),
            },
            "instore": {
                "instore_intent":     str(row.get("instore_intent", "")),
                "nba_strategy":       str(row.get("nba_strategy", "")),
                "psychology_trigger": str(row.get("psychology_trigger", "")),
                "product_focus":      str(row.get("product_focus", "")),
                "product_rec_1":      str(row.get("product_rec_1", "")),
                "product_rec_2":      str(row.get("product_rec_2", "")),
                "product_rec_3":      str(row.get("product_rec_3", "")),
                "online_insight":     str(row.get("online_insight", "")),
                "urgency_signal":     str(row.get("urgency_signal", "")),
                "key_insight":        str(row.get("key_insight", "")),
                "script": {
                    "opening":   str(row.get("script_opening", "")),
                    "khai_thac": str(row.get("script_khai_thac", "")),
                    "goi_y":     str(row.get("script_goi_y", "")),
                    "chot":      str(row.get("script_chot", "")),
                    "upsell":    str(row.get("script_upsell", "")),
                },
                "script_source": str(row.get("script_source", "fallback")),
                "tokens_used":   int(row.get("tokens_used", 0)),
            },
        }
        new_entries.append(entry)

    if existing_cache and existing_cache.get("customers"):
        existing_ids = {c["customer_id"] for c in existing_cache["customers"]}
        merged = existing_cache["customers"] + [
            e for e in new_entries if e["customer_id"] not in existing_ids
        ]
    else:
        merged = new_entries

    cache = {
        "version":          "2.0",
        "generated_at":     datetime.now().isoformat(),
        "excel_path":       str(DATA_PATH),
        "total_customers":  len(merged),
        "customers":        merged,
    }
    save_cache(cache)
    print(f"\n[DONE] Đã lưu {len(merged)} khách vào {CACHE_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh lại instore_scripts.json")
    parser.add_argument(
        "--incremental", action="store_true",
        help="Chỉ xử lý khách mới chưa có trong cache (mặc định: chạy lại toàn bộ)",
    )
    parser.add_argument(
        "--api-key", default="",
        help="OpenAI API key để dùng GPT-4o (hoặc set OPENAI_API_KEY trong .env)",
    )
    args = parser.parse_args()
    run(incremental=args.incremental, api_key=args.api_key)
