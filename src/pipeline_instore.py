"""
src/pipeline_instore.py
────────────────────────────────────────────────────────────────────────────
In-Store NBA Pipeline — Nhánh 2

Flow:
  Step 1: Load data (Excel)
  Step 2: LEP — train hoặc predict_only
  Step 3: InstoreScriptEngine → classify intent + sinh script 5 bước
  Step 4: Save nba_results_instore.xlsx với đầy đủ thông tin TVV

Output file: outputs/nba_results_instore.xlsx
Sheets:
  - lep_predictions       : Toàn bộ LEP output
  - instore_scripts_full  : Đầy đủ thông tin + script (view cho manager)
  - tvv_quick_view        : View gọn cho TVV dùng trên app/tablet
  - intent_distribution   : Phân phối instore intent types
  - feature_importance    : Top features của LEP model
  - run_summary           : Tóm tắt run

Cách chạy:
  python -m src.pipeline_instore                          # LLM mode (cần OPENAI_API_KEY)
  python -m src.pipeline_instore --no-llm                 # Fallback template mode
  python -m src.pipeline_instore --predict-only           # Dùng model đã lưu, không train
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.lep_pipeline        import LEPModel, get_feature_importance, DEFAULT_MODEL_DIR
from src.instore_script_engine import InstoreScriptEngine


# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = ROOT / "data" / "customer_data_poc_enhanced.xlsx"
OUTPUT_PATH = ROOT / "outputs" / "nba_results_instore.xlsx"
MODEL_PATH  = DEFAULT_MODEL_DIR / "lep_model.pkl"
OUTPUT_PATH.parent.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Step functions
# ══════════════════════════════════════════════════════════════════════════════

def step_load_data(path: Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    print("=" * 70)
    print("JEWELRY NBA — IN-STORE PIPELINE (Nhánh 2)")
    print("=" * 70)
    print(f"\n[Step 1] Loading data: {path}")

    sheets = pd.read_excel(path, sheet_name=None)
    for name, df in sheets.items():
        print(f"  ✓ {name:35s} → {len(df):4d} rows × {len(df.columns):2d} cols")
    return sheets


def step_lep(df_profiles: pd.DataFrame,
             df_ml: pd.DataFrame,
             predict_only: bool = False) -> tuple[LEPModel, pd.DataFrame]:
    print("\n" + "─" * 70)
    mode_label = "PREDICT ONLY" if predict_only else "TRAIN"
    print(f"[Step 2] LEP — Life Event Prediction ({mode_label})")
    print("─" * 70)

    if predict_only:
        try:
            lep = LEPModel.load(MODEL_PATH)
            print("[LEP] Model đã load từ file.")
        except FileNotFoundError:
            print("[LEP] Chưa có model → train từ đầu.")
            lep = LEPModel(n_estimators=100)
            lep.train(df_profiles, df_ml, verbose=True)
            lep.save()
    else:
        lep = LEPModel(n_estimators=100)
        lep.train(df_profiles, df_ml, verbose=True)
        lep.save()

    preds = lep.predict(df_profiles)

    # Print distribution
    dist = preds["predicted_intent"].value_counts()
    print("\n[LEP] Intent distribution:")
    for intent, n in dist.items():
        bar = "█" * n
        print(f"  {intent:15s} {n:3d}  {bar}")

    # Top features
    print("\n[LEP] Top 8 predictive features:")
    fi = get_feature_importance(lep, top_n=8)
    for _, r in fi.iterrows():
        bar = "▓" * int(r["importance"] * 80)
        print(f"  {r['feature']:35s} {r['importance']:.4f}  {bar}")

    return lep, preds


def step_instore(lep_preds: pd.DataFrame,
                 df_profiles: pd.DataFrame,
                 api_key: str = "",
                 use_llm: bool = True) -> pd.DataFrame:
    print("\n" + "─" * 70)
    print("[Step 3] IN-STORE SCRIPT GENERATION")
    print("─" * 70)

    engine = InstoreScriptEngine(
        api_key=api_key or None if use_llm else None,
        use_cache=True,
    )
    result = engine.generate_scripts(lep_preds, df_profiles, verbose=True)
    engine.print_script_report(result)
    return result


def step_save(lep_preds: pd.DataFrame,
              instore_df: pd.DataFrame,
              lep_model: LEPModel,
              output_path: Path = OUTPUT_PATH) -> str:
    print("\n" + "─" * 70)
    print("[Step 4] SAVING OUTPUTS")
    print("─" * 70)

    # ── TVV Quick View (gọn, dễ đọc trên tablet) ──────────────────────────────
    tvv_cols = [
        "customer_id",
        "segment_rfm_tier",
        "budget",
        "style",
        "preferred_type",
        "material",
        # Intent & Signal
        "instore_intent",
        "nba_strategy",
        "psychology_trigger",
        "priority",
        "confidence",
        "lep_intent",
        # Online insight
        "online_insight",
        "urgency_signal",
        "key_insight",
        # Products
        "product_focus",
        "product_rec_1",
        "product_rec_2",
        "product_rec_3",
        # Script
        "script_opening",
        "script_khai_thac",
        "script_goi_y",
        "script_chot",
        "script_upsell",
    ]
    tvv_view = instore_df[[c for c in tvv_cols if c in instore_df.columns]]

    # ── Intent distribution ────────────────────────────────────────────────────
    intent_dist = (
        instore_df.groupby("instore_intent")
        .agg(
            count=("customer_id", "count"),
            avg_confidence=("confidence", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
        .rename(columns={"instore_intent": "intent_type"})
    )
    intent_dist["avg_monetary"] = intent_dist["avg_monetary"].round(0)
    intent_dist["avg_confidence"] = intent_dist["avg_confidence"].round(3)

    # ── Run summary ────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    llm_n      = (instore_df["script_source"] == "llm").sum()
    fallback_n = instore_df["script_source"].str.startswith("fallback").sum()

    instore_intent_counts = instore_df["instore_intent"].value_counts().to_dict()
    summary = pd.DataFrame([{
        "run_at":             now_str,
        "total_customers":    len(instore_df),
        "llm_scripts":        int(llm_n),
        "fallback_scripts":   int(fallback_n),
        "total_tokens_used":  int(instore_df["tokens_used"].sum()),
        "high_purchase_count": instore_intent_counts.get("High Purchase", 0),
        "exploration_count":  instore_intent_counts.get("Exploration", 0),
        "premium_count":      instore_intent_counts.get("Premium", 0),
        "low_intent_count":   instore_intent_counts.get("Low Intent", 0),
        "lep_intent_dist":    str(lep_preds["predicted_intent"].value_counts().to_dict()),
        "output_file":        str(output_path),
    }])

    # ── Feature importance ─────────────────────────────────────────────────────
    fi_df = get_feature_importance(lep_model, top_n=20)

    # ── Write Excel ────────────────────────────────────────────────────────────
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        lep_preds.to_excel(writer, sheet_name="lep_predictions", index=False)
        instore_df.to_excel(writer, sheet_name="instore_scripts_full", index=False)
        tvv_view.to_excel(writer, sheet_name="tvv_quick_view", index=False)
        intent_dist.to_excel(writer, sheet_name="intent_distribution", index=False)
        fi_df.to_excel(writer, sheet_name="feature_importance", index=False)
        summary.to_excel(writer, sheet_name="run_summary", index=False)

    print(f"\n  ✓ Saved → {output_path}")
    print(f"  Sheets:")
    print(f"    lep_predictions       — {len(lep_preds)} rows")
    print(f"    instore_scripts_full  — {len(instore_df)} rows, {len(instore_df.columns)} cols")
    print(f"    tvv_quick_view        — {len(tvv_view)} rows (gọn cho TVV)")
    print(f"    intent_distribution   — {len(intent_dist)} intent types")
    print(f"    feature_importance    — top 20 features")
    print(f"    run_summary            — 1 row")

    return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main(predict_only: bool = False, use_llm: bool = True):
    os.chdir(ROOT)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if use_llm:
        if api_key:
            print(f"[Config] OpenAI API key: {'*' * 10}{api_key[-4:]}")
        else:
            print("[Config] Không có OPENAI_API_KEY → chạy fallback template mode")
            print("[Config] Để dùng LLM: export OPENAI_API_KEY='sk-...'")
    else:
        print("[Config] --no-llm flag → chạy fallback template mode")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    sheets      = step_load_data()
    df_profiles = sheets["profiles_enhanced"]
    df_ml       = sheets.get("ml_predictions", pd.DataFrame())

    # ── Step 2: LEP ───────────────────────────────────────────────────────────
    lep_model, lep_preds = step_lep(
        df_profiles=df_profiles,
        df_ml=df_ml,
        predict_only=predict_only,
    )

    # ── Step 3: Instore scripts ────────────────────────────────────────────────
    instore_df = step_instore(
        lep_preds=lep_preds,
        df_profiles=df_profiles,
        api_key=api_key,
        use_llm=use_llm,
    )

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    out_path = step_save(
        lep_preds=lep_preds,
        instore_df=instore_df,
        lep_model=lep_model,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("IN-STORE PIPELINE COMPLETE")
    print("=" * 70)

    instore_dist = instore_df["instore_intent"].value_counts()
    print(f"\n  Tổng khách xử lý   : {len(instore_df)}")
    print(f"  LLM scripts        : {(instore_df['script_source'] == 'llm').sum()}")
    print(f"  Fallback scripts   : {instore_df['script_source'].str.startswith('fallback').sum()}")
    print(f"\n  Phân phối Instore Intent:")
    for intent_type, count in instore_dist.items():
        bar = "█" * count
        print(f"    {intent_type:18s} {count:3d}  {bar}")
    print(f"\n  Output: {out_path}")
    print("=" * 70)

    return lep_preds, instore_df


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jewelry NBA — In-Store Pipeline (Nhánh 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python -m src.pipeline_instore                   # LLM mode (cần OPENAI_API_KEY)
  python -m src.pipeline_instore --no-llm           # Template fallback
  python -m src.pipeline_instore --predict-only     # Dùng model đã lưu
  python -m src.pipeline_instore --predict-only --no-llm
        """,
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Load model đã lưu, không train lại (nhanh hơn)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Dùng fallback template thay vì LLM (không cần API key)"
    )
    args = parser.parse_args()

    main(
        predict_only=args.predict_only,
        use_llm=not args.no_llm,
    )
