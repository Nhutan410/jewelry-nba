"""
src/pipeline.py
────────────────────────────────────────────────────────────────────────────
End-to-end pipeline: LEP → NBA → Feedback → Report

Hỗ trợ 3 chế độ vận hành:
  1. full_run (mặc định): Train mới từ đầu với toàn bộ dữ liệu gốc.
  2. predict_only         : Load model đã lưu → predict ngay (không train lại).
  3. retrain_with_new     : Train lại khi có file dữ liệu mới; có thể gộp
                            dữ liệu mới vào pool cũ (append_to_existing=True).

Cách dùng:
    python -m src.pipeline                          # full_run
    python -m src.pipeline --mode predict_only
    python -m src.pipeline --mode retrain_with_new --new-data path/to/new.xlsx
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.lep_pipeline      import LEPModel, get_feature_importance, DEFAULT_MODEL_DIR
from src.nba_engine        import NBAEngine
from src.nba_engine_llm    import NBAEngineLLM
from src.feedback_loop     import FeedbackProcessor
from src.instore_script_engine import InstoreScriptEngine


# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = ROOT / "data" / "customer_data_poc_enhanced.xlsx"
OUTPUTS_DIR  = ROOT / "outputs"
MODEL_PATH   = DEFAULT_MODEL_DIR / "lep_model.pkl"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_all_sheets(path: Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    print("=" * 60)
    print("JEWELRY NBA — END-TO-END PIPELINE")
    print("=" * 60)
    print(f"\n[Data] Loading from: {path}")

    sheets = pd.read_excel(path, sheet_name=None)
    for name, df in sheets.items():
        print(f"  ✓ {name:30s} → {len(df):4d} rows × {len(df.columns):2d} cols")
    return sheets


def load_new_data_sheets(path: Path) -> dict[str, pd.DataFrame]:
    """
    Load file dữ liệu mới (Excel).
    File mới cần có ít nhất sheet 'profiles_enhanced'.
    Các sheet khác (ml_predictions, interactions, purchases, nba_actions) là tuỳ chọn.
    """
    print(f"\n[Data] Loading NEW data from: {path}")
    sheets = pd.read_excel(path, sheet_name=None)
    for name, df in sheets.items():
        print(f"  ✓ [NEW] {name:30s} → {len(df):4d} rows × {len(df.columns):2d} cols")

    if "profiles_enhanced" not in sheets:
        raise ValueError(
            "File dữ liệu mới phải có sheet 'profiles_enhanced'. "
            f"Các sheet hiện có: {list(sheets.keys())}"
        )
    return sheets


# ── Step 1: LEP ───────────────────────────────────────────────────────────────
def run_lep_train(df_profiles: pd.DataFrame,
                  df_ml: pd.DataFrame | None = None,
                  save_model: bool = True) -> tuple[LEPModel, pd.DataFrame]:
    """Train LEP model từ đầu và predict."""
    print("\n" + "─" * 60)
    print("STEP 1 — LIFE EVENT PREDICTION (LEP) — TRAIN MODE")
    print("─" * 60)

    model = LEPModel(n_estimators=100)
    metrics = model.train(df_profiles, df_ml, verbose=True)

    preds = model.predict(df_profiles)
    _print_lep_results(preds, model)

    if save_model:
        model.save()

    return model, preds


def run_lep_predict_only(df_profiles: pd.DataFrame,
                          model_path: Path | None = None) -> tuple[LEPModel, pd.DataFrame]:
    """
    Load model đã lưu và predict — không train lại.
    Dùng khi chỉ có dữ liệu mới cần dự đoán mà không muốn train lại.
    """
    print("\n" + "─" * 60)
    print("STEP 1 — LIFE EVENT PREDICTION (LEP) — PREDICT ONLY MODE")
    print("─" * 60)

    model = LEPModel.load(model_path)
    preds = model.predict(df_profiles)
    _print_lep_results(preds, model)
    return model, preds


def run_lep_retrain(df_profiles: pd.DataFrame,
                    df_new_profiles: pd.DataFrame,
                    df_ml: pd.DataFrame | None = None,
                    df_new_ml: pd.DataFrame | None = None,
                    append_to_existing: bool = True,
                    model_path: Path | None = None,
                    save_model: bool = True) -> tuple[LEPModel, pd.DataFrame]:
    """
    Retrain LEP khi có dữ liệu mới.

    Chiến lược:
      - append_to_existing=True  → gộp data mới vào pool đã train → train lại
      - append_to_existing=False → train lại chỉ với data mới (bỏ data cũ)

    Nếu chưa có model đã lưu → tự động fall back về train từ đầu với data gốc + mới.
    """
    print("\n" + "─" * 60)
    print("STEP 1 — LIFE EVENT PREDICTION (LEP) — RETRAIN WITH NEW DATA")
    print(f"        append_to_existing={append_to_existing}")
    print("─" * 60)

    try:
        model = LEPModel.load(model_path)
    except FileNotFoundError:
        print("[LEP] Chưa có model đã lưu → train từ đầu với toàn bộ dữ liệu.")
        # Gộp data gốc + mới nếu có
        if df_profiles is not None and len(df_profiles) > 0:
            df_combined = pd.concat([df_profiles, df_new_profiles], ignore_index=True)
            df_ml_combined = None
            if df_ml is not None and df_new_ml is not None:
                df_ml_combined = pd.concat([df_ml, df_new_ml], ignore_index=True)
            elif df_ml is not None:
                df_ml_combined = df_ml
            elif df_new_ml is not None:
                df_ml_combined = df_new_ml
        else:
            df_combined = df_new_profiles
            df_ml_combined = df_new_ml

        model = LEPModel(n_estimators=100)
        model.train(df_combined, df_ml_combined, verbose=True)
    else:
        print(f"[LEP] Đã load model. Bắt đầu retrain với {len(df_new_profiles)} bản ghi mới.")
        model.retrain(df_new_profiles, df_new_ml,
                      append_to_existing=append_to_existing, verbose=True)

    # Predict trên data mới
    preds_new = model.predict(df_new_profiles)
    print(f"\n[LEP] Predictions cho {len(df_new_profiles)} khách MỚI:")
    _print_lep_results(preds_new, model)

    if save_model:
        model.save()

    return model, preds_new


def _print_lep_results(preds: pd.DataFrame, model: LEPModel):
    """In kết quả LEP prediction."""
    print("\n[LEP] Predictions:")
    print(preds[["customer_id", "predicted_intent", "confidence", "priority"]].to_string(index=False))

    dist = preds["predicted_intent"].value_counts()
    print("\n[LEP] Intent distribution:")
    for intent, count in dist.items():
        bar = "█" * count
        print(f"  {intent:20s} {count:3d} {bar}")

    fi = get_feature_importance(model, top_n=10)
    print("\n[LEP] Top 10 features:")
    for _, row in fi.iterrows():
        bar = "▓" * int(row["importance"] * 100)
        print(f"  {row['feature']:35s} {row['importance']:.4f} {bar}")


# ── Step 2: NBA ───────────────────────────────────────────────────────────────
def run_nba(preds: pd.DataFrame, df_profiles: pd.DataFrame,
            use_llm: bool = False) -> pd.DataFrame:
    print("\n" + "─" * 60)
    print("STEP 2 — NEXT-BEST-ACTION (NBA)" + (" [LLM MODE]" if use_llm else ""))
    print("─" * 60)

    if use_llm:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[NBA-LLM] Không có OPENAI_API_KEY → dùng fallback template.")
            print("[NBA-LLM] Để dùng LLM: set OPENAI_API_KEY=sk-...")
        engine = NBAEngineLLM(api_key=api_key or None, use_cache=True)
        nba_df = engine.generate_actions_llm(preds, df_profiles, verbose=True)
        engine.print_message_report(nba_df)
    else:
        engine = NBAEngine()
        nba_df = engine.generate_actions(preds, df_profiles)

        print("\n[NBA] Action plan:")
        print(nba_df[["customer_id", "predicted_intent", "confidence",
                      "priority", "channel", "campaign_id", "rule_status"]].to_string(index=False))

    summary = engine.get_action_summary(nba_df)
    print(f"\n[NBA] Summary:")
    print(f"  Total customers  : {summary['total_customers']}")
    print(f"  Actions allowed  : {summary['actions_allowed']}")
    print(f"  Actions blocked  : {summary['actions_blocked']}")
    print(f"  By intent        : {summary['by_intent']}")
    print(f"  By channel       : {summary['by_channel']}")
    print(f"  By priority      : {summary['by_priority']}")

    return nba_df


# ── Step 3: Feedback ──────────────────────────────────────────────────────────
def run_feedback(df_interactions: pd.DataFrame,
                 df_purchases: pd.DataFrame) -> FeedbackProcessor:
    print("\n" + "─" * 60)
    print("STEP 3 — FEEDBACK LOOP ANALYSIS")
    print("─" * 60)

    fb = FeedbackProcessor()
    perf = fb.process_interactions(df_interactions)

    best = fb.get_best_channel_per_customer()
    print("\n[Feedback] Best channel per customer:")
    print(best.to_string(index=False))

    funnel = fb.get_conversion_funnel(df_interactions, df_purchases)
    print("\n[Feedback] Conversion funnel by channel:")
    print(funnel.to_string(index=False))

    camp = fb.get_campaign_performance(df_interactions)
    print("\n[Feedback] Top campaigns by reward:")
    print(camp.head(8).to_string(index=False))

    return fb


# ── Step 4: Save outputs ──────────────────────────────────────────────────────
def save_outputs(preds: pd.DataFrame, nba_df: pd.DataFrame,
                 fb: FeedbackProcessor,
                 df_interactions: pd.DataFrame,
                 df_purchases: pd.DataFrame,
                 suffix: str = "") -> str:
    print("\n" + "─" * 60)
    print("STEP 4 — SAVING OUTPUTS")
    print("─" * 60)

    fname = f"nba_results{suffix}.xlsx"
    out_path = OUTPUTS_DIR / fname
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        preds.to_excel(writer, sheet_name="lep_predictions", index=False)
        nba_df.to_excel(writer, sheet_name="nba_actions_plan", index=False)

        if fb.channel_perf is not None:
            fb.channel_perf.to_excel(writer, sheet_name="channel_performance", index=False)

        best = fb.get_best_channel_per_customer()
        best.to_excel(writer, sheet_name="best_channel", index=False)

        funnel = fb.get_conversion_funnel(df_interactions, df_purchases)
        funnel.to_excel(writer, sheet_name="conversion_funnel", index=False)

        camp = fb.get_campaign_performance(df_interactions)
        camp.to_excel(writer, sheet_name="campaign_performance", index=False)

        summary_rows = []
        for _, row in nba_df.iterrows():
            # Lấy message từ cột đúng tuỳ engine (template vs LLM)
            if "llm_body" in nba_df.columns:
                msg_text = str(row.get("llm_body", ""))
                cta_text = str(row.get("llm_cta", row.get("cta", "")))
            else:
                msg_text = str(row.get("message", ""))
                cta_text = str(row.get("cta", ""))

            msg_preview = msg_text[:80] + "..." if len(msg_text) > 80 else msg_text

            entry = {
                "customer_id":         row["customer_id"],
                "intent":              row["predicted_intent"],
                "confidence":          row["confidence"],
                "priority":            row["priority"],
                "recommended_channel": row["channel"],
                "campaign":            row["campaign_id"],
                "message_preview":     msg_preview,
                "product_focus":       row.get("product_focus", ""),
                "cta":                 cta_text,
                "action_status":       row["rule_status"],
            }

            # Thêm cột LLM nếu có
            if "llm_body" in nba_df.columns:
                entry["llm_subject"]      = row.get("llm_subject", "")
                entry["tone"]             = row.get("tone", "")
                entry["tokens_used"]      = row.get("tokens_used", 0)
                entry["message_source"]   = row.get("message_source", "")

            summary_rows.append(entry)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="final_summary", index=False)

    print(f"\n[Output] Saved → {out_path}")
    return str(out_path)


# ── Main orchestrators ────────────────────────────────────────────────────────
def run_instore(preds: pd.DataFrame,
                df_profiles: pd.DataFrame,
                api_key: str = "") -> pd.DataFrame:
    """Nhánh 2 (In-store): sinh Sales Script 5 bước cho TVV."""
    print("\n" + "─" * 60)
    print("NHÁNH 2 — IN-STORE SALES SCRIPT GENERATION")
    print("─" * 60)

    from src.pipeline_instore import step_save as instore_step_save

    engine = InstoreScriptEngine(api_key=api_key or None, use_cache=True)
    instore_df = engine.generate_scripts(preds, df_profiles, verbose=True)
    engine.print_script_report(instore_df)

    # Save riêng file instore
    from src.lep_pipeline import LEPModel, DEFAULT_MODEL_DIR
    try:
        _lep = LEPModel.load(DEFAULT_MODEL_DIR / "lep_model.pkl")
    except Exception:
        _lep = None

    if _lep is not None:
        instore_step_save(lep_preds=preds, instore_df=instore_df, lep_model=_lep)
    else:
        # Lưu tối giản khi không load được model
        out = OUTPUTS_DIR / "nba_results_instore.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            preds.to_excel(w, sheet_name="lep_predictions", index=False)
            instore_df.to_excel(w, sheet_name="instore_scripts_full", index=False)
        print(f"[InStore] Saved (compact) → {out}")

    return instore_df


def main_full_run(use_llm: bool = False, run_instore_branch: bool = False):
    """Chế độ mặc định: train từ đầu với dữ liệu gốc."""
    os.chdir(ROOT)
    sheets          = load_all_sheets()
    df_profiles     = sheets["profiles_enhanced"]
    df_interactions = sheets["interactions"]
    df_purchases    = sheets["purchases"]
    df_ml           = sheets.get("ml_predictions", pd.DataFrame())

    model, preds    = run_lep_train(df_profiles, df_ml, save_model=True)
    nba_df          = run_nba(preds, df_profiles, use_llm=use_llm)
    fb              = run_feedback(df_interactions, df_purchases)
    out_path        = save_outputs(preds, nba_df, fb, df_interactions, df_purchases)

    if run_instore_branch:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        run_instore(preds, df_profiles, api_key=api_key)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE (full_run)")
    print(f"Output: {out_path}")
    print("=" * 60)
    return preds, nba_df, fb


def main_predict_only(new_data_path: Path | None = None,
                      use_llm: bool = False,
                      run_instore_branch: bool = False):
    """
    Chế độ predict_only: load model đã lưu → predict trên dữ liệu mới.
    Không train lại. Cần đã có file model được lưu trước đó.

    Args:
        new_data_path: Nếu None → dùng file dữ liệu gốc.
    """
    os.chdir(ROOT)

    if new_data_path and Path(new_data_path).exists():
        sheets = load_new_data_sheets(Path(new_data_path))
    else:
        sheets = load_all_sheets()

    df_profiles     = sheets["profiles_enhanced"]
    df_interactions = sheets.get("interactions", pd.DataFrame())
    df_purchases    = sheets.get("purchases", pd.DataFrame())

    model, preds    = run_lep_predict_only(df_profiles)
    nba_df          = run_nba(preds, df_profiles, use_llm=use_llm)

    out_path = None
    if not df_interactions.empty and not df_purchases.empty and not nba_df.empty:
        fb = run_feedback(df_interactions, df_purchases)
        out_path = save_outputs(preds, nba_df, fb, df_interactions, df_purchases,
                                suffix="_predict_only")

    if run_instore_branch:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        run_instore(preds, df_profiles, api_key=api_key)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE (predict_only)")
    if out_path:
        print(f"Output: {out_path}")
    print("=" * 60)
    return preds, nba_df


def main_retrain_with_new(new_data_path: Path,
                           append_to_existing: bool = True,
                           use_llm: bool = False,
                           run_instore_branch: bool = False):
    """
    Chế độ retrain_with_new: nhận dữ liệu mới → retrain/update model → predict.

    Args:
        new_data_path     : Đường dẫn file Excel mới (bắt buộc có sheet 'profiles_enhanced')
        append_to_existing: True → gộp data mới vào pool cũ; False → train lại hoàn toàn.
    """
    os.chdir(ROOT)

    # Load dữ liệu gốc (dùng để fallback khi chưa có model)
    base_sheets     = load_all_sheets()
    df_profiles     = base_sheets["profiles_enhanced"]
    df_interactions = base_sheets["interactions"]
    df_purchases    = base_sheets["purchases"]
    df_ml           = base_sheets.get("ml_predictions")

    # Load dữ liệu mới
    new_sheets      = load_new_data_sheets(Path(new_data_path))
    df_new_profiles = new_sheets["profiles_enhanced"]
    df_new_ml       = new_sheets.get("ml_predictions")

    # Retrain + predict trên data mới
    model, preds    = run_lep_retrain(
        df_profiles=df_profiles,
        df_new_profiles=df_new_profiles,
        df_ml=df_ml,
        df_new_ml=df_new_ml,
        append_to_existing=append_to_existing,
        save_model=True,
    )

    # NBA + Feedback dùng data gốc (hoặc merge nếu có thêm interactions mới)
    df_interactions_combined = df_interactions
    df_purchases_combined = df_purchases
    if "interactions" in new_sheets:
        df_interactions_combined = pd.concat(
            [df_interactions, new_sheets["interactions"]], ignore_index=True)
    if "purchases" in new_sheets:
        df_purchases_combined = pd.concat(
            [df_purchases, new_sheets["purchases"]], ignore_index=True)

    nba_df  = run_nba(preds, df_new_profiles, use_llm=use_llm)
    fb      = run_feedback(df_interactions_combined, df_purchases_combined)
    out_path = save_outputs(preds, nba_df, fb,
                            df_interactions_combined, df_purchases_combined,
                            suffix="_retrained")

    if run_instore_branch:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        run_instore(preds, df_new_profiles, api_key=api_key)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE (retrain_with_new)")
    print(f"Output: {out_path}")
    print("=" * 60)
    return preds, nba_df, fb


# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Jewelry NBA Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python -m src.pipeline                                              # Nhánh 1: Train + template
  python -m src.pipeline --llm                                        # Nhánh 1: Train + LLM
  python -m src.pipeline --instore                                    # Nhánh 2: In-store scripts
  python -m src.pipeline --llm --instore                              # Cả 2 nhánh
  python -m src.pipeline --mode predict_only --llm --instore          # Predict + cả 2 nhánh
  python -m src.pipeline --mode predict_only --new-data new.xlsx      # Predict dữ liệu mới
  python -m src.pipeline --mode retrain_with_new --new-data new.xlsx  # Retrain + gộp
        """,
    )
    parser.add_argument(
        "--mode", choices=["full_run", "predict_only", "retrain_with_new"],
        default="full_run",
        help="Chế độ chạy pipeline (mặc định: full_run)",
    )
    parser.add_argument(
        "--new-data", type=str, default=None,
        help="Đường dẫn file Excel mới (dùng với predict_only hoặc retrain_with_new)",
    )
    parser.add_argument(
        "--no-append", action="store_true",
        help="Với retrain_with_new: train lại chỉ với data mới, không gộp pool cũ",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Dùng LLM cho Nhánh 1 (Online) — sinh message cá nhân hoá. "
             "Yêu cầu OPENAI_API_KEY trong biến môi trường.",
    )
    parser.add_argument(
        "--instore", action="store_true",
        help="Chạy Nhánh 2 (In-store) — sinh Sales Script 5 bước cho TVV. "
             "Dùng LLM nếu có OPENAI_API_KEY, fallback template nếu không.",
    )

    args = parser.parse_args()

    if args.mode == "full_run":
        main_full_run(use_llm=args.llm, run_instore_branch=args.instore)

    elif args.mode == "predict_only":
        main_predict_only(new_data_path=args.new_data,
                          use_llm=args.llm,
                          run_instore_branch=args.instore)

    elif args.mode == "retrain_with_new":
        if not args.new_data:
            parser.error("--mode retrain_with_new yêu cầu --new-data <path>")
        main_retrain_with_new(
            new_data_path=Path(args.new_data),
            append_to_existing=not args.no_append,
            use_llm=args.llm,
            run_instore_branch=args.instore,
        )


if __name__ == "__main__":
    main()



"""
Ví dụ:
  python -m src.pipeline                                              # Train + template message
  python -m src.pipeline --llm                                        # Train + LLM message
  python -m src.pipeline --mode predict_only --llm                    # Predict + LLM
  python -m src.pipeline --mode predict_only --new-data new.xlsx      # Predict dữ liệu mới
  python -m src.pipeline --mode retrain_with_new --new-data new.xlsx  # Retrain + gộp
"""