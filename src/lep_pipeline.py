"""
src/lep_pipeline.py
────────────────────────────────────────────────────────────────────────────
Life Event Prediction (LEP) Pipeline
Đọc profiles_enhanced + ml_predictions → tính feature → predict 4 intent

Hỗ trợ:
  - train()      : train mới từ đầu với dữ liệu cho trước
  - retrain()    : train lại khi có dữ liệu mới (thay thế model cũ)
  - predict()    : predict trên model đã train
  - save() / load() : lưu và tải model đã train (không cần train lại mỗi lần)
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import os
import pickle
import json
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


# ── Hằng số ──────────────────────────────────────────────────────────────────
INTENT_LABELS   = ["engagement", "anniversary", "self_reward", "gift"]
PROBA_COLS      = ["proba_self", "proba_ann", "proba_selfreward", "proba_gift"]

# Cột proba có trong profiles_enhanced (tên khác với PROBA_COLS ở ml_predictions)
# proba_eng → engagement, proba_ann → anniversary, proba_self → self_reward, proba_gift → gift
_PROFILES_PROBA_MAP = {
    "proba_eng":  "engagement",
    "proba_ann":  "anniversary",
    "proba_self": "self_reward",
    "proba_gift": "gift",
}

NUMERIC_FEATURES = [
    "recency_days", "frequency", "monetary", "online_share",
    "avg_unit_price", "avg_discount",
    "tp_web", "tp_app", "tp_email", "tp_zns", "tp_social_ads",
    "tp_crm", "tp_store", "tp_direct",
    "web_pdp_views", "add_to_cart",
    "email_open", "email_click", "zns_open", "zns_click",
    "ads_clicks", "ad_spend",
    "camp_engagement", "camp_anniversary", "camp_selfreward", "camp_birthday",
    "sig_birthday_in_days", "sig_view_diamond",
    "sig_view_engagement_ring", "sig_search_propose",
]

CAT_FEATURES = ["budget", "style", "preferred_type", "material", "segment_rfm_tier"]

# Thư mục mặc định lưu model
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "outputs" / "models"

# Tên cột customer ID ưu tiên tìm theo thứ tự này
_ID_CANDIDATES = ["c", "customer_id", "cust_id", "id"]


def _resolve_id_col(df: pd.DataFrame) -> str:
    """Tự động tìm cột customer ID trong DataFrame."""
    for col in _ID_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"Không tìm thấy cột customer ID. "
        f"Cần có một trong: {_ID_CANDIDATES}. "
        f"Các cột hiện có: {list(df.columns)}"
    )


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo derived features từ raw profiles."""
    d = df.copy()

    # Engagement rate per channel
    d["email_engagement_rate"] = np.where(
        d["tp_email"] > 0, d["email_open"] / d["tp_email"], 0)
    d["zns_engagement_rate"] = np.where(
        d["tp_zns"] > 0, d["zns_open"] / d["tp_zns"], 0)

    # Intent composite scores
    d["engagement_intent_score"] = (
        d["sig_view_engagement_ring"] * 3
        + d["sig_search_propose"] * 3
        + d["sig_view_diamond"] * 2
        + d["camp_engagement"]
        + (d["sig_birthday_in_days"] < 30).astype(int)
    )
    d["anniversary_intent_score"] = (
        d["camp_anniversary"] * 2
        + d["sig_birthday_in_days"].apply(lambda x: 2 if x < 45 else 0)
        + d["tp_store"]
    )
    d["self_reward_intent_score"] = (
        d["camp_selfreward"] * 2
        + d["add_to_cart"]
        + d["web_pdp_views"]
    )
    d["gift_intent_score"] = (
        d["camp_birthday"]
        + d["tp_crm"]
        + d["tp_store"]
    )

    # RFM derived
    d["spend_per_visit"] = np.where(
        (d["tp_web"] + d["tp_app"]) > 0,
        d["monetary"] / (d["tp_web"] + d["tp_app"]),
        0,
    )
    d["recency_bucket"] = pd.cut(
        d["recency_days"],
        bins=[0, 30, 60, 90, 180, 9999],
        labels=[4, 3, 2, 1, 0],
    ).astype(float)

    # Online vs offline affinity
    d["is_omnichannel"] = ((d["online_share"] > 0) & (d["online_share"] < 1)).astype(int)

    return d


def encode_categoricals(df: pd.DataFrame, fit: bool = True,
                         encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Label encode categorical columns."""
    d = df.copy()
    if encoders is None:
        encoders = {}
    for col in CAT_FEATURES:
        if col not in d.columns:
            continue
        if fit:
            le = LabelEncoder()
            d[col + "_enc"] = le.fit_transform(d[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            d[col + "_enc"] = d[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )
    return d, encoders


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Trả về danh sách feature columns để train/predict."""
    derived = [
        "email_engagement_rate", "zns_engagement_rate",
        "engagement_intent_score", "anniversary_intent_score",
        "self_reward_intent_score", "gift_intent_score",
        "spend_per_visit", "recency_bucket", "is_omnichannel",
    ]
    cat_enc = [c + "_enc" for c in CAT_FEATURES if c + "_enc" in df.columns]
    return NUMERIC_FEATURES + derived + cat_enc


# ── Helpers: tạo label thực từ dữ liệu ──────────────────────────────────────
def build_labels_from_data(df_profiles: pd.DataFrame,
                            df_ml: pd.DataFrame | None = None) -> pd.Series:
    """
    Tạo nhãn intent thực từ dữ liệu. Thứ tự ưu tiên:
      1. df_ml (ml_predictions) có PROBA_COLS  → intent có xác suất cao nhất
      2. profiles_enhanced có cột proba_eng/ann/self/gift → intent có xác suất cao nhất
      3. Cột 'intent' trực tiếp trong df_profiles
      4. Cột 'persona' trong df_profiles (lowest quality fallback)
    """
    persona_map = {
        "Người sắp cầu hôn":       "engagement",
        "Khách mua dịp kỷ niệm":   "anniversary",
        "Người thích tự thưởng":   "self_reward",
        "Người mua quà tặng":      "gift",
    }
    col_to_intent = dict(zip(PROBA_COLS, INTENT_LABELS))

    # ── Ưu tiên 1: df_ml với PROBA_COLS ──────────────────────────────────────
    if df_ml is not None and not df_ml.empty:
        id_col_profiles = _resolve_id_col(df_profiles)
        id_col_ml = _resolve_id_col(df_ml)
        merged = df_profiles[[id_col_profiles]].merge(
            df_ml[[id_col_ml] + PROBA_COLS],
            left_on=id_col_profiles, right_on=id_col_ml, how="left"
        )
        has_proba = merged[PROBA_COLS].notna().any(axis=1)

        labels = []
        for i, (_, row) in enumerate(merged.iterrows()):
            if has_proba.iloc[i]:
                dominant_col = merged.loc[row.name, PROBA_COLS].idxmax()
                labels.append(col_to_intent[dominant_col])
            else:
                profile_row = df_profiles.iloc[i]
                if "intent" in profile_row.index and pd.notna(profile_row["intent"]):
                    labels.append(profile_row["intent"])
                elif "persona" in profile_row.index:
                    labels.append(persona_map.get(profile_row["persona"], "gift"))
                else:
                    labels.append("gift")
        return pd.Series(labels, name="intent")

    # ── Ưu tiên 2: proba columns nhúng sẵn trong profiles_enhanced ───────────
    # (proba_eng, proba_ann, proba_self, proba_gift)
    _available_proba = {
        col: intent
        for col, intent in _PROFILES_PROBA_MAP.items()
        if col in df_profiles.columns
    }
    if len(_available_proba) >= 2:
        proba_df = df_profiles[list(_available_proba.keys())].copy()
        proba_df.columns = list(_available_proba.values())
        dominant = proba_df.idxmax(axis=1)
        # Với khách mới không có proba → fallback persona
        labels = []
        for i, (_, row) in enumerate(df_profiles.iterrows()):
            proba_vals = proba_df.iloc[i]
            if proba_vals.notna().any() and proba_vals.max() > 0:
                labels.append(dominant.iloc[i])
            elif "persona" in row.index:
                labels.append(persona_map.get(row["persona"], "gift"))
            else:
                labels.append("gift")
        return pd.Series(labels, name="intent")

    # ── Ưu tiên 3: cột 'intent' trực tiếp ────────────────────────────────────
    if "intent" in df_profiles.columns:
        return df_profiles["intent"].fillna("gift")

    # ── Ưu tiên 4 (thấp nhất): suy từ persona ────────────────────────────────
    if "persona" in df_profiles.columns:
        return df_profiles["persona"].map(persona_map).fillna("gift")

    raise ValueError(
        "Không thể xác định label. Dữ liệu cần có ít nhất một trong: "
        "df_ml (ml_predictions), cột proba_*, cột 'intent', hoặc cột 'persona'."
    )


# ── Model ─────────────────────────────────────────────────────────────────────
class LEPModel:
    """
    Multi-class classifier: predict dominant intent (engagement/anniversary/
    self_reward/gift) từ profile features.
    Dùng RandomForest.

    Workflow:
        # --- Lần đầu: train từ dữ liệu gốc ---
        model = LEPModel()
        metrics = model.train(df_profiles, df_ml)
        model.save()                          # Lưu model

        # --- Lần sau: load model đã lưu ---
        model = LEPModel.load()
        preds = model.predict(df_profiles_new)

        # --- Khi có dữ liệu mới: retrain ---
        model.retrain(df_new_profiles, df_new_ml)
        model.save()

        # --- Hoặc: thêm data mới vào pool hiện có rồi train lại ---
        model.retrain(df_extra, df_extra_ml, append_to_existing=True)
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self.encoders: dict = {}
        self.feature_cols: list[str] = []
        self.label_enc = LabelEncoder()
        self.is_trained = False

        # Metadata về quá trình train
        self.train_metadata: dict = {}

        # Lưu trữ dữ liệu training để hỗ trợ append_to_existing
        self._train_X: pd.DataFrame | None = None
        self._train_y: np.ndarray | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare_features(self, df_profiles: pd.DataFrame,
                           fit_encoders: bool) -> pd.DataFrame:
        """Feature engineering + encoding."""
        df = engineer_features(df_profiles)
        df, self.encoders = encode_categoricals(df, fit=fit_encoders,
                                                encoders=self.encoders)
        if fit_encoders:
            self.feature_cols = get_feature_cols(df)
        return df

    def _fit_classifier(self, X: pd.DataFrame, y: np.ndarray,
                         verbose: bool) -> dict:
        """Fit và trả về metrics."""
        if len(X) < 4:
            self.clf.fit(X, y)
            if verbose:
                print(f"[LEP] Trained on full dataset ({len(X)} samples — too few to split).")
            return {"note": "trained_on_full_dataset", "n_samples": len(X)}

        # Tắt stratify nếu bất kỳ class nào có < 2 mẫu (sklearn yêu cầu tối thiểu 2)
        from collections import Counter
        class_counts = Counter(y)
        min_class_count = min(class_counts.values()) if class_counts else 0
        stratify = y if (len(set(y)) > 1 and min_class_count >= 2) else None
        if stratify is None and len(set(y)) > 1:
            if verbose:
                print(f"[LEP] WARNING: Một số class chỉ có 1 mẫu "
                      f"({dict(class_counts)}) — tắt stratify để tránh lỗi split.")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        self.clf.fit(X_tr, y_tr)
        y_pred = self.clf.predict(X_te)
        report = classification_report(
            y_te, y_pred,
            target_names=self.label_enc.classes_,
            output_dict=True,
            zero_division=0,
        )
        if verbose:
            print(f"[LEP] Trained. n={len(X)}, test accuracy={report['accuracy']:.3f}")
            print(f"      Classes: {list(self.label_enc.classes_)}")
        return report

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, df_profiles: pd.DataFrame,
              df_ml: pd.DataFrame | None = None,
              verbose: bool = True) -> dict:
        """
        Train model từ đầu.
        Label được suy tự động từ dữ liệu thực (df_ml hoặc persona/intent column).
        Không có giá trị nào bị hard-code.

        Args:
            df_profiles : DataFrame profiles_enhanced (bắt buộc có cột 'c')
            df_ml       : DataFrame ml_predictions (tuỳ chọn, ưu tiên hơn persona)
            verbose     : In thông tin training

        Returns:
            dict metrics (classification report hoặc note)
        """
        if verbose:
            print(f"[LEP] train() — n_profiles={len(df_profiles)}, "
                  f"has_ml_predictions={'yes' if df_ml is not None else 'no'}")

        df = self._prepare_features(df_profiles, fit_encoders=True)

        # Xây dựng label thực từ dữ liệu (không hard-code)
        y_str = build_labels_from_data(df_profiles, df_ml)
        y = self.label_enc.fit_transform(y_str.fillna("gift"))

        X = df[self.feature_cols].fillna(0)

        # Lưu training pool để hỗ trợ retrain với append
        self._train_X = X.copy()
        self._train_y = y.copy()

        metrics = self._fit_classifier(X, y, verbose)
        self.is_trained = True
        self.train_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "label_source": "ml_predictions" if df_ml is not None else "persona/intent",
            "classes": list(self.label_enc.classes_),
            "metrics": metrics,
        }
        return metrics

    def retrain(self, df_new_profiles: pd.DataFrame,
                df_new_ml: pd.DataFrame | None = None,
                append_to_existing: bool = True,
                verbose: bool = True) -> dict:
        """
        Train lại model khi có dữ liệu mới.

        Args:
            df_new_profiles    : DataFrame profiles MỚI (bắt buộc có cột 'c')
            df_new_ml          : DataFrame ml_predictions MỚI (tuỳ chọn)
            append_to_existing : True → gộp dữ liệu mới vào pool cũ rồi train lại.
                                 False → train lại chỉ với dữ liệu mới (thay thế hoàn toàn).
            verbose            : In thông tin

        Returns:
            dict metrics
        """
        if verbose:
            print(f"[LEP] retrain() — n_new={len(df_new_profiles)}, "
                  f"append={append_to_existing}, "
                  f"has_ml_predictions={'yes' if df_new_ml is not None else 'no'}")

        # Feature engineering trên data mới (dùng encoders đã fit nếu append)
        df_new = self._prepare_features(df_new_profiles,
                                         fit_encoders=not append_to_existing)

        y_str_new = build_labels_from_data(df_new_profiles, df_new_ml)
        y_new = self.label_enc.transform(
            y_str_new.fillna("gift").apply(
                lambda x: x if x in self.label_enc.classes_ else "gift"
            )
        ) if append_to_existing and self.is_trained else None

        X_new = df_new[self.feature_cols].fillna(0)

        if append_to_existing and self.is_trained and self._train_X is not None:
            # Gộp vào pool cũ
            if y_new is None:
                y_new = self.label_enc.transform(y_str_new.fillna("gift"))
            X_combined = pd.concat([self._train_X, X_new], ignore_index=True)
            y_combined = np.concatenate([self._train_y, y_new])
            if verbose:
                print(f"[LEP] Pool mở rộng: {len(self._train_X)} → {len(X_combined)} samples")
        else:
            # Không append: train lại từ đầu hoàn toàn với data mới
            y_new_str = y_str_new.fillna("gift")
            y_combined = self.label_enc.fit_transform(y_new_str)
            X_combined = X_new

        # Lưu lại pool mới
        self._train_X = X_combined.copy() if isinstance(X_combined, pd.DataFrame) else pd.DataFrame(X_combined, columns=self.feature_cols)
        self._train_y = y_combined.copy()

        metrics = self._fit_classifier(X_combined, y_combined, verbose)
        self.is_trained = True
        prev_meta = self.train_metadata.copy()
        self.train_metadata = {
            "retrained_at": datetime.now().isoformat(),
            "n_new_samples": len(df_new_profiles),
            "total_pool": len(X_combined),
            "append_to_existing": append_to_existing,
            "label_source": "ml_predictions" if df_new_ml is not None else "persona/intent",
            "classes": list(self.label_enc.classes_),
            "metrics": metrics,
            "previous_train": prev_meta,
        }
        return metrics

    def predict(self, df_profiles: pd.DataFrame) -> pd.DataFrame:
        """
        Predict intent + xác suất cho từng khách.
        Mọi xác suất đều do model tính toán — không có giá trị cố định.

        Returns DataFrame với cols:
            customer_id, predicted_intent, proba_engagement,
            proba_anniversary, proba_self_reward, proba_gift, confidence, priority
        """
        assert self.is_trained, "Gọi train() hoặc load() trước khi predict()"

        df = self._prepare_features(df_profiles, fit_encoders=False)
        X = df[self.feature_cols].fillna(0)

        proba = self.clf.predict_proba(X)

        # clf.classes_ là các nhãn (encoded) mà model THỰC SỰ thấy khi train
        # (có thể ít hơn label_enc.classes_ nếu dữ liệu ít và một class bị rớt khỏi train set)
        clf_classes_encoded = list(self.clf.classes_)          # numeric encoded labels
        clf_classes_names   = list(self.label_enc.inverse_transform(clf_classes_encoded))

        # Predict label: map qua clf_classes chứ không phải toàn bộ label_enc
        pred_proba_idx = proba.argmax(axis=1)
        pred_labels    = [clf_classes_names[i] for i in pred_proba_idx]

        # Map proba columns theo 4-intent order cố định
        id_col = _resolve_id_col(df_profiles)
        result = pd.DataFrame({"customer_id": df_profiles[id_col].values})
        intent_order = ["engagement", "anniversary", "self_reward", "gift"]
        for intent in intent_order:
            if intent in clf_classes_names:
                col_idx = clf_classes_names.index(intent)
                result[f"proba_{intent}"] = proba[:, col_idx]
            else:
                # Class này không xuất hiện trong training (dữ liệu quá ít)
                result[f"proba_{intent}"] = 0.0

        result["predicted_intent"] = pred_labels
        result["confidence"] = proba.max(axis=1)
        result["priority"] = pd.cut(
            result["confidence"],
            bins=[0, 0.5, 0.7, 1.01],
            labels=["low", "medium", "high"],
        )
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> str:
        """
        Lưu model xuống file .pkl.
        Nếu không truyền path → tự động lưu vào outputs/models/lep_model.pkl.

        Returns:
            str đường dẫn file đã lưu
        """
        assert self.is_trained, "Model chưa được train, không thể save."

        if path is None:
            DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = DEFAULT_MODEL_DIR / "lep_model.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "clf": self.clf,
            "encoders": self.encoders,
            "feature_cols": self.feature_cols,
            "label_enc": self.label_enc,
            "train_metadata": self.train_metadata,
            # Lưu training pool để hỗ trợ retrain với append
            "_train_X": self._train_X,
            "_train_y": self._train_y,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        # Lưu metadata dạng JSON để dễ đọc
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.train_metadata, f, ensure_ascii=False, indent=2,
                      default=str)

        print(f"[LEP] Model saved → {path}")
        print(f"[LEP] Metadata   → {meta_path}")
        return str(path)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LEPModel":
        """
        Load model từ file .pkl đã lưu.
        Nếu không truyền path → tự động load từ outputs/models/lep_model.pkl.

        Returns:
            LEPModel instance đã train sẵn, predict được ngay.
        """
        if path is None:
            path = DEFAULT_MODEL_DIR / "lep_model.pkl"
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model tại: {path}\n"
                "Hãy chạy model.train(...) và model.save() trước."
            )

        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls.__new__(cls)
        instance.clf          = payload["clf"]
        instance.encoders     = payload["encoders"]
        instance.feature_cols = payload["feature_cols"]
        instance.label_enc    = payload["label_enc"]
        instance.train_metadata = payload.get("train_metadata", {})
        instance._train_X     = payload.get("_train_X")
        instance._train_y     = payload.get("_train_y")
        instance.is_trained   = True

        trained_at = instance.train_metadata.get(
            "trained_at", instance.train_metadata.get("retrained_at", "unknown"))
        print(f"[LEP] Model loaded from: {path}")
        print(f"[LEP] Trained at: {trained_at}, "
              f"pool size: {instance.train_metadata.get('n_samples', instance.train_metadata.get('total_pool', '?'))}")
        return instance

    def get_metadata(self) -> dict:
        """Trả về thông tin về model đã train."""
        return self.train_metadata.copy()


# ── Feature importance ─────────────────────────────────────────────────────────
def get_feature_importance(model: LEPModel, top_n: int = 15) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": model.feature_cols,
        "importance": model.clf.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)
    return imp


# ── Standalone run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Demo các workflow:
      1. Train lần đầu + save
      2. Load model đã lưu + predict (không train lại)
      3. Retrain với dữ liệu mới
    """
    DATA_PATH = "data/customer_data_poc_enhanced.xlsx"
    df_p = pd.read_excel(DATA_PATH, sheet_name="profiles_enhanced")
    df_ml = pd.read_excel(DATA_PATH, sheet_name="ml_predictions")

    print("=" * 60)
    print("WORKFLOW 1: Train từ đầu và lưu model")
    print("=" * 60)
    model = LEPModel()
    metrics = model.train(df_p, df_ml, verbose=True)
    saved_path = model.save()

    preds = model.predict(df_p)
    print("\n[LEP] Predictions sample:")
    print(preds[["customer_id", "predicted_intent", "confidence", "priority"]].to_string(index=False))

    print("\n[LEP] Top features:")
    print(get_feature_importance(model).to_string(index=False))

    print("\n" + "=" * 60)
    print("WORKFLOW 2: Load model đã lưu (không train lại)")
    print("=" * 60)
    model2 = LEPModel.load()
    preds2 = model2.predict(df_p)
    print("[LEP] Predictions từ model đã load:")
    print(preds2[["customer_id", "predicted_intent", "confidence"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("WORKFLOW 3: Retrain khi có dữ liệu mới (giả lập bằng 3 dòng đầu)")
    print("=" * 60)
    df_new = df_p.head(3).copy()       # Giả lập dữ liệu mới
    df_new_ml = df_ml.head(3).copy()
    model2.retrain(df_new, df_new_ml, append_to_existing=True, verbose=True)
    model2.save()

    preds3 = model2.predict(df_p)
    print("[LEP] Predictions sau retrain:")
    print(preds3[["customer_id", "predicted_intent", "confidence"]].to_string(index=False))
