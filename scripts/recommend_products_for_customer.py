#!/usr/bin/env python3
"""Recommend real PNJ products for one customer without external APIs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.product_recommender import DEFAULT_METADATA_PATH, ProductRecommender


DEFAULT_CUSTOMER_XLSX = ROOT / "data" / "customer_data_poc_enhanced.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get top real PNJ products for one customer from local data.",
    )
    parser.add_argument("customer_id", help="Customer ID in customer_data_poc_enhanced.xlsx")
    parser.add_argument("--customer-xlsx", type=Path, default=DEFAULT_CUSTOMER_XLSX)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--recipient-gender", default="", help='Giới tính người thụ hưởng: "Nam", "Nữ" hoặc bỏ trống.')
    parser.add_argument("--recipient-audience", default="", help='Nhóm người thụ hưởng: "Người lớn" hoặc "Trẻ em".')
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def load_customer_profile(path: Path, customer_id: str) -> dict[str, Any]:
    sheets = pd.read_excel(path, sheet_name=None)
    profiles = sheets["profiles_enhanced"].copy()
    id_col = "c" if "c" in profiles.columns else "customer_id"
    profiles["customer_id"] = profiles[id_col].astype(str)

    if "profiles" in sheets and "gender" not in profiles.columns:
        gender_df = sheets["profiles"][["customer_id", "gender"]].copy()
        gender_df["customer_id"] = gender_df["customer_id"].astype(str)
        profiles = profiles.merge(gender_df, on="customer_id", how="left")

    row = profiles[profiles["customer_id"] == str(customer_id)]
    if row.empty:
        raise SystemExit(f"Không tìm thấy customer_id={customer_id}")

    profile = row.iloc[0].to_dict()
    profile["customer_id"] = str(customer_id)
    return profile


def print_human(customer_id: str, profile: dict[str, Any], recs: list[dict[str, Any]]) -> None:
    print(f"Khách: {customer_id}")
    print(
        "Tín hiệu: "
        f"segment={profile.get('segment_rfm_tier', '')}, "
        f"budget={profile.get('budget', '')}, "
        f"preferred_type={profile.get('preferred_type', '')}, "
        f"material={profile.get('material', '')}, "
        f"style={profile.get('style', '')}, "
        f"recipient_gender={profile.get('recipient_gender', '')}, "
        f"recipient_audience={profile.get('recipient_audience', '')}"
    )
    print()

    if not recs:
        print("Không tìm được sản phẩm phù hợp.")
        return

    for idx, rec in enumerate(recs, 1):
        print(f"{idx}. {rec['name']}")
        print(f"   SKU: {rec['sku']} | Giá: {rec['price_text']} | Điểm: {rec['score']}/100")
        if rec.get("url"):
            print(f"   Link: {rec['url']}")
        breakdown = rec.get("score_breakdown") or {}
        if breakdown:
            parts = [f"{k}={v:g}" for k, v in breakdown.items()]
            print("   Breakdown: " + ", ".join(parts))
        for reason in rec.get("evidence", []):
            print(f"   - {reason}")
        print()


def main() -> None:
    args = parse_args()
    profile = load_customer_profile(args.customer_xlsx, args.customer_id)
    if args.recipient_gender:
        profile["recipient_gender"] = args.recipient_gender
    if args.recipient_audience:
        profile["recipient_audience"] = args.recipient_audience
    recommender = ProductRecommender(catalog_path=args.catalog)
    recs = recommender.recommend_for_profile(profile, top_n=args.top_n)

    if args.json:
        print(json.dumps({"customer_id": args.customer_id, "recommendations": recs}, ensure_ascii=False, indent=2))
        return

    print_human(args.customer_id, profile, recs)


if __name__ == "__main__":
    main()
