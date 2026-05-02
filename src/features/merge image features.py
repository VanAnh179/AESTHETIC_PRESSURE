"""
merge_features.py
-----------------
Gop dac trung thanh image_extracted_features.csv voi them cot AP:

    img_id               String  (ten file khong co phan mo rong)
    edge_density         Float   (tu banner_visual_structure_and_ocr_extract.csv)
    geometric_blocks     Integer/Float (tu banner_visual_structure_and_ocr_extract.csv)
    color_entropy        Float   (tu features_color_and_text.csv)
    compression_ratio    Float   (tu features_color_and_text.csv)
    text_area_ratio      Float   (tu features_color_and_text.csv)
    wcag_contrast        Float   (WCAG contrast tu anh + text boxes)
    rule_violation_score Float   (vi tri text so voi ty le vang / rule of thirds)
    ap_interaction_mean  Float   (AP tuong tac, gop trung binh theo img_id)
    design_score         Float   (diem thiet ke tu wcag_contrast + rule_violation_score)
    ap_final             Float   (AP cuoi: 0.70*ap_interaction_mean + 0.30*design_score)

Nguon:
    data/processed/banner_visual_structure_and_ocr_extract.csv
    data/processed/features_color_and_text.csv
    data/processed/image_processed/
    data/processed/interaction_ap_scores.csv

Dau ra:
    data/processed/image_extracted_features.csv

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from design_rules_processor import compute_design_rules_features

# ── Đường dẫn mặc định ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED    = PROJECT_ROOT / "data" / "processed"

DEFAULT_OCR_CSV   = PROCESSED / "banner_visual_structure_and_ocr_extract.csv"
DEFAULT_COLOR_CSV = PROCESSED / "features_color_and_text.csv"
DEFAULT_IMAGE_DIR = PROCESSED / "image_processed"
DEFAULT_AP_CSV    = PROCESSED / "interaction_ap_scores.csv"
DEFAULT_OUTPUT    = PROCESSED / "image_extracted_features.csv"

AP_INTERACTION_WEIGHT = 0.70
DESIGN_SCORE_WEIGHT = 0.30
DESIGN_WCAG_WEIGHT = 0.60
DESIGN_RULE_WEIGHT = 1.0 - DESIGN_WCAG_WEIGHT

FINAL_COLUMNS = [
    "img_id",
    "edge_density",
    "geometric_blocks",
    "color_entropy",
    "compression_ratio",
    "text_area_ratio",
    "wcag_contrast",
    "rule_violation_score",
    "ap_interaction_mean",
    "design_score",
    "ap_final",
]


def minmax_normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v <= min_v:
        return values * 0.0
    return (values - min_v) / (max_v - min_v)


def load_ocr_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        print(f"[LOI] Khong tim thay OCR CSV: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"image_name", "edge_density", "geometric_blocks"}
    missing = required - set(df.columns)
    if missing:
        print(f"[LOI] OCR CSV thieu cot: {missing}", file=sys.stderr)
        sys.exit(1)
    df["img_id"] = df["image_name"].apply(lambda x: Path(str(x)).stem)
    return df[["img_id", "edge_density", "geometric_blocks"]].copy()


def load_color_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        print(f"[LOI] Khong tim thay Color CSV: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"image_id", "color_entropy", "compression_ratio", "text_area_ratio"}
    missing = required - set(df.columns)
    if missing:
        print(f"[LOI] Color CSV thieu cot: {missing}", file=sys.stderr)
        sys.exit(1)
    df = df.rename(columns={"image_id": "img_id"})
    return df[["img_id", "color_entropy", "compression_ratio", "text_area_ratio"]].copy()


def load_ap_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        print(f"[CANH BAO] Khong tim thay AP CSV: {path}")
        return pd.DataFrame(columns=["img_id", "ap_interaction_mean"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"img_id", "ap_sentiment_adjusted_chuan_hoa"}
    missing = required - set(df.columns)
    if missing:
        print(f"[CANH BAO] AP CSV thieu cot: {missing}. Bo qua AP merge.")
        return pd.DataFrame(columns=["img_id", "ap_interaction_mean"])

    grouped = (
        df[["img_id", "ap_sentiment_adjusted_chuan_hoa"]]
        .copy()
        .groupby("img_id", as_index=False)
        .mean(numeric_only=True)
    )
    grouped = grouped.rename(
        columns={"ap_sentiment_adjusted_chuan_hoa": "ap_interaction_mean"}
    )
    return grouped


def merge_features(
    ocr_csv: Path,
    color_csv: Path,
    image_dir: Path,
    ap_csv: Path,
    output_csv: Path,
    join: str = "outer",
) -> None:
    print(f"Doc OCR CSV   : {ocr_csv}")
    print(f"Doc Color CSV : {color_csv}")

    df_ocr   = load_ocr_csv(ocr_csv)
    df_color = load_color_csv(color_csv)

    print(f"\nSo dong OCR CSV   : {len(df_ocr):>5}")
    print(f"So dong Color CSV : {len(df_color):>5}")

    df = pd.merge(df_ocr, df_color, on="img_id", how=join)

    if image_dir.is_dir():
        print(f"Doc anh       : {image_dir}")
        design_df = compute_design_rules_features(image_dir, ocr_csv, df["img_id"].tolist())
        df = pd.merge(df, design_df, on="img_id", how="left")
    else:
        print(f"[CANH BAO] Khong tim thay thu muc anh: {image_dir}")
        df["wcag_contrast"] = 0.0
        df["rule_violation_score"] = 0.0

    ap_df = load_ap_csv(ap_csv)
    if not ap_df.empty:
        df = pd.merge(df, ap_df, on="img_id", how="left")
    else:
        df["ap_interaction_mean"] = 0.0

    df["ap_interaction_mean"] = (
        pd.to_numeric(df["ap_interaction_mean"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    wcag_norm = minmax_normalize(df["wcag_contrast"])
    rule_norm = minmax_normalize(df["rule_violation_score"])
    design_score = (DESIGN_WCAG_WEIGHT * wcag_norm) + (DESIGN_RULE_WEIGHT * (1.0 - rule_norm))
    df["design_score"] = design_score
    df["ap_final"] = (
        AP_INTERACTION_WEIGHT * df["ap_interaction_mean"]
        + DESIGN_SCORE_WEIGHT * df["design_score"]
    )

    df = df[FINAL_COLUMNS].sort_values("img_id", ignore_index=True)

    n_total    = len(df)
    n_complete = df.dropna().shape[0]
    n_missing  = n_total - n_complete

    print(f"\nKet qua sau merge ({join} join):")
    print(f"   Tong so anh      : {n_total}")
    print(f"   Du du lieu       : {n_complete}")
    print(f"   Co gia tri thieu : {n_missing}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig", float_format="%.6f")

    print(f"\n{'=' * 60}")
    print(f"Hoan tat! Da ghi: {output_csv}")
    print(f"   So dong : {n_total}  |  So cot : {len(FINAL_COLUMNS)}")
    print(f"   Cac cot : {', '.join(FINAL_COLUMNS)}")
    print(f"{'=' * 60}")
    print("\nXem truoc (5 dong dau):")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gop dac trung thanh image_extracted_features.csv (them AP)")
    parser.add_argument("--ocr",    type=Path, default=DEFAULT_OCR_CSV)
    parser.add_argument("--color",  type=Path, default=DEFAULT_COLOR_CSV)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--ap",      type=Path, default=DEFAULT_AP_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--join",   choices=["outer", "inner", "left", "right"], default="outer")
    args = parser.parse_args()
    merge_features(args.ocr, args.color, args.images, args.ap, args.output, args.join)