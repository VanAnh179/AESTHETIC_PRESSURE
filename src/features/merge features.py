"""
merge_features.py
-----------------
Gộp hai bộ đặc trưng đã trích xuất thành extracted_features.csv với 6 cột:

    img_id               String  (tên file không có phần mở rộng)
    edge_density         Float   (từ banner_visual_structure_and_ocr_extract.csv)
    geometric_blocks     Integer/Float (từ banner_visual_structure_and_ocr_extract.csv)
    color_entropy        Float   (từ features_color_and_text.csv)
    compression_ratio    Float   (từ features_color_and_text.csv)
    text_area_ratio      Float   (từ features_color_and_text.csv)

Nguồn:
    data/processed/banner_visual_structure_and_ocr_extract.csv
    data/processed/features_color_and_text.csv

Đầu ra:
    data/processed/extracted_features.csv

Cách dùng:
    python merge_features.py
    python merge_features.py --ocr   path/to/banner_visual_structure_and_ocr_extract.csv
    python merge_features.py --color path/to/features_color_and_text.csv
    python merge_features.py --output path/to/extracted_features.csv
    python merge_features.py --join inner
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Đường dẫn mặc định ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED    = PROJECT_ROOT / "data" / "processed"

DEFAULT_OCR_CSV   = PROCESSED / "banner_visual_structure_and_ocr_extract.csv"
DEFAULT_COLOR_CSV = PROCESSED / "features_color_and_text.csv"
DEFAULT_OUTPUT    = PROCESSED / "extracted_features.csv"

FINAL_COLUMNS = [
    "img_id",
    "edge_density",
    "geometric_blocks",
    "color_entropy",
    "compression_ratio",
    "text_area_ratio",
]


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


def merge_features(ocr_csv: Path, color_csv: Path, output_csv: Path, join: str = "outer") -> None:
    print(f"Doc OCR CSV   : {ocr_csv}")
    print(f"Doc Color CSV : {color_csv}")

    df_ocr   = load_ocr_csv(ocr_csv)
    df_color = load_color_csv(color_csv)

    print(f"\nSo dong OCR CSV   : {len(df_ocr):>5}")
    print(f"So dong Color CSV : {len(df_color):>5}")

    df = pd.merge(df_ocr, df_color, on="img_id", how=join)
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
    parser = argparse.ArgumentParser(description="Gop dac trung thanh extracted_features.csv (6 cot)")
    parser.add_argument("--ocr",    type=Path, default=DEFAULT_OCR_CSV)
    parser.add_argument("--color",  type=Path, default=DEFAULT_COLOR_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--join",   choices=["outer", "inner", "left", "right"], default="outer")
    args = parser.parse_args()
    merge_features(args.ocr, args.color, args.output, args.join)