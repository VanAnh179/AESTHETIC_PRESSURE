"""
comment_processed.py
--------------------
Data Aggregator — Gom tụ comment từ 3 platform:
    data/raw/facebook/
    data/raw/shopee/
    data/raw/instagram/

Với mỗi folder, script tự động phát hiện file CSV tổng hợp (file có nhiều row nhất).
Sau đó liên kết với image_id và gộp tất cả vào một bảng.

Nhiệm vụ: CHỈ làm nhiệm vụ "bốc vác" (data collection).
Xử lý NLP (làm sạch text, tokenize, quét từ khóa) sẽ được xử lý ở nlp_engine.py.

Đầu ra: data/processed/merged_comments_raw.csv
Cột   : image_id | source | raw_text

"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
#   Vị trí file: src/features/comment_processed.py
#   parents[0]  = src/features/
#   parents[1]  = src/
#   parents[2]  = project root

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT  = PROJECT_ROOT / "data" / "processed" / "merged_comments_raw.csv"

# Thứ tự ưu tiên tên các thư mục platform (phải khớp với SOURCE_PREFIX)
PLATFORM_DIRS: dict[str, str] = {
    "facebook":  "fac",
    "shopee":    "sho",
    "instagram": "ins",
}

# ── Tên cột có thể xuất hiện trong CSV raw ───────────────────────────────────
# Script sẽ dò tự động; thứ tự = ưu tiên cao → thấp

CANDIDATE_IMAGE_COLS = [
    "img_id", "image_id",
    "image_name", "banner_id", "post_id",
    "ad_id", "product_id", "item_id", "id",
]

CANDIDATE_TEXT_COLS = [
    "raw_comment",
    "comment", "text", "content", "caption",
    "body", "message", "review", "description",
    "noi_dung", "binh_luan", "comment_text",
]


# ── Tiện ích detect cột ───────────────────────────────────────────────────────

def detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> Optional[str]:
    """
    Trả về tên cột đầu tiên trong `candidates` tồn tại trong df (không phân biệt hoa/thường).
    Nếu không tìm thấy, trả None và in cảnh báo.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    print(f"  [CANH BAO] Khong tim thay cot '{label}'. "
          f"Cac cot hien co: {list(df.columns)}", file=sys.stderr)
    return None


# ── Đọc CSV tổng hợp từ một thư mục platform ─────────────────────────────────

def find_main_csv(folder: Path, verbose: bool = False) -> Optional[Path]:
    """
    Tìm file CSV "tổng hợp" trong folder: ưu tiên file có nhiều row nhất.
    Trả None nếu không có CSV nào.
    """
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        return None
    if len(csvs) == 1:
        return csvs[0]

    # Nhiều file: chọn file có nhiều dòng nhất (file tổng hợp)
    best: Optional[Path] = None
    best_rows = -1
    for csv_path in csvs:
        try:
            # Đọc nhanh với chunksize để đếm dòng, tránh load toàn bộ vào RAM
            n = sum(1 for _ in open(csv_path, encoding="utf-8-sig", errors="ignore")) - 1
            if verbose:
                print(f"    {csv_path.name}: {n} rows")
            if n > best_rows:
                best_rows = n
                best = csv_path
        except OSError:
            continue
    return best


def load_platform_csv(
    folder: Path,
    platform: str,
    prefix: str,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Đọc CSV tổng hợp của một platform, chuẩn hoá thành DataFrame với cột:
        image_id | source | raw_text

    Logic ánh xạ image_id:
        - Nếu CSV đã có cột image_id/image_name → dùng trực tiếp (strip .jpg/.png nếu có)
        - Nếu CSV có cột tương tự post_id/ad_id/item_id → dùng làm image_id
        - Nếu không có cột nào match → tạo image_id = {prefix}_row_{i}
          (trường hợp CSV không có khoá liên kết, vẫn giữ lại comment để làm sạch)

    Returns:
        DataFrame[image_id, source, raw_text] hoặc None nếu lỗi đọc file.
    """
    csv_path = find_main_csv(folder, verbose=verbose)
    if csv_path is None:
        print(f"  [CANH BAO] Khong tim thay CSV nao trong: {folder}", file=sys.stderr)
        return None

    print(f"  [DOC] {csv_path.relative_to(folder.parent.parent)} ({platform})")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    except Exception as e:
        print(f"  [LOI] Khong doc duoc {csv_path.name}: {e}", file=sys.stderr)
        return None

    if df.empty:
        print(f"  [CANH BAO] File rong: {csv_path.name}", file=sys.stderr)
        return None

    if verbose:
        print(f"    Cot: {list(df.columns)}")
        print(f"    So hang: {len(df)}")

    # ── Detect cột text ──────────────────────────────────────────────────────
    text_col = detect_column(df, CANDIDATE_TEXT_COLS, "text/comment")
    if text_col is None:
        print(f"  [BO QUA] {platform}: khong xac dinh duoc cot comment.", file=sys.stderr)
        return None

    # ── Detect cột image_id ──────────────────────────────────────────────────
    id_col = detect_column(df, CANDIDATE_IMAGE_COLS, "image_id")

    # Xây kết quả
    result = pd.DataFrame()
    result["raw_text"] = df[text_col].astype(str)

    if id_col:
        # Xoá đuôi .jpg/.jpeg/.png/.webp nếu có (để khớp với image_id trong features CSV)
        result["image_id"] = (
            df[id_col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.(jpg|jpeg|png|webp|gif|bmp|tiff?)$", "", flags=re.IGNORECASE, regex=True)
        )
    else:
        # Fallback: sinh image_id tạm từ prefix và thứ tự dòng
        result["image_id"] = [f"{prefix}_{i+1:04d}" for i in range(len(df))]
        print(f"  [INFO] Khong co cot image_id ro rang — "
              f"su dung '{prefix}_XXXX' lam placeholder.", file=sys.stderr)

    result["source"] = platform
    return result[["image_id", "source", "raw_text"]]


# ── Hàm chính ─────────────────────────────────────────────────────────────────

def process_comments(
    raw_dir: Path,
    output_csv: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """
    Điều phối toàn bộ pipeline:
        1. Đọc CSV từ 3 platform
        2. Gộp thành một DataFrame duy nhất
        3. Làm sạch cột raw_text → cleaned_text
        4. Lọc bỏ comment rỗng
        5. Ghi ra output_csv

    Parameters
    ----------
    raw_dir  : Path – thư mục gốc chứa facebook/, shopee/, instagram/
    output_csv : Path – đường dẫn file CSV đầu ra
    dry_run  : bool – nếu True, chỉ in thống kê, không ghi file
    verbose  : bool – in chi tiết từng bước
    """
    print("=" * 65)
    print("COMMENT PROCESSOR — AESTHETIC PRESSURE PROJECT")
    print("=" * 65)
    print(f"Thu muc raw  : {raw_dir}")
    print(f"Output CSV   : {output_csv}")
    print(f"Dry-run      : {dry_run}")
    print("=" * 65 + "\n")

    frames: list[pd.DataFrame] = []

    for platform, prefix in PLATFORM_DIRS.items():
        folder = raw_dir / platform
        print(f"[{platform.upper()}]")

        if not folder.is_dir():
            print(f"  [BO QUA] Khong tim thay thu muc: {folder}\n",
                  file=sys.stderr)
            continue

        df = load_platform_csv(folder, platform, prefix, verbose=verbose)
        if df is None or df.empty:
            print(f"  [BO QUA] Khong co du lieu tu {platform}.\n",
                  file=sys.stderr)
            continue

        print(f"  -> {len(df):,} comment doc duoc\n")
        frames.append(df)

    if not frames:
        print("[LOI] Khong co du lieu nao duoc doc thanh cong. Ket thuc.",
              file=sys.stderr)
        sys.exit(1)

    # ── Gộp tất cả platform ─────────────────────────────────────────────────
    combined = pd.concat(frames, ignore_index=True)
    total_raw = len(combined)

    # ── Không làm sạch tại đây (chỉ gom dữ liệu) ────────────────────────────
    # Xử lý NLP sẽ được thực hiện ở nlp_engine.py
    dropped = 0

    # Sắp xếp theo image_id và source
    combined = combined[["image_id", "source", "raw_text"]]  # Giữ 3 cột
    combined.sort_values(["image_id", "source"], inplace=True, ignore_index=True)

    # ── Thống kê ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("THONG KE")
    print("=" * 65)
    print(f"  Tong comment doc vao        : {total_raw:>8,}")
    print(f"  Comment ghi ra CSV          : {len(combined):>8,}")
    print(f"  Image ID duy nhat           : {combined['image_id'].nunique():>8,}")
    print()

    # Thống kê theo platform
    for src, grp in combined.groupby("source", sort=False):
        print(f"  {src:<12} : {len(grp):>7,} comment  |  "
              f"{grp['image_id'].nunique():>5} image_id")

    print("=" * 65)

    if dry_run:
        print("\n[DRY-RUN] Khong ghi file. Su dung --no-dry-run de xuat CSV.")
        return

    # ── Ghi CSV ───────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n[XUAT] {output_csv}")
    print(f"       {len(combined):,} dong  |  "
          f"{output_csv.stat().st_size / 1024:.1f} KB")
    print("=" * 65)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Gom tụ comment từ facebook/shopee/instagram -> merged_comments_raw.csv"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Thu muc goc chua cac folder facebook/, shopee/, instagram/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Duong dan file CSV dau ra",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chi in thong ke, khong ghi file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In chi tiet ten cot va so dong moi file CSV",
    )
    args = parser.parse_args()

    process_comments(
        raw_dir=args.raw_dir,
        output_csv=args.output,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )