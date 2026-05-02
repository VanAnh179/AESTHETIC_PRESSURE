"""
collect_images.py
-----------------
Gom tất cả ảnh từ MỌI cấp thư mục bên trong:
    data/raw/facebook/
    data/raw/shopee/
    data/raw/instagram/
vào thư mục đích:
    data/raw/image/

Tên file GIỮ NGUYÊN như ban đầu.
Nếu trùng tên thì thêm hậu tố _1, _2, ...

"""

import argparse
import shutil
from pathlib import Path

# ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
#   Vị trí file: src/features/collect_images.py
#   parents[0] = src/features/
#   parents[1] = src/
#   parents[2] = AESTHETIC_PRESSURE/  ← project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

SOURCE_DIRS = {
    "facebook":  RAW_DIR / "facebook",
    "shopee":    RAW_DIR / "shopee",
    "instagram": RAW_DIR / "instagram",
}

DEST_DIR = RAW_DIR / "image"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def unique_dest(dest_path: Path) -> Path:
    """Thêm hậu tố _1, _2, … nếu file đích đã tồn tại."""
    if not dest_path.exists():
        return dest_path
    stem, suffix = dest_path.stem, dest_path.suffix
    i = 1
    while True:
        candidate = dest_path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


# ── Hàm chính ─────────────────────────────────────────────────────────────────

def collect_images(dry_run: bool = False, move: bool = False) -> None:
    action = "di chuyển" if move else "sao chép"
    prefix = "[DRY-RUN] " if dry_run else ""

    if not dry_run:
        DEST_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Thu muc dich (se tao neu chua co): {DEST_DIR}\n")

    total   = 0
    renamed = 0

    for source_name, source_root in SOURCE_DIRS.items():

        if not source_root.exists():
            print(f"Khong tim thay thu muc: {source_root}  -> bo qua.\n")
            continue

        # Lấy TẤT CẢ ảnh ở mọi cấp độ
        images = sorted(
            f for f in source_root.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{source_name.upper():<12} | {len(images):>4} anh  ({source_root})")

        if not images:
            print("      Khong co anh nao.\n")
            continue

        for img in images:
            # Giữ nguyên tên file gốc
            dest     = DEST_DIR / img.name
            resolved = unique_dest(dest)

            if resolved != dest:
                renamed += 1
                print(f"      Trung ten '{img.name}' -> doi thanh: {resolved.name}")

            print(f"      {prefix}{action}: {img.relative_to(source_root)}  ->  {resolved.name}")

            if not dry_run:
                if move:
                    shutil.move(str(img), resolved)
                else:
                    shutil.copy2(img, resolved)

            total += 1

        print()

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    print("-" * 60)
    print(f"Hoan tat {'(dry-run) ' if dry_run else ''}!")
    print(f"   Tong anh da {action:<13}: {total}")
    print(f"   Doi ten do trung       : {renamed}")
    print(f"   Thu muc dich           : {DEST_DIR}")
    print("-" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gom anh tu facebook/shopee/instagram -> data/raw/image/"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chi in danh sach, khong thuc su copy/move"
    )
    parser.add_argument(
        "--move", action="store_true",
        help="Di chuyen file thay vi sao chep"
    )
    args = parser.parse_args()
    collect_images(dry_run=args.dry_run, move=args.move)