"""
preprocess_images.py
-------------------
Tiền xử lý hàng loạt ảnh trong thư mục nguồn:
  1) Kiểm tra file hỏng (corrupted) — bỏ qua, không crash.
  2) Resize chiều rộng 1000px, giữ tỷ lệ (LANCZOS).
  3) Chuyển về RGB; ảnh có alpha: ghép nền trắng trước khi RGB.

Đọc:  data/raw/image/
Ghi:   data/processed/image_processed/  (cùng tên file, không ghi đè thư mục raw)

Chạy:  python preprocess_images.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageFile
from tqdm import tqdm

# Cho phép load ảnh hơi lỗi ở cuối file (một số trường hợp vẫn dùng được)
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ── Cấu hình đường dẫn (đổi ở đây theo môi trường của bạn) ──────────────────
# Vị trí file: src/features/preprocess_images.py  →  parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Thư mục ảnh gốc (input)
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "image"

# Thư mục ảnh đã xử lý (output)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "image_processed"

# Chiều rộng chuẩn sau resize (px)
TARGET_WIDTH = 1000

# Các phần mở rộng được coi là ảnh
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

# Nội suy chất lượng cao (Pillow 9+ dùng Resampling; bản cũ dùng hằng số cũ)
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - Pillow < 9
    RESAMPLE = Image.LANCZOS  # type: ignore[attr-defined]


# ── Bước 3: chuyển về RGB (nền trắng nếu có alpha) ───────────────────────────

def to_rgb_on_white(image: Image.Image) -> Image.Image:
    """
    Đưa mọi mode về RGB.
    - RGBA / LA / PA: ghép lên nền trắng theo kênh alpha.
    - P (palette): nếu có transparency trong info, xử lý qua RGBA rồi nền trắng.
    - CMYK, L, …: convert chuẩn sang RGB.
    """
    mode = image.mode

    if mode == "RGB":
        return image

    # Ảnh palette: cần kiểm tra transparency
    if mode == "P":
        if "transparency" in image.info:
            image = image.convert("RGBA")
            return to_rgb_on_white(image)
        return image.convert("RGB")

    # Có kênh alpha: dán lên nền trắng
    if mode in ("RGBA", "LA"):
        rgb = Image.new("RGB", image.size, (255, 255, 255))
        if mode == "RGBA":
            rgb.paste(image, mask=image.split()[-1])
        else:  # LA: grayscale + alpha
            rgb.paste(image.convert("RGB"), mask=image.split()[-1])
        return rgb

    if mode == "PA":  # palette + alpha (hiếm)
        return to_rgb_on_white(image.convert("RGBA"))

    if mode == "CMYK":
        return image.convert("RGB")

    if mode in ("L", "1", "I", "F"):
        return image.convert("RGB")

    # Các mode khác: thử chuyển thẳng
    return image.convert("RGB")


# ── Mở ảnh an toàn (bước 1: corrupted check) ────────────────────────────────

def try_open_image(path: Path) -> Image.Image | None:
    """
    Mở ảnh trong try/except. Nếu hỏng hoặc không đọc được, trả None (caller in cảnh báo).
    """
    try:
        im = Image.open(path)
        im.load()  # buộc giải mã pixel — bắt lỗi sớm nếu file vỡ
    except (OSError, ValueError) as e:
        print(
            f'Skipped corrupted file: {path.name}  ({e!s})',
            file=sys.stderr,
        )
        return None

    # GIF/WebP nhiều frame: chỉ lấy frame đầu cho pipeline thống nhất
    try:
        im.seek(0)
    except EOFError:
        print(f"Skipped corrupted file: {path.name}  (no frame)", file=sys.stderr)
        return None

    return im


# ── Bước 2: resize giữ tỷ lệ, width = TARGET_WIDTH ─────────────────────────

def resize_fixed_width(image: Image.Image, target_w: int) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    new_w = target_w
    new_h = max(1, int(round(h * (target_w / float(w)))))
    if (new_w, new_h) == (w, h):
        return image
    return image.resize((new_w, new_h), RESAMPLE)


# ── Pipeline từng file ─────────────────────────────────────────────────────

def process_one(path: Path, out_dir: Path, target_w: int) -> bool:
    """
    Trả True nếu lưu thành công, False nếu bỏ qua (corrupt) hoặc lỗi lưu.
    """
    im = try_open_image(path)
    if im is None:
        return False

    try:
        # Thứ tự: RGB (nền trắng nếu cần) trước, rồi resize — tránh scale alpha lạ
        rgb = to_rgb_on_white(im)
        resized = resize_fixed_width(rgb, target_w)
        out_path = out_dir / path.name

        # Lưu RGB; JPEG không hỗ trợ transparency (đã xử lý ở trên)
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            resized.save(out_path, quality=95, subsampling=0, optimize=True)
        else:
            resized.save(out_path)
        return True
    except (OSError, ValueError) as e:
        print(
            f'Skipped / failed to save: {path.name}  ({e!s})',
            file=sys.stderr,
        )
        return False
    finally:
        im.close()


# ── Quét thư mục & chạy toàn bộ ────────────────────────────────────────────

def list_image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for name in os.listdir(root):
        p = root / name
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


def main() -> None:
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    target_w = TARGET_WIDTH

    if not input_dir.is_dir():
        print(f"Khong tim thay thu muc nguon: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_image_files(input_dir)
    n = len(files)
    if n == 0:
        print(f"Khong co file anh nao trong: {input_dir}")
        return

    print(f"Nguon     : {input_dir}")
    print(f"Dich      : {output_dir}")
    print(f"Rong chuan: {target_w} px | Tong file: {n}\n")

    ok = 0
    with tqdm(
        files,
        desc="Tien xu ly",
        unit="file",
        ncols=80,
        file=sys.stdout,
    ) as pbar:
        for path in pbar:
            pbar.set_postfix_str(path.name, refresh=True)
            if process_one(path, output_dir, target_w):
                ok += 1

    print()
    print("-" * 60)
    print(f"Hoan tat. Thanh cong: {ok}/{n}  |  Bo qua/loi: {n - ok}")
    print(f"Thu muc xuat: {output_dir}")
    print("-" * 60)


if __name__ == "__main__":
    main()
