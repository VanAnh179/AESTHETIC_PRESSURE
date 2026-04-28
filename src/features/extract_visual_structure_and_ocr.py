"""
extract_visual_structure_and_ocr.py
-----------------------------------
Trich xuat feature tu anh banner da preprocess:
1) edge_density bang Canny.
2) geometric_blocks: do muc do phan bo khoi hinh hoc lon.
3) OCR: quet toan anh va lay bounding boxes vung text.

Input : data/processed/image_processed/
Output: data/processed/banner_feature_extract.csv
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output
from tqdm import tqdm

# ── Cau hinh duong dan ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "image_processed"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "banner_visual_structure_and_ocr_extract.csv"
TESSDATA_DIR = PROJECT_ROOT / "data" / "tessdata"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Windows thuong can tro truc tiep toi tesseract.exe neu PATH chua cap nhat.
TESSERACT_CMD = Path(
    os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
)
if TESSERACT_CMD.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_CMD)
if TESSDATA_DIR.is_dir():
    os.environ.setdefault("TESSDATA_PREFIX", str(TESSDATA_DIR))

# ── Cau hinh Canny / geometric blocks / OCR ─────────────────────────────────
CANNY_LOW = 70
CANNY_HIGH = 170
MIN_BLOCK_AREA_RATIO = 0.005  # >=0.5% dien tich anh duoc coi la khoi lon
MIN_BLOCK_DIM_RATIO = 0.04  # moi canh khoi >=4% canh anh
MAX_BLOCKS_COUNTED = 25
OCR_MIN_CONFIDENCE = 60.0
OCR_LANG = os.environ.get("OCR_LANG", "vie+eng")
OCR_CONFIG = (
    f"--tessdata-dir {TESSDATA_DIR} "
    "--oem 3 --psm 11 -c preserve_interword_spaces=1"
)
TEXT_LIKE_PATTERN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]")


@dataclass
class FeatureRow:
    image_name: str
    width: int
    height: int
    edge_density: float
    geometric_blocks: int
    geometric_block_area_ratio: float
    text_region_count: int
    text_boxes_json: str


def list_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    files: list[Path] = []
    for name in os.listdir(input_dir):
        p = input_dir / name
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return sorted(files, key=lambda x: x.name.lower())


def compute_edge_density(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edge_pixels = int(np.count_nonzero(edges))
    total_pixels = int(edges.shape[0] * edges.shape[1])
    if total_pixels == 0:
        return 0.0
    return float(edge_pixels / total_pixels)


def compute_geometric_blocks(image_bgr: np.ndarray) -> tuple[int, float]:
    """
    Do luong khoi hinh hoc lon:
    - Tim contour tu mask canh.
    - Loc contour theo dien tich toi thieu (theo ty le anh).
    - Tra ve:
      + so khoi lon (geometric_blocks)
      + tong dien tich khoi lon / dien tich anh
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny + nguong nhi phan de lay khoi lon ngay ca khi bien canh yeu.
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = cv2.bitwise_or(edges, binary)

    # Dong/mo de tao vung lien thong on dinh cho contour.
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    if img_area <= 0:
        return 0, 0.0

    min_block_area = MIN_BLOCK_AREA_RATIO * img_area
    img_h, img_w = image_bgr.shape[:2]
    min_block_w = img_w * MIN_BLOCK_DIM_RATIO
    min_block_h = img_h * MIN_BLOCK_DIM_RATIO
    block_count = 0
    block_area_sum = 0.0

    # Uu tien khoi lon hon de han che dem nhieu thanh phan nho.
    contour_areas = sorted(
        ((float(cv2.contourArea(cnt)), cnt) for cnt in contours),
        key=lambda x: x[0],
        reverse=True,
    )
    for area, cnt in contour_areas:
        if block_count >= MAX_BLOCKS_COUNTED:
            break
        if area < min_block_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_block_w or h < min_block_h:
            continue

        # Day dac contour (dien tich contour / dien tich box) de loai vienh rat mong.
        rect_area = float(max(w * h, 1))
        solidity = area / rect_area
        if solidity < 0.15:
            continue

        block_count += 1
        block_area_sum += area

    return block_count, float(block_area_sum / img_area)


def prepare_ocr_image(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Tien xu ly rieng cho OCR song ngu Viet/Anh.
    Dung RGB goc de tranh texture/banner bi nham thanh chu sau khi tang tuong phan.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return rgb, 1.0


def extract_text_boxes(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """
    OCR toan bo anh va tra ve danh sach box text de ban giao cho Linh Khanh.
    Moi phan tu gom:
    - text, conf
    - left, top, width, height
    - x_min, y_min, x_max, y_max
    """
    ocr_image, scale = prepare_ocr_image(image_bgr)
    data = pytesseract.image_to_data(
        ocr_image,
        lang=OCR_LANG,
        config=OCR_CONFIG,
        output_type=Output.DICT,
    )

    boxes: list[dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = " ".join((data["text"][i] or "").strip().split())
        conf_raw = str(data["conf"][i]).strip()
        try:
            conf = float(conf_raw)
        except ValueError:
            conf = -1.0

        if not text or conf < OCR_MIN_CONFIDENCE:
            continue

        # Loai token nhieu nhu "..", "/", "-", ky tu loi OCR hoac icon bi nham text.
        if "�" in text or TEXT_LIKE_PATTERN.search(text) is None:
            continue

        left = int(round(int(data["left"][i]) / scale))
        top = int(round(int(data["top"][i]) / scale))
        width = int(round(int(data["width"][i]) / scale))
        height = int(round(int(data["height"][i]) / scale))

        if width <= 0 or height <= 0:
            continue

        # Banner co nhieu texture; token qua nho thuong la nhieu, khong phai text vung.
        if width < 8 or height < 8:
            continue

        box = {
            "text": text,
            "conf": conf,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "x_min": left,
            "y_min": top,
            "x_max": left + width,
            "y_max": top + height,
            "block_num": int(data["block_num"][i]),
            "par_num": int(data["par_num"][i]),
            "line_num": int(data["line_num"][i]),
            "word_num": int(data["word_num"][i]),
        }
        boxes.append(box)

    return boxes


def process_image(path: Path) -> FeatureRow | None:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        print(f"Skipped unreadable image: {path.name}")
        return None

    h, w = image_bgr.shape[:2]
    edge_density = compute_edge_density(image_bgr)
    block_count, block_area_ratio = compute_geometric_blocks(image_bgr)
    text_boxes = extract_text_boxes(image_bgr)

    row = FeatureRow(
        image_name=path.name,
        width=w,
        height=h,
        edge_density=round(edge_density, 6),
        geometric_blocks=block_count,
        geometric_block_area_ratio=round(block_area_ratio, 6),
        text_region_count=len(text_boxes),
        # Chuoi JSON de Linh Khanh co the parse truc tiep tinh ty le dien tich text
        text_boxes_json=json.dumps(text_boxes, ensure_ascii=False),
    )
    return row


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc input: {INPUT_DIR}")

    files = list_image_files(INPUT_DIR)
    if not files:
        print(f"Khong tim thay anh nao trong: {INPUT_DIR}")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with tqdm(files, desc="Extract banner features", unit="img", ncols=100) as pbar:
        for path in pbar:
            pbar.set_postfix_str(path.name, refresh=True)
            result = process_image(path)
            if result is not None:
                rows.append(asdict(result))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("-" * 70)
    print(f"Done. Images scanned: {len(files)} | Rows exported: {len(df)}")
    print(f"CSV exported to: {OUTPUT_CSV}")
    print("-" * 70)


if __name__ == "__main__":
    main()
