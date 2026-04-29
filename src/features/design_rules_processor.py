"""
design_rules_processor.py
-------------------------
Tinh cac chi so quy chuan thiet ke tu anh da preprocess:

1) wcag_contrast         - ti le tuong phan chu/nen theo WCAG
2) rule_violation_score  - do lech vi tri chu so voi ty le vang va quy tac 1/3

Input:
  - data/processed/image_processed/
  - data/processed/banner_visual_structure_and_ocr_extract.csv (text_boxes_json)

Output:
  - DataFrame voi cot: img_id, wcag_contrast, rule_violation_score
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def srgb_to_linear(channel: float) -> float:
    c = channel / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(bgr: tuple[float, float, float]) -> float:
    b, g, r = bgr
    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def wcag_contrast_ratio(l1: float, l2: float) -> float:
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def clamp_box(box: dict[str, Any], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    try:
        x = int(box.get("left", box.get("x_min", 0)))
        y = int(box.get("top", box.get("y_min", 0)))
        w = int(box.get("width", 0))
        h = int(box.get("height", 0))
    except (TypeError, ValueError):
        return None

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)

    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def compute_box_contrast(image_bgr: np.ndarray, box: dict[str, Any]) -> tuple[float, int] | None:
    img_h, img_w = image_bgr.shape[:2]
    coords = clamp_box(box, img_w, img_h)
    if coords is None:
        return None

    x0, y0, x1, y1 = coords
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    area_white = int(cv2.countNonZero(mask))
    area_black = int(mask.size - area_white)
    if area_black == 0 or area_white == 0:
        return 1.0, int(roi.shape[0] * roi.shape[1])

    # Nhom it pixel hon duoc xem la text.
    if area_black < area_white:
        text_mask = cv2.bitwise_not(mask)
        bg_mask = mask
    else:
        text_mask = mask
        bg_mask = cv2.bitwise_not(mask)

    text_mean = cv2.mean(roi, mask=text_mask)[:3]
    bg_mean = cv2.mean(roi, mask=bg_mask)[:3]

    l_text = relative_luminance((text_mean[0], text_mean[1], text_mean[2]))
    l_bg = relative_luminance((bg_mean[0], bg_mean[1], bg_mean[2]))
    contrast = wcag_contrast_ratio(l_text, l_bg)

    area = int(roi.shape[0] * roi.shape[1])
    return contrast, area


def compute_wcag_contrast(image_bgr: np.ndarray, boxes: list[dict[str, Any]]) -> float:
    if not boxes:
        return 0.0

    total = 0.0
    total_weight = 0.0

    for box in boxes:
        result = compute_box_contrast(image_bgr, box)
        if result is None:
            continue
        contrast, area = result
        if area <= 0:
            continue
        total += contrast * area
        total_weight += area

    if total_weight <= 0:
        return 0.0

    value = total / total_weight
    return float(round(min(value, 21.0), 6))


def compute_rule_violation_score(img_w: int, img_h: int, boxes: list[dict[str, Any]]) -> float:
    if img_w <= 0 or img_h <= 0 or not boxes:
        return 0.0

    third_x = [img_w / 3.0, 2.0 * img_w / 3.0]
    third_y = [img_h / 3.0, 2.0 * img_h / 3.0]

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    golden_a = 1.0 / phi
    golden_b = 1.0 - golden_a
    golden_x = [img_w * golden_b, img_w * golden_a]
    golden_y = [img_h * golden_b, img_h * golden_a]

    third_points = [(x, y) for x in third_x for y in third_y]
    golden_points = [(x, y) for x in golden_x for y in golden_y]

    diag = math.sqrt((img_w * img_w) + (img_h * img_h))
    if diag <= 0:
        return 0.0

    total = 0.0
    total_weight = 0.0

    for box in boxes:
        coords = clamp_box(box, img_w, img_h)
        if coords is None:
            continue
        x0, y0, x1, y1 = coords
        area = float((x1 - x0) * (y1 - y0))
        if area <= 0:
            continue

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        min_third = min(math.hypot(cx - px, cy - py) for px, py in third_points)
        min_golden = min(math.hypot(cx - px, cy - py) for px, py in golden_points)
        min_dist = min(min_third, min_golden)

        score = min_dist / diag
        total += score * area
        total_weight += area

    if total_weight <= 0:
        return 0.0

    return float(round(total / total_weight, 6))


def load_text_boxes_lookup(ocr_csv: Path) -> dict[str, list[dict[str, Any]]]:
    if not ocr_csv.is_file():
        return {}

    df = pd.read_csv(ocr_csv, encoding="utf-8-sig")
    required = {"image_name", "text_boxes_json"}
    missing = required - set(df.columns)
    if missing:
        return {}

    lookup: dict[str, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        img_id = Path(str(row["image_name"])).stem
        raw = row.get("text_boxes_json", "[]")
        try:
            boxes = json.loads(raw) if isinstance(raw, str) else []
        except (json.JSONDecodeError, TypeError):
            boxes = []
        lookup[img_id] = boxes

    return lookup


def build_image_lookup(image_dir: Path) -> dict[str, Path]:
    if not image_dir.is_dir():
        return {}
    files = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return {p.stem: p for p in files}


def compute_design_rules_features(
    image_dir: Path,
    ocr_csv: Path,
    img_ids: list[str] | None = None,
) -> pd.DataFrame:
    boxes_lookup = load_text_boxes_lookup(ocr_csv)
    image_lookup = build_image_lookup(image_dir)

    if img_ids is None:
        img_ids = sorted(set(boxes_lookup.keys()))

    rows: list[dict[str, Any]] = []
    for img_id in img_ids:
        boxes = boxes_lookup.get(img_id, [])
        if not boxes:
            rows.append({
                "img_id": img_id,
                "wcag_contrast": 0.0,
                "rule_violation_score": 0.0,
            })
            continue

        img_path = image_lookup.get(img_id)
        if img_path is None:
            rows.append({
                "img_id": img_id,
                "wcag_contrast": 0.0,
                "rule_violation_score": 0.0,
            })
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            rows.append({
                "img_id": img_id,
                "wcag_contrast": 0.0,
                "rule_violation_score": 0.0,
            })
            continue

        wcag = compute_wcag_contrast(image_bgr, boxes)
        rule = compute_rule_violation_score(image_bgr.shape[1], image_bgr.shape[0], boxes)

        rows.append({
            "img_id": img_id,
            "wcag_contrast": wcag,
            "rule_violation_score": rule,
        })

    return pd.DataFrame(rows)
