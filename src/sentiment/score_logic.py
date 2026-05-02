"""
score_logic.py
--------------
Tầng tính điểm: tổng hợp matched_keywords từ comment_processed.csv
thành điểm áp lực thẩm mỹ (Aesthetic Pressure Score) per image_id.

Quy trình:
    1. Đọc comment_processed.csv (output của nlp_engine.py)
    2. Đọc score_logic.json để lấy trọng số từng từ khóa
    3. Với mỗi comment: parse matched_keywords JSON → tra trọng số
    4. Group by image_id: cộng dồn trọng số theo nhóm & cực tính
    5. Tính pressure_score & gán pressure_label
    6. Ghi labels_nlp.csv (1 dòng / image_id)

Cách tính điểm:
    - {group}_neg  : tổng trọng số của các từ negative khớp (>= 0)
    - {group}_pos  : tổng trọng số tuyệt đối của các từ positive khớp (>= 0)
    - total_neg    : sum(layout_neg, color_neg, visual_neg, text_info_neg)
    - total_pos    : sum(layout_pos, color_pos, visual_pos, text_info_pos)
    - pressure_score = total_neg - total_pos
      (dương → nghiêng về High Pressure, âm → Low Pressure)

    ⚠ Nếu image_id không có bất kỳ comment nào khớp từ khóa:
      → pressure_score = 0.0, pressure_label = "neutral"
      (không để NaN hay lỗi ảnh hưởng tới pipeline ML)

Đầu vào:
    data/processed/comment_processed.csv   (output nlp_engine.py)
    src/sentiment/score_logic.json         (trọng số từ khóa)

Đầu ra:
    data/processed/labels_nlp.csv
    Cột: image_id | source
         | layout_neg | layout_pos | color_neg | color_pos
         | visual_neg | visual_pos | text_info_neg | text_info_pos
         | total_neg  | total_pos  | pressure_score | pressure_label

Cách dùng:
    python score_logic.py
    python score_logic.py --comments data/processed/comment_processed.csv
    python score_logic.py --score-json src/sentiment/score_logic.json
    python score_logic.py --output data/processed/labels_nlp.csv
    python score_logic.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
#   src/sentiment/score_logic.py
PROJECT_ROOT        = Path(__file__).resolve().parents[2]
DEFAULT_COMMENTS    = PROJECT_ROOT / "data" / "processed" / "comment_processed.csv"
DEFAULT_SCORE_JSON  = Path(__file__).resolve().parent / "score_logic.json"
DEFAULT_OUTPUT      = PROJECT_ROOT / "data" / "processed" / "labels_nlp.csv"

GROUPS = ["layout", "color", "visual", "text_info"]

GROUP_LABELS = {
    "layout":    "Bố cục (Layout)",
    "color":     "Màu sắc (Color)",
    "visual":    "Độ rõ nét (Visual)",
    "text_info": "Nội dung (Text/Info)",
}

# Trọng số mặc định khi từ khóa có trong KEYWORD_DICT nhưng không có trong score_logic.json
DEFAULT_WEIGHT_NEG =  0.5
DEFAULT_WEIGHT_POS = -0.5

# ── Hệ số nhân theo mức độ cường độ (Intensity Multiplier) ───────────────────
# Được áp dụng lên weight gốc khi comment có từ cường độ đứng trước từ khóa.
# Giá trị này khớp với trường "intensity" trong matched_keywords (output của nlp_engine.py).
#
# Ví dụ minh hoạ:
#   "hơi chói"   → weight_gốc(chói=0.8) × 0.7  = 0.56   (giảm nhẹ)
#   "khá chói"   → weight_gốc(chói=0.8) × 1.0  = 0.80   (giữ nguyên)
#   "rất chói"   → weight_gốc(chói=0.8) × 1.5  = 1.20   (tăng gấp rưỡi)
#   "chói vl"    → weight_gốc(chói=0.8) × 2.0  = 1.60   (tăng gấp đôi)
#
# None (không có từ cường độ) → multiplier = 1.0 (mặc định, không thay đổi).
INTENSITY_MULTIPLIERS: dict[str, float] = {
    "weak":    0.7,   # hơi, chút, nhẹ ...
    "medium":  1.0,   # khá, cũng, vừa ...
    "strong":  1.5,   # rất, quá, siêu, lắm ...
    "extreme": 2.0,   # cực_kỳ, vl, vcl, wtf ...
}


# ── Tải score_logic.json ─────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower().strip()


def load_score_lookup(score_json: Path) -> dict[str, dict[str, float]]:
    """
    Đọc score_logic.json và build lookup:
        { "layout|negative|rối": 0.7, "layout|positive|thoáng": 0.7, ... }

    Giá trị trả về luôn là số dương (absolute weight) kèm polarity riêng.
    Dùng key-pattern: "{group}|{polarity}|{keyword_normalized}"
    """
    if not score_json.is_file():
        print(f"[CANH BAO] Khong tim thay score_logic.json: {score_json}", file=sys.stderr)
        print("           Se dung trong so mac dinh cho tat ca tu khoa.\n", file=sys.stderr)
        return {}

    with open(score_json, encoding="utf-8") as f:
        raw: dict = json.load(f)

    lookup: dict[str, float] = {}
    for group, pol_dict in raw.items():
        for polarity, kw_dict in pol_dict.items():
            for kw, weight in kw_dict.items():
                key = f"{group}|{polarity}|{_normalize(kw)}"
                # Lưu absolute weight (score_logic.json dùng âm cho positive)
                lookup[key] = abs(float(weight))

    return lookup


# ── Tính score cho 1 comment từ hits list ────────────────────────────────────

def score_hits(
    hits: list[dict],
    lookup: dict[str, float],
) -> dict[str, float]:
    """
    Từ list hits của 1 comment, tính contribution theo nhóm & polarity.
    Áp dụng INTENSITY_MULTIPLIERS nếu hit có trường "intensity".

    Returns:
        dict { "layout_neg": float, "layout_pos": float, ... }
        Tất cả giá trị >= 0.
    """
    result = {f"{g}_{p}": 0.0 for g in GROUPS for p in ("neg", "pos")}

    for h in hits:
        group    = h.get("group", "")
        polarity = h.get("polarity", "")
        kw       = _normalize(h.get("keyword", ""))

        if group not in GROUPS or polarity not in ("negative", "positive"):
            continue

        # Tra cứu weight gốc trong lookup
        key = f"{group}|{polarity}|{kw}"
        weight = lookup.get(key)

        if weight is None:
            # Từ khóa không có trong score_logic.json → dùng default
            weight = DEFAULT_WEIGHT_NEG if polarity == "negative" else abs(DEFAULT_WEIGHT_POS)

        # Áp dụng hệ số nhân cường độ (None → multiplier = 1.0, giữ nguyên)
        intensity   = h.get("intensity")          # "weak" | "medium" | "strong" | "extreme" | None
        multiplier  = INTENSITY_MULTIPLIERS.get(intensity, 1.0) if intensity else 1.0
        weight      = weight * multiplier

        col = f"{group}_{'neg' if polarity == 'negative' else 'pos'}"
        result[col] += weight

    return result


# ── Hàm chính ─────────────────────────────────────────────────────────────────

def run_score_logic(
    comments_csv: Path,
    score_json:   Path,
    output_csv:   Path,
    dry_run:      bool = False,
) -> None:
    print("=" * 65)
    print("SCORE LOGIC — AESTHETIC PRESSURE PROJECT")
    print("=" * 65)
    print(f"Comments : {comments_csv}")
    print(f"Weights  : {score_json}")
    print(f"Output   : {output_csv}")
    print("=" * 65 + "\n")

    # ── Đọc dữ liệu ───────────────────────────────────────────────────────────
    if not comments_csv.is_file():
        print(f"[LOI] Khong tim thay: {comments_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(comments_csv, encoding="utf-8-sig", low_memory=False)
    print(f"Doc comment_processed: {len(df):,} dong\n")

    required = {"image_id", "matched_keywords"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[LOI] Thieu cot: {missing}", file=sys.stderr)
        sys.exit(1)

    has_source = "source" in df.columns

    # ── Tải trọng số ─────────────────────────────────────────────────────────
    lookup = load_score_lookup(score_json)
    print(f"Trong so: {len(lookup)} entry tu score_logic.json\n")

    # ── Parse matched_keywords & tính score mỗi comment ──────────────────────
    score_cols = [f"{g}_{p}" for g in GROUPS for p in ("neg", "pos")]

    def parse_and_score(raw_json: str) -> dict[str, float]:
        try:
            hits = json.loads(raw_json) if isinstance(raw_json, str) else []
        except (json.JSONDecodeError, TypeError):
            hits = []
        return score_hits(hits, lookup)

    print("Dang tinh diem pressure...")
    scores_df = pd.DataFrame(
        df["matched_keywords"].fillna("[]").apply(parse_and_score).tolist()
    )

    # ── Gắn thêm image_id và source ──────────────────────────────────────────
    scores_df["image_id"] = df["image_id"].values
    if has_source:
        scores_df["source"] = df["source"].values

    # ── Group by image_id: cộng dồn trọng số ─────────────────────────────────
    group_keys = ["image_id"] + (["source"] if has_source else [])

    # Dùng agg first cho source (giữ source đầu tiên xuất hiện per image_id)
    agg_dict = {col: "sum" for col in score_cols}
    if has_source:
        agg_dict["source"] = "first"

    agg = scores_df.groupby("image_id", sort=False).agg(agg_dict).reset_index()

    # ── Tính total_neg, total_pos, pressure_score, pressure_label ────────────
    agg["total_neg"] = agg[[f"{g}_neg" for g in GROUPS]].sum(axis=1)
    agg["total_pos"] = agg[[f"{g}_pos" for g in GROUPS]].sum(axis=1)
    agg["pressure_score"] = (agg["total_neg"] - agg["total_pos"]).round(4)

    # ⚠ Image không có từ khóa nào khớp → pressure_score = 0.0, label = "neutral"
    has_any_hit = (agg["total_neg"] + agg["total_pos"]) > 0

    def _label(row) -> str:
        if (row["total_neg"] + row["total_pos"]) == 0:
            return "neutral"
        if row["pressure_score"] > 0:
            return "high"
        if row["pressure_score"] < 0:
            return "low"
        return "neutral"

    agg["pressure_label"] = agg.apply(_label, axis=1)

    # ── Làm tròn các cột float ────────────────────────────────────────────────
    for col in score_cols + ["total_neg", "total_pos"]:
        agg[col] = agg[col].round(4)

    # ── Sắp xếp cột output ───────────────────────────────────────────────────
    output_cols = (
        ["image_id"]
        + (["source"] if has_source else [])
        + [f"{g}_neg" for g in GROUPS]
        + [f"{g}_pos" for g in GROUPS]
        + ["total_neg", "total_pos", "pressure_score", "pressure_label"]
    )
    agg = agg[output_cols].sort_values("image_id", ignore_index=True)

    # ── Thống kê ──────────────────────────────────────────────────────────────
    n_total   = len(agg)
    n_high    = (agg["pressure_label"] == "high").sum()
    n_low     = (agg["pressure_label"] == "low").sum()
    n_neutral = (agg["pressure_label"] == "neutral").sum()
    n_no_hit  = (~has_any_hit).sum()

    print(f"\n{'─'*65}")
    print("THONG KE KET QUA (per image_id)")
    print(f"{'─'*65}")
    print(f"  Tong image_id            : {n_total:>7,}")
    print(f"  HIGH pressure  (score>0) : {n_high:>7,}  ({n_high/n_total*100:.1f}%)")
    print(f"  LOW  pressure  (score<0) : {n_low:>7,}  ({n_low/n_total*100:.1f}%)")
    print(f"  NEUTRAL (score=0)        : {n_neutral:>7,}  ({n_neutral/n_total*100:.1f}%)")
    print(f"    Trong do khong co hit  : {n_no_hit:>7,}")
    print(f"\n  Pressure score trung binh: {agg['pressure_score'].mean():>+.4f}")
    print(f"  Pressure score cao nhat  : {agg['pressure_score'].max():>+.4f}")
    print(f"  Pressure score thap nhat : {agg['pressure_score'].min():>+.4f}")

    print(f"\n  DONG GOP TRUNG BINH THEO NHOM (image co hit):")
    hit_df = agg[has_any_hit]
    if not hit_df.empty:
        print(f"  {'Nhom':<25} {'NEG_avg':>9} {'POS_avg':>9}")
        print(f"  {'─'*43}")
        for group, label in GROUP_LABELS.items():
            neg_avg = hit_df[f"{group}_neg"].mean()
            pos_avg = hit_df[f"{group}_pos"].mean()
            print(f"  {label:<25} {neg_avg:>9.4f} {pos_avg:>9.4f}")

    print(f"{'─'*65}")

    if dry_run:
        print(f"\n[DRY-RUN] Mau 15 dong (sap xep theo pressure_score giam dan):")
        pd.set_option("display.max_colwidth", 18)
        pd.set_option("display.float_format", "{:.4f}".format)
        preview_cols = (
            ["image_id"]
            + (["source"] if has_source else [])
            + ["total_neg", "total_pos", "pressure_score", "pressure_label"]
        )
        print(
            agg[preview_cols]
            .sort_values("pressure_score", ascending=False)
            .head(15)
            .to_string(index=False)
        )
        print("\n[DRY-RUN] Khong ghi file.")
        return

    # ── Ghi CSV ───────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_csv, index=False, encoding="utf-8-sig")
    size_kb = output_csv.stat().st_size / 1024
    print(f"\n[XUAT] {output_csv}")
    print(f"       {len(agg):,} dong  |  {size_kb:.1f} KB")
    print("=" * 65)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tinh diem pressure: comment_processed.csv -> labels_nlp.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--comments",    type=Path, default=DEFAULT_COMMENTS,
        help="CSV dau vao tu nlp_engine.py (comment_processed.csv)",
    )
    parser.add_argument(
        "--score-json",  type=Path, default=DEFAULT_SCORE_JSON,
        help="File JSON trong so tu khoa (score_logic.json)",
    )
    parser.add_argument(
        "--output",      type=Path, default=DEFAULT_OUTPUT,
        help="CSV dau ra (labels_nlp.csv)",
    )
    parser.add_argument(
        "--dry-run",     action="store_true",
        help="In 15 dong mau, khong ghi file",
    )
    args = parser.parse_args()
    run_score_logic(args.comments, args.score_json, args.output, args.dry_run)