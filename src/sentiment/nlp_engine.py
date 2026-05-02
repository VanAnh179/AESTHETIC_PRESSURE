"""
nlp_engine.py
=============
Pipeline xử lý NLP cho comment UI/UX.

Input  : merged_comments_raw.csv  (cột: "image_id", "source", "raw_text")
Output : comment_processed.csv    (thêm cột "cleaned_text", "matched_keywords")

Yêu cầu:
    pip install underthesea nltk pandas emoji
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# ── Thư viện NLP ───────────────────────────────────────────────────────────────
try:
    from underthesea import word_tokenize as vn_tokenize
except ImportError:
    print("[LỖI] Thiếu thư viện underthesea. Cài bằng: pip install underthesea")
    sys.exit(1)

try:
    import nltk
    from nltk.tokenize import word_tokenize as en_tokenize
    for _resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(_resource)
        except LookupError:
            nltk.download(_resource.split("/")[-1], quiet=True)
except ImportError:
    print("[LỖI] Thiếu thư viện nltk. Cài bằng: pip install nltk")
    sys.exit(1)

try:
    import emoji as emoji_lib
except ImportError:
    print("[LỖI] Thiếu thư viện emoji. Cài bằng: pip install emoji")
    sys.exit(1)

# ── Đường dẫn mặc định ────────────────────────────────────────────────────────
HERE           = Path(__file__).parent
PROJECT_ROOT   = HERE.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

SCORE_LOGIC_PATH = HERE / "score_logic.json"
INPUT_CSV        = DATA_PROCESSED / "merged_comments_raw.csv"
OUTPUT_CSV       = DATA_PROCESSED / "comment_processed.csv"


# ══════════════════════════════════════════════════════════════════════════════
# BẢNG TỪ CƯỜNG ĐỘ (INTENSITY MODIFIERS)
#
# Mục đích: phát hiện từ đứng TRƯỚC một từ khóa để gắn nhãn mức độ.
# Ví dụ:  "quá phèn"   → keyword "phèn",  intensity = "strong"
#          "hơi rối"    → keyword "rối",   intensity = "weak"
#          "cực kỳ xịn" → keyword "xịn",  intensity = "extreme"
#
# Bốn mức:  weak | medium | strong | extreme
# ══════════════════════════════════════════════════════════════════════════════

INTENSITY_MODIFIERS: dict[str, str] = {
    # ── weak (hơi yếu / một chút) ───────────────────────────────────────────
    "hơi":       "weak",   "hoi":       "weak",
    "chút":      "weak",   "chut":      "weak",
    "tí":        "weak",   "ti":        "weak",
    "tí_chút":   "weak",   "ti_chut":   "weak",
    "một_chút":  "weak",   "mot_chut":  "weak",
    "nhẹ":       "weak",   "nhe":       "weak",
    "khẽ":       "weak",   "khe":       "weak",
    # English weak
    "slightly":  "weak",
    "somewhat":  "weak",
    "little":    "weak",

    # ── medium (khá / vừa phải) ──────────────────────────────────────────────
    "khá":       "medium", "kha":       "medium",
    "cũng":      "medium", "cung":      "medium",
    "vừa":       "medium", "vua":       "medium",
    "tương_đối": "medium", "tuong_doi": "medium",
    "khá_là":    "medium", "kha_la":    "medium",
    "cũng_khá":  "medium",
    # English medium
    "quite":     "medium",
    "pretty":    "medium", 
    "fairly":    "medium",
    "rather":    "medium",

    # ── strong (rất / quá / cực) ─────────────────────────────────────────────
    "rất":       "strong", "rat":       "strong",
    "quá":       "strong", "qua":       "strong",
    "thật_sự":   "strong", "that_su":   "strong",
    "thực_sự":   "strong", "thuc_su":   "strong",
    "cực":       "strong", "cuc":       "strong",
    "siêu":      "strong", "sieu":      "strong",
    "vô_cùng":   "strong", "vo_cung":   "strong",
    "lắm":       "strong", "lam":       "strong",
    "kinh":      "strong",
    "ghê":       "strong", "ghe":       "strong",
    "thật":      "strong", "that":      "strong",
    "quá_là":    "strong", "qua_la":    "strong",
    # English strong
    "very":      "strong",
    "really":    "strong",
    "so":        "strong",
    "too":       "strong",
    "truly":     "strong",
    "highly":    "strong",

    # ── extreme (cực đoan / tiếng lóng mạng) ─────────────────────────────────
    "cực_kỳ":        "extreme", "cuc_ky":     "extreme",
    "tột_độ":        "extreme", "tot_do":     "extreme",
    "vãi":           "extreme", "vai":        "extreme",
    "vl":            "extreme",
    "vcl":           "extreme",
    "wtf":           "extreme",
    "wth":           "extreme",
    "kinh_khủng":    "extreme", "kinh_khung": "extreme",
    "tởm_kinh":      "extreme",
    "đỉnh_của_đỉnh": "extreme",
    # English extreme
    "extremely":     "extreme",
    "incredibly":    "extreme",
    "insanely":      "extreme",
    "absolutely":    "extreme",
    "super":         "extreme",
    "damn":          "extreme",
    "fucking":       "extreme",
}

# ── Regex compile sẵn ─────────────────────────────────────────────────────────
_RE_URL      = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENTION  = re.compile(r"@\w+")
_RE_HASHTAG  = re.compile(r"#\w+")
_RE_REPEAT   = re.compile(r"(.)\1{2,}")  
_RE_SPACES   = re.compile(r"\s+")

_RE_STRIP = re.compile(
    r"[^\w\s:"
    r"\u00C0-\u024F"    # Latin Extended
    r"\u1E00-\u1EFF"    # Latin Extended Additional (tiếng Việt precomposed)
    r"\u0300-\u036F"    # Combining Diacritical Marks
    r"]",
    re.UNICODE,
)

# Nhận diện token emoji sau demojize, ví dụ :fire: hoặc :thumbs_up:
_RE_EMOJI_TOKEN = re.compile(r"(:[a-z0-9_+\-]+:)")

def build_keyword_index(score_logic_path: Path) -> dict[str, dict]:
    """
    Trả về flat dict  {keyword: {"group": ..., "polarity": ...}}.

    Ví dụ:
        "messy"    -> {"group": "layout", "polarity": "negative"}
        ":fire:"   -> {"group": "visual", "polarity": "positive"}
    """
    with score_logic_path.open(encoding="utf-8") as f:
        logic: dict = json.load(f)

    index: dict[str, dict] = {}
    for group, polarities in logic.items():
        for polarity, keywords in polarities.items():
            for keyword in keywords:
                if keyword not in index:
                    index[keyword] = {"group": group, "polarity": polarity}
    return index

def clean_text(text: str) -> str:
    """
    Pipeline làm sạch + "phiên dịch" emoji thành token có thể match.

    Tám bước theo thứ tự:

    [1] DEMOJIZE
        Chuyển emoji Unicode -> ký hiệu văn bản :tên_emoji: bằng thư viện `emoji`.
        Ví dụ:  🔥 -> :fire:    👍 -> :thumbs_up:    💩 -> :pile_of_poo:
        Chuẩn hóa thêm alias đặc biệt: :+1: -> :thumbs_up:, :-1: -> :thumbs_down:
        Bước này PHẢI chạy TRƯỚC khi xóa ký tự đặc biệt để emoji không bị mất.
        Sau bước này, token :fire: sẽ khớp trực tiếp với key ":fire:" trong
        score_logic.json.

    [2] NFC NORMALIZE
        Chuẩn hóa Unicode NFC để hợp nhất ký tự tiếng Việt ghép tổ hợp.
        Tránh lỗi "chữ giống nhau nhưng không so khớp được" do khác cách
        encode (VD: "ệ" viết bằng 2 codepoint vs 1 codepoint precomposed).

    [3] LOWERCASE
        Toàn bộ chữ hoa -> chữ thường.

    [4] NOISE REMOVAL
        Xóa URL, @mention, #hashtag — những phần không mang nghĩa về UI/UX.

    [5] STRIP NOISE CHARS
        Xóa mọi ký tự ngoài tập được phép:
            - \w  : chữ cái, số, dấu gạch dưới (bảo toàn _  trong :smiling_face:)
            - khoảng trắng
            - `:` : bảo toàn dấu hai chấm trong token emoji :fire:
            - Unicode Latin/Việt mở rộng
        Dấu câu thừa, ký hiệu lạ, emoticon text dạng :-) bị xóa hoàn toàn.

    [6] INTENSITY MODIFIERS — KHÔNG LÀM GÌ (bước chủ đích giữ nguyên)
        Các từ cường độ (hơi, khá, rất, quá, cực_kỳ, vl ...) là ký tự \w
        nên tự nhiên được giữ lại sau bước [5]. extract_keywords() sẽ đọc
        context của chúng ở bước sau — KHÔNG được xóa hay thay thế chúng ở đây.

    [7] STRETCH COMPRESSION
        Nén ký tự bị kéo dài >= 3 lần xuống còn 2.
        "phènnnnnn" -> "phènn"    "quáaaa" -> "quáa"
        (Giữ 2 thay vì 1 để tokenizer nhận diện kiểu viết nhấn mạnh của mạng.)

    [8] WHITESPACE NORMALIZATION
        Thu gọn khoảng trắng thừa, strip đầu cuối.

    Parameters
    ----------
    text : str  Văn bản gốc từ comment mạng xã hội.

    Returns
    -------
    str  Chuỗi đã làm sạch, sẵn sàng đưa vào tokenize().
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # [1] DEMOJIZE ─────────────────────────────────────────────────────────────
    # language="alias" dùng tên alias ngắn phổ biến (:fire:, :heart: ...)
    text = emoji_lib.demojize(text, language="alias")
    # Chuẩn hóa alias đặc biệt có ký tự +/- sang dạng chữ thuần
    text = text.replace(":+1:",  ":thumbs_up:")
    text = text.replace(":-1:", ":thumbs_down:")

    # [2] NFC NORMALIZE ────────────────────────────────────────────────────────
    text = unicodedata.normalize("NFC", text)

    # [3] LOWERCASE ────────────────────────────────────────────────────────────
    text = text.lower()

    # [4] NOISE REMOVAL ────────────────────────────────────────────────────────
    text = _RE_URL.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)
    text = _RE_HASHTAG.sub(" ", text)

    # [5] STRIP NOISE CHARS ────────────────────────────────────────────────────
    # Giữ lại dấu `:` để token emoji :fire: không bị bẻ vỡ.
    # \w đã bao gồm [a-z0-9_] nên _ trong :smiling_face_with_heart_eyes: an toàn.
    text = _RE_STRIP.sub(" ", text)

    # [6] INTENSITY MODIFIERS — bước trống có chủ đích, xem docstring ─────────

    # [7] STRETCH COMPRESSION ──────────────────────────────────────────────────
    text = _RE_REPEAT.sub(r"\1\1", text)

    # [8] WHITESPACE NORMALIZATION ─────────────────────────────────────────────
    text = _RE_SPACES.sub(" ", text).strip()

    return text

def _is_vietnamese(text: str) -> bool:
    """Heuristic: văn bản chứa ký tự có dấu Việt -> coi là tiếng Việt."""
    vn_chars = set(
        "àáâãèéêìíòóôõùúýăđơư"
        "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    )
    return any(ch in vn_chars for ch in text)


def tokenize(text: str) -> str:
    """
    Tokenize thông minh theo ngôn ngữ, bảo toàn token emoji :snake_case:.

    Chiến lược:
    - Tách token emoji ra trước bằng regex _RE_EMOJI_TOKEN.
    - Phần văn bản xen kẽ:
        · Có ký tự tiếng Việt -> underthesea (cụm từ nối dấu _)
        · Thuần Latin/Anh      -> NLTK word_tokenize
    - Ghép tất cả lại bằng khoảng trắng.

    Returns
    -------
    str  Chuỗi token cách nhau bằng dấu cách.
    """
    if not text:
        return ""

    parts  = _RE_EMOJI_TOKEN.split(text)   # luân phiên [text, :emoji:, text, ...]
    result = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _RE_EMOJI_TOKEN.fullmatch(part):
            # Emoji token giữ nguyên, không qua tokenizer
            result.append(part)
        elif _is_vietnamese(part):
            result.append(vn_tokenize(part, format="text"))
        else:
            tokens = en_tokenize(part)
            result.append(" ".join(tokens))

    return " ".join(result)

def extract_keywords(
    tokenized_text: str,
    keyword_index: dict[str, dict],
) -> list[dict]:
    """
    So khớp từng token với keyword_index.
    Với mỗi match, nhìn lùi 1-2 token để phát hiện từ cường độ gần nhất.

    Ví dụ input tokens: ["quá", "phèn", "lắm", ":fire:"]
        -> "phèn"  : intensity = "strong"  (do "quá" đứng trước)
        -> ":fire:" : intensity = None      (không có từ cường độ trước đó)

    Returns
    -------
    list[dict]  Mỗi phần tử:
        {
            "keyword":   "phèn",
            "group":     "color",
            "polarity":  "negative",
            "intensity": "strong"    # hoặc null nếu không có từ cường độ
        }
    Dedup theo keyword (không trả trùng lặp).
    """
    tokens = tokenized_text.split()
    seen: set[str] = set()
    matched: list[dict] = []

    for idx, token in enumerate(tokens):
        if token not in keyword_index or token in seen:
            continue

        seen.add(token)

        # Nhìn lùi tối đa 2 vị trí để tìm từ cường độ gần nhất
        intensity: str | None = None
        for lookback in (1, 2):
            prev_idx = idx - lookback
            if prev_idx >= 0 and tokens[prev_idx] in INTENSITY_MODIFIERS:
                intensity = INTENSITY_MODIFIERS[tokens[prev_idx]]
                break   # lấy từ cường độ gần nhất, dừng tìm tiếp

        matched.append(
            {
                "keyword":   token,
                "group":     keyword_index[token]["group"],
                "polarity":  keyword_index[token]["polarity"],
                "intensity": intensity,
            }
        )

    return matched

def process_csv(
    input_path: Path = INPUT_CSV,
    output_path: Path = OUTPUT_CSV,
    score_logic_path: Path = SCORE_LOGIC_PATH,
    text_column: str = "raw_text",
) -> pd.DataFrame:
    """
    Đọc merged_comments_raw.csv -> pipeline NLP -> ghi comment_processed.csv.

    Cột output thêm vào:
        cleaned_text     : văn bản sau clean_text() (có emoji dạng :fire:)
        matched_keywords : JSON array của {keyword, group, polarity, intensity}

    Returns
    -------
    DataFrame đã xử lý.
    """
    # [1] Load keyword index
    print(f"[1/5] Đang load keyword index từ: {score_logic_path}")
    keyword_index = build_keyword_index(score_logic_path)
    print(f"      -> {len(keyword_index)} keywords.")

    # [2] Đọc CSV
    print(f"[2/5] Đang đọc input: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    if text_column not in df.columns:
        raise ValueError(
            f"Cột '{text_column}' không tồn tại. "
            f"Các cột hiện có: {list(df.columns)}"
        )
    total = len(df)
    print(f"      -> {total} dòng.")

    # [3] Làm sạch (demojize + strip + stretch)
    print("[3/5] Đang chạy clean_text…")
    df["cleaned_text"] = df[text_column].apply(clean_text)

    # [4] Tokenize (bảo toàn :emoji:)
    print("[4/5] Đang tokenize (underthesea + NLTK)…")
    df["_tokenized"] = df["cleaned_text"].apply(tokenize)

    # [5] Quét từ khóa + gắn nhãn cường độ
    print("[5/5] Đang quét từ khóa & gắn intensity…")

    def _row_to_json(tok: str) -> str:
        return json.dumps(extract_keywords(tok, keyword_index), ensure_ascii=False)

    df["matched_keywords"] = df["_tokenized"].apply(_row_to_json)
    df.drop(columns=["_tokenized"], inplace=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    hit_count   = df["matched_keywords"].apply(lambda s: s != "[]").sum()
    emoji_hits  = df["cleaned_text"].apply(lambda s: ":" in s).sum()

    print(
        f"\n[OK] Hoàn thành!\n"
        f"   Tổng dòng        : {total}\n"
        f"   Có từ khóa       : {hit_count}\n"
        f"   Không có từ khóa : {total - hit_count}\n"
        f"   Dòng chứa emoji  : {emoji_hits}\n"
        f"   Cột output       : image_id | source | raw_text | cleaned_text | matched_keywords\n"
        f"   File CSV         : {output_path}"
    )
    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NLP Engine – quét từ khóa UI/UX từ comment CSV."
    )
    parser.add_argument("--input",       default=str(INPUT_CSV))
    parser.add_argument("--output",      default=str(OUTPUT_CSV))
    parser.add_argument("--score-logic", default=str(SCORE_LOGIC_PATH))
    parser.add_argument("--text-col",    default="raw_text")
    args = parser.parse_args()

    process_csv(
        input_path=Path(args.input),
        output_path=Path(args.output),
        score_logic_path=Path(args.score_logic),
        text_column=args.text_col,
    )