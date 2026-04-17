"""
Facebook Scraper - Auto version
Tự động cà dữ liệu từ danh sách fanpage
"""
import sys
import io

# Fix encoding for Vietnamese characters and emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import time
import os
import base64
import re
import urllib.parse
from html import unescape

try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ Error: python-dotenv not installed. Run: pip install python-dotenv")
    exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

ENV_PATHS = [
    os.path.join(PROJECT_ROOT, '.env'),
    os.path.join(SCRIPT_DIR, '.env'),
]
for env_path in ENV_PATHS:
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        break

# Import data extractor module
try:
    from scraping.fb_scraper.data_extractor import run_data_extraction
except ImportError:
    try:
        # Fallback when running this file directly: python src/scraping/fb_scraper/main.py
        from data_extractor import run_data_extraction
    except ImportError:
        print("⚠️ Warning: data_extractor module not found")
        run_data_extraction = None

# ========= CONFIG =========
GRAPHQL_URL = "https://www.facebook.com/api/graphql/"

# Read DOC_ID from .env, fallback to default
DOC_ID_POSTS = os.getenv('DOC_ID_POSTS', "25430544756617998")
DOC_ID_COMMENTS_ROOT = os.getenv('DOC_ID_COMMENTS_ROOT', "26567970629522780")
DOC_ID_COMMENTS_PAGINATION = os.getenv('DOC_ID_COMMENTS_PAGINATION', "26233394729665676")
DOC_ID_COMMENTS_LEGACY = os.getenv('DOC_ID_COMMENTS_LEGACY', "25550760954572974")
DOC_ID_COMMENT_REPLIES = os.getenv('DOC_ID_COMMENT_REPLIES', "26570577339199586")
DOC_ID_REACTIONS_DIALOG = os.getenv('DOC_ID_REACTIONS_DIALOG', "33437545572555426")
DOC_ID_REACTIONS_TOOLTIP = os.getenv('DOC_ID_REACTIONS_TOOLTIP', "29619394521038216")
COMMENTS_INTENT_TOKEN = os.getenv('COMMENTS_INTENT_TOKEN', "RANKED_UNFILTERED_CHRONOLOGICAL_REPLIES_INTENT_V1")
COMMENTS_FEED_LOCATION = os.getenv('COMMENTS_FEED_LOCATION', "COMET_MEDIA_VIEWER")

REACTION_FIELD_BY_ID = {
    "1635855486666999": "react_like",
    "1678524932434102": "react_love",
    "613557422527858": "react_care",
    "478547315650144": "react_wow",
    "444813342392137": "react_angry",
}

BASE_HEADERS = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "content-type": "application/x-www-form-urlencoded",
    "origin": "https://www.facebook.com",
}

# Get credentials from .env
COOKIES = {}
FB_DTSG = os.getenv('FB_DTSG', '').strip().strip('"')

c_user = os.getenv('COOKIES_C_USER', '').strip().strip('"')
xs = os.getenv('COOKIES_XS', '').strip().strip('"')
datr = os.getenv('COOKIES_DATR', '').strip().strip('"')

if c_user:
    COOKIES['c_user'] = c_user
if xs:
    COOKIES['xs'] = xs  
if datr:
    COOKIES['datr'] = datr

PROXY = os.getenv('PROXY', '').strip()
PROXIES = {'http': PROXY, 'https': PROXY} if PROXY else None

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
SAVE_IMAGES = True

# ============= DANH SÁCH FANPAGE CẦN CÀ =============
# Format: Full URL từ Facebook
FANPAGES = [
    {
        "url": "https://www.facebook.com/SamsungVietnam",
        "name": "Samsung"
    },
    # Thêm fanpage khác ở đây (dạng full URL)
]

POST_LIMIT_PER_FANPAGE = 3  # Chỉ cà 3 posts per page

# ========= HELPER FUNCTIONS =========
def sanitize_fanpage_name(name):
    """Sanitize fanpage name để dùng làm tên folder (chỉ cho phép alphanumeric + space/dash/underscore)"""
    if not name:
        return "Unknown"
    # Loại bỏ các ký tự không hợp lệ, chỉ giữ alphanumeric, space, dash, underscore
    sanitized = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    return sanitized or "Unknown"

# ========= PARSER =========
def extract_data_blocks(raw_text):
    blocks = []
    i = 0
    n = len(raw_text)
    while True:
        idx = raw_text.find('"data"', i)
        if idx == -1:
            break
        brace_start = raw_text.find('{', idx)
        if brace_start == -1:
            break
        depth = 0
        for j in range(brace_start, n):
            if raw_text[j] == '{':
                depth += 1
            elif raw_text[j] == '}':
                depth -= 1
                if depth == 0:
                    block_text = raw_text[brace_start:j+1]
                    try:
                        block = json.loads(block_text)
                        blocks.append(block)
                    except:
                        pass
                    i = j + 1
                    break
        else:
            break
    return blocks

def clean_data_blocks(blocks):
    cleaned = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block.pop("errors", None)
        block.pop("extensions", None)
        cleaned.append(block)
    return cleaned

def parse_fb_response(text):
    text = text.replace("for (;;);", "").strip()
    extracted = extract_data_blocks(text)
    cleaned = clean_data_blocks(extracted)
    return cleaned

def fb_json(response_text):
    text = (response_text or "").strip()
    if text.startswith("for (;;);"):
        text = text[len("for (;;);"):]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            continue

    try:
        return json.loads(text)
    except Exception:
        return {}

def parse_graphql_blocks(response_text):
    blocks = parse_fb_response(response_text or "")
    if blocks:
        return blocks

    single = fb_json(response_text or "")
    if isinstance(single, dict) and single:
        return [single]

    return []

# ========= NETWORK =========
def retry_request(url, headers, data, proxies, cookies=None, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, data=data, 
                            proxies=proxies, cookies=cookies, timeout=30)
            if r.status_code == 200:
                return r
            else:
                print(f"    ⚠️ Attempt {attempt}/{max_retries}: Status {r.status_code}")
        except Exception as e:
            print(f"    ⚠️ Attempt {attempt}/{max_retries}: {str(e)}")
        
        if attempt < max_retries:
            time.sleep(attempt * 2)
    
    return None

# ========= COMMENTS =========
def _get_nested_dict(data, path):
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur

def _iter_values_for_key(data, key_name):
    values = []

    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k) == key_name:
                    values.append(v)
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)
    return values

def _extract_likes_shares_from_payload(payload):
    likes = 0
    shares = 0

    like_keys = ("i18n_reaction_count", "likers_count", "liking_subscribers_count", "reaction_count")
    share_keys = ("i18n_share_count", "share_count")

    for key in like_keys:
        for value in _iter_values_for_key(payload, key):
            likes = max(likes, _parse_count(value))

    for key in share_keys:
        for value in _iter_values_for_key(payload, key):
            shares = max(shares, _parse_count(value))

    if likes == 0:
        likes = _find_max_count_by_key(
            payload,
            include_keywords=("react", "liker", "liking", "reaction", "top_reaction"),
            exclude_keywords=("comment", "share"),
            require_keywords=("count", "i18n"),
            blocked_keywords=("id", "fbid", "legacy", "token", "cursor", "url", "uri")
        )

    if shares == 0:
        shares = _find_max_count_by_key(
            payload,
            include_keywords=("share",),
            exclude_keywords=("comment", "react"),
            require_keywords=("count", "i18n"),
            blocked_keywords=("id", "fbid", "legacy", "token", "cursor", "url", "uri")
        )

    return _sanitize_metric(likes), _sanitize_metric(shares)

def _parse_count(value):
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip().upper().replace(" ", "")
        if not s:
            return 0

        match = re.search(r"([0-9]+(?:[\.,][0-9]+)?)([KMB])", s)
        if match:
            base = match.group(1).replace(",", ".")
            try:
                number = float(base)
                factor = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(match.group(2), 1)
                return int(number * factor)
            except Exception:
                pass

        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 0
    if isinstance(value, dict):
        for key in ("count", "count_reduced", "total_count", "value", "text"):
            if key in value:
                parsed = _parse_count(value.get(key))
                if parsed > 0:
                    return parsed
    return 0

def _find_max_count_by_key(data, include_keywords, exclude_keywords=(), require_keywords=(), blocked_keywords=()):
    max_value = 0

    def _walk(obj):
        nonlocal max_value
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()
                include_ok = any(word in key for word in include_keywords)
                exclude_ok = not any(word in key for word in exclude_keywords)
                require_ok = True if not require_keywords else any(word in key for word in require_keywords)
                blocked_ok = not any(word in key for word in blocked_keywords)

                if include_ok and exclude_ok and require_ok and blocked_ok:
                    parsed = _parse_count(v)
                    if parsed > max_value:
                        max_value = parsed
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)
    return max_value

def _sanitize_metric(value):
    try:
        val = int(value or 0)
    except Exception:
        return 0

    if val < 0:
        return 0

    # Chặn parse nhầm ID thành count (thường 13-17 chữ số)
    if val > 1_000_000_000:
        return 0

    return val

def _extract_max_by_regex(text, patterns):
    max_val = 0
    for pattern in patterns:
        for m in re.findall(pattern, text, flags=re.IGNORECASE):
            val = _parse_count(m)
            if val > max_val:
                max_val = val
    return max_val

def _extract_metrics_from_plugin_html(html_text):
    likes = 0
    comments = 0
    shares = 0

    metric_blocks = re.findall(r'<div[^>]*title="([^"]+)"[^>]*>(.*?)</div>', html_text, flags=re.IGNORECASE | re.DOTALL)
    for raw_title, inner_html in metric_blocks:
        title = unescape(raw_title or "").strip().lower()
        inner_text = re.sub(r"<[^>]+>", "", inner_html or "")
        value = _parse_count(unescape(inner_text).strip())
        if value <= 0:
            continue

        if "chia sẻ" in title or "share" in title:
            shares = max(shares, value)
        elif "bình luận" in title or "comment" in title:
            comments = max(comments, value)
        elif "thích" in title or "like" in title or "reaction" in title:
            likes = max(likes, value)

    return likes, comments, shares

def _extract_page_slug_from_url(url):
    if not url:
        return None
    m = re.search(r"facebook\.com/([^/?#]+)/?", url)
    if not m:
        return None
    slug = (m.group(1) or "").strip()
    if not slug:
        return None
    if slug in {"profile.php", "permalink.php", "story.php", "posts"}:
        return None
    return slug

def _extract_page_slug_from_node(node, fallback_page_url=None):
    actor_url = None
    try:
        actors = (
            node.get("comet_sections", {})
            .get("content", {})
            .get("story", {})
            .get("actors", [])
        )
        if actors and isinstance(actors[0], dict):
            actor_url = actors[0].get("url")
    except Exception:
        actor_url = None

    return _extract_page_slug_from_url(actor_url) or _extract_page_slug_from_url(fallback_page_url)

def _build_fallback_post_urls(post_id, owner_id=None, page_slug=None, permalink=None):
    urls = []
    if permalink:
        urls.append(permalink)
    if page_slug and post_id:
        urls.append(f"https://www.facebook.com/{page_slug}/posts/{post_id}")
    if post_id:
        urls.append(f"https://www.facebook.com/{post_id}")
    if post_id and owner_id:
        urls.append(f"https://www.facebook.com/story.php?story_fbid={post_id}&id={owner_id}")

    deduped = []
    seen = set()
    for u in urls:
        if not u or u in seen:
            continue
        seen.add(u)
        deduped.append(u)
    return deduped

def fetch_post_engagement_fallback(post_id, owner_id=None, cookies=None, page_slug=None, permalink=None):
    """Fallback: lấy like/comment/share từ plugin endpoint của permalink post."""
    if not post_id:
        return 0, 0, 0

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
        }

        best_likes = 0
        best_comments = 0
        best_shares = 0

        for post_url in _build_fallback_post_urls(
            post_id,
            owner_id=owner_id,
            page_slug=page_slug,
            permalink=permalink,
        ):
            plugin_url = (
                "https://www.facebook.com/plugins/post.php?href="
                + urllib.parse.quote(post_url, safe="")
                + "&show_text=true&width=500"
            )

            r = requests.get(plugin_url, headers=headers, cookies=cookies, proxies=PROXIES, timeout=30)
            if r.status_code != 200:
                continue

            likes, comments, shares = _extract_metrics_from_plugin_html(r.text)
            best_likes = max(best_likes, likes)
            best_comments = max(best_comments, comments)
            best_shares = max(best_shares, shares)

        # Nếu plugin không trả về thì thử parse nhanh từ HTML permalink.
        if best_likes == 0 and best_shares == 0:
            for post_url in _build_fallback_post_urls(post_id, owner_id=owner_id, page_slug=page_slug, permalink=permalink):
                r = requests.get(post_url, headers=headers, cookies=cookies, proxies=PROXIES, timeout=30)
                if r.status_code != 200:
                    continue

                html = r.text
                best_likes = max(best_likes, _extract_max_by_regex(html, [
                    r'"i18n_reaction_count"\s*:\s*"([^\"]+)"',
                    r'"reaction_count"\s*:\s*\{[^\}]*"count"\s*:\s*([0-9]+)',
                    r'"reactors"\s*:\s*\{[^\}]*"count"\s*:\s*([0-9]+)',
                    r'([0-9][0-9\.,KMBkmb]*)\s*lượt\s*thích',
                    r'([0-9][0-9\.,KMBkmb]*)\s*reactions?',
                ]))
                best_shares = max(best_shares, _extract_max_by_regex(html, [
                    r'"i18n_share_count"\s*:\s*"([^\"]+)"',
                    r'"share_count"\s*:\s*\{[^\}]*"count"\s*:\s*([0-9]+)',
                    r'"share_count"\s*:\s*([0-9]+)',
                    r'([0-9][0-9\.,KMBkmb]*)\s*lượt\s*chia\s*sẻ',
                    r'([0-9][0-9\.,KMBkmb]*)\s*shares?',
                ]))

        return _sanitize_metric(best_likes), _sanitize_metric(best_comments), _sanitize_metric(best_shares)
    except Exception:
        return 0, 0, 0

def _extract_count_from_feedback(feedback):
    if not isinstance(feedback, dict):
        return 0, 0

    like_values = [
        feedback.get("likers_count"),
        feedback.get("liking_subscribers_count"),
        feedback.get("i18n_reaction_count"),
        _get_nested_dict(feedback, ["reactors", "count"]),
        _get_nested_dict(feedback, ["reactors", "count_reduced"]),
        _get_nested_dict(feedback, ["likers", "count"]),
        _get_nested_dict(feedback, ["likers", "count_reduced"]),
        _get_nested_dict(feedback, ["reaction_count", "count"]),
        _get_nested_dict(feedback, ["reaction_count", "count_reduced"]),
    ]

    for v in _iter_values_for_key(feedback, "i18n_reaction_count"):
        like_values.append(v)

    share_values = [
        feedback.get("share_count"),
        feedback.get("i18n_share_count"),
        _get_nested_dict(feedback, ["share_count", "count"]),
        _get_nested_dict(feedback, ["share_count", "count_reduced"]),
    ]

    for v in _iter_values_for_key(feedback, "i18n_share_count"):
        share_values.append(v)

    likes = 0
    for candidate in like_values:
        likes = max(likes, _parse_count(candidate))

    shares = 0
    for candidate in share_values:
        shares = max(shares, _parse_count(candidate))

    return likes, shares

def _collect_feedback_candidates(node):
    candidates = [node.get("feedback")]
    candidates.append(_get_nested_dict(node, ["feedback", "comet_ufi_summary_and_actions_renderer", "feedback"]))

    # Common deep paths in timeline payload
    candidates.append(_get_nested_dict(node, [
        "comet_sections", "feedback", "story", "story_ufi_container", "story",
        "feedback_context", "feedback_target_with_context", "comet_ufi_summary_and_actions_renderer", "feedback"
    ]))
    candidates.append(_get_nested_dict(node, [
        "comet_sections", "feedback", "story", "story_ufi_container", "story",
        "feedback_context", "feedback_target_with_context"
    ]))
    candidates.append(_get_nested_dict(node, [
        "comet_sections", "feedback", "story", "feedback_context",
        "feedback_target_with_context", "comet_ufi_summary_and_actions_renderer", "feedback"
    ]))
    candidates.append(_get_nested_dict(node, [
        "comet_sections", "feedback", "story", "feedback_context", "feedback_target_with_context"
    ]))

    # Legacy renderer variants
    candidates.append(_get_nested_dict(node, ["comet_sections", "feedback", "story", "feedback_context", "feedback_target_with_context", "ufi_renderer", "feedback"]))
    candidates.append(_get_nested_dict(node, ["comet_sections", "feedback", "story", "feedback", "ufi_renderer", "feedback"]))
    return [c for c in candidates if isinstance(c, dict)]

def extract_likes_shares(node):
    likes = 0
    shares = 0

    for feedback in _collect_feedback_candidates(node):
        l, s = _extract_count_from_feedback(feedback)
        likes = max(likes, l)
        shares = max(shares, s)

    if likes == 0:
        likes = _find_max_count_by_key(
            node,
            include_keywords=("react", "liker", "liking", "reaction", "top_reaction"),
            exclude_keywords=("comment", "share"),
            require_keywords=("count", "i18n"),
            blocked_keywords=("id", "fbid", "legacy", "token", "cursor", "url", "uri")
        )

    if shares == 0:
        shares = _find_max_count_by_key(
            node,
            include_keywords=("share",),
            exclude_keywords=("comment", "react"),
            require_keywords=("count", "i18n"),
            blocked_keywords=("id", "fbid", "legacy", "token", "cursor", "url", "uri")
        )

    return _sanitize_metric(likes), _sanitize_metric(shares)

def _reaction_bucket(item):
    if not isinstance(item, dict):
        return None

    reaction = item.get("reaction") or {}
    reaction_id = str(reaction.get("id") or "")
    if reaction_id in REACTION_FIELD_BY_ID:
        return REACTION_FIELD_BY_ID[reaction_id]

    label = (reaction.get("localized_name") or "").strip().lower()
    if not label:
        return None
    if "thích" in label or "like" in label:
        return "react_like"
    if "yêu" in label or "love" in label:
        return "react_love"
    if "thương" in label or "care" in label:
        return "react_care"
    if "wow" in label:
        return "react_wow"
    if "phẫn nộ" in label or "angry" in label:
        return "react_angry"
    return None

def fetch_reactions_breakdown(feedback_id, cookies=None):
    result = {
        "likes": 0,
        "react_like": 0,
        "react_love": 0,
        "react_care": 0,
        "react_wow": 0,
        "react_angry": 0,
    }
    if not feedback_id:
        return result

    try:
        # Query nhẹ để lấy tổng số người react.
        tooltip_headers = {**BASE_HEADERS, "x-fb-friendly-name": "CometUFIReactionIconTooltipContentQuery"}
        tooltip_data = {
            "av": cookies.get("c_user", "0") if cookies else "0",
            "__user": cookies.get("c_user", "0") if cookies else "0",
            "__a": "1",
            "fb_dtsg": FB_DTSG if FB_DTSG else "",
            "doc_id": DOC_ID_REACTIONS_TOOLTIP,
            "variables": json.dumps({
                "feedbackTargetID": feedback_id,
                "reactionID": None,
            })
        }
        rt = retry_request(GRAPHQL_URL, tooltip_headers, tooltip_data, PROXIES, cookies)
        if rt:
            for block in parse_graphql_blocks(rt.text):
                root = block.get("data") if isinstance(block, dict) and isinstance(block.get("data"), dict) else block
                if not isinstance(root, dict):
                    continue
                feedback = root.get("feedback") if isinstance(root.get("feedback"), dict) else None
                if not feedback:
                    continue
                fb_id = str(feedback.get("id") or "")
                if fb_id and fb_id != str(feedback_id):
                    continue
                result["likes"] = max(result["likes"], _parse_count(_get_nested_dict(feedback, ["reactors", "count"])))

        headers = {**BASE_HEADERS, "x-fb-friendly-name": "CometUFIReactionsDialogQuery"}
        data = {
            "av": cookies.get("c_user", "0") if cookies else "0",
            "__user": cookies.get("c_user", "0") if cookies else "0",
            "__a": "1",
            "fb_dtsg": FB_DTSG if FB_DTSG else "",
            "doc_id": DOC_ID_REACTIONS_DIALOG,
            "variables": json.dumps({
                "feedbackTargetID": feedback_id,
                "reactionID": None,
                "scale": 2,
            })
        }

        r = retry_request(GRAPHQL_URL, headers, data, PROXIES, cookies)
        if not r:
            return result

        blocks = parse_graphql_blocks(r.text)
        chosen = None

        for block in blocks:
            root = block.get("data") if isinstance(block, dict) and isinstance(block.get("data"), dict) else block
            if not isinstance(root, dict):
                continue

            node = root.get("node") if isinstance(root.get("node"), dict) else None
            if not node:
                continue

            node_id = str(node.get("id") or "")
            if node_id != str(feedback_id):
                continue

            top_reactions = node.get("top_reactions") if isinstance(node.get("top_reactions"), dict) else {}
            summary_items = top_reactions.get("summary") or []
            candidate = {
                "likes": 0,
                "react_like": 0,
                "react_love": 0,
                "react_care": 0,
                "react_wow": 0,
                "react_angry": 0,
            }

            for item in summary_items:
                candidate["likes"] += _parse_count(item.get("reaction_count"))
                bucket = _reaction_bucket(item)
                if not bucket:
                    continue
                candidate[bucket] = max(candidate[bucket], _parse_count(item.get("reaction_count")))

            # Ưu tiên node match đúng feedback_id và có likes lớn nhất.
            if not chosen or candidate["likes"] > chosen["likes"]:
                chosen = candidate

        if chosen:
            result.update(chosen)

        if result["likes"] == 0:
            result["likes"] = sum(result[k] for k in ("react_like", "react_love", "react_care", "react_wow", "react_angry"))
    except Exception:
        return result

    return result

def _extract_comments_block(root):
    node = root.get("node") if isinstance(root.get("node"), dict) else {}
    return (
        _get_nested_dict(node, ["comment_rendering_instance_for_feed_location", "comments"])
        or _get_nested_dict(node, ["comment_rendering_instance", "comments"])
        or node.get("comments")
        or {}
    )

def _request_comments_blocks(doc_id, friendly_name, variables, cookies=None):
    headers = {**BASE_HEADERS, "x-fb-friendly-name": friendly_name}
    data = {
        "av": cookies.get("c_user", "0") if cookies else "0",
        "__user": cookies.get("c_user", "0") if cookies else "0",
        "__a": "1",
        "fb_dtsg": FB_DTSG if FB_DTSG else "",
        "doc_id": doc_id,
        "variables": json.dumps(variables),
    }

    r = retry_request(GRAPHQL_URL, headers, data, PROXIES, cookies)
    if not r:
        return []
    return parse_graphql_blocks(r.text)

def fetch_replies_for_comment(comment_feedback_id, expansion_token, cookies=None):
    if not comment_feedback_id or not expansion_token:
        return []

    replies = []
    seen_reply_ids = set()
    token = expansion_token
    page = 0

    while token and page < 120:
        try:
            headers = {**BASE_HEADERS, "x-fb-friendly-name": "Depth1CommentsListPaginationQuery"}
            data = {
                "av": cookies.get("c_user", "0") if cookies else "0",
                "__user": cookies.get("c_user", "0") if cookies else "0",
                "__a": "1",
                "fb_dtsg": FB_DTSG if FB_DTSG else "",
                "doc_id": DOC_ID_COMMENT_REPLIES,
                "variables": json.dumps({
                    "clientKey": None,
                    "expansionToken": token,
                    "feedLocation": "POST_PERMALINK_DIALOG",
                    "focusCommentID": None,
                    "scale": 2,
                    "useDefaultActor": False,
                    "id": comment_feedback_id
                })
            }

            r = retry_request(GRAPHQL_URL, headers, data, PROXIES, cookies)
            if not r:
                break

            blocks = parse_graphql_blocks(r.text)
            if not blocks:
                break

            has_next = False
            next_token = None
            new_items = 0

            for block in blocks:
                root = block.get("data") if isinstance(block, dict) and isinstance(block.get("data"), dict) else block
                if not isinstance(root, dict):
                    continue

                replies_connection = _get_nested_dict(root, ["node", "replies_connection"]) or {}
                for e in replies_connection.get("edges") or []:
                    n = (e or {}).get("node") or {}
                    reply_id = str(n.get("legacy_fbid") or n.get("id") or "")
                    text = (n.get("body") or {}).get("text", "")
                    clean_text = re.sub(r"\s+", " ", (text or "")).strip()
                    if not clean_text:
                        clean_text = f"[non-text-reply:{reply_id or 'unknown'}]"
                    if reply_id and reply_id in seen_reply_ids:
                        continue
                    if reply_id:
                        seen_reply_ids.add(reply_id)
                    replies.append({"id": reply_id, "text": clean_text})
                    new_items += 1

                page_info = replies_connection.get("page_info") or {}
                has_next = has_next or bool(page_info.get("has_next_page"))
                if page_info.get("end_cursor"):
                    next_token = page_info.get("end_cursor")

            if not has_next or not next_token or next_token == token:
                break
            if new_items == 0 and next_token == token:
                break

            token = next_token
            page += 1
            time.sleep(0.2)
        except Exception:
            break

    return replies

def fetch_comments_from_feedback(feedback_id, cookies=None, max_pages=None):
    """Fetch comments + replies từ feedback_id với nhiều chiến lược để tăng độ phủ."""
    results = []
    seen_comment_ids = set()
    seen_reply_ids = set()
    seen_text_fallback = set()
    metrics_likes = 0
    metrics_shares = 0
    comment_total = 0

    if not feedback_id:
        return results, {"likes": 0, "shares": 0, "comment_total": 0}

    strategies = [
        {
            "intent": COMMENTS_INTENT_TOKEN,
            "location": COMMENTS_FEED_LOCATION,
            "root_doc": DOC_ID_COMMENTS_ROOT,
            "root_name": "CommentListComponentsRootQuery",
            "page_doc": DOC_ID_COMMENTS_PAGINATION,
            "page_name": "CommentsListComponentsPaginationQuery",
            "use_root": True,
        },
        {
            "intent": "CHRONOLOGICAL_UNFILTERED_INTENT_V1",
            "location": "COMET_MEDIA_VIEWER",
            "root_doc": DOC_ID_COMMENTS_ROOT,
            "root_name": "CommentListComponentsRootQuery",
            "page_doc": DOC_ID_COMMENTS_PAGINATION,
            "page_name": "CommentsListComponentsPaginationQuery",
            "use_root": True,
        },
        {
            "intent": "REVERSE_CHRONOLOGICAL_UNFILTERED_INTENT_V1",
            "location": "COMET_MEDIA_VIEWER",
            "root_doc": DOC_ID_COMMENTS_ROOT,
            "root_name": "CommentListComponentsRootQuery",
            "page_doc": DOC_ID_COMMENTS_PAGINATION,
            "page_name": "CommentsListComponentsPaginationQuery",
            "use_root": True,
        },
        {
            "intent": "REVERSE_CHRONOLOGICAL_UNFILTERED_INTENT_V1",
            "location": "DEDICATED_COMMENTING_SURFACE",
            "root_doc": DOC_ID_COMMENTS_LEGACY,
            "root_name": "CommentsListComponentsPaginationQuery",
            "page_doc": DOC_ID_COMMENTS_LEGACY,
            "page_name": "CommentsListComponentsPaginationQuery",
            "use_root": False,
        },
        {
            "intent": "RANKED_UNFILTERED_CHRONOLOGICAL_REPLIES_INTENT_V1",
            "location": "DEDICATED_COMMENTING_SURFACE",
            "root_doc": DOC_ID_COMMENTS_ROOT,
            "root_name": "CommentListComponentsRootQuery",
            "page_doc": DOC_ID_COMMENTS_PAGINATION,
            "page_name": "CommentsListComponentsPaginationQuery",
            "use_root": True,
        },
        {
            "intent": "RANKED_UNFILTERED_CHRONOLOGICAL_REPLIES_INTENT_V1",
            "location": "PERMALINK",
            "root_doc": DOC_ID_COMMENTS_ROOT,
            "root_name": "CommentListComponentsRootQuery",
            "page_doc": DOC_ID_COMMENTS_PAGINATION,
            "page_name": "CommentsListComponentsPaginationQuery",
            "use_root": True,
        },
    ]

    for strategy in strategies:
        cursor = None
        page = 0

        while True:
            if max_pages is not None and page >= max_pages:
                break
            if page >= 220:
                break

            try:
                common_vars = {
                    "commentsIntentToken": strategy["intent"],
                    "feedLocation": strategy["location"],
                    "focusCommentID": None,
                    "id": feedback_id,
                    "scale": 2,
                    "useDefaultActor": False,
                }

                if page == 0 and strategy["use_root"]:
                    variables = {
                        **common_vars,
                        "feedbackSource": 65,
                    }
                    blocks = _request_comments_blocks(
                        strategy["root_doc"],
                        strategy["root_name"],
                        variables,
                        cookies,
                    )
                else:
                    variables = {
                        **common_vars,
                        "commentsAfterCount": -1,
                        "commentsAfterCursor": cursor,
                        "commentsBeforeCount": None,
                        "commentsBeforeCursor": None,
                    }
                    blocks = _request_comments_blocks(
                        strategy["page_doc"],
                        strategy["page_name"],
                        variables,
                        cookies,
                    )

                if not blocks:
                    break

                page_has_edges = False
                has_next = False
                next_cursor = None

                for block in blocks:
                    root = block.get("data") if isinstance(block, dict) and isinstance(block.get("data"), dict) else block
                    if not isinstance(root, dict):
                        continue

                    extra_likes, extra_shares = _extract_likes_shares_from_payload(root)
                    metrics_likes = max(metrics_likes, extra_likes)
                    metrics_shares = max(metrics_shares, extra_shares)

                    comments_block = _extract_comments_block(root)
                    if not isinstance(comments_block, dict):
                        continue

                    comment_total = max(comment_total, _parse_count(comments_block.get("total_count")))

                    edges = comments_block.get("edges") or []
                    if edges:
                        page_has_edges = True

                    for e in edges:
                        n = (e or {}).get("node") or {}
                        comment_id = str(n.get("legacy_fbid") or n.get("id") or "")
                        text = (n.get("body") or {}).get("text", "")
                        clean_text = re.sub(r"\s+", " ", (text or "")).strip()
                        if not clean_text:
                            clean_text = f"[non-text-comment:{comment_id or 'unknown'}]"

                        can_store_comment = False
                        if comment_id:
                            if comment_id not in seen_comment_ids:
                                seen_comment_ids.add(comment_id)
                                can_store_comment = True
                        elif clean_text and clean_text not in seen_text_fallback:
                            seen_text_fallback.add(clean_text)
                            can_store_comment = True

                        if can_store_comment:
                            results.append(clean_text)

                        fb = n.get("feedback")
                        if not isinstance(fb, dict):
                            fb = {}

                        replies_connection = fb.get("replies_connection")
                        if not isinstance(replies_connection, dict):
                            replies_connection = {}

                        for re_edge in replies_connection.get("edges") or []:
                            reply_node = (re_edge or {}).get("node") or {}
                            reply_id = str(reply_node.get("legacy_fbid") or reply_node.get("id") or "")
                            reply_text = re.sub(r"\s+", " ", ((reply_node.get("body") or {}).get("text") or "")).strip()
                            if not reply_text:
                                reply_text = f"[non-text-reply:{reply_id or 'unknown'}]"
                            if reply_id and reply_id in seen_reply_ids:
                                continue
                            if reply_id:
                                seen_reply_ids.add(reply_id)
                            elif reply_text in seen_text_fallback:
                                continue
                            else:
                                seen_text_fallback.add(reply_text)
                            results.append(reply_text)

                        page_info_reply = replies_connection.get("page_info") or {}
                        reply_total = _parse_count(_get_nested_dict(fb, ["replies_fields", "total_count"]))
                        visible_reply_count = len(replies_connection.get("edges") or [])
                        reply_token = _get_nested_dict(fb, ["expansion_info", "expansion_token"]) or page_info_reply.get("end_cursor")
                        need_more_reply = bool(reply_token) and (
                            bool(page_info_reply.get("has_next_page"))
                            or reply_total > visible_reply_count
                        )

                        if need_more_reply:
                            reply_items = fetch_replies_for_comment(fb.get("id"), reply_token, cookies)
                            for reply in reply_items:
                                reply_id = str(reply.get("id") or "")
                                reply_text = re.sub(r"\s+", " ", (reply.get("text") or "")).strip()
                                if not reply_text:
                                    reply_text = f"[non-text-reply:{reply_id or 'unknown'}]"
                                if reply_id and reply_id in seen_reply_ids:
                                    continue
                                if reply_id:
                                    seen_reply_ids.add(reply_id)
                                elif reply_text in seen_text_fallback:
                                    continue
                                else:
                                    seen_text_fallback.add(reply_text)
                                results.append(reply_text)

                    page_info = comments_block.get("page_info") or {}
                    has_next = has_next or bool(page_info.get("has_next_page"))
                    if page_info.get("end_cursor"):
                        next_cursor = page_info.get("end_cursor")

                if not page_has_edges:
                    break
                if not has_next or not next_cursor or next_cursor == cursor:
                    break

                cursor = next_cursor
                page += 1
                time.sleep(0.25)

                if comment_total > 0 and len(seen_comment_ids) >= comment_total:
                    break
            except Exception as e:
                print(f"    ⚠️ Lỗi fetch comments: {e}")
                break

        if comment_total > 0 and len(seen_comment_ids) >= comment_total:
            break

    # Bù các comment không thể truy xuất body (privacy/filter) để giữ đủ tổng lượng.
    if comment_total > 0 and len(results) < comment_total:
        missing = comment_total - len(results)
        for idx in range(missing):
            results.append(f"[missing-comment:{idx + 1}]")

    return results, {
        "likes": _sanitize_metric(metrics_likes),
        "shares": _sanitize_metric(metrics_shares),
        "comment_total": _sanitize_metric(comment_total),
    }

# ========= IMAGE =========
def download_image(url, post_id, image_index=1, post_dir=None):
    if not url or not post_id:
        return None
    try:
        if not post_dir:
            post_dir = os.path.join(OUTPUT_DIR, str(post_id))
        os.makedirs(post_dir, exist_ok=True)
        
        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        elif ".jpeg" in url.lower():
            ext = ".jpeg"
        
        filename = f"{post_id}{ext}" if image_index == 1 else f"{post_id}_{image_index}{ext}"
        filepath = os.path.join(post_dir, filename)
        
        response = requests.get(url, timeout=30, proxies=PROXIES)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return filename
    except:
        return None

_image_counters = {}

def extract_media(node, post_id, post_dir=None):
    global _image_counters
    
    if post_id not in _image_counters:
        _image_counters[post_id] = 0
    
    media = []
    attachments = node.get("attachments") or []

    first_image_url = None
    for att in attachments:
        styles = att.get("styles") or {}
        attachment = styles.get("attachment") or {}

        single_media = attachment.get("media")
        if single_media and isinstance(single_media, dict) and "photo_image" in single_media:
            first_image_url = (single_media.get("photo_image") or {}).get("uri")
            if first_image_url:
                break

        for m in attachment.get("all_subattachments", {}).get("nodes", []):
            media_node = m.get("media") or {}
            if "image" in media_node:
                first_image_url = (media_node.get("image") or {}).get("uri")
                if first_image_url:
                    break

        if first_image_url:
            break

    # Chỉ lấy ảnh đầu tiên đại diện cho post (album/carousel)
    if first_image_url:
        _image_counters[post_id] += 1
        saved = download_image(first_image_url, post_id, _image_counters[post_id], post_dir)
        if saved:
            media.append({"type": "photo", "url": first_image_url})

    return media

# ========= EXTRACT =========
def extract_comment_count(node):
    try:
        comment_count = node.get("feedback", {}).get("comment_rendering_instance", {}).get("comments", {}).get("total_count")
        if comment_count is not None:
            return comment_count
        return 0
    except:
        return 0

def is_reel_or_video_post(node):
    story_type = node.get("__typename", "")
    if "reel" in story_type.lower():
        return True
    
    attachments = node.get("attachments") or []
    for att in attachments:
        styles = att.get("styles") or {}
        attachment = styles.get("attachment") or {}
        single_media = attachment.get("media")
        if single_media and single_media.get("__typename") == "Video":
            return True
    
    return False

def extract_page_name(node):
    try:
        actors = node.get('comet_sections', {}).get('content', {}).get('story', {}).get('actors', [])
        if actors and len(actors) > 0:
            return actors[0].get('name')
        return None
    except:
        return None

def post_already_exists(post_id, base_folder=OUTPUT_DIR, fanpage_name=""):
    if not post_id or not fanpage_name:
        return False
    fanpage_name_clean = sanitize_fanpage_name(fanpage_name)
    post_file = os.path.join(base_folder, "facebook", fanpage_name_clean, str(post_id), f"{post_id}.json")
    return os.path.exists(post_file)

# ========= EXTRACT USER ID FROM URL =========
def extract_user_id_from_url(url, cookies=None):
    """Extract Facebook User ID from a profile URL"""
    if not url:
        return None
    
    # First, try direct URL patterns
    url_patterns = [
        r'profile\.php\?id=(\d+)',
        r'/profile/(\d+)',
        r'id=(\d+)',
        r'facebook\.com/(\d+)(?:/|$)',
    ]
    
    for pattern in url_patterns:
        match = re.search(pattern, url)
        if match:
            user_id = match.group(1)
            if user_id.isdigit():
                print(f"  ✅ Found User ID in URL: {user_id}")
                return user_id
    
    # If no ID in URL, fetch page and search in HTML
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    try:
        print(f"  🔍 Fetching page to extract ID...")
        response = requests.get(url, headers=headers, proxies=PROXIES, timeout=20)
        response.raise_for_status()
        html = response.text
        
        # Try multiple patterns to find user ID in HTML
        patterns = [
            r'fb://profile/(\d+)',           # BEST signal
            r'"profile_owner":"(\d+)"',
            r'"userID":"(\d+)"',
            r'owner_id["\']?\s*:\s*["\']?(\d+)',
            r'"page_id":"(\d+)"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                user_id = match.group(1)
                if user_id.isdigit():
                    print(f"  ✅ Found User ID: {user_id}")
                    return user_id
        
        print("  ❌ User ID not found (profile may be private or login wall)")
        return None
    
    except Exception as e:
        print(f"  ❌ Error fetching URL: {e}")
        return None

# ========= FETCH POSTS =========
def fetch_posts(user_id, page_name, limit=3, page_url=None):
    """Chỉ lấy posts có ảnh"""
    all_posts = []
    cursor = None
    page_num = 1
    
    print(f"  🔍 Lấy posts từ {page_name} (max {limit} posts có ảnh)...")
    
    while len(all_posts) < limit:
        variables = {
            "count": 5,
            "cursor": cursor,
            "id": user_id,
            "feedLocation": "TIMELINE",
            "renderLocation": "timeline",
            "scale": 2,
            "useDefaultActor": False
        }
        
        payload = {
            "av": COOKIES.get("c_user", "0"),
            "__user": COOKIES.get("c_user", "0"),
            "__a": "1",
            "fb_dtsg": FB_DTSG if FB_DTSG else "",
            "doc_id": DOC_ID_POSTS,
            "variables": json.dumps(variables),
        }
        
        max_empty_retries = 3
        empty_retry_count = 0
        cleaned_data = []
        
        while empty_retry_count < max_empty_retries:
            try:
                r = retry_request(GRAPHQL_URL, BASE_HEADERS, payload, PROXIES, COOKIES)
                if not r:
                    empty_retry_count += 1
                    if empty_retry_count < max_empty_retries:
                        time.sleep(2)
                    continue
                
                cleaned_data = parse_fb_response(r.text)
                
                if cleaned_data and len(cleaned_data) > 0:
                    break
                else:
                    empty_retry_count += 1
                    if empty_retry_count < max_empty_retries:
                        time.sleep(2)
            except Exception as e:
                print(f"    ❌ Lỗi: {e}")
                empty_retry_count += 1
                if empty_retry_count < max_empty_retries:
                    time.sleep(2)
        
        if not cleaned_data or len(cleaned_data) == 0:
            print(f"    ⚠️ Không có dữ liệu")
            break
        
        story_nodes = []
        timeline_block = None
        
        for block in cleaned_data:
            if not isinstance(block, dict):
                continue
            
            node = block.get("node") or {}
            if not isinstance(node, dict):
                node = {}
            
            if "timeline_list_feed_units" in node:
                timeline_block = block
                edges = node["timeline_list_feed_units"].get("edges", [])
                for edge in edges:
                    edge_node = edge.get("node")
                    if edge_node and edge_node.get("__typename") == "Story":
                        story_nodes.append(edge_node)
        
        print(f"    📄 Tìm được {len(story_nodes)} posts trên trang {page_num}")
        
        for node in story_nodes:
            if len(all_posts) >= limit:
                break
            
            if is_reel_or_video_post(node):
                continue
            
            post_id = node.get("post_id")
            if not post_id:
                continue
            
            temp_page_name = extract_page_name(node) or page_name
            
            if post_already_exists(post_id, OUTPUT_DIR, temp_page_name):
                continue

            # Trích xuất comments trước và bỏ qua post không có comments
            feedback = node.get("feedback") or {}
            feedback_id = feedback.get("id")
            comments_text, feedback_metrics = fetch_comments_from_feedback(feedback_id, COOKIES)
            if not comments_text and int((feedback_metrics or {}).get("comment_total", 0) or 0) == 0:
                print(f"      ↷ Skip post {post_id}: không có comment")
                continue

            reaction_metrics = fetch_reactions_breakdown(feedback_id, COOKIES)

            # CHỈ LẤY POSTS CÓ ẢNH
            fanpage_name_clean = sanitize_fanpage_name(temp_page_name)
            post_dir = os.path.join(OUTPUT_DIR, "facebook", fanpage_name_clean, str(post_id))
            media = extract_media(node, post_id, post_dir)
            if not media:
                continue
            
            message = (
                node.get("comet_sections", {})
                .get("content", {})
                .get("story", {})
                .get("message", {})
                .get("text")
            )

            page_slug = _extract_page_slug_from_node(node, page_url)
            permalink = f"https://www.facebook.com/{page_slug}/posts/{post_id}" if page_slug else None
            
            likes, shares = extract_likes_shares(node)
            likes = max(likes, int((feedback_metrics or {}).get("likes", 0) or 0))
            shares = max(shares, int((feedback_metrics or {}).get("shares", 0) or 0))
            likes = max(likes, int((reaction_metrics or {}).get("likes", 0) or 0))

            # Fallback plugin/permalink để tránh thiếu like/share trên timeline GraphQL.
            fallback_likes, fallback_comment_total, fallback_shares = fetch_post_engagement_fallback(
                post_id,
                owner_id=user_id,
                cookies=COOKIES,
                page_slug=page_slug,
                permalink=permalink,
            )
            likes = max(likes, fallback_likes)
            shares = max(shares, fallback_shares)

            comment_total = max(
                len(comments_text),
                int((feedback_metrics or {}).get("comment_total", 0) or 0),
                int(fallback_comment_total or 0),
            )
            
            post = {
                "post_id": post_id,
                "text": message,
                "media_count": len(media),
                "media": media,
                "page_name": temp_page_name,
                "likes": likes,
                "shares": shares,
                "react_like": int((reaction_metrics or {}).get("react_like", 0) or 0),
                "react_love": int((reaction_metrics or {}).get("react_love", 0) or 0),
                "react_care": int((reaction_metrics or {}).get("react_care", 0) or 0),
                "react_wow": int((reaction_metrics or {}).get("react_wow", 0) or 0),
                "react_angry": int((reaction_metrics or {}).get("react_angry", 0) or 0),
                "comment_count": comment_total,
                "comments": comments_text,
            }
            
            os.makedirs(post_dir, exist_ok=True)
            
            post_file = os.path.join(post_dir, f"{post_id}.json")
            with open(post_file, "w", encoding="utf-8") as f:
                json.dump(post, f, ensure_ascii=False, indent=2)
            
            print(
                f"      ✓ Post {post_id} ({len(media)} ảnh, "
                f"likes={likes}, shares={shares}, comments={len(comments_text)}/{comment_total})"
            )
            
            all_posts.append(post)
            time.sleep(1)
        
        # Get next cursor
        page_info = {}
        if timeline_block:
            page_info = timeline_block["node"]["timeline_list_feed_units"].get("page_info", {})
        
        if not page_info:
            for block in cleaned_data:
                if isinstance(block, dict) and "page_info" in block:
                    page_info = block["page_info"]
                    break
        
        cursor = page_info.get("end_cursor")
        
        if not cursor:
            break
        
        time.sleep(1)
        page_num += 1
    
    return all_posts

# ========= EXTRACT USER ID =========
# ========= MAIN =========
def main():
    print("\n" + "="*60)
    print("   📘 FACEBOOK AUTO SCRAPER")
    print("="*60 + "\n")
    
    if not COOKIES or 'c_user' not in COOKIES:
        print("❌ LỖI: Chưa có cookies trong .env!")
        print("\n📋 HƯỚNG DẪN CẬP NHẬT CREDENTIALS:")
        print("════════════════════════════════════════════")
        print("1. ⏰ Lưu ý: Cookies hết hạn sau vài ngày - cần lấy LẠI")
        print("2. 🔓 Mở Facebook trong Chrome")
        print("3. 🔧 Nhấn F12 → Application tab")
        print("4. 🗂️ Left panel → Cookies → facebook.com")
        print("5. 📋 Copy: c_user, xs, datr → cập nhật .env")
        print("6. 🌐 Network tab → Refresh →Find 'graphql' request")
        print("7. 📦 Payload tab → Copy fb_dtsg value")
        print("8. 💾 Paste vào FB_DTSG trong .env")
        print("════════════════════════════════════════════")
        return
    
    if not FB_DTSG:
        print("⚠️ CẢNH BÁO: Chưa có FB_DTSG trong .env!")
        print("Scraper có thể không hoạt động.")
    
    print(f"✅ User: {COOKIES.get('c_user')}")
    print(f"✅ DOC_ID: {DOC_ID_POSTS[:20]}...")
    print(f"✅ Credentials ready!\n")
    
    total_posts = 0
    
    for fanpage in FANPAGES:
        fanpage_url = fanpage.get("url")
        name = fanpage.get("name", "Unknown")
        
        print(f"\n📍 Xử lý: {name}")
        print(f"   URL: {fanpage_url}")
        
        # Extract user ID from URL
        try:
            user_id = extract_user_id_from_url(fanpage_url)
        except Exception as ex:
            print(f"   ❌ Exception in extract: {ex}")
            user_id = None
        
        if not user_id:
            print(f"   ❌ Không thể trích xuất ID từ URL, bỏ qua\n")
            continue
        
        print(f"   ID: {user_id}")
        
        try:
            posts = fetch_posts(user_id, name, limit=POST_LIMIT_PER_FANPAGE, page_url=fanpage_url)
            print(f"   ✅ Lấy được {len(posts)} posts\n")
            total_posts += len(posts)
        except Exception as e:
            print(f"   ❌ Lỗi: {e}\n")
            import traceback
            traceback.print_exc()
        
        time.sleep(2)  # Delay giữa các fanpage
    
    print("\n" + "="*60)
    print(f"✅ HOÀN THÀNH! Tổng {total_posts} posts")
    print(f"📂 Dữ liệu lưu tại: {OUTPUT_DIR}/")
    print("="*60 + "\n")
    
    # ========= TRÍCH XUẤT DỮ LIỆU & XUẤT CSV =========
    if run_data_extraction and total_posts > 0:
        print("\n" + "="*60)
        print("   🔄 NÂNG CẤP: Trích xuất dữ liệu & Xuất CSV")
        print("="*60)
        
        try:
            csv_path, record_count = run_data_extraction(
                output_dir=OUTPUT_DIR,
                rename_images=True
            )
            
            if csv_path:
                print(f"\n🎉 CSV EXPORT THÀNH CÔNG!")
                print(f"   📊 File: {csv_path}")
                print(f"   📈 Records: {record_count} dòng")
        
        except Exception as e:
            print(f"\n⚠️ Lỗi trong quá trình export: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Dừng chương trình theo yêu cầu người dùng.")
