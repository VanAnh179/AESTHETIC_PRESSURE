"""
Instagram Scraper - Auto version
Tự động cà dữ liệu từ danh sách fanpage Instagram sử dụng instagrapi
"""
import sys
import io
import os
import json
import csv
import re
import time
import random
import shutil
import requests
from datetime import datetime, timezone

# Fix encoding for Vietnamese characters and emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ========= PATH SETUP =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ Error: python-dotenv not installed. Run: pip install python-dotenv")
    exit(1)

# Load .env
ENV_PATHS = [
    os.path.join(SCRIPT_DIR, '.env'),
    os.path.join(PROJECT_ROOT, '.env'),
]
for env_path in ENV_PATHS:
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        break

try:
    from instagrapi import Client
    from instagrapi.exceptions import (
        LoginRequired, ChallengeRequired, FeedbackRequired,
        PleaseWaitFewMinutes, ClientError,
    )
except ImportError:
    print("❌ Error: instagrapi not installed. Run: pip install instagrapi")
    exit(1)

# ========= CONFIG =========
IG_USERNAME = os.getenv('IG_USERNAME', '').strip().strip('"')
IG_PASSWORD = os.getenv('IG_PASSWORD', '').strip().strip('"')
IG_SESSION_ID = os.getenv('IG_SESSION_ID', '').strip().strip('"')
IG_PROXY = os.getenv('IG_PROXY', '').strip().strip('"')

FANPAGES_FILE = os.path.join(SCRIPT_DIR, 'fanpages.json')
SESSION_FILE = os.path.join(SCRIPT_DIR, 'session.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'instagram')
CSV_FILENAME = 'raw_ig_data.csv'

# Scraping rules
MAX_POSTS_PER_PAGE = 5          # Mỗi page chỉ cà tối đa 5 post
MIN_COMMENT_COUNT = 20          # Post phải trên 20 comment mới được cà
MAX_COMMENTS_PER_POST = 60      # Chỉ cà top 60 comment
MAX_POSTS_TO_SCAN = 30          # Chỉ duyệt tối đa 30 post mỗi page
CUTOFF_DATE = datetime(2026, 4, 18, tzinfo=timezone.utc)  # Chỉ cà post trước ngày này
SAVE_EVERY_N_PAGES = 10         # Mỗi 10 page sẽ tự động lưu CSV + đổi tên ảnh

# CSV columns
CSV_FIELDNAMES = [
    'img_id', 'source', 'fanpage', 'total_react',
    'comment_count', 'hashtag', 'raw_comment'
]

IMG_ID_PATTERN = re.compile(r'^ins_(\d+)$', re.IGNORECASE)
PROCESSED_IMAGE_PATTERN = re.compile(r'^ins_\d+\.\w+$', re.IGNORECASE)


# ========= HELPER FUNCTIONS =========
def sanitize_name(name):
    """Sanitize fanpage name for folder usage."""
    if not name:
        return "Unknown"
    sanitized = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    return sanitized or "Unknown"


def extract_username_from_url(url):
    """Extract Instagram username from URL."""
    if not url:
        return None
    # Handle various URL formats
    m = re.search(r'instagram\.com/([^/?#]+)/?', url)
    if m:
        username = m.group(1).strip()
        # Filter out non-profile paths
        if username in ('p', 'reel', 'explore', 'stories', 'accounts', 'direct'):
            return None
        return username
    return None


def extract_hashtags(caption_text):
    """Extract hashtags from caption text, joined by |."""
    if not caption_text:
        return ""
    tags = re.findall(r'#\w+', caption_text)
    return "|".join(tags) if tags else ""


def is_text_comment(text):
    """
    Check if comment contains real text (not just emojis/GIFs/hashtags-only).
    Returns False for: empty, emoji-only, GIF placeholders, hashtag-only.
    """
    s = str(text or "").strip()
    if not s:
        return False
    # Remove all emoji and symbol unicode characters
    stripped = re.sub(
        r'[\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF'
        r'\U00002702-\U000027B0'
        r'\U000024C2-\U0001F251'
        r'\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F'
        r'\U0001FA70-\U0001FAFF'
        r'\U00002600-\U000026FF'
        r'\U0000FE00-\U0000FE0F'
        r'\U0000200D'
        r'\U00000020'
        r']+', '', s
    )
    # After removing emojis, must still have at least one letter or digit
    if not any(ch.isalnum() for ch in stripped):
        return False
    # Remove all hashtags and whitespace; if nothing remains → hashtag-only
    without_hashtags = re.sub(r'#\w+', '', s).strip()
    # Also strip emojis from the remainder
    without_hashtags = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251'
        r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
        r'\U00002600-\U000026FF\U0000FE00-\U0000FE0F\U0000200D\s]+', '', without_hashtags
    )
    if not any(ch.isalnum() for ch in without_hashtags):
        return False
    return True


def format_comments(comments):
    """Format comment objects into a single string separated by |. Newlines removed."""
    texts = []
    for c in comments:
        text = ""
        if hasattr(c, 'text'):
            text = str(c.text or "").strip()
        elif isinstance(c, dict):
            text = str(c.get('text', '') or "").strip()
        elif isinstance(c, str):
            text = c.strip()

        # Replace newlines with space to keep CSV on one line
        text = re.sub(r'[\r\n]+', ' ', text).strip()

        if text and is_text_comment(text):
            texts.append(text)
    return "|".join(texts)


def load_fanpages():
    """Load fanpages from JSON file."""
    if not os.path.exists(FANPAGES_FILE):
        print(f"❌ Không tìm thấy file: {FANPAGES_FILE}")
        return []
    try:
        with open(FANPAGES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Lỗi đọc fanpages.json: {e}")
        return []

    if isinstance(data, dict):
        data = data.get("fanpages", [])
    if not isinstance(data, list):
        print("❌ fanpages.json phải là một mảng.")
        return []

    fanpages = []
    for item in data:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        name = str(item.get("name") or "Unknown").strip() or "Unknown"
        fanpages.append({"url": url, "name": name})
    return fanpages


# ========= IMAGE COUNTER BOOTSTRAP =========
def get_existing_max_img_counter():
    """Scan existing CSV and files to find the highest ins_XXX counter."""
    max_idx = 0

    # Check CSV
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    m = IMG_ID_PATTERN.match(str(row.get('img_id', '')))
                    if m:
                        idx = int(m.group(1))
                        if idx > max_idx:
                            max_idx = idx
        except Exception as e:
            print(f"⚠️ Không đọc được CSV hiện có: {e}")

    # Check existing renamed files
    if os.path.exists(OUTPUT_DIR):
        for root, _, files in os.walk(OUTPUT_DIR):
            for filename in files:
                name_no_ext = os.path.splitext(filename)[0]
                m = IMG_ID_PATTERN.match(name_no_ext)
                if m:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx

    return max_idx


# ========= LOGIN =========
def _configure_client(cl):
    """Apply common client settings for anti-detection."""
    cl.delay_range = [5, 12]
    cl.set_country("VN")
    cl.set_country_code(84)
    cl.set_locale("vi_VN")
    cl.set_timezone_offset(7 * 3600)  # UTC+7
    if IG_PROXY:
        cl.set_proxy(IG_PROXY)


def create_client():
    """Create and login instagrapi Client with session persistence and anti-detection."""

    # === Strategy 1: Load saved session file ===
    if os.path.exists(SESSION_FILE):
        try:
            cl = Client()
            _configure_client(cl)
            cl.load_settings(SESSION_FILE)
            cl.login(IG_USERNAME, IG_PASSWORD)
            try:
                cl.account_info()
            except LoginRequired:
                old_session = cl.get_settings()
                cl.set_settings({})
                cl.set_uuids(old_session["uuids"])
                cl.login(IG_USERNAME, IG_PASSWORD)
            cl.dump_settings(SESSION_FILE)
            print("✅ Đăng nhập thành công (session cũ)")
            return cl
        except Exception as e:
            print(f"⚠️ Session cũ không dùng được: {e}")
            try:
                os.remove(SESSION_FILE)
            except OSError:
                pass

    # === Strategy 2: Login by session ID (from browser) ===
    if IG_SESSION_ID and len(IG_SESSION_ID) > 30:
        try:
            print("   🍪 Đăng nhập bằng Session ID...")
            cl = Client()
            _configure_client(cl)
            cl.login_by_sessionid(IG_SESSION_ID)
            cl.dump_settings(SESSION_FILE)
            print("✅ Đăng nhập thành công (Session ID)")
            return cl
        except Exception as e:
            print(f"   ⚠️ Session ID không hợp lệ: {e}")

    # === Strategy 3: Username/password login with retries ===
    if IG_USERNAME and IG_PASSWORD:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                print(f"   🔑 Đăng nhập lần {attempt}/{max_retries}...")
                cl = Client()
                _configure_client(cl)
                cl.login(IG_USERNAME, IG_PASSWORD)
                cl.dump_settings(SESSION_FILE)
                print("✅ Đăng nhập thành công (mới)")
                return cl
            except Exception as e:
                print(f"   ⚠️ Lần {attempt} thất bại: {e}")
                if attempt < max_retries:
                    wait = attempt * 10
                    print(f"   ⏳ Chờ {wait}s trước khi thử lại...")
                    time.sleep(wait)

    print("❌ Không thể đăng nhập.")
    print("   💡 Gợi ý:")
    print("   1. Mở Chrome -> đăng nhập Instagram")
    print("   2. F12 -> Application -> Cookies -> instagram.com")
    print("   3. Copy giá trị 'sessionid'")
    print("   4. Dán vào IG_SESSION_ID trong file .env")
    print("   5. Chạy lại script")
    return None


# ========= SCRAPE ONE PAGE =========
def scrape_page(cl, username, page_name):
    """Scrape posts from a single Instagram page. Returns list of post data dicts."""
    print(f"\n📍 Đang xử lý page: {page_name} (@{username})")

    page_dir = os.path.join(OUTPUT_DIR, sanitize_name(page_name))
    os.makedirs(page_dir, exist_ok=True)

    # Get user ID
    try:
        user_id = cl.user_id_from_username(username)
        print(f"   User ID: {user_id}")
    except Exception as e:
        print(f"   ❌ Không tìm được user: {e}")
        return []

    time.sleep(random.uniform(5, 10))
    print(f"   ⏳ Đợi xong, bắt đầu lấy media...")

    # Fetch medias (scan up to MAX_POSTS_TO_SCAN)
    try:
        medias = cl.user_medias(user_id, amount=MAX_POSTS_TO_SCAN)
        print(f"   📦 Lấy được {len(medias)} posts để duyệt")
    except Exception as e:
        print(f"   ❌ Lỗi lấy media: {e}")
        return []

    collected_posts = []
    scanned = 0

    for media in medias:
        if scanned >= MAX_POSTS_TO_SCAN:
            print(f"   ⚠️ Đã duyệt tối đa {MAX_POSTS_TO_SCAN} posts, chuyển page")
            break

        if len(collected_posts) >= MAX_POSTS_PER_PAGE:
            print(f"   ✅ Đã đủ {MAX_POSTS_PER_PAGE} posts, chuyển page")
            break

        scanned += 1
        media_pk = str(media.pk)

        # Check date cutoff
        post_date = media.taken_at
        if post_date and post_date.tzinfo is None:
            post_date = post_date.replace(tzinfo=timezone.utc)

        if post_date and post_date >= CUTOFF_DATE:
            print(f"      ⏭️ Post {media_pk}: ngày {post_date.strftime('%Y-%m-%d')} >= cutoff, bỏ qua")
            continue

        # Check comment count
        comment_count = media.comment_count or 0
        if comment_count < MIN_COMMENT_COUNT:
            print(f"      ⏭️ Post {media_pk}: {comment_count} comments (< {MIN_COMMENT_COUNT}), bỏ qua")
            continue

        print(f"      📝 Post {media_pk}: {comment_count} comments, likes={media.like_count}")

        # Create post directory
        post_dir = os.path.join(page_dir, media_pk)
        os.makedirs(post_dir, exist_ok=True)

        # Fetch comments (delay dài để tránh bị phát hiện)
        wait_comment = random.uniform(8, 15)
        print(f"         ⏳ Chờ {wait_comment:.1f}s trước khi lấy comments...")
        time.sleep(wait_comment)
        try:
            comments = cl.media_comments(media.pk, amount=MAX_COMMENTS_PER_POST)
            print(f"         💬 Lấy được {len(comments)} comments")
        except Exception as e:
            print(f"         ⚠️ Lỗi lấy comments: {e}")
            comments = []

        # Download image
        image_path = None
        try:
            time.sleep(random.uniform(5, 8))
            if media.media_type == 1:
                # Photo: download directly
                image_path = cl.photo_download(media.pk, folder=post_dir)
            elif media.media_type == 8:
                # Album: chỉ lấy ảnh đầu tiên
                if media.resources and len(media.resources) > 0:
                    first_res = media.resources[0]
                    # Download thumbnail of first resource
                    thumb_url = str(first_res.thumbnail_url) if first_res.thumbnail_url else None
                    if thumb_url:
                        thumb_path = os.path.join(post_dir, f"{media_pk}.jpg")
                        r = requests.get(thumb_url, timeout=30)
                        if r.status_code == 200:
                            with open(thumb_path, 'wb') as f:
                                f.write(r.content)
                            image_path = thumb_path
                    if not image_path:
                        # Fallback: download first via album_download
                        paths = cl.album_download(media.pk, folder=post_dir)
                        if paths:
                            image_path = paths[0]
                            for extra in paths[1:]:
                                try:
                                    os.remove(str(extra))
                                except OSError:
                                    pass
            elif media.media_type == 2:
                # Video: download thumbnail
                if media.thumbnail_url:
                    thumb_path = os.path.join(post_dir, f"{media_pk}.jpg")
                    try:
                        r = requests.get(str(media.thumbnail_url), timeout=30)
                        if r.status_code == 200:
                            with open(thumb_path, 'wb') as f:
                                f.write(r.content)
                            image_path = thumb_path
                    except Exception:
                        pass

            if image_path:
                print(f"         🖼️ Ảnh: {os.path.basename(str(image_path))}")
            else:
                print(f"         ⚠️ Không tải được ảnh")
                continue  # Skip post nếu không có ảnh
        except Exception as e:
            print(f"         ⚠️ Lỗi tải ảnh: {e}")
            continue

        # Extract data
        caption = media.caption_text or ""
        hashtags = extract_hashtags(caption)
        comments_text = format_comments(comments)
        total_react = media.like_count or 0

        # Build raw data dict
        raw_data = {
            "media_pk": media_pk,
            "media_id": str(media.id),
            "code": media.code,
            "taken_at": str(post_date),
            "media_type": media.media_type,
            "caption": caption,
            "like_count": media.like_count or 0,
            "comment_count": comment_count,
            "play_count": media.play_count or 0,
            "page_name": page_name,
            "username": username,
            "hashtags": hashtags,
            "comments": [
                str(c.text or "") for c in comments
            ] if comments else []
        }

        # Save raw JSON
        json_path = os.path.join(post_dir, f"{media_pk}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2, default=str)

        collected_posts.append({
            "media_pk": media_pk,
            "page_name": page_name,
            "post_dir": post_dir,
            "total_react": total_react,

            "comment_count": comment_count,
            "hashtags": hashtags,
            "comments_text": comments_text,
            "image_path": str(image_path) if image_path else None,
        })

        print(f"         ✅ Lưu thành công!")
        wait_next = random.uniform(10, 20)
        print(f"         ⏳ Nghỉ {wait_next:.1f}s trước post tiếp theo...")
        time.sleep(wait_next)

    print(f"   📊 Thu thập được {len(collected_posts)}/{scanned} posts từ @{username}")
    return collected_posts


# ========= CSV & IMAGE RENAME =========
def save_to_csv_and_rename(all_posts_data, img_counter_start):
    """
    Append posts to CSV and rename images.
    Returns the new img_counter value after processing.
    """
    if not all_posts_data:
        return img_counter_start

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    counter = img_counter_start
    new_records = []

    for post in all_posts_data:
        image_path = post.get("image_path")
        if not image_path or not os.path.exists(image_path):
            continue

        # Skip already processed images
        basename = os.path.basename(image_path)
        if PROCESSED_IMAGE_PATTERN.match(basename):
            continue

        counter += 1
        img_id = f"ins_{counter:03d}"

        # Rename image to ins_XXX.png
        post_dir = os.path.dirname(image_path)
        _, ext = os.path.splitext(image_path)

        # Convert to PNG if not already
        new_filename = f"{img_id}.png"
        new_path = os.path.join(post_dir, new_filename)

        if ext.lower() != '.png':
            # Copy and rename (simple rename, keep original format with .png extension)
            try:
                shutil.copy2(image_path, new_path)
                os.remove(image_path)
            except Exception as e:
                print(f"   ⚠️ Lỗi đổi tên ảnh: {e}")
                continue
        else:
            try:
                os.rename(image_path, new_path)
            except Exception as e:
                print(f"   ⚠️ Lỗi đổi tên ảnh: {e}")
                continue

        print(f"   🖼️ {basename} → {new_filename}")

        record = {
            'img_id': img_id,
            'source': 'Instagram',
            'fanpage': post.get('page_name', 'Unknown'),
            'total_react': post.get('total_react', 0),

            'comment_count': post.get('comment_count', 0),
            'hashtag': post.get('hashtags', ''),
            'raw_comment': post.get('comments_text', ''),
        }
        new_records.append(record)

    # Write CSV
    if new_records:
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                if write_header:
                    writer.writeheader()
                for record in new_records:
                    writer.writerow(record)
            print(f"\n📝 Đã thêm {len(new_records)} dòng vào CSV")
        except Exception as e:
            print(f"❌ Lỗi ghi CSV: {e}")

    return counter


# ========= MAIN =========
def main():
    print("\n" + "=" * 60)
    print("   📸 INSTAGRAM AUTO SCRAPER")
    print("=" * 60 + "\n")

    # Validate credentials
    has_credentials = bool(IG_USERNAME and IG_PASSWORD)
    has_session = bool(IG_SESSION_ID and len(IG_SESSION_ID) > 30)
    if not has_credentials and not has_session:
        print("❌ Chưa có thông tin đăng nhập trong .env!")
        print("   Cần IG_USERNAME + IG_PASSWORD hoặc IG_SESSION_ID")
        return

    print(f"👤 Username: {IG_USERNAME}")

    # Login
    cl = create_client()
    if not cl:
        return

    # Load fanpages
    fanpages = load_fanpages()
    if not fanpages:
        print("❌ Danh sách fanpage rỗng.")
        return

    print(f"📋 Có {len(fanpages)} page cần cà\n")

    # Bootstrap image counter
    img_counter = get_existing_max_img_counter()
    print(f"🔢 Image counter bắt đầu từ: {img_counter}")

    # Scrape loop
    total_posts = 0
    batch_posts = []  # Accumulate posts for batch save
    pages_in_batch = 0

    for i, fanpage in enumerate(fanpages):
        url = fanpage["url"]
        name = fanpage["name"]
        username = extract_username_from_url(url)

        if not username:
            print(f"\n⚠️ Không trích xuất được username từ: {url}")
            continue

        try:
            posts = scrape_page(cl, username, name)
            batch_posts.extend(posts)
            total_posts += len(posts)
            pages_in_batch += 1
        except Exception as e:
            print(f"   ❌ Lỗi scrape {name}: {e}")
            import traceback
            traceback.print_exc()

        # Save every N pages
        if pages_in_batch >= SAVE_EVERY_N_PAGES:
            print(f"\n💾 Tự động lưu sau {pages_in_batch} pages...")
            img_counter = save_to_csv_and_rename(batch_posts, img_counter)
            batch_posts = []
            pages_in_batch = 0

        # Delay between pages
        if i < len(fanpages) - 1:
            delay = random.uniform(30, 60)
            print(f"\n⏳ Nghỉ {delay:.0f}s trước page tiếp theo...")
            time.sleep(delay)

    # Final save for remaining posts
    if batch_posts:
        print(f"\n💾 Lưu batch cuối cùng ({len(batch_posts)} posts)...")
        img_counter = save_to_csv_and_rename(batch_posts, img_counter)

    print("\n" + "=" * 60)
    print(f"✅ HOÀN THÀNH! Tổng: {total_posts} posts")
    print(f"📂 Dữ liệu: {OUTPUT_DIR}")
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    if os.path.exists(csv_path):
        print(f"📊 CSV: {csv_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Dừng chương trình theo yêu cầu người dùng.")
