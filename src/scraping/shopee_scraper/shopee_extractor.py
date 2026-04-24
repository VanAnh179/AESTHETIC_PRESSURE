import os
import re
import time
import hashlib
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException

# ── Cấu hình ──────────────────────────────────────────────────────────────────
MAX_PRODUCTS      = 20   # tối đa SP mỗi banner
MIN_COMMENTS      = 2    # giữ lại để thu raw_comment
MIN_REVIEWS       = 2    # SP có >= 2 review hợp lệ sẽ được tính là đạt
MAX_COMMENTS_EACH = 5    # chỉ lấy 5 comment/SP cho raw_comment
MAX_PAGES_PER_SP  = 3    # tối đa 3 trang review mỗi SP

# ── Selectors & Regex ─────────────────────────────────────────────────────────
KNOWN_REVIEW_SELECTORS = [
    'div.q2b7Oq[data-cmtid]',
    'div.q2b7Oq',
    'div.shopee-product-rating',
    '[class*="product-ratings__list"] > div',
    '[class*="product-rating__main"]',
    '[data-cmtid]',
    '[class*="rating"]',
]
COMMENT_SELECTORS = [
    'div.YNedDV', '[class*="YNedDV"]',
    'div.meQyXP', '[class*="meQyXP"]',
    '[class*="review-content"]', '[class*="comment-content"]',
]

DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?')

SELLER_REPLY_RE = re.compile(
    r'phản\s*hồi\s*của\s*người\s*bán|seller\s*reply|shop\s*reply', re.I
)
LABEL_RE = re.compile(
    r'^(màu sắc|chất liệu|kích cỡ|kích thước|số lượng|size|color|material)\s*:\s*', re.I
)
EXCLUDE_RE = re.compile(
    r'Phân loại hàng|Chất liệu\s*:|Màu sắc\s*:|Kích cỡ\s*:|Số lượng\s*:'
    r'|Phản Hồi Của Người Bán|hữu ích|báo cáo|trả lời',
    re.I
)


# ── Text helpers ───────────────────────────────────────────────────────────────
def normalize_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def clean_comment_text(text: str) -> str:
    t = normalize_space(text)
    if not t:
        return ''
    if SELLER_REPLY_RE.search(t):
        return ''
    t = LABEL_RE.sub('', t)
    t = re.sub(
        r'(màu sắc|chất liệu|kích cỡ|kích thước|số lượng|size|color|material)\s*:\s*',
        ' ', t, flags=re.I
    )
    t = normalize_space(t)
    if len(t) < 2 or len(t) > 1200:
        return ''
    if t.lower() in {'hữu ích?', 'báo cáo', 'trả lời'}:
        return ''
    return t


def post_clean(t: str) -> str:
    t = normalize_space(t)
    if not t:
        return ''
    t = re.sub(r'\b(hữu ích\??|báo cáo|trả lời)\b', ' ', t, flags=re.I)
    t = re.sub(r'\b\d{1,2}:\d{2}\b', ' ', t)
    t = re.sub(r'Phân loại hàng\s*:.*$', ' ', t, flags=re.I)
    t = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}', ' ', t)
    t = normalize_space(t)
    return t if len(t) >= 2 else ''


def normalize_product_url(href: str) -> str:
    href = (href or '').strip()
    if not href:
        return ''
    if href.startswith('//'):
        href = 'https:' + href
    elif href.startswith('/'):
        href = 'https://shopee.vn' + href
    elif not href.startswith('http'):
        return ''
    href = href.split('?')[0]
    return href if ('/product/' in href or re.search(r'-i\.\d+\.\d+', href)) else ''


def parse_review_date(date_str: str):
    if not date_str:
        return None
    from datetime import datetime
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def wait_if_verification_page(driver):
    """Nếu Shopee bật captcha/xác thực thì chờ user xử lý tay."""
    try:
        url = (driver.current_url or '').lower()
    except Exception:
        return
    if any(k in url for k in ('verify', 'captcha', 'anti-bot', 'bot-check')):
        print()
        print('🚧  Shopee đang yêu cầu xác thực / captcha.')
        print('   Hãy xử lý trên Chrome rồi quay lại đây.')
        input('>>> Nhấn Enter để tiếp tục scraping: ')


# ── Star extraction ────────────────────────────────────────────────────────────
def extract_stars_from_card(card) -> int:
    solid = len(card.select('svg.icon-rating-solid, svg[class*="icon-rating-solid"]'))
    if 1 <= solid <= 5:
        return solid
    active = len(card.select(
        '[class*="star--active"], [class*="icon-star-active"], [class*="ShopeeRating--active"]'
    ))
    if 1 <= active <= 5:
        return active
    for node in card.select('[aria-label]'):
        label = node.get('aria-label', '')
        m = re.search(r'([1-5])\s*(sao|star|out|/)', label, re.I)
        if m:
            return int(m.group(1))
    return 0


def get_stars_for_cards(driver, selector: str) -> list:
    js = r"""
var cards = document.querySelectorAll(arguments[0]);
return Array.from(cards).map(function(card) {
    var i, m;
    var labeled = card.querySelectorAll('[aria-label]');
    for (i = 0; i < labeled.length; i++) {
        var lb = (labeled[i].getAttribute('aria-label') || '').toLowerCase();
        m = lb.match(/([1-5])\s*(sao|star|out|\/)/);
        if (m) return parseInt(m[1]);
    }
    var active = card.querySelectorAll(
        'svg[class*="icon-rating-solid"],'
      + '[class*="star--active"],'
      + '[class*="icon-star-active"],'
      + '[class*="ShopeeRating--active"]'
    ).length;
    if (active >= 1 && active <= 5) return active;
    var svgs = card.querySelectorAll('svg'), n = 0;
    for (i = 0; i < svgs.length; i++) {
        var col = window.getComputedStyle(svgs[i]).color || '';
        var rgb = col.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (rgb) {
            var r = parseInt(rgb[1]), g = parseInt(rgb[2]), b = parseInt(rgb[3]);
            if (r > g + 20 && b < 110) n++;
        }
    }
    if (n >= 1 && n <= 5) return n;
    return 0;
});
"""
    try:
        result = driver.execute_script(js, selector) or []
        return [max(0, min(int(x or 0), 5)) for x in result]
    except:
        return []


# ── Comment extraction ─────────────────────────────────────────────────────────
def extract_comment_from_card(card) -> str:
    candidates = []

    # 1) div.YNedDV — div chứa text thô của user
    for el in card.select('div.YNedDV, [class*="YNedDV"]'):
        t = post_clean(clean_comment_text(el.get_text(' ', strip=True)))
        if t:
            candidates.append(t)

    # 2) div.meQyXP — block chứa cả label lẫn comment
    me = card.select_one('div.meQyXP, [class*="meQyXP"]')
    if me:
        me_clone = BeautifulSoup(str(me), 'html.parser')
        for lbl in me_clone.select('.K5v3lN, [class*="K5v3lN"]'):
            lbl.decompose()
        for node in me_clone.select('div, span, p'):
            if node.find(['div', 'span', 'p']):
                continue
            t = post_clean(clean_comment_text(node.get_text(' ', strip=True)))
            if t:
                candidates.append(t)
        block = post_clean(clean_comment_text(me_clone.get_text(' ', strip=True)))
        if block:
            candidates.append(block)

    # dedup
    uniq = []
    for t in candidates:
        if t and t not in uniq:
            uniq.append(t)
    if not uniq:
        return ''
    return max(uniq, key=len)[:500]


# ── Review card validation ─────────────────────────────────────────────────────
def is_review_card(el) -> bool:
    text = normalize_space(el.get_text(' ', strip=True))
    if not text:
        return False
    date_hits = DATE_RE.findall(text)
    if len(date_hits) > 2:
        return False
    if len(text) > 5000:
        return False
    classes    = ' '.join(el.get('class', []))
    classes_l  = classes.lower()
    has_marker = bool(el.get('data-cmtid')) or any(
        token in classes_l for token in ('q2b7oq', 'product-rating', 'review', 'rating')
    )
    has_stars  = bool(
        el.select('svg.icon-rating-solid, svg[class*="icon-rating-solid"]')
        or el.select('svg.icon-rating, svg[class*="icon-rating"]')
        or el.select('[class*="star--active"], [class*="icon-star-active"]')
        or el.find(attrs={'aria-label': re.compile(r'[1-5]\s*(sao|star|out|/)', re.I)})
    )
    has_variant = 'Phân loại hàng' in text
    has_comment_block = any(el.select(sel) for sel in COMMENT_SELECTORS)
    has_date = len(date_hits) >= 1
    return (has_marker or has_comment_block or has_date) and (has_stars or has_variant or has_comment_block)


# ── Auto-detect selector ───────────────────────────────────────────────────────
def auto_detect_selector(soup) -> str | None:
    candidates = {}
    for node in soup.find_all(string=DATE_RE):
        el = node.parent
        for _ in range(8):
            if el is None or el.name == 'body':
                break
            cls = el.get('class', [])
            if cls and el.name in ['div', 'li', 'article']:
                siblings = el.parent.find_all(el.name, class_=cls[0]) if el.parent else []
                valid_count = sum(1 for s in siblings if is_review_card(s))
                if valid_count >= 1:
                    key = f'{el.name}[class*="{cls[0]}"]'
                    candidates[key] = valid_count
                    break
            el = el.parent
    if candidates:
        best = max(candidates, key=candidates.get)
        return best
    return None


def detect_selector(soup) -> str:
    for s in KNOWN_REVIEW_SELECTORS:
        cards = soup.select(s)
        valid = [c for c in cards if is_review_card(c)]
        if valid:
            print(f'  [SELECTOR] known: {s} ({len(valid)} cards)')
            return s
    sel = auto_detect_selector(soup)
    if sel:
        valid = [c for c in soup.select(sel) if is_review_card(c)]
        if valid:
            print(f'  [SELECTOR] auto: {sel} ({len(valid)} cards)')
            return sel
    fallback = 'div.q2b7Oq[data-cmtid], div.q2b7Oq'
    print(f'  [SELECTOR] fallback: {fallback}')
    return fallback


def collect_product_links(driver, url_kind: str) -> list[str]:
    """Cuộn trang và gom link sản phẩm nhiều vòng để giảm miss do lazy-load."""
    product_links: list[str] = []
    seen: set[str] = set()
    max_rounds = 10 if url_kind == 'search' else 6
    stagnant_rounds = 0
    prev_count = -1

    for _ in range(max_rounds):
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        for a in soup.find_all('a', href=True):
            full = normalize_product_url(a['href'])
            if full and full not in seen:
                seen.add(full)
                product_links.append(full)

        for m in re.finditer(r'(https://shopee\.vn[^\s"\']*(?:/product/\d+/\d+|-[iI]\.\d+\.\d+)[^\s"\']*)', html):
            full = normalize_product_url(m.group(1))
            if full and full not in seen:
                seen.add(full)
                product_links.append(full)

        if len(product_links) == prev_count:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
            prev_count = len(product_links)

        if stagnant_rounds >= 2 and product_links:
            break

        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(2 if url_kind == 'search' else 1.5)

    return product_links


# ── Scroll đến khu vực review ─────────────────────────────────────────────────
def scroll_to_review_section(driver) -> bool:
    for heading in ['ĐÁNH GIÁ SẢN PHẨM', 'Đánh Giá Sản Phẩm', 'Product Ratings']:
        try:
            el = driver.find_element(By.XPATH, f"//*[normalize-space(text())='{heading}']")
            driver.execute_script(
                'arguments[0].scrollIntoView({block:"center",behavior:"smooth"});', el
            )
            time.sleep(2.5)
            driver.execute_script('window.scrollBy(0, 350);')
            time.sleep(1.5)
            print('  [SCROLL] Tìm thấy heading đánh giá ✓')
            return True
        except NoSuchElementException:
            continue

    print('  [SCROLL] Không thấy heading, cuộn từng bước...')
    total_h = driver.execute_script('return document.body.scrollHeight')
    for pct in range(55, 105, 5):
        driver.execute_script(f'window.scrollTo(0, {int(total_h * pct / 100)});')
        time.sleep(1.5)
        soup  = BeautifulSoup(driver.page_source, 'html.parser')
        found = any(
            [c for c in soup.select(s) if is_review_card(c)]
            for s in KNOWN_REVIEW_SELECTORS
        ) or auto_detect_selector(soup) is not None
        if found:
            driver.execute_script('window.scrollBy(0, 200);')
            time.sleep(1)
            print(f'  [SCROLL] Cards xuất hiện tại {pct}% ✓')
            return True

    print('  [SCROLL] Cuộn hết trang nhưng chưa thấy cards')
    return False


# ── Wait for reviews (poll DOM) ───────────────────────────────────────────────
def wait_for_reviews(driver, selector: str | None, max_wait: int = 22):
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            WebDriverWait(driver, 2).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
        except:
            pass
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        if selector:
            cards = soup.select(selector)
            valid = [c for c in cards if is_review_card(c)]
            if valid:
                return soup, valid
        else:
            sel = detect_selector(soup)
            if sel:
                valid = [c for c in soup.select(sel) if is_review_card(c)]
                return soup, valid
        time.sleep(1.5)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup, []


# ── Remove seller reply nodes ─────────────────────────────────────────────────
def remove_seller_reply_nodes(card):
    for reply in card.select(
        "div.p5tg3L,[class*='p5tg3L'],[class*='seller-reply'],"
        "[class*='shop-reply'],[class*='response']"
    ):
        reply.decompose()
    for node in card.find_all(string=SELLER_REPLY_RE):
        container = node.parent
        for _ in range(4):
            if container is None:
                break
            if container.name in ['div', 'section'] and len(container.get_text(' ', strip=True)) > 12:
                container.decompose()
                break
            container = container.parent


# ── Parse 1 review card ────────────────────────────────────────────────────────
def parse_card(card, stars_js=None) -> dict | None:
    try:
        remove_seller_reply_nodes(card)

        try:
            stars_js_int = int(stars_js) if stars_js is not None else None
        except:
            stars_js_int = None

        stars = stars_js_int if (stars_js_int is not None and 1 <= stars_js_int <= 5) \
                else extract_stars_from_card(card)

        full_text = normalize_space(card.get_text(' ', strip=True))
        m = DATE_RE.search(full_text)
        date_str = m.group(0) if m else ''

        review_date = parse_review_date(date_str)

        vm = re.search(r'Phân loại hàng\s*:\s*([^|]+)', full_text)
        variant = ''
        if vm:
            variant = normalize_space(vm.group(1))
            variant = re.sub(r'\d{4}-\d{2}-\d{2}.*', '', variant).strip()
            variant = variant.split('phản hồi của')[0].strip()

        comment = extract_comment_from_card(card)

        return {
            'date'   : review_date,
            'stars'  : max(0, min(int(stars or 0), 5)),
            'variant': variant,
            'comment': comment,
        }
    except Exception as e:
        print(f'  [ERR] parse_card: {e}')
        return None


# ── Scrape 1 sản phẩm ─────────────────────────────────────────────────────────
def scrape_product(driver, url: str) -> list[dict]:
    driver.get(url)
    time.sleep(3)
    wait_if_verification_page(driver)

    scroll_to_review_section(driver)

    reviews  = []
    seen_ids = set()
    selector = None

    for page_num in range(1, MAX_PAGES_PER_SP + 1):
        soup, valid_cards = wait_for_reviews(driver, selector, max_wait=22)

        if selector is None and valid_cards:
            selector = detect_selector(soup)
        elif selector is None:
            for _ in range(3):
                driver.execute_script('window.scrollBy(0, 400);')
                time.sleep(2)
                soup     = BeautifulSoup(driver.page_source, 'html.parser')
                selector = detect_selector(soup)
                if selector:
                    valid_cards = [c for c in soup.select(selector) if is_review_card(c)]
                    break

        if selector is None:
            break

        all_cards   = soup.select(selector)
        valid_cards = [c for c in all_cards if is_review_card(c)]

        if not valid_cards and page_num > 1:
            scroll_to_review_section(driver)
            time.sleep(2)
            soup        = BeautifulSoup(driver.page_source, 'html.parser')
            all_cards   = soup.select(selector)
            valid_cards = [c for c in all_cards if is_review_card(c)]

        if not valid_cards:
            break

        # Dedup theo hash nội dung card
        page_ids = {
            hashlib.md5(c.get_text(' ', strip=True)[:80].encode()).hexdigest()
            for c in valid_cards
        }
        new_ids = page_ids - seen_ids
        if not new_ids and page_num > 1:
            deadline = time.time() + 10
            while time.time() < deadline:
                time.sleep(1.5)
                soup        = BeautifulSoup(driver.page_source, 'html.parser')
                all_cards   = soup.select(selector)
                valid_cards = [c for c in all_cards if is_review_card(c)]
                page_ids    = {
                    hashlib.md5(c.get_text(' ', strip=True)[:80].encode()).hexdigest()
                    for c in valid_cards
                }
                new_ids = page_ids - seen_ids
                if new_ids:
                    break
            if not new_ids:
                break
        seen_ids |= new_ids

        js_stars_list = get_stars_for_cards(driver, selector) if selector else []

        for idx, card in enumerate(valid_cards):
            js_s = js_stars_list[idx] if idx < len(js_stars_list) else None
            rv   = parse_card(card, stars_js=js_s)
            if rv is None:
                continue
            has_signal = (
                rv.get('date') is not None
                or int(rv.get('stars') or 0) > 0
                or bool(rv.get('comment'))
            )
            if not has_signal:
                continue
            reviews.append(rv)

        # Click next page
        moved = False
        for sel in [
            'button.shopee-icon-button--right:not([disabled])',
            "[class*='page-next']:not([disabled])",
            "button[aria-label*='next']:not([disabled])",
        ]:
            try:
                btn = driver.find_element(By.CSS_SELECTOR, sel)
                driver.execute_script('arguments[0].scrollIntoView({block:"center"});', btn)
                time.sleep(0.5)
                driver.execute_script('arguments[0].click();', btn)
                moved = True
                break
            except NoSuchElementException:
                pass
        if not moved:
            break

    return reviews


# ── URL classification ─────────────────────────────────────────────────────────
def _classify_url(url: str) -> str:
    """Phân loại collection_url:
       - 'product'   : trỏ thẳng đến 1 sản phẩm  (vd /product/xxx/yyy hoặc -i.xxx.yyy)
       - 'search'    : trang search/mall-search  (cần scroll nhiều để lazy-load)
       - 'collection': trang shop/collection bình thường (giữ logic cũ).
    """
    u = (url or '').lower()
    if '/product/' in u or re.search(r'-i\.\d+\.\d+', u):
        return 'product'
    if ('entrypoint=shopbysearch' in u
            or '/mall/search' in u
            or '/search?' in u):
        return 'search'
    return 'collection'


def _build_output_frames(image_id: str, brand: str, df: pd.DataFrame,
                         raw_comment_str: str):
    """Chuẩn hoá output cho main.py, chỉ còn banner_summary + stats."""
    total_review = len(df)
    avg_stars = round(df['stars'].mean(), 2) if total_review > 0 else 0

    banner_row = pd.DataFrame([{
        'image_id'    : image_id,
        'source'      : 'shopee',
        'shop'        : brand,
        'total_review': total_review,
        'avg_stars'   : avg_stars,
        'raw_comment' : raw_comment_str,
    }])[['image_id', 'source', 'shop', 'total_review', 'avg_stars', 'raw_comment']]

    stats = {
        'total_review': total_review,
        'avg_stars'   : avg_stars,
        'raw_comment' : raw_comment_str,
    }
    return banner_row, stats


# ── Main extraction function ───────────────────────────────────────────────────
def run_shopee_extraction(driver, camp: dict, image_id: str, data_dir: str):
    brand    = camp['brand']
    camp_url = camp['collection_url']
    url_kind = _classify_url(camp_url)

    # ── Nhánh A: URL là 1 sản phẩm → scrape trực tiếp, bỏ bước gom product_links
    if url_kind == 'product':
        print(f"  🎯 URL trỏ thẳng đến 1 sản phẩm → scrape trực tiếp")
        product_links = [camp_url.split('?')[0]]
    else:
        # ── Nhánh B/C: trang shop / collection / search → gom link sản phẩm ──
        driver.get(camp_url)
        time.sleep(5)
        wait_if_verification_page(driver)
        product_links = collect_product_links(driver, url_kind)

        print(f"  📦 Tìm thấy {len(product_links)} link sản phẩm "
              f"cho [{brand}] (kind={url_kind})")

    all_reviews         = []
    banner_raw_comments = []
    qualified_count     = 0
    scanned             = 0
    product_results     = []   # lưu lại metadata để dùng cho fallback

    # ── PASS 1: Lấy SP có >= MIN_REVIEWS review hợp lệ ───────────────────
    for link in product_links:
        if qualified_count >= MAX_PRODUCTS:
            break
        scanned += 1
        print(f"  -> SP {scanned}/{len(product_links)} | đạt={qualified_count}/{MAX_PRODUCTS}...", end='\r')

        rv = scrape_product(driver, link)
        valid_cmts = [r['comment'] for r in rv if r.get('comment')]
        review_count = len(rv)
        is_qualified = (review_count >= MIN_REVIEWS) or (len(valid_cmts) >= MIN_COMMENTS)
        product_results.append({
            'reviews': rv,
            'review_count': review_count,
            'comment_count': len(valid_cmts),
            'qualified': is_qualified,
        })

        if is_qualified:
            qualified_count += 1
            all_reviews.extend(rv)
            banner_raw_comments.extend(valid_cmts[:MAX_COMMENTS_EACH])

        time.sleep(1.5)

    print()

    # ── PASS 2 (fallback): Nếu chưa đủ MAX_PRODUCTS ──────────────────────
    if qualified_count < MAX_PRODUCTS:
        print(f"  ⚠️  Chỉ có {qualified_count} SP đạt ≥{MIN_REVIEWS} review. Chạy fallback...")
        added = 0
        # Tận dụng kết quả đã quét nhưng bị loại, ưu tiên SP có nhiều review hơn
        fallback_results = sorted(
            [p for p in product_results if (not p['qualified']) and p['reviews']],
            key=lambda p: (p['review_count'], p['comment_count']),
            reverse=True,
        )
        for product in fallback_results:
            if qualified_count + added >= MAX_PRODUCTS:
                break
            rv = product['reviews']
            valid_cmts = [r['comment'] for r in rv if r.get('comment')]
            if rv:
                all_reviews.extend(rv)
                banner_raw_comments.extend(valid_cmts[:MAX_COMMENTS_EACH])
                added += 1

        # Quét thêm link chưa duyệt
        for link in product_links[scanned:]:
            if qualified_count + added >= MAX_PRODUCTS:
                break
            scanned += 1
            print(f"  -> [fallback] SP {scanned}/{len(product_links)}...", end='\r')
            rv = scrape_product(driver, link)
            if rv:
                all_reviews.extend(rv)
                valid_cmts = [r['comment'] for r in rv if r.get('comment')]
                banner_raw_comments.extend(valid_cmts[:MAX_COMMENTS_EACH])
                added += 1
            time.sleep(1.5)
        print()
        print(f"  📊 Fallback: {qualified_count} đạt chuẩn + {added} fallback = {qualified_count + added} SP")
    else:
        print(f"  📊 Đủ {qualified_count}/{MAX_PRODUCTS} SP đạt chuẩn.")

    if not all_reviews:
        print(f"⚠️  Không lấy được review nào cho: {brand}")
        empty_df = pd.DataFrame(columns=['date', 'stars', 'variant', 'comment'])
        return _build_output_frames(image_id, brand, empty_df, '')

    df = pd.DataFrame(all_reviews)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    raw_comment_str = ' | '.join(banner_raw_comments)

    banner_row, stats = _build_output_frames(
        image_id, brand, df, raw_comment_str
    )

    print(f"✅ [{brand}] ({image_id}) scraped | Reviews: {len(df)} | SP quét: {scanned} | SP ≥{MIN_COMMENTS} cmt: {qualified_count}")

    # Trả về banner_summary + stats để main.py ghi JSON và gom batch CSV
    return banner_row, stats