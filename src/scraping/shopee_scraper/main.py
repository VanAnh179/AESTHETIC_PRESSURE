import sys
import re
import time
import json
import random
import urllib.request
from pathlib import Path
import pandas as pd
import undetected_chromedriver as uc
import winreg

# === XỬ LÝ ĐƯỜNG DẪN ===
CURRENT_FILE = Path(__file__).resolve()
SCRAPE_DIR   = CURRENT_FILE.parent              # src/scraping/shopee_scraper/
ROOT_DIR     = SCRAPE_DIR.parent.parent.parent  # project root (Aesthetic-Pressure-ML/)

if str(SCRAPE_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPE_DIR))

from shopee_extractor import run_shopee_extraction

# Output: data/raw/shopee/
SHOPEE_DIR  = ROOT_DIR / 'data' / 'raw' / 'shopee'
BATCH_SIZE  = 5   # ghi CSV sau mỗi N shop


def get_chrome_major_version() -> int | None:
    """Đọc version Chrome đã cài trên Windows và trả về major version."""
    reg_targets = [
        (winreg.HKEY_CURRENT_USER, r'Software\Google\Chrome\BLBeacon', 'version'),
        (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Google\Chrome\BLBeacon', 'version'),
        (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\WOW6432Node\Google\Chrome\BLBeacon', 'version'),
    ]
    for root, subkey, value_name in reg_targets:
        try:
            with winreg.OpenKey(root, subkey) as key:
                version = str(winreg.QueryValueEx(key, value_name)[0]).strip()
                major = version.split('.', 1)[0]
                if major.isdigit():
                    return int(major)
        except Exception:
            continue
    return None


def init_driver():
    opt = uc.ChromeOptions()
    opt.add_argument('--lang=vi-VN,vi')
    opt.add_argument('--window-size=1366,768')
    opt.add_argument('--no-first-run')
    opt.add_argument('--no-default-browser-check')
    version_main = get_chrome_major_version()
    if version_main is not None:
        print(f'🔧 Chrome local major version: {version_main}')
    else:
        print('⚠️  Không dò được version Chrome, thử khởi động driver mặc định...')
    driver = uc.Chrome(
        options=opt,
        headless=False,
        version_main=version_main,
        use_subprocess=True,
    )
    driver.set_page_load_timeout(60)
    return driver


def wait_for_manual_login(driver):
    """Dừng hoàn toàn bằng input() — không chạy ngầm — tránh anti-bot Shopee."""
    driver.get('https://shopee.vn/buyer/login')
    time.sleep(3)

    print()
    print('=' * 55)
    print('  ĐÃ MỞ TRANG ĐĂNG NHẬP SHOPEE TRÊN CỬA SỔ CHROME')
    print()
    print('  Làm theo các bước:')
    print('  1. Đăng nhập tài khoản Shopee')
    print('  2. Vượt captcha nếu có (làm thủ công trên Chrome)')
    print('  3. Chờ trang chủ load xong hoàn toàn')
    print('  4. Quay lại đây và nhấn Enter')
    print('=' * 55)

    input('\n>>> Nhấn Enter khi đã đăng nhập xong: ')

    if 'login' in driver.current_url or 'buyer/login' in driver.current_url:
        print()
        print('⚠️  Có vẻ vẫn đang ở trang login.')
        print('   Hãy kiểm tra lại trên Chrome xem đã đăng nhập chưa.')
        input('>>> Nhấn Enter lần nữa khi chắc chắn đã đăng nhập: ')

    print()
    print('✅ OK! Bắt đầu scraping...')
    print()
    time.sleep(2)


def download_image(url: str, save_path: Path) -> bool:
    """Tải ảnh banner từ URL về save_path. Trả về True nếu thành công."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            save_path.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f'  ⚠️  Không tải được ảnh: {e}')
        return False


def find_campaign_json() -> Path | None:
    for directory in [ROOT_DIR, SCRAPE_DIR]:
        for name in ['campaign.json', 'compaign.json']:
            p = directory / name
            if p.exists():
                return p
    return None


def is_banner_done(camp: dict, shopee_dir: Path) -> bool:
    """Banner được coi là đã scrape OK nếu:
       - JSON có scrape_status='done', hoặc
       - JSON cũ có total_review > 0
       Trả về True → skip trong lần chạy này (resume).
    """
    cid   = camp.get('id', '')
    brand = camp.get('brand', '')
    if not cid or not brand:
        return False
    json_file = shopee_dir / brand / cid / f'{cid}.json'
    if not json_file.exists():
        return False
    try:
        data = json.loads(json_file.read_text(encoding='utf-8'))
        if data.get('scrape_status') == 'done':
            return True
        if data.get('scrape_status') == 'failed':
            return False
        return int(data.get('total_review', 0)) > 0
    except Exception:
        return False


def check_captcha(driver) -> None:
    """Nếu URL hiện tại có dấu hiệu captcha / verify → chờ user giải tay."""
    try:
        url = (driver.current_url or '').lower()
    except Exception:
        return
    if any(k in url for k in ('verify', 'captcha', 'anti-bot', 'bot-check')):
        print()
        print('🚧  PHÁT HIỆN CAPTCHA / TRANG XÁC THỰC Ở CHROME')
        print('   → Mở cửa sổ Chrome, giải captcha tay, rồi quay lại đây.')
        input('>>> Nhấn Enter khi đã giải xong để tiếp tục: ')


def get_existing_image_id(camp: dict, shopee_dir: Path) -> str | None:
    """Nếu banner đã từng được chạy dở trước đó, ưu tiên dùng lại image_id cũ."""
    cid   = camp.get('id', '')
    brand = camp.get('brand', '')
    if not cid or not brand:
        return None

    banner_dir = shopee_dir / brand / cid
    json_file  = banner_dir / f'{cid}.json'

    if json_file.exists():
        try:
            data = json.loads(json_file.read_text(encoding='utf-8'))
            image_id = str(data.get('image_id', '')).strip()
            if re.fullmatch(r'sho_\d+', image_id):
                return image_id
        except Exception:
            pass

    for img_path in banner_dir.glob('sho_*.png'):
        if img_path.stem.startswith('sho_'):
            return img_path.stem
    return None


def get_next_image_idx(shopee_dir: Path) -> int:
    """Lấy image_id tiếp theo theo kiểu đánh tiếp `max + 1`."""
    max_idx = 0
    csv_path = shopee_dir / 'banner_summary.csv'
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=['image_id'], encoding='utf-8-sig')
            nums = df['image_id'].astype(str).str.extract(r'sho_(\d+)')[0].dropna().astype(int)
            if not nums.empty:
                max_idx = max(max_idx, int(nums.max()))
        except Exception:
            pass

    for json_path in shopee_dir.rglob('*.json'):
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
            image_id = str(data.get('image_id', '')).strip()
            nums = pd.Series([image_id]).str.extract(r'sho_(\d+)')[0].dropna()
            if not nums.empty:
                max_idx = max(max_idx, int(nums.iloc[0]))
        except Exception:
            continue

    for img_path in shopee_dir.rglob('sho_*.png'):
        nums = pd.Series([img_path.stem]).str.extract(r'sho_(\d+)')[0].dropna()
        if not nums.empty:
            max_idx = max(max_idx, int(nums.iloc[0]))

    return max_idx + 1 if max_idx > 0 else 1


def flush_csv(pending_banner: list, shopee_dir: Path):
    """Ghi append banner_summary.csv, không ghi đè dữ liệu cũ."""
    if not pending_banner:
        print('  ⏭  Batch này không có dữ liệu mới để ghi CSV.')
        return

    csv_banner = shopee_dir / 'banner_summary.csv'

    pd.concat(pending_banner, ignore_index=True).to_csv(
        csv_banner, mode='a', index=False,
        header=not csv_banner.exists(), encoding='utf-8-sig'
    )
    print(f'  💾 Đã append {len(pending_banner)} banner vào banner_summary.csv')


def main():
    json_path = find_campaign_json()
    if json_path is None:
        print()
        print('❌ LỖI: Không tìm thấy file campaign.json!')
        print(f'   Đã tìm ở:\n   • {ROOT_DIR}\n   • {SCRAPE_DIR}')
        return

    print(f'📄 Đọc: {json_path}')
    with open(json_path, 'r', encoding='utf-8') as f:
        campaigns = json.load(f).get('campaigns', [])

    print(f'📋 Tìm thấy {len(campaigns)} campaign.')

    SHOPEE_DIR.mkdir(parents=True, exist_ok=True)

    next_idx = get_next_image_idx(SHOPEE_DIR)
    print(f'🔢 Image id mới sẽ đánh tiếp từ: sho_{next_idx:03d}')
    print()

    done_flags       = [is_banner_done(camp, SHOPEE_DIR) for camp in campaigns]
    remaining_total  = sum(1 for done in done_flags if not done)
    skipped_total    = len(campaigns) - remaining_total

    print(f'✅ Đã hoàn tất từ trước: {skipped_total} banner')
    print(f'🆕 Cần scrape tiếp: {remaining_total} banner')
    print(f'📦 Cứ {BATCH_SIZE} banner mới liên tiếp sẽ dừng để append CSV.')
    print()

    if remaining_total == 0:
        print('🎉 Không còn banner nào cần scrape tiếp.')
        return

    print('🌐 Đang khởi động Chrome (hiển thị cửa sổ thật)...')
    driver = init_driver()

    wait_for_manual_login(driver)

    pending_banner: list = []
    processed_new = 0

    for offset, camp in enumerate(campaigns):
        brand = camp['brand']

        if done_flags[offset]:
            print(f'⏭  Bỏ qua [{camp["id"]}] — banner này đã scrape xong trước đó.')
            continue

        existing_image_id = get_existing_image_id(camp, SHOPEE_DIR)
        if existing_image_id:
            image_id = existing_image_id
            print(f'🔁 Resume banner [{camp["id"]}] với image_id cũ: {image_id}')
        else:
            image_id = f'sho_{next_idx:03d}'
            next_idx += 1

        processed_new += 1

        print(f"{'='*55}")
        print(f"🚀 BANNER {processed_new}/{remaining_total}: {brand} — {camp.get('campaign_name', '')}")
        print(f"   id={camp['id']} | image_id={image_id}")
        print(f"{'='*55}")

        check_captcha(driver)

        banner_dir = SHOPEE_DIR / brand / camp['id']
        banner_dir.mkdir(parents=True, exist_ok=True)

        img_url  = camp.get('image', '')
        img_path = banner_dir / f'{image_id}.png'
        if img_path.exists() and img_path.stat().st_size > 0:
            print(f'  🖼  Ảnh đã tồn tại → dùng lại {img_path.name}')
        elif img_url:
            print(f'  📥 Đang tải ảnh banner → {img_path.name}')
            download_image(img_url, img_path)
        else:
            print('  ⚠️  Không có URL ảnh banner trong config.')

        banner_row = None
        stats = {
            'total_review': 0,
            'avg_stars': 0,
            'raw_comment': '',
        }
        scrape_status = 'failed'

        try:
            result = run_shopee_extraction(driver, camp, image_id, str(SHOPEE_DIR))
            if result is not None:
                banner_row, stats = result
                if stats.get('total_review', 0) > 0:
                    if banner_row is not None:
                        pending_banner.append(banner_row)
                    scrape_status = 'done'
                else:
                    print('  ⚠️  Không thu được review hợp lệ -> không append CSV cho banner này.')
        except Exception as e:
            print(f'❌ Lỗi [{brand}]: {e}')
            import traceback
            traceback.print_exc()

        json_name = f"{camp['id']}.json"
        info = {
            **camp,
            'image_id': image_id,
            'total_review': stats['total_review'],
            'avg_stars': stats['avg_stars'],
            'raw_comment': stats['raw_comment'],
            'scrape_status': scrape_status,
        }
        (banner_dir / json_name).write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(f'  💾 Đã lưu {json_name} '
              f'(total_review={info["total_review"]}, avg_stars={info["avg_stars"]})')

        is_last_new = processed_new == remaining_total
        batch_done  = processed_new % BATCH_SIZE == 0
        if batch_done or is_last_new:
            print()
            print(f'{"─"*55}')
            print(f'⏸  CHECKPOINT: đã xử lý {processed_new}/{remaining_total} banner mới.')
            print(f'{"─"*55}')
            flush_csv(pending_banner, SHOPEE_DIR)
            pending_banner.clear()
            if not is_last_new:
                print()

        if not is_last_new:
            pause = random.uniform(6, 14)
            print(f'  ⏳ Nghỉ {pause:.1f}s trước banner tiếp theo...')
            time.sleep(pause)

    print()
    print("🎉 HOÀN THÀNH! Kiểm tra thư mục 'data/raw/shopee' để xem kết quả.")

    try:
        driver.service.stop()
        driver.quit()
    except Exception:
        pass
    finally:
        try:
            driver._keep_user_data_dir = False
        except Exception:
            pass


if __name__ == '__main__':
    main()