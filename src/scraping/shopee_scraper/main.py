import sys
import time
import json
import urllib.request
from pathlib import Path
import pandas as pd
import undetected_chromedriver as uc

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


def init_driver():
    opt = uc.ChromeOptions()
    opt.add_argument('--lang=vi-VN,vi')
    opt.add_argument('--window-size=1366,768')
    opt.add_argument('--no-first-run')
    opt.add_argument('--no-default-browser-check')
    driver = uc.Chrome(options=opt, headless=False, version_main=None, use_subprocess=True)
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


def get_next_image_idx(shopee_dir: Path) -> int:
    """Đọc banner_summary.csv để lấy id cuối cùng, trả về index tiếp theo."""
    csv_path = shopee_dir / 'banner_summary.csv'
    if not csv_path.exists():
        return 1
    try:
        df = pd.read_csv(csv_path, usecols=['image_id'], encoding='utf-8-sig')
        nums = df['image_id'].str.extract(r'sho_(\d+)')[0].dropna().astype(int)
        return int(nums.max()) + 1 if not nums.empty else 1
    except Exception:
        return 1


def flush_csv(pending_banner: list, pending_daily: list, shopee_dir: Path):
    """Ghi append 2 danh sách dataframe vào 2 file CSV."""
    if not pending_banner:
        return

    csv_banner = shopee_dir / 'banner_summary.csv'
    csv_daily  = shopee_dir / 'daily_summary.csv'

    pd.concat(pending_banner, ignore_index=True).to_csv(
        csv_banner, mode='a', index=False,
        header=not csv_banner.exists(), encoding='utf-8-sig'
    )
    if pending_daily:
        pd.concat(pending_daily, ignore_index=True).to_csv(
            csv_daily, mode='a', index=False,
            header=not csv_daily.exists(), encoding='utf-8-sig'
        )
    print(f'  💾 Đã lưu {len(pending_banner)} shop vào banner_summary.csv & daily_summary.csv')


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

    # Lấy index tiếp theo dựa vào CSV đã có (tránh ghi đè id cũ)
    start_idx = get_next_image_idx(SHOPEE_DIR)
    print(f'🔢 Bắt đầu đánh id từ: sho_{start_idx:03d}')
    print()

    print('🌐 Đang khởi động Chrome (hiển thị cửa sổ thật)...')
    driver = init_driver()

    wait_for_manual_login(driver)

    pending_banner: list = []
    pending_daily:  list = []

    for offset, camp in enumerate(campaigns):
        image_id = f'sho_{start_idx + offset:03d}'
        brand    = camp['brand']

        print(f"={'='*54}")
        print(f"🚀 CAMPAIGN {offset+1}/{len(campaigns)}: {brand} — {camp.get('campaign_name', '')}")
        print(f"={'='*54}")

        # ── Tạo thư mục: data/raw/shopee/{brand}/{campaign_id}/ ──────────
        banner_dir = SHOPEE_DIR / brand / camp['id']
        banner_dir.mkdir(parents=True, exist_ok=True)

        # ── Tải ảnh banner → luôn lưu dạng .png ─────────────────────────
        img_url  = camp.get('image', '')
        img_path = banner_dir / f'{image_id}.png'
        if img_url:
            print(f'  📥 Đang tải ảnh banner → {img_path.name}')
            download_image(img_url, img_path)
        else:
            print('  ⚠️  Không có URL ảnh banner trong config.')

        # ── Lưu {campaign_id}.json ────────────────────────────────────────
        json_name = f"{camp['id']}.json"
        info = {**camp, 'image_id': image_id}
        (banner_dir / json_name).write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(f'  💾 Lưu {json_name}')

        # ── Scrape — nhận lại dataframe thay vì ghi thẳng ────────────────
        try:
            result = run_shopee_extraction(driver, camp, image_id, str(SHOPEE_DIR))
            if result is not None:
                banner_row, daily_rows = result
                pending_banner.append(banner_row)
                pending_daily.append(daily_rows)
        except Exception as e:
            print(f'❌ Lỗi [{brand}]: {e}')
            import traceback
            traceback.print_exc()

        # ── Ghi CSV mỗi BATCH_SIZE shop hoặc shop cuối cùng ──────────────
        is_last = (offset + 1) == len(campaigns)
        if len(pending_banner) >= BATCH_SIZE or is_last:
            flush_csv(pending_banner, pending_daily, SHOPEE_DIR)
            pending_banner.clear()
            pending_daily.clear()

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