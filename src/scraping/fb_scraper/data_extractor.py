"""
Data Extraction & CSV Export Module
Trích xuất dữ liệu (likes, shares, comments) từ posts và tạo file CSV chuẩn
"""

import os
import json
import csv


class FacebookDataExtractor:
    """
    Trích xuất dữ liệu từ Facebook posts và xlsx ra CSV chuẩn
    """
    
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        self.image_counter = 0
        self.data_records = []
        self.img_id_mapping = {}  # Map từ image path cũ sang img_id mới
    
    def get_next_img_id(self):
        """Tạo mã định danh ảnh tiếp theo: fac_001, fac_002, ..."""
        self.image_counter += 1
        return f"fac_{self.image_counter:03d}"

    def _extract_comments_text(self, comments):
        texts = []

        if isinstance(comments, list):
            for item in comments:
                if isinstance(item, str):
                    value = item.strip()
                    if value:
                        texts.append(value)
                elif isinstance(item, dict):
                    value = (
                        item.get("text")
                        or item.get("comment_text")
                        or item.get("message")
                        or item.get("body")
                        or ""
                    )
                    value = str(value).strip()
                    if value:
                        texts.append(value)

        return " | ".join(texts)
    
    def extract_all_posts(self):
        """Quét toàn bộ folder facebook/ và tìm tất cả posts"""
        print("\n📊 Bắt đầu xử lý ảnh...")
        
        total_images = 0
        total_posts = 0
        
        # Quét folder facebook/[fanpage_name]/[post_id]/
        facebook_dir = os.path.join(self.output_dir, "facebook")
        if not os.path.exists(facebook_dir):
            print(f"❌ Không tìm thấy: {facebook_dir}")
            return 0
        
        for fanpage_name in os.listdir(facebook_dir):
            fanpage_dir = os.path.join(facebook_dir, fanpage_name)
            if not os.path.isdir(fanpage_dir):
                continue
            
            for post_id in os.listdir(fanpage_dir):
                post_dir = os.path.join(fanpage_dir, post_id)
                if not os.path.isdir(post_dir):
                    continue
                
                json_file = os.path.join(post_dir, f"{post_id}.json")
                if os.path.exists(json_file):
                    images = self.process_post_json(json_file)
                    if images > 0:
                        total_posts += 1
                        total_images += images
        
        print(f"✅ Tìm được: {total_posts} posts, {total_images} ảnh")
        return total_images
    
    def process_post_json(self, json_file_path):
        """Xử lý một file JSON post"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                post_data = json.load(f)
            
            post_id = post_data.get("post_id")
            if not post_id:
                return 0

            fanpage_name = str(
                post_data.get("page_name")
                or os.path.basename(os.path.dirname(os.path.dirname(json_file_path)))
                or "Unknown"
            )
            
            # Lấy dữ liệu từ JSON
            likes = int(post_data.get("likes", 0))
            shares = int(post_data.get("shares", 0))
            
            # Lấy reaction breakdown
            react_like = int(post_data.get("react_like", 0))
            react_love = int(post_data.get("react_love", 0))
            react_care = int(post_data.get("react_care", 0))
            react_wow = int(post_data.get("react_wow", 0))
            react_angry = int(post_data.get("react_angry", 0))
            
            # Tính tổng reactions
            total_react = react_like + react_love + react_care + react_wow + react_angry
            
            # Combine comments thành chuỗi
            comments_data = post_data.get("comments", [])
            comments_text = self._extract_comments_text(comments_data)
            if isinstance(comments_data, list):
                comment_count = len([c for c in comments_data if str(c).strip()])
            else:
                comment_count = int(post_data.get("comment_count", 0) or 0)
            
            # BỎ QUA posts không có comments
            if not comments_text or not comments_text.strip():
                return 0

            media_urls = []
            for m in post_data.get("media", []):
                if isinstance(m, dict):
                    url = (m.get("url") or "").strip()
                    if url:
                        media_urls.append(url)
            
            # Tìm tất cả ảnh trong folder post
            post_folder = os.path.dirname(json_file_path)
            image_files = []
            
            if os.path.exists(post_folder):
                for file in os.listdir(post_folder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                        image_files.append(file)
            
            # Sắp xếp ảnh
            image_files.sort()
            
            # Xử lý từng ảnh
            image_count = 0
            for idx, image_file in enumerate(image_files):
                image_path = os.path.join(post_folder, image_file)
                img_id = self.get_next_img_id()
                image_url = media_urls[idx] if idx < len(media_urls) else ""
                
                # Tạo record CSV với các cột mới
                record = {
                    "img_id": img_id,
                    "source": "Facebook",
                    "fanpage": fanpage_name,
                    "url": image_url,
                    "total_react": total_react,
                    "share_count": shares,
                    "react_like": react_like,
                    "react_love": react_love,
                    "react_care": react_care,
                    "react_wow": react_wow,
                    "react_angry": react_angry,
                    "comment_count": comment_count,
                    "raw_comment": comments_text
                }
                
                # Lưu mapping old → new
                self.img_id_mapping[image_path] = img_id
                self.data_records.append(record)
                
                image_count += 1
            
            return image_count
        
        except Exception as e:
            print(f"  ⚠️ Lỗi: {json_file_path} - {e}")
            return 0
    
    def rename_images(self):
        """Đổi tên ảnh theo img_id"""
        print("\n🖼️  Đổi tên ảnh...")
        
        rename_count = 0
        
        for old_path, img_id in self.img_id_mapping.items():
            try:
                if not os.path.exists(old_path):
                    continue
                
                # Lấy extension
                _, ext = os.path.splitext(old_path)
                
                # Tạo đường dẫn mới
                post_dir = os.path.dirname(old_path)
                new_filename = f"{img_id}{ext}"
                new_path = os.path.join(post_dir, new_filename)
                
                # Đổi tên file
                if old_path != new_path and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    rename_count += 1
                    print(f"  ✓ {os.path.basename(old_path)} → {new_filename}")
            
            except Exception as e:
                print(f"  ✗ Lỗi đổi tên {old_path}: {e}")
        
        print(f"✅ Đã đổi tên: {rename_count} ảnh\n")
        return rename_count
    
    def export_csv(self, csv_filename="raw_fb_data.csv"):
        """Xuất dữ liệu ra CSV tại data/raw/facebook/raw_fb_data.csv"""
        facebook_dir = os.path.join(self.output_dir, "facebook")
        
        # Tạo folder facebook nếu chưa có
        if not os.path.exists(facebook_dir):
            os.makedirs(facebook_dir, exist_ok=True)
        
        csv_path = os.path.join(facebook_dir, csv_filename)
        
        print(f"📝 Xuất CSV: {os.path.basename(csv_path)}")
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['img_id', 'source', 'fanpage', 'url', 'total_react', 'share_count', 
                             'react_like', 'react_love', 'react_care', 'react_wow', 'react_angry',
                             'comment_count', 'raw_comment']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for record in self.data_records:
                    writer.writerow(record)
            
            print(f"✅ Xuất thành công: {len(self.data_records)} dòng\n")
            return csv_path
        
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return None
    
    def run(self, rename_images=True):
        """Chạy toàn bộ quy trình"""
        self.extract_all_posts()
        if rename_images:
            self.rename_images()
        csv_path = self.export_csv()
        return csv_path, len(self.data_records)


def run_data_extraction(output_dir="data/raw", rename_images=True):
    """Hàm tiện lợi"""
    extractor = FacebookDataExtractor(output_dir)
    return extractor.run(rename_images)


if __name__ == "__main__":
    csv_path, count = run_data_extraction()
    if csv_path:
        print(f"🎉 Hoàn tất! CSV: {csv_path}")

