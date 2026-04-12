# AESTHETIC PRESSURE PROJECT
---
## ⚙️ Quy chuẩn kỹ thuật
- **Python Version:** 3.9+ (Thống nhất để tránh lỗi thư viện).
- **Image Standard:**
  - Resize all images to `width=1000px` before processing.
  - Convert to `RGB` (tránh lỗi kênh màu Alpha của ảnh .png).
- **NLP Library:** `underthesea` cho tiếng Việt, `nltk` cho tiếng Anh.
- **Git Branching Policy:**
  - Naming: `feature/[tên-thành-viên]-[tên-module]` (vd: `feature/vananh-canny-edge`).
  - Merge: Yêu cầu ít nhất 1 thành viên khác review code trước khi Merge vào Main.
---
## 📂 CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT STRUCTURE)
```markdown
Aesthetic-Pressure-ML/
├── data/                       # Chứa toàn bộ dữ liệu (Không push ảnh lên Git nếu quá nặng)
│   ├── raw/                    # Dữ liệu thô vừa cào về
│   │   ├── fb_ads/             # Ảnh & Review từ Facebook Ads (Vân Anh)
│   │   ├── shopee_lazada/      # Ảnh & Review từ TMĐT (Linh Khánh)
│   │   └── social_media/       # Ảnh từ Insta/Pinterest (Hà Ngọc)
│   ├── processed/              # Dữ liệu sau khi xử lý
│   │   ├── features_cv.csv     # Chỉ số toán học từ GĐ 2 (Shared)
│   │   ├── labels_nlp.csv      # Nhãn áp lực từ GĐ 3 (Shared)
│   │   └── final_dataset.csv   # Dataset cuối cùng dùng cho ML (Hà Ngọc merge)
├── notebooks/                  # File Google Colab / Jupyter Notebook để thử nghiệm
│   ├── 01_scraping_demo.ipynb
│   ├── 02_cv_extraction.ipynb
│   └── 03_model_training.ipynb
├── src/                        # Mã nguồn chính của dự án (Python scripts)
│   ├── scraping/               # Module cào dữ liệu (GĐ 1)
│   │   ├── fb_scraper.py       # (Vân Anh)
│   │   ├── ecom_scraper.py     # (Linh Khánh)
│   │   └── social_scraper.py   # (Hà Ngọc)
│   ├── features/               # Module xử lý ảnh OpenCV (GĐ 2)
│   │   ├── edge_processor.py   # (Vân Anh)
│   │   ├── color_processor.py  # (Linh Khánh)
│   │   └── text_processor.py   # (Hà Ngọc)
│   ├── sentiment/              # Module xử lý NLP (GĐ 3)
│   │   ├── score_logic.py      # Tính AP Score (Vân Anh)
│   │   └── nlp_engine.py       # Phân tích cảm xúc (Linh Khánh)
│   └── models/                 # Module huấn luyện ML (GĐ 4)
│       ├── random_forest.py    # (Vân Anh)
│       ├── svm_knn.py          # (Linh Khánh)
│       └── evaluation.py       # (Hà Ngọc)
├── models/                     # Lưu trữ model đã huấn luyện (.pkl)
├── reports/                    # Kết quả nghiên cứu (GĐ 5)
│   ├── figures/                # Biểu đồ, Heatmap, Visualization (Linh Khánh)
│   └── essay/                  # File nháp tiểu luận (Vân Anh & Team)
├── demo/                       # Sản phẩm cuối cùng
│   └── app_predict.py          # Giao diện dự báo mẫu (Hà Ngọc)
├── .gitignore                  # Chặn các file rác, file ảnh nặng không cần thiết
├── README.md                   # Hướng dẫn dự án & Phân chia công việc
└── requirements.txt            # Danh sách thư viện cần cài đặt (OpenCV, Pandas,...)
```

---
## 👥 Team Members
* **Lê Vân Anh (Leader):** Chịu trách nhiệm Logic tổng, CV (Cạnh), Scraping FB và Model RF.
* **Linh Khánh:** Chịu trách nhiệm NLP Sentiment, CV (Màu sắc), Scraping TMĐT và Model SVM/KNN.
* **Hà Ngọc:** Chịu trách nhiệm Merging Data, CV (Chữ/Khoảng trắng), Scraping Social và Code Demo.

## 🚀 Quy trình làm việc trên Git (Workflow)
Để tránh xung đột code, team thống nhất:
1. **Branching:** Không push trực tiếp lên `main`. Mỗi người làm trên branch riêng: `dev-vananh`, `dev-khanh`, `dev-ngoc`.
2. **Merging:** Sau khi hoàn thành một module, thực hiện **Pull Request (PR)** để Leader duyệt trước khi merge vào `main`.
3. **Commit Message:** Ghi rõ nội dung: `feat: add edge detection`, `fix: bug in nlp script`.

## 🛠 Hướng dẫn cài đặt
```bash
# Clone dự án
git clone https://github.com/[username]/Aesthetic-Pressure-ML.git

# Cài đặt thư viện
pip install -r requirements.txt
```

## 📊 Quản lý dự án
link notion quản lý dự án: [AESTHETIC PRESSURE](https://www.notion.so/AESTHETIC-PRESSURE-337001c683f180769b8efc89ef4d5f0f?source=copy_link)
* [x] GĐ 1: Thu thập dữ liệu (In Progress)
* [ ] GĐ 2: Trích xuất đặc trưng (Pending)
* [ ] GĐ 3: NLP & Merging (Pending)
* [ ] GĐ 4: Huấn luyện Model (Pending)
* [ ] GĐ 5: Tiểu luận & Demo (Pending)

---

### 💡 Lưu ý:
1. **File `requirements.txt`:** Mỗi khi ai đó cài thêm thư viện mới (ví dụ: `EasyOCR` hay `Underthesea`), hãy cập nhật vào file này để 2 bạn còn lại chỉ cần chạy lệnh `pip install` là có môi trường giống hệt nhau.
2. **Notion Sync:** Mỗi khi Merge code thành công trên Git, hãy chuyển trạng thái Task trên Notion sang **Done**.
