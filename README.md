# Phân loại tin tức tiếng Việt và gợi ý bài tương tự

Bài tập lớn môn Trí tuệ nhân tạo, Học viện Công nghệ Bưu chính Viễn thông (PTIT) - mục 5b.

## Thành viên nhóm

| Họ tên | Mã sinh viên |
|---|---|
| ... | ... |

## Mô tả

Hệ thống nhận một bài báo tiếng Việt, dự đoán chuyên mục (10 chuyên mục) và gợi ý 5 bài tương tự.
Dữ liệu: bộ VNTC (84.128 bài dùng được). Tiền xử lý: tách từ bằng pyvi, loại stopword, vector hóa TF-IDF (50.000 từ).
Thuật toán: Naive Bayes, kNN (cosine), Decision Tree.

## Kết quả (tập test 16.826 bài)

| Mô hình | Accuracy | Macro-F1 |
|---|---|---|
| Naive Bayes cơ bản (alpha=1,0) | 0,8728 | 0,8443 |
| **Naive Bayes tinh chỉnh (alpha=0,1, prior đều)** | **0,8957** | **0,8855** |
| Naive Bayes tinh chỉnh + chi-square (10.000 đặc trưng) | 0,8927 | 0,8822 |
| kNN cosine (k=11) | 0,8565 | 0,8364 |
| Decision Tree (2.000 đặc trưng, độ sâu 40) | 0,7488 | 0,7307 |

## Chạy demo

```
pip install -r requirements.txt
python -m streamlit run app.py
```

Mô hình đã huấn luyện nằm sẵn ở `model/demo_model.pkl`.

## Huấn luyện lại từ đầu

1. Tải dữ liệu theo hướng dẫn ở `data/README.md`.
2. `python build_dataset.py` (tiền xử lý toàn bộ, tạo `data/clean_full.csv`)
3. `python train.py` (Naive Bayes cơ bản)
4. `python compare.py` (chọn tham số, so sánh các thuật toán)
5. `python build_demo_model.py` (tạo `model/demo_model.pkl`)

## Hạn chế

- Bộ dữ liệu VNTC khá cũ, nên tin tức hiện nay có thể bị phân loại kém chính xác hơn.
- Chuyên mục khoa học và đời sống có recall thấp hơn các chuyên mục khác.