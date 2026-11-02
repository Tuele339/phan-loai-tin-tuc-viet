import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

# ====== CẤU HÌNH (chọn từ kết quả compare.py) ======
ALPHA = 0.1
MAX_FEATURES = 50000
N_INDEX = 20000      # số bài đưa vào chỉ mục gợi ý "bài tương tự"
# ===================================================

os.makedirs("model", exist_ok=True)

df = pd.read_csv(os.path.join("data", "clean_full.csv"), encoding="utf-8",
                 usecols=["label", "snippet", "clean"])
df["clean"] = df["clean"].fillna("")
df["snippet"] = df["snippet"].fillna("")
df = df[df["clean"].str.split().str.len() >= 20].reset_index(drop=True)
print("Số bài dùng được:", len(df))

# Chia train/test giống hệt train.py và compare.py (80:20, random_state=42)
idx_all = np.arange(len(df))
idx_trval, idx_test = train_test_split(
    idx_all, test_size=0.2, random_state=42, stratify=df["label"])

# Huấn luyện mô hình cuối cùng
tfidf = TfidfVectorizer(max_features=MAX_FEATURES, min_df=2)
X_trval = tfidf.fit_transform(df["clean"].iloc[idx_trval])
y_trval = df["label"].iloc[idx_trval].values
nb = MultinomialNB(alpha=ALPHA, fit_prior=False).fit(X_trval, y_trval)

# Kiểm tra: kết quả phải khớp với bảng so sánh của compare.py
X_test = tfidf.transform(df["clean"].iloc[idx_test])
y_test = df["label"].iloc[idx_test].values
pred = nb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(f"Macro-F1: {f1_score(y_test, pred, average='macro'):.4f}")

# Chỉ mục gợi ý bài tương tự: lấy ngẫu nhiên N_INDEX bài trong tập train
rng = np.random.default_rng(42)
pos = np.sort(rng.choice(len(idx_trval), size=min(N_INDEX, len(idx_trval)), replace=False))
X_index = X_trval[pos].astype(np.float32).tocsr()
index_labels = [str(x) for x in y_trval[pos]]
index_snippets = [str(x) for x in df["snippet"].iloc[idx_trval].values[pos]]

# Bỏ thuộc tính chỉ dùng để tham khảo cho file nhẹ hơn
if hasattr(tfidf, "stop_words_"):
    tfidf.stop_words_ = None

bundle = {
    "tfidf": tfidf,
    "nb": nb,
    "X_index": X_index,
    "index_labels": index_labels,
    "index_snippets": index_snippets,
}
duong_dan = os.path.join("model", "demo_model.pkl")
joblib.dump(bundle, duong_dan, compress=3)
mb = os.path.getsize(duong_dan) / 1024 / 1024
print(f"\nĐã lưu {duong_dan}: {mb:.1f} MB")
if mb > 90:
    print("CẢNH BÁO: file lớn hơn 90 MB, khó đưa lên GitHub. Hãy giảm N_INDEX xuống (ví dụ 10000) rồi chạy lại.")