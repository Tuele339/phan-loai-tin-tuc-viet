import gc
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

MAX_FEATURES = 50000
os.makedirs("images", exist_ok=True)
pd.set_option("display.width", 250)


def tim_lang_gieng(X_query, XT_train, k_max, batch=100):
    """Với mỗi bài trong X_query, trả về chỉ số của k_max bài train giống nhất
    (sắp xếp từ giống nhất đến kém hơn). Xử lý theo lô nhỏ để tiết kiệm RAM.
    Vector TF-IDF đã chuẩn hóa L2 nên cosine similarity = tích vô hướng."""
    X_query = X_query.astype(np.float32).tocsr()
    ket_qua = []
    for i in range(0, X_query.shape[0], batch):
        sim = (X_query[i:i + batch] @ XT_train).toarray()
        top = np.argpartition(-sim, k_max - 1, axis=1)[:, :k_max]
        hang = np.arange(top.shape[0])[:, None]
        thu_tu = np.argsort(-sim[hang, top], axis=1)
        ket_qua.append(top[hang, thu_tu])
    return np.vstack(ket_qua)


class KNNCosine:
    """kNN với độ đo cosine, dùng hàm tim_lang_gieng ở trên."""

    def __init__(self, k, batch=100):
        self.k = k
        self.batch = batch

    def fit(self, X, y):
        self.XT = X.astype(np.float32).T.tocsr()
        self.classes_, self.y_id = np.unique(np.asarray(y), return_inverse=True)
        return self

    def predict(self, X):
        idx = tim_lang_gieng(X, self.XT, self.k, self.batch)
        votes = self.y_id[idx]
        n_cls = len(self.classes_)
        pred = np.array([np.bincount(r, minlength=n_cls).argmax() for r in votes])
        return self.classes_[pred]


# ---------- Đọc dữ liệu (chỉ đọc 2 cột cần thiết cho nhẹ RAM) ----------
df = pd.read_csv(os.path.join("data", "clean_full.csv"), encoding="utf-8",
                 usecols=["label", "clean"])
df["clean"] = df["clean"].fillna("")
df = df[df["clean"].str.split().str.len() >= 20].reset_index(drop=True)
print("Số bài dùng được:", len(df))

# ---------- Chia train / validation / test ----------
# Tập test giống hệt train.py (test_size=0.2, random_state=42)
X_trval, X_test, y_trval, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
X_tr, X_val, y_tr, y_val = train_test_split(
    X_trval, y_trval, test_size=0.2, random_state=42, stratify=y_trval)
print(f"Train: {len(X_tr)} | Validation: {len(X_val)} | Test: {len(X_test)}")
del df
gc.collect()

classes = np.array(sorted(y_tr.unique()))
lab2id = {c: i for i, c in enumerate(classes)}

# ---------- TF-IDF cho phần chọn tham số ----------
t0 = time.time()
tfidf = TfidfVectorizer(max_features=MAX_FEATURES, min_df=2)
Xtr = tfidf.fit_transform(X_tr)
Xval = tfidf.transform(X_val)
print(f"TF-IDF xong sau {time.time() - t0:.1f} giây | số đặc trưng: {Xtr.shape[1]}")

# ---------- [1] Chọn alpha cho Naive Bayes (prior đều, chọn theo Macro-F1) ----------
print("\n[1] Chọn alpha cho Naive Bayes (tập validation)")
alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
f1_alpha = []
for a in alphas:
    m = MultinomialNB(alpha=a, fit_prior=False).fit(Xtr, y_tr)
    p = m.predict(Xval)
    f1 = f1_score(y_val, p, average="macro")
    f1_alpha.append(f1)
    print(f"  alpha={a}: Macro-F1={f1:.4f} | Accuracy={accuracy_score(y_val, p):.4f}")
best_alpha = alphas[int(np.argmax(f1_alpha))]
print("  => alpha tốt nhất:", best_alpha)

# ---------- [2] Chọn số đặc trưng chi-square ----------
print("\n[2] Chọn số đặc trưng chi-square (Naive Bayes)")
diem, _ = chi2(Xtr, y_tr)
thu_tu = np.argsort(np.nan_to_num(diem))[::-1]      # đặc trưng tốt nhất đứng đầu
ks = [1000, 2000, 5000, 10000, 20000, "all"]
f1_k = []
for k in ks:
    cols = np.arange(Xtr.shape[1]) if k == "all" else thu_tu[:k]
    m = MultinomialNB(alpha=best_alpha, fit_prior=False).fit(Xtr[:, cols], y_tr)
    p = m.predict(Xval[:, cols])
    f1 = f1_score(y_val, p, average="macro")
    f1_k.append(f1)
    print(f"  k={k}: Macro-F1={f1:.4f} | Accuracy={accuracy_score(y_val, p):.4f}")
best_k = ks[int(np.argmax(f1_k))]
print("  => số đặc trưng cho Macro-F1 cao nhất:", best_k)

# Chọn số đặc trưng nhỏ nhất mà Macro-F1 chỉ kém bản tốt nhất không quá 0.005
nguong = max(f1_k) - 0.005
k_chi = next(k for k, f in zip(ks, f1_k) if f >= nguong)
print(f"  => số đặc trưng dùng cho dòng chi-square trong bảng so sánh: {k_chi}")

# ---------- [3] Chọn k cho kNN (cosine) ----------
print("\n[3] Chọn k cho kNN (khoảng cách cosine) - dùng 5000 bài validation cho nhanh")
knn_ks = [1, 3, 5, 7, 11, 15, 21]
n_v = min(5000, Xval.shape[0])
ytr_id = y_tr.map(lab2id).values
yv_id = y_val.map(lab2id).values[:n_v]
t0 = time.time()
XT_tr = Xtr.astype(np.float32).T.tocsr()
idx = tim_lang_gieng(Xval[:n_v], XT_tr, max(knn_ks), batch=100)
print(f"  Tìm láng giềng xong sau {time.time() - t0:.1f} giây")
neigh = ytr_id[idx]
f1_knn = []
for kk in knn_ks:
    pred = np.array([np.bincount(r, minlength=len(classes)).argmax() for r in neigh[:, :kk]])
    f1 = f1_score(yv_id, pred, average="macro")
    f1_knn.append(f1)
    print(f"  k={kk}: Macro-F1={f1:.4f} | Accuracy={(pred == yv_id).mean():.4f}")
best_knn = knn_ks[int(np.argmax(f1_knn))]
print("  => k tốt nhất của kNN:", best_knn)

# ---------- Vẽ biểu đồ ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot([str(a) for a in alphas], f1_alpha, marker="o")
ax[0].set_title("Naive Bayes: ảnh hưởng của alpha")
ax[0].set_xlabel("alpha"); ax[0].set_ylabel("Macro-F1 (validation)")
ax[1].plot([str(k) for k in ks], f1_k, marker="o", color="green")
ax[1].set_title("Naive Bayes: số đặc trưng chi-square")
ax[1].set_xlabel("Số đặc trưng giữ lại")
ax[2].plot([str(k) for k in knn_ks], f1_knn, marker="o", color="red")
ax[2].set_title("kNN (cosine): ảnh hưởng của k")
ax[2].set_xlabel("k")
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join("images", "tune_params.png"), dpi=150)
print("\nĐã lưu images/tune_params.png")

# ---------- Giải phóng bộ nhớ trước phần so sánh trên tập test ----------
del XT_tr, idx, neigh, Xtr, Xval, tfidf, diem, thu_tu
gc.collect()

# ---------- [4] So sánh trên tập test (huấn luyện lại bằng train + validation) ----------
print("\n[4] So sánh trên tập test (chấm một lần duy nhất)")
t0 = time.time()
tfidf_f = TfidfVectorizer(max_features=MAX_FEATURES, min_df=2)
Xf = tfidf_f.fit_transform(X_trval)
Xt = tfidf_f.transform(X_test)
print(f"  TF-IDF (train+validation) xong sau {time.time() - t0:.1f} giây")

diem_f, _ = chi2(Xf, y_trval)
thu_tu_f = np.argsort(np.nan_to_num(diem_f))[::-1]


def chon_dac_trung(k):
    cols = np.arange(Xf.shape[1]) if k == "all" else thu_tu_f[:k]
    return Xf[:, cols], Xt[:, cols]


def danh_gia(ten, clf, Xa, Xb):
    t0 = time.time()
    clf.fit(Xa, y_trval)
    t_fit = time.time() - t0
    t0 = time.time()
    pred = clf.predict(Xb)
    t_pred = time.time() - t0
    print(f"  Xong: {ten}")
    return {
        "Mô hình": ten,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Macro-F1": round(f1_score(y_test, pred, average="macro"), 4),
        "Recall khoa_hoc": round(recall_score(y_test, pred, labels=["khoa_hoc"], average=None)[0], 3),
        "Huấn luyện (giây)": round(t_fit, 2),
        "Dự đoán (giây)": round(t_pred, 2),
    }


k_tree = 2000       # Decision Tree dùng ít đặc trưng và giới hạn độ sâu để chạy nhanh
ket_qua = []
ket_qua.append(danh_gia("Naive Bayes cơ bản (alpha=1.0)", MultinomialNB(alpha=1.0), Xf, Xt))
ket_qua.append(danh_gia(f"Naive Bayes tinh chỉnh (alpha={best_alpha}, prior đều)",
                        MultinomialNB(alpha=best_alpha, fit_prior=False), Xf, Xt))
Xa, Xb = chon_dac_trung(k_chi)
ket_qua.append(danh_gia(f"Naive Bayes tinh chỉnh + chi-square (k={k_chi})",
                        MultinomialNB(alpha=best_alpha, fit_prior=False), Xa, Xb))
ket_qua.append(danh_gia(f"kNN cosine (k={best_knn})", KNNCosine(best_knn), Xf, Xt))
Xa, Xb = chon_dac_trung(k_tree)
ket_qua.append(danh_gia(f"Decision Tree (k={k_tree}, max_depth=40)",
                        DecisionTreeClassifier(max_depth=40, random_state=42), Xa, Xb))

bang = pd.DataFrame(ket_qua)
print()
print(bang.to_string(index=False))
bang.to_csv(os.path.join("images", "ket_qua_so_sanh.csv"), index=False, encoding="utf-8-sig")
print("\nĐã lưu images/ket_qua_so_sanh.csv")

# ---------- [5] Top từ đặc trưng của từng chuyên mục ----------
vocab = np.array(tfidf_f.get_feature_names_out())
with open(os.path.join("images", "top_tu_dac_trung.txt"), "w", encoding="utf-8") as f:
    for c in classes:
        d, _ = chi2(Xf, (y_trval == c).astype(int))
        top = vocab[np.argsort(np.nan_to_num(d))[::-1][:10]]
        dong = f"{c}: " + ", ".join(top)
        print(dong)
        f.write(dong + "\n")
print("\nĐã lưu images/top_tu_dac_trung.txt")
