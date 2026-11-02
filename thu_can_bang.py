import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, f1_score, recall_score

os.makedirs("images", exist_ok=True)

# ---------- Đọc dữ liệu, chia giống compare.py ----------
df = pd.read_csv(os.path.join("data", "clean_full.csv"), encoding="utf-8")
df["clean"] = df["clean"].fillna("")
df = df[df["clean"].str.split().str.len() >= 20].reset_index(drop=True)

X_trval, X_test, y_trval, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
X_tr, X_val, y_tr, y_val = train_test_split(
    X_trval, y_trval, test_size=0.2, random_state=42, stratify=y_trval)
print(f"Train: {len(X_tr)} | Validation: {len(X_val)}")

tfidf = TfidfVectorizer(max_features=50000, min_df=2)
Xtr = tfidf.fit_transform(X_tr)
Xval = tfidf.transform(X_val)
print("TF-IDF xong, số đặc trưng:", Xtr.shape[1])

# ---------- Các biến thể Naive Bayes ----------
mo_hinh = {}
for a in [0.1, 0.5, 1.0]:
    mo_hinh[f"MultinomialNB a={a} (prior theo dữ liệu)"] = MultinomialNB(alpha=a)
    mo_hinh[f"MultinomialNB a={a} (prior đều)"] = MultinomialNB(alpha=a, fit_prior=False)
    mo_hinh[f"ComplementNB a={a}"] = ComplementNB(alpha=a)

classes = sorted(y_tr.unique())
rows = []
for ten, clf in mo_hinh.items():
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xval)
    rec = recall_score(y_val, pred, labels=classes, average=None)
    rows.append({
        "Mô hình": ten,
        "Accuracy": round(accuracy_score(y_val, pred), 4),
        "Macro-F1": round(f1_score(y_val, pred, average="macro"), 4),
        "Recall khoa_hoc": round(rec[classes.index("khoa_hoc")], 3),
        "Recall doi_song": round(rec[classes.index("doi_song")], 3),
    })

bang = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\nKết quả trên tập validation:\n")
print(bang.to_string(index=False))
bang.to_csv(os.path.join("images", "thu_can_bang.csv"), index=False, encoding="utf-8-sig")
print("\nĐã lưu images/thu_can_bang.csv")