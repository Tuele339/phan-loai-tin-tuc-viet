import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

os.makedirs("model", exist_ok=True)
os.makedirs("images", exist_ok=True)

# 1. Đọc dữ liệu đã xử lý, bỏ các bài quá ngắn
df = pd.read_csv(os.path.join("data", "clean_full.csv"), encoding="utf-8")
df["clean"] = df["clean"].fillna("")
df = df[df["clean"].str.split().str.len() >= 20].reset_index(drop=True)
print("Số bài dùng được:", len(df))
print(df["label"].value_counts())

# 2. Chia train/test = 80:20, giữ tỉ lệ các chuyên mục
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"])
print("\nTrain:", len(X_train), "| Test:", len(X_test))

# 3. TF-IDF (chỉ fit trên tập train)
tfidf = TfidfVectorizer(max_features=50000, min_df=2)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)
print("Số đặc trưng TF-IDF:", Xtr.shape[1])

# 4. Huấn luyện Naive Bayes
nb = MultinomialNB(alpha=1.0)      # alpha=1 là Laplace smoothing
nb.fit(Xtr, y_train)
pred = nb.predict(Xte)

# 5. Đánh giá
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")
print(f"\nAccuracy : {acc:.4f}")
print(f"Macro-F1 : {f1:.4f}\n")
bao_cao = classification_report(y_test, pred)
print(bao_cao)

with open(os.path.join("images", "bao_cao_nb.txt"), "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\nMacro-F1: {f1:.4f}\n\n{bao_cao}")

# 6. Vẽ và lưu confusion matrix
cm = confusion_matrix(y_test, pred, labels=nb.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=nb.classes_, yticklabels=nb.classes_)
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Confusion matrix - Naive Bayes (toàn bộ dữ liệu)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join("images", "confusion_nb.png"), dpi=150)
print("Đã lưu images/confusion_nb.png")

# 7. Lưu mô hình
joblib.dump((tfidf, nb), os.path.join("model", "nb_model.pkl"))
print("Đã lưu model/nb_model.pkl")