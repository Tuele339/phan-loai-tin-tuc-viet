import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess

ROOT = "external_test"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)

if not os.path.isdir(ROOT):
    print("Chưa có thư mục external_test.")
    raise SystemExit

b = joblib.load(os.path.join("model", "demo_model.pkl"))
tfidf, nb = b["tfidf"], b["nb"]
classes = list(nb.classes_)

rows = []
for label in sorted(os.listdir(ROOT)):
    folder = os.path.join(ROOT, label)
    if not os.path.isdir(folder):
        continue
    if label not in classes:
        print(f"Bỏ qua thư mục '{label}' vì tên không trùng nhãn của mô hình")
        continue
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".txt"):
            continue
        with open(os.path.join(folder, fn), encoding="utf-8-sig", errors="ignore") as f:
            text = f.read()
        v = tfidf.transform([preprocess(text)])
        if v.nnz == 0:
            print("Không đọc được nội dung:", label, fn)
            continue
        p = nb.predict_proba(v)[0]
        o = np.argsort(-p)
        thu_tu = [classes[i] for i in o]
        rows.append({
            "Bài": f"{label}/{fn}",
            "Thực tế": label,
            "Dự đoán": thu_tu[0],
            "Xác suất": round(float(p[o[0]]), 3),
            "Hạng 2": thu_tu[1],
            "Hạng 3": thu_tu[2],
            "Hạng của nhãn thật": thu_tu.index(label) + 1,
        })

if not rows:
    print("Không có bài nào để chấm.")
    raise SystemExit

df = pd.DataFrame(rows)
df["Đúng"] = df["Hạng của nhãn thật"] == 1
n = len(df)

print("\n=== KẾT QUẢ TỪNG BÀI ===")
print(df.to_string(index=False))

print(f"\n=== TỔNG KẾT ({n} bài) ===")
for k in (1, 2, 3):
    so = int((df["Hạng của nhãn thật"] <= k).sum())
    print(f"  Top-{k} accuracy: {so}/{n} = {so / n:.1%}")

dung = df[df["Đúng"]]["Xác suất"]
sai = df[~df["Đúng"]]["Xác suất"]
print(f"  Xác suất trung bình khi đoán đúng: {dung.mean():.3f}" if len(dung) else "  Không có bài đúng")
print(f"  Xác suất trung bình khi đoán sai : {sai.mean():.3f}" if len(sai) else "  Không có bài sai")
print(f"  Số bài sai nhưng xác suất > 0.9  : {int((sai > 0.9).sum())}/{len(sai)}")

print("\n=== THEO TỪNG CHUYÊN MỤC ===")
tk = df.groupby("Thực tế").agg(
    Tong=("Đúng", "count"),
    Top1=("Đúng", "sum"),
    Top2=("Hạng của nhãn thật", lambda s: int((s <= 2).sum())),
)
print(tk.to_string())

print("\n=== CÁC BÀI ĐOÁN SAI (để cả nhóm xem lại) ===")
print(df[~df["Đúng"]][["Bài", "Thực tế", "Dự đoán", "Xác suất", "Hạng 2", "Hạng của nhãn thật"]].to_string(index=False))

# Ma trận nhầm lẫn của các bài thử
os.makedirs("images", exist_ok=True)
ct = pd.crosstab(df["Thực tế"], df["Dự đoán"]).reindex(index=classes, columns=classes, fill_value=0)
plt.figure(figsize=(10, 8))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title(f"Ma trận nhầm lẫn trên {n} bài báo thật")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join("images", "confusion_ngoai_mien.png"), dpi=150)

df.to_csv(os.path.join("images", "ket_qua_ngoai_mien.csv"), index=False, encoding="utf-8-sig")
print("\nĐã lưu images/ket_qua_ngoai_mien.csv và images/confusion_ngoai_mien.png")