import os
import joblib
import numpy as np
import pandas as pd
from preprocess import preprocess

ROOT = "external_test"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)

if not os.path.isdir(ROOT):
    print("Chưa có thư mục external_test. Hãy tạo theo hướng dẫn.")
    raise SystemExit

b = joblib.load(os.path.join("model", "demo_model.pkl"))
tfidf, nb = b["tfidf"], b["nb"]
hop_le = set(nb.classes_)

rows = []
for label in sorted(os.listdir(ROOT)):
    folder = os.path.join(ROOT, label)
    if not os.path.isdir(folder):
        continue
    if label not in hop_le:
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
        rows.append({
            "Bài": f"{label}/{fn}",
            "Thực tế": label,
            "Dự đoán": nb.classes_[o[0]],
            "Xác suất": round(float(p[o[0]]), 3),
            "Hạng 2": nb.classes_[o[1]],
            "Đúng": bool(nb.classes_[o[0]] == label),
        })

if not rows:
    print("Không có bài nào để chấm.")
    raise SystemExit

df = pd.DataFrame(rows)
print("\n=== KẾT QUẢ TỪNG BÀI ===")
print(df.to_string(index=False))

print(f"\n=== TỔNG KẾT: đúng {df['Đúng'].sum()}/{len(df)} bài ({df['Đúng'].mean():.1%}) ===")

print("\n=== THEO TỪNG CHUYÊN MỤC ===")
tk = df.groupby("Thực tế")["Đúng"].agg(Đúng="sum", Tổng="count")
print(tk.to_string())

sai = df[~df["Đúng"]]
if len(sai):
    print("\n=== CÁC CẶP NHẦM LẪN (thực tế -> dự đoán) ===")
    print(sai.groupby(["Thực tế", "Dự đoán"]).size().to_string())

os.makedirs("images", exist_ok=True)
df.to_csv(os.path.join("images", "ket_qua_ngoai_mien.csv"), index=False, encoding="utf-8-sig")
print("\nĐã lưu images/ket_qua_ngoai_mien.csv")