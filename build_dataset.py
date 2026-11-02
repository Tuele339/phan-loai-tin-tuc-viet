import os
import time
import multiprocessing as mp
import pandas as pd
from preprocess import preprocess

# ====== CẤU HÌNH ======
ROOT = os.path.join("data", "VNTC")
OUT_FILE = os.path.join("data", "clean_full.csv")
MAX_CHARS = 3000        # chỉ lấy 3000 ký tự đầu mỗi bài cho nhanh
BATCH = 4000            # xử lý và lưu theo từng đợt 4000 bài
N_PROC = max(1, min(6, (os.cpu_count() or 2) - 1))   # số luồng chạy song song
# ======================


def doc_du_lieu():
    nhan_list, text_list = [], []
    for fn in sorted(os.listdir(ROOT)):
        if not fn.lower().endswith(".txt"):
            continue
        print("Đang đọc", fn, "...")
        with open(os.path.join(ROOT, fn), encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("__label__"):
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                nhan_list.append(parts[0].replace("__label__", ""))
                text_list.append(parts[1][:MAX_CHARS])
    return nhan_list, text_list


if __name__ == "__main__":
    mp.freeze_support()
    nhan_list, text_list = doc_du_lieu()
    tong = len(text_list)
    print("Tổng số bài đọc được:", tong)
    print("Số luồng chạy song song:", N_PROC)

    os.makedirs("data", exist_ok=True)

    # Nếu đã chạy dở trước đó thì chạy tiếp từ chỗ dừng
    da_xong = 0
    if os.path.exists(OUT_FILE):
        da_xong = len(pd.read_csv(OUT_FILE, usecols=["idx"], encoding="utf-8"))
        print(f"Tìm thấy file cũ: đã xử lý {da_xong} bài -> chạy tiếp từ đó")

    t0 = time.time()
    with mp.Pool(N_PROC) as pool:
        for bat_dau in range(da_xong, tong, BATCH):
            ket_thuc = min(bat_dau + BATCH, tong)
            texts = text_list[bat_dau:ket_thuc]
            cleans = pool.map(preprocess, texts, chunksize=50)

            rows = pd.DataFrame({
                "idx": range(bat_dau, ket_thuc),
                "label": nhan_list[bat_dau:ket_thuc],
                "snippet": [t[:300] for t in texts],   # đoạn đầu để hiển thị khi demo
                "clean": cleans,
            })
            rows.to_csv(OUT_FILE, mode="a", index=False, encoding="utf-8",
                        header=not os.path.exists(OUT_FILE))

            so_bai = ket_thuc - da_xong
            toc_do = so_bai / (time.time() - t0)
            con_lai = (tong - ket_thuc) / toc_do / 60
            print(f"{ket_thuc}/{tong} bài | {toc_do:.0f} bài/giây | còn khoảng {con_lai:.1f} phút")

    df = pd.read_csv(OUT_FILE, encoding="utf-8")
    df["clean"] = df["clean"].fillna("")
    hop_le = df[df["clean"].str.split().str.len() >= 20]
    print("\n=== XONG ===")
    print("Tổng số dòng:", len(df), "| Số bài đủ dài (dùng được):", len(hop_le))
    print(hop_le["label"].value_counts())