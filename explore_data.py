import os

# Nếu tải bằng cách A, giữ nguyên dòng dưới.
# Nếu tải bằng cách B, bỏ dấu # ở dòng dưới và sửa lại đường dẫn thật:
ROOT = os.path.join("data", "VNTC")

if not os.path.exists(ROOT):
    print("KHÔNG TÌM THẤY THƯ MỤC:", ROOT)
    raise SystemExit

def doc_file(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "utf-16", "cp1258"):
        try:
            text = raw.decode(enc)
            if enc == "utf-8" and "\x00" in text:
                continue
            return enc, " ".join(text.split())
        except UnicodeDecodeError:
            continue
    return "loi", ""

mau = None
print("=== CÁC THƯ MỤC CÓ FILE ===")
for dirpath, dirnames, filenames in os.walk(ROOT):
    if filenames:
        print(f"{dirpath}  ->  {len(filenames)} file")
        if mau is None:
            mau = os.path.join(dirpath, filenames[0])

if mau:
    enc, text = doc_file(mau)
    print("\n=== FILE MẪU ===")
    print("Đường dẫn:", mau)
    print("Mã hóa đọc được:", enc)
    print("Nội dung 300 ký tự đầu:", text[:300])