import os
import re
from pyvi import ViTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "stopwords_vi.txt"), encoding="utf-8") as f:
    STOPWORDS = set(w.strip() for w in f if w.strip())


def read_text(path):
    """Đọc 1 file bài báo, tự thử các kiểu mã hóa phổ biến."""
    raw = open(path, "rb").read()
    text = None
    for enc in ("utf-8", "utf-16", "cp1258"):
        try:
            text = raw.decode(enc)
            if enc == "utf-8" and "\x00" in text:
                text = None
                continue
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")
    return " ".join(text.split())


def preprocess(text, max_chars=3000):
    """Chữ thường -> bỏ số, dấu câu -> tách từ -> bỏ stopword."""
    text = text[:max_chars].lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = ViTokenizer.tokenize(text)          # "sinh viên" -> "sinh_viên"
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


if __name__ == "__main__":
    mau = "Đội tuyển Việt Nam giành chiến thắng 3-0 trong trận đấu tối qua, sinh viên cổ vũ rất nhiệt tình."
    print("Trước:", mau)
    print("Sau  :", preprocess(mau))