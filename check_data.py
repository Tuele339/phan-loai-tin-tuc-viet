import os
import collections

ROOT = os.path.join("data", "VNTC")

for fn in sorted(os.listdir(ROOT)):
    path = os.path.join(ROOT, fn)
    if not os.path.isfile(path):
        continue
    dem = collections.Counter()
    tong = 0
    sai_dinh_dang = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tong += 1
            if line.startswith("__label__"):
                nhan = line.split(" ", 1)[0].replace("__label__", "")
                dem[nhan] += 1
            else:
                sai_dinh_dang += 1
    print(f"\n=== {fn}: {tong} dòng, {sai_dinh_dang} dòng không có nhãn ===")
    for nhan, so in sorted(dem.items()):
        print(f"  {nhan}: {so}")