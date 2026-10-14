"""缺角尺寸调参：不同 size_ratio / count 的角部破损效果对比（放大拼接）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from core.utils import imread, imwrite  # noqa: E402
from core.mlcc_region import segment_mlcc  # noqa: E402
from core.defects import apply_defects  # noqa: E402

OUT = os.path.join("build", "chip_tune")
os.makedirs(OUT, exist_ok=True)
rgb = imread(os.path.join("samples_mlcc", "mlcc_h_std.png"))
res = segment_mlcc(rgb, bg_side="auto")
rm = {"body": res.body, "electrode": res.electrode}

combos = []
for sr in (0.26, 0.40, 0.55, 0.75):
    for cnt in (1, 2):
        combos.append((sr, cnt))

tiles = []
print(f"{'size_ratio':>11}{'count':>7}{'覆盖%':>9}{'越界%':>8}  file")
for sr, cnt in combos:
    st = {"chip": {"enable": True, "strength": 0.9, "target": "both",
                   "severity": "obvious", "count": cnt, "size_ratio": sr}}
    img, ap = apply_defects(rgb, st, seed=5, region_masks=rm)
    r = ap[0]
    tot = float((r.mask > 0.05).sum())
    out = float(((r.mask > 0.05) & (res.product < 0.5)).sum())
    lk = out / tot * 100 if tot else 0
    cov = float((r.mask > 0.35).mean()) * 100
    fn = f"sr{int(sr*100)}_c{cnt}.png"
    imwrite(os.path.join(OUT, fn), img)
    print(f"{sr:>11.2f}{cnt:>7}{cov:>8.2f}%{lk:>7.2f}%  {fn}")
    # 取左上角与右上角拼一块放大
    im = Image.open(os.path.join(OUT, fn)).convert("RGB")
    a = im.crop((70, 150, 270, 330)).resize((240, 216), Image.LANCZOS)
    b = im.crop((740, 150, 940, 330)).resize((240, 216), Image.LANCZOS)
    row = Image.new("RGB", (240 * 2 + 6, 216), (20, 20, 20))
    row.paste(a, (0, 0))
    row.paste(b, (246, 0))
    tiles.append(row)

cols = 4
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (246 * cols, (216 + 8) * rows), (14, 14, 14))
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    sheet.paste(t, (c * 246, r * (216 + 8)))
sheet.save(os.path.join(OUT, "_sheet.png"))
print("\n对比图:", os.path.join(OUT, "_sheet.png"), sheet.size)