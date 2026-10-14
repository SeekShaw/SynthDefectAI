"""缺角缺陷生成测试：不同强度/明显度/表面的效果与落点约束。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from core.utils import imread, imwrite, to_u8  # noqa: E402
from core.mlcc_region import segment_mlcc  # noqa: E402
from core.defects import apply_defects, ChipDefect  # noqa: E402

OUT = os.path.join("build", "chip_test")
os.makedirs(OUT, exist_ok=True)
SMP = os.path.join("samples_mlcc", "mlcc_h_std.png")

rgb = imread(SMP)
res = segment_mlcc(rgb, bg_side="auto")
rm = {"body": res.body, "electrode": res.electrode}
print(f"分割 ok={res.ok} 覆盖率={res.coverage:.1%}")

# 1) 参数扫描：强度 × 明显度
print("\n" + "=" * 78)
print("缺角：强度 × 明显度")
print("=" * 78)
print(f"{'强度':>6}{'明显度':>10}{'个数':>6}{'掩膜占比':>11}{'越界%':>9}{'种子':>6}")
rows = []
for s in (0.35, 0.7, 1.0):
    for sev in ("obvious", "medium", "subtle"):
        leaks, covers = [], []
        for seed in range(4):
            st = {"chip": {"enable": True, "strength": s, "target": "both",
                           "severity": sev, "count": 2, "size_ratio": 0.16}}
            img, ap = apply_defects(rgb, st, seed=seed, region_masks=rm)
            r = ap[0]
            tot = float((r.mask > 0.05).sum())
            out = float(((r.mask > 0.05) & (res.product < 0.5)).sum())
            leaks.append(out / tot * 100 if tot else 0.0)
            covers.append(float((r.mask > 0.35).mean()) * 100)
            if seed == 0:
                imwrite(os.path.join(OUT, f"chip_s{int(s*100)}_{sev}.png"), img)
        print(f"{s:>6.2f}{sev:>10}{r.count:>6}"
              f"{np.mean(covers):>10.2f}%{max(leaks):>8.2f}%{seed:>6}")
        rows.append((s, sev, r.count, np.mean(covers), max(leaks)))
        assert max(leaks) <= 1.0, f"缺角越界 {max(leaks):.2f}% > 1%"

# 2) 目标表面
print("\n" + "=" * 78)
print("目标表面约束")
print("=" * 78)
for target, key in (("body", "body"), ("electrode", "electrode")):
    st = {"chip": {"enable": True, "strength": 0.85, "target": target,
                   "severity": "obvious", "count": 2}}
    img, ap = apply_defects(rgb, st, seed=11, region_masks=rm)
    r = ap[0]
    # 目标掩膜是羽化软掩膜，用 >0.05 判定与掩膜本身一致
    inside_tgt = float(((r.mask > 0.05) & (rm[key] > 0.05)).sum())
    tot = float((r.mask > 0.05).sum())
    rate = inside_tgt / tot * 100 if tot else 0.0
    print(f"  target={target:<10} 个数={r.count} 落在目标表面内={rate:.2f}%")
    assert rate >= 99.5, f"缺角未落在 {target} 内（{rate:.1f}%）"
    imwrite(os.path.join(OUT, f"chip_{target}.png"), img)

# 3) 换其他样例（含竖放）
print("\n其他样例：")
for name in ("mlcc_v_std.png", "mlcc_v_dark.png", "mlcc_h_bright.png"):
    r0 = imread(os.path.join("samples_mlcc", name))
    s0 = segment_mlcc(r0, bg_side="auto")
    rm0 = {"body": s0.body, "electrode": s0.electrode}
    st = {"chip": {"enable": True, "strength": 0.8, "target": "both",
                   "severity": "obvious", "count": 2}}
    img, ap = apply_defects(r0, st, seed=3, region_masks=rm0)
    r = ap[0]
    tot = float((r.mask > 0.05).sum())
    out = float(((r.mask > 0.05) & (s0.product < 0.5)).sum())
    lk = out / tot * 100 if tot else 0.0
    print(f"  {name:<20} 个数={r.count} 占比={float((r.mask>0.35).mean())*100:.2f}% 越界={lk:.2f}%")
    imwrite(os.path.join(OUT, f"chip_{name}"), img)
    assert lk <= 1.0, f"{name} 缺角越界 {lk:.2f}%"

# 4) 与其它缺陷共存
print("\n与其它缺陷共存：")
st = {
    "scratch": {"enable": True, "strength": 0.6, "target": "both", "severity": "obvious"},
    "smudge": {"enable": True, "strength": 0.5, "target": "body", "severity": "medium"},
    "discolor": {"enable": True, "strength": 0.5, "target": "body", "severity": "mixed"},
    "chip": {"enable": True, "strength": 0.8, "target": "both", "severity": "obvious",
             "count": 2},
}
img, ap = apply_defects(rgb, st, seed=77, region_masks=rm)
print("  叠加顺序:", [r.kind for r in ap])
tot = sum(float((r.mask > 0.05).sum()) for r in ap)
out = sum(float(((r.mask > 0.05) & (res.product < 0.5)).sum()) for r in ap)
print(f"  总缺陷占比={float((np.maximum.reduce([r.mask for r in ap])>0.35).mean())*100:.2f}% "
      f"总越界={out/tot*100:.2f}%")
imwrite(os.path.join(OUT, "chip_combo.png"), img)
assert out / tot * 100 <= 1.0

print("\n输出目录:", OUT)
print("ALL CHIP TESTS PASSED")