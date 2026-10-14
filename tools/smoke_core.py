import sys, traceback, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils import imread
from core.defects import apply_defects, DEFECT_KINDS
from core.batch import DEFAULT_SETTINGS

img = imread("samples/product_dark.png")
print("input:", img.shape, img.dtype)
print("kinds:", DEFECT_KINDS)

fails = []
for kind in DEFECT_KINDS:
    s = {k: dict(v) for k, v in DEFAULT_SETTINGS.items()}
    for k in s:
        s[k]["enable"] = (k == kind)
    try:
        out, applied = apply_defects(img, s, seed=1)
        print(f"{kind:9s} OK  shape={out.shape}  mask_sum={float(applied[0].mask.sum()):.1f}")
    except Exception:
        print(f"{kind:9s} FAIL")
        fails.append(kind)
        traceback.print_exc()

# 四类共存
s = {k: dict(v) for k, v in DEFAULT_SETTINGS.items()}
for k in s:
    s[k]["enable"] = True
out, applied = apply_defects(img, s, seed=2)
print("all-kinds OK", [r.kind for r in applied],
      "mask_sum=%.1f" % sum(float(r.mask.sum()) for r in applied))
if len(applied) != len(DEFECT_KINDS):
    fails.append("coexist")
print("SMOKE CORE", "FAILED: " + ",".join(fails) if fails else "PASSED")
sys.exit(1 if fails else 0)
