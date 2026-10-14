"""端到端校验：自动查找 MLCC 产品 + 缺陷只落在产品表面（改进 5）。

对 5 张 MLCC 样例 × 3 种目标表面 × 4 档明显度 = 60 组，
统计缺陷像素落在产品外的比例（越界率），并核对分割精度（与真值 region_*.png 比对）。
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from core.utils import imread, imwrite  # noqa: E402
from core.mlcc_region import segment_mlcc, region_overlay  # noqa: E402
from core.defects import apply_defects, SEVERITY_LEVELS  # noqa: E402

SMP = "samples_mlcc"
GT_DIR = os.path.join(SMP, "_truth")   # 真值图单独放_truth/，不与原图同目录
PRODUCTS = ["mlcc_h_std.png", "mlcc_h_bright.png", "mlcc_h_warm.png",
            "mlcc_v_std.png", "mlcc_v_dark.png"]
# 真值：regionmap_*.png 中 127=瓷体 255=电极 0=背景（干净标签图）
GTS = {"mlcc_h_std.png": "regionmap_h_std.png",
       "mlcc_h_bright.png": "regionmap_h_bright.png",
       "mlcc_h_warm.png": "regionmap_h_warm.png",
       "mlcc_v_std.png": "regionmap_v_std.png",
       "mlcc_v_dark.png": "regionmap_v_dark.png"}

TARGETS = ["body", "electrode", "both"]
SEVS = ["obvious", "medium", "subtle", "mixed"]

rows = []
fails = []
out_dir = os.path.join("build", "verify_region")
os.makedirs(out_dir, exist_ok=True)

print("=" * 96)
print("A. 分割精度（自动查找 vs 真值）")
print("=" * 96)
print(f"{'样例':<20}{'主轴°':>7}{'方法':>10}{'产品IoU':>10}{'瓷体IoU':>10}"
      f"{'电极IoU':>10}{'覆盖率':>9}")
seg_ok = 0
for p in PRODUCTS:
    rgb = imread(os.path.join(SMP, p))
    gt = imread(os.path.join(GT_DIR, GTS[p]))
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    gt_prod = (gt > 0).astype(np.float32)
    gt_body = np.isclose(gt, 127).astype(np.float32)
    gt_elec = np.isclose(gt, 255).astype(np.float32)

    res = segment_mlcc(rgb, bg_side="auto")

    def iou(a, b):
        a = a > 0.5
        b = b > 0.5
        u = (a | b).sum()
        return float((a & b).sum()) / u if u else 0.0

    i_p = iou(res.product, gt_prod)
    i_b = iou(res.body, gt_body)
    i_e = iou(res.electrode, gt_elec)
    print(f"{p:<20}{res.axis_angle_deg:>7.0f}{res.method:>10}"
          f"{i_p:>10.3f}{i_b:>10.3f}{i_e:>10.3f}{res.coverage:>9.1%}")
    if not res.ok:
        fails.append(f"{p}: 分割失败 {res.notes}")
    elif i_p < 0.90:
        fails.append(f"{p}: 产品 IoU 偏低 {i_p:.3f}")
    else:
        seg_ok += 1
    imwrite(os.path.join(out_dir, f"seg_{p}"), region_overlay(rgb, res))

print(f"\n分割通过：{seg_ok}/{len(PRODUCTS)}\n")

# ---------------------------------------------------------------------------
print("=" * 96)
print("B. 缺陷越界率矩阵（每格 = 缺陷像素落在产品外的百分比）")
print("=" * 96)
ALL_KINDS = ("scratch", "smudge", "discolor", "chip")
for target in TARGETS:
    print(f"\n--- 目标表面 = {target} ---")
    head = f"{'样例':<20}" + "".join(f"{s:>14}" for s in SEVS)
    print(head)
    for p in PRODUCTS:
        rgb = imread(os.path.join(SMP, p))
        res = segment_mlcc(rgb, bg_side="auto")
        if not res.ok:
            print(f"{p:<20}  [分割失败]")
            continue
        region_masks = {"body": res.body, "electrode": res.electrode}
        outside = (res.product < 0.5)
        cells = []
        for sev in SEVS:
            settings = {}
            for kind in ALL_KINDS:
                settings[kind] = {"enable": True, "target": target,
                                  "severity": sev}
            # 用多种子取平均，降低随机性
            rates, pix = [], 0
            for seed in range(6):
                img, applied = apply_defects(rgb, settings, seed=seed,
                                             region_masks=region_masks)
                tot = sum(float((r.mask > 0.05).sum()) for r in applied)
                out = sum(float(((r.mask > 0.05) & outside).sum())
                          for r in applied)
                pix += int(tot)
                rates.append((out / tot * 100) if tot else 0.0)
            mean = float(np.mean(rates))
            worst = float(np.max(rates))
            cells.append((mean, worst, pix))
            if worst > 1.0:
                fails.append(f"{p}/{target}/{sev}: 最差越界 {worst:.2f}%")
            rows.append({"product": p, "target": target, "severity": sev,
                         "leak_mean": round(mean, 4), "leak_worst": round(worst, 4),
                         "defect_px": pix})
        print(f"{p:<20}" + "".join(f"{m:>7.3f}/{w:<5.2f}%" for m, w, _ in cells))

# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("B2. 缺角单独越界矩阵（含最低分项定位）")
print("=" * 96)
print(f"{'样例':<20}{'target':>10}{'severity':>10}{'越界%':>9}{'像素':>10}{'个数':>6}")
for p in PRODUCTS:
    rgb = imread(os.path.join(SMP, p))
    res = segment_mlcc(rgb, bg_side="auto")
    if not res.ok:
        continue
    region_masks = {"body": res.body, "electrode": res.electrode}
    outside = (res.product < 0.5)
    for target in TARGETS:
        for sev in SEVS:
            settings = {"chip": {"enable": True, "target": target, "severity": sev}}
            rates = []
            for seed in range(6):
                _, applied = apply_defects(rgb, settings, seed=seed,
                                           region_masks=region_masks)
                if not applied:
                    continue
                tot = sum(float((r.mask > 0.05).sum()) for r in applied)
                out = sum(float(((r.mask > 0.05) & outside).sum())
                          for r in applied)
                rates.append((out / tot * 100) if tot else 0.0)
            if not rates:
                continue
            worst = float(np.max(rates))
            pix = int(tot)
            if worst > 1.0:
                fails.append(f"chip/{p}/{target}/{sev}: 最差越界 {worst:.2f}%")
            print(f"{p:<20}{target:>10}{sev:>10}{worst:>8.2f}%{pix:>10}{len(applied):>6}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("C. 缺陷占比（覆盖率）与大小差异：明显 vs 不明显")
print("=" * 96)
p = "mlcc_h_std.png"
rgb = imread(os.path.join(SMP, p))
res = segment_mlcc(rgb, bg_side="auto")
region_masks = {"body": res.body, "electrode": res.electrode}
print(f"{'明显度':>8}{'缺陷占比':>12}{'划痕像素':>12}{'脏污像素':>12}{'异色像素':>12}{'缺角像素':>12}")
cov = {}
for sev in SEVS:
    settings = {k: {"enable": True, "target": "both", "severity": sev}
                for k in ALL_KINDS}
    acc = {k: [] for k in ALL_KINDS}
    for seed in range(6):
        _, applied = apply_defects(rgb, settings, seed=seed,
                                   region_masks=region_masks)
        for r in applied:
            acc[r.kind].append(int((r.mask > 0.35).sum()))
    tot = sum(np.mean(v) for v in acc.values())
    cov[sev] = tot / res.product.size
    print(f"{sev:>8}{tot / res.product.size:>11.3%}"
          f"{np.mean(acc['scratch']):>12.0f}{np.mean(acc['smudge']):>12.0f}"
          f"{np.mean(acc['discolor']):>12.0f}{np.mean(acc['chip']):>12.0f}")
if cov["obvious"] <= cov["subtle"] * 1.5:
    fails.append(f"明显度未拉开差距: obvious={cov['obvious']:.4%} "
                 f"subtle={cov['subtle']:.4%}")
else:
    ratio = cov["obvious"] / max(cov["subtle"], 1e-9)
    print(f"\n[OK] 明显/不明显 缺陷量比 = {ratio:.1f}x")

with open(os.path.join(out_dir, "leak_matrix.json"), "w", encoding="utf-8") as f:
    json.dump({"rows": rows, "fails": fails,
               "coverage": {k: v for k, v in cov.items()}},
              f, ensure_ascii=False, indent=2)

print("\n" + "=" * 96)
if fails:
    print(f"发现 {len(fails)} 个问题：")
    for m in fails:
        print("  -", m)
    raise SystemExit(1)
print("全部通过：分割 IoU ≥ 0.90，缺陷越界率 ≤ 1%，明显度可分。")
print("=" * 96)
