# -*- coding: utf-8 -*-
"""第六轮细化可视化验证：黑点(颜色/形状)、凹坑(凹)、鼓包(凸)、崩缺(软边/同色)。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import math

from core.utils import imread, imwrite
from core.defects import apply_defects
from core.mlcc_region import segment_mlcc

OUT = "build"
os.makedirs(OUT, exist_ok=True)

img = imread("samples/product_dark.png")
seg = segment_mlcc(img)
rm = {"body": seg.body, "electrode": seg.electrode}


def run(kind_cfgs, seed, tag):
    s = {k: dict(v) for k, v in __import__("core.batch", fromlist=["DEFAULT_SETTINGS"]).DEFAULT_SETTINGS.items()}
    for k in s:
        s[k]["enable"] = k in kind_cfgs
        s[k].update(kind_cfgs.get(k, {}))
    out, applied = apply_defects(img, s, seed=seed, region_masks=rm)
    print(tag, "->", [(r.kind, r.count) for r in applied])
    return out, applied


def zoom_save(img_, applied, kind, path, scale=3):
    """找该类缺陷掩膜质心，裁一块局部放大图并拼接原图。"""
    m = None
    for r in applied:
        if r.kind == kind:
            m = r.mask
    if m is None or m.max() <= 0.01:
        print(f"  !! {kind} 未生成，跳过 {path}")
        return
    ys, xs = np.nonzero(m > 0.3)
    i = int(np.argmax(m[ys, xs]))
    cy, cx = int(ys[i]), int(xs[i])
    half = 60
    y0, y1 = max(0, cy - half), min(img.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(img.shape[1], cx + half)
    a = img[y0:y1, x0:x1]
    b = img_[y0:y1, x0:x1]
    a = cv2.resize(a, (a.shape[1]*scale, a.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, (b.shape[1]*scale, b.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    sep = np.full((a.shape[0], 4, 3), 255, np.uint8)
    imwrite(path, np.hstack([a, sep, b]))
    print("  saved", path)


# 1) 黑点：默认(自动深黑) / 自定义红色 / 高柔和度
out1, ap1 = run({"dot": {"enable": True, "count": 3, "size_ratio": 0.02}}, 7, "dot默认")
zoom_save(out1, ap1, "dot", os.path.join(OUT, "v6_dot_auto.png"))
out2, ap2 = run({"dot": {"enable": True, "count": 3, "size_ratio": 0.02,
                         "color": "custom", "color_rgb": [10, 10, 10],
                         "edge_soft": 0.9}}, 7, "dot自定义纯黑+高柔和")
zoom_save(out2, ap2, "dot", os.path.join(OUT, "v6_dot_custom.png"))

# 2) 凹坑：应读作「凹」（背光侧亮弧、迎光侧暗）
out3, ap3 = run({"pit": {"enable": True, "count": 3, "size_ratio": 0.02,
                         "depth": 0.8, "highlight": 0.7}}, 11, "pit凹")
zoom_save(out3, ap3, "pit", os.path.join(OUT, "v6_pit_concave.png"))

# 3) 鼓包：应读作「凸」，颜色与邻近区域一致
out4, ap4 = run({"bump": {"enable": True, "count": 3, "size_ratio": 0.02,
                          "height": 0.8, "gloss": 0.6}}, 13, "bump凸")
zoom_save(out4, ap4, "bump", os.path.join(OUT, "v6_bump.png"))

# 4) 崩缺：默认(柔和+同色) 与 柔和度拉满
out5, ap5 = run({"chip": {"enable": True, "count": 1, "size_ratio": 0.35}}, 17, "chip默认")
zoom_save(out5, ap5, "chip", os.path.join(OUT, "v6_chip_default.png"))
out6, ap6 = run({"chip": {"enable": True, "count": 1, "size_ratio": 0.35,
                          "edge_soft": 0.9, "color_match": 1.0}}, 17, "chip最柔")
zoom_save(out6, ap6, "chip", os.path.join(OUT, "v6_chip_soft.png"))

# 定量检查：凹坑 vs 鼓包 明暗极性
def arc_polarity(img_, mask, seg_mask):
    """沿光照方向划分半圆，比较光源侧与对侧的平均亮度变化。

    light_deg=135 → 光向量 (cos135°, sin135°) = (-0.707, +0.707)（y 向下）。
    凹坑：光源侧内壁背光变暗（Δ_lit < Δ_opp）；鼓包：光源侧被照亮（Δ_lit > Δ_opp）。
    """
    ys, xs = np.nonzero(mask > 0.4)
    if xs.size < 20:
        return None
    cy, cx = ys.mean(), xs.mean()
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 2
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = (d < r * 0.85) & (seg_mask > 0.5)
    th = math.radians(135.0)
    lx, ly = math.cos(th), math.sin(th)
    proj = (xx - cx) * lx + (yy - cy) * ly
    lit = inside & (proj > 0)      # 光源所在半边
    opp = inside & (proj < 0)      # 对侧
    orig = img.astype(np.float32).mean(axis=2)
    new = img_.astype(np.float32).mean(axis=2)
    return (new - orig)[lit].mean(), (new - orig)[opp].mean()


for name, out_, ap_ in (("pit", out3, ap3), ("bump", out4, ap4)):
    # 极性测量用「单实例」渲染，避免多实例互相污染半圆统计
    so, apo = run({name: {"enable": True, "count": 1, "size_ratio": 0.025,
                          "depth" if name == "pit" else "height": 0.8,
                          "highlight" if name == "pit" else "gloss": 0.7}},
                  101, f"{name} 单实例(极性测量)")
    m = next((r.mask for r in apo if r.kind == name), None)
    pol = arc_polarity(so, m, seg.body) if m is not None else None
    if pol:
        dlit, dopp = pol
        if name == "pit":
            ok = dlit < dopp
            rel = "光源侧更暗(凹) ✓" if ok else "光源侧更亮(凸) ✗"
        else:
            ok = dlit > dopp
            rel = "光源侧更亮(凸) ✓" if ok else "光源侧更暗(凹) ✗"
        print(f"  {name} 极性: 光源侧Δ={dlit:+.1f} 对侧Δ={dopp:+.1f} → {rel}")
        if not ok:
            print("  !! 极性不符")
    else:
        print(f"  !! {name} 单实例未生成，无法测极性")

print("DONE")
