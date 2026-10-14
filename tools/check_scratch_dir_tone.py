"""划痕方向与颜色观感核查：旋转 ROI → 方向跟随；颜色白↔黑、深浅可调。

产出 ``build/scratch_dir_tone.png``：
- 第一行：同一个产品在 0° / 30° / 60° / 90° 的 ROI 下，划痕方向随 ROI 走
- 第二行：颜色 -1(白) / 0(随机) / +1(黑) × 深浅 0.2 / 1.5
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.defects import ScratchDefect, resolve_target  # noqa: E402
from core.mlcc_region import segment_mlcc  # noqa: E402
from core.utils import Rng, imread, imwrite  # noqa: E402

OUT = os.path.join(HERE, "build")
FAIL = []


def check(cond, msg):
    print(f"[{'OK ' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAIL.append(msg)


def roi_mask(shape, cx, cy, rw, rh, deg):
    """生成旋转矩形掩膜（与 GUI 的 ROI 数据结构一致）。"""
    h, w = shape
    ang = math.radians(deg)
    pts = cv2.boxPoints(((float(cx), float(cy)), (float(rw), float(rh)), float(deg)))
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, pts.astype(np.int32), 255)
    return m.astype(np.float32) / 255.0


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (30, 34, 40), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (235, 240, 245), 1, cv2.LINE_AA)
    return out


def measure(mask):
    """逐连通域统计主轴相对目标角度的偏差。"""
    m = (mask > 0.05).astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    devs = []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 120:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float64)
        pts -= pts.mean(axis=0)
        ev, evec = np.linalg.eigh(np.cov(pts.T))
        v = evec[:, int(np.argmax(ev))]
        a = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180
        devs.append(min(abs(a - TARGET), 180 - abs(a - TARGET)))
    return devs


TARGET = 0.0


def main() -> int:
    global TARGET
    img = imread(os.path.join(HERE, "samples_mlcc", "mlcc_h_std.png"))
    seg = segment_mlcc(img)
    prod = resolve_target("both", {"body": seg.body, "electrode": seg.electrode})
    h, w = img.shape[:2]
    print(f"样例 {img.shape}")

    # ---------- 1. 方向跟随 ROI ----------
    print("\n[1] 划痕方向跟随 ROI 角度")
    tiles = [label(cv2.resize(img, (360, int(h * 360 / w))), "original")]
    cx, cy = w * 0.5, h * 0.5
    for deg in (0, 30, 60, 90):
        TARGET = float(deg)
        rm = roi_mask((h, w), cx, cy, w * 0.62, h * 0.34, deg)
        gen = ScratchDefect(strength=0.75, severity_mult=1.0, count=4,
                            curve=0.55, length_ratio=0.55, width_px=2.6,
                            tone=1.0, tone_depth=0.9)
        r = gen.render(img, Rng(7), region=prod, hint=(math.radians(deg), cx, cy))
        devs = measure(r.mask)
        m0 = float(np.mean(devs)) if devs else 999.0
        check(m0 <= 14.0, f"ROI {deg}°：逐条主轴平均偏差 {m0:.1f}°（n={len(devs)}）")
        # 叠一张 ROI 轮廓便于观察
        vis = r.image.copy()
        edge = (rm > 0.5).astype(np.uint8) - (cv2.erode((rm > 0.5).astype(np.uint8),
                                                    np.ones((5, 5), np.uint8)))
        vis[edge > 0] = (255, 210, 60)
        tiles.append(label(cv2.resize(vis, (360, int(h * 360 / w))),
                           f"ROI {deg}deg dev={m0:.1f}"))
    row1 = np.concatenate(tiles, axis=1)

    # ---------- 2. 颜色白 ↔ 黑 + 深浅 ----------
    print("\n[2] 颜色（白↔黑）与深浅")
    tiles2 = []
    for tone in (-1.0, 0.0, 1.0):
        for depth in (0.25, 1.4):
            gen = ScratchDefect(strength=0.75, severity_mult=1.0, count=4,
                                curve=0.5, length_ratio=0.5, width_px=3.0,
                                tone=tone, tone_depth=depth)
            acc = []
            vis = None
            for sd in range(3):
                r = gen.render(img, Rng(sd), region=prod)
                g0 = img.astype(np.float32).mean(axis=2)
                g1 = r.image.astype(np.float32).mean(axis=2)
                sel = r.mask > 0.5
                acc.append(float((g1 - g0)[sel].mean()) if sel.any() else 0.0)
                vis = r.image
            d = float(np.mean(acc))
            name = {(-1.0, 0.25): "tone=-1 white shallow", (-1.0, 1.4): "tone=-1 white deep",
                    (0.0, 0.25): "tone=0 random shallow", (0.0, 1.4): "tone=0 random deep",
                    (1.0, 0.25): "tone=+1 black shallow", (1.0, 1.4): "tone=+1 black deep"}[(tone, depth)]
            print(f"    {name:<24} 平均亮度变化 {d:+6.1f}")
            tiles2.append(label(cv2.resize(vis, (360, int(h * 360 / w))),
                              f"tone={tone:+.0f} depth={depth} d={d:+.0f}"))
    row2 = np.concatenate(tiles2[:3], axis=1)
    row3 = np.concatenate(tiles2[3:], axis=1)

    def pad_to(row, width):
        if row.shape[1] >= width:
            return row
        return np.hstack([row, np.zeros((row.shape[0], width - row.shape[1], 3),
                                        np.uint8)])

    W = max(r.shape[1] for r in (row1, row2, row3))
    sheet = np.concatenate([pad_to(r, W) for r in (row1, row2, row3)], axis=0)
    p = os.path.join(OUT, "scratch_dir_tone.png")
    imwrite(p, sheet)
    print(f"\n对比图：{p}  {sheet.shape}")
    if FAIL:
        print("\n==== 失败项 ====")
        for m in FAIL:
            print(" -", m)
        return 1
    print("\nALL SCRATCH DIR/TONE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
