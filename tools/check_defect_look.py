"""缺陷观感专项核查：数量 / 划痕路径 / 边缘柔和度。

产出 ``build/defect_look.png``（划痕三种弯曲度 + 脏污/异色边缘放大），
并打印量化指标：
- 数量参数是否精确生效（设 1 → 实例数 1）
- 划痕曲折度 fill = 掩膜面积 / (外接框对角线 × 线宽)，直线≈1.0，越大越弯
- 边缘过渡带占比 = 0.02<mask<0.98 的像素 / 掩膜面积，越大边缘越糊
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.batch import DEFAULT_SETTINGS  # noqa: E402
from core.defects import apply_defects  # noqa: E402
from core.mlcc_region import segment_mlcc  # noqa: E402
from core.utils import imread, imwrite  # noqa: E402

OUT = os.path.join(HERE, "build")


def one(kind: str, img: np.ndarray, rm: dict, **over) -> tuple:
    cfg = {"enable": True, "target": "both", "severity": "obvious",
           "strength": DEFAULT_SETTINGS[kind]["strength"]}
    cfg.update(DEFAULT_SETTINGS[kind])
    cfg.update(over)
    cfg["enable"] = True
    out, applied = apply_defects(img, {kind: cfg}, seed=7, region_masks=rm)
    res = applied[-1]
    return out, res


def fill_ratio(mask: np.ndarray) -> float:
    """掩膜面积 / (外接框对角线 × 名义线宽)：直线≈1，越大越弯。

    名义线宽 = 面积 / 覆盖行数（对直线与曲线都成立）。
    """
    ys, xs = np.nonzero(mask > 0.05)
    if len(xs) == 0:
        return 0.0
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    diag = float(np.hypot(bw, bh))
    area = float(len(xs))
    wpx = max(1.0, area / max(len(np.unique(ys)), 1))
    return area / max(diag * wpx, 1.0)


def path_curvature(curve: float, trials: int = 24, L: float = 400.0) -> tuple:
    """直接量测生成路径的「实际长度 / 首尾直线距离」，越大越蜿蜒。"""
    from core.defects import ScratchDefect, ScratchParams
    from core.utils import Rng

    gen = ScratchDefect(strength=0.6, severity_mult=1.0)
    p = ScratchParams(curve=curve)
    rng = Rng(1234)
    ratios = []
    for i in range(trials):
        ang = (i % 8) * (np.pi / 8)
        p0 = (500.0, 260.0)
        pts = gen._draw_path(p0, ang, L, rng, curve, n=180)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        direct = float(np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))
        ratios.append(seg / max(direct, 1e-6))
    arr = np.asarray(ratios)
    return float(arr.mean()), float(arr.max())


def soft_ratio(mask: np.ndarray) -> float:
    """过渡带像素占比：0.02<mask<0.98 的像素 / (mask>0.02 的像素)。"""
    m = mask.astype(np.float32)
    band = float(((m > 0.02) & (m < 0.98)).sum())
    solid = float((m > 0.02).sum())
    return band / max(solid, 1.0)


def grad_at_edge(mask: np.ndarray) -> float:
    """掩膜边界处的平均梯度（越小越柔）。"""
    m = clip = np.clip(mask.astype(np.float32), 0, 1)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    band = (m > 0.02) & (m < 0.98)
    if band.sum() < 10:
        return 0.0
    return float(g[band].mean())


def crop_around(img: np.ndarray, mask: np.ndarray, size: int = 320,
                pad: int = 0) -> np.ndarray:
    ys, xs = np.nonzero(mask > 0.05)
    if len(xs) == 0:
        return np.zeros((size, size, 3), np.uint8)
    cx, cy = int(xs.mean()), int(ys.mean())
    half = size // 2 + pad
    x0 = int(np.clip(cx - half, 0, img.shape[1] - size))
    y0 = int(np.clip(cy - half, 0, img.shape[0] - size))
    sub = img[y0:y0 + size, x0:x0 + size]
    if sub.shape[0] != size or sub.shape[1] != size:
        out = np.zeros((size, size, 3), np.uint8)
        out[:sub.shape[0], :sub.shape[1]] = sub
        return out
    return sub


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (30, 34, 40), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (230, 235, 240), 1, cv2.LINE_AA)
    return out


def stack(rows, gap=6):
    w = max(r.shape[1] for r in rows)
    h = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    out = np.full((h, w, 3), 24, np.uint8)
    y = 0
    for r in rows:
        out[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + gap
    return out


def main() -> int:
    img = imread(os.path.join(HERE, "samples_mlcc", "mlcc_h_std.png"))
    seg = segment_mlcc(img)
    rm = {"body": seg.body, "electrode": seg.electrode}
    print(f"样例 {img.shape}  分割 ok={seg.ok}")

    # ---------- 1) 数量参数精确性 ----------
    print("\n[1] 数量参数（设 1 应只出 1 个实例）")
    for kind, key in (("scratch", "count"), ("smudge", "blob_count"),
                      ("discolor", "region_count"), ("chip", "count")):
        _, res = one(kind, img, rm, **{key: 1}, severity="obvious")
        ok = res.count == 1
        print(f"    {kind:<9} {key}={DEFAULT_SETTINGS[kind][key]} → 实例 {res.count} "
              f"{'OK' if ok else 'FAIL'}")

    # ---------- 2) 划痕弯曲度 ----------
    print("\n[2] 划痕路径（弯曲度 0 / 0.55 / 1.0）")
    base = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
    tiles = [label(base, "original")]
    for cv_ in (0.0, 0.55, 1.0):
        out, res = one("scratch", img, rm, count=1, curve=cv_, strength=0.75,
                       width_px=2.6, length_ratio=0.85)
        fr = fill_ratio(res.mask)
        mean_r, max_r = path_curvature(cv_)
        print(f"    curve={cv_:<4} 路径长度/首尾距离 平均={mean_r:.2f} 最大={max_r:.2f}"
              f"   掩膜 fill={fr:.2f}")
        tiles.append(label(crop_around(out, res.mask, 320),
                           f"curve={cv_} path={mean_r:.2f}"))
    row_scratch = np.concatenate(tiles, axis=1)
    row_scratch = cv2.resize(row_scratch, (row_scratch.shape[1] // 2,
                                           row_scratch.shape[0] // 2))

    # ---------- 3) 脏污 / 异色边缘柔和度 ----------
    print("\n[3] 脏污 / 异色边缘（过渡带占比与边界梯度）")
    rows = []
    for kind, key in (("smudge", "edge_soft"), ("discolor", "edge_blur")):
        tiles = [label(base, "original")]
        for es in (0.0, 1.0, 2.0):
            over = {key: es}
            if kind == "smudge":
                over["dirt_color"] = "dark"      # 深色污渍更便于目视边缘
            out, res = one(kind, img, rm, strength=0.8, severity="obvious", **over)
            sr = soft_ratio(res.mask)
            gr = grad_at_edge(res.mask)
            print(f"    {kind:<9} {key}={es:<4} 过渡带={sr:.0%}  边界梯度={gr:.3f}")
            tiles.append(label(crop_around(out, res.mask, 320),
                               f"{kind} {key}={es} band={sr:.0%}"))
        r = np.concatenate(tiles, axis=1)
        rows.append(cv2.resize(r, (r.shape[1] // 2, r.shape[0] // 2)))

    sheet = stack([row_scratch] + rows)

    # ---------- 4) 默认参数下的四类缺陷总览 ----------
    print("\n[4] 默认参数（数量=1 / 划痕弯曲度 0.55 / 边缘模糊 1.0）")
    full_tiles = [cv2.resize(img, (320, int(img.shape[0] * 320 / img.shape[1])),
                             interpolation=cv2.INTER_AREA)]
    zoom_tiles = [label(base, "original")]
    for kind in ("scratch", "smudge", "discolor", "chip"):
        out, res = one(kind, img, rm, severity="obvious")
        full_tiles.append(cv2.resize(out, (320, int(out.shape[0] * 320 / out.shape[1])),
                                     interpolation=cv2.INTER_AREA))
        zoom_tiles.append(label(crop_around(out, res.mask, 320),
                                f"{kind} n={res.count} default"))
        print(f"    {kind:<9} 实例={res.count}  面积占比={float((res.mask > 0.05).mean()) * 100:.2f}%"
              f"  过渡带={soft_ratio(res.mask):.0%}  边界梯度={grad_at_edge(res.mask):.3f}")
    fh = max(t.shape[0] for t in full_tiles)
    full_tiles = [cv2.copyMakeBorder(t, 0, fh - t.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT, value=(24, 26, 30))
                  for t in full_tiles]
    rows.append(np.concatenate(full_tiles, axis=1))
    rows.append(np.concatenate(zoom_tiles, axis=1))

    sheet = stack([row_scratch] + rows)
    sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))
    path = os.path.join(OUT, "defect_look.png")
    imwrite(path, sheet)
    print(f"\n对比图: {path}  {sheet.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
