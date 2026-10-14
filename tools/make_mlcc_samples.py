"""合成结构真实的 MLCC 样图，用于开发与验证区域分割算法。

MLCC 外观结构（沿长度方向）：
    端电极 | 瓷体 | 端电极
端电极（terminal）为银灰金属镀层，并向上下表面包裹一小段（wrap），
瓷体（ceramic body）为灰白/米黄陶瓷，带轻微分层纹理与高点状杂质。
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils import Rng, fractal_noise, clip01, to_u8, imwrite  # noqa: E402


def rounded_rect_mask(h, w, x0, y0, x1, y1, r):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.maximum(np.maximum(x0 + r - xx, xx - (x1 - r)), 0)
    dy = np.maximum(np.maximum(y0 + r - yy, yy - (y1 - r)), 0)
    inside = (np.sqrt(dx**2 + dy**2) <= r) & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    return inside.astype(np.float32)


def make_mlcc(seed=1, W=1000, H=520, horizontal=True, bg=(0.14, 0.15, 0.17),
              ceramic=(0.72, 0.71, 0.66), metal=(0.60, 0.62, 0.65),
              term_ratio=0.22, cap_ratio=0.42, label=""):
    """生成一张 MLCC 产品图 + region map。

    region map: 0=背景 1=瓷体 2=端电极
    """
    rng = Rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[:] = np.array(bg, np.float32)[None, None, :]

    # 底色加环境光渐变
    gl = 1.0 - 0.30 * (yy / H) - 0.10 * (xx / W)
    img *= gl[:, :, None]

    region = np.zeros((H, W), np.uint8)

    if horizontal:
        bw, bh = W * 0.72, H * 0.42
        cx, cy = W * 0.5, H * 0.5
    else:
        bw, bh = W * 0.30, H * 0.70
        cx, cy = W * 0.5, H * 0.5

    x0, y0 = cx - bw / 2, cy - bh / 2
    x1, y1 = cx + bw / 2, cy + bh / 2
    r = min(bw, bh) * 0.16
    cap = rounded_rect_mask(H, W, x0, y0, x1, y1, r)

    # ---- 端电极宽度（沿长度方向） ----
    tw = bh * term_ratio            # 电极沿长度方向的宽度
    if horizontal:
        term_zone = ((xx >= x0) & (xx <= x0 + tw)) | ((xx >= x1 - tw) & (xx <= x1))
    else:
        term_zone = ((yy >= y0) & (yy <= y0 + tw)) | ((yy >= y1 - tw) & (yy <= y1))

    body = cap * (1 - term_zone.astype(np.float32))
    term = cap * term_zone.astype(np.float32)

    # ---- 瓷体：陶瓷基色 + 分层纹理 + 微杂质点 ----
    cer_col = np.array(ceramic, np.float32) * rng.uniform(0.94, 1.06)
    cbase = np.broadcast_to(cer_col[None, None, :], (H, W, 3)).copy()
    # 陶瓷层状纹（沿短边方向的细微条纹）
    layer = fractal_noise((H, W), rng, base_cells=int(bh / 6), octaves=3, persistence=0.6)
    if horizontal:
        stripe = np.sin(yy / max(bh, 1) * np.pi * rng.uniform(8, 16))
    else:
        stripe = np.sin(xx / max(bw, 1) * np.pi * rng.uniform(8, 16))
    cbase += ((stripe * 0.018 + (layer - 0.5) * 0.05))[:, :, None]
    # 陶瓷整体受光
    cgrad = 1.0 - 0.18 * ((yy - y0) / max(bh, 1)) - 0.10 * ((xx - x0) / max(bw, 1))
    cbase *= cgrad[:, :, None]
    # 极少量的自然杂质点（用于验证「不要把正常杂质当缺陷」）
    nat = (fractal_noise((H, W), rng, base_cells=110, octaves=2) > 0.86).astype(np.float32)
    cbase -= (nat * 0.06)[:, :, None]

    # ---- 端电极：金属基色 + 竖向镀层纹理 + 强反光带 ----
    m_col = np.array(metal, np.float32) * rng.uniform(0.95, 1.05)
    mbase = np.broadcast_to(m_col[None, None, :], (H, W, 3)).copy()
    if horizontal:
        pgrad = np.gradient(xx + yy * 0.15)  # 斜向镀层
    else:
        pgrad = np.gradient(yy + xx * 0.15)
    mbase += (rng.g.normal(0, 1, (H, W)).astype(np.float32) * 0.022)[:, :, None]
    # 电极上的横向反光带
    if horizontal:
        hl = np.exp(-((xx - (x0 + tw * 0.5)) ** 2) / (2 * (tw * 0.28) ** 2))
        hr = np.exp(-((xx - (x1 - tw * 0.5)) ** 2) / (2 * (tw * 0.28) ** 2))
    else:
        hl = np.exp(-((yy - (y0 + tw * 0.5)) ** 2) / (2 * (tw * 0.28) ** 2))
        hr = np.exp(-((yy - (y1 - tw * 0.5)) ** 2) / (2 * (tw * 0.28) ** 2))
    hl, hr = hl * rng.uniform(0.7, 1.2), hr * rng.uniform(0.7, 1.2)
    mbase += (np.clip(hl + hr, 0, 2) * 0.10)[:, :, None]

    # ---- 合成到画布 ----
    img = img * (1 - body[:, :, None]) + cbase * body[:, :, None]
    img = img * (1 - term[:, :, None]) + mbase * term[:, :, None]

    region[(body > 0.5)] = 1
    region[(term > 0.5)] = 2

    # ---- 阴影与边缘暗化，增强真实感 ----
    from core.utils import polyline_mask  # noqa

    sh = np.zeros((H, W), np.float32)
    off = max(3, int(bh * 0.05))
    if horizontal:
        sh = rounded_rect_mask(H, W, x0 + off, y0 + off, x1 + off, y1 + off * 2, r)
    else:
        sh = rounded_rect_mask(H, W, x0 + off, y0 + off, x1 + off * 2, y1 + off, r)
    sh = np.clip(sh - cap, 0, 1) * 0.55
    img = img * (1 - sh[:, :, None] * 0.7)

    elev = np.clip(np.sqrt(np.maximum(cap, 0)), 0, 1)
    img = img * (0.92 + 0.08 * elev[:, :, None])

    return to_u8(clip01(img)), region


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(base, "samples_mlcc")
    # 真值图（region / regionmap）放子目录，避免被 list_images 当成输入原图
    truth = os.path.join(out, "_truth")
    os.makedirs(truth, exist_ok=True)

    cfgs = [
        dict(seed=11, W=1000, H=520, horizontal=True, label="h_std",
             ceramic=(0.74, 0.73, 0.68), metal=(0.62, 0.64, 0.67),
             bg=(0.13, 0.14, 0.16)),
        dict(seed=12, W=1000, H=520, horizontal=True, label="h_warm",
             ceramic=(0.79, 0.74, 0.62), metal=(0.58, 0.60, 0.64),
             bg=(0.10, 0.11, 0.13)),
        dict(seed=13, W=760, H=760, horizontal=False, label="v_std",
             ceramic=(0.71, 0.70, 0.66), metal=(0.63, 0.65, 0.68),
             bg=(0.15, 0.16, 0.18)),
        dict(seed=14, W=1100, H=460, horizontal=True, label="h_bright",
             ceramic=(0.82, 0.81, 0.76), metal=(0.70, 0.72, 0.75),
             bg=(0.22, 0.23, 0.25)),
        dict(seed=15, W=900, H=620, horizontal=False, label="v_dark",
             ceramic=(0.66, 0.65, 0.61), metal=(0.55, 0.57, 0.60),
             bg=(0.08, 0.09, 0.10)),
    ]

    for c in cfgs:
        label = c.pop("label")
        img, region = make_mlcc(**c)
        imwrite(os.path.join(out, f"mlcc_{label}.png"), img)
        # 干净标签图：0=背景 127=瓷体 255=端电极（供分割精度评估）
        lab = np.zeros_like(region)
        lab[region == 1] = 127
        lab[region == 2] = 255
        imwrite(os.path.join(truth, f"regionmap_{label}.png"), lab)
        # 区域可视化：瓷体绿、电极蓝
        vis = img.astype(np.float32).copy()
        vis[region == 1] = vis[region == 1] * 0.35 + np.array([40, 220, 130], np.float32) * 0.65
        vis[region == 2] = vis[region == 2] * 0.35 + np.array([60, 140, 255], np.float32) * 0.65
        imwrite(os.path.join(truth, f"region_{label}.png"), np.clip(vis, 0, 255).astype(np.uint8))
        print(f"  {label}: body={int((region == 1).sum())} term={int((region == 2).sum())}")

    print("MLCC samples written to", out)


if __name__ == "__main__":
    main()
