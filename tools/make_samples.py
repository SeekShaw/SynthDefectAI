"""生成测试用的「合格产品」样图 —— 模拟金属/塑料外壳的电子元器件。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils import imwrite, fractal_noise, Rng, to_u8, clip01  # noqa: E402


def make_product(seed=1, w=900, h=700, base=(0.62, 0.64, 0.68)):
    rng = Rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    base = np.array(base, dtype=np.float32)

    # 外壳主体：圆角矩形
    margin = 0.07
    x0, x1 = w * margin, w * (1 - margin)
    y0, y1 = h * margin, h * (1 - margin)
    r = min(w, h) * 0.06
    dx = np.maximum(np.maximum(x0 + r - xx, xx - (x1 - r)), 0)
    dy = np.maximum(np.maximum(y0 + r - yy, yy - (y1 - r)), 0)
    body = (np.sqrt(dx**2 + dy**2) <= r).astype(np.float32)
    body *= ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)).astype(np.float32)
    from core.utils import polyline_mask  # noqa
    body = np.clip(body, 0, 1)

    # 环境渐变光（左上亮）
    grad = 1.0 - 0.42 * ((xx / w) * 0.6 + (yy / h) * 0.4)
    img[:] = 0.12  # 背景暗
    body_col = base[None, None, :] * grad[:, :, None]
    img = img * (1 - body[:, :, None]) + body_col * body[:, :, None]

    # 主体表面轻微纹理
    tex = fractal_noise((h, w), rng, base_cells=40, octaves=4, persistence=0.6)
    img += ((tex - 0.5) * 0.05)[:, :, None] * body[:, :, None]

    # 元件：芯片黑块
    def rect(cx, cy, rw, rh, color, round_r=6):
        m = np.zeros((h, w), dtype=np.float32)
        xa, xb = cx - rw / 2, cx + rw / 2
        ya, yb = cy - rh / 2, cy + rh / 2
        ddx = np.maximum(np.maximum(xa + round_r - xx, xx - (xb - round_r)), 0)
        ddy = np.maximum(np.maximum(ya + round_r - yy, yy - (yb - round_r)), 0)
        m = (np.sqrt(ddx**2 + ddy**2) <= round_r).astype(np.float32)
        m *= ((xx >= xa) & (xx <= xb) & (yy >= ya) & (yy <= yb)).astype(np.float32)
        return m, color

    parts = [
        (w * 0.35, h * 0.42, w * 0.30, h * 0.34, (0.09, 0.09, 0.11)),
        (w * 0.70, h * 0.36, w * 0.14, h * 0.16, (0.14, 0.14, 0.16)),
        (w * 0.70, h * 0.66, w * 0.14, h * 0.14, (0.14, 0.14, 0.16)),
    ]
    for cx, cy, rw, rh, col in parts:
        m, c = rect(cx, cy, rw, rh, np.array(col, dtype=np.float32))
        img = img * (1 - m[:, :, None] * 0.98) + c[None, None, :] * (m[:, :, None] * 0.98)

    # 引脚
    for i in range(6):
        px = w * (0.22 + i * 0.032)
        m, c = rect(px, h * 0.79, w * 0.014, h * 0.10, np.array((0.55, 0.56, 0.58), np.float32), 2)
        img = img * (1 - m[:, :, None]) + c[None, None, :] * m[:, :, None]

    # 高光条
    hi = np.exp(-((xx - w * 0.30) ** 2) / (2 * (w * 0.055) ** 2)) * body
    img += (hi * 0.10)[:, :, None]

    return to_u8(clip01(img))


def make_metal(seed=2, w=800, h=600):
    """浅色金属件 + 拉丝。"""
    rng = Rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array((0.78, 0.79, 0.80), np.float32)
    grad = 1.0 - 0.30 * (yy / h) - 0.12 * (xx / w)
    img = 0.10 + 0.0 * xx[:, :, None]
    img = np.repeat(np.full((h, w, 1), 0.10, np.float32), 3, axis=2)
    m = np.ones((h, w), np.float32)
    img = img * (1 - m[:, :, None]) + (base[None, None, :] * grad[:, :, None]) * m[:, :, None]

    brush = fractal_noise((h, w), rng, base_cells=180, octaves=2)
    brush = np.tile(brush.mean(axis=1, keepdims=True), (1, w))
    img += ((brush - 0.5) * 0.055)[:, :, None]
    return to_u8(clip01(img))


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    os.makedirs(out, exist_ok=True)
    imwrite(os.path.join(out, "product_dark.png"), make_product(1, 900, 700, (0.58, 0.60, 0.64)))
    imwrite(os.path.join(out, "product_gray.png"), make_product(2, 820, 640, (0.42, 0.43, 0.45)))
    imwrite(os.path.join(out, "product_metal.png"), make_metal(3, 800, 600))
    print("samples written to", out)
    for f in sorted(os.listdir(out)):
        print("  ", f)
