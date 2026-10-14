"""基础工具：随机数、噪声场、曲线采样、图像 IO 与色彩空间辅助。

全部依赖 numpy / PIL / cv2，无网络依赖，可在打包后的 exe 内离线运行。
"""
from __future__ import annotations

import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------
# 随机数
# --------------------------------------------------------------------------
class Rng:
    """对 numpy Generator 的轻量封装，统一产出 float / int / choice。"""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        self.g = np.random.default_rng(self.seed)

    def uniform(self, low: float, high: float) -> float:
        return float(self.g.uniform(low, high))

    def randint(self, low: int, high: int) -> int:
        return int(self.g.integers(low, high + 1))

    def choice(self, seq: Sequence, p: Optional[Sequence[float]] = None):
        if p is None:
            idx = int(self.g.integers(0, len(seq)))
        else:
            p = np.asarray(p, dtype=np.float64)
            p = p / p.sum()
            idx = int(self.g.choice(len(seq), p=p))
        return seq[idx]

    def bool(self, prob: float = 0.5) -> bool:
        return bool(self.g.random() < prob)

    def normal(self, mean: float, std: float) -> float:
        return float(self.g.normal(mean, std))


# --------------------------------------------------------------------------
# 噪声场
# --------------------------------------------------------------------------
def _value_noise(shape: Tuple[int, int], cells: int, rng: Rng) -> np.ndarray:
    """单层 value noise：在低分辨率网格上取随机值再双三次插值。"""
    h, w = shape
    ch = max(2, int(cells))
    cw = max(2, int(cells * w / max(h, 1))) if h > 0 else ch
    grid = rng.g.uniform(0.0, 1.0, size=(ch, cw)).astype(np.float32)

    if cv2 is not None:
        return cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)

    from PIL import Image

    img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    img = img.resize((w, h), resample=Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def fractal_noise(shape: Tuple[int, int], rng: Rng,
                  base_cells: int = 4, octaves: int = 5,
                  persistence: float = 0.55, lacunarity: float = 2.0) -> np.ndarray:
    """分形噪声（fBm），返回 [0,1] 的 float32 场。

    用于生成脏污的斑块分布、异色的不均匀过渡、划痕底部的粗糙感。
    """
    h, w = shape
    total = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    cells = float(base_cells)
    for _ in range(max(1, int(octaves))):
        total += amp * _value_noise((h, w), cells, rng)
        norm += amp
        amp *= persistence
        cells *= lacunarity
    out = total / max(norm, 1e-6)
    lo, hi = float(out.min()), float(out.max())
    if hi - lo < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    return ((out - lo) / (hi - lo)).astype(np.float32)


def perlinish_mask(shape: Tuple[int, int], rng: Rng, threshold: float = 0.55,
                   softness: float = 0.12, **kw) -> np.ndarray:
    """用分形噪声阈值化得到软边斑块掩膜，值域 [0,1]。"""
    f = fractal_noise(shape, rng, **kw)
    lo = threshold - max(softness, 1e-4)
    hi = threshold + max(softness, 1e-4)
    m = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(m, dtype=np.float32)


# --------------------------------------------------------------------------
# 曲线 / 几何
# --------------------------------------------------------------------------
def bezier_points(p0, p1, p2, p3, n: int = 128) -> np.ndarray:
    """三次贝塞尔采样，返回 (n,2) float32，使用随机控制点生成自然划痕走向。"""
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float32)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float32)[None, :] for p in (p0, p1, p2, p3))
    mt = 1.0 - t
    return (mt**3) * p0 + 3 * (mt**2) * t * p1 + 3 * mt * (t**2) * p2 + (t**3) * p3


def blur_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """高斯模糊软化掩膜边缘，并把峰值重新归一到 1.0。

    脏污/异色的斑块在覆盖率阈值化之后会出现明显的硬边（像贴纸），
    这里补一次高斯模糊，生成 0→1 的渐变过渡带；sigma 越大边缘越糊。
    """
    m = np.asarray(mask, dtype=np.float32)
    if sigma <= 0.05 or float(m.max()) <= 0.0:
        return m
    if cv2 is not None:
        out = cv2.GaussianBlur(m, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    else:  # pragma: no cover - 无 cv2 时退化为 PIL
        from PIL import Image, ImageFilter

        img = Image.fromarray((np.clip(m, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        out = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=float(sigma))),
                         dtype=np.float32) / 255.0
    mx = float(out.max())
    if mx > 1e-6:
        out = out / mx
    return clip01(out)


def polyline_mask(shape: Tuple[int, int], pts: Sequence[Tuple[float, float]],
                  width: float, taper: Optional[Sequence[float]] = None,
                  feather: float = 1.0) -> np.ndarray:
    """沿折线/曲线绘制变宽度线条掩膜，返回 [0,1] float32。

    taper: 与 pts 等长的宽度系数序列，用来做「两端渐细」的划痕。
    优先走 cv2 的分段绘制；无 cv2 时退化为 PIL 逐段绘制。
    """
    h, w = shape
    n = len(pts)
    if n < 2 or width <= 0:
        return np.zeros((h, w), dtype=np.float32)

    if taper is None:
        taper = np.ones(n, dtype=np.float32)
    taper = np.asarray(taper, dtype=np.float32)
    if taper.shape[0] != n:
        taper = np.ones(n, dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    if cv2 is not None:
        # 用密集子段模拟变宽度，避免 cv2.line 不能变宽的局限
        steps_per_seg = 6
        prev = np.asarray(pts[0], dtype=np.float32)
        prev_t = float(taper[0])
        for i in range(1, n):
            cur = np.asarray(pts[i], dtype=np.float32)
            cur_t = float(taper[i])
            for s in range(1, steps_per_seg + 1):
                a = s / steps_per_seg
                p = prev * (1 - a) + cur * a
                tw = prev_t * (1 - a) + cur_t * a
                wpx = max(1, int(round(width * tw)))
                cv2.line(mask, (int(round(prev[0])), int(round(prev[1]))),
                         (int(round(p[0])), int(round(p[1]))), 1.0, wpx,
                         lineType=cv2.LINE_AA)
                prev, prev_t = p, tw
            prev, prev_t = cur, cur_t
        if feather > 0:
            k = int(max(1, round(feather))) * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return np.clip(mask, 0.0, 1.0)

    from PIL import Image, ImageDraw

    ss = 2  # 超采样抗锯齿
    img = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(img)
    prev = np.asarray(pts[0], dtype=np.float32)
    prev_t = float(taper[0])
    for i in range(1, n):
        cur = np.asarray(pts[i], dtype=np.float32)
        cur_t = float(taper[i])
        wid = max(1, int(round(width * (prev_t + cur_t) * 0.5))) * ss
        d.line([tuple(prev * ss), tuple(cur * ss)], fill=255, width=wid)
        prev, prev_t = cur, cur_t
    img = img.resize((w, h), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if feather > 0:
        from PIL import ImageFilter

        img2 = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=feather))
        arr = np.asarray(img2, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


# --------------------------------------------------------------------------
# 色彩空间
# --------------------------------------------------------------------------
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055).astype(np.float32)


def rgb_to_lab(img_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 Lab（L:0..100, a/b:-128..127）。"""
    if cv2 is not None:
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    from PIL import Image

    return np.asarray(Image.fromarray(img_u8).convert("LAB"), dtype=np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float32 Lab -> uint8 RGB。"""
    lab_u8 = np.clip(np.round(lab), 0, 255).astype(np.uint8)
    if cv2 is not None:
        return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)
    from PIL import Image

    return np.asarray(Image.fromarray(lab_u8, mode="LAB").convert("RGB"), dtype=np.uint8)


# --------------------------------------------------------------------------
# 图像 IO（兼容中文路径）
# --------------------------------------------------------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def imread(path: str) -> np.ndarray:
    """读入为 RGB uint8，支持中文/空格路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise IOError(f"无法读取图片（空文件）: {path}")
    if cv2 is not None:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"无法解码图片: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def imwrite(path: str, rgb: np.ndarray) -> None:
    """写出 RGB uint8 图片，支持中文路径。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if cv2 is not None:
        ext = os.path.splitext(path)[1].lower() or ".png"
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(ext, bgr)
        if not ok:
            raise IOError(f"无法编码图片: {path}")
        buf.tofile(path)
        return
    from PIL import Image

    Image.fromarray(rgb, mode="RGB").save(path)


def list_images(folder: str, recursive: bool = True,
                skip_internal: bool = True) -> List[str]:
    """列出目录下所有受支持的图片文件。

    skip_internal=True 时跳过 ``_`` 前缀的子目录（如 ``_truth/``、``_labels/``），
    避免把标注/真值/中间产物当作输入原图。
    """
    out: List[str] = []
    if not os.path.isdir(folder):
        return out
    if recursive:
        for root, dirs, files in os.walk(folder):
            if skip_internal:
                dirs[:] = [d for d in dirs if not d.startswith("_")]
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS):
                    out.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(folder)):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and f.lower().endswith(IMG_EXTS):
                out.append(p)
    return out


def safe_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in str(name)).strip()
    return out or "unnamed"


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def to_u8(x: np.ndarray) -> np.ndarray:
    """[0,1] float -> uint8。"""
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def blend(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """base/overlay 为 [0,1] float32 (H,W,3)，alpha 为 (H,W) 或 (H,W,1)。"""
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    return clip01(base * (1.0 - alpha) + overlay * alpha)


def filled_circle_mask(shape, cx: float, cy: float, r: float, feather: float = 1.5) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    lo, hi = max(r - feather, 1e-3), r + feather
    return np.clip((hi - d) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
