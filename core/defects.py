"""四类表面缺陷的物理仿真生成算法。

设计原则（对齐工业视觉合成缺陷的主流做法）：
1. 先用几何/噪声模型生成「缺陷区域掩膜」；
2. 再基于光照模型计算缺陷区域内的像素变化（而非直接叠加固定颜色）；
3. 最后按掩膜 alpha 混合回原图，并对缺陷内部做材质相关的灰度/色度扰动。

四类缺陷：
- ScratchDefect  划痕：贝塞尔曲线路径 + 变宽度 + 亮暗双边 + 微观抖动
- SmudgeDefect   脏污：分形噪声斑块 + 颗粒质感 + 半透明污染色
- DiscolorDefect 异色：区域掩膜 + Lab 色域偏移 + 边缘渐晕
- ChipDefect     缺角：轮廓啃缺口 + 粗糙断面颗粒 + 交界暗边 + 碎屑/裂纹
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .utils import (Rng, blend, blur_mask, clip01,
                    filled_circle_mask, fractal_noise, perlinish_mask,
                    polyline_mask, rgb_to_lab, lab_to_rgb, to_u8,
                    srgb_to_linear, linear_to_srgb)
from .mlcc_region import sample_point


# 明显度分级 → 物理参数缩放
SEVERITY_LEVELS = {
    "obvious": (0.80, 1.00),    # 明显
    "medium": (0.45, 0.70),     # 中等
    "subtle": (0.12, 0.35),     # 不明显（接近检测阈值的困难样本）
    "mixed": (0.12, 1.00),      # 随机混合
}
SEVERITY_CN = {"obvious": "明显", "medium": "中等", "subtle": "不明显", "mixed": "随机混合"}


# ==========================================================================
# 通用基类
# ==========================================================================
@dataclass
class DefectResult:
    """一次缺陷仿真的产物。"""
    image: np.ndarray          # (H,W,3) uint8 缺陷图
    mask: np.ndarray           # (H,W) float32 [0,1] 缺陷区域
    kind: str = "defect"       # 缺陷类型名
    count: int = 1             # 该类型下实际生成的缺陷实例个数
    severity: str = ""         # 明显度等级（obvious/medium/subtle/mixed）
    surface: str = ""          # 目标表面（body/electrode/both）


class BaseDefect:
    """缺陷生成器基类。子类实现 `render`。"""

    kind = "base"

    def __init__(self, strength: float = 0.5, severity_mult: float = 1.0, **params) -> None:
        # strength 统一为 0..1 的强度刻度，内部再映射到物理参数区间
        self.strength = float(np.clip(strength, 0.0, 1.0))
        # 明显度乘子（0.12~1.0），用于对物理参数做二次缩放
        self.severity_mult = float(np.clip(severity_mult, 0.05, 1.0))
        self.params = params

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None,
               hint=None) -> DefectResult:  # pragma: no cover
        raise NotImplementedError


def _bbox_short_side(region: np.ndarray) -> float:
    """区域外接框的短边长度（区域为空时返回 0）。"""
    ys, xs = np.nonzero(region > 0.5)
    if len(xs) == 0:
        return 0.0
    return float(min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))


# ==========================================================================
# 1. 划痕
# ==========================================================================
@dataclass
class ScratchParams:
    count: int = 1                  # 划痕条数
    length_ratio: float = 0.55      # 长度相对图像短边的比例
    width_px: float = 2.2           # 基准线宽（像素）
    curve: float = 0.55             # 弯曲度 0=近直线，1=明显蜿蜒（按拖拽路径生成）
    tone: float = 0.0               # 颜色：-1=白 … +1=黑，0=按 dark_ratio 随机
    tone_depth: float = 0.85        # 颜色深浅程度（1=达到最大对比）
    dark_ratio: float = 0.45        # tone=0 时呈「凹陷暗线」的概率，其余为「亮反光线」
    jitter: float = 1.6             # 微观抖动幅度（像素），让边缘不平滑
    feather: float = 1.2            # 羽化半径
    double_edge: bool = True         # 是否在暗线两侧叠加亮边（金属/镀膜反光）
    double_edge_gain: float = 0.35


def _mask_direction(mask: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    """掩膜的形状主方向：返回 (角度rad, 离心率0~1)；太圆则返回 None。

    划痕要「沿着生成区域的方向」走，所以这里直接用区域掩膜的 PCA 主轴。
    """
    if mask is None:
        return None
    ys, xs = np.nonzero(np.asarray(mask) > 0.5)
    if len(xs) < 32:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mu = pts.mean(axis=0)
    cov = np.atleast_2d(np.cov((pts - mu).T))
    try:
        vals, vecs = np.linalg.eigh(cov)
    except Exception:  # pragma: no cover
        return None
    vals = np.clip(vals, 1e-9, None)
    order = np.argsort(vals)[::-1]
    l1, l2 = vals[order[0]], vals[order[1]]
    elong = float(np.sqrt(l1 / max(l2, 1e-9)))
    if elong < 1.25:            # 接近圆形/方形，方向不可靠
        return None
    v = vecs[:, order[0]]
    ang = math.atan2(float(v[1]), float(v[0]))
    ecc = float(np.clip(1.0 - l2 / l1, 0.0, 1.0))
    return ang, ecc


class ScratchDefect(BaseDefect):
    kind = "scratch"

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None,
               hint=None, paths=None) -> DefectResult:
        """生成划痕。

        paths 不为空（手绘线条模式）：沿每条手绘线生成，**起止位置与手绘一致**；
        否则退回「沿生成区域方向」自由生成（批量流程无手绘时使用）。
        """
        h, w = rgb.shape[:2]
        s = self.strength
        sv = self.severity_mult
        tone = float(np.clip(self.params.get("tone", 0.0), -1.0, 1.0))
        p = ScratchParams(
            # 数量参数即最终条数（不再随强度浮动），避免「设 1 却出 3 条」的错觉
            count=max(1, int(self.params.get("count", 1))),
            length_ratio=self.params.get("length_ratio", 0.55),
            width_px=self.params.get("width_px", 2.2) * (0.6 + 1.1 * s) * (0.5 + 0.5 * sv),
            curve=float(np.clip(self.params.get("curve", 0.55), 0.0, 1.0)),
            tone=tone,
            tone_depth=float(np.clip(self.params.get("tone_depth", 0.85), 0.05, 2.0)),
            dark_ratio=self.params.get("dark_ratio", 0.45),
            jitter=self.params.get("jitter", 1.6),
            feather=self.params.get("feather", 1.2) + (1.0 - sv) * 1.3,
        )

        base = rgb.astype(np.float32) / 255.0
        total_mask = np.zeros((h, w), dtype=np.float32)
        # 累积「亮度偏移」，正=提亮，负=压暗
        total_shift = np.zeros((h, w), dtype=np.float32)
        self.last_dir_source = "随机"
        made = 0

        # ---------------- 模式 A：沿手绘线条（起止点一致） ----------------
        if paths:
            self.last_dir_source = "手绘线条"
            for path in paths:
                pts = self._resample_path(path, n=220)
                if pts is None:
                    continue
                for _ in range(p.count):
                    lm, pol, gain = self._scratch_along((h, w), pts, p, rng, region)
                    if lm is None:
                        continue
                    made += 1
                    total_mask = np.maximum(total_mask, lm)
                    total_shift += lm * pol * gain
            if made == 0:
                return DefectResult(image=rgb.copy(), mask=total_mask,
                                    kind=self.kind, count=0)
            return self._compose(base, total_mask, total_shift, p, s, sv, rng, made)

        # ---------------- 模式 B：沿生成区域方向自由生成 ----------------
        base_ang: Optional[float] = None
        dir_src = "随机"
        if hint is not None:
            try:
                base_ang = float(hint[0])
                dir_src = "手绘区域"
            except Exception:
                base_ang = None
        if base_ang is None:
            md = _mask_direction(region)
            if md is not None:
                base_ang = md[0]
                dir_src = "区域主轴"
        if base_ang is None and region is not None:
            ax = self._major_axis(region)
            if ax is not None:
                base_ang = ax
                dir_src = "产品主轴"
        short = min(h, w)
        if region is not None:
            short = int(max(np.sqrt(max((region > 0.5).sum(), 1.0)) * 1.6, 24))

        attempts = 0
        while made < p.count and attempts < p.count * 8:
            attempts += 1
            line_mask, polarity, gain = self._one_scratch(
                (h, w), short, p, rng, region, base_ang)
            if line_mask is None:
                continue
            made += 1
            total_mask = np.maximum(total_mask, line_mask)
            total_shift += line_mask * polarity * gain
        if made == 0:
            return DefectResult(image=rgb.copy(), mask=total_mask,
                                kind=self.kind, count=0)
        self.last_dir_source = dir_src
        return self._compose(base, total_mask, total_shift, p, s, sv, rng, made)

    # ---------------- 合成 ----------------
    def _compose(self, base, total_mask, total_shift, p: ScratchParams,
                 s: float, sv: float, rng: Rng, made: int) -> DefectResult:
        """把累积的掩膜/亮度偏移合成为缺陷图（两种模式共用）。"""
        h, w = total_mask.shape[:2]
        total_mask = clip01(total_mask)
        total_shift = np.clip(total_shift, -1.0, 1.0)

        # 物理化：凹陷划痕散射掉部分入射光 -> 变暗；反光划痕镜面反射 -> 变亮
        out = base.copy()
        dark_sel = np.clip(-total_shift, 0.0, 1.0)
        out = out * (1.0 - dark_sel[:, :, None] * 0.85)
        bright_sel = np.clip(total_shift, 0.0, 1.0)
        sat = out.mean(axis=2, keepdims=True)
        out = out + (bright_sel[:, :, None] * (0.55 * (out * 0.35 + 0.65)))
        out = out * (1 - bright_sel[:, :, None] * 0.18) + sat * (bright_sel[:, :, None] * 0.18)
        out = clip01(out)

        # 划痕内部微观粗糙（金属拉丝感）
        grain = fractal_noise((h, w), rng, base_cells=64, octaves=3, persistence=0.5)
        rough = (grain - 0.5) * 0.22 * (0.4 + s) * (0.5 + 0.5 * sv)
        out = clip01(out + (rough * total_mask)[:, :, None])

        return DefectResult(image=to_u8(out), mask=total_mask,
                            kind=self.kind, count=made)

    def _polarity_gain(self, p: ScratchParams, rng: Rng):
        """按颜色设定返回 (polarity, gain)。

        tone < -0.06 → 亮白线；tone > +0.06 → 暗黑线；|tone| 越大颜色越极端；
        tone ≈ 0 → 按 dark_ratio 随机（凹陷暗线 / 亮反光线）。
        """
        gain_scale = 0.45 + 0.55 * self.severity_mult
        depth = float(np.clip(p.tone_depth, 0.05, 2.0))
        if abs(p.tone) < 0.06:
            dark = rng.bool(p.dark_ratio)
            gain = (rng.uniform(0.55, 0.95) if dark else rng.uniform(0.30, 0.75)) \
                * (0.55 + 0.45 * self.strength) * gain_scale
            return (-1.0 if dark else 1.0), float(gain)
        dark = p.tone > 0
        mag = 0.35 + 0.65 * abs(p.tone)
        base_g = rng.uniform(0.55, 0.95) if dark else rng.uniform(0.45, 0.85)
        gain = base_g * mag * depth * (0.55 + 0.45 * self.strength) * gain_scale
        return (-1.0 if dark else 1.0), float(np.clip(gain, 0.0, 1.0))

    # ---------------- 沿手绘线条生成 ----------------
    @staticmethod
    def _resample_path(path, n: int = 220) -> Optional[np.ndarray]:
        """把手绘折线按弧长重采样并轻度平滑，返回 (n,2) float32。"""
        pts = np.asarray([(float(x), float(y)) for x, y in path], dtype=np.float64)
        if len(pts) < 2:
            return None
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 1e-6:
            return None
        t = np.linspace(0.0, 1.0, n)
        xs = np.interp(t, s / s[-1], pts[:, 0])
        ys = np.interp(t, s / s[-1], pts[:, 1])
        out = np.stack([xs, ys], axis=1)
        if len(pts) >= 3:                 # 手绘折线轻度平滑，去掉鼠标抖动
            k = 5
            pad = k // 2
            ker = np.ones(k, dtype=np.float64) / k
            out = np.stack([
                np.convolve(np.pad(out[:, 0], pad, mode="edge"), ker, mode="valid"),
                np.convolve(np.pad(out[:, 1], pad, mode="edge"), ker, mode="valid"),
            ], axis=1)
        return out.astype(np.float32)

    def _scratch_along(self, shape, pts: np.ndarray, p: ScratchParams, rng: Rng,
                       region: Optional[np.ndarray]):
        """在给定路径上生成一条划痕：起止位置与路径端点一致。"""
        n = len(pts)
        t = np.linspace(0, 1, n, dtype=np.float32)
        # 两端轻微渐细但保留 0.72 宽度，保证起止位置就是手绘的起止位置
        taper = 0.72 + 0.28 * np.clip(np.sin(np.pi * t), 0.0, None) ** 0.5
        width = max(1.0, p.width_px)

        if p.jitter > 0:                 # 轻微侧向游走（两端为 0，不影响起止点）
            jn = fractal_noise((1, n), rng, base_cells=10, octaves=3)
            jn = (jn.ravel() - 0.5) * 2.0 * p.jitter * 0.35
            tg = np.gradient(pts, axis=0)
            nm = np.stack([-tg[:, 1], tg[:, 0]], axis=1)
            nl = np.linalg.norm(nm, axis=1, keepdims=True) + 1e-6
            env = (np.clip(np.sin(np.pi * t), 0.0, None) ** 0.7)[:, None]
            pts = pts + (nm / nl) * (jn[:, None] * env)

        m = polyline_mask(shape, [(float(x), float(y)) for x, y in pts],
                          width=width, taper=taper, feather=p.feather)
        if region is not None:
            raw_pos = float((m > 0.05).sum())
            m = m * clip01(region)
            if float((m > 0.05).sum()) < 0.25 * max(raw_pos, 1.0):
                return None, 0.0, 0.0
        polarity, gain = self._polarity_gain(p, rng)
        if polarity < 0 and p.double_edge:      # 暗线两侧加亮边
            off = max(1.0, width * 0.9)
            tg = np.gradient(pts, axis=0)
            nm = np.stack([-tg[:, 1], tg[:, 0]], axis=1)
            nl = np.linalg.norm(nm, axis=1, keepdims=True) + 1e-6
            unit = nm / nl
            for sign in (1.0, -1.0):
                sp = pts + unit * (off * sign)
                me = polyline_mask(shape, [(float(x), float(y)) for x, y in sp],
                                   width=max(1.0, width * 0.55), taper=taper,
                                   feather=p.feather * 1.4)
                if region is not None:
                    me = me * clip01(region)
                m = np.maximum(m, me * 0.55)
        m = np.clip(m, 0.0, 1.0) ** 0.75
        return m, polarity, gain

    @staticmethod
    def _major_axis(region: np.ndarray) -> Optional[float]:
        """求区域主轴方向（弧度），用于让划痕沿产品长边延伸。"""
        try:
            import cv2

            m = cv2.moments((region > 0.5).astype(np.uint8), binaryImage=True)
            if m["m00"] > 16 and (m["mu20"] - m["mu02"]) ** 2 + 4 * m["mu11"] ** 2 > 1e-6:
                return 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
        except Exception:
            pass
        return None

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float((a + math.pi) % (2 * math.pi) - math.pi)

    def _draw_path(self, p0, ang: float, L: float, rng: Rng, curve: float,
                   n: int = 180) -> np.ndarray:
        """按「手工拖拽」的方式生成划痕路径，返回 (n,2) float32。

        三步叠加，避免出现「一条数学直线」：
        1. **方向随机游走**：把路径切成若干段，每段方向做小幅随机转动
           （模拟手拖时不稳），并朝初始方向轻微回正，避免卷成团；
        2. **偶发转折 kink**：以一定概率来一个明显拐弯；
        3. **低频蛇形**：沿法线叠加 1~4 个周期的正弦横向偏移，形成 S 形摆动，
           最后做移动平均平滑去掉折线拐点。

        curve=0 退化为近似直线（旧行为），curve=1 明显蜿蜒。
        """
        c = float(np.clip(curve, 0.0, 1.0))
        segs = int(np.clip(round(3 + 7 * c), 3, 12))     # 分段数
        seg_len = L / segs
        sigma = 0.05 + 0.52 * c                         # 每段方向抖动（弧度）
        pull = 0.14 + 0.14 * (1.0 - c)                  # 朝初始方向回正
        kink_p = 0.12 + 0.28 * c                         # 明显转折概率
        amp = L * (0.003 + 0.075 * c)                    # 蛇形横向幅度
        cycles = 1.0 + 3.0 * c

        pts = [np.asarray(p0, dtype=np.float64)]
        cur = float(ang)
        for _ in range(segs):
            step = cur + rng.normal(0.0, sigma)
            if rng.bool(kink_p):
                step += rng.choice((-1.0, 1.0)) * rng.uniform(0.30, 0.85)
            step -= pull * self._wrap_angle(step - ang)
            cur = step
            ln = seg_len * rng.uniform(0.82, 1.18)
            pts.append(pts[-1] + np.array([math.cos(cur), math.sin(cur)]) * ln)

        raw = np.asarray(pts, dtype=np.float64)
        seg = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 1e-6:
            return np.repeat(raw[:1], n, axis=0).astype(np.float32)
        s = s / s[-1]

        t = np.linspace(0.0, 1.0, n)
        xs = np.interp(t, s, raw[:, 0])
        ys = np.interp(t, s, raw[:, 1])

        # 低频蛇形（两端收敛到 0，避免端点被推离原位置）
        nx, ny = -math.sin(ang), math.cos(ang)
        phase = rng.uniform(0.0, 2 * np.pi)
        env = np.sin(np.pi * t) ** 0.6
        wave = np.sin(2 * np.pi * cycles * t + phase) * amp * env
        xs = xs + nx * wave
        ys = ys + ny * wave

        out = np.stack([xs, ys], axis=1)
        k = int(np.clip(round(3 + 12 * c), 3, 15)) | 1   # 移动平均窗口（奇数）
        if k > 1:
            pad = k // 2
            ker = np.ones(k, dtype=np.float64) / k
            px = np.convolve(np.pad(out[:, 0], pad, mode="edge"), ker, mode="valid")
            py = np.convolve(np.pad(out[:, 1], pad, mode="edge"), ker, mode="valid")
            out = np.stack([px, py], axis=1)

        # ---- 主轴对齐：把整条路径绕起点旋转到 ang ----
        # 随机游走只负责「形状」，方向必须由调用方（手绘区域/目标表面）决定，
        # 否则多条划痕的走向会互相抵消，看起来又变成「随机方向」。
        centered = out - out.mean(axis=0)
        cov = np.atleast_2d(np.cov(centered.T))
        try:
            vals, vecs = np.linalg.eigh(cov)
            v = vecs[:, int(np.argmax(vals))]
            cur = math.atan2(float(v[1]), float(v[0]))
            d = ang - cur
            d = (d + math.pi) % (2 * math.pi) - math.pi
            if abs(d) > 1e-6:
                ca, sa = math.cos(d), math.sin(d)
                rel = out - out[0]
                out = np.stack([rel[:, 0] * ca - rel[:, 1] * sa,
                                rel[:, 0] * sa + rel[:, 1] * ca], axis=1) + out[0]
        except Exception:  # pragma: no cover
            pass
        return out.astype(np.float32)

    def _one_scratch(self, shape, short, p: ScratchParams, rng: Rng,
                     region: Optional[np.ndarray] = None,
                     base_angle: Optional[float] = None):
        h, w = shape
        if base_angle is not None:
            # 沿区域方向：只给 ±10° 的自然抖动，绝不随机乱向
            ang = base_angle + rng.uniform(-0.18, 0.18)
            if region is not None:
                sx, sy = sample_point(region, rng, margin=0.02)
                p0 = (sx, sy)
            else:
                p0 = (rng.uniform(0.15 * w, 0.85 * w), rng.uniform(0.15 * h, 0.85 * h))
        elif region is not None:
            # 区域模式但方向不可靠（区域接近圆形）：位置在区域内，方向随机
            sx, sy = sample_point(region, rng, margin=0.02)
            ang = rng.uniform(0, 2 * np.pi)
            p0 = (sx, sy)
        else:
            # 自由模式：从图像边缘附近出发更符合真实刮擦
            edge = rng.randint(0, 3)
            if edge == 0:      # 上边
                p0 = (rng.uniform(0.1 * w, 0.9 * w), rng.uniform(-0.05 * h, 0.10 * h))
                ang = rng.uniform(0.35, 1.20)   # 向下
            elif edge == 1:    # 下边
                p0 = (rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.90 * h, 1.05 * h))
                ang = rng.uniform(0.35, 1.20) + np.pi
            elif edge == 2:    # 左边
                p0 = (rng.uniform(-0.05 * w, 0.10 * w), rng.uniform(0.1 * h, 0.9 * h))
                ang = rng.uniform(-0.45, 0.45)
            else:              # 右边
                p0 = (rng.uniform(0.90 * w, 1.05 * w), rng.uniform(0.1 * h, 0.9 * h))
                ang = rng.uniform(-0.45, 0.45) + np.pi

        L = short * p.length_ratio * rng.uniform(0.6, 1.35)

        # 路径：按拖拽轨迹生成（分段随机游走 + 蛇形 + 平滑）
        pts = self._draw_path(p0, ang, L, rng, p.curve, n=180)

        # 微观抖动：沿法线方向做高频噪声偏移，避免过于「数学化」
        if p.jitter > 0:
            n = len(pts)
            jn = fractal_noise((1, n), rng, base_cells=12, octaves=3)
            jn = (jn.ravel() - 0.5) * 2.0 * p.jitter
            tang = np.gradient(pts, axis=0)
            norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
            nlen = np.linalg.norm(norm, axis=1, keepdims=True) + 1e-6
            pts = pts + (norm / nlen) * jn[:, None]

        # 两端渐细（裁剪掉 sin 在端点处的浮点负值，避免负数开方出 NaN）
        t = np.linspace(0, 1, len(pts), dtype=np.float32)
        taper = np.clip(np.sin(np.pi * t), 0.0, None) ** 0.55
        taper = 0.25 + 0.75 * taper

        width = max(1.0, p.width_px)
        m = polyline_mask(shape, [(float(x), float(y)) for x, y in pts],
                          width=width, taper=taper, feather=p.feather)

        # 区域约束：裁剪到目标表面内；留存太少则判定失败重采样
        if region is not None:
            raw_pos = float((m > 0.05).sum())
            m = m * clip01(region)
            if float((m > 0.05).sum()) < 0.25 * max(raw_pos, 1.0):
                return None, 0.0, 0.0

        # 极性：暗线 or 亮线
        #   tone < -0.06 → 亮白线；tone > +0.06 → 暗黑线；|tone| 越大颜色越极端
        #   tone ≈ 0     → 按 dark_ratio 随机（保持旧行为）
        polarity, gain = self._polarity_gain(p, rng)
        if polarity < 0 and p.double_edge:
            # 暗线两侧加亮边（金属/镀膜反光）
            off = max(1.0, width * 0.9)
            tang = np.gradient(pts, axis=0)
            norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
            nlen = np.linalg.norm(norm, axis=1, keepdims=True) + 1e-6
            unit = norm / nlen
            for sign in (1.0, -1.0):
                sp = pts + unit * (off * sign)
                me = polyline_mask(shape, [(float(x), float(y)) for x, y in sp],
                                   width=max(1.0, width * 0.55), taper=taper,
                                   feather=p.feather * 1.4)
                me = me if region is None else me * clip01(region)
                m = np.maximum(m, me * 0.55)

        # 中心比边缘更实（横截面近似高斯）
        m = np.clip(m, 0.0, 1.0) ** 0.75
        return m, polarity, gain


# ==========================================================================
# 2. 脏污
# ==========================================================================
def _smoothstep(x: np.ndarray) -> np.ndarray:
    """smoothstep：一阶导在两端为 0，梯度变化平缓，不会出现生硬的折角。"""
    t = clip01(x)
    return t * t * (3.0 - 2.0 * t)


def _soft_blob_field(shape: Tuple[int, int], cx: float, cy: float, rad: float,
                     ang: float, stretch: float, rng: "Rng",
                     edge_soft: float = 0.6, irregular: float = 1.0,
                     grad_cells: Optional[int] = None) -> np.ndarray:
    """构造「软边斑块场」：中心接近 1，经**空间上真实存在的过渡带**衰减到 0。

    为什么不用旧的「模糊 + 全局阈值」：
    阈值 ``clip01((mask-thr)/(1-thr))`` 会把所有低于 thr 的像素直接归零，
    等于在 thr 处切出一圈**阶跃边界**——绘制区域越大，斑块越大，
    这圈硬边越显眼。这里改为纯空间域的径向衰减：
    ``alpha = smoothstep((1 - d) / band)``，band 由 ``edge_soft`` 控制，
    外缘的 alpha 本身就很低（<0.1），肉眼几乎看不出边界。

    edge_soft : 0=边界清晰（过渡带窄） … 1=几乎无边界（过渡带很宽）
    irregular : 分形噪声对边界的扰动强度（越大越不规则）
    """
    h, w = int(shape[0]), int(shape[1])
    rad = max(float(rad), 1e-3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    ca, sa = math.cos(ang), math.sin(ang)
    rx = dx * ca + dy * sa
    ry = -dx * sa + dy * ca
    d = np.sqrt((rx / max(stretch, 1e-3)) ** 2 + ry ** 2) / rad      # 归一化距离

    # 过渡带宽度：edge_soft 越大，带越宽 → 边界感越弱
    es = float(np.clip(edge_soft, 0.0, 1.0))
    band = 0.06 + 0.94 * (es ** 1.4)
    field = _smoothstep((1.0 - d) / max(band, 1e-3))

    if irregular > 1e-3:
        # 分形噪声扰动半径：让边界不规则（而不是一个规整椭圆）
        cells = grad_cells or max(3, int(min(h, w) / max(rad, 1.0) * 2.2))
        n = fractal_noise((h, w), rng, base_cells=max(2, cells),
                          octaves=4, persistence=0.55)
        # 噪声在过渡带内权重最大：实心区保持干净，边界才毛糙
        rim_w = np.clip(1.0 - np.abs(d - (1.0 - band * 0.5)) / max(band * 1.4, 1e-3), 0, 1)
        field = clip01(field * (1.0 - 0.30 * irregular)
                       + field * n * (0.62 * irregular) * rim_w
                       + 0.22 * irregular * rim_w * (n - 0.5) * 2.0 * 0.5)
    return clip01(field)


def _radius_for_coverage(coverage: float, base_short: float, stretch: float,
                         scale: float, frame_area: float,
                         k: float = 0.78) -> float:
    """按目标覆盖率反推斑块半径（避免用阈值硬切面积）。

    alpha>0.5 的等效面积 ≈ π·rad²·stretch·k（k 计入软边造成的内缩）。
    """
    cov = float(np.clip(coverage, 0.0005, 0.9))
    area = max(cov * max(frame_area, 1.0), 4.0)
    return max(2.0, math.sqrt(area / max(math.pi * max(stretch, 1.0) * k, 1e-3)))


def _region_inset(region: np.ndarray) -> np.ndarray:
    """区域内部到边界的距离场（0 = 边界，向内递增）。"""
    try:
        import cv2
        m = (np.asarray(region, np.float32) > 0.5).astype(np.uint8)
        if m.max() <= 0:
            return np.zeros_like(m, np.float32)
        return cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
    except Exception:
        return (np.asarray(region, np.float32) > 0.5).astype(np.float32)


def _fit_blob_in_region(region: np.ndarray, rng: "Rng",
                        want_rad: float, stretch: float) -> Tuple[float, float, float]:
    """在手绘区域内挑一个「离边界足够远」的斑块中心，并返回 (cx, cy, rad)。

    目的：避免大斑块被手绘区域边界**笔直切开**——那条直线正是
    「区域画得越大、边界感越明显」的主要来源。
    做法：只在「到边界距离 ≥ 0.75·rad」的像素里采样；若区域内没有
    这么深的点，就自动把半径压到区域内最大内切深度，缺陷仍能生成。
    """
    inset = _region_inset(region)
    rad = float(max(want_rad, 2.0))
    need = 0.75 * rad
    ys, xs = np.nonzero(inset >= need)
    if xs.size >= 8:
        i = rng.randint(0, int(xs.size) - 1)
        return float(xs[i]), float(ys[i]), rad
    deep = float(inset.max())
    if deep <= 2.0:                       # 区域太薄，退回普通采样
        cx, cy = sample_point(region, rng, margin=0.04)
        return cx, cy, max(2.0, min(rad, deep))
    rad = min(rad, deep / 0.75)            # 压到刚好放得下
    need = 0.75 * rad
    ys, xs = np.nonzero(inset >= need)
    if xs.size < 4:
        cx, cy = sample_point(region, rng, margin=0.04)
        return cx, cy, max(2.0, rad)
    i = rng.randint(0, int(xs.size) - 1)
    return float(xs[i]), float(ys[i]), rad


def _feather_width_for(region: np.ndarray, h: int, w: int) -> float:
    """区域向内羽化的像素宽度。

    ⚠️ 必须相对**区域自身**尺寸来定，不能相对整幅图：
    增强模式给每个缺陷的是一个「局部小椭圆」（半径可能只有二三十像素），
    若按 ``0.06*min(h,w)``（上千像素的图就是 ~28px）去腐蚀，
    小椭圆会被**整个抹掉**，缺陷一个都生成不出来。
    """
    short = _bbox_short_side(region) if region is not None else 0.0
    if short <= 0:
        return 0.0
    return float(np.clip(0.10 * short, 1.5, 10.0))


def _feather_region(region: np.ndarray, width: float) -> np.ndarray:
    """把手绘区域掩膜**向内**羽化：区域边界处为 0，内部渐增到 1。

    作用：``mask * feathered`` 在区域外严格为 0（不越界），
    但不会在区域边界留下一条笔直的硬切痕；越靠近边界 alpha 越低，
    视觉上自然消失。
    """
    if width <= 0.4:
        return clip01(region)
    try:
        import cv2
        m = (np.asarray(region, np.float32) > 0.5).astype(np.uint8)
        if m.max() <= 0:
            return np.zeros_like(m, np.float32)
        k = max(1, int(round(width)))
        er = cv2.erode(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        f = cv2.GaussianBlur(er.astype(np.float32), (0, 0), max(0.8, width * 0.55))
        return clip01(f * clip01(region))
    except Exception:
        return clip01(region)

@dataclass
class SmudgeParams:
    blob_count: int = 1             # 斑块数量
    blob_scale: float = 0.30        # 斑块尺寸相对短边的比例
    coverage: float = 0.35          # 覆盖面积大致占比（影响阈值）
    dirt_color: str = "auto"        # auto / dark / dust / oil / rust
    opacity: float = 0.55           # 最大不透明度
    granular: float = 0.35          # 颗粒感强度
    speckle: bool = True            # 是否叠加细小飞溅点
    edge_soft: float = 1.0          # 边缘模糊强度（相对斑块半径的 sigma 系数）


_DIRT_PALETTE = {
    "dark": (0.10, 0.09, 0.08),     # 深色污渍
    "dust": (0.62, 0.58, 0.52),     # 灰尘
    "oil": (0.22, 0.19, 0.12),      # 油污（偏黄褐）
    "rust": (0.42, 0.22, 0.10),     # 锈迹（橙褐）
}


class SmudgeDefect(BaseDefect):
    kind = "smudge"

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None, hint=None) -> DefectResult:
        h, w = rgb.shape[:2]
        s = self.strength
        sv = self.severity_mult
        short = min(h, w)
        frame_area = float(h * w)
        if region is not None:
            short = int(max(np.sqrt(max((region > 0.5).sum(), 1.0)) * 1.6, 24))
            frame_area = float(max((region > 0.5).sum(), 1.0))

        blob_count = max(1, int(self.params.get("blob_count", 1)))
        blob_scale = self.params.get("blob_scale", 0.30) * (0.55 + 0.45 * sv)
        opacity = float(np.clip(self.params.get("opacity", 0.55) * (0.55 + 0.65 * s)
                                * (0.30 + 0.70 * sv), 0.0, 0.95))
        granular = self.params.get("granular", 0.35) * (0.4 + s) * (0.5 + 0.5 * sv)
        speckle = self.params.get("speckle", True)
        # edge_soft 归一到 0~1：0=边界清晰，1=几乎看不出边界（默认偏软）
        edge_soft = float(np.clip(self.params.get("edge_soft", 1.0) / 2.5, 0.0, 1.0))
        # 覆盖率不再用「全局阈值」实现（阈值会切出可见硬边），
        # 改为按目标覆盖率反推每个斑块的半径。
        cov = float(np.clip(self.params.get("coverage", 0.35) * (0.6 + 0.9 * s)
                            * (0.5 + 0.5 * sv), 0.02, 0.85))

        mask = np.zeros((h, w), dtype=np.float32)
        radii = []
        # 区域向内羽化：避免斑块被手绘区域的直边「切」出一条硬边
        reg_f = (_feather_region(region, _feather_width_for(region, h, w))
                 if region is not None else None)
        for i in range(blob_count):
            ang = rng.uniform(0, np.pi)
            stretch = rng.uniform(1.0, 2.2)
            # 多个斑块时每个只占覆盖率的一部分
            share = cov / float(blob_count)
            rad = _radius_for_coverage(share, short, stretch, blob_scale, frame_area)
            # 斑块尺寸滑条仍作为「相对倍率」生效，但限制在合理范围内，
            # 避免在大区域上把单个斑块撑得过大
            rad *= float(np.clip(blob_scale / 0.30, 0.35, 2.2))
            rad = float(np.clip(rad, 3.0, max(6.0, short * 0.9)))
            if region is not None:
                cx, cy, rad = _fit_blob_in_region(region, rng, rad, stretch)
            else:
                cx = rng.uniform(0.08 * w, 0.92 * w)
                cy = rng.uniform(0.08 * h, 0.92 * h)
            radii.append(rad)
            blob = _soft_blob_field((h, w), cx, cy, rad, ang, stretch, rng,
                                    edge_soft=edge_soft,
                                    irregular=float(np.clip(0.35 + 0.5 * blob_scale
                                                           + 0.35 * s, 0.0, 1.3)))
            mask = np.maximum(mask, blob)

        if speckle:
            sp = self._speckle((h, w), rng, short,
                               density=0.35 * (0.5 + s) * (0.4 + 0.6 * sv),
                               region=region)
            rad_avg = float(np.mean(radii)) if radii else float(short * blob_scale)
            mask = np.maximum(mask, blur_mask(sp, max(0.8, rad_avg * 0.08)))
        # 区域约束：污渍只留在目标表面（用羽化后的区域，边界处自然消失）
        if reg_f is not None:
            mask = mask * reg_f
        mask = np.clip(mask, 0.0, 1.0)
        if mask.max() <= 0:
            return DefectResult(image=rgb.copy(), mask=mask, kind=self.kind, count=0)

        # 污染色
        mode = str(self.params.get("dirt_color", "auto")).lower()
        if mode == "auto":
            mean_rgb = rgb.reshape(-1, 3).mean(axis=0) / 255.0
            # 深色产品上偏浅灰污，浅色产品上偏深色污
            mode = "dust" if mean_rgb.mean() < 0.45 else "dark"
        dirt = np.array(_DIRT_PALETTE.get(mode, _DIRT_PALETTE["dark"]), dtype=np.float32)
        dirt = np.clip(dirt * rng.uniform(0.85, 1.15), 0.0, 1.0)

        base = rgb.astype(np.float32) / 255.0
        overlay = np.broadcast_to(dirt[None, None, :], (h, w, 3)).copy()

        # 颗粒质感：让脏污区域内部有明暗斑驳
        grain = fractal_noise((h, w), rng, base_cells=42, octaves=4, persistence=0.62)
        g = (grain - 0.5) * 2.0
        overlay = clip01(overlay + (g * granular * 0.35)[:, :, None])

        alpha = np.clip(mask * opacity, 0.0, 1.0)
        # 边缘羽化：中心浓、边缘淡（指数 >1 让过渡带更缓，避免边界感）
        alpha = alpha ** 1.10
        out = blend(base, overlay, alpha)

        # 脏污会轻微降低局部对比度（覆盖了一层污染物）
        local = out * (1.0 - alpha[:, :, None] * 0.10) + \
            np.clip(base * 0.5 + 0.5, 0, 1) * (alpha[:, :, None] * 0.10)
        out = clip01(local)

        return DefectResult(image=to_u8(out), mask=mask, kind=self.kind,
                            count=blob_count)

    def _speckle(self, shape, rng: Rng, short: float, density: float,
                 region: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = shape
        out = np.zeros((h, w), dtype=np.float32)
        n = max(1, int(120 * density))
        for _ in range(n):
            if region is not None:
                cx, cy = sample_point(region, rng, margin=0.01)
                if cx == 0.0 and cy == 0.0:
                    continue
            else:
                cx = rng.uniform(0, w)
                cy = rng.uniform(0, h)
            r = rng.uniform(0.4, max(1.4, short * 0.013))
            out = np.maximum(out, filled_circle_mask((h, w), cx, cy, r, feather=0.9))
        return out


# ==========================================================================
# 3. 异色
# ==========================================================================
@dataclass
class DiscolorParams:
    region_count: int = 1
    region_scale: float = 0.35
    hue_shift: float = 22.0     # a/b 通道偏移幅度
    lum_shift: float = -8.0     # L 通道偏移
    sat_gain: float = 1.25      # 色度增益
    edge_soft: float = 0.45     # 边缘过渡宽度（相对区域）
    edge_blur: float = 1.0      # 边缘模糊强度（相对区域半径的 sigma 系数）

_COLOR_CASTS = [
    ("泛黄", (6.0, 14.0, 30.0)),
    ("泛蓝", (-8.0, -6.0, 34.0)),
    ("泛绿", (-6.0, -26.0, 6.0)),
    ("泛红", (8.0, 38.0, 8.0)),
    ("泛紫", (2.0, 26.0, -22.0)),
    ("褪色", (10.0, -10.0, -16.0)),
]

# 异色「指定色调」预设：A/B 通道相对 128 中心的偏移（LAB：A+ = 偏红、A- = 偏绿；
# B+ = 偏黄、B- = 偏蓝；L 由明暗偏移/自定义目标决定）。方向按标准 Lab 语义标定，
# 让下拉里「泛黄/泛蓝/泛绿/泛红/泛紫」字面与实际色偏一致。
_DISCOLOR_PRESETS = {
    "yellow": (8.0, 30.0),     # 泛黄：偏黄（B+）
    "blue":   (-4.0, -32.0),   # 泛蓝：偏蓝（B-）
    "green":  (-34.0, 6.0),    # 泛绿：偏绿（A-）
    "red":    (40.0, 8.0),     # 泛红：偏红（A+）
    "purple": (24.0, -22.0),   # 泛紫：偏红+偏蓝（A+ B-）
    "fade":   (-10.0, -14.0),  # 褪色：去饱和（A/B 向 128 收缩）
}


def _is_rgb_triple(v) -> bool:
    """判断 v 是否为 (r,g,b) 三元组（0~255）。"""
    return (isinstance(v, (list, tuple)) and len(v) == 3
            and all(isinstance(x, (int, float)) for x in v))


def _discolor_target_cast(rgb: np.ndarray, mask: np.ndarray,
                          rgb_target, rng: "Rng") -> Tuple[float, float, float]:
    """把掩膜内平均 LAB 推向目标 RGB 对应的 LAB，返回 (cast_a, cast_b, lum)。

    这样用户挑出的任意 RGB 颜色都会成为该异色区域的「平均目标色」，
    而非简单的整体叠色，避免出现「贴纸感」。
    """
    t = np.asarray(rgb_target, dtype=np.uint8).reshape(1, 1, 3)
    try:
        tlab = rgb_to_lab(t)[0, 0]
    except Exception:
        return 0.0, 0.0, 0.0
    m = mask > 0.02
    if int(m.sum()) == 0:
        return 0.0, 0.0, 0.0
    src = rgb_to_lab(rgb)
    w = mask[m]
    sa = float(np.average(src[m, 1], weights=w))
    sb = float(np.average(src[m, 2], weights=w))
    sl = float(np.average(src[m, 0], weights=w))
    j = float(rng.uniform(0.92, 1.08))  # 轻微抖动，避免多次生成完全一致
    ca = float((tlab[1] - sa) * j)
    cb = float((tlab[2] - sb) * j)
    cl = float((tlab[0] - sl) * j)
    return ca, cb, cl


class DiscolorDefect(BaseDefect):
    kind = "discolor"

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None, hint=None) -> DefectResult:
        h, w = rgb.shape[:2]
        s = self.strength
        sv = self.severity_mult
        # 异色区尺寸以「产品外接框短边」为基准（与缺角一致的口径）：
        # 若沿用 sqrt(面积)*1.6，斑块会几乎吞掉整个产品，面积对明显度就不敏感了。
        short = min(h, w)
        if region is not None:
            short = int(max(_bbox_short_side(region), 24))

        region_count = max(1, int(self.params.get("region_count", 1)))
        region_scale = self.params.get("region_scale", 0.35) * (0.55 + 0.45 * sv)
        # edge_soft 直接作为「边界感」：0=边界清晰 … 1=几乎无边界
        edge_soft = float(np.clip(self.params.get("edge_soft", 0.75), 0.0, 1.0))
        # 覆盖率用于反推半径（不再用全局阈值硬切，避免可见等值线）
        cov = float(np.clip(0.06 + 0.30 * s, 0.02, 0.6))
        frame_area = float(h * w)
        if region is not None:
            frame_area = float(max((region > 0.5).sum(), 1.0))

        mask = np.zeros((h, w), dtype=np.float32)
        radii = []
        # 区域向内羽化：避免异色块被手绘区域的直边切出硬边
        reg_f = (_feather_region(region, _feather_width_for(region, h, w))
                 if region is not None else None)
        for _ in range(region_count):
            ang = rng.uniform(0, np.pi)
            stretch = rng.uniform(1.0, 2.4)
            share = cov / float(region_count)
            rad = _radius_for_coverage(share, short, stretch, region_scale, frame_area)
            rad *= float(np.clip(region_scale / 0.35, 0.35, 2.2))
            rad = float(np.clip(rad, 4.0, max(8.0, short * 0.95)))
            if region is not None:
                cx, cy, rad = _fit_blob_in_region(region, rng, rad, stretch)
            else:
                cx = rng.uniform(0.10 * w, 0.90 * w)
                cy = rng.uniform(0.10 * h, 0.90 * h)
            radii.append(rad)
            blob = _soft_blob_field((h, w), cx, cy, rad, ang, stretch, rng,
                                    edge_soft=edge_soft,
                                    irregular=float(np.clip(0.30 + 0.45 * region_scale
                                                           + 0.30 * s, 0.0, 1.2)))
            mask = np.maximum(mask, blob)

        mask = clip01(mask)
        # 区域约束：异色只留在目标表面（羽化后边界处自然消失，且严格不越界）
        if reg_f is not None:
            mask = mask * reg_f
        if mask.max() <= 0:
            return DefectResult(image=rgb.copy(), mask=mask, kind=self.kind, count=0)

        # 颜色设置：随机 / 预设色调 / 自定义目标色
        color_mode = str(self.params.get("color", "random")).lower()
        custom_rgb = self.params.get("color_rgb", None)

        if color_mode == "custom" and _is_rgb_triple(custom_rgb):
            # 自定义目标色：把掩膜内平均色推向目标 RGB（色度增益归 1，颜色精准还原）
            cast_a, cast_b, lum = _discolor_target_cast(rgb, mask, custom_rgb, rng)
            cast = np.array([0.0, cast_a, cast_b], dtype=np.float32)
            sat_gain = 1.0
        else:
            if color_mode in _DISCOLOR_PRESETS:
                da, db = _DISCOLOR_PRESETS[color_mode]
                cast = np.array([0.0, da, db], dtype=np.float32)
            else:  # random
                _, cast = rng.choice(_COLOR_CASTS)
                cast = np.array(cast, dtype=np.float32)
            cast = cast * (0.35 + 0.95 * s) * (0.25 + 0.75 * sv)
            lum = self.params.get("lum_shift", -8.0) * (0.4 + 1.0 * s) * (0.25 + 0.75 * sv)
            sat_gain = 1.0 + (self.params.get("sat_gain", 1.25) - 1.0) * (0.35 + 0.9 * s) \
                * (0.3 + 0.7 * sv)

        lab = rgb_to_lab(rgb)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        L2 = L + (lum * mask)
        A2 = 128.0 + (A - 128.0) * sat_gain + cast[1] * mask
        B2 = 128.0 + (B - 128.0) * sat_gain + cast[2] * mask

        # 边缘/内部不均匀：叠加低频噪声，避免出现「贴纸」感
        uneven = (fractal_noise((h, w), rng, base_cells=10, octaves=3) - 0.5) * 2.0
        A2 = A2 + uneven * 6.0 * mask
        B2 = B2 + uneven * 6.0 * mask

        new_lab = np.stack([L2, A2, B2], axis=2)
        recolored = lab_to_rgb(new_lab)

        # 区域外保持原样，区域边缘做渐晕过渡
        a = clip01(mask)[:, :, None]
        out = np.clip(rgb.astype(np.float32) * (1 - a) + recolored.astype(np.float32) * a,
                      0, 255).astype(np.uint8)

        return DefectResult(image=out, mask=mask, kind=self.kind, count=region_count)


# ==========================================================================
# 4. 崩缺（边沿崩缺 / 缺损）
# ==========================================================================
@dataclass
class ChipParams:
    count: int = 1                 # 崩缺个数
    size_ratio: float = 0.22       # 崩缺「沿边沿」长度相对产品短边的比例
    jag_ratio: float = 0.45        # 断口边缘的锯齿/崩缺程度
    depth_ratio: float = 0.30      # 崩缺深度相对局部半厚度（浅！实际崩缺很浅）
    rim_width: float = 2.0         # 断口内侧暗边宽度（像素）
    crack_ratio: float = 0.30      # 从崩缺延伸出的裂纹长度比
    grain: float = 0.45            # 断面颗粒感
    darkness: float = 0.32         # 崩缺内壁变暗程度
    edge_soft: float = 0.45        # 边界柔和度（0=断口边界清晰 … 1=平滑过渡）
    color_match: float = 0.65      # 断面颜色接近度（1=与邻近区域颜色几乎一致）
    debris: bool = True            # 是否产生细小碎屑
    corner_bias: float = 0.35      # 落在转角处的概率（转角更易磕碰）


class ChipDefect(BaseDefect):
    """崩缺：产品**边沿**被磕掉一小块（实际缺陷多为边沿浅层缺损，而非整角崩落）。

    物理表现：
    1. 位置 —— 沿产品轮廓**任意位置**取点（转角处概率更高），不再是「只啃角点」；
    2. 形状 —— 沿边沿呈月牙状浅缺口：跨度沿切向、深度沿内法线，
       深度受**局部半厚度**约束（浅缺损不会贯穿产品）；
    3. 断面 —— 新鲜断面略亮、粗糙（陶瓷颗粒感），带多频角向抖动形成崩口锯齿；
    4. 交界 —— 原轮廓线上有一圈细微暗边（材料崩落后的阴影）；
    5. 碎屑 —— 缺口外侧（背景侧）散落细小崩渣；
    6. 可选 —— 自缺口向内延伸短裂纹。
    """

    kind = "chip"

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None, hint=None) -> DefectResult:
        import cv2

        h, w = rgb.shape[:2]
        s = self.strength
        sv = self.severity_mult
        base = rgb.astype(np.float32) / 255.0

        reg = clip01(region) if region is not None else np.ones((h, w), np.float32)
        binm = (reg > 0.5).astype(np.uint8)
        if int(binm.sum()) < 64:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)

        # ---- 距离场：0 在轮廓上，向内递增；梯度方向即「内法线」 ----
        dt = cv2.distanceTransform(binm, cv2.DIST_L2, 5).astype(np.float32)
        gy, gx = np.gradient(dt)
        gn = np.hypot(gx, gy)
        inside = binm > 0
        if not inside.any():
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        # 局部半厚度：崩缺深度以它为基准，保证「浅」且不会贯穿
        half_t = float(np.percentile(dt[inside], 90.0))
        if half_t < 2.0:
            half_t = 2.0
        prod_short = float(max(math.sqrt(float(binm.sum())), 12.0))

        # ---- 候选落点：轮廓上的像素 ----
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        bpts = np.vstack(cnts).reshape(-1, 2).astype(np.float32)
        if bpts.shape[0] < 8:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)

        count = max(1, int(self.params.get("count", 1)))
        size_ratio = float(self.params.get("size_ratio", 0.22)) \
            * (0.55 + 0.55 * s) * (0.6 + 0.4 * sv)
        jag = float(self.params.get("jag_ratio", 0.45))
        depth_ratio = float(self.params.get("depth_ratio", 0.30)) \
            * (0.5 + 0.7 * s) * (0.6 + 0.4 * sv)
        corner_bias = float(np.clip(self.params.get("corner_bias", 0.35), 0.0, 1.0))
        edge_soft = float(np.clip(self.params.get("edge_soft", 0.45), 0.0, 1.0))
        color_match = float(np.clip(self.params.get("color_match", 0.65), 0.0, 1.0))

        mask = np.zeros((h, w), dtype=np.float32)
        out = base.copy()
        n_made = 0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        used: List[Tuple[float, float, float]] = []   # (cx, cy, 半径) 避免叠在一起

        for _ in range(count):
            # 落点：默认按曲率（凸角）加权；给了 hint 就优先选**离锚点最近**的一段边沿
            idx = self._pick_edge_point(bpts, rng, corner_bias, hint)
            if idx is None:
                break
            cx, cy = float(bpts[idx, 0]), float(bpts[idx, 1])

            # 该点的内法线（距离场梯度方向）
            ix = int(np.clip(cx, 0, w - 1))
            iy = int(np.clip(cy, 0, h - 1))
            nx, ny = float(gx[iy, ix]), float(gy[iy, ix])
            nrm = math.hypot(nx, ny)
            if nrm < 1e-3:
                continue
            n_in = np.array([nx / nrm, ny / nrm], np.float32)     # 指向产品内部
            tan = np.array([-n_in[1], n_in[0]], np.float32)

            # 缺口尺寸：跨度沿边沿（较明显），深度浅（受局部半厚度限制）
            half = max(2.0, prod_short * float(np.clip(size_ratio, 0.02, 0.6))
                       * rng.uniform(0.7, 1.35))
            local_t = max(2.0, float(dt[iy, ix]) + half * 0.5)
            depth_len = float(np.clip(half * depth_ratio * rng.uniform(0.7, 1.3),
                                      1.2, local_t * 0.75))

            # 与已生成的缺口保持距离
            if any(math.hypot(cx - ux, cy - uy) < (half + ur) * 0.85
                   for ux, uy, ur in used):
                continue

            dd = (xx - cx) * n_in[0] + (yy - cy) * n_in[1]   # 沿内法线（向内为正）
            tt = (xx - cx) * tan[0] + (yy - cy) * tan[1]     # 沿边沿

            # 横向剖面：中间最深、两端收口成尖（典型崩口形状）
            span = clip01(1.0 - (tt / max(half, 1e-6)) ** 2)
            span = span * span * (3.0 - 2.0 * span)          # smoothstep，收口更自然

            # 深度轮廓：多频抖动 + 分形噪声 → 崩口参差不齐
            ph = rng.uniform(0, 6.283)
            wob = (np.sin(tt / max(half, 1e-6) * rng.uniform(2.0, 3.6) + ph) * 0.55
                   + np.sin(tt / max(half, 1e-6) * rng.uniform(5.0, 9.0) + ph * 1.7) * 0.30
                   + np.sin(tt / max(half, 1e-6) * rng.uniform(11.0, 17.0) + ph * 0.6) * 0.15)
            enoise = fractal_noise((h, w), rng, base_cells=max(4, int(half * 0.9)),
                                   octaves=4, persistence=0.62)
            reach = depth_len * (1.0 + jag * wob * 0.34
                                 + jag * (enoise - 0.5) * 0.85)
            reach = np.maximum(reach, 0.8)

            # 缺口本体：从轮廓向内啃进 reach 处。
            # 过渡带宽度由「边界柔和度」控制：越大断口边缘越平滑、
            # 与周围材料的衔接越自然（不再是一圈生硬的边界线）
            soft = max(1.0, depth_len * (0.10 + 0.55 * edge_soft))
            bite = clip01((reach - dd) / soft) * span
            bite = clip01(bite) * clip01(reg)
            if bite.max() <= 0.06 or int((bite > 0.5).sum()) < 10:
                continue
            used.append((cx, cy, half))
            n_made += 1

            # ---- 断面材质：颜色与邻近区域接近 + 粗糙颗粒感 ----
            solid = (bite > 0.5).astype(np.uint8)
            depth_w = clip01(1.0 - dd / max(float(np.percentile(reach, 99.0)), 1e-6))
            lum = out.mean(axis=2, keepdims=True)
            grain_amt = float(self.params.get("grain", 0.45)) * (0.4 + 0.7 * s) * (0.4 + 0.6 * sv)
            grain = fractal_noise((h, w), rng, base_cells=max(6, int(half * 1.1)),
                                  octaves=4, persistence=0.62)
            # 模型A（旧）：断口比原表面略亮（+3%~11%），靠粗糙感与暗边区分
            frac_old = clip01(lum * rng.uniform(1.03, 1.11)
                              + (grain - 0.5)[:, :, None] * grain_amt * 0.45)
            # 模型B（新）：取崩口周围环带的平均色 → 断面颜色与邻近区域几乎
            # 一致（真实崩缺露出的是同一块材料），只保留轻微颗粒起伏
            r2 = max(3, int(half * 0.6))
            near = (cv2.dilate(solid, np.ones((2 * r2 + 1, 2 * r2 + 1), np.uint8)) > 0)
            sel = near & (reg > 0.5) & (bite <= 0.05)
            if int(sel.sum()) >= 32:
                surr = base[sel].mean(axis=0)
            elif (reg > 0.5).any():
                surr = base[reg > 0.5].mean(axis=0)
            else:
                surr = np.array([0.7, 0.7, 0.7], np.float32)
            frac_near = clip01(surr[None, None, :]
                               + (grain - 0.5)[:, :, None] * grain_amt * 0.30)
            # 「颜色接近度」在两个模型间插值：1=完全跟随邻近区域颜色
            fracture = clip01(frac_old * (1.0 - color_match) + frac_near * color_match)
            dark = float(self.params.get("darkness", 0.32)) * (0.4 + 0.8 * s) * (0.4 + 0.6 * sv)
            fracture = clip01(fracture - dark * depth_w[:, :, None] * 0.34)
            # 露出未烧结体，断面略偏灰褐
            fracture = clip01(fracture * np.array([1.012, 0.996, 0.962], np.float32)[None, None, :])
            a_chip = bite[:, :, None]
            out = out * (1 - a_chip) + fracture * a_chip

            # ---- 交界暗边：原轮廓线内侧一圈暗线（崩落后的阴影） ----
            k = max(1, int(round(float(self.params.get("rim_width", 2.0)))))
            solid = (bite > 0.5).astype(np.uint8)
            k3 = np.ones((3, 3), np.uint8)
            grown = cv2.dilate(solid, k3, iterations=k)
            rim = clip01((grown - solid).astype(np.float32)) * (1.0 - bite * 0.5)
            rim = clip01(rim) * clip01(reg)
            # 边界柔和度越大，交界暗边越淡（暗边也是「边界感」的来源之一）
            out = clip01(out * (1.0 - (0.42 - 0.18 * edge_soft) * rim[:, :, None]))

            # ---- 缺口外侧（背景侧）的细小崩渣 ----
            if self.params.get("debris", True):
                deb = np.zeros((h, w), np.float32)
                for _k in range(rng.randint(2, 6)):
                    a = rng.uniform(0, math.pi * 2)
                    rr = half * rng.uniform(0.25, 0.95)
                    dcx = cx - n_in[0] * rr * rng.uniform(0.1, 0.6) \
                        + tan[0] * math.cos(a) * rr * 0.7
                    dcy = cy - n_in[1] * rr * rng.uniform(0.1, 0.6) \
                        + tan[1] * math.cos(a) * rr * 0.7
                    r = max(0.5, half * rng.uniform(0.012, 0.042))
                    dm2 = filled_circle_mask((h, w), dcx, dcy, r, feather=0.6)
                    dn = fractal_noise((h, w), rng, base_cells=max(8, int(r * 6) + 8),
                                       octaves=2, persistence=0.5)
                    deb = np.maximum(deb, clip01(dm2 * (0.35 + 1.5 * (dn > 0.55))))
                dmask = clip01(deb) * (1.0 - clip01(reg))
                dcol = np.array([0.74, 0.72, 0.68], np.float32)
                out = clip01(out * (1 - dmask[:, :, None] * 0.8)
                             + dcol[None, None, :] * dmask[:, :, None] * 0.8)

            # ---- 自缺口向内延伸的短裂纹 ----
            cr = float(self.params.get("crack_ratio", 0.30)) * (0.3 + 0.9 * s) * (0.4 + 0.6 * sv)
            if cr > 0.02:
                from .utils import polyline_mask

                clen = half * rng.uniform(1.0, 2.4) * cr
                side = float(rng.choice([-1.0, 1.0]))
                sx = cx + n_in[0] * depth_len * 0.75 + tan[0] * side * half * 0.28
                sy = cy + n_in[1] * depth_len * 0.75 + tan[1] * side * half * 0.28
                a_dir = math.atan2(n_in[1], n_in[0]) + side * rng.uniform(0.30, 0.85)
                p0 = (sx, sy)
                p1 = (sx + math.cos(a_dir) * clen * 0.38 + tan[0] * side * clen * 0.20,
                      sy + math.sin(a_dir) * clen * 0.38 + tan[1] * side * clen * 0.20)
                p2 = (sx + math.cos(a_dir) * clen * 0.72 - tan[0] * side * clen * 0.10,
                      sy + math.sin(a_dir) * clen * 0.72 - tan[1] * side * clen * 0.10)
                p3 = (sx + math.cos(a_dir) * clen, sy + math.sin(a_dir) * clen)
                cm = polyline_mask((h, w), [p0, p1, p2, p3],
                                   width=max(0.6, half * rng.uniform(0.012, 0.030)),
                                   taper=[0.9, 0.65, 0.35, 0.04], feather=0.7)
                cm = clip01(cm) * clip01(reg)
                out = clip01(out * (1 - 0.40 * cm[:, :, None]))
                bite = np.maximum(bite, cm)

            mask = np.maximum(mask, bite)

        if n_made == 0 or mask.max() <= 0.02:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)

        return DefectResult(image=to_u8(out), mask=clip01(mask), kind=self.kind,
                            count=n_made)

    @staticmethod
    def _pick_edge_point(bpts: np.ndarray, rng: Rng, corner_bias: float,
                         hint=None) -> Optional[int]:
        """在轮廓点里挑一个落点。

        - 给了 ``hint=(ang, ax, ay)``（增强模式传入锚点）：只在**离锚点最近的
          20%** 轮廓点里按曲率加权挑，这样崩缺既贴着产品边沿、又不远离锚点；
        - 没给 hint：按曲率（凸角）加权，转角处概率更高，真实磕碰也多发生在棱边。
        """
        n = bpts.shape[0]
        if n == 0:
            return None
        pool = np.arange(n)
        if hint is not None and len(hint) >= 3:
            try:
                ax, ay = float(hint[1]), float(hint[2])
                d = np.hypot(bpts[:, 0] - ax, bpts[:, 1] - ay)
                k = max(4, int(n * 0.2))
                pool = np.argsort(d)[:k]
            except Exception:
                pool = np.arange(n)
        if pool.size == 0:
            pool = np.arange(n)
        # 用轮廓点的局部曲率近似：三点夹角越小越尖锐
        k = max(2, n // 220)
        prev = np.roll(bpts, k, axis=0)
        nxt = np.roll(bpts, -k, axis=0)
        v1 = bpts - prev
        v2 = nxt - bpts
        n1 = np.linalg.norm(v1, axis=1) + 1e-6
        n2 = np.linalg.norm(v2, axis=1) + 1e-6
        cosang = np.clip((v1 * v2).sum(1) / (n1 * n2), -1.0, 1.0)
        sharp = np.clip(1.0 - cosang, 0.0, 2.0)      # 越尖锐越大
        wts = (1.0 - corner_bias) + corner_bias * (sharp / max(sharp.max(), 1e-6))
        w = np.clip(wts[pool], 1e-6, None)
        tot = float(w.sum())
        if tot <= 1e-9:
            return int(pool[int(rng.randint(0, pool.size - 1))])
        r = rng.uniform(0.0, tot)
        acc = 0.0
        for i in range(pool.size):
            acc += float(w[i])
            if acc >= r:
                return int(pool[i])
        return int(pool[-1])

# ==========================================================================


def resolve_target(target: str, region_masks: Optional[dict]) -> Optional[np.ndarray]:
    """把 'body'/'electrode'/'both' 解析为一个联合软掩膜。"""
    if region_masks is None:
        return None
    parts = []
    if target in ("body", "both") and region_masks.get("body") is not None:
        parts.append(region_masks["body"])
    if target in ("electrode", "both") and region_masks.get("electrode") is not None:
        parts.append(region_masks["electrode"])
    if not parts:
        return None
    out = parts[0].copy()
    for p in parts[1:]:
        out = np.maximum(out, p)
    out = clip01(out)
    return out if out.max() > 0.1 else None


def apply_defects(rgb: np.ndarray, settings: dict, seed: Optional[int] = None,
                  region_masks: Optional[dict] = None, progress=None,
                  region_hint=None, scratch_paths=None):
    """按 settings 依次叠加多类缺陷。

    settings 形如::

        {
          "scratch":  {"enable": True, "strength": 0.6, "count": 6,
                       "target": "body", "severity": "obvious"},
          ...
        }

    region_masks: {"body": floatmask, "electrode": floatmask} 或 None。
                  提供时缺陷位置被约束在目标表面内。

    region_hint: 手绘区域的方向提示 ``(角度rad, cx, cy)``（可None）。
                 划痕会沿着这个方向生成，而不是随机方向。

    scratch_paths: 手绘线条的点列列表（``[[(x,y), ...], ...]``）。
                   非空时划痕**沿这些线**生成，起止位置与手绘一致；
                   为空时划痕按区域方向自由生成（批量流程）。

    返回 (image, applied) —— applied 是 [DefectResult, ...]，按应用顺序。
    """
    rng = Rng(seed)
    img = rgb.copy()
    applied: List[DefectResult] = []
    order = [k for k in DEFECT_KINDS if settings.get(k, {}).get("enable")]

    for i, kind in enumerate(order):
        cfg = dict(settings.get(kind, {}))
        strength = float(cfg.pop("strength", 0.5))
        cfg.pop("enable", None)
        target = str(cfg.pop("target", "both"))
        severity = str(cfg.pop("severity", "mixed"))
        lo, hi = SEVERITY_LEVELS.get(severity, (0.12, 1.0))
        sv = rng.uniform(lo, hi)
        region = resolve_target(target, region_masks)

        gen = DEFECT_REGISTRY.get(kind)
        if gen is None:
            continue
        kw = {}
        if kind == "scratch":
            kw["paths"] = scratch_paths
        res = gen(strength=strength, severity_mult=sv, **cfg).render(
            img, rng, region=region, hint=region_hint, **kw)
        res.kind = kind
        res.severity = severity
        res.surface = target
        img = res.image
        applied.append(res)
        if progress is not None:
            progress((i + 1) / max(len(order), 1))

    return img, applied


# ==========================================================================
# 5. 黑点（黑斑 / 针孔）
# ==========================================================================
@dataclass
class BlackDotParams:
    count: int = 1                 # 黑点个数
    size_ratio: float = 0.010      # 黑点半径相对产品短边的比例
    darkness: float = 0.88         # 颜色深度（1=中心达到纯色）
    cluster: float = 0.35          # 成簇出现的概率
    edge_soft: float = 0.65        # 边界柔和度（0=边界清晰 … 1=几乎无边界）
    color: str = "auto"            # auto=自动深色 / custom=自定义颜色
    color_rgb: tuple = (26, 22, 20)  # 黑点颜色（custom 模式生效）


class BlackDotDefect(BaseDefect):
    """黑点：陶瓷表面的黑色针孔/黑斑（烧结缺陷、污染点、金属异物灼点）。

    真实特征：
    - 形状是**不规则的圆/椭圆**（长宽比、朝向、边界均随机），
      边缘经软边场平滑衰减，没有一圈明显的边界线；
    - 手绘区域只**限定范围**（黑点落在区域内），不改变黑点自身形状；
    - 颜色支持自定义（取色），darkness 直接控制不透明度——
      调到 1 时黑点中心就是所选颜色本身；
    - 常成簇出现，而不是均匀散布。
    """

    kind = "dot"

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None, hint=None) -> DefectResult:
        h, w = rgb.shape[:2]
        s = self.strength
        sv = self.severity_mult
        base = rgb.astype(np.float32) / 255.0

        reg = clip01(region) if region is not None else np.ones((h, w), np.float32)
        ys, xs = np.nonzero(reg > 0.5)
        if xs.size < 16:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        short = float(min(h, w))
        # 区域**向内羽化**：黑点只被「限定范围」，即使中心贴近区域边界，
        # 也是平滑淡出，而不会留下一条笔直的区域切痕
        regf = _feather_region(reg, _feather_width_for(reg, h, w)) \
            if region is not None else reg

        count = max(1, int(self.params.get("count", 1)))
        size_ratio = float(self.params.get("size_ratio", 0.010)) * (0.5 + 0.9 * s) * (0.6 + 0.5 * sv)
        darkness = float(np.clip(self.params.get("darkness", 0.88), 0.2, 1.0))
        cluster = float(np.clip(self.params.get("cluster", 0.35), 0.0, 1.0))
        es = float(np.clip(self.params.get("edge_soft", 0.65), 0.0, 1.0))
        cm = str(self.params.get("color", "auto")).lower()
        crgb = self.params.get("color_rgb", (26, 22, 20))
        if not _is_rgb_triple(crgb):
            crgb = (26, 22, 20)
        if cm == "custom":
            dot_cols = [tuple(float(c) for c in crgb)]
        else:
            # 自动：几种接近黑的自然污染色（略带冷暖差异）
            dot_cols = [(22, 20, 18), (34, 30, 26), (16, 16, 18), (40, 34, 28)]

        dot_mask = np.zeros((h, w), dtype=np.float32)
        out = base.copy()
        n_made = 0
        clusters: List[Tuple[float, float]] = []

        for i in range(count):
            # 成簇：一部分点围绕已有簇生成
            if clusters and rng.bool(cluster):
                ax, ay = clusters[int(rng.randint(0, len(clusters) - 1))]
                a = rng.uniform(0, math.pi * 2)
                rr = rng.uniform(0.4, 3.2) * short * size_ratio
                cx = ax + math.cos(a) * rr
                cy = ay + math.sin(a) * rr
            else:
                if region is not None:
                    cx, cy = sample_point(region, rng, margin=0.03)
                else:
                    cx = rng.uniform(0.05 * w, 0.95 * w)
                    cy = rng.uniform(0.05 * h, 0.95 * h)
                clusters.append((cx, cy))
            rad = max(0.8, short * size_ratio * rng.uniform(0.55, 1.7))
            col = np.array(dot_cols[int(rng.randint(0, len(dot_cols) - 1))],
                           np.float32) / 255.0
            # 不规则的圆/椭圆：长宽比与朝向随机，边界不规则度与柔和度可调
            blob = _soft_blob_field((h, w), cx, cy, rad, rng.uniform(0, math.pi),
                                    rng.uniform(0.80, 1.45), rng,
                                    edge_soft=es, irregular=0.55 + 0.35 * es)
            blob = clip01(blob) * regf
            if blob.max() <= 0.08:
                continue
            n_made += 1
            # darkness 直接控制不透明度：调到 1 时中心就是所选纯色；
            # 边缘 alpha 随软边场自然衰减，不会有硬边
            halo = np.clip(blur_mask(blob, max(1.0, rad * 0.9)) - blob, 0, 1)
            tot = clip01(blob * darkness
                         + halo * 0.30 * (1.0 - darkness * 0.6))[:, :, None]
            out = out * (1.0 - tot) + col[None, None, :] * tot
            dot_mask = np.maximum(dot_mask, blob)

        if n_made == 0 or dot_mask.max() <= 0.02:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        return DefectResult(image=to_u8(out), mask=clip01(dot_mask), kind=self.kind,
                            count=n_made)


# ==========================================================================
# 6. 凹坑（麻点 / 缩孔）
# ==========================================================================
@dataclass
class PitParams:
    count: int = 1                 # 凹坑个数
    size_ratio: float = 0.011      # 凹坑半径相对产品短边
    depth: float = 0.55            # 凹陷深度（影响明暗对比）
    highlight: float = 0.55        # 坑内高光强度
    light_deg: float = 135.0       # 光照方向（度，0=从右）


class PitDefect(BaseDefect):
    """凹坑：陶瓷表面的麻点/缩孔（烧结收缩形成的小凹洞）。

    真实特征（**立体感来自方向性明暗**，这是与"黑点"的本质区别）：
    - 「凹」的高度场是负的（中心低、边缘高）→ 法线朝坑心倾斜：
      迎光侧内壁背光变暗、**背光侧内壁被照亮**——这正是凹的读法；
      若极性反了（法线向外倾），看起来就像鼓包凸起；
    - 坑中心偏暗（光进不去）；
    - 边缘轮廓不规则（噪声只作用在轮廓环带，坑内高度场保持光滑）。
    """

    kind = "pit"

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None, hint=None) -> DefectResult:
        h, w = rgb.shape[:2]
        s = self.strength
        sv = self.severity_mult
        base = rgb.astype(np.float32) / 255.0

        reg = clip01(region) if region is not None else np.ones((h, w), np.float32)
        ys, xs = np.nonzero(reg > 0.5)
        if xs.size < 16:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        short = float(min(h, w))

        count = max(1, int(self.params.get("count", 1)))
        size_ratio = float(self.params.get("size_ratio", 0.011)) * (0.5 + 0.9 * s) * (0.6 + 0.5 * sv)
        depth = float(np.clip(self.params.get("depth", 0.55), 0.05, 1.0))
        hi_amt = float(np.clip(self.params.get("highlight", 0.55), 0.0, 1.5))
        light_deg = float(self.params.get("light_deg", 135.0))
        th = math.radians(light_deg)
        lx, ly = math.cos(th), math.sin(th)      # 光照方向（图像坐标，y 向下）

        out = base.copy()
        mask = np.zeros((h, w), np.float32)
        n_made = 0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        for _ in range(count):
            if region is not None:
                cx, cy = sample_point(region, rng, margin=0.03)
            else:
                cx = rng.uniform(0.05 * w, 0.95 * w)
                cy = rng.uniform(0.05 * h, 0.95 * h)
            rad = max(1.2, short * size_ratio * rng.uniform(0.6, 1.6))
            # ---- 高度场（用于明暗着色）必须**光滑** ----
            # 凹坑是规则的内凹球冠，若把分形噪声加进高度场，梯度求出的法线会
            # 到处乱转，着色出现漩涡/尖刺状伪影。这里噪声只作用于「轮廓掩膜」，
            # 高度场保持干净，着色才有正确的方向性明暗。
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(rad, 1e-6)
            cap = np.clip(1.0 - d * d, 0.0, 1.0)          # 平滑球冠
            if cap.max() <= 0.05:
                continue

            # 法向（由光滑球冠的梯度得到）→ Lambert 着色。
            # ⚠️ 凹坑高度场取**负**（中心低、边缘高）→ 法线朝坑心倾斜：
            # 迎光侧内壁背光变暗、背光侧内壁被照亮 —— 这才是「凹」的明暗。
            # （若取正高度场，法线向外倾，视觉上读成鼓包凸起。）
            hy, hx = np.gradient(-cap * rad)
            nlen = np.sqrt(hx * hx + hy * hy + 1.0)
            nx_ = -hx / nlen
            ny_ = -hy / nlen
            nz_ = 1.0 / nlen
            lam = clip01(nx_ * lx + ny_ * ly + nz_ * 0.88)      # 0..1
            # 凹坑只要「一侧亮弧 + 一侧暗弧」这一条信息。
            # 不要再叠加独立的 rimband 高光环 —— 它与镜面高光弧位置接近，
            # 两条弧叠在一起会形成「逗号/波浪」状的假纹。
            shade = 1.0 - depth * (1.0 - lam) * 1.20             # 背光侧变暗
            shade = shade + hi_amt * 0.26 * np.power(clip01(lam), 6.0)   # 单一亮弧
            shade = np.where(d < 1.0, shade, 1.0)                # 坑外不受影响
            shade = np.clip(shade, 0.35, 1.55)

            # ---- 轮廓掩膜：只在边缘环带加噪声，让坑口不规则但内部干净 ----
            n = fractal_noise((h, w), rng, base_cells=max(4, int(rad * 0.8)),
                              octaves=3, persistence=0.5)
            ring = np.clip(1.0 - np.abs(d - 0.92) / 0.22, 0, 1)   # 边缘环带权重
            rmod = d * (1.0 + 0.30 * (n - 0.5) * ring)             # 半径被噪声扰动
            dome = np.clip(1.0 - rmod * rmod, 0.0, 1.0)
            if dome.max() <= 0.05:
                continue
            n_made += 1

            a = clip01(dome * np.clip(0.78 + 0.3 * s, 0, 1))[:, :, None]
            out = clip01(out * (1 - a) + out * shade[:, :, None] * a)
            mask = np.maximum(mask, dome * clip01(reg))

        if n_made == 0 or mask.max() <= 0.02:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        return DefectResult(image=to_u8(out), mask=clip01(mask), kind=self.kind,
                            count=n_made)


# ==========================================================================
# 7. 异物（粘附物 / 灰尘纤维 / 碎屑）
# ==========================================================================
@dataclass
class DebrisParams:
    count: int = 1                 # 异物个数
    size_ratio: float = 0.013      # 异物尺寸相对产品短边
    shadow: float = 0.45           # 投影强度
    gloss: float = 0.35            # 高光强度
    fiber_ratio: float = 0.35      # 呈纤维/毛发状的比例
    color_mode: str = "auto"       # auto / dark / light / metal / fiber


_DEBRIS_COLORS = {
    "dark": (0.13, 0.12, 0.11),     # 深色颗粒（最常见）
    "light": (0.72, 0.70, 0.64),    # 浅色碎屑（纸屑/灰尘）
    "metal": (0.55, 0.57, 0.61),    # 金属屑（带冷调）
    "fiber": (0.40, 0.38, 0.34),    # 纤维/毛发
}


class DebrisDefect(BaseDefect):
    """异物：粘附在产品表面的外来物（灰尘颗粒、纤维、金属屑、碎纸屑）。

    真实特征（**立体感来自投影 + 自身明暗**）：
    - 物体本身有独立的颜色（与瓷体不同），形状不规则；
    - 在光照反方向有一道**投影**（贴着表面的暗影）——这是"贴附感"的关键；
    - 物体顶部有高光、底部偏暗，表现出轻微的厚度；
    - 一部分呈纤维状（细长、弯曲、两端渐细）。
    """

    kind = "debris"

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None, hint=None) -> DefectResult:
        h, w = rgb.shape[:2]
        s = self.strength
        sv = self.severity_mult
        base = rgb.astype(np.float32) / 255.0

        reg = clip01(region) if region is not None else np.ones((h, w), np.float32)
        ys, xs = np.nonzero(reg > 0.5)
        if xs.size < 16:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        short = float(min(h, w))

        count = max(1, int(self.params.get("count", 4)))
        size_ratio = float(self.params.get("size_ratio", 0.022)) * (0.5 + 0.9 * s) * (0.6 + 0.5 * sv)
        shadow_amt = float(np.clip(self.params.get("shadow", 0.45), 0.0, 1.0))
        gloss = float(np.clip(self.params.get("gloss", 0.35), 0.0, 1.5))
        fiber_ratio = float(np.clip(self.params.get("fiber_ratio", 0.35), 0.0, 1.0))
        mode = str(self.params.get("color_mode", "auto")).lower()
        if mode == "auto":
            mode = "fiber" if rng.bool(fiber_ratio) else str(
                rng.choice(["dark", "light", "metal"]))

        # 光照方向：默认左上 → 投影落向右下
        th = math.radians(rng.uniform(200.0, 250.0))
        lx, ly = math.cos(th), math.sin(th)
        out = base.copy()
        mask = np.zeros((h, w), np.float32)
        n_made = 0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        for _ in range(count):
            if region is not None:
                cx, cy = sample_point(region, rng, margin=0.03)
            else:
                cx = rng.uniform(0.05 * w, 0.95 * w)
                cy = rng.uniform(0.05 * h, 0.95 * h)
            rad = max(1.2, short * size_ratio * rng.uniform(0.5, 1.8))
            is_fiber = (mode == "fiber") or rng.bool(fiber_ratio * 0.5)
            ang = rng.uniform(0, math.pi)
            stretch = rng.uniform(3.0, 7.0) if is_fiber else rng.uniform(0.85, 1.9)
            obj = _soft_blob_field((h, w), cx, cy, rad, ang, stretch, rng,
                                   edge_soft=0.30, irregular=1.25)
            # 真实异物多是**碎屑/颗粒**：轮廓要有明显的棱角与缺口，
            # 而不是规整的椭圆。仅在掩膜上做一次阈值化"削角"，让它更像碎块。
            facet = fractal_noise((h, w), rng, base_cells=max(3, int(rad * 0.55)),
                                  octaves=2, persistence=0.45)
            if not is_fiber:
                obj = clip01(obj * (0.72 + 0.62 * facet))
            else:
                # 纤维：中段实、两端渐细
                obj = clip01(obj ** 1.7)
            obj = obj * clip01(reg)
            if obj.max() <= 0.10 or int((obj > 0.4).sum()) < 5:
                continue
            n_made += 1

            col = np.array(_DEBRIS_COLORS.get(mode, _DEBRIS_COLORS["dark"]), np.float32)
            col = np.clip(col * rng.uniform(0.80, 1.20), 0.0, 1.0)

            # ---- 投影：沿光照反方向偏移并模糊（贴附感的关键） ----
            if shadow_amt > 0.02:
                off = max(1.0, rad * 0.42)
                sh = np.zeros((h, w), np.float32)
                sx0 = int(max(0, cx - lx * off - rad * 1.5))
                sy0 = int(max(0, cy - ly * off - rad * 1.5))
                sx1 = int(min(w, cx - lx * off + rad * 1.5 + 1))
                sy1 = int(min(h, cy - ly * off + rad * 1.5 + 1))
                if sx1 > sx0 and sy1 > sy0:
                    sh[sy0:sy1, sx0:sx1] = obj[sy0:sy1, sx0:sx1]
                sh = blur_mask(sh, max(0.7, rad * 0.22))
                sh = clip01(sh) * (1.0 - obj) * clip01(reg)
                out = clip01(out * (1.0 - shadow_amt * sh[:, :, None]))

            # ---- 物体自身明暗：**很扁**的起伏，避免变成"球" ----
            # 真实附着物贴在表面，厚度很小，明暗范围窄；
            # 不再用物体轮廓梯度（那会让它看起来像颗珠子），改用低频面噪声。
            oy_, ox_ = np.gradient(clip01(obj) * rad)
            slope = clip01(0.5 + 0.5 * (-ox_ * lx - oy_ * ly) * 0.8)
            lit = 0.84 + 0.16 * slope
            lit = lit + gloss * 0.18 * np.power(clip01(slope), 6.0)
            lit = np.clip(np.where(obj > 0.02, lit, 1.0), 0.45, 1.22)
            body = np.clip(col[None, None, :] * lit[:, :, None], 0, 1)
            a = (clip01(obj) * np.clip(0.85 + 0.2 * s, 0, 1))[:, :, None]
            out = clip01(out * (1 - a) + body * a)
            mask = np.maximum(mask, obj)

        if n_made == 0 or mask.max() <= 0.02:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        return DefectResult(image=to_u8(out), mask=clip01(mask), kind=self.kind,
                            count=n_made)


# ==========================================================================
# 8. 鼓包凸起（局部隆起 / 起泡）
# ==========================================================================
@dataclass
class BumpParams:
    count: int = 1                 # 鼓包个数
    size_ratio: float = 0.012      # 鼓包半径相对产品短边
    height: float = 0.55           # 凸起高度（影响立体感强弱）
    gloss: float = 0.40            # 顶部高光强度
    light_deg: float = 135.0       # 光照方向（度，0=从右）
    edge_soft: float = 0.60        # 边缘过渡柔和度（0=边界清晰 … 1=平滑融入）
    color: str = "auto"            # auto=保持邻近区域材质 / custom=自定义颜色
    color_rgb: tuple = (180, 170, 160)  # 鼓包颜色（custom 模式生效）


class BumpDefect(BaseDefect):
    """鼓包凸起：表层局部隆起/起泡（分层、烧结鼓胀）。

    真实特征（与凹坑正好相反的明暗极性）：
    - 颜色与邻近区域几乎一致（还是那块材料，只是被顶了起来）——
      auto 模式直接保留原表面材质，只叠加方向性明暗；
    - 「凸」的高度场是正的（中心高）→ 法线向外倾：
      迎光侧被照亮、背光侧落下暗影，顶部有一小块柔和高光；
    - 轮廓是光滑的圆/椭圆（边缘环带轻噪声），边缘经软过渡融入周围，
      没有边界线。
    """

    kind = "bump"

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None, hint=None) -> DefectResult:
        h, w = rgb.shape[:2]
        s = self.strength
        sv = self.severity_mult
        base = rgb.astype(np.float32) / 255.0

        reg = clip01(region) if region is not None else np.ones((h, w), np.float32)
        ys, xs = np.nonzero(reg > 0.5)
        if xs.size < 16:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        short = float(min(h, w))

        count = max(1, int(self.params.get("count", 1)))
        size_ratio = float(self.params.get("size_ratio", 0.012)) * (0.5 + 0.9 * s) * (0.6 + 0.5 * sv)
        height_amt = float(np.clip(self.params.get("height", 0.55), 0.05, 1.0))
        gloss = float(np.clip(self.params.get("gloss", 0.40), 0.0, 1.5))
        light_deg = float(self.params.get("light_deg", 135.0))
        es = float(np.clip(self.params.get("edge_soft", 0.60), 0.0, 1.0))
        cm = str(self.params.get("color", "auto")).lower()
        crgb = self.params.get("color_rgb", (180, 170, 160))
        if not _is_rgb_triple(crgb):
            crgb = (180, 170, 160)
        th = math.radians(light_deg)
        lx, ly = math.cos(th), math.sin(th)      # 光照方向（图像坐标，y 向下）

        out = base.copy()
        mask = np.zeros((h, w), np.float32)
        n_made = 0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        for _ in range(count):
            if region is not None:
                cx, cy = sample_point(region, rng, margin=0.03)
            else:
                cx = rng.uniform(0.05 * w, 0.95 * w)
                cy = rng.uniform(0.05 * h, 0.95 * h)
            rad = max(1.5, short * size_ratio * rng.uniform(0.6, 1.6))

            # ---- 高度场：光滑**正**球冠（中心高）→ 凸起的明暗极性 ----
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(rad, 1e-6)
            cap = np.clip(1.0 - d * d, 0.0, 1.0)
            if cap.max() <= 0.05:
                continue
            hy, hx = np.gradient(cap * rad)      # 正号 → 法线向外倾（凸）
            nlen = np.sqrt(hx * hx + hy * hy + 1.0)
            nx_ = -hx / nlen
            ny_ = -hy / nlen
            nz_ = 1.0 / nlen
            lam = clip01(nx_ * lx + ny_ * ly + nz_ * 0.88)
            # 迎光侧亮、背光侧暗 + 顶部柔和高光（单一信息，不做多余环带）
            shade = 1.0 - height_amt * 0.35 * (1.0 - lam)
            shade = shade + gloss * 0.30 * np.power(clip01(lam), 5.0)
            shade = np.clip(np.where(d < 1.15, shade, 1.0), 0.55, 1.5)

            # ---- 轮廓：只在边缘环带加噪声，鼓包内部高度场保持光滑 ----
            n = fractal_noise((h, w), rng, base_cells=max(4, int(rad * 0.8)),
                              octaves=3, persistence=0.5)
            ring = np.clip(1.0 - np.abs(d - 0.92) / 0.22, 0, 1)
            rmod = d * (1.0 + 0.18 * (n - 0.5) * ring)
            dome = np.clip(1.0 - rmod * rmod, 0.0, 1.0)
            if dome.max() <= 0.05:
                continue

            # ---- 颜色：auto 保持原表面材质（只加明暗）；custom 用指定颜色 ----
            if cm == "custom":
                col = np.array([float(c) for c in crgb], np.float32) / 255.0
                detail = fractal_noise((h, w), rng,
                                       base_cells=max(4, int(rad * 1.4)),
                                       octaves=2, persistence=0.5)
                body = clip01(col[None, None, :]
                              * (0.96 + 0.08 * detail[:, :, None]) * shade[:, :, None])
            else:
                body = base * shade[:, :, None]   # 材质不变，立体感全部来自明暗

            # ---- 边缘平滑融入：alpha 随 smoothstep 衰减，边缘处接近 0 ----
            band = 0.10 + 0.55 * es               # es 越大过渡越宽、越无边界感
            a = _smoothstep((1.0 - rmod) / max(band, 1e-3))
            a = clip01(a * np.clip(0.80 + 0.3 * s, 0, 1)) * clip01(reg)
            if a.max() <= 0.05:
                continue
            n_made += 1
            out = clip01(out * (1 - a[:, :, None]) + body * a[:, :, None])
            mask = np.maximum(mask, dome * clip01(reg))

        if n_made == 0 or mask.max() <= 0.02:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)
        return DefectResult(image=to_u8(out), mask=clip01(mask), kind=self.kind,
                            count=n_made)

# 组合器
# ==========================================================================
DEFECT_KINDS = ("scratch", "smudge", "discolor", "chip", "dot", "pit",
                "debris", "bump")

#: 类型名 → 生成器类（新增缺陷类型只需在此登记一次）
DEFECT_REGISTRY = {
    "scratch": ScratchDefect,
    "smudge": SmudgeDefect,
    "discolor": DiscolorDefect,
    "chip": ChipDefect,
    "dot": BlackDotDefect,
    "pit": PitDefect,
    "debris": DebrisDefect,
    "bump": BumpDefect,
}

#: 各类「个数」在参数里的字段名（增强模式据此决定每张图生成几个）
COUNT_FIELD = {
    "scratch": "count",
    "smudge": "blob_count",
    "discolor": "region_count",
    "chip": "count",
    "dot": "count",
    "pit": "count",
    "debris": "count",
    "bump": "count",
}

_KIND_TO_COLOR = {
    "scratch": (231, 76, 60),     # 红
    "smudge": (241, 196, 15),     # 黄
    "discolor": (52, 152, 219),   # 蓝
    "chip": (155, 89, 182),       # 紫
    "dot": (44, 44, 52),          # 深灰（黑点）
    "pit": (86, 160, 210),        # 青蓝（凹坑）
    "debris": (230, 138, 74),     # 橙（异物）
    "bump": (255, 128, 170),      # 粉（鼓包）
}


# ==========================================================================
# 5. 缺陷样本增强：在用户手绘区域内随机生成多个、随机类型/位置/形状的缺陷
# ==========================================================================
def _soft_ellipse(shape, cx: float, cy: float, r: float,
                  ang: float, stretch: float = 1.0) -> np.ndarray:
    """生成柔边椭圆掩膜 [0,1]：中心=1，半径 r 处=0。

    用作「局部缺陷范围」——生成器只在该范围内放置缺陷，
    从而把单个实例约束在区域内的随机锚点附近（实现"随机位置"）。
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    rx = (dx * math.cos(ang) + dy * math.sin(ang)) / max(stretch, 1e-3)
    ry = (-dx * math.sin(ang) + dy * math.cos(ang))
    d = np.sqrt(rx * rx + ry * ry)
    return clip01(1.0 - d / max(r, 1e-6)).astype(np.float32)


def _random_defect_params(kind: str, rng: Rng, strength: float,
                          scale: float, cx: float, cy: float,
                          extra: Optional[dict] = None) -> dict:
    """为"缺陷样本增强"构造单类缺陷的随机参数（只生成 1 个实例）。

    scale 控制缺陷相对"局部区域短边"的尺寸系数（0..1）；
    各类内部再用分形噪声 / 随机游走路径产生"随机形状"，颜色与明显度也随机。
    extra 可传入用户在界面里对该类的设定（如异色的 color/color_rgb），
    以便增强样本也沿用指定的色调。
    """
    if kind == "scratch":
        return {
            "count": 1,
            "length_ratio": float(np.clip(scale * rng.uniform(0.6, 1.4), 0.08, 0.95)),
            "width_px": rng.uniform(1.2, 3.6),
            "curve": rng.uniform(0.1, 0.9),
            "tone": rng.uniform(-0.6, 0.6),
            "tone_depth": rng.uniform(0.5, 1.0),
            "dark_ratio": rng.uniform(0.2, 0.8),
            "jitter": rng.uniform(0.5, 2.0),
            "feather": rng.uniform(0.8, 1.6),
            "double_edge": bool(rng.bool(0.6)),
        }
    if kind == "smudge":
        return {
            "blob_count": 1,
            "blob_scale": float(np.clip(scale * rng.uniform(0.6, 1.3), 0.05, 0.6)),
            "coverage": rng.uniform(0.2, 0.5),
            "dirt_color": str(rng.choice(["auto", "dark", "dust", "oil", "rust"])),
            "opacity": rng.uniform(0.35, 0.8),
            "granular": rng.uniform(0.2, 0.6),
            "speckle": bool(rng.bool(0.6)),
            "edge_soft": rng.uniform(0.35, 0.95),
        }
    if kind == "discolor":
        p = {
            "region_count": 1,
            "region_scale": float(np.clip(scale * rng.uniform(0.6, 1.3), 0.05, 0.6)),
            "lum_shift": rng.uniform(-18, 6),
            "sat_gain": rng.uniform(1.05, 1.6),
            "edge_soft": rng.uniform(0.35, 0.95),
        }
        # 沿用用户在界面里选定的异色颜色（随机/预设/自定义），让增强样本也带指定色调
        if extra:
            if "color" in extra:
                p["color"] = str(extra["color"])
            if _is_rgb_triple(extra.get("color_rgb")):
                p["color_rgb"] = list(extra["color_rgb"])
        return p
    if kind == "chip":
        return {
            "count": 1,
            "size_ratio": float(np.clip(scale * rng.uniform(0.5, 1.1), 0.05, 0.5)),
            "jag_ratio": rng.uniform(0.25, 0.7),
            "depth_ratio": rng.uniform(0.18, 0.45),     # 浅崩缺
            "rim_width": rng.uniform(1.5, 3.0),
            "crack_ratio": rng.uniform(0.0, 0.45),
            "grain": rng.uniform(0.3, 0.6),
            "darkness": rng.uniform(0.25, 0.5),
            "edge_soft": rng.uniform(0.25, 0.75),
            "color_match": rng.uniform(0.4, 0.9),
            "debris": bool(rng.bool(0.6)),
            "corner_bias": rng.uniform(0.15, 0.6),
        }
    if kind == "dot":
        p = {
            "count": 1,
            "size_ratio": float(np.clip(scale * rng.uniform(0.5, 1.4), 0.004, 0.05)),
            "darkness": rng.uniform(0.70, 0.95),
            "cluster": rng.uniform(0.0, 0.6),
            "edge_soft": rng.uniform(0.40, 0.90),
        }
        # 沿用用户在界面里选定的黑点颜色（自动/自定义），让增强样本也带指定色调
        if extra:
            if "color" in extra:
                p["color"] = str(extra["color"])
            if _is_rgb_triple(extra.get("color_rgb")):
                p["color_rgb"] = list(extra["color_rgb"])
        return p
    if kind == "pit":
        return {
            "count": 1,
            "size_ratio": float(np.clip(scale * rng.uniform(0.6, 1.5), 0.006, 0.06)),
            "depth": rng.uniform(0.35, 0.85),
            "highlight": rng.uniform(0.3, 0.9),
            "light_deg": rng.uniform(0.0, 360.0),
        }
    if kind == "debris":
        return {
            "count": 1,
            "size_ratio": float(np.clip(scale * rng.uniform(0.5, 1.5), 0.008, 0.08)),
            "shadow": rng.uniform(0.25, 0.6),
            "gloss": rng.uniform(0.15, 0.6),
            "fiber_ratio": rng.uniform(0.0, 0.7),
            "color_mode": str(rng.choice(["auto", "dark", "light", "metal", "fiber"])),
        }
    if kind == "bump":
        return {
            "count": 1,
            "size_ratio": float(np.clip(scale * rng.uniform(0.6, 1.5), 0.005, 0.06)),
            "height": rng.uniform(0.30, 0.80),
            "gloss": rng.uniform(0.15, 0.70),
            "edge_soft": rng.uniform(0.35, 0.90),
            "light_deg": rng.uniform(0.0, 360.0),
        }
    return {}


def _read_augment_counts(settings) -> dict:
    """从缺陷生成算法参数里读出各类的"个数"，作为每张增强图的缺陷数。

    与 apply_defects 保持一致：每张图的总缺陷数 = 各启用类型的个数之和。
    各类"个数"字段名统一由 ``COUNT_FIELD`` 给出（scratch.count /
    smudge.blob_count / discolor.region_count / chip·dot·pit·debris.count），
    新增类型只需在 ``COUNT_FIELD`` 与 ``DEFECT_REGISTRY`` 各登记一次。
    未启用任何类型时退化为每类各 1 个。
    """
    counts: dict = {}
    if settings:
        for k in DEFECT_KINDS:
            cfg = settings.get(k, {}) or {}
            if not cfg.get("enable", False):
                continue
            c = int(cfg.get(COUNT_FIELD.get(k, "count"), 1))
            c = max(0, c)
            if c > 0:
                counts[k] = c
    if not counts:
        # 未启用任何类型时，退化为每类各 1 个，保证有产物
        counts = {k: 1 for k in DEFECT_KINDS}
    return counts


def _augment_one_image(rgb: np.ndarray, region, counts: dict,
                       rng: "Rng", severity: str, settings=None):
    """生成"一张"缺陷图：缺陷类型/位置/形状随机，数量 = counts 之和。

    类型来自 settings 中启用的种类，位置在用户手绘区域内随机锚定，
    形状由各生成器自带的分形噪声 / 随机游走路径产生，从而"随机位置 + 随机形状"。
    """
    img = rgb.copy()
    h, w = rgb.shape[:2]
    reg = clip01(region).astype(np.float32) if region is not None else np.ones((h, w), np.float32)
    if reg.max() <= 0.01:
        reg = np.ones((h, w), np.float32)
    ys, xs = np.nonzero(reg > 0.5)
    if xs.size:
        R_short = float(min(int(xs.max() - xs.min() + 1),
                            int(ys.max() - ys.min() + 1)))
    else:
        R_short = float(min(h, w))
    R_short = max(R_short, 24.0)

    # 每张图的缺陷总数 = 各启用类型"个数"之和
    total = max(1, sum(counts.values()))
    # 单个缺陷的局部范围（相对区域短边）：实例越多，单个范围越小，避免全挤成一块
    per_radius = float(np.clip(0.10 + 0.30 / math.sqrt(total), 0.05, 0.40))
    lo, hi = SEVERITY_LEVELS.get(severity, (0.12, 1.0))

    applied: List[DefectResult] = []
    for kind, c in counts.items():
        for _ in range(c):
            sv = rng.uniform(lo, hi)
            strength = rng.uniform(0.35, 0.9)
            # 随机锚点（在用户手绘区域内）→ 实现"随机位置"
            ax, ay = sample_point(reg, rng, margin=0.04)
            if ax == 0.0 and ay == 0.0:
                continue
            r = R_short * per_radius * rng.uniform(0.8, 1.25)
            # 局部区域形状也随机（各向异性椭圆），让落点更自然
            ang = rng.uniform(0, np.pi)
            stretch = rng.uniform(1.0, 2.0)
            local = _soft_ellipse((h, w), ax, ay, r, ang, stretch)
            # 关键约束：把随机椭圆与用户手绘区域求交，确保缺陷绝不越出限定区域。
            # 椭圆虽以区域内锚点为中心，但半径可大于区域边界，不裁剪会溢出到手绘区之外。
            local = clip01(local) * reg
            cfg = _random_defect_params(kind, rng, strength, 0.6, ax, ay,
                                        settings.get(kind))
            gen = DEFECT_REGISTRY.get(kind)
            if gen is None:
                continue
            # 崩缺是「边沿」缺陷：局部椭圆是产品内部的，压根没有轮廓可用。
            # 这里把整个手绘区域交给它，并用 hint 告诉它锚点在哪，
            # 让它去挑**离锚点最近的那段边沿**，既保证有轮廓可依，
            # 又不会跑到离锚点很远的地方去。
            if kind == "chip":
                res = gen(strength=strength, severity_mult=sv, **cfg).render(
                    img, rng, region=reg, hint=(0.0, ax, ay))
            else:
                res = gen(strength=strength, severity_mult=sv, **cfg).render(
                    img, rng, region=local)
            res.kind = kind
            res.severity = severity
            res.surface = "augment"
            if res.count == 0 or res.mask.max() <= 0.02:
                continue
            img = res.image
            applied.append(res)
    return img, applied


def apply_augmentation(rgb: np.ndarray,
                       region: Optional[np.ndarray] = None,
                       n_images: int = 6,
                       seed: Optional[int] = None,
                       settings: Optional[dict] = None,
                       severity: str = "mixed"):
    """缺陷样本增强：在用户手绘区域内，按设定数量生成多张缺陷图。

    与"缺陷生成算法参数"对齐：每张图的缺陷数量由 settings 中各类的"个数"
    （scratch.count / smudge.blob_count / discolor.region_count / chip.count）
    之和决定；未启用任何类型时退化为四类各 1 个。各缺陷的类型/位置/形状随机。

    rgb      : (H,W,3) uint8 原图（建议为一张缺陷样板图）
    region   : float32 [0,1] 手绘区域掩膜；为 None 时退化为整图随机生成
    n_images : 生成的缺陷图张数（即"生成图片数量"）
    seed     : 随机种子（同种子可复现）
    settings : 缺陷生成算法参数（用于读取"个数"），可为 None
    severity : 明显度等级（"mixed"=随机混合）

    返回 list[(image, applied)] —— 每张图一个 (图像, [DefectResult,...])。
    """
    counts = _read_augment_counts(settings)
    n_images = max(1, int(n_images))
    base = int(seed) if seed is not None else 1
    results = []
    for i in range(n_images):
        irng = Rng((base * 1000003 + i * 2654435761 + 12345) & 0x7FFFFFFF)
        img, applied = _augment_one_image(rgb, region, counts, irng, severity, settings)
        results.append((img, applied))
    return results


def merge_masks(applied, shape) -> np.ndarray:
    """把多类缺陷掩膜合并为 (H,W) float32。"""
    h, w = shape
    m = np.zeros((h, w), dtype=np.float32)
    for r in applied:
        m = np.maximum(m, r.mask)
    return clip01(m)


def make_comparison(original: np.ndarray, result: DefectResult,
                    mask_rgb: bool = True) -> np.ndarray:
    """生成「原图 | 掩膜 | 缺陷图」三联对比图。"""
    h, w = original.shape[:2]
    m = clip01(result.mask)
    mask_vis = np.stack([to_u8(m)] * 3, axis=2)
    if mask_rgb:
        overlay = original.astype(np.float32).copy()
        color = np.array(_KIND_TO_COLOR.get(result.kind, (255, 255, 255)), dtype=np.float32)
        a = (m * 0.65)[:, :, None]
        overlay = original.astype(np.float32) * (1 - a) + color[None, None, :] * a
        mask_vis = np.clip(overlay, 0, 255).astype(np.uint8)

    gap = np.full((h, 8, 3), 40, dtype=np.uint8)
    out = np.concatenate([original, gap, mask_vis, gap, result.image], axis=1)
    return out
