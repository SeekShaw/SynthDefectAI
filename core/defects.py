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
        if region is not None:
            short = int(max(np.sqrt(max((region > 0.5).sum(), 1.0)) * 1.6, 24))

        blob_count = max(1, int(self.params.get("blob_count", 1)))
        blob_scale = self.params.get("blob_scale", 0.30) * (0.55 + 0.45 * sv)
        opacity = float(np.clip(self.params.get("opacity", 0.55) * (0.55 + 0.65 * s)
                                * (0.30 + 0.70 * sv), 0.0, 0.95))
        granular = self.params.get("granular", 0.35) * (0.4 + s) * (0.5 + 0.5 * sv)
        speckle = self.params.get("speckle", True)
        edge_soft = float(self.params.get("edge_soft", 1.0))

        mask = np.zeros((h, w), dtype=np.float32)
        radii = []
        for _ in range(blob_count):
            if region is not None:
                cx, cy = sample_point(region, rng, margin=0.04)
            else:
                cx = rng.uniform(0.08 * w, 0.92 * w)
                cy = rng.uniform(0.08 * h, 0.92 * h)
            rad = short * blob_scale * rng.uniform(0.45, 1.25)
            radii.append(rad)
            # 各向异性：脏污常沿某个方向拉长
            ang = rng.uniform(0, np.pi)
            stretch = rng.uniform(1.0, 2.2)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dx, dy = xx - cx, yy - cy
            rx = dx * np.cos(ang) + dy * np.sin(ang)
            ry = -dx * np.sin(ang) + dy * np.cos(ang)
            d = np.sqrt((rx / stretch) ** 2 + ry ** 2)
            base = np.clip((rad - d) / max(rad, 1e-6), 0.0, 1.0)

            # 用噪声把圆形斑块打碎成不规则形状
            cells = max(3, int(short / max(rad, 1.0) * 2.2))
            n = fractal_noise((h, w), rng, base_cells=cells, octaves=4,
                              persistence=0.58)
            blob = clip01(base * (0.35 + 1.30 * n) * rng.uniform(0.75, 1.15))
            mask = np.maximum(mask, blob)

        # 边缘柔化：先高斯模糊把硬边化成渐变带，再做覆盖率阈值。
        # 顺序很关键：若先阈值后模糊，软晕会吃掉明显度差异（模糊把面积重新撑大），
        # 且必须保证「模糊在前、区域裁剪在后」，否则边缘会溢出产品表面。
        rad_avg = float(np.mean(radii)) if radii else float(short * blob_scale)
        sigma = max(2.0, rad_avg * 0.12 * edge_soft)
        mask = blur_mask(mask, sigma)

        # 全局阈值化，控制实际覆盖率
        cov = float(np.clip(self.params.get("coverage", 0.35) * (0.6 + 0.9 * s)
                            * (0.5 + 0.5 * sv), 0.02, 0.85))
        thr = float(np.quantile(mask, 1.0 - cov)) if mask.max() > 0 else 1.0
        mask = clip01((mask - thr) / max(1.0 - thr, 1e-6))

        if speckle:
            sp = self._speckle((h, w), rng, short, density=0.35 * (0.5 + s) * (0.4 + 0.6 * sv),
                               region=region)
            # 飞溅点同样轻微模糊，避免出现一颗颗硬边小圆
            mask = np.maximum(mask, blur_mask(sp, max(0.8, sigma * 0.30)) * 0.75)
        # 区域约束：污渍只留在目标表面
        if region is not None:
            mask = mask * clip01(region)
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
        edge_soft = float(np.clip(self.params.get("edge_soft", 0.45), 0.05, 1.0))
        edge_blur = float(self.params.get("edge_blur", 1.0))

        mask = np.zeros((h, w), dtype=np.float32)
        radii = []
        for _ in range(region_count):
            if region is not None:
                cx, cy = sample_point(region, rng, margin=0.05)
            else:
                cx = rng.uniform(0.10 * w, 0.90 * w)
                cy = rng.uniform(0.10 * h, 0.90 * h)
            rad = short * region_scale * rng.uniform(0.5, 1.3)
            radii.append(rad)
            ang = rng.uniform(0, np.pi)
            stretch = rng.uniform(1.0, 2.4)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dx, dy = xx - cx, yy - cy
            rx = dx * np.cos(ang) + dy * np.sin(ang)
            ry = -dx * np.sin(ang) + dy * np.cos(ang)
            d = np.sqrt((rx / stretch) ** 2 + ry ** 2)
            base = np.clip((rad - d) / max(rad * edge_soft, 1e-6), 0.0, 1.0)

            # 不规则边界
            cells = max(3, int(short / max(rad, 1.0) * 1.8))
            n = fractal_noise((h, w), rng, base_cells=cells, octaves=4, persistence=0.55)
            blob = clip01(base * (0.45 + 1.05 * n) * rng.uniform(0.8, 1.1))
            mask = np.maximum(mask, blob)

        mask = clip01(mask)

        # 边缘柔化：先高斯模糊把硬边化成渐变带，再做覆盖率上限控制。
        # 顺序很重要——先模糊后阈值，既保住柔软边缘，又不会让软晕把明显度差异冲淡；
        # 「区域裁剪」必须放在最后，否则模糊的晕会溢出产品表面。
        rad_avg = float(np.mean(radii)) if radii else float(short * region_scale)
        mask = blur_mask(mask, max(2.5, rad_avg * 0.16 * edge_blur))

        # 覆盖率上限控制：只在超调时收缩（阈值低于零时掩膜保持不变）
        cov = float(np.clip(0.06 + 0.30 * s, 0.02, 0.6))
        thr = float(np.quantile(mask, 1.0 - cov))
        if thr > 0:
            mask = clip01((mask - thr) / max(1.0 - thr, 1e-6))

        # 区域约束：异色只留在目标表面
        if region is not None:
            mask = mask * clip01(region)
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
# 4. 缺角（崩边 / 掉角）
# ==========================================================================
@dataclass
class ChipParams:
    count: int = 1                 # 缺角个数
    size_ratio: float = 0.40       # 缺角尺寸相对产品短边的比例
    jag_ratio: float = 0.42        # 断面边缘的锯齿/崩口程度
    depth_ratio: float = 0.85      # 崩口深度（相对缺角尺寸）
    rim_width: float = 2.2         # 断面亮边宽度（像素）
    crack_ratio: float = 0.35      # 从缺角延伸出的裂纹长度比
    grain: float = 0.42            # 断面颗粒感
    darkness: float = 0.34         # 崩口内壁变暗程度
    debris: bool = True            # 是否产生细小碎屑


class ChipDefect(BaseDefect):
    """缺角：产品边角被磕碰掉一块，露出粗糙断面。

    物理表现：
    1. 轮廓缺口 —— 以角点为圆心、沿边界内切的不规则多边形把产品"啃掉"一块；
    2. 新断面 —— 粗糙陶瓷颗粒感，比原表面更亮（新鲜断面反光更强）且更粗糙；
    3. 断面与原表面交界处有一圈细微暗边（凹凸处的阴影/污染）；
    4. 缺角外侧（背景侧）有细小碎屑；
    5. 可选从缺角顶点延伸出短裂纹。
    """

    kind = "chip"

    def render(self, rgb: np.ndarray, rng: Rng,
               region: Optional[np.ndarray] = None, hint=None) -> DefectResult:
        h, w = rgb.shape[:2]
        s = self.strength
        sv = self.severity_mult
        base = rgb.astype(np.float32) / 255.0

        reg = region if region is not None else np.ones((h, w), np.float32)
        ys, xs = np.nonzero(reg > 0.5)
        if len(xs) < 64:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        prod_short = float(min(x1 - x0 + 1, y1 - y0 + 1))
        if prod_short < 12:
            return DefectResult(image=rgb.copy(), mask=np.zeros((h, w), np.float32),
                                kind=self.kind, count=0)

        count = max(1, int(self.params.get("count", 1)))
        size_ratio = self.params.get("size_ratio", 0.40) * (0.5 + 0.6 * s) * (0.55 + 0.45 * sv)
        radius = max(3.0, prod_short * size_ratio * 0.34)
        jag = self.params.get("jag_ratio", 0.42)
        depth = self.params.get("depth_ratio", 0.85)

        mask = np.zeros((h, w), dtype=np.float32)
        out = base.copy()
        n_made = 0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        for _ in range(count):
            corner = self._pick_corner(reg, rng, exclude=None)
            if corner is None:
                break
            cx, cy, ox, oy = corner

            # ---- 缺角几何：沿角点切掉一块（半平面断面） ----
            # n_in: 由角点指向产品内部的内法线；tan: 沿断面的走向
            n_in = np.array([-ox, -oy], np.float32)
            tan = np.array([-n_in[1], n_in[0]], np.float32)
            dd = (xx - cx) * n_in[0] + (yy - cy) * n_in[1]   # 深度（沿内法线）
            tt = (xx - cx) * tan[0] + (yy - cy) * tan[1]     # 横向（沿断面）

            half = radius * rng.uniform(1.05, 1.75)             # 断面半宽
            depth_len = half * float(np.clip(depth, 0.35, 1.5)) * rng.uniform(0.8, 1.3)

            # 断面线：多频角向抖动 + 分形噪声 → 贝壳状不规则崩口
            th = np.arctan2(tt, np.maximum(dd, 1e-3))
            wob = (np.sin(th * rng.uniform(2.0, 3.5) + rng.uniform(0, 6.28)) * 0.55
                   + np.sin(th * rng.uniform(5.0, 8.5) + rng.uniform(0, 6.28)) * 0.30
                   + np.sin(th * rng.uniform(11.0, 16.0) + rng.uniform(0, 6.28)) * 0.15)
            edge_noise = fractal_noise((h, w), rng, base_cells=max(3, int(radius * 0.9)),
                                       octaves=4, persistence=0.62)
            cut = depth_len * (1.0 + jag * wob * 0.42
                               + jag * (edge_noise - 0.5) * 0.90)

            # 横向收口：中间深、两端尖（角被磕掉的典型形状）
            span = clip01(1.0 - np.abs(tt) / max(half, 1e-6))
            span = clip01(span * (0.5 + 0.5 * span))

            bite = clip01((cut - dd) / max(0.18, jag * 0.30 + 0.12))
            bite = clip01(bite * (0.30 + 0.90 * span))
            bite = clip01(bite * clip01(reg))     # 缺口必须在产品轮廓内侧
            if bite.max() <= 0.05 or float((bite > 0.5).sum()) < 12:
                continue

            depth_w = clip01(1.0 - dd / max(float(np.percentile(cut, 99.5)), 1e-6))
            chip = bite
            n_made += 1

            # ---- 断面材质：略亮、粗糙（新鲜断口的陶瓷颗粒） ----
            grain = fractal_noise((h, w), rng, base_cells=max(6, int(radius * 1.2)),
                                  octaves=4, persistence=0.62)
            grain_amt = self.params.get("grain", 0.42) * (0.4 + 0.7 * s) * (0.4 + 0.6 * sv)
            lum = out.mean(axis=2, keepdims=True)
            # 断面只比原表面略亮（+4%~12%），主要靠粗糙感而非亮度区分
            fracture = clip01(lum * rng.uniform(1.04, 1.12)
                              + (grain - 0.5)[:, :, None] * grain_amt * 0.42)
            # 崩口内壁（靠外侧）稍暗，模拟深度阴影
            dark = self.params.get("darkness", 0.34) * (0.4 + 0.8 * s) * (0.4 + 0.6 * sv)
            fracture = clip01(fracture - dark * depth_w[:, :, None] * 0.30)
            # 断面少量色偏（露出未烧结体，偏灰褐）
            tint = np.array([1.01, 0.995, 0.965], np.float32)
            fracture = clip01(fracture * tint[None, None, :])

            a_chip = chip[:, :, None]
            out = out * (1 - a_chip) + fracture * a_chip

            # ---- 交界暗边：断面轮廓内侧一圈暗线 ----
            rim = np.zeros((h, w), np.float32)
            try:
                import cv2

                k = max(1, int(round(self.params.get("rim_width", 2.2))))
                dil = cv2.dilate((chip > 0.5).astype(np.uint8),
                                 np.ones((3, 3), np.uint8), iterations=max(1, k // 2))
                inner = (cv2.dilate((chip > 0.5).astype(np.uint8),
                                    np.ones((3, 3), np.uint8), iterations=k) - dil)
                rim = clip01(inner.astype(np.float32))
            except Exception:
                rim = clip01(chip - clip01(bite * 0.82))
            rim = rim * (1.0 - chip * 0.35)
            rim = clip01(rim) * clip01(reg)
            out = clip01(out * (1.0 - 0.45 * rim[:, :, None]))

            # ---- 断口外侧（背景侧）碎屑 ----
            if self.params.get("debris", True):
                debris = np.zeros((h, w), np.float32)
                for _ in range(rng.randint(3, 8)):
                    a = rng.uniform(0, math.pi * 2)
                    rr = radius * rng.uniform(0.9, 2.0)
                    # 沿断面向外散落
                    dcx = cx + ox * rr * 0.35 + tan[0] * math.cos(a) * rr \
                        + n_in[0] * math.sin(a) * rr * 0.5
                    dcy = cy + oy * rr * 0.35 + tan[1] * math.cos(a) * rr \
                        + n_in[1] * math.sin(a) * rr * 0.5
                    r = max(0.6, radius * rng.uniform(0.015, 0.05))
                    # 碎屑用小的不规则团块而不是正圆
                    dm = filled_circle_mask((h, w), dcx, dcy, r, feather=0.6)
                    dn = fractal_noise((h, w), rng, base_cells=max(8, int(r * 6) + 8),
                                       octaves=2, persistence=0.5)
                    dm = clip01(dm * (0.35 + 1.5 * (dn > 0.55)))
                    debris = np.maximum(debris, dm)
                # 碎屑不计入缺陷掩膜（不是产品缺陷本身），但要画到图上
                dmask = clip01(debris) * (1.0 - clip01(reg))
                dcol = np.array([0.72, 0.70, 0.66], np.float32)
                out = clip01(out * (1 - dmask[:, :, None] * 0.85)
                             + dcol[None, None, :] * dmask[:, :, None] * 0.85)

            # ---- 可选：从缺角延伸短裂纹（沿断面走向向内延伸） ----
            cr = self.params.get("crack_ratio", 0.35) * (0.3 + 0.9 * s) * (0.4 + 0.6 * sv)
            if cr > 0.02:
                from .utils import polyline_mask

                clen = radius * rng.uniform(1.2, 2.8) * cr
                side = rng.choice([-1.0, 1.0])
                # 起点在断面中部，方向斜向产品内部
                sx = cx + n_in[0] * depth_len * 0.55 + tan[0] * side * half * 0.35
                sy = cy + n_in[1] * depth_len * 0.55 + tan[1] * side * half * 0.35
                a_dir = math.atan2(n_in[1], n_in[0]) + side * rng.uniform(0.35, 0.95)
                p0 = (sx, sy)
                p1 = (sx + math.cos(a_dir) * clen * 0.4 + tan[0] * side * clen * 0.22,
                      sy + math.sin(a_dir) * clen * 0.4 + tan[1] * side * clen * 0.22)
                p2 = (sx + math.cos(a_dir) * clen * 0.75 - tan[0] * side * clen * 0.12,
                      sy + math.sin(a_dir) * clen * 0.75 - tan[1] * side * clen * 0.12)
                p3 = (sx + math.cos(a_dir) * clen,
                      sy + math.sin(a_dir) * clen)
                cm = polyline_mask((h, w), [p0, p1, p2, p3],
                                   width=max(0.6, radius * rng.uniform(0.015, 0.035)),
                                   taper=[0.9, 0.7, 0.4, 0.05],
                                   feather=0.7)
                cm = clip01(cm) * clip01(reg)
                out = clip01(out * (1 - 0.42 * cm[:, :, None]))
                chip = np.maximum(chip, cm)

            mask = np.maximum(mask, chip)

        mask = clip01(mask) * clip01(reg)
        if mask.max() <= 0:
            return DefectResult(image=rgb.copy(), mask=mask, kind=self.kind, count=0)

        return DefectResult(image=to_u8(out), mask=mask, kind=self.kind, count=n_made)

    def _pick_corner(self, reg: np.ndarray, rng: Rng, exclude=None):
        """在区域轮廓上找一个「凸角」作为缺角位置。

        做法：距离变换找最宽处 → 用凸缺陷（convexity defect）挑角点。
        这里用更稳的替代方案：沿区域主轴的 4 个斜向极值点，
        再用局部邻域的凸性做二次筛选。
        """
        import cv2

        binm = (reg > 0.5).astype(np.uint8)
        if binm.sum() < 64:
            return None
        # 主轴（PCA）
        ys, xs = np.nonzero(binm)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        mu = pts.mean(0)
        cov = np.cov((pts - mu).T)
        if np.ndim(cov) == 0:
            cov = np.array([[cov, 0.0], [0.0, cov]], np.float32)
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
        minor = np.array([-major[1], major[0]], np.float32)

        # 距离变换：内部最大处 = 厚度最大（产品中心）
        dt = cv2.distanceTransform(binm, cv2.DIST_L2, 3)
        cys, cxs = np.unravel_index(int(np.argmax(dt)), dt.shape)

        cands = []
        for a_deg in (45, 135, 225, 315):
            a = math.radians(a_deg)
            d = major * math.cos(a) + minor * math.sin(a)
            proj = pts @ d
            i = int(np.argmax(proj))
            cands.append((float(pts[i, 0]), float(pts[i, 1])))

        rng.g.shuffle(cands)
        h, w = reg.shape[:2]
        for (cx, cy) in cands:
            if exclude is not None and \
                    abs(cx - exclude[0]) < exclude[2] * 1.5 and \
                    abs(cy - exclude[1]) < exclude[2] * 1.5:
                continue
            # 该点必须在区域内
            ix, iy = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
            if binm[iy, ix] == 0:
                continue
            # 凸性检查：角点应远离"厚度最大"方向（也就是在外轮廓上）
            # 用到中心的距离做归一化，距离越大越像角
            dctr = math.hypot(cx - float(cxs), cy - float(cys))
            if dctr < 8:
                continue
            # 外法线：从中心指向该点
            ox, oy = cx - float(cxs), cy - float(cys)
            n = math.hypot(ox, oy)
            if n < 1e-6:
                continue
            return (cx, cy, ox / n, oy / n)
        return None


# ==========================================================================
# 组合器
# ==========================================================================
DEFECT_KINDS = ("scratch", "smudge", "discolor", "chip")

#: 类型名 → 生成器类（新增缺陷类型只需在此登记一次）
DEFECT_REGISTRY = {
    "scratch": ScratchDefect,
    "smudge": SmudgeDefect,
    "discolor": DiscolorDefect,
    "chip": ChipDefect,
}

_KIND_TO_COLOR = {
    "scratch": (231, 76, 60),     # 红
    "smudge": (241, 196, 15),     # 黄
    "discolor": (52, 152, 219),   # 蓝
    "chip": (155, 89, 182),       # 紫
}


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
            "edge_soft": rng.uniform(0.6, 1.4),
        }
    if kind == "discolor":
        p = {
            "region_count": 1,
            "region_scale": float(np.clip(scale * rng.uniform(0.6, 1.3), 0.05, 0.6)),
            "lum_shift": rng.uniform(-18, 6),
            "sat_gain": rng.uniform(1.05, 1.6),
            "edge_soft": rng.uniform(0.2, 0.7),
            "edge_blur": rng.uniform(0.6, 1.4),
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
            "size_ratio": float(np.clip(scale * rng.uniform(0.6, 1.2), 0.05, 0.6)),
            "jag_ratio": rng.uniform(0.2, 0.7),
            "depth_ratio": rng.uniform(0.6, 1.2),
            "rim_width": rng.uniform(1.5, 3.0),
            "crack_ratio": rng.uniform(0.0, 0.5),
            "grain": rng.uniform(0.3, 0.6),
            "darkness": rng.uniform(0.25, 0.5),
            "debris": bool(rng.bool(0.6)),
        }
    return {}


def _read_augment_counts(settings) -> dict:
    """从缺陷生成算法参数里读出各类的"个数"，作为每张增强图的缺陷数。

    与 apply_defects 保持一致：每张图的总缺陷数 = 各启用类型的个数之和。
    各类对应的"个数"字段：scratch.count / smudge.blob_count /
    discolor.region_count / chip.count。未启用任何类型时退化为四类各 1 个。
    """
    counts: dict = {}
    if settings:
        for k in DEFECT_KINDS:
            cfg = settings.get(k, {}) or {}
            if not cfg.get("enable", False):
                continue
            if k == "scratch":
                c = int(cfg.get("count", 1))
            elif k == "smudge":
                c = int(cfg.get("blob_count", 1))
            elif k == "discolor":
                c = int(cfg.get("region_count", 1))
            elif k == "chip":
                c = int(cfg.get("count", 1))
            else:
                c = 1
            c = max(0, c)
            if c > 0:
                counts[k] = c
    if not counts:
        # 未启用任何类型时，退化为四类各 1 个，保证有产物
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
