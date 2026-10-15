"""MLCC 区域自动分割：从产品照片中分割出 背景 / 瓷体 / 端电极。

流程（针对 MLCC 结构定制，避免通用聚类的不稳定）：
1. 前景分割 —— 分别尝试「亮侧/暗侧」两个阈值候选，
   按「最大连通域的实心度」评分取优（产品是单一紧凑块，噪声是散点）；
2. 主轴估计 —— 前景像素 PCA，横放/竖放/斜放通吃；
3. 端电极检测 —— 轴向颜色剖面法：沿主轴分 48 个条带，每条带取 Lab 中值，
   与瓷体中部参考色做距离剖面 d(bin)。电极在两端形成双峰，
   对 d 做 Otsu 自适应阈值后从两端向内扩展得到电极宽度；
4. 兜底 —— 剖面无峰（瓷体电极颜色几乎一致）时按几何先验 15% 切分；
   也可手动指定 electrode_ratio 强制几何切分；
5. 形态学清理 + 软边羽化。

输出 RegionMap：0=背景 1=瓷体 2=端电极。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .utils import Rng, clip01

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


BODY, ELECTRODE, BG = 1, 2, 0


# --------------------------------------------------------------------------
# 基础形态学（cv2 优先，PIL 兜底）
# --------------------------------------------------------------------------
def _morph(mask: np.ndarray, ksize: int, op: str) -> np.ndarray:
    if ksize < 1:
        return mask
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize * 2 + 1,) * 2)
        m = (mask > 0).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN if op == "open" else cv2.MORPH_CLOSE, k)
        return m.astype(np.float32)
    from PIL import Image, ImageFilter

    size = ksize * 2 + 1
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    flt = ImageFilter.MinFilter(size) if op == "open" else ImageFilter.MaxFilter(size)
    img = img.filter(flt)
    flt2 = ImageFilter.MaxFilter(size) if op == "open" else ImageFilter.MinFilter(size)
    img = img.filter(flt2)
    return (np.asarray(img) > 127).astype(np.float32)


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.float32)
    if cv2 is not None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == best).astype(np.float32)
    from collections import deque

    h, w = m.shape
    seen = np.zeros_like(m, dtype=np.uint8)
    labels = np.zeros_like(m, dtype=np.int32)
    best_id, best_size, cur = -1, 0, 0
    for sy in range(h):
        for sx in range(w):
            if m[sy, sx] and not seen[sy, sx]:
                cur += 1
                q = deque([(sy, sx)])
                seen[sy, sx] = 1
                size = 0
                while q:
                    y, x = q.popleft()
                    labels[y, x] = cur
                    size += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = 1
                            q.append((ny, nx))
                if size > best_size:
                    best_size, best_id = size, cur
    return (labels == best_id).astype(np.float32) if best_id > 0 else np.zeros_like(m, dtype=np.float32)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(np.float32)
    if cv2 is not None:
        ff = m.copy()
        h, w = m.shape
        pad = np.zeros((h + 2, w + 2), np.uint8)
        flood = ff.copy()
        cv2.floodFill(flood, pad, (0, 0), 1)
        holes = (flood == 0)
        ff[holes] = 1
        return ff.astype(np.float32)
    out = m.copy()
    for row in out:
        idx = np.nonzero(row == 1)[0]
        if idx.size >= 2:
            row[idx[0]:idx[-1] + 1] = 1
    return out.astype(np.float32)


def _otsu_on_values(values: np.ndarray) -> float:
    """对任意一维取值集合做 Otsu（映射到 0..255）。"""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        return lo
    q = np.clip((v - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return lo + (_otsu_u8(q) / 255.0) * (hi - lo)


def _otsu_u8(gray: np.ndarray) -> float:
    if cv2 is not None:
        thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(thr)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = (np.arange(256) * hist).sum()
    wB = sumB = 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        var = wB * wF * ((sumB / wB) - ((sum_all - sumB) / wF)) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return float(best_t)


def _rgb_to_lab_f(rgb_u8: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    from PIL import Image

    return np.asarray(Image.fromarray(rgb_u8).convert("LAB"), dtype=np.float32)


def _feather(mask: np.ndarray, radius: float = 1.5) -> np.ndarray:
    if radius <= 0:
        return clip01(mask)
    if cv2 is not None:
        k = int(max(1, round(radius))) * 2 + 1
        return clip01(cv2.GaussianBlur(clip01(mask), (k, k), 0))
    from PIL import Image, ImageFilter

    img = Image.fromarray((clip01(mask) * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return clip01(np.asarray(img, dtype=np.float32) / 255.0)


# --------------------------------------------------------------------------
# 前景候选评分
# --------------------------------------------------------------------------
def _fg_candidate(gray_u8: np.ndarray, side_mask: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """对一侧阈值候选做形态学清理并打分。

    评分 = 实心度 × 尺寸项 × 环形惩罚 × 触边惩罚 × 圆角矩形度
    （产品是单一紧凑的圆角矩形块；背景常是「包住产品的环」，填洞后会吃掉整个画面）
    """
    fg = side_mask.astype(np.float32)
    if fg.sum() < 50:
        return np.zeros_like(fg), 0.0
    fg = _morph(fg, k, "close")
    cc = _largest_cc(fg)
    if cc.sum() < 50:
        return np.zeros_like(fg), 0.0
    area_before = float(cc.sum())
    filled = _fill_holes(cc)
    area_after = float(filled.sum())
    cc = _morph(filled, max(1, k // 2), "open")
    area = float(cc.sum())

    ys, xs = np.nonzero(cc > 0.5)
    if xs.size < 50:
        return np.zeros_like(fg), 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    fill = area / float(bw * bh)
    size_term = min(1.0, (area / (cc.size * 0.015)))

    # 环形惩罚：填洞显著增大 → 该候选是「背景环」而非产品
    ring_ratio = area_after / max(area_before, 1.0)
    ring_penalty = 0.25 if ring_ratio > 1.20 else 1.0

    # 触边惩罚：四边全触 → 多半是背景
    h, w = cc.shape
    top = (cc[0, :] > 0.5).any()
    bottom = (cc[h - 1, :] > 0.5).any()
    left = (cc[:, 0] > 0.5).any()
    right = (cc[:, w - 1] > 0.5).any()
    border_penalty = 0.4 if (top and bottom and left and right) else 1.0

    # 长宽比惩罚：MLCC 外形近似矩形（含圆角），极端细长多为噪声
    ar = max(bw, bh) / max(min(bw, bh), 1)
    ar_penalty = 1.0 if ar <= 6.0 else max(0.3, 6.0 / ar)

    # 矩形度：用外接框内的实际占比与「圆角矩形填充率」比较
    rect_penalty = 1.0 if fill >= 0.80 else max(0.45, fill / 0.80)

    score = fill * size_term * ring_penalty * border_penalty * ar_penalty * rect_penalty
    return cc, float(score)


def _thresholds_candidates(gray_u8: np.ndarray) -> List[float]:
    """生成多个阈值候选：全局 Otsu + 分位数，覆盖光照不均的情形。"""
    outs = []
    try:
        outs.append(_otsu_u8(gray_u8))
    except Exception:
        pass
    for q in (10, 20, 30, 70, 80, 90):
        outs.append(float(np.percentile(gray_u8, q)))
    uniq = []
    for t in outs:
        t = float(np.clip(t, 1, 254))
        if all(abs(t - u) > 3 for u in uniq):
            uniq.append(t)
    return uniq


def auto_find_product(rgb: np.ndarray, bg_side: str = "auto",
                      k: Optional[int] = None) -> Tuple[np.ndarray, float, dict]:
    """自动查找产品目标（多阈值候选 + 评分选优）。

    返回 (product_mask float32, score, diag)。
    """
    h, w = rgb.shape[:2]
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    if k is None:
        k = max(2, int(min(h, w) * 0.01))

    best = (np.zeros((h, w), np.float32), -1.0, {})
    thr_list = _thresholds_candidates(gray_u8)

    for thr in thr_list:
        for sign in ("bright", "dark"):
            if bg_side == "bright_fg" and sign == "dark":
                continue
            if bg_side == "dark_fg" and sign == "bright":
                continue
            cand = (gray_u8 > thr) if sign == "bright" else (gray_u8 <= thr)
            m, sc = _fg_candidate(gray_u8, cand, k)
            if sc > best[1]:
                best = (m, sc, {"thr": thr, "sign": sign})

    return best[0], best[1], best[2]


def refine_product_from_body(fg: np.ndarray, lab: np.ndarray,
                            body_hint: np.ndarray) -> np.ndarray:
    """当产品掩膜不完整时，用「类瓷体色」区域扩展补全产品掩膜。

    body_hint: 已知的瓷体参考区域（例如当前产品掩膜的中心块）。
    """
    sel = body_hint > 0.5
    if sel.sum() < 50:
        return fg
    ref = np.median(lab[sel], axis=0)
    # 与参考色足够接近的像素视为产品（瓷体/电极通常同一材质族）
    d = np.linalg.norm((lab - ref[None, None, :]) * np.array([0.6, 1.0, 1.0], np.float32),
                       axis=2)
    thr = float(np.percentile(d[sel], 92)) + 12.0
    similar = (d < thr).astype(np.float32)
    merged = np.maximum(fg, similar)
    merged = _morph(merged, max(2, int(min(fg.shape) * 0.006)), "close")
    merged = _largest_cc(merged)
    return _fill_holes(merged)


# --------------------------------------------------------------------------
# 轴向颜色剖面 → 端电极宽度
# --------------------------------------------------------------------------
def _electrode_widths(lab: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                      along_n: np.ndarray, n_bins: int = 48
                      ) -> Tuple[float, float, dict]:
    """返回 (左端电极宽度占比, 右端宽度占比, 诊断)。均为相对主轴总长的比例。"""
    diag = {}
    L = lab[ys, xs][:, 0]
    A = lab[ys, xs][:, 1]
    B = lab[ys, xs][:, 2]

    bin_idx = np.clip(((along_n + 1.0) / 2.0 * n_bins).astype(np.int32), 0, n_bins - 1)

    med = np.full((n_bins, 3), np.nan, np.float32)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() >= 8:
            med[b] = np.median(np.stack([L[m], A[m], B[m]], axis=1), axis=0)

    valid = ~np.isnan(med[:, 0])
    if valid.sum() < n_bins * 0.5:
        return 0.15, 0.15, {"reason": "剖面样本不足"}

    # 参考色 = 中部 30% 条带的中值
    c0, c1 = int(n_bins * 0.35), int(n_bins * 0.65)
    ref = np.median(med[c0:c1][valid[c0:c1]], axis=0)

    d = np.linalg.norm((med - ref[None]) * np.array([0.7, 1.0, 1.0]), axis=1)
    d[~valid] = np.nan
    # 填补无效 bin 并平滑（边缘复制填充，避免端点被零填充拉低）
    din = d.copy()
    idx = np.arange(n_bins)
    din[np.isnan(din)] = np.interp(idx[np.isnan(din)], idx[~np.isnan(d)], d[~np.isnan(d)])
    pad = np.pad(din, 1, mode="edge")
    ds = np.convolve(pad, np.ones(3) / 3.0, mode="valid")

    # Otsu 自适应阈值
    thr = _otsu_on_values(ds)
    if not np.isfinite(thr):
        thr = float(np.nanmedian(ds))
    diag["thr"] = float(thr)
    diag["d_max"] = float(np.nanmax(ds))
    diag["d_min"] = float(np.nanmin(ds))

    # 剖面无峰 → 电极瓷体颜色几乎一致
    if diag["d_max"] - diag["d_min"] < 4.0:
        return 0.15, 0.15, {"reason": "剖面无显著色差，几何兜底"}

    # 从两端向内游走，容忍连续 2 个 bin 的低值（电极/瓷体过渡带）
    left_w = 0.0
    miss = 0
    for b in range(n_bins):
        if ds[b] > thr:
            left_w = (b + 1) / n_bins
            miss = 0
        else:
            miss += 1
            if miss >= 3:
                break
    right_w = 0.0
    miss = 0
    for b in range(n_bins - 1, -1, -1):
        if ds[b] > thr:
            right_w = (n_bins - b) / n_bins
            miss = 0
        else:
            miss += 1
            if miss >= 3:
                break

    # 合理性约束
    if left_w > 0.45 or right_w > 0.45:
        return 0.15, 0.15, {"reason": "电极带宽异常，几何兜底"}
    if left_w < 0.02 and right_w < 0.02:
        return 0.15, 0.15, {"reason": "未检出电极，几何兜底"}

    return float(max(left_w, 0.015)), float(max(right_w, 0.015)), diag


# --------------------------------------------------------------------------
# 主分割
# --------------------------------------------------------------------------
@dataclass
class RegionResult:
    region: np.ndarray                      # (H,W) uint8: 0/1/2
    product: np.ndarray                     # float32 [0,1]
    body: np.ndarray                        # float32 [0,1] 软边
    electrode: np.ndarray                   # float32 [0,1] 软边
    ok: bool = True
    method: str = "profile"                 # profile / geometric / failed
    axis_angle_deg: float = 0.0
    coverage: float = 0.0
    elec_left: float = 0.0
    elec_right: float = 0.0
    notes: List[str] = field(default_factory=list)


def segment_mlcc(rgb: np.ndarray,
                 electrode_ratio: Optional[float] = None,
                 bg_side: str = "auto",
                 feather: float = 1.5,
                 user_region: Optional[np.ndarray] = None,
                 user_region_mode: str = "intersect") -> RegionResult:
    """分割 MLCC 区域。

    electrode_ratio: 手动指定端电极宽度占比（0.03~0.45，相对主轴总长），
                     None=自动剖面检测。分割不准时的手动兜底。
    bg_side: "auto" | "bright_fg" | "dark_fg" —— 前景亮度先验。
    user_region: 用户手绘区域掩膜（float32 [0,1]），可选。
    user_region_mode: "intersect"=缺陷限制在 手绘∩产品；
                      "union"=手绘区域并入产品（允许手动圈定背景里的目标）；
                      "override"=完全以手绘区域为准（忽略自动分割）。
    """
    h, w = rgb.shape[:2]
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    notes: list = []

    # ---------- 1. 自动查找产品（多阈值候选 + 评分选优） ----------
    k = max(2, int(min(h, w) * 0.01))
    fg, fg_score, fg_diag = auto_find_product(rgb, bg_side=bg_side, k=k)
    if fg_diag:
        notes.append(f"目标查找: {fg_diag.get('sign')}侧 thr={fg_diag.get('thr'):.0f} "
                     f"评分={fg_score:.2f}")

    cov = float(fg.mean())
    ok = 0.015 <= cov <= 0.92
    if not ok:
        notes.append(f"产品占比异常({cov:.2%})")

    ys, xs = np.nonzero(fg > 0.5)
    if xs.size < 200:
        region = np.zeros((h, w), np.uint8)
        region[int(h * 0.3):int(h * 0.7), int(w * 0.2):int(w * 0.8)] = BODY
        res = RegionResult(region=region, product=fg,
                           body=(region == BODY).astype(np.float32),
                           electrode=np.zeros((h, w), np.float32),
                           ok=False, method="failed", coverage=cov,
                           notes=notes + ["目标查找失败，请使用手绘区域模式"])
        if user_region is None:
            return res
        fg = res.product
        ys, xs = np.nonzero(fg > 0.5)
        if xs.size < 200:
            return res

    # ---------- 2. 主轴 PCA ----------
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    covm = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(covm)
    major = evecs[:, int(np.argmax(evals))]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    along = (pts - mean) @ major
    span = float(along.max() - along.min()) or 1.0
    along_n = (along - along.min()) / span * 2 - 1

    # ---------- 3. 电极宽度 ----------
    lab = _rgb_to_lab_f(rgb)
    if electrode_ratio is not None:
        r = float(np.clip(electrode_ratio, 0.03, 0.45))
        lw = rw = r
        method = "geometric"
        diag = {}
    else:
        lw, rw, diag = _electrode_widths(lab, ys, xs, along_n)
        method = "profile" if "兜底" not in diag.get("reason", "") else "geometric"
        if "reason" in diag:
            notes.append(diag["reason"])

    # ---------- 4. 生成掩膜 ----------
    elec_bool = np.zeros(fg.shape, dtype=bool)
    elec_bool[ys, xs] = (along_n < -1 + 2 * lw) | (along_n > 1 - 2 * rw)
    body_bool = (fg > 0.5) & (~elec_bool)

    body_m = _morph(body_bool.astype(np.float32), max(1, int(min(h, w) * 0.005)), "open")
    body_m = _largest_cc(body_m)
    body_m = _morph(body_m, max(1, int(min(h, w) * 0.005)), "close")

    elec_m = _morph(elec_bool.astype(np.float32), max(1, int(min(h, w) * 0.004)), "open")
    if cv2 is not None:
        n_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
            (elec_m > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(elec_m)
        if n_cc > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            for j in np.argsort(areas)[::-1][:2]:   # 两个端头
                keep += (labels_cc == (j + 1)).astype(np.float32)
        elec_m = keep
    elec_m = _morph(elec_m, max(1, int(min(h, w) * 0.005)), "close")

    product = clip01(np.maximum(body_m, elec_m))
    body_soft = _feather(body_m, feather) * product
    elec_soft = _feather(elec_m, feather) * product

    # 先做「瓷体 / 电极」软分配（两者互补，和为 1）。
    # ⚠️ 必须排在「用户手绘区域」之前：归一化是除以 (body+elec)，
    #    若先乘了手绘区域 ur 再归一化，ur 会被整除抵消（除零下限 1e-6 还会把
    #    产品羽化边缘 × 微弱手绘的像素重新放大回 1.0），导致限定区域形同虚设、
    #    缺陷落到手绘区之外。现在先归一化定表面，再由 ur 做硬性裁剪。
    tot = np.maximum(body_soft + elec_soft, 1e-6)
    body_soft = np.minimum(body_soft / tot, 1.0)
    elec_soft = np.minimum(elec_soft / tot, 1.0)

    # ---------- 5. 用户手绘区域 ----------
    if user_region is not None and user_region.max() > 0.1:
        ur = clip01(user_region)
        mode = (user_region_mode or "intersect").lower()
        if mode == "override":
            # 完全以手绘为准：手绘区域内按原分割结果区分瓷体/电极，
            # 分割未覆盖处按瓷体处理，保证缺陷一定能落在手绘区域内
            inside = (ur > 0.5)
            body_soft = np.where(inside, np.maximum(body_soft, ur * (~(elec_soft > 0.5) * 1.0)), 0)
            elec_soft = np.where(inside, elec_soft, 0)
            body_soft = clip01(body_soft) * inside
            elec_soft = clip01(elec_soft) * inside
            product = clip01(np.maximum(body_soft, elec_soft))
        elif mode == "union":
            add = clip01(ur) * (1.0 - product)
            body_soft = clip01(body_soft + add)
            product = clip01(np.maximum(product, ur))
        else:  # intersect（默认）
            # 与产品表面取交集：缺陷严格限制在「手绘区域 ∩ 产品表面」内
            body_soft = clip01(body_soft * ur)
            elec_soft = clip01(elec_soft * ur)
            product = clip01(product * ur)
        notes.append(f"已应用手绘区域（{mode}）")

    region = np.zeros((h, w), np.uint8)
    region[body_soft > 0.5] = BODY
    region[elec_soft > 0.5] = ELECTRODE

    return RegionResult(region=region, product=product, body=body_soft,
                        electrode=elec_soft, ok=ok, method=method,
                        axis_angle_deg=angle, coverage=cov,
                        elec_left=lw, elec_right=rw, notes=notes)


def region_overlay(rgb: np.ndarray, res: RegionResult,
                   body_alpha: float = 0.30, elec_alpha: float = 0.38) -> np.ndarray:
    """分割结果着色叠加：绿=瓷体，蓝=端电极。"""
    out = rgb.astype(np.float32).copy()
    b = res.body[:, :, None]
    e = res.electrode[:, :, None]
    body_col = np.array([60, 210, 120], np.float32)[None, None, :]
    elec_col = np.array([70, 140, 255], np.float32)[None, None, :]
    out = out * (1 - b * body_alpha) + body_col * (b * body_alpha)
    out = out * (1 - e * elec_alpha) + elec_col * (e * elec_alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def sample_point(mask: np.ndarray, rng: Rng, margin: float = 0.05) -> Tuple[float, float]:
    """从掩膜内随机取一点 (x, y)，避开边缘 margin 比例。"""
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size == 0:
        return 0.0, 0.0
    h, w = mask.shape
    mx, my = int(w * margin), int(h * margin)
    keep = (xs >= mx) & (xs < w - mx) & (ys >= my) & (ys < h - my)
    if keep.sum() < 10:
        keep = np.ones_like(keep)
    i = rng.randint(0, int(keep.sum()) - 1)
    return float(xs[keep][i]), float(ys[keep][i])
