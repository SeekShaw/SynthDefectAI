"""验证异色缺陷的颜色「设置 + 调整」功能。

- 随机 / 预设色调（泛黄/泛蓝/泛绿/泛红/泛紫/褪色）/ 自定义目标色 均能正常生成；
- 自定义目标色：缺陷区平均 RGB 应明显向所选目标色移动（距离小于原图到目标的距离）；
- 各预设色调应产生可区分的色偏方向。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.defects import DiscolorDefect, rgb_to_lab  # noqa: E402
from core.utils import Rng  # noqa: E402


def make_image(h=200, w=200, base=(170, 170, 170)):
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.clip(base[c] + rng.integers(-6, 6, (h, w)), 0, 255).astype(np.uint8)
    return img


def mean_rgb_in_mask(img, mask):
    m = mask > 0.5
    if m.sum() == 0:
        return None
    return img[m].reshape(-1, 3).mean(axis=0)


def dist(a, b):
    return float(np.sqrt(((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2).sum()))


def render(mode, target=None, seed=1, strength=0.8):
    img = make_image()
    params = {"region_scale": 0.5, "lum_shift": 0.0, "sat_gain": 1.25,
              "edge_soft": 0.45, "edge_blur": 1.0}
    if mode == "custom":
        params["color"] = "custom"
        params["color_rgb"] = list(target)
    else:
        params["color"] = mode
    res = DiscolorDefect(strength=strength, severity_mult=1.0, **params).render(
        img, Rng(seed))
    return img, res


def main():
    ok = True

    # 1) 所有模式都能生成合法图像
    modes = ["random", "yellow", "blue", "green", "red", "purple", "fade"]
    for md in modes:
        img, res = render(md)
        assert res.image.dtype == np.uint8 and res.image.shape == img.shape, \
            f"{md} 输出异常"
        assert res.mask.max() > 0.1, f"{md} 未产生有效掩膜"
    print("[1] 所有预设/随机色调均正常生成 OK")

    # 2) 自定义目标色：缺陷区平均色应向目标移动
    targets = {
        "红": [220, 30, 30],
        "绿": [40, 200, 60],
        "蓝": [40, 90, 230],
        "橙": [240, 150, 30],
    }
    for name, tgt in targets.items():
        img, res = render("custom", target=tgt, strength=0.9)
        src_mean = mean_rgb_in_mask(img, res.mask)
        new_mean = mean_rgb_in_mask(res.image, res.mask)
        d_src = dist(src_mean, tgt)
        d_new = dist(new_mean, tgt)
        moved = d_new < d_src * 0.6   # 至少拉近到 60% 以内
        print(f"    {name:>2} 目标{tgt} 原距{d_src:.0f} → 新距{d_new:.0f}  {'OK' if moved else 'FAIL'}")
        ok = ok and moved
        # 主导通道方向应与目标一致（如红色目标 → R 应为三通道最大）
        assert new_mean.argmax() == np.asarray(tgt).argmax(), \
            f"{name} 主色通道与目标不一致"
    print("[2] 自定义目标色：缺陷区平均色向目标移动 OK" if ok else "[2] FAIL")

    # 3) 不同预设应产生不同色偏（泛黄 vs 泛蓝 的平均 B 通道 LAB 应相反方向）
    _, r_y = render("yellow")
    _, r_b = render("blue")
    lab_y = rgb_to_lab(r_y.image)[r_y.mask > 0.5].reshape(-1, 3)[:, 1:].mean(0)
    lab_b = rgb_to_lab(r_b.image)[r_b.mask > 0.5].reshape(-1, 3)[:, 1:].mean(0)
    # 泛黄应让 B(LAB) 更大（更黄），泛蓝应让 B 更小
    sep = (lab_y[1] - lab_b[1])  # B 通道差
    print(f"    泛黄 vs 泛蓝 LAB-B 差 = {sep:.1f}（应明显 >0）")
    ok = ok and sep > 5

    # 4) 同目标同种子可复现
    _, a = render("custom", target=[240, 150, 30], seed=7)
    _, b = render("custom", target=[240, 150, 30], seed=7)
    repro = bool(np.array_equal(a.image, b.image))
    print(f"[4] 同目标同种子复现 = {repro}")
    ok = ok and repro

    print("\nALL_DISCOLOR_COLOR_CHECKS_OK" if ok else "\nDISCOLOR_COLOR_CHECK_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
