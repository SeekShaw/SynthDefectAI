"""快速复核「明显度是否拉得开」+ 边缘柔和度（verify_region C 段的快版）。

用法::

    python tools/check_severity.py [样例名]

判定：obvious / subtle 的缺陷像素量比需≥ 1.5（verify_region 的门槛）。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.defects import DEFECT_KINDS, apply_defects  # noqa: E402
from core.mlcc_region import segment_mlcc  # noqa: E402
from core.utils import imread  # noqa: E402

SEVS = ("obvious", "medium", "subtle", "mixed")


def band_ratio(mask: np.ndarray) -> float:
    m = np.clip(mask.astype(np.float32), 0, 1)
    band = float(((m > 0.02) & (m < 0.98)).sum())
    return band / max(float((m > 0.02).sum()), 1.0)


def edge_grad(mask: np.ndarray) -> float:
    m = np.clip(mask.astype(np.float32), 0, 1)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    band = (m > 0.02) & (m < 0.98)
    return float(g[band].mean()) if band.sum() > 10 else 0.0


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "mlcc_h_std.png"
    rgb = imread(os.path.join(HERE, "samples_mlcc", name))
    res = segment_mlcc(rgb, bg_side="auto")
    rm = {"body": res.body, "electrode": res.electrode}

    print(f"样例 {name} {rgb.shape}")
    print(f"{'明显度':>8}{'缺陷占比':>11}" + "".join(f"{k:>11}" for k in DEFECT_KINDS))
    cov, soft = {}, {}
    for sev in SEVS:
        settings = {k: {"enable": True, "target": "both", "severity": sev}
                    for k in DEFECT_KINDS}
        acc = {k: [] for k in DEFECT_KINDS}
        for seed in range(6):
            _, applied = apply_defects(rgb, settings, seed=seed, region_masks=rm)
            for r in applied:
                acc[r.kind].append(int((r.mask > 0.35).sum()))
                if seed == 0:
                    soft.setdefault(r.kind, []).append(
                        (band_ratio(r.mask), edge_grad(r.mask)))
        tot = sum(float(np.mean(v)) for v in acc.values())
        cov[sev] = tot / res.product.size
        print(f"{sev:>8}{cov[sev]:>11.3%}" +
              "".join(f"{np.mean(acc[k]):>11.0f}" for k in DEFECT_KINDS))

    ratio = cov["obvious"] / max(cov["subtle"], 1e-9)
    ok = ratio >= 1.5
    print(f"\n明显/不明显 缺陷量比 = {ratio:.2f}x  {'OK' if ok else 'FAIL（门槛 1.5）'}")

    print("\n边缘柔和度（过渡带占比 / 边界梯度，越低越柔）")
    for k, vals in soft.items():
        b = float(np.mean([v[0] for v in vals]))
        g = float(np.mean([v[1] for v in vals]))
        print(f"    {k:<9} 过渡带={b:.0%}  边界梯度={g:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
