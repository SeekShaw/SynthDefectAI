"""验证「缺陷样本增强模式」核心逻辑（headless，无需 GUI）。

覆盖：
1. apply_augmentation 按 n_images 生成多张缺陷图（每张缺陷数由 settings「个数」决定）；
2. 缺陷主要落在手绘区域内（允许少量溢出）；
3. 同种子可完全复现；
4. 无区域（整图）时也能生成；
5. 批量引擎 BatchConfig(augment=True, augment_images=N) 产出 N 张图 + 掩膜 + 标注。
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.defects import apply_augmentation  # noqa: E402
from core.batch import BatchConfig, run_batch  # noqa: E402
from core.utils import imwrite  # noqa: E402

H, W = 480, 640


def make_image():
    img = np.full((H, W, 3), 205, np.uint8)
    img[40:200, 120:520] = 190  # 瓷体色块
    return img


def make_region():
    m = np.zeros((H, W), np.float32)
    m[60:420, 120:520] = 1.0  # 大区域，尽量保证每个缺陷都能落进去
    return m


def make_settings(counts: dict):
    """构造缺陷生成算法参数：counts 形如 {'scratch':3,'smudge':2,...}。"""
    key = {"scratch": "count", "smudge": "blob_count",
           "discolor": "region_count", "chip": "count"}
    cfg = {}
    for k in ("scratch", "smudge", "discolor", "chip"):
        c = counts.get(k, 0)
        d = {"enable": c > 0, "strength": 0.6, "target": "both", "severity": "mixed"}
        d[key[k]] = max(1, c)
        cfg[k] = d
    return cfg


def main():
    img = make_image()
    region = make_region()

    settings = make_settings({"scratch": 3, "smudge": 2, "discolor": 1, "chip": 1})
    expect = 3 + 2 + 1 + 1  # 每张图的缺陷总数（来自算法参数「个数」）

    # ---- 1. 生成 N 张缺陷图，每张缺陷数 = settings 个数之和 ----
    N = 6
    results = apply_augmentation(img, region=region, n_images=N, seed=123,
                                 settings=settings, severity="mixed")
    assert isinstance(results, list) and len(results) == N, \
        f"应返回 {N} 张图，实际 {len(results)}"
    for i, (out, applied) in enumerate(results):
        assert out.shape == img.shape, "输出尺寸应与原图一致"
        assert len(applied) >= int(expect * 0.8), \
            f"第{i}张缺陷数应≈{expect}，实际 {len(applied)}"
    kinds = set()
    for r in results[0][1]:
        kinds.add(r.kind)
    print(f"[1] 生成 {N} 张，每张≈{expect}个缺陷，类型涵盖={sorted(kinds)}")

    # ---- 2. 缺陷主要落在手绘区域内 ----
    total_px = inside_px = 0
    for _, applied in results:
        for r in applied:
            m = (r.mask > 0.05)
            total_px += int(m.sum())
            inside_px += int((m & (region > 0.5)).sum())
    assert total_px > 500, "缺陷总像素应明显大于噪声"
    ratio = inside_px / max(total_px, 1)
    assert ratio > 0.7, f"缺陷应主要落在区域内，实际 {ratio:.2f}"
    print(f"[2] 区域内占比={ratio:.2f} ✓")

    # ---- 3. 同种子可完全复现 ----
    results_b = apply_augmentation(img, region=region, n_images=N, seed=123,
                                  settings=settings, severity="mixed")
    for (a, _), (b, _) in zip(results, results_b):
        assert np.array_equal(a, b), "同种子应完全复现"
    print("[3] 同种子复现一致 ✓")

    # ---- 4. 无区域（整图）也能生成 ----
    results_c = apply_augmentation(img, region=None, n_images=4, seed=7,
                                   settings=settings, severity="mixed")
    assert len(results_c) == 4 and all(len(a) >= 3 for _, a in results_c), \
        "整图随机也应生成缺陷"
    print(f"[4] 整图随机：4 张均生成缺陷 ✓")

    # ---- 5. 数量确实由 settings 个数驱动 ----
    few = make_settings({"scratch": 1})
    res_few = apply_augmentation(img, region=region, n_images=3, seed=5,
                                 settings=few, severity="mixed")
    avg_few = sum(len(a) for _, a in res_few) / len(res_few)
    avg_many = sum(len(a) for _, a in results) / len(results)
    assert avg_few < avg_many, f"少个数应明显少于多个数（{avg_few} vs {avg_many}）"
    print(f"[5] 数量随 settings 个数变化：少={avg_few:.1f} < 多={avg_many:.1f} ✓")

    # ---- 6. 批量引擎：augment_images=N ----
    tmp = tempfile.mkdtemp(prefix="aug_verify_")
    src = os.path.join(tmp, "sample.png")
    imwrite(src, img)
    cfg = BatchConfig(
        input_paths=[src],
        output_dir=os.path.join(tmp, "out"),
        images_per_input=2,
        seed=42,
        augment=True,
        augment_region=region.copy(),
        augment_images=N,
        augment_severity="mixed",
        save_defect=True, save_mask=True, save_preview=True,
        save_region=True, save_yolo=True, save_voc=True, save_coco=True,
        use_region=False,
    )
    info = run_batch(cfg)
    assert info["images_generated"] == N, \
        f"应生成 {N} 张，实际 {info['images_generated']}"
    assert info["augment_mode"] is True
    img_dir = os.path.join(cfg.output_dir, "images")
    files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    assert len(files) == N, f"images/ 应有 {N} 张，实际 {len(files)}"
    mask_root = os.path.join(cfg.output_dir, "masks")
    mask_kinds = [d for d in os.listdir(mask_root)
                  if os.path.isdir(os.path.join(mask_root, d)) and
                  any(f.endswith(".png") for f in os.listdir(os.path.join(mask_root, d)))]
    assert mask_kinds, "应至少生成一类缺陷掩膜"
    print(f"[6] 批量增强：生成 {info['images_generated']} 张，"
          f"类别掩膜={sorted(mask_kinds)}，YOLO/VOC/COCO 已写出 ✓")

    print("\nALL_AUGMENT_CHECKS_PASS")


if __name__ == "__main__":
    main()
