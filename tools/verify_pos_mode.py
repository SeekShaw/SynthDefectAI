"""回归校验：「缺陷位置方式」两种模式的行为。

- ``POS_MODE_PRODUCT``（限定在产品轮廓内）：缺陷 = 产品表面 ∩ 手绘区域；
  手绘区域没画到产品上时应给出明确提示，而不是把缺陷洒满整图。
- ``POS_MODE_PIXEL``（按图像坐标位置）：缺陷直接落在手绘区域的像素坐标上，
  **完全不依赖产品识别** —— 即使产品识别彻底失败也不影响落点。

另含产品轮廓 / 限定范围叠加层的校验（能画出、颜色可改、开关有效）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import glob  # noqa: E402

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import gui.main_window as mw  # noqa: E402
from core.defects import apply_defects  # noqa: E402
from core.mlcc_region import product_outline_overlay, segment_mlcc  # noqa: E402
from gui.preview_pane import POS_MODE_PIXEL, POS_MODE_PRODUCT  # noqa: E402

HARD_OUT = 0.001
fails = []


def check(cond, msg):
    print(("  [OK ] " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


class _DeadSeg:
    """模拟「产品识别彻底失败」：body/electrode/product 全空。"""

    ok = False
    notes = ["forced-fail"]

    def __init__(self, shape):
        z = np.zeros(shape, np.float32)
        self.body = z
        self.electrode = z.copy()
        self.product = z.copy()


def main():
    app = QApplication.instance() or QApplication([])
    win = mw.MainWindow()
    win.add_inputs(sorted(glob.glob(os.path.join("samples_mlcc", "*.png"))))
    if win.lst.count() == 0:
        print("没有可用样例，无法校验")
        return 1
    win.lst.setCurrentRow(0)
    app.processEvents()
    H, W = win.current_image.shape[:2]
    view = win.preview.views[0]

    def set_mode(mode):
        win.preview.cmb_posmode.setCurrentIndex(
            win.preview.cmb_posmode.findData(mode))
        app.processEvents()

    def draw_rect(rect):
        view.clear_regions()
        app.processEvents()
        win.preview.cb_draw.setChecked(True)
        view.set_mode(view.MODE_RECT)
        view._rects.append(rect)
        app.processEvents()
        for k in win.pages:
            win.pages[k].enable.setChecked(k == "smudge")
        win.sp_seed.setValue(3)
        win.refresh_preview()
        app.processEvents()

    def info():
        return win.preview.lbl_info.text()

    in_rect = (0.35 * W, 0.44 * H, 0.20 * W, 0.22 * H, 0.0)   # 压在产品上
    out_rect = (0.35 * W, 0.09 * H, 0.20 * W, 0.09 * H, 0.0)  # 完全在产品外

    print("[1] 产品轮廓叠加层")
    set_mode(POS_MODE_PRODUCT)
    draw_rect(in_rect)
    arr = win.preview._overlay_arr
    check(arr is not None and arr.shape == (H, W, 4), "叠加层尺寸与图像一致")
    check(arr is not None and int((arr[:, :, 3] > 0).sum()) > 0,
          "叠加层已画出内容（产品轮廓 + 限定范围）")
    # 轮廓颜色可改
    win.preview._outline_color = (0, 255, 0)
    win.preview.overlayChanged.emit()
    app.processEvents()
    arr2 = win.preview._overlay_arr
    green = (arr2[:, :, 1] > 200) & (arr2[:, :, 0] < 60) if arr2 is not None else None
    check(bool(green is not None and green.any()), "轮廓颜色可修改（改为绿色后出现绿像素）")
    win.preview._outline_color = (255, 80, 80)
    win.preview.overlayChanged.emit()
    app.processEvents()
    # 线宽可改
    win.preview.sp_outline_w.setValue(6)
    app.processEvents()
    w6 = int((win.preview._overlay_arr[:, :, 3] > 0).sum())
    win.preview.sp_outline_w.setValue(1)
    app.processEvents()
    w1 = int((win.preview._overlay_arr[:, :, 3] > 0).sum())
    check(w6 > w1, "线宽可修改（6px 覆盖 %d > 1px 覆盖 %d）" % (w6, w1))
    win.preview.sp_outline_w.setValue(2)
    # 显隐开关
    win.preview.cb_show_outline.setChecked(False)
    app.processEvents()
    only_draw = win.preview._overlay_arr
    n_only_draw = int((only_draw[:, :, 3] > 0).sum()) if only_draw is not None else 0
    check(n_only_draw >= 0 and n_only_draw < w1,
          "可单独关闭产品轮廓（仅剩限定范围）")
    win.preview.cb_show_outline.setChecked(True)
    app.processEvents()

    print("[2] 模式①产品轮廓内：区域压在产品边缘 → 交集生效")
    set_mode(POS_MODE_PRODUCT)
    # 取一条跨越产品上边界的窄条：交集必然小于手绘区域本身，
    # 这样才能证明「区域 = 产品 ∩ 手绘」而不是直接照搬手绘掩膜。
    seg0 = win._ensure_segmentation()
    ptop = int(np.nonzero(seg0.product[:, seg0.product.shape[1] // 2] > 0.5)[0].min())
    edge_rect = (0.45 * W, float(ptop), 0.30 * W, 0.05 * H, 0.0)
    draw_rect(edge_rect)
    check("位置[产品轮廓内]" in info(), "信息栏标注为产品轮廓模式")
    dm = win._draw_mask()
    rm = win._region_masks_for(seg0)
    check(rm is not None, "得到有效区域掩膜")
    n_dm = int((dm > 0.5).sum())
    n_rm = int((rm["body"] > 0.5).sum() + (rm["electrode"] > 0.5).sum())
    check(n_rm < n_dm,
          "交集小于手绘区域（手绘 %d px → 交集 %d px，说明按产品裁剪生效）"
          % (n_dm, n_rm))
    img, applied = apply_defects(win.current_image, win.current_settings(),
                                 seed=3, region_masks=rm)
    if applied:
        m = applied[0].mask > 0.05
        check(int((m & (dm <= HARD_OUT)).sum()) == 0, "缺陷未越出手绘区域")
        check(int((m & (seg0.product <= HARD_OUT)).sum()) == 0,
              "缺陷未越出产品轮廓")
    else:
        check(False, "应生成缺陷")

    print("[3] 模式①产品轮廓内：区域画在产品外 → 明确提示，不生成")
    draw_rect(out_rect)
    check("未覆盖产品表面" in info() or "没有可用" in info(),
          "给出「手绘区域未覆盖产品表面」提示：%s" % info()[:40])

    print("[4] 模式②按图像坐标位置：区域画在产品外 → 仍正常生成")
    set_mode(POS_MODE_PIXEL)
    draw_rect(out_rect)
    check("位置[图像坐标]" in info(), "信息栏标注为图像坐标模式")
    dm = win._draw_mask()
    rm = win._region_masks_for(win._ensure_segmentation())
    check(rm is not None and np.array_equal(rm["body"], dm),
          "区域掩膜就是手绘掩膜本身")
    img, applied = apply_defects(win.current_image, win.current_settings(),
                                 seed=3, region_masks=rm)
    check(bool(applied), "在产品外也能生成缺陷（不受产品识别限制）")
    if applied:
        m = applied[0].mask > 0.05
        ys, xs = np.nonzero(dm > 0.5)
        inside = (xs.min() <= xs.max() and ys.min() <= ys.max())
        out = int((m & (dm <= HARD_OUT)).sum())
        check(out == 0, "缺陷完全落在手绘区域内（越界 %d）" % out)
        yy, xx = np.nonzero(m)
        check(xx.min() >= xs.min() and xx.max() <= xs.max()
              and yy.min() >= ys.min() and yy.max() <= ys.max(),
              "缺陷 bbox 完全在手绘矩形 bbox 之内")

    print("[5] 模式②：产品识别彻底失败也不影响落点（核心诉求）")
    draw_rect(in_rect)
    dm = win._draw_mask()
    ys, xs = np.nonzero(dm > 0.5)
    bb = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    orig = win._ensure_segmentation
    try:
        win._ensure_segmentation = lambda: _DeadSeg((H, W))
        win.refresh_preview()
        app.processEvents()
        rm = win._region_masks_for(win._ensure_segmentation())
        check(rm is not None and np.array_equal(rm["body"], dm),
              "识别失败时区域掩膜仍是手绘掩膜")
        img, applied = apply_defects(win.current_image, win.current_settings(),
                                     seed=3, region_masks=rm)
        check(bool(applied), "识别失败时仍能生成缺陷")
        if applied:
            m = applied[0].mask > 0.05
            yy, xx = np.nonzero(m)
            check(int((m & (dm <= HARD_OUT)).sum()) == 0, "识别失败时缺陷仍不越界")
            check(xx.min() >= bb[0] and xx.max() <= bb[1]
                  and yy.min() >= bb[2] and yy.max() <= bb[3],
                  "识别失败时缺陷 bbox 仍等于手绘区域 bbox 之内")
    finally:
        win._ensure_segmentation = orig
        win._seg_key = None
        win._seg_res = None

    print("[6] 未画区域：两种模式都应整图随机")
    for mode in (POS_MODE_PRODUCT, POS_MODE_PIXEL):
        set_mode(mode)
        view.clear_regions()
        app.processEvents()
        for k in win.pages:
            win.pages[k].enable.setChecked(k == "smudge")
        win.refresh_preview()
        app.processEvents()
        check(win._draw_mask() is None, "[%s] 未绘制时掩膜为 None" % mode)
        seg = win._ensure_segmentation()
        rm = win._region_masks_for(seg)
        img, applied = apply_defects(win.current_image, win.current_settings(),
                                     seed=3, region_masks=rm)
        check(bool(applied) and bool((applied[0].mask > 0.05).any()),
              "[%s] 未画区域时仍能整图生成" % mode)

    print("[7] 叠加层生成函数")
    seg = segment_mlcc(win.current_image)
    dm = np.zeros((H, W), np.float32)
    dm[H // 4: H // 2, W // 4: W // 2] = 1.0
    rgba = product_outline_overlay((H, W), product=seg.product, draw_region=dm,
                                   color=(255, 80, 80), width=2)
    check(rgba.shape == (H, W, 4) and rgba.dtype == np.uint8, "叠加层形状/类型正确")
    px = rgba[rgba[:, :, 3] > 200][:, :3]
    check(len(px) > 0, "叠加层有可见像素")
    uniq = {tuple(int(x) for x in c) for c in np.unique(px, axis=0)}
    check((255, 80, 80) in uniq, "产品轮廓使用指定颜色 (255,80,80)")
    empty = product_outline_overlay((H, W))
    check(int((empty[:, :, 3] > 0).sum()) == 0, "无输入时返回全透明图层")

    print()
    if fails:
        print("VERIFY_POS_MODE_FAILED (%d 项)" % len(fails))
        for f in fails:
            print("   - " + f)
        return 1
    print("ALL_POS_MODE_CHECKS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())