"""回归校验：手绘限定区域必须真正约束脏污/异色/缺角的生成位置。

覆盖两个历史缺陷：
1. `segment_mlcc` 先乘手绘区域再归一化 → ur 被整除抵消，限定区域形同虚设；
2. `MainWindow._ensure_segmentation` 缓存键只用 dm.sum()（面积），
   等面积不同位置的手绘区域会复用上一块的分割结果。

判定标准：缺陷掩膜不得落在「真正没有手绘像素」的位置（dm <= 0.001）。
画笔抗锯齿软边（0 < dm < 0.5）属正常羽化，不计入越界。
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

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import gui.main_window as mw  # noqa: E402
from core.defects import apply_defects  # noqa: E402

HARD_OUT = 0.001          # 低于此值视为「完全没有手绘像素」= 真正越界
KINDS = ("smudge", "discolor", "chip")

fails = []


def check(cond, msg):
    print(("  [OK ] " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


def phantom(region_masks, dm):
    """「幽灵像素」：真正传给生成器的区域掩膜越出了手绘范围。

    注意：现在的架构里 ``segment_mlcc`` **不**再吃手绘区域，``seg.product``
    是完整的产品真实轮廓（用于画轮廓线），它超出手绘范围是完全正常的。
    真正必须被约束的是 ``region_masks`` —— 也就是 ``MainWindow._region_masks_for``
    算出来、直接喂给 ``apply_defects`` 的那份掩膜。它必须落在手绘区域内。
    """
    if not region_masks:
        return 0
    region = np.maximum(region_masks["body"], region_masks["electrode"])
    return int(((region > 0.5) & (dm < 0.5)).sum())


def main():
    app = QApplication.instance() or QApplication([])
    win = mw.MainWindow()
    import glob
    win.add_inputs(sorted(glob.glob(os.path.join("samples_mlcc", "*.png"))))
    win.lst.setCurrentRow(0)
    app.processEvents()
    if win.current_image is None:
        print("没有可用样例，无法校验")
        return 1

    H, W = win.current_image.shape[:2]
    view = win.preview.views[0]

    # 本脚本需同时启用 脏污/异色/崩缺 三类；默认上限 1 会触发互斥弹窗
    # （offscreen 下模态阻塞），因此先把「同时生成缺陷种类」上调。
    win.sp_kind_limit.setValue(len(KINDS))

    def brush(fy):
        view.set_mode(view.MODE_BRUSH)
        view.set_brush_radius(16)
        y = int(fy * H)
        view._strokes.append([(0.20 * W, y), (0.55 * W, y)])

    def rect(fx, fy):
        view.set_mode(view.MODE_RECT)
        view._rects.append((fx * W, fy * H, 0.22 * W, 0.22 * H, 0.0))

    def poly():
        view.set_mode(view.MODE_POLY)
        view._polys.append([(0.30 * W, 0.30 * H), (0.55 * W, 0.32 * H),
                            (0.52 * W, 0.55 * H), (0.28 * W, 0.50 * H)])

    def run(setup, seed=3, mode=None):
        view.clear_regions()
        app.processEvents()
        if mode is not None:
            win.preview.cmb_posmode.setCurrentIndex(
                win.preview.cmb_posmode.findData(mode))
            app.processEvents()
        win.preview.cb_draw.setChecked(True)
        setup()
        app.processEvents()
        for k in win.pages:
            win.pages[k].enable.setChecked(k in KINDS)
        win.sp_seed.setValue(seed)
        win.refresh_preview()
        app.processEvents()
        dm = win._draw_mask()
        seg = win._ensure_segmentation()
        # 直接复用界面真实使用的区域计算（而不是在这里另写一份），
        # 否则测试验证的就不是产品真正跑的那条路径了。
        rm = win._region_masks_for(seg)
        if dm is not None and rm is None:
            return "off_product", dm, 0
        # 「幽灵像素」：真正用于生成缺陷的区域掩膜越出了手绘范围
        ph = phantom(rm, dm) if dm is not None else 0
        img, applied = apply_defects(
            win.current_image, win.current_settings(), seed=seed,
            region_masks=rm, region_hint=win._draw_hint())
        worst = 0
        for r in applied:
            m = r.mask > 0.05
            if not m.any():
                continue
            worst = max(worst, int((m & (dm <= HARD_OUT)).sum()))
        return worst, dm, ph

    print("[1] 画笔涂抹：不同纵向位置（等面积，考验缓存键）")
    for fy in (0.30, 0.40, 0.55, 0.70):
        worst, dm, ph = run(lambda fy=fy: brush(fy))
        check(worst == 0, "笔迹 y=%.2fH：硬越界像素=%d" % (fy, worst))
        check(ph == 0, "笔迹 y=%.2fH：生成用区域掩膜越出手绘=%d" % (fy, ph))

    print("[2] 矩形框：不同位置")
    for fx, fy in ((0.38, 0.45), (0.25, 0.30), (0.55, 0.60), (0.70, 0.50)):
        worst, dm, ph = run(lambda fx=fx, fy=fy: rect(fx, fy))
        check(worst == 0, "矩形 (%.2f,%.2f)：硬越界像素=%d" % (fx, fy, worst))
        check(ph == 0, "矩形 (%.2f,%.2f)：生成用区域掩膜越出手绘=%d" % (fx, fy, ph))

    print("[3] 多边形")
    worst, dm, ph = run(poly)
    check(worst == 0, "多边形：硬越界像素=%d" % worst)
    check(ph == 0, "多边形：生成用区域掩膜越出手绘=%d" % ph)

    print("[4] 不同随机种子（笔迹 y=0.35H）")
    for s in (1, 7, 42, 99):
        worst, dm, ph = run(lambda: brush(0.35), seed=s)
        check(worst == 0, "seed=%d：硬越界像素=%d" % (s, worst))

    print("[5] 位置方式对「画在产品外」的行为差异")
    # 默认已是「按图像坐标位置」：区域就是手绘掩膜本身，画在产品外也照常生成
    worst, dm, ph = run(lambda: brush(0.22), mode="pixel")
    check(worst != "off_product" and worst == 0,
          "按图像坐标位置：画在产品外仍按手绘坐标生成且不越界（返回 %r）" % (worst,))
    # 「限定在产品轮廓内」：交集为空 → 应阻止生成
    worst, dm, ph = run(lambda: brush(0.22), mode="product")
    check(worst == "off_product",
          "限定在产品轮廓内：区域未覆盖产品时已阻止生成（返回 %r）" % (worst,))

    print("[6] 未画区域 → 仍应整图随机")
    view.clear_regions()
    app.processEvents()
    check(win._draw_mask() is None, "未绘制时 _draw_mask() 为 None")
    for k in win.pages:
        win.pages[k].enable.setChecked(k == "smudge")
    win.refresh_preview()
    app.processEvents()
    seg = win._ensure_segmentation()
    img, applied = apply_defects(
        win.current_image, win.current_settings(), seed=3,
        region_masks={"body": seg.body, "electrode": seg.electrode})
    got = bool(applied) and bool((applied[0].mask > 0.05).any())
    check(got, "未画区域时脏污仍能生成（覆盖整图）")

    print()
    if fails:
        print("VERIFY_REGION_PLACEMENT_FAILED (%d 项)" % len(fails))
        for f in fails:
            print("   - " + f)
        return 1
    print("ALL_REGION_PLACEMENT_CHECKS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())