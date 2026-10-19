# -*- coding: utf-8 -*-
"""实证探针：调整位置后保存的文件，到底等于三栏中的哪一栏？

流程（用户真实操作）：
  1. 载图，启用 smudge，画矩形区域（stub）
  2. 生成预览 -> 缓存三栏内容（原图 / 区域与缺陷 mid / 缺陷图 comp）
  3. ↑ 微调 5px
  4. 点「保存缺陷图」
  5. 对保存文件分别与以下候选取最大像素差：
     a) 缺陷图（第三栏，调整后）      <- 应当匹配
     b) 缺陷图（第三栏，调整前）
     c) 区域与缺陷（第二栏 mid，调整后）
     d) 原图
"""
import os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox

QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)

from core.utils import imread
from gui.main_window import MainWindow

OUT = os.path.join(ROOT, "build", "out_probe_which")
if os.path.isdir(OUT):
    for f in glob.glob(os.path.join(OUT, "*.png")):
        os.remove(f)


def maxdiff(a, b):
    if a is None or b is None:
        return -1.0
    if a.shape != b.shape:
        return -2.0
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())


def main():
    app = QApplication([])
    win = MainWindow()
    samples = sorted(glob.glob(os.path.join(ROOT, "samples_mlcc", "*.png")))
    win.add_inputs([samples[0]])
    win.lst.setCurrentRow(0)
    app.processEvents()

    win._mutual_lock = True
    win.pages["smudge"].enable.setChecked(True)
    win._mutual_lock = False
    app.processEvents()

    # stub：画一个覆盖产品上半部的矩形区域（有交集，正常生成）
    seg = win._ensure_segmentation()
    h, w = seg.body.shape
    rect = np.zeros((h, w), np.float32)
    rect[:, :] = 1.0
    win.preview.draw_enabled = lambda: True
    win.preview.has_drawn_regions = lambda: True
    win._draw_mask = lambda kind=None: rect.copy()
    win._draw_paths = lambda: []

    win.refresh_preview()
    app.processEvents()
    assert win._defect_shift_base is not None, "未生成缺陷"

    # 拦截三栏内容：记录 set_images 的最新参数（orig, mid, comp）
    latest = {}
    _orig_set = win.preview.set_images

    def _spy(o, m, c):
        latest["orig"], latest["mid"], latest["comp"] = o, m, c
        return _orig_set(o, m, c)

    win.preview.set_images = _spy

    # 调整：↓ 5 次（向下 5px）
    for _ in range(5):
        win._shift_defect(0, 1)
    app.processEvents()
    print("偏移:", win._defect_offset, flush=True)

    # 保存
    win.ed_out.setText(OUT)
    win.btn_save.click()
    app.processEvents()
    pngs = sorted(glob.glob(os.path.join(OUT, "*.png")))
    print("保存产物:", [os.path.basename(p) for p in pngs], flush=True)
    assert pngs, "没有保存文件"
    saved = imread(pngs[0])

    # 候选对比（mid=第二栏「区域与缺陷」，comp=第三栏「缺陷图」，均为调整后）
    orig_img = win._defect_shift_base[0]
    from gui.main_window import _compose_shifted
    exp_shifted = _compose_shifted(win._defect_shift_base, *win._defect_offset)
    mid_pane = latest.get("mid")
    comp_pane = latest.get("comp")
    cands = [
        ("A 缺陷图·第三栏实时内容(调整后)", comp_pane),
        ("B 缺陷图·基准平移(调整后·理论值)", exp_shifted),
        ("C 区域与缺陷·第二栏(调整后)", mid_pane),
        ("D 原图", orig_img),
    ]
    print("\n=== 保存文件与各候选的最大像素差 ===", flush=True)
    for name, cand in cands:
        print(f"  {name}: {maxdiff(saved, cand):.2f}", flush=True)

    diffs = {name: maxdiff(saved, cand) for name, cand in cands}
    best = min(diffs, key=diffs.get)
    print(f"\n保存文件最接近: {best}", flush=True)
    ok = diffs["A 缺陷图·第三栏实时内容(调整后)"] < 1.0
    print("结论:", "保存的正是第三栏「缺陷图」（调整后）" if ok else "保存内容异常！",
          flush=True)
    win.close()
    return 0 if ok else 1


def _shift(a, dx, dy):
    from gui.main_window import _shift_arr
    return _shift_arr(a, dx, dy)


if __name__ == "__main__":
    raise SystemExit(main())
