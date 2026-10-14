"""离屏测量布局几何：三卡片高度 / 是否裁切 / 预览起始位置。

用法::

    QT_QPA_PLATFORM=offscreen python tools/measure_layout.py [宽 高]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

import gui.main_window as mw  # noqa: E402
from gui.theme import apply_theme  # noqa: E402


def children_bottom(w):
    """返回 w 下所有**可见**子控件的最大底边（相对 w 坐标系）。

    只统计可见控件：QComboBox 的下拉弹层等隐藏子控件几何不可信。
    """
    best = 0
    for c in w.findChildren(QWidget):
        if not c.isVisible() or c.window() is not w.window():
            continue
        g = c.geometry()
        x, y = g.x(), g.y()
        p = c.parentWidget()
        while p is not None and p is not w:
            x += p.x()
            y += p.y()
            p = p.parentWidget()
        if p is w:
            best = max(best, y + g.height())
    return best


def report(tag, w, size):
    w.resize(*size)
    w.show()
    app = QApplication.instance()
    for _ in range(3):
        app.processEvents()
    w._equalize_cards()
    for _ in range(3):
        app.processEvents()

    print(f"\n===== {tag}  窗口 {w.width()}x{w.height()} =====")
    for name, c in (("输入", w._card_in), ("缺陷", w._card_def), ("输出", w._card_out)):
        lay_min = c.layout().minimumSize().height()
        print(f"  [{name}] geo=({c.x()},{c.y()},{c.width()}x{c.height()}) "
              f"sizeHint={c.sizeHint().height()} minHint={c.minimumSizeHint().height()} "
              f"layoutMin={lay_min} childrenBottom={children_bottom(c)}")
        spill = children_bottom(c) - c.height()
        if spill > 0:
            print(f"        !! 内容溢出卡片 {spill}px")
    pv = w.preview
    top_bottom = max(c.y() + c.height() for c in (w._card_in, w._card_def, w._card_out))
    print(f"  卡片底边={top_bottom}  预览 y={pv.y()} h={pv.height()} "
          f"minH={pv.minimumSizeHint().height()}")
    print(f"  重叠={top_bottom - pv.y()}px  预览是否可见={pv.height() > 0}")
    cols = {k: p._cols for k, p in w.pages.items()}
    print(f"  参数列数={cols}")
    print(f"  tabs h={w.tabs.height()} 需={w.tabs.tabBar().sizeHint().height()} + "
          f"{max(p.sizeHint().height() for p in w.pages.values())}")


def main():
    global app
    app = QApplication(sys.argv)
    apply_theme(app)
    sizes = [(1100, 600), (1240, 780), (1280, 800), (1366, 768), (1560, 940),
             (1920, 1080)]
    if len(sys.argv) >= 3:
        sizes = [(int(sys.argv[1]), int(sys.argv[2]))]
    for s in sizes:
        w = mw.MainWindow()
        report("布局", w, s)
        w.close()
        w.deleteLater()
        app.processEvents()


if __name__ == "__main__":
    main()
