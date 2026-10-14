"""离屏渲染主窗口并截图，用于人工核查布局。"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from PyQt6.QtWidgets import QApplication, QTabWidget  # noqa: E402
import numpy as np  # noqa: E402
from PyQt6.QtCore import QPoint, QPointF, Qt, QTimer  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

import gui.main_window as mw  # noqa: E402

OUT = os.path.join(HERE, "build", "gui_shots")
os.makedirs(OUT, exist_ok=True)


def main():
    app = QApplication(sys.argv)
    win = mw.MainWindow()
    win.resize(1760, 1020)
    win.show()

    smp = os.path.join(HERE, "samples_mlcc")
    samples = [os.path.join(smp, f) for f in sorted(os.listdir(smp))
               if f.startswith("mlcc_") and f.endswith(".png")]
    win.add_inputs(samples)
    win.lst.setCurrentRow(0)
    win.ed_out.setText(os.path.join(HERE, "build", "out_shot"))
    win.sp_count.setValue(2)
    app.processEvents()

    win.refresh_preview()
    app.processEvents()

    def shot(tag):
        app.processEvents()
        p = os.path.join(OUT, f"main_{tag}.png")
        win.grab().save(p)
        print("[shot]", p)

    def _wpt(widget, img_pt):
        p = widget.image_to_widget(QPointF(*img_pt))
        return QPoint(int(np.clip(p.x(), 1, widget.width() - 2)),
                      int(np.clip(p.y(), 1, widget.height() - 2)))

    def _drag(widget, p_from, p_to, steps=14):
        """真实鼠标拖动（截图脚本也走用户实际的操作路径）。"""
        QTest.mousePress(widget, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, _wpt(widget, p_from))
        for i in range(1, steps + 1):
            t = i / float(steps)
            QTest.mouseMove(widget, _wpt(widget,
                             (p_from[0] + (p_to[0] - p_from[0]) * t,
                              p_from[1] + (p_to[1] - p_from[1]) * t)))
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, _wpt(widget, p_to))

    shot("00_overview")

    # 小窗口 / 最小尺寸回归：三卡片不再压到下方预览
    for tag, size in (("10_small_1366x768", (1366, 768)),
                      ("11_min_1180x720", (1180, 720))):
        win.resize(*size)
        app.processEvents()
        win._equalize_cards()
        app.processEvents()
        shot(tag)
    win.resize(1760, 1020)
    app.processEvents()
    win._equalize_cards()
    app.processEvents()

    tabs = win.findChildren(QTabWidget)[0]
    tabs.setCurrentIndex(1)
    shot("01_tab_smudge")
    tabs.setCurrentIndex(2)
    shot("02_tab_discolor")

    # 缺角 Tab
    tabs.setCurrentIndex(3)
    win.pages["scratch"].enable.setChecked(False)
    win.pages["smudge"].enable.setChecked(False)
    win.pages["discolor"].enable.setChecked(False)
    win.pages["chip"].enable.setChecked(True)
    win.pages["chip"].rows["size_ratio"].setValue(0.55)
    win.pages["chip"].rows["count"].setValue(2)
    win.refresh_preview()
    app.processEvents()
    shot("03_tab_chip")
    print("      info:", win.preview.lbl_info.text())

    tabs.setCurrentIndex(0)
    win.pages["scratch"].enable.setChecked(True)
    win.pages["smudge"].enable.setChecked(True)
    win.pages["discolor"].enable.setChecked(True)
    win.pages["chip"].enable.setChecked(False)
    app.processEvents()

    # 手绘区域 + 矩形模式
    win.preview.cb_draw.setChecked(True)
    app.processEvents()
    win.preview.views[0]._rects.append((390, 235, 420, 160, math.radians(12)))
    win.preview.views[0].update()
    win.refresh_preview()
    app.processEvents()
    shot("04_draw_rect")
    win.preview.clear_regions()
    win.preview.cb_draw.setChecked(False)
    app.processEvents()

    # 缩放
    win.preview.zoom_1to1()
    app.processEvents()
    shot("05_zoom_1to1")
    win.preview.fit_all()
    app.processEvents()

    # 明显度对比
    for k in ("scratch", "smudge", "discolor"):
        cb = win.pages[k].combos["severity"]
        cb.setCurrentIndex(cb.findData("obvious"))
    win.refresh_preview()
    app.processEvents()
    shot("08_sev_obvious")
    print("      info:", win.preview.lbl_info.text())

    for k in ("scratch", "smudge", "discolor"):
        cb = win.pages[k].combos["severity"]
        cb.setCurrentIndex(cb.findData("subtle"))
    win.refresh_preview()
    app.processEvents()
    shot("09_sev_subtle")
    print("      info:", win.preview.lbl_info.text())

    # 手绘：线条（划痕）/ 多边形（脏污异色）/ 旋转 ROI / 画笔
    win.preview.cb_draw.setChecked(True)
    win.preview.cmb_mode.setCurrentIndex(1)      # 画线条
    v0 = win.preview.views[0]
    h_img, w_img = win.current_image.shape[:2]
    _drag(v0, (w_img * 0.22, h_img * 0.42), (w_img * 0.80, h_img * 0.46), steps=16)
    win._force_resegment()
    win.pages["smudge"].enable.setChecked(False)
    win.pages["discolor"].enable.setChecked(False)
    win.pages["chip"].enable.setChecked(False)
    win.pages["scratch"].enable.setChecked(True)
    win.pages["scratch"].rows["count"].setValue(3)
    win.pages["scratch"].rows["tone"].setValue(1.0)
    win.refresh_preview()
    for _ in range(3):
        app.processEvents()
    shot("06_line_scratch")
    print("      info:", win.preview.lbl_info.text())

    # 多边形：脏污 / 异色 / 缺角
    win.preview.cmb_mode.setCurrentIndex(3)      # 画多边形
    v0._lines.clear()
    v0._polys.append([(w_img * 0.26, h_img * 0.30), (w_img * 0.52, h_img * 0.26),
                      (w_img * 0.60, h_img * 0.52), (w_img * 0.44, h_img * 0.74),
                      (w_img * 0.22, h_img * 0.62)])
    win.pages["scratch"].enable.setChecked(False)
    win.pages["smudge"].enable.setChecked(True)
    win.pages["discolor"].enable.setChecked(True)
    win.pages["chip"].enable.setChecked(True)
    win._force_resegment()
    win.refresh_preview()
    for _ in range(3):
        app.processEvents()
    shot("07_poly_area")
    print("      info:", win.preview.lbl_info.text())

    # 旋转 ROI + 画笔
    win.preview.cmb_mode.setCurrentIndex(2)      # 画矩形
    v0._polys.clear()
    v0._rects.append((w_img * 0.5, h_img * 0.5, w_img * 0.60, h_img * 0.34,
                      math.radians(28)))
    win.preview.sp_angle.setValue(28)
    win._force_resegment()
    win.refresh_preview()
    for _ in range(3):
        app.processEvents()
    shot("10_roi28")
    print("      info:", win.preview.lbl_info.text())

    win.preview.cmb_mode.setCurrentIndex(4)      # 画笔（笔刷控件此时才出现）
    v0._rects.clear()
    v0._strokes.append([(w_img * 0.20, h_img * 0.40), (w_img * 0.45, h_img * 0.55),
                        (w_img * 0.78, h_img * 0.45)])
    win._force_resegment()
    win.refresh_preview()
    for _ in range(3):
        app.processEvents()
    shot("11_brush")
    print("      info:", win.preview.lbl_info.text())

    # 翻页：切到下一张（输入卡片的「下一张 ▶」）
    win.preview.cb_draw.setChecked(False)
    v0.clear_regions()
    win.btn_next.click()
    for _ in range(3):
        app.processEvents()
    shot("12_next_image")
    print("      第", win.lst.currentRow() + 1, "张:", win.lbl_count.text())
    win.btn_prev.click()
    for _ in range(3):
        app.processEvents()
    win._force_resegment()

    win.close()
    print("DONE")
    app.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    rc = {"code": 0}

    def run():
        try:
            main()
        except Exception:
            import traceback
            traceback.print_exc()
            rc["code"] = 1
        finally:
            app.quit()

    QTimer.singleShot(0, run)
    app.exec()
    sys.exit(rc["code"])