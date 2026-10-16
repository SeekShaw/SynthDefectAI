"""探针：验证「限定在产品轮廓内」模式下的交集与「绘制区域不在限定范围内」提示。
通过 monkeypatch 绘制访问器（_draw_mask / _draw_paths / has_*）构造受控的
矩形/线条区域，验证：
  1) 矩形框画在产品外 -> 提示「绘制区域不在限定范围内」且不生成
  2) 矩形框部分在轮廓内 -> 缺陷只落在 轮廓∩矩形 内
  3) 划痕线画在产品外 -> 提示且不生成
  4) 划痕线部分在轮廓内 -> 划痕只落在轮廓内的那段
"""
import os, glob, sys
import numpy as np
print("SCRIPT START", flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtTest import QTest
from core.utils import imread, clip01
from gui.main_window import MainWindow, POS_MODE_PRODUCT, POS_MODE_PIXEL, _shift_arr
print("IMPORTS OK", flush=True)

_captured = []
def _mb_info(parent, title, text, *a, **k):
    _captured.append((title, text)); return QMessageBox.StandardButton.Ok
QMessageBox.information = staticmethod(_mb_info)
QMessageBox.warning = staticmethod(_mb_info)

def gen(win, app, reset=True):
    win._preview_timer.stop()
    win.refresh_preview(reset_offset=reset)
    app.processEvents()

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    app.processEvents()
    print("[step] built", flush=True)
    samples = sorted(glob.glob(os.path.join(ROOT, "samples_mlcc", "*.png")))
    samples = [s for s in samples if not os.path.basename(s).startswith("region")]
    win.add_inputs([samples[0]]); app.processEvents()
    print("[step] inputs", flush=True)
    win.lst.setCurrentRow(0); app.processEvents()
    win.preview.cmb_posmode.setCurrentIndex(win.preview.cmb_posmode.findData(POS_MODE_PRODUCT))
    app.processEvents()
    print("[step] pos=product", flush=True)
    win._mutual_lock = True
    win.pages["smudge"].enable.setChecked(True)
    win.pages["scratch"].enable.setChecked(True)
    win._mutual_lock = False
    app.processEvents()
    print("[step] enabled", flush=True)
    win.preview.draw_enabled = lambda: True
    win.preview.has_drawn_regions = lambda: True
    win.preview.has_line_regions = lambda: True
    win.ed_out.setText(os.path.join(ROOT, "build", "out_probe"))

    h, w = win.current_image.shape[:2]
    seg = win._ensure_segmentation()
    prod = np.maximum(seg.body, seg.electrode) > 0.5
    prod_ys, prod_xs = np.where(prod)
    cy, cx = int(prod_ys.mean()), int(prod_xs.mean())
    print(f"[info] 产品中心≈({cx},{cy}) 图像({w}x{h})")

    results = []
    def rec(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})", flush=True)
        results.append(ok)

    # ---- 1) 矩形框画在产品外（角落）----
    rect_out = np.zeros((h, w), np.float32); rect_out[5:55, 5:55] = 1.0
    win._draw_mask = lambda kind=None: rect_out.copy()
    win._draw_paths = lambda: []
    print("[chk] 1 矩形外", flush=True)
    nb, msg = win._region_out_of_scope(seg, win._region_masks_for(seg))
    rec("矩形框在产品外->提示", nb and "绘制区域不在限定范围内" in msg, f"nb={nb}")
    gen(win, app)
    rec("矩形外->不生成缺陷", win._defect_shift_base is None, "shift_base=None")

    # ---- 2) 矩形框部分在轮廓内 ----
    rect_part = np.zeros((h, w), np.float32)
    rect_part[0:h//2, 0:w] = 1.0
    win._draw_mask = lambda kind=None: rect_part.copy()
    win._draw_paths = lambda: []
    print("[chk] 2 矩形部分", flush=True)
    nb2, _ = win._region_out_of_scope(seg, win._region_masks_for(seg))
    rec("矩形部分内->不提示", not nb2, f"nb={nb2}")
    gen(win, app)
    o, d, m, mk = win._defect_shift_base
    mask = mk > 0.35
    inside = float((mask & prod).sum()) / max(1, mask.sum())
    rec("矩形部分内->缺陷在轮廓内", inside >= 0.97, f"inside={inside:.2%}")

    # ---- 3) 划痕线画在产品外 ----
    win._draw_mask = lambda kind=None: None
    win._draw_paths = lambda: [[(10,10),(20,20),(30,30),(40,40)]]
    print("[chk] 3 划痕外", flush=True)
    nb3, msg3 = win._region_out_of_scope(seg, win._region_masks_for(seg))
    rec("划痕线在产品外->提示", nb3 and "绘制区域不在限定范围内" in msg3, f"nb={nb3}")
    _captured.clear()
    win.save_defect_image(); app.processEvents()
    saved_prompt = any("绘制区域不在限定范围内" in t for _, t in _captured)
    rec("划痕外->保存弹窗提示", saved_prompt, f"captured={len(_captured)}")

    # ---- 4) 划痕线部分在轮廓内 ----
    win._draw_paths = lambda: [[(10,10),(cx,cy)]]
    print("[chk] 4 划痕部分", flush=True)
    nb4, _ = win._region_out_of_scope(seg, win._region_masks_for(seg))
    rec("划痕部分内->不提示", not nb4, f"nb={nb4}")
    gen(win, app)
    o, d, m, mk = win._defect_shift_base
    mask = mk > 0.35
    inside = float((mask & prod).sum()) / max(1, mask.sum())
    rec("划痕部分内->缺陷在轮廓内", inside >= 0.97, f"inside={inside:.2%}")

    # ---- 5) 像素坐标模式：不裁剪、不提示 ----
    win.preview.cmb_posmode.setCurrentIndex(win.preview.cmb_posmode.findData(POS_MODE_PIXEL))
    app.processEvents()
    win._draw_mask = lambda kind=None: rect_out.copy()
    win._draw_paths = lambda: [[(10,10),(20,20)]]
    print("[chk] 5 像素模式", flush=True)
    nb5, _ = win._region_out_of_scope(seg, win._region_masks_for(seg))
    rec("像素模式->不提示(线外)", not nb5, f"nb={nb5}")

    ok_all = all(results)
    print("\nRESULT:", "全部通过" if ok_all else "存在失败", flush=True)
    win.close()
    return 0 if ok_all else 1

if __name__ == "__main__":
    sys.exit(main())
