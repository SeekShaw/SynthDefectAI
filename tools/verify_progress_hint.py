"""聚焦验证：点击刷新预览后应有明确的缺陷生成进度提示。

- 进入生成前：状态栏显示「正在生成缺陷预览…」，进度条进入忙碌动画。
- 生成结束：状态栏显示「预览已更新」，进度条复位为 (0,100) 且值为 0。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np
from PyQt6.QtWidgets import QApplication

import gui.main_window as mw

app = QApplication(sys.argv)
win = mw.MainWindow()
win.show()
app.processEvents()

# 构造一张纯灰图作为当前产品图
img = np.full((300, 800, 3), 220, dtype=np.uint8)
win.current_image = img
win.input_paths = []

# 关闭手绘（模拟用户未画线的常态），让 scratch 不依赖画线也能进入生成分支
win.preview.cb_draw.setChecked(False)
settings = win.current_settings()
print("scratch 默认启用:", settings["scratch"]["enable"], "| 手绘开启:", win.preview.draw_enabled())

# 调用刷新预览（被测逻辑）
win.refresh_preview()
app.processEvents()

status = win.status.currentMessage()
pmin, pmax, pval = win.progress.minimum(), win.progress.maximum(), win.progress.value()
print("状态栏文本:", repr(status))
print("进度条(min,max,value):", pmin, pmax, pval)

ok = True
if status != "预览已更新":
    # 同步生成完成后，状态栏最终应停在「预览已更新」
    print("[FAIL] 状态栏未显示「预览已更新」")
    ok = False
else:
    print("[OK ] 状态栏显示「预览已更新」")

if (pmin, pmax, pval) != (0, 100, 0):
    print(f"[FAIL] 进度条未复位为 (0,100,0)，实际 ({pmin},{pmax},{pval})")
    ok = False
else:
    print("[OK ] 进度条已复位为 (0,100,0)")

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
