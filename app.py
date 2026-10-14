"""SyntheticDefectAI — 工业表面缺陷合成工具（图形界面）。

用法:
    python app.py                # 启动 GUI
    python app.py --cli ...      # 命令行批量生成（见 core/cli.py）
"""
from __future__ import annotations

import os
import sys

# 保证以脚本方式运行或打包后都能正确定位 core 包
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QIcon, QFont  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.theme import apply_theme  # noqa: E402


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("SyntheticDefectAI")
    app.setApplicationDisplayName("SynthDefect AI — 表面缺陷合成工具")
    app.setStyle("Fusion")
    apply_theme(app)

    base = getattr(sys, "_MEIPASS", _HERE)
    ico = os.path.join(base, "assets", "app.ico")
    if os.path.exists(ico):
        app.setWindowIcon(QIcon(ico))

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        from core.cli import cli_main

        sys.exit(cli_main(sys.argv[2:]))
    sys.exit(main())
