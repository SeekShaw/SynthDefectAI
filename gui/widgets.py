"""预留：可折叠参数分组控件（当前主界面未强制依赖）。"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QGroupBox, QToolButton, QVBoxLayout, QWidget,
                             QHBoxLayout, QLabel)


class CollapsibleGroup(QWidget):
    """带折叠按钮的参数分组。"""

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, parent=None, expanded: bool = True) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.btn = QToolButton()
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(expanded)
        self.btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.btn.setStyleSheet(
            "QToolButton{border:none;font-weight:600;color:#4c9aff;padding:4px 2px;}")
        self.btn.toggled.connect(self._on_toggle)
        root.addWidget(self.btn)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(10, 0, 0, 0)
        self.body.setVisible(expanded)
        root.addWidget(self.body)

    def _on_toggle(self, on: bool):
        self.body.setVisible(on)
        self.btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.toggled.emit(on)
