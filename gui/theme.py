"""深色工业风主题（与 WorkBuddy IDE 深色主题一致）。"""
from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

# 取色板
BG = "#14161a"
PANEL = "#1b1f25"
PANEL_ALT = "#22272f"
BORDER = "#2e353f"
TEXT = "#e6e9ee"
TEXT_DIM = "#98a2b3"
ACCENT = "#4c9aff"
ACCENT_DIM = "#2f6fd0"
GOOD = "#33d17a"
WARN = "#f2c94c"
BAD = "#eb5757"

QSS = f"""
* {{
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 12px;
}}
QWidget {{
    background: {BG};
    color: {TEXT};
}}
QMainWindow, QDialog {{ background: {BG}; }}

QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 18px;
    padding: 9px 10px 8px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 2px;
    padding: 1px 7px;
    color: {ACCENT};
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 5px;
}}

QLabel {{ background: transparent; }}
QLabel[role="hint"] {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel[role="title"] {{ font-size: 15px; font-weight: 700; color: {TEXT}; }}
QLabel[role="defect"] {{ font-size: 16px; font-weight: 700; color: {ACCENT}; }}

QFrame[role="card"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

QPushButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 10px;
    color: {TEXT};
}}
QPushButton:hover {{ background: #2b323c; border-color: {ACCENT_DIM}; }}
QPushButton:pressed {{ background: #262c35; }}
QPushButton:disabled {{ color: #5c6675; background: #1a1e24; border-color: #262c34; }}

QPushButton[accent="true"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #0b1220;
    font-weight: 700;
}}
QPushButton[accent="true"]:hover {{ background: #66acff; }}
QPushButton[accent="true"]:disabled {{ background: #2c3a4d; border-color: #2c3a4d; color: #6b7b90; }}

QPushButton[danger="true"] {{ border-color: #6b2f2f; color: #ff9d9d; }}
QPushButton[danger="true"]:hover {{ background: #3a2222; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 3px 6px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
    outline: none;
}}

/* ---- 数字框上下箭头：深色主题下必须显式定义，否则箭头不可见 ---- */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: #2b323c;
    border: none;
    width: 17px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {ACCENT_DIM};
}}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
    background: {ACCENT};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {TEXT_DIM};
}}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover,
QSpinBox::up-arrow:pressed, QDoubleSpinBox::up-arrow:pressed {{
    border-bottom-color: #ffffff;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
}}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover,
QSpinBox::down-arrow:pressed, QDoubleSpinBox::down-arrow:pressed {{
    border-top-color: #ffffff;
}}

QCheckBox, QRadioButton {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid #4a5464;
    border-radius: 4px;
    background: {PANEL_ALT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QRadioButton::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{
    height: 4px; background: #2b323c; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #dfe6ef; width: 11px; margin: -5px 0; border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}

QProgressBar {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 16px;
    text-align: center;
    color: {TEXT};
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {PANEL}; top: -1px; }}
QTabBar::tab {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 4px 12px;
    margin-right: 3px;
    color: {TEXT_DIM};
}}
QTabBar::tab:selected {{ background: {PANEL}; color: {ACCENT}; font-weight: 600; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #39424f; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #4a5666; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #39424f; border-radius: 5px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QListWidget, QTreeWidget, QTableWidget {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    outline: none;
    alternate-background-color: #252b34;
}}
QListWidget::item, QTreeWidget::item {{ padding: 2px 6px; border-radius: 4px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {ACCENT_DIM}; }}
QHeaderView::section {{
    background: #262c35; border: none; border-right: 1px solid {BORDER};
    padding: 6px; color: {TEXT_DIM};
}}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QToolTip {{
    background: #262c35; color: {TEXT}; border: 1px solid {BORDER};
    padding: 5px 7px; border-radius: 5px;
}}
QStatusBar {{ background: {PANEL}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyleSheet(QSS)
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(PANEL_ALT))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(PANEL_ALT))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_DIM))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    app.setPalette(pal)
