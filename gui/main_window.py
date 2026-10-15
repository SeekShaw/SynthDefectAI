"""主窗口：输入 / Tab 分页缺陷参数 / 可缩放预览 / 可旋转 ROI 手绘 / 批量生成。"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QKeySequence
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox,
                             QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QInputDialog, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                             QMessageBox, QProgressBar, QApplication, QPushButton, QScrollArea,
                             QSlider, QSpinBox, QSplitter, QStatusBar,
                             QTabWidget, QVBoxLayout, QWidget)

from core.batch import (BatchConfig, DEFAULT_SETTINGS, KIND_CN, SEVERITY_CN,
                        TARGET_CN, ItemResult, run_batch)
from core.defects import (DEFECT_KINDS, apply_augmentation, apply_defects,
                        _KIND_TO_COLOR)
from core.mlcc_region import (product_outline_overlay, region_overlay,
                             segment_mlcc)
from core.utils import clip01, imread, imwrite, list_images, to_u8

from .image_view import ImageView, np_to_qpixmap
from .preview_pane import POS_MODE_PIXEL, POS_MODE_PRODUCT, PreviewPane

_KIND_COLOR = dict(_KIND_TO_COLOR)   # 各缺陷类型在叠加图里的颜色

# 异色「色调」下拉选项（随机 / 预设 / 自定义）
DISCOLOR_COLOR_OPTS = [
    ("随机", "random"), ("泛黄", "yellow"), ("泛蓝", "blue"), ("泛绿", "green"),
    ("泛红", "red"), ("泛紫", "purple"), ("褪色", "fade"), ("自定义", "custom"),
]


# --------------------------------------------------------------------------
# 后台任务
# --------------------------------------------------------------------------
class BatchWorker(QObject):
    progress = pyqtSignal(int, int, str)
    item_done = pyqtSignal(object)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, cfg: BatchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            info = run_batch(
                self.cfg,
                on_progress=lambda d, t, n: self.progress.emit(d, t, n),
                on_item=lambda r: self.item_done.emit(r),
                should_stop=lambda: self._stop,
            )
            self.finished.emit(info)
        except Exception as e:
            import traceback

            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------
# 参数行
# --------------------------------------------------------------------------
class SliderRow(QWidget):
    """标签 + 滑条 + 数值 的一行参数控件。"""

    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, vmin: float, vmax: float, value: float,
                 step: float = 0.01, decimals: int = 2, suffix: str = "",
                 integer: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._min, self._max = vmin, vmax
        self._int = integer
        self._dec = 0 if integer else decimals

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.lbl = QLabel(label)
        self.lbl.setFixedWidth(58)
        lay.addWidget(self.lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(self._to_slider(value))
        lay.addWidget(self.slider, 1)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(self._dec)
        self.spin.setValue(value)
        self.spin.setFixedWidth(68)
        if suffix:
            self.spin.setSuffix(suffix)
        lay.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    def _to_slider(self, v: float) -> int:
        span = self._max - self._min
        return int(round((v - self._min) / span * 1000)) if span else 0

    def _from_slider(self, s: int) -> float:
        return self._min + (self._max - self._min) * s / 1000.0

    def _on_slider(self, s: int):
        v = self._from_slider(s)
        if self._int:
            v = float(round(v))
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(v)

    def _on_spin(self, v: float):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(v))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, v: float):
        self.spin.setValue(v)


class DefectPage(QWidget):
    """单个缺陷类型的参数页（放入 Tab）。

    参数行按可用宽度自适应排成 1 / 2 列（由主窗口调用 ``set_columns`` 指定）：
    宽屏两列排布能把卡片高度压到最矮，窄屏回落单列避免文字被裁。
    """

    changed = pyqtSignal()

    TWO_COL_MIN = 520       # 页宽 ≥ 该值时参数行排两列
    LABEL_W = 58            # 参数名标签宽度（与 SliderRow 内标签对齐）

    def __init__(self, kind: str, cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.rows: dict[str, SliderRow] = {}
        self.combos: dict[str, QComboBox] = {}
        self._items: list[QWidget] = []
        self._cols = 1
        self.color_rgb = (150, 130, 80)   # 异色自定义目标色（供 config() 统一输出）

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(5)

        self.enable = QCheckBox(f"启用「{KIND_CN.get(kind, kind)}」缺陷")
        self.enable.setChecked(bool(cfg.get("enable", False)))
        self.enable.toggled.connect(self._emit)
        outer.addWidget(self.enable)

        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(4)
        outer.addWidget(self.body, 1)

        # 明显度（目标表面已按需求移除，缺陷统一落在产品可见表面上）
        cb_s = QComboBox()
        for k, txt in SEVERITY_CN.items():
            cb_s.addItem(txt, k)
        cb_s.setCurrentIndex(max(0, cb_s.findData(cfg.get("severity", "mixed"))))
        cb_s.currentIndexChanged.connect(self._emit)
        self.combos["severity"] = cb_s
        self.w_severity = self._labeled("明显度", cb_s)

        self.hline = self._hline()

        if kind == "scratch":
            self._add("strength", "强度", 0.0, 1.0, cfg.get("strength", 0.55))
            self._add("count", "条数", 1, 40, cfg.get("count", 1), integer=True)
            self._add("length_ratio", "长度比", 0.05, 1.5, cfg.get("length_ratio", 0.55))
            self._add("width_px", "线宽px", 0.5, 12.0, cfg.get("width_px", 2.2))
            self._add("curve", "弯曲度", 0.0, 1.0, cfg.get("curve", 0.55))
            self._add("tone", "颜色", -1.0, 1.0, cfg.get("tone", 0.0))
            self._add("tone_depth", "深浅", 0.1, 1.5, cfg.get("tone_depth", 0.85))
        elif kind == "smudge":
            self._add("strength", "强度", 0.0, 1.0, cfg.get("strength", 0.50))
            self._add("blob_count", "斑块数", 1, 20, cfg.get("blob_count", 1), integer=True)
            self._add("blob_scale", "斑块尺寸", 0.05, 0.8, cfg.get("blob_scale", 0.30))
            self._add("coverage", "覆盖率", 0.02, 0.9, cfg.get("coverage", 0.35))
            self._add("edge_soft", "边缘模糊", 0.0, 2.5, cfg.get("edge_soft", 1.0))
            cb = QComboBox()
            for k, txt in [("auto", "自动（按底色）"), ("dark", "深色污渍"),
                           ("dust", "灰尘灰"), ("oil", "油污褐"), ("rust", "锈迹橙")]:
                cb.addItem(txt, k)
            cb.setCurrentIndex(max(0, cb.findData(cfg.get("dirt_color", "auto"))))
            cb.currentIndexChanged.connect(self._emit)
            self.combos["dirt_color"] = cb
            self._items.append(self._labeled("污染色", cb))
        elif kind == "discolor":
            self._add("strength", "强度", 0.0, 1.0, cfg.get("strength", 0.50))
            self._add("region_count", "区域数", 1, 12, cfg.get("region_count", 1), integer=True)
            self._add("region_scale", "区域尺寸", 0.05, 0.9, cfg.get("region_scale", 0.35))
            self._add("sat_gain", "色度增益", 0.5, 2.0, cfg.get("sat_gain", 1.25))
            self._add("lum_shift", "明暗偏移", -40, 40, cfg.get("lum_shift", -8), step=1)
            self._add("edge_blur", "边缘模糊", 0.0, 2.5, cfg.get("edge_blur", 1.0))
            # 颜色设置：指定色调 + 自定义取色
            cb = QComboBox()
            for txt, key in DISCOLOR_COLOR_OPTS:
                cb.addItem(txt, key)
            cb.setCurrentIndex(max(0, cb.findData(cfg.get("color", "random"))))
            cb.currentIndexChanged.connect(self._emit)
            self.combos["color"] = cb
            self._items.append(self._labeled("色调", cb))
            # 自定义取色按钮（仅「自定义」生效）
            self.color_rgb = tuple(int(x) for x in cfg.get("color_rgb", [150, 130, 80]))
            self.btn_color = QPushButton("取色…")
            self.btn_color.setFixedWidth(72)
            self._apply_color_swatch()
            self.btn_color.clicked.connect(self._pick_color)
            self._items.append(self._labeled("自定义色", self.btn_color))
        elif kind == "chip":
            self._add("strength", "崩口强度", 0.0, 1.0, cfg.get("strength", 0.60))
            self._add("count", "缺角数", 1, 8, cfg.get("count", 1), integer=True)
            self._add("size_ratio", "缺角尺寸", 0.15, 0.90, cfg.get("size_ratio", 0.40))
            self._add("depth_ratio", "崩口深度", 0.30, 1.50, cfg.get("depth_ratio", 0.85))
            self._add("jag_ratio", "断面锯齿", 0.05, 0.90, cfg.get("jag_ratio", 0.42))
            self._add("crack_ratio", "延伸裂纹", 0.0, 0.90, cfg.get("crack_ratio", 0.35))

        self._relayout()
        self.enable.toggled.connect(self._sync_enable)
        self._sync_enable(self.enable.isChecked())

    # ---------------- 自适应排版 ----------------
    def set_columns(self, cols: int):
        """由主窗口按卡片可用宽度指定参数列数（1 或 2）。"""
        cols = 2 if int(cols) >= 2 else 1
        if cols == self._cols:
            return
        self._cols = cols
        self._relayout()

    def _relayout(self):
        g = self.grid
        while g.count():
            g.takeAt(0)
        cols = self._cols or 1
        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)
        # 明显度单独一行（目标表面已移除）
        g.addWidget(self.w_severity, 0, 0, 1, 2)
        g.addWidget(self.hline, 1, 0, 1, 2)
        r = 2
        if cols == 2:
            for i, w in enumerate(self._items):
                g.addWidget(w, r + i // 2, i % 2)
        else:
            # 单列：参数行横跨两列（每列等宽，保证滑条不塌缩）
            for i, w in enumerate(self._items):
                g.addWidget(w, r + i, 0, 1, 2)
        self.body.updateGeometry()

    def _labeled(self, label: str, w: QWidget) -> QWidget:
        """标签 + 控件的组合行（标签宽度与滑条行一致，便于对齐）。"""
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        lb = QLabel(label)
        lb.setFixedWidth(self.LABEL_W)
        h.addWidget(lb)
        h.addWidget(w, 1)
        return box

    @staticmethod
    def _hline() -> QFrame:
        ln = QFrame()
        ln.setFrameShape(QFrame.Shape.HLine)
        ln.setStyleSheet("color:#2e353f; background:#2e353f; max-height:1px;")
        return ln

    def _add(self, key, label, vmin, vmax, value, step=0.01, integer=False):
        r = SliderRow(label, vmin, vmax, value, step=step, integer=integer)
        r.valueChanged.connect(self._emit)
        self.rows[key] = r
        self._items.append(r)

    def _sync_enable(self, on: bool):
        self.body.setEnabled(on)

    def _emit(self, *_):
        self.changed.emit()

    def config(self) -> dict:
        cfg = {"enable": self.enable.isChecked()}
        for k, r in self.rows.items():
            v = r.value()
            cfg[k] = int(round(v)) if r._int else round(v, 4)
        for k, cb in self.combos.items():
            cfg[k] = cb.currentData()
        cfg["color_rgb"] = list(self.color_rgb)
        return cfg

    def _apply_color_swatch(self):
        """把取色按钮背景刷成当前自定义色（无按钮的页面安全跳过）。"""
        btn = getattr(self, "btn_color", None)
        if btn is None:
            return
        r, g, b = self.color_rgb
        btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color:#0b1220; "
            f"border:1px solid #49525e; border-radius:4px;")

    def _pick_color(self):
        """打开取色对话框，选定后切到「自定义」并广播变更。"""
        init = QColor(*self.color_rgb)
        c = QColorDialog.getColor(init, self, "选择异色目标色")
        if not c.isValid():
            return
        self.color_rgb = (c.red(), c.green(), c.blue())
        self._apply_color_swatch()
        idx = self.combos["color"].findData("custom")
        if idx >= 0:
            self.combos["color"].setCurrentIndex(idx)
        self._emit()


# --------------------------------------------------------------------------
# 主窗口
# --------------------------------------------------------------------------
class ThumbnailGallery(QWidget):
    """增强样本缩略图画廊：生成的多张缺陷图以缩略图网格展示，点击即选中。"""

    selected = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        cap = QLabel("增强样本缩略图")
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setProperty("role", "title")
        lay.addWidget(cap)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.grid = QWidget()
        self.gl = QGridLayout(self.grid)
        self.gl.setSpacing(8)
        self.scroll.setWidget(self.grid)
        lay.addWidget(self.scroll, 1)
        self.images: list = []
        self.thumbs: list = []
        self.current = -1
        self._placeholder()

    def _placeholder(self):
        self.clear()
        tip = QLabel("勾选「缺陷样本增强模式」并点「刷新预览」后，\n"
                     "这里显示生成的缩略图；点击缩略图可在右侧查看大图。")
        tip.setWordWrap(True)
        tip.setProperty("role", "hint")
        tip.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.gl.addWidget(tip, 0, 0)

    def clear(self):
        while self.gl.count():
            item = self.gl.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.setParent(None)
        self.images = []
        self.thumbs = []
        self.current = -1

    def set_thumbnails(self, images: list, captions=None):
        self.clear()
        self.images = list(images)
        cols = 2
        for i, img in enumerate(self.images):
            cell = self._make_cell(
                img, i, (captions[i] if captions else f"#{i + 1}"))
            self.gl.addWidget(cell, i // cols, i % cols)
            self.thumbs.append(cell)
        if self.images:
            self._select(0)

    def _make_cell(self, img, idx, caption):
        cell = QFrame()
        cell.setFrameShape(QFrame.Shape.Box)
        cell.setLineWidth(1)
        cell.setStyleSheet(
            "ThumbnailGallery QFrame{border:1px solid #2e353f; "
            "border-radius:6px; background:#0e1013;}")
        v = QVBoxLayout(cell)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)
        pm = np_to_qpixmap(img).scaled(
            110, 110, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        lab = QLabel()
        lab.setPixmap(pm)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab.setStyleSheet("background:#0e1013;")
        v.addWidget(lab, 1)
        cap = QLabel(caption)
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setProperty("role", "hint")
        v.addWidget(cap)
        cell.setCursor(Qt.CursorShape.PointingHandCursor)
        cell.mousePressEvent = lambda _e, i=idx: self._select(i)
        return cell

    def _select(self, idx):
        if idx < 0 or idx >= len(self.images):
            return
        self.current = idx
        for i, t in enumerate(self.thumbs):
            t.setStyleSheet(
                "ThumbnailGallery QFrame{border:%s; border-radius:6px; background:#0e1013;}"
                % ("2px solid #4c9aff" if i == idx else "1px solid #2e353f"))
        self.selected.emit(idx)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SynthDefect AI — MLCC 表面缺陷合成工具")
        self.resize(1560, 940)
        # 最小尺寸按「三卡片 + 预览最小高度 + 底部行」实测取值，低于此值会互相挤压
        self.setMinimumSize(1180, 740)

        self.input_paths: list[str] = []
        self.current_image: Optional[np.ndarray] = None
        self.current_path: str = ""
        self.worker: Optional[BatchWorker] = None
        self.thread: Optional[QThread] = None
        self._seg_key: Optional[tuple] = None
        self._seg_res = None
        self.aug_results: list = []     # [(img, applied), ...] 增强模式生成的样本
        self.aug_selected: int = -1     # 当前在画廊中选中的样本下标
        self._adjusting = False        # 卡片等高重算防重入
        self._top_h = 0                # 顶部三卡片的统一高度

        # 状态栏先于界面构建创建：_defect_card() 会同步 Tab 标题并写状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self.refresh_preview)

        self._build_ui()
        self._build_menu()
        self.status.showMessage("就绪：请选择 MLCC 产品图片")

    # ---------------- 布局 ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 6)
        root.setSpacing(8)

        # 上半部分：输入设置 | 缺陷图生成配置 | 输出设置（三卡等高、文字不裁切）
        top = QHBoxLayout()
        top.setSpacing(10)
        self._card_in = self._input_card()
        self._card_def = self._defect_card()
        self._card_out = self._output_card()
        top.addWidget(self._card_in, 4)
        top.addWidget(self._card_def, 6)
        top.addWidget(self._card_out, 4)
        root.addLayout(top)

        # 下半部分：左=源图/手绘（框选部分移到最左），中=增强缩略图画廊，右=缺陷图显示
        self.split_main = QSplitter(Qt.Orientation.Horizontal)
        self.split_main.setContentsMargins(0, 0, 0, 0)
        self.split_main.setHandleWidth(6)

        # 左侧：源图 + 手绘区域（"框选部分"移到最左）
        self.preview = PreviewPane()
        self.preview.regionChanged.connect(self._on_region_drawn)
        self.preview.modeChanged.connect(self._on_draw_mode_changed)
        self.preview.drawToggled.connect(self._on_draw_toggled)
        self.preview.rectAngleChanged.connect(self._on_rect_angle)
        self.preview.posModeChanged.connect(self._on_pos_mode_changed)
        self.preview.overlayChanged.connect(self._refresh_overlay)
        self.split_main.addWidget(self.preview)

        # 中间：增强样本缩略图画廊（框选区域空出后放这里）
        self.gallery = ThumbnailGallery()
        self.gallery.selected.connect(self._on_thumb_selected)
        self.split_main.addWidget(self.gallery)

        # 右侧：缺陷图显示（点击左侧缩略图即在此查看大图）
        self.defect_view = ImageView()
        self.defect_view.setStyleSheet(
            "background:#0e1013; border:1px solid #2e353f; border-radius:8px;")
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(4)
        rcap = QLabel("缺陷图（点击左侧缩略图查看）")
        rcap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rcap.setProperty("role", "hint")
        rv.addWidget(self.defect_view, 1)
        rv.addWidget(rcap)
        self.split_main.addWidget(right)

        self.split_main.setStretchFactor(0, 5)
        self.split_main.setStretchFactor(1, 3)
        self.split_main.setStretchFactor(2, 5)
        root.addWidget(self.split_main, 1)

        # 三张卡片的等高在 showEvent 中按真实布局高度统一（保证文字不裁切）
        # 底部：进度 + 操作（保持不变）
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        bl.addWidget(self.progress, 1)
        self.btn_preview = QPushButton("刷新预览")
        self.btn_preview.clicked.connect(self.refresh_preview)
        bl.addWidget(self.btn_preview)
        self.btn_save = QPushButton("保存缺陷图")
        self.btn_save.setProperty("accent", True)
        self.btn_save.setMinimumHeight(36)
        self.btn_save.setMinimumWidth(140)
        self.btn_save.setToolTip("把当前这张预览的缺陷图保存到输出目录")
        self.btn_save.clicked.connect(self.save_defect_image)
        bl.addWidget(self.btn_save)
        self.btn_run = QPushButton("批量生成全部")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.setMinimumWidth(140)
        self.btn_run.setToolTip("对已导入的全部图片批量生成")
        self.btn_run.clicked.connect(self.start_batch)
        bl.addWidget(self.btn_run)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setProperty("danger", True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_batch)
        bl.addWidget(self.btn_stop)
        root.addWidget(bottom)
        self.status.showMessage("就绪")

    def showEvent(self, e):
        """窗口显示后按真实布局高度统一三张卡片高度（保证文字不裁切）。"""
        super().showEvent(e)
        self._equalize_cards()

    def resizeEvent(self, e):
        """窗口尺寸变化后重算卡片高度 / 参数列数（窄屏回落单列）。"""
        super().resizeEvent(e)
        self._equalize_cards()

    def _equalize_cards(self):
        """按真实内容高度统一三张卡片高度，杜绝内容溢出压到下方预览。

        要点：
        - 参数行按缺陷卡片可用宽度自适应 1 / 2 列，窗口宽时两列更矮；
        - 高度取 ``max(sizeHint, minimumSizeHint)``，保证任何一页的文字都不被裁；
        - 用 ``_adjusting`` 防重入（setFixedHeight 会触发布局重算）。
        """
        if self._adjusting or not getattr(self, "_card_in", None):
            return
        self._adjusting = True
        try:
            cards = (self._card_in, self._card_def, self._card_out)
            page_w = self._card_def.width() - 46      # 扣掉卡片内边距与 Tab 边框
            for p in self.pages.values():
                p.set_columns(2 if page_w >= DefectPage.TWO_COL_MIN else 1)
            tabbar_h = self.tabs.tabBar().sizeHint().height()
            tallest = 0
            for p in self.pages.values():
                p.body.updateGeometry()
                tallest = max(tallest,
                              p.sizeHint().height(), p.minimumSizeHint().height())
            self.tabs.setFixedHeight(tabbar_h + tallest + 2)
            need = 0
            for c in cards:
                need = max(need, c.sizeHint().height(), c.minimumSizeHint().height())
            for c in cards:
                c.setFixedHeight(need)
            self._top_h = need
        finally:
            self._adjusting = False

    # 上半部分卡片 1：输入设置
    def _input_card(self) -> QGroupBox:
        gb_in = QGroupBox("  1 · 输入设置  ")
        v = QVBoxLayout(gb_in)
        v.setSpacing(7)
        r1 = QHBoxLayout()
        b_file = QPushButton("选择图片…")
        b_file.clicked.connect(self.pick_files)
        b_dir = QPushButton("选择文件夹…")
        b_dir.clicked.connect(self.pick_dir)
        b_clr = QPushButton("清空")
        b_clr.clicked.connect(self.clear_inputs)
        r1.addWidget(b_file)
        r1.addWidget(b_dir)
        r1.addWidget(b_clr)
        v.addLayout(r1)
        # 上一张 / 下一张
        rn = QHBoxLayout()
        b_prev = QPushButton("◀ 上一张")
        b_prev.setToolTip("切换到上一张图片")
        b_prev.clicked.connect(lambda: self.step_image(-1))
        b_next = QPushButton("下一张 ▶")
        b_next.setToolTip("切换到下一张图片")
        b_next.clicked.connect(lambda: self.step_image(+1))
        self.btn_prev = b_prev
        self.btn_next = b_next
        rn.addWidget(b_prev)
        rn.addWidget(b_next)
        rn.addStretch(1)
        v.addLayout(rn)
        self.lst = QListWidget()
        self.lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst.setMinimumHeight(96)
        self.lst.currentRowChanged.connect(self.on_select_preview)
        v.addWidget(self.lst)
        self.lbl_count = QLabel("未选择任何图片")
        self.lbl_count.setProperty("role", "hint")
        v.addWidget(self.lbl_count)

        # 缺陷样本增强模式：导入一张缺陷图 → 在预览上手绘大致区域 →
        # 在该区域内随机生成多个、随机类型 / 位置 / 形状的缺陷（批量扩充训练样本）
        self.cb_augment = QCheckBox("缺陷样本增强模式")
        self.cb_augment.setToolTip(
            "在左侧预览上手绘「大致区域」后勾选本项：\n"
            "将按「生成图片数量」生成多张缺陷图（每张缺陷数由算法参数的「个数」决定），\n"
            "缺陷类型/位置/形状随机；生成后在中间画廊点击缩略图可在右侧查看大图。\n"
            "未绘制区域时退化为整图随机生成。")
        self.cb_augment.toggled.connect(self._on_augment_toggled)
        va = QHBoxLayout()
        va.addWidget(self.cb_augment)
        va.addStretch(1)
        v.addLayout(va)

        ra = QHBoxLayout()
        ra.addWidget(QLabel("生成图片数量"))
        self.sp_aug_count = QSpinBox()
        self.sp_aug_count.setRange(1, 100)
        self.sp_aug_count.setValue(6)
        self.sp_aug_count.setSuffix(" 张")
        self.sp_aug_count.setEnabled(False)
        ra.addWidget(self.sp_aug_count)
        ra.addStretch(1)
        v.addLayout(ra)
        return gb_in

    # 上半部分卡片 3：输出设置
    def _output_card(self) -> QGroupBox:
        gb_o = QGroupBox("  3 · 输出设置  ")
        vo = QVBoxLayout(gb_o)
        vo.setSpacing(7)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("输出目录"))
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("选择输出目录…")
        r2.addWidget(self.ed_out, 1)
        b_out = QPushButton("…")
        b_out.setFixedWidth(34)
        b_out.clicked.connect(self.pick_out)
        r2.addWidget(b_out)
        vo.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("每图生成"))
        self.sp_count = QSpinBox()
        self.sp_count.setRange(1, 100)
        self.sp_count.setValue(3)
        self.sp_count.setSuffix(" 张")
        r3.addWidget(self.sp_count)
        r3.addWidget(QLabel("随机种子"))
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 2_000_000_000)
        self.sp_seed.setValue(42)
        r3.addWidget(self.sp_seed)
        r3.addStretch(1)
        vo.addLayout(r3)

        r4 = QHBoxLayout()
        r4.addWidget(QLabel("长边缩放"))
        self.sp_max = QSpinBox()
        self.sp_max.setRange(0, 8192)
        self.sp_max.setSingleStep(64)
        self.sp_max.setValue(0)
        self.sp_max.setSuffix(" px（0=原尺寸）")
        r4.addWidget(self.sp_max, 1)
        vo.addLayout(r4)

        # 产物类型：按需求只有「缺陷图」默认开启，其余全部默认关闭
        self.cb_defect = QCheckBox("缺陷图")
        self.cb_defect.setChecked(True)
        self.cb_mask = QCheckBox("缺陷掩膜")
        self.cb_mask.setChecked(False)
        self.cb_prev = QCheckBox("对比图")
        self.cb_prev.setChecked(False)
        self.cb_region = QCheckBox("区域分割图")
        self.cb_region.setChecked(False)
        rc1 = QHBoxLayout()
        for cb in (self.cb_defect, self.cb_mask, self.cb_prev, self.cb_region):
            rc1.addWidget(cb)
        rc1.addStretch(1)
        vo.addLayout(rc1)

        self.cb_yolo = QCheckBox("YOLO txt")
        self.cb_yolo.setChecked(False)
        self.cb_voc = QCheckBox("VOC xml")
        self.cb_voc.setChecked(False)
        self.cb_coco = QCheckBox("COCO json")
        self.cb_coco.setChecked(False)
        rc2 = QHBoxLayout()
        for cb in (self.cb_yolo, self.cb_voc, self.cb_coco):
            rc2.addWidget(cb)
        rc2.addStretch(1)
        vo.addLayout(rc2)

        rp = QHBoxLayout()
        b_open = QPushButton("打开输出目录")
        b_open.clicked.connect(self.open_out)
        b_save = QPushButton("保存参数…")
        b_save.clicked.connect(self.save_preset)
        b_load = QPushButton("载入参数…")
        b_load.clicked.connect(self.load_preset)
        for b in (b_open, b_save, b_load):
            rp.addWidget(b)
        vo.addLayout(rp)
        return gb_o

    # 上半部分卡片 2：缺陷图生成配置（顶部直接是四个缺陷名的 Tab 分页）
    def _defect_card(self) -> QGroupBox:
        gb = QGroupBox("  2 · 缺陷图生成配置  ")
        vp = QVBoxLayout(gb)
        vp.setContentsMargins(12, 12, 12, 12)
        vp.setSpacing(8)

        # Tab 放在卡片顶部，标签即四个缺陷名（● 已启用 / ○ 未启用）
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._sync_defect_header)
        self.pages: dict[str, DefectPage] = {}
        for kind in DEFECT_KINDS:
            page = DefectPage(kind, DEFAULT_SETTINGS[kind])
            page.changed.connect(self.schedule_preview)
            self.pages[kind] = page
            self.tabs.addTab(page, f"  {KIND_CN[kind]}  ")
            page.enable.toggled.connect(self._sync_tab_titles)
        self._sync_tab_titles()
        vp.addWidget(self.tabs, 1)
        return gb

    def current_defect_kind(self) -> str:
        i = max(0, self.tabs.currentIndex())
        kinds = list(DEFECT_KINDS)
        return kinds[i] if i < len(kinds) else kinds[0]

    def _sync_defect_header(self, *_):
        """Tab 标签已含缺陷名与启用状态，这里只需刷新提示（状态栏）。"""
        kind = self.current_defect_kind()
        page = self.pages.get(kind)
        if page is None:
            return
        bar = getattr(self, "status", None)
        if bar is None or page is None:
            return
        if page.enable.isChecked():
            bar.showMessage(
                f"当前缺陷：{KIND_CN.get(kind, kind)} · 已启用"
                + ("　（限定在手绘区域内）"
                   if getattr(getattr(self, "preview", None), "draw_enabled", bool)()
                   else ""))

    def _sync_tab_titles(self):
        for i, kind in enumerate(DEFECT_KINDS):
            on = self.pages[kind].enable.isChecked()
            self.tabs.setTabText(i, f"  {'●' if on else '○'} {KIND_CN[kind]}  ")
        self._sync_defect_header()

    def _build_menu(self):
        m = self.menuBar()
        fm = m.addMenu("文件(&F)")
        a = QAction("选择图片…", self)
        a.setShortcut(QKeySequence("Ctrl+O"))
        a.triggered.connect(self.pick_files)
        fm.addAction(a)
        a = QAction("选择文件夹…", self)
        a.setShortcut(QKeySequence("Ctrl+Shift+O"))
        a.triggered.connect(self.pick_dir)
        fm.addAction(a)
        fm.addSeparator()
        a = QAction("打开输出目录", self)
        a.triggered.connect(self.open_out)
        fm.addAction(a)
        fm.addSeparator()
        a = QAction("退出", self)
        a.setShortcut(QKeySequence("Ctrl+Q"))
        a.triggered.connect(self.close)
        fm.addAction(a)

        hm = m.addMenu("帮助(&H)")
        a = QAction("使用说明", self)
        a.triggered.connect(self.show_help)
        hm.addAction(a)
        a = QAction("关于", self)
        a.triggered.connect(self.show_about)
        hm.addAction(a)

    # ---------------- 输入 ----------------
    def pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 MLCC 产品图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if files:
            self.add_inputs(files)

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择包含 MLCC 图片的文件夹")
        if not d:
            return
        found = list_images(d, recursive=True)
        if not found:
            QMessageBox.information(self, "提示", "该文件夹下没有找到受支持的图片文件。")
            return
        self.add_inputs(found)
        if not self.ed_out.text().strip():
            self.ed_out.setText(os.path.join(d, "_synth_defect_out"))

    def add_inputs(self, paths):
        for p in paths:
            if p not in self.input_paths:
                self.input_paths.append(p)
        self._refresh_list()
        if self.current_image is None and self.input_paths:
            self.lst.setCurrentRow(0)

    def _refresh_list(self):
        self.lst.clear()
        for i, p in enumerate(self.input_paths):
            it = QListWidgetItem(f"{i + 1:>3}.  {os.path.basename(p)}")
            it.setToolTip(p)
            self.lst.addItem(it)
        n = len(self.input_paths)
        self.lbl_count.setText(f"已选择 {n} 张图片" if n else "未选择任何图片")

    def clear_inputs(self):
        self.input_paths.clear()
        self.current_image = None
        self.current_path = ""
        self._refresh_list()
        self.preview.set_images(None, None, None)
        self.preview.clear_regions()
        self.preview.set_info("—")

    def on_select_preview(self, row: int):
        if 0 <= row < len(self.input_paths):
            try:
                self.current_path = self.input_paths[row]
                self.current_image = imread(self.current_path)
                self._seg_key = None
                self._seg_res = None
            except Exception as e:
                self.current_image = None
                self.status.showMessage(f"读取失败：{e}")
        else:
            self.current_image = None
            self.current_path = ""
        self.preview.clear_regions()
        self.schedule_preview()

    # ---------------- 上一张 / 下一张 ----------------
    def step_image(self, delta: int):
        """切换上一张 / 下一张图片。"""
        if not self.input_paths:
            self.status.showMessage("尚未导入图片")
            return
        row = self.lst.currentRow()
        nxt = row + int(delta)
        if row < 0:
            nxt = 0 if delta > 0 else len(self.input_paths) - 1
        nxt = max(0, min(len(self.input_paths) - 1, nxt))
        self.lst.setCurrentRow(nxt)
        self.lbl_count.setText(
            f"第 {nxt + 1} / {len(self.input_paths)} 张 ｜ "
            f"{os.path.basename(self.input_paths[nxt])}")

    # ---------------- 区域检测（内置恒开，不再暴露为设置项） ----------------
    def _seg_params(self) -> tuple:
        return (self.preview.draw_enabled(), self.preview.draw_mode())

    def _force_resegment(self):
        self._seg_key = None
        self._seg_res = None
        self.schedule_preview()

    def _on_region_drawn(self):
        self._seg_key = None
        self._seg_res = None
        self._refresh_overlay()      # 黄色虚线（限定范围）随手绘实时更新
        self.schedule_preview()

    def _on_draw_mode_changed(self, mode: str):
        self.status.showMessage(
            "绘制模式：在左侧「原图 / 手绘区域」上按住左键绘制" if mode != ImageView.MODE_VIEW
            else "浏览模式：滚轮缩放，左键拖拽平移，双击适应窗口")

    def _on_draw_toggled(self, on: bool):
        self._seg_key = None
        self._seg_res = None
        self.schedule_preview()
        self.status.showMessage(
            "已限定手绘区域：划痕沿手绘线条，脏污/异色/缺角在手绘区域内生成" if on
            else "已取消手绘限定：缺陷回到整个产品表面")

    #: 脏污/异色/缺角 只认这几种手绘方式（不含线条）
    AREA_KINDS = ("rect", "poly", "brush")

    def _draw_mask(self):
        """脏污/异色/缺角 的限定区域：矩形 + 多边形 + 画笔（不含线条）。"""
        if self.current_image is None:
            return None
        if not (self.preview.draw_enabled() and self.preview.has_drawn_regions()):
            return None
        return self.preview.draw_region_mask(self.current_image.shape[:2],
                                             include=self.AREA_KINDS)

    def _draw_paths(self):
        """划痕用的手绘线条点列（起止点即划痕起止点）。"""
        if self.current_image is None:
            return []
        if not (self.preview.draw_enabled() and self.preview.has_line_regions()):
            return []
        return self.preview.line_paths()

    def _draw_hint(self):
        """手绘区域的方向提示 (角度rad, cx, cy)：无手绘线条时用于划痕方向。"""
        if not (self.preview.draw_enabled() and self.preview.has_drawn_regions()):
            return None
        return self.preview.region_hint()

    def _hint_note(self) -> str:
        n = len(self._draw_paths())
        if n:
            return f" ｜ 划痕沿手绘线条 ×{n}"
        h = self._draw_hint()
        if h is None:
            return ""
        import math as _m
        return f" ｜ 划痕方向 {_m.degrees(h[0]) % 180:.0f}°"

    def _on_rect_angle(self, _deg: float):
        self._force_resegment()
        self.status.showMessage(
            f"ROI 角度已改为 {self.preview.sp_angle.value()}°，"
            "划痕会沿该方向生成")

    # ---------------- 缺陷位置方式 ----------------
    def _pos_mode(self) -> str:
        return self.preview.pos_mode()

    def _on_pos_mode_changed(self, mode: str):
        """切换「产品轮廓内 / 图像坐标」。两种模式的区域来源不同，需重算。"""
        self._seg_key = None
        self._seg_res = None
        self.schedule_preview()
        self.status.showMessage(
            "缺陷位置：限定在产品轮廓内（缺陷 = 产品表面 ∩ 手绘区域）"
            if mode == POS_MODE_PRODUCT else
            "缺陷位置：按图像坐标位置（忽略产品识别，手绘区域像素坐标直接生效）")

    def _region_masks_for(self, seg) -> Optional[dict]:
        """按当前「缺陷位置方式」算出用于生成缺陷的区域掩膜。

        - ``POS_MODE_PRODUCT``：**产品表面 ∩ 手绘区域**。
          交集在这里显式相乘完成（而不是把手绘区域喂给 segment_mlcc），
          这样 seg 保持为产品真实轮廓，轮廓线画出来才有参考意义。
        - ``POS_MODE_PIXEL``：直接把手绘区域像素掩膜当作可用表面，
          **完全绕开产品识别**，因此产品识别准不准都不影响缺陷落点。
        """
        dm = self._draw_mask()
        if self._pos_mode() == POS_MODE_PIXEL:
            if dm is None:
                return None
            # body/electrode 都填同一份手绘掩膜：target='both' 时取并集即为手绘区域
            return {"body": dm, "electrode": dm}
        if seg is None or (seg.body.max() <= 0.1 and seg.electrode.max() <= 0.1):
            return None
        if dm is None:
            return {"body": seg.body, "electrode": seg.electrode}
        body = clip01(seg.body * dm)
        elec = clip01(seg.electrode * dm)
        # ⚠️ 交集可能整块落空（手绘区域完全画在产品外）。此时必须返回 None，
        #    否则调用方拿到的是「两个全零掩膜」，resolve_target 会退化成 None，
        #    缺陷反而会洒满整图 —— 与限定区域的意图正好相反。
        if body.max() <= 0.1 and elec.max() <= 0.1:
            return None
        return {"body": body, "electrode": elec}

    def _refresh_overlay(self, *_):
        """重画产品轮廓 / 限定范围叠加层（颜色、线宽、显隐或区域变化时调用）。"""
        if self.current_image is None:
            return
        h, w = self.current_image.shape[:2]
        # 两种模式都画产品轮廓：在「图像坐标」模式下轮廓仅供对照参考，
        # 不参与缺陷落点计算，因此产品识别不准也不影响结果。
        seg = self._ensure_segmentation()
        show_p = self.preview.show_product_outline() and seg is not None
        show_d = self.preview.show_draw_area()
        if not show_p and not show_d:
            self.preview.set_overlay(None)
            return
        self.preview.set_overlay(product_outline_overlay(
            (h, w),
            product=seg.product if show_p else None,
            draw_region=self._draw_mask() if show_d else None,
            color=self.preview.outline_color(),
            width=self.preview.outline_width(),
            show_product=show_p, show_draw=show_d))

    def _ensure_segmentation(self):
        """对当前预览图运行（或取缓存）区域检测。"""
        if self.current_image is None:
            return None
        dm = self._draw_mask()
        # 分割**不喂**手绘区域：得到的是产品的真实轮廓，既用于画轮廓线，
        # 也用于「产品 ∩ 手绘」的交集计算。若把手绘区域喂进去，
        # seg.product 会被裁到手绘区里，画出来的「产品轮廓」就只是手绘框的复制，
        # 用户反而看不出产品真正的边界在哪。
        # 因此这里与 dm 无关，缓存键也不含 dm（画笔改动不必重新分割）。
        key = (self.lst.currentRow(),) + self._seg_params()
        if self._seg_key == key and self._seg_res is not None:
            return self._seg_res
        try:
            res = segment_mlcc(self.current_image)
        except Exception as e:
            self.status.showMessage(f"区域检测错误：{e}")
            res = None
        self._seg_key = key
        self._seg_res = res
        return self._seg_res

    # ---------------- 输出 ----------------
    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.ed_out.setText(d)

    def open_out(self):
        d = self.ed_out.text().strip()
        if not d or not os.path.isdir(d):
            QMessageBox.information(self, "提示", "输出目录还不存在，请先生成一次。")
            return
        try:
            os.startfile(d)  # type: ignore[attr-defined]
        except Exception:
            import subprocess

            subprocess.Popen(["explorer", os.path.normpath(d)])

    # ---------------- 参数 ----------------
    def current_settings(self) -> dict:
        return {k: p.config() for k, p in self.pages.items()}

    def schedule_preview(self):
        self._preview_timer.start()

    # ---------------- 缺陷样本增强模式 ----------------
    def _on_augment_toggled(self, on: bool):
        self.sp_aug_count.setEnabled(bool(on))
        if on:
            note = "" if (self.preview.draw_enabled()
                          and self.preview.has_drawn_regions()) else \
                "（尚未绘制区域，将在整图随机生成）"
            self.status.showMessage("缺陷样本增强模式：将生成设定张数的缺陷图，"
                                    "每张缺陷数由算法参数决定" + note)
        else:
            self.status.showMessage("已退出缺陷样本增强模式")
        self.schedule_preview()

    def _augment_region(self):
        """增强模式使用的手绘区域（rect/poly/brush）。未绘制则为 None（整图）。"""
        if self.current_image is None:
            return None
        if not (self.preview.draw_enabled() and self.preview.has_drawn_regions()):
            return None
        return self._draw_mask()

    def _aug_overlay(self, applied):
        """把缺陷掩膜叠加到原图上，生成"区域与缺陷"中间视图。"""
        mid = self.current_image.astype(np.float32).copy()
        for r in applied:
            color = _KIND_COLOR.get(r.kind, (255, 255, 255))
            a = (clip01(r.mask) * 0.6)[:, :, None]
            mid = mid * (1 - a) + np.array(color, np.float32)[None, None, :] * a
        return np.clip(mid, 0, 255).astype(np.uint8)

    def _refresh_augment_preview(self):
        self.status.showMessage("正在生成缺陷样本（增强模式）…")
        self.progress.setRange(0, 0)
        QApplication.processEvents()
        ok = False
        try:
            seed = int(self.sp_seed.value())
            region = self._augment_region()
            n = int(self.sp_aug_count.value())
            settings = self.current_settings()
            # 一次性生成 n 张缺陷图；每张缺陷数由算法参数的"个数"决定
            results = apply_augmentation(
                self.current_image, region=region, n_images=n, seed=seed,
                settings=settings, severity="mixed")
            self.aug_results = results
            self.gallery.set_thumbnails([img for img, _ in results])
            if results:
                img0, applied0 = results[0]
                mid = self._aug_overlay(applied0)
                self.preview.set_images(self.current_image, mid, img0)
                self.defect_view.set_image(img0)
                self.aug_selected = 0
                kinds = " + ".join(
                    f"{KIND_CN.get(r.kind, r.kind)}"
                    f"[{SEVERITY_CN.get(getattr(r, 'severity', 'mixed'), '?')}]"
                    for r in applied0)
                per = sum(len(a) for _, a in results)
                self.preview.set_info(
                    f"增强：共{n}张 ｜ 首图 {kinds} ｜ 缺陷占比"
                    f"{sum(float((r.mask > 0.35).mean()) for r in applied0) * 100:.2f}%"
                    f" ｜ 平均每图{per / max(1, len(results)):.0f}个 ｜ seed={seed}")
            else:
                self.preview.set_images(self.current_image, None, self.current_image)
                self.defect_view.set_image(self.current_image)
                self.preview.set_info("增强模式未生成缺陷"
                                     "（区域可能过小或类型不可行，可换个区域）"
                                     f" ｜ seed={seed}")
            ok = True
        except Exception as e:
            self.status.showMessage(f"增强预览错误：{e}")
            self.preview.set_images(self.current_image, None, self.current_image)
            self.defect_view.set_image(self.current_image)
            self.preview.set_info(f"增强预览出错：{e}")
        finally:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            if ok:
                self.status.showMessage("增强预览已更新")

    def _on_thumb_selected(self, idx: int):
        """点击画廊缩略图 → 在右侧缺陷视图（及左侧预览缺陷栏）展示该图。"""
        results = getattr(self, "aug_results", None)
        if not results or idx < 0 or idx >= len(results):
            return
        img, applied = results[idx]
        self.aug_selected = idx
        self.preview.views[2].set_image(img)
        self.defect_view.set_image(img)
        kinds = " + ".join(
            f"{KIND_CN.get(r.kind, r.kind)}"
            f"[{SEVERITY_CN.get(getattr(r, 'severity', 'mixed'), '?')}]"
            for r in applied)
        self.preview.set_info(
            f"增强：第{idx + 1}张 ｜ {kinds} ｜ 缺陷占比"
            f"{sum(float((r.mask > 0.35).mean()) for r in applied) * 100:.2f}%"
            f" ｜ 共{len(applied)}个")

    def _save_augment_image(self):
        out = self.ed_out.text().strip()
        if not out:
            QMessageBox.warning(self, "缺少输出", "请先在「输出设置」里选择输出目录。")
            return
        results = getattr(self, "aug_results", None)
        if not results:
            QMessageBox.information(self, "尚未生成",
                                    "请先点「刷新预览」生成增强样本，再保存。")
            return
        idx = getattr(self, "aug_selected", 0)
        if idx < 0 or idx >= len(results):
            idx = 0
        img, applied = results[idx]
        if not applied:
            QMessageBox.information(self, "未生成缺陷",
                                    "当前选中的样本没有生成任何缺陷，请换一张缩略图。")
            return
        try:
            seed = int(self.sp_seed.value())
            base = os.path.splitext(os.path.basename(self.current_path))[0] \
                if self.current_path else "defect"
            os.makedirs(out, exist_ok=True)
            kinds = "_".join(sorted({r.kind for r in applied}))
            name = f"{base}_aug{idx + 1}_{kinds}_n{len(applied)}_seed{seed}.png"
            dst = os.path.join(out, name)
            imwrite(dst, img)
            if self.cb_mask.isChecked():
                for r in applied:
                    imwrite(os.path.join(out, f"{base}_aug{idx + 1}_{r.kind}_mask.png"),
                            to_u8(clip01(r.mask) * 255.0))
            if self.cb_prev.isChecked():
                pre = self.defect_view.grab().toImage()
                imwrite(os.path.join(out, f"{base}_aug{idx + 1}_preview.png"),
                        np.array(pre))
            self.status.showMessage(f"已保存（增强 第{idx + 1}张）：{dst}")
            QMessageBox.information(self, "保存完成", f"已保存到：\n{dst}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            self.status.showMessage(f"保存失败：{e}")

    def _start_augment_batch(self, out: str):
        try:
            os.makedirs(out, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "无法创建输出目录", str(e))
            return
        region = self._augment_region()
        cfg = BatchConfig(
            input_paths=list(self.input_paths),
            output_dir=out,
            settings=self.current_settings(),
            images_per_input=int(self.sp_count.value()),
            seed=int(self.sp_seed.value()),
            max_side=int(self.sp_max.value()),
            use_region=False,                 # 增强用「手绘区域」约束，不依赖自动分割
            save_defect=self.cb_defect.isChecked(),
            save_mask=self.cb_mask.isChecked(),
            save_preview=self.cb_prev.isChecked(),
            save_region=self.cb_region.isChecked(),
            save_yolo=self.cb_yolo.isChecked(),
            save_voc=self.cb_voc.isChecked(),
            save_coco=self.cb_coco.isChecked(),
            augment=True,
            augment_region=region,
            augment_images=int(self.sp_aug_count.value()),
            augment_severity="mixed",
        )
        self.thread = QThread(self)
        self.worker = BatchWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.item_done.connect(self.on_item)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        note = "（按手绘区域生成）" if region is not None else "（未绘制区域，整图随机）"
        self.status.showMessage(
            f"开始增强生成：{len(self.input_paths)} 张 × {cfg.augment_images} 张/图 {note}")
        self.thread.start()

    def refresh_preview(self):
        if self.current_image is None:
            self.preview.set_images(None, None, None)
            self.preview.set_info("—")
            return
        # 用户可能画完线还没双击/回车结束就先点了刷新 —— 这里帮他提交
        self.preview.commit_pending()
        # 产品轮廓 / 限定范围叠加层：两种模式都画（先画好，之后切换模式不必重算分割）
        self._refresh_overlay()
        if self.cb_augment.isChecked():
            self._refresh_augment_preview()
            return
        # 非增强模式：清空画廊，缺陷图同步显示在右侧缺陷视图
        self.gallery.set_thumbnails([])
        settings = self.current_settings()
        seg = self._ensure_segmentation()
        seg_overlay = None
        if seg is not None and seg.ok and self._pos_mode() == POS_MODE_PRODUCT:
            seg_overlay = region_overlay(self.current_image, seg)

        # 划痕只认「手绘线条」：勾选了划痕但没画线时跳过，并在信息栏提示
        paths = self._draw_paths()
        scratch_on = bool(settings.get("scratch", {}).get("enable"))
        need_line = scratch_on and self.preview.draw_enabled()
        line_hint = ""
        if need_line and not paths:
            # 勾选了划痕但没画线：跳过划痕，并提示先画线
            settings = {k: dict(v) for k, v in settings.items()}
            settings["scratch"]["enable"] = False
            line_hint = " ｜ 划痕待画线（请用「线条」模式画一条）"
        elif paths:
            line_hint = f" ｜ 划痕沿手绘线条 ×{len(paths)}"

        if not any(v.get("enable") for v in settings.values()):
            # 第三栏仍显示当前图（而不是空白），否则看起来像「没有图片」
            self.preview.set_images(self.current_image, seg_overlay,
                                    self.current_image)
            self.preview.set_info(
                ("划痕待画线：请在左栏用「画线条」模式按住左键拖一条线"
                 "（松开即成线），或点「结束当前线条」"
                 if need_line and not paths else "未启用任何缺陷类型"))
            return
        # 进入缺陷生成：先给出明确的进度提示，避免界面在同步计算时看似卡死
        self.status.showMessage("正在生成缺陷预览…")
        self.progress.setRange(0, 100)
        self.progress.setRange(0, 0)  # 进度条进入忙碌动画
        QApplication.processEvents()  # 强制刷新，让提示先显示出来
        ok = False
        try:
            seed = int(self.sp_seed.value())
            region_masks = self._region_masks_for(seg)
            # 限定了手绘区域却算不出可用区域时，不生成缺陷并明确说明原因。
            # 两种模式的原因不同：产品轮廓模式是「手绘区没画到产品上」，
            # 像素坐标模式则不应发生（区域就是手绘掩膜本身）。
            dm = self._draw_mask()
            if dm is not None and region_masks is None:
                self.preview.set_images(self.current_image, seg_overlay,
                                        self.current_image)
                self.defect_view.set_image(self.current_image)
                if self._pos_mode() == POS_MODE_PIXEL:
                    msg = "手绘区域无效，请重新绘制限定区域"
                else:
                    msg = ("手绘区域未覆盖产品表面（缺陷位置方式为「限定在产品轮廓内」）。"
                           "请把限定区域画在产品上，或改用「按图像坐标位置」")
                self.preview.set_info(msg)
                self.status.showMessage(msg)
                return
            img, applied = apply_defects(self.current_image, settings, seed=seed,
                                         region_masks=region_masks,
                                         region_hint=self._draw_hint(),
                                         scratch_paths=paths)

            mid = (seg_overlay.astype(np.float32) if seg_overlay is not None
                   else self.current_image.astype(np.float32).copy())
            for r in applied:
                color = _KIND_COLOR.get(r.kind, (255, 255, 255))
                a = (clip01(r.mask) * 0.6)[:, :, None]
                mid = mid * (1 - a) + np.array(color, np.float32)[None, None, :] * a
            mid = np.clip(mid, 0, 255).astype(np.uint8)

            if applied:
                kinds = " + ".join(
                    f"{KIND_CN.get(r.kind, r.kind)}"
                    f"[{TARGET_CN.get(getattr(r, 'surface', 'both'), '?')}"
                    f"/{SEVERITY_CN.get(getattr(r, 'severity', 'mixed'), '?')}]"
                    for r in applied)
                cover = sum(float((r.mask > 0.35).mean()) for r in applied) * 100
                # 落点校验：缺陷是否越出允许范围
                # （产品轮廓模式看产品表面；像素坐标模式看手绘区域本身）
                limit = None
                if self._pos_mode() == POS_MODE_PIXEL:
                    limit = self._draw_mask()
                elif seg is not None and seg.product.max() > 0.1:
                    limit = seg.product
                leak = 0.0
                if limit is not None:
                    out_mask = (limit < 0.5)
                    total = sum(float((r.mask > 0.05).sum()) for r in applied)
                    if total > 0:
                        leak = sum(float(((r.mask > 0.05) & out_mask).sum())
                                   for r in applied) / total * 100
                mode_tag = ("产品轮廓内" if self._pos_mode() == POS_MODE_PRODUCT
                            else "图像坐标")
                self.preview.set_images(self.current_image, mid, img)
                self.defect_view.set_image(img)
                self.preview.set_info(
                    f"{kinds} ｜ 位置[{mode_tag}] ｜ 缺陷占比{cover:.2f}% ｜ 越界{leak:.2f}%"
                    f"{line_hint or self._hint_note()} ｜ seed={seed}")
            else:
                self.preview.set_images(self.current_image, mid, img)
                self.defect_view.set_image(img)
                why = ("（线条可能画在产品表面之外，已被区域约束剔除）"
                       if paths else "")
                self.preview.set_info(f"未生成缺陷{why} ｜ seed={seed}")
            ok = True
        except Exception as e:
            self.status.showMessage(f"预览错误：{e}")
            # 出错时也保证三栏有图（而不是第三栏空白）
            self.preview.set_images(self.current_image, seg_overlay,
                                    self.current_image)
            self.defect_view.set_image(self.current_image)
            self.preview.set_info(f"预览出错：{e}")
        finally:
            self.progress.setRange(0, 100)  # 复位为批量模式的默认范围
            self.progress.setValue(0)
            if ok:
                self.status.showMessage("预览已更新")

    # ---------------- 批量 ----------------
    def save_defect_image(self):
        """把「当前预览」这张缺陷图保存到输出目录（单张）。"""
        if self.current_image is None:
            QMessageBox.warning(self, "没有可保存的图", "请先选择一张图片。")
            return
        if self.cb_augment.isChecked():
            self._save_augment_image()
            return
        settings = self.current_settings()
        if not any(v.get("enable") for v in settings.values()):
            QMessageBox.warning(self, "未启用缺陷", "请至少启用一类缺陷。")
            return
        out = self.ed_out.text().strip()
        if not out:
            QMessageBox.warning(self, "缺少输出", "请先在「输出设置」里选择输出目录。")
            return
        # 复用预览的完整计算路径，保证保存结果与预览完全一致
        try:
            self.preview.commit_pending()
            seed = int(self.sp_seed.value())
            paths = self._draw_paths()
            scratch_on = bool(settings.get("scratch", {}).get("enable"))
            if scratch_on and self.preview.draw_enabled() and not paths:
                settings = {k: dict(v) for k, v in settings.items()}
                settings["scratch"]["enable"] = False
            seg = self._ensure_segmentation()
            region_masks = self._region_masks_for(seg)
            # 与预览一致：算不出可用区域时直接不生成，避免缺陷洒满整图
            if self._draw_mask() is not None and region_masks is None:
                tip = ("请把限定区域画在产品表面上。"
                       if self._pos_mode() == POS_MODE_PRODUCT else
                       "请重新绘制限定区域。")
                QMessageBox.information(
                    self, "未生成缺陷",
                    "手绘区域没有可用的生成范围。\n\n" + tip
                    + "\n\n（可在「缺陷位置」里改为「按图像坐标位置」，"
                      "此时不受产品识别影响）")
                return
            img, applied = apply_defects(self.current_image, settings, seed=seed,
                                         region_masks=region_masks,
                                         region_hint=self._draw_hint(),
                                         scratch_paths=paths)
            if not applied:
                QMessageBox.information(
                    self, "未生成缺陷",
                    "当前设置下没有生成任何缺陷。\n\n"
                    +("若已勾选「限定手绘区域」，请确认已用「画线条」模式画一条线。"
                      if settings.get("scratch", {}).get("enable") is None else ""))
                return
            base = os.path.splitext(os.path.basename(self.current_path))[0] \
                if self.current_path else "defect"
            os.makedirs(out, exist_ok=True)
            # 文件名带缺陷类型与 seed，便于区分多次保存
            kinds = "_".join(sorted({r.kind for r in applied}))
            name = f"{base}_{kinds}_seed{seed}.png"
            dst = os.path.join(out, name)
            imwrite(dst, img)
            # 可选产物
            if self.cb_mask.isChecked():
                for r in applied:
                    imwrite(os.path.join(out, f"{base}_{r.kind}_seed{seed}_mask.png"),
                            to_u8(clip01(r.mask) * 255.0))
            if self.cb_prev.isChecked():
                pre = self.preview.views[2].grab().toImage()
                imwrite(os.path.join(out, f"{base}_{kinds}_seed{seed}_preview.png"),
                        np.array(pre))
            self.status.showMessage(f"已保存：{dst}")
            QMessageBox.information(self, "保存完成", f"已保存到：\n{dst}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            self.status.showMessage(f"保存失败：{e}")

    def start_batch(self):
        if self.worker is not None:
            return
        if not self.input_paths:
            QMessageBox.warning(self, "缺少输入", "请先选择至少一张图片。")
            return
        out = self.ed_out.text().strip()
        if not out:
            QMessageBox.warning(self, "缺少输出", "请选择输出目录。")
            return
        if self.cb_augment.isChecked():
            self._start_augment_batch(out)
            return
        settings = self.current_settings()
        if not any(v.get("enable") for v in settings.values()):
            QMessageBox.warning(self, "未启用缺陷", "请至少启用一类缺陷。")
            return
        dm = self._draw_mask()
        if self.preview.draw_enabled() and dm is None:
            QMessageBox.information(
                self, "未绘制区域",
                "已勾选「限定手绘区域」但尚未绘制。请在预览工具栏选择"
                "「画矩形区域」或「画笔涂抹」绘制，或取消勾选该项。\n\n"
                "注意：手绘区域仅作用于当前选中的这张图片。")
            return
        try:
            os.makedirs(out, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "无法创建输出目录", str(e))
            return

        cfg = BatchConfig(
            input_paths=list(self.input_paths),
            output_dir=out,
            settings=settings,
            images_per_input=int(self.sp_count.value()),
            seed=int(self.sp_seed.value()),
            max_side=int(self.sp_max.value()),
            use_region=True,
            save_defect=self.cb_defect.isChecked(),
            save_mask=self.cb_mask.isChecked(),
            save_preview=self.cb_prev.isChecked(),
            save_region=self.cb_region.isChecked(),
            save_yolo=self.cb_yolo.isChecked(),
            save_voc=self.cb_voc.isChecked(),
            save_coco=self.cb_coco.isChecked(),
        )

        self.thread = QThread(self)
        self.worker = BatchWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.item_done.connect(self.on_item)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.showMessage(
            f"开始生成：{len(self.input_paths)} 张 × {cfg.images_per_input}")
        self.thread.start()

    def stop_batch(self):
        if self.worker:
            self.worker.stop()
            self.btn_stop.setEnabled(False)
            self.status.showMessage("已请求停止，正在收尾…")

    def on_progress(self, done: int, total: int, name: str):
        pct = int(done / max(total, 1) * 100)
        self.progress.setValue(pct)
        self.status.showMessage(f"处理中 {done}/{total}（{pct}%）  {name}")

    def on_item(self, rec: ItemResult):
        if not rec.ok:
            self.status.showMessage(f"跳过 {os.path.basename(rec.src)}："
                                    f"{rec.error.splitlines()[0]}")

    def on_finished(self, info: dict):
        self._teardown_thread()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(100)
        msg = (f"成功 {info.get('images_generated', 0)} 张，"
               f"失败 {info.get('images_failed', 0)} 张")
        self.status.showMessage("完成：" + msg)
        detail = "、".join(f"{KIND_CN.get(k, k)} {n} 框"
                          for k, n in (info.get("boxes_per_class") or {}).items())
        QMessageBox.information(
            self, "生成完成",
            f"{msg}\n{detail}\n\n输出目录：\n{self.ed_out.text().strip()}")

    def on_failed(self, msg: str):
        self._teardown_thread()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.showMessage("运行失败：" + msg.splitlines()[0])
        QMessageBox.critical(self, "运行失败", msg[:2000])

    def _teardown_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait(4000)
            self.thread = None
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    # ---------------- 预设 ----------------
    def preset_data(self) -> dict:
        """打包当前全部参数（供保存 / 测试使用）。"""
        return {
            "settings": self.current_settings(),
            "images_per_input": self.sp_count.value(),
            "seed": self.sp_seed.value(),
            "max_side": self.sp_max.value(),
            "region": {"auto": True,
                       "use_draw": self.preview.draw_enabled(),
                       "draw_mode": self.preview.draw_mode(),
                       "roi_angle": self.preview.sp_angle.value()},
        }

    def save_preset_checked(self, path: str) -> bool:
        """无对话框保存（测试友好）。"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.preset_data(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.status.showMessage(f"保存失败：{e}")
            return False
        self.status.showMessage(f"参数已保存：{path}")
        return True

    def load_preset_checked(self, path: str) -> bool:
        """无对话框载入（测试友好）。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.status.showMessage(f"载入失败：{e}")
            return False
        self._apply_preset(data)
        return True

    def save_preset(self):
        p, _ = QFileDialog.getSaveFileName(self, "保存参数", "defect_preset.json",
                                           "JSON 文件 (*.json)")
        if not p:
            return
        self.save_preset_checked(p)

    def load_preset(self):
        p, _ = QFileDialog.getOpenFileName(self, "载入参数", "", "JSON 文件 (*.json)")
        if not p:
            return
        if not self.load_preset_checked(p):
            QMessageBox.critical(self, "载入失败", "无法读取该参数文件。")

    def _apply_preset(self, data: dict):
        for kind, cfg in (data.get("settings") or {}).items():
            page = self.pages.get(kind)
            if not page:
                continue
            page.enable.setChecked(bool(cfg.get("enable", True)))
            for k, v in cfg.items():
                if k in page.rows:
                    page.rows[k].setValue(v)
                elif k in page.combos:
                    i = page.combos[k].findData(v)
                    if i >= 0:
                        page.combos[k].setCurrentIndex(i)
            # 自定义异色目标色：恢复取色按钮的色板
            if "color_rgb" in cfg and getattr(page, "btn_color", None) is not None:
                try:
                    page.color_rgb = tuple(int(x) for x in cfg["color_rgb"])
                    page._apply_color_swatch()
                except Exception:
                    pass
        if "images_per_input" in data:
            self.sp_count.setValue(int(data["images_per_input"]))
        if "seed" in data:
            self.sp_seed.setValue(int(data["seed"]))
        if "max_side" in data:
            self.sp_max.setValue(int(data["max_side"]))
        rg = data.get("region") or {}
        if "use_draw" in rg:
            self.preview.cb_draw.setChecked(bool(rg["use_draw"]))
        if "roi_angle" in rg:
            self.preview.sp_angle.setValue(int(rg["roi_angle"]))
        self._sync_tab_titles()
        self.schedule_preview()

    # ---------------- 杂项 ----------------
    def show_help(self):
        QMessageBox.information(self, "使用说明", HELP_TEXT)

    def show_about(self):
        QMessageBox.about(
            self, "关于 SynthDefect AI",
            "<h3>SynthDefect AI <span style='color:#98a2b3'>v1.11</span></h3>"
            "<p>MLCC 表面缺陷合成工具：自动查找产品 → 约束缺陷落点 → "
            "合成划痕 / 脏污 / 异色 / 缺角。</p>"
            "<p>支持可旋转 ROI 手绘限定、划痕方向与颜色可调、"
            "缺陷明显度分级、预览缩放平移。</p>"
            "<p>纯本地运行，无需联网、无需 GPU、无需大模型 API。</p>")

    def closeEvent(self, e):
        if self.worker is not None:
            r = QMessageBox.question(self, "正在生成", "任务尚未完成，确定要退出吗？")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.worker.stop()
            self._teardown_thread()
        e.accept()


HELP_TEXT = """\
【适用场景】
导入合格的 MLCC 产品照片，自动定位瓷体与端电极表面，
只在产品表面合成「划痕 / 脏污 / 异色 / 缺角」缺陷，并支持明显 / 中等 / 不明显分级，
用于扩充工业视觉检测模型的训练集。

【操作步骤】
1. 顶部「1 · 输入设置」卡片选择图片或文件夹导入 MLCC 原图。
2. 「限定手绘区域」**默认开启**，在「原图 / 手绘区域」栏选择绘制方式：
   · **画线条（划痕专用）**：**按住左键拖一条线**（松开即成一条划痕），
     也可逐点点击画折线，双击 / 右键 / 回车 / 「结束当前线条」结束；
     **划痕沿这条线生成，起止位置与手绘一致**。
   · **画矩形区域**：拖拽画框，按住 **Ctrl 左键拖动**可旋转，
     也可用第二行的「ROI角度」输入框或「＋90° / －90°」精确调整。
   · **画多边形**：逐点点击，点回起点或双击 / 右键 / 回车闭合。
   · **画笔涂抹**：按住左键自由涂抹（笔刷大小可调，笔刷仅在该模式下出现）。
   手绘范围与产品区域**固定取交集**，画到产品外的部分不会生成缺陷。
   线条只服务于划痕；脏污 / 异色 / 缺角只认矩形框、多边形、画笔。
   没画完的线条 / 多边形，点「刷新预览」或切换绘制模式时会自动提交。
3. 「2 · 缺陷图生成配置」卡片顶部就是**四个缺陷名的 Tab 分页**（● 已启用 / ○ 未启用），
   在各 Tab 内配置参数（明显度、强度、数量等），预览实时刷新。
   **划痕默认开启**，脏污 / 异色 / 缺角默认关闭（需勾选各页内的「启用」复选框）。
   · 数量（条数 / 斑块数 / 区域数 / 缺角数）默认 1。画了 N 条线时，
     「条数」表示**每条线生成几条平行划痕**；
   · 划痕「弯曲度」0=近直线、1=明显蜿蜒（仅未画线、按区域方向自动生成时生效）；
   · 划痕「颜色」-1=白 … +1=黑，0=随机暗/亮；「深浅」0.1~1.5 控制偏离原色的程度；
   · 脏污 / 异色的「边缘模糊」0~2.5，默认 1.0（边缘为渐变过渡，没有硬边）。
4. 输入设置卡片用「◀ 上一张 / 下一张 ▶」逐张查看。
5. 设置每图张数、随机种子、输出目录，点底部「**保存缺陷图**」把当前这张预览
   的缺陷图存到输出目录；点「**批量生成全部**」可对所有导入图片批量生成。
   输出设置里只有「缺陷图」默认勾选，其余产物（掩膜 / 对比图 / 区域图 / 标注）默认关闭。

【划痕的两种生成方式】
· **手绘线条（预览）**：勾选「限定手绘区域」→ 选「画线条」→ **按住左键拖一条线**，
  松开即成一条划痕；划痕严格沿这条线，起止点与手绘一致。
  没画线时预览会提示「划痕待画线」，第三栏仍显示原图（不会空白）。
· **批量生成**：批处理是逐图自动跑，没有手绘，
  此时划痕按「目标表面长轴方向」自动生成（与 v1.8 行为一致）。

【产品区域自动定位】
区域检测已内置并常开：程序自动查找 MLCC 产品，
绿色 = 瓷体，蓝色 = 端电极，缺陷不会落到背景上。

【划痕方向与颜色】
· 方向：不再随机。手绘了 ROI / 笔迹时沿绘制方向；未手绘时沿目标表面（瓷体/端电极）
  的长轴方向，只保留 ±10° 的自然抖动。
· 颜色：「颜色」滑条从白到黑连续可调；「深浅」控制发白/发黑的程度。
  颜色取 0 时按「凹陷暗线 / 亮反光线」随机（与真实划痕一致）。

【预览操作】
· 滚轮缩放（以光标为锚点）；左键拖拽或中键拖拽平移
· 双击画面 = 适应窗口；工具栏提供 适应 / 1:1 / ＋ / －
· 三栏视图缩放同步

【界面自适应】
· 顶部三张卡片（输入 / 缺陷配置 / 输出）高度一致，
  窗口变宽时参数行自动排成两列、变窄时回落单列，保证文字始终完整。
· 预览工具条分两行（图像操作 / 状态信息），窄窗口下控件不会被挤扁。
· 窗口最小尺寸 1180×720，再小会互相遮挡，故不再允许继续缩小。

【手绘区域三种作用方式】
· 与产品求交（推荐）：手绘 ∩ 自动产品，最稳妥
· 并入产品区域：手绘区域被并入产品（可圈定自动漏检的目标）
· 完全以手绘为准：忽略自动分割，完全按手绘范围生成

【缺角缺陷】
在产品角部沿断面方向崩掉一块：断面呈锯齿状贝壳纹，
带颗粒感与深度阴影，可选延伸裂纹与细小碎屑。
参数：缺角数 / 缺角尺寸 / 崩口深度 / 断面锯齿 / 延伸裂纹。

【产物】
· images/ 缺陷图 ｜ masks/<类型>/ 掩膜 ｜ region/ 区域分割图
· previews/<类型>/ 三联对比图
· labels/ YOLO txt ｜ annotations_voc/ VOC xml ｜ annotations_coco.json
· dataset_info.json 记录全部参数与统计

【提示】
· 手绘区域仅对当前选中的那一张图片生效。
· 抓拍建议：产品充满画面、与背景对比清晰。
· 检测任务建议明显度选「随机混合」+ 多种子，覆盖困难样本谱系。
"""
