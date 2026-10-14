"""三栏预览面板（原图 / 区域与缺陷 / 缺陷图），支持缩放平移与可旋转 ROI 手绘。"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame,
                             QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                             QSlider, QSpinBox, QVBoxLayout, QWidget)

from .image_view import ImageView


class PreviewPane(QWidget):
    """原图 | 区域与缺陷 | 缺陷图 三栏预览。

    - 三栏缩放/平移同步
    - 最左栏可切换为「手绘区域」模式（矩形 / 涂抹）
    - 顶部工具条承载手绘开关（画笔 / 可旋转矩形 ROI）+ 缩放
    """

    regionChanged = pyqtSignal()
    modeChanged = pyqtSignal(str)
    drawToggled = pyqtSignal(bool)          # 启用手绘区域
    rectAngleChanged = pyqtSignal(float)    # ROI 框角度（度）

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # views / caps 必须先于 _build_bar() 存在：工具条构建时会把绘制模式
        # 切到「画线条」，从而触发 _on_mode() 访问 views[0]
        self.views: List[ImageView] = []
        self.caps: List[QLabel] = []

        lay.addLayout(self._build_bar())

        # ---- 三栏 ----
        row = QHBoxLayout()
        row.setSpacing(8)
        for title in ("原图 / 手绘区域", "区域与缺陷", "缺陷图"):
            box = QVBoxLayout()
            box.setSpacing(4)
            cap = QLabel(title)
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap.setProperty("role", "hint")
            view = ImageView()
            view.setSizePolicy(QSizePolicy.Policy.Expanding,
                               QSizePolicy.Policy.Expanding)
            view.zoomChanged.connect(self._on_zoom_changed)
            box.addWidget(view, 1)
            box.addWidget(cap)
            row.addLayout(box, 1)
            self.views.append(view)
            self.caps.append(cap)

        lay.addLayout(row, 1)

        self._syncing = False
        # 三栏就绪后再切默认模式（_on_mode 会访问 views[0]，必须晚于上面创建）
        if self.cb_draw.isChecked() and self.cmb_mode.currentIndex() == 0:
            self.cmb_mode.setCurrentIndex(1)      # 默认进入「画线条（划痕）」

    # ---------------- 工具栏 ----------------
    def _build_bar(self) -> QVBoxLayout:
        """三条工具行：① 手绘与 ROI ② 缩放③ 状态信息。

        控件按「当前绘制模式」动态显隐（画笔滑条仅涂抹模式、ROI角度仅矩形模式），
        否则一行放不下会被挤扁（QBoxLayout 空间不足时压固定宽度子项）。
        """
        wrap = QVBoxLayout()
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.setSpacing(6)
        wrap.addLayout(self._bar_draw())
        wrap.addLayout(self._bar_zoom())
        wrap.addLayout(self._bar_info())
        self._sync_brush_enabled()
        return wrap

    def _bar_draw(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(6)
        t = QLabel("效果预览")
        t.setProperty("role", "title")
        bar.addWidget(t)
        bar.addSpacing(8)

        # ---- 手绘区域 ----
        self.cb_draw = QCheckBox("限定手绘区域")
        self.cb_draw.setChecked(True)          # 默认开启
        self.cb_draw.setToolTip("缺陷只在手绘范围内生成（划痕沿手绘线条，"
                                 "脏污/异色/缺角在手绘的矩形·多边形·画笔内）")
        self.cb_draw.toggled.connect(self._on_draw_toggle)
        bar.addWidget(self.cb_draw)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("浏览", ImageView.MODE_VIEW)
        self.cmb_mode.addItem("画线条（划痕）", ImageView.MODE_LINE)
        self.cmb_mode.addItem("画矩形区域", ImageView.MODE_RECT)
        self.cmb_mode.addItem("画多边形", ImageView.MODE_POLY)
        self.cmb_mode.addItem("画笔涂抹", ImageView.MODE_BRUSH)
        self.cmb_mode.setFixedWidth(150)
        self.cmb_mode.setToolTip("线条：划痕专用（可拖动或逐点点击，起止点即划痕起止点）\n"
                                 "矩形/多边形/画笔：脏污、异色、缺角的区域限定")
        self.cmb_mode.currentIndexChanged.connect(self._on_mode)
        bar.addWidget(self.cmb_mode)

        # 画笔相关（仅涂抹模式显示）
        self.lbl_brush = QLabel("笔刷")
        bar.addWidget(self.lbl_brush)
        self.sld_brush = QSlider(Qt.Orientation.Horizontal)
        self.sld_brush.setRange(4, 80)
        self.sld_brush.setValue(14)
        self.sld_brush.setFixedWidth(90)
        self.sld_brush.valueChanged.connect(
            lambda v: [v0.set_brush_radius(v) for v0 in self.views])
        bar.addWidget(self.sld_brush)

        b_clr = QPushButton("清除手绘")
        b_clr.clicked.connect(self.clear_regions)
        bar.addWidget(b_clr)

        bar.addStretch(1)

        # 结束当前线条/多边形（点回起点、双击、右键、回车之外再给一个按钮）
        b_fin = QPushButton("结束当前线条")
        b_fin.setToolTip("把正在画的线条/多边形提交（等同双击或回车）")
        b_fin.clicked.connect(self.finish_pending)
        self.btn_finish = b_fin
        bar.addWidget(b_fin)
        return bar

    def _bar_zoom(self) -> QHBoxLayout:
        """缩放 + ROI 角度。ROI 角度只在矩形模式显示。"""
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self.lbl_angle = QLabel("ROI角度")
        self.lbl_angle.setToolTip("旋转矩形 ROI 的角度（度）。"
                                   "也可以在图上按住 Ctrl 左键拖动来旋转")
        bar.addWidget(self.lbl_angle)
        self.sp_angle = QSpinBox()
        self.sp_angle.setRange(-180, 180)
        self.sp_angle.setValue(0)
        self.sp_angle.setSuffix("°")
        self.sp_angle.setFixedWidth(126)
        self.sp_angle.setToolTip("ROI 矩形角度，0=水平；划痕会沿该方向生成")
        self.sp_angle.valueChanged.connect(self._on_angle_changed)
        bar.addWidget(self.sp_angle)

        b_rot_l = QPushButton("－90°")
        b_rot_l.setFixedWidth(70)
        b_rot_l.clicked.connect(lambda: self.rotate_rect(-90))
        bar.addWidget(b_rot_l)
        b_rot_r = QPushButton("＋90°")
        b_rot_r.setFixedWidth(70)
        b_rot_r.clicked.connect(lambda: self.rotate_rect(90))
        bar.addWidget(b_rot_r)

        bar.addStretch(1)

        b_out = QPushButton("－")
        b_out.setFixedWidth(40)
        b_out.setToolTip("缩小")
        b_out.clicked.connect(lambda: self.zoom_all(1 / 1.25))
        bar.addWidget(b_out)
        b_1to1 = QPushButton("1:1")
        b_1to1.setToolTip("按1 图像像素 = 1 屏幕像素显示")
        b_1to1.clicked.connect(self.zoom_1to1)
        bar.addWidget(b_1to1)
        b_fit = QPushButton("适应窗口")
        b_fit.setToolTip("三栏都缩放到刚好放下整图")
        b_fit.clicked.connect(self.fit_all)
        bar.addWidget(b_fit)
        b_in = QPushButton("＋")
        b_in.setFixedWidth(40)
        b_in.setToolTip("放大")
        b_in.clicked.connect(lambda: self.zoom_all(1.25))
        bar.addWidget(b_in)
        return bar

    def _bar_info(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.lbl_info = QLabel("—")
        self.lbl_info.setProperty("role", "hint")
        self.lbl_info.setWordWrap(False)
        bar.addWidget(self.lbl_info, 1)
        return bar

    def rotate_rect(self, deg: float):
        """±90° 快捷旋转：直接改矩形角度（没有矩形时提示）。"""
        v = self.views[0]
        if v is not None and v.rotate_rect_by(float(deg)):
            self.sp_angle.setValue(int(round(v.rect_angle_deg() - 180)) % 360 - 180
                                   if v.rect_angle_deg() > 180
                                   else int(round(v.rect_angle_deg())))
            return
        self.set_info("请先在「原图/ 手绘区域」栏画一个矩形 ROI，再旋转")

    def finish_pending(self):
        """提交正在画的线条 / 多边形（工具栏按钮入口）。"""
        v = self.views[0]
        if v is not None and v.finish_poly():
            self.regionChanged.emit()

    def _on_angle_changed(self, deg: int):
        """角度输入框 → 旋转最后一个 ROI 框 → 通知主窗口重算。"""
        v = self.views[0]
        if v is not None and v.set_rect_angle(float(deg)):
            self.rectAngleChanged.emit(float(deg))
        elif v is not None and not v.has_rects():
            self.set_info("请先画一个矩形 ROI（模式选「画矩形区域」），再设置角度")

    def has_roi(self) -> bool:
        """是否已有矩形 ROI（角度控件的可用性）。"""
        v = self.views[0]
        return bool(v is not None and v.has_rects())

    def commit_pending(self) -> bool:
        """若还有未提交的线条/多边形，先提交（刷新预览前调用）。"""
        v = self.views[0]
        if v is None:
            return False
        if v.has_pending():
            if v.finish_poly():
                return True
        return False

    # ---------------- 数据 ----------------
    @staticmethod
    def _sep() -> QFrame:
        ln = QFrame()
        ln.setFrameShape(QFrame.Shape.VLine)
        ln.setStyleSheet("color:#2e353f; background:#2e353f; max-width:1px;")
        return ln

    def set_images(self, orig, mask_rgb, defect):
        for i, arr in enumerate((orig, mask_rgb, defect)):
            self.views[i].set_image(arr)

    def set_info(self, text: str):
        self.lbl_info.setText(text)

    def clear_regions(self):
        self.views[0].clear_regions()
        self.regionChanged.emit()

    def draw_region_mask(self, shape, include=None):
        return self.views[0].region_mask(shape, include=include)

    def has_drawn_regions(self) -> bool:
        return self.views[0].has_regions()

    def has_line_regions(self) -> bool:
        """是否画了线条（划痕专用）。"""
        return self.views[0].has_line_regions()

    def line_paths(self):
        """手绘线条点列（划痕沿其生成）。"""
        return self.views[0].line_paths()

    def region_hint(self):
        """手绘区域方向提示 (角度rad, cx, cy)。"""
        return self.views[0].region_hint()

    def rect_angle_deg(self) -> float:
        return float(self.sp_angle.value())

    def draw_enabled(self) -> bool:
        return self.cb_draw.isChecked()

    def draw_mode(self) -> str:
        """手绘范围与产品区域固定取交集（已移除合并方式下拉）。"""
        return "intersect"

    def _on_zoom_changed(self, zoom: float):
        """把源视图的缩放/平移同步到其余两栏。

        注意：只在 **用户交互（滚轮/按钮/拖拽）** 时触发同步。
        fit_all / zoom_1to1 这类「批量归一化」操作会一次性处理三栏，
        它们各自发出的 zoomChanged 不能再触发同步，否则会互相覆盖
        _fit_mode，导致「适应」被误标为「非适应」。
        """
        if self._syncing:
            return
        self._syncing = True
        src = self.sender()
        for v in self.views:
            if v is not src:
                v._zoom = zoom
                v._fit_mode = getattr(src, "_fit_mode", False)
                if v._fit_mode:
                    v._fit_scale = min(v.width() / max(v._img_size.width(), 1),
                                       v.height() / max(v._img_size.height(), 1))
                v._offset = QPointF(src._offset)
                v.update()
        self._syncing = False

    def zoom_all(self, factor: float):
        self._syncing = True
        try:
            for v in self.views:
                v.zoom_by(factor)
        finally:
            self._syncing = False
        self._realign()

    def fit_all(self):
        self._syncing = True
        try:
            for v in self.views:
                v.fit_to_window()
        finally:
            self._syncing = False
        self._realign()

    def zoom_1to1(self):
        self._syncing = True
        try:
            for v in self.views:
                v.zoom_1to1()
        finally:
            self._syncing = False
        self._realign()

    def _realign(self):
        """三栏按第一栏的缩放状态对齐（fit 模式下各栏自算适配比例）。"""
        ref = self.views[0]
        for v in self.views[1:]:
            v._zoom = ref._zoom
            v._fit_mode = ref._fit_mode
            if ref._fit_mode:
                v.reanchor()
            else:
                v._offset = QPointF(ref._offset)
                v.update()

    # ---------------- 模式 ----------------
    def _on_mode(self, _idx: int):
        # 切换模式前先提交上一个模式未完成的线条/多边形，避免被丢弃
        self.commit_pending()
        mode = self.cmb_mode.currentData()
        # 只在第一栏开启绘制，其余两栏保持浏览
        self.views[0].set_mode(mode)
        for v in self.views[1:]:
            v.set_mode(ImageView.MODE_VIEW)
        self._sync_brush_enabled()
        self.modeChanged.emit(mode)

    def _on_draw_toggle(self, on: bool):
        """手绘总开关：关闭时回到浏览模式并清掉范围，缺陷回到全产品面。"""
        if not on:
            self.commit_pending()
            self.cmb_mode.setCurrentIndex(0)   # 浏览
            self.clear_regions()
        else:
            self.cmb_mode.setCurrentIndex(1)   # 画线条（划痕专用）
        self.modeChanged.emit(self.cmb_mode.currentData())
        self.drawToggled.emit(on)

    def _sync_brush_enabled(self):
        """按「是否手绘 + 当前绘制模式」显隐无关控件，保证工具条不被挤扁。"""
        draw_on = self.cb_draw.isChecked()
        mode = self.cmb_mode.currentData()
        self.cmb_mode.setEnabled(draw_on)
        brush = draw_on and mode == ImageView.MODE_BRUSH
        self.lbl_brush.setVisible(brush)
        self.sld_brush.setVisible(brush)
        rect = draw_on and mode == ImageView.MODE_RECT
        for w in (self.lbl_angle, self.sp_angle):
            w.setVisible(rect)
        self._rot_btns = [b for b in (getattr(self, "_rot_l", None),
                                     getattr(self, "_rot_r", None)) if b is not None]
        # 线条 / 多边形模式下才需要「结束当前线条」
        self.btn_finish.setVisible(mode in (ImageView.MODE_LINE, ImageView.MODE_POLY))
