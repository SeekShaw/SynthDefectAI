"""三栏预览面板（原图 / 区域与缺陷 / 缺陷图），支持缩放平移与可旋转 ROI 手绘。"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
                             QFrame, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget)

from .image_view import ImageView

#: 缺陷位置方式：限定在自动识别到的产品轮廓内
POS_MODE_PRODUCT = "product"
#: 缺陷位置方式：直接按手绘区域的图像像素坐标，不依赖产品识别
POS_MODE_PIXEL = "pixel"


class PreviewPane(QWidget):
    """原图 | 区域与缺陷 | 缺陷图 三栏预览。

    - 三栏缩放/平移同步
    - 最左栏可切换为「手绘区域」模式（矩形 / 涂抹）
    - 顶部工具条承载手绘开关（画笔 / 可旋转矩形 ROI）+ 缩放
    - 「缺陷位置方式」决定限定区域语义：产品轮廓内 / 纯图像坐标
    """

    regionChanged = pyqtSignal()
    modeChanged = pyqtSignal(str)
    drawToggled = pyqtSignal(bool)          # 启用手绘区域
    rectAngleChanged = pyqtSignal(float)    # ROI 框角度（度）
    posModeChanged = pyqtSignal(str)        # 缺陷位置方式变化
    overlayChanged = pyqtSignal()           # 轮廓颜色/线宽/显隐变化
    defectMove = pyqtSignal(int, int)       # 缺陷位置微调 (dx, dy) 像素

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # views / caps 必须先于 _build_bar() 存在：工具条构建时会把绘制模式
        # 切到「画线条」，从而触发 _on_mode() 访问 views[0]
        self.views: List[ImageView] = []
        self.caps: List[QLabel] = []
        self._overlay_arr = None      # 最近一次的产品轮廓叠加层 (H,W,4)

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
        wrap.addLayout(self._bar_posmode())
        wrap.addLayout(self._bar_zoom())
        wrap.addLayout(self._bar_info())
        self._sync_brush_enabled()
        return wrap

    def _bar_posmode(self) -> QHBoxLayout:
        """缺陷位置方式：产品轮廓内 / 图像坐标 + 轮廓显示设置。"""
        bar = QHBoxLayout()
        bar.setSpacing(6)

        t = QLabel("缺陷位置")
        t.setProperty("role", "title")
        bar.addWidget(t)

        # ---- 位置方式 ----
        self.cmb_posmode = QComboBox()
        self.cmb_posmode.addItem("限定在产品轮廓内", POS_MODE_PRODUCT)
        self.cmb_posmode.addItem("按图像坐标位置", POS_MODE_PIXEL)
        # 默认「按图像坐标位置」：产品识别不准时也能按手绘坐标精确落点，
        # 比「限定在产品轮廓内」更少受分割误差影响。
        self.cmb_posmode.setCurrentIndex(self.cmb_posmode.findData(POS_MODE_PIXEL))
        self.cmb_posmode.setFixedWidth(190)
        self.cmb_posmode.setToolTip(
            "① 限定在产品轮廓内：先自动识别产品位置，缺陷只能生成在\n"
            "「产品表面 ∩ 手绘区域」内（识别不准时缺陷位置也会受影响）。\n"
            "② 按图像坐标位置：不依赖产品识别，手绘区域的像素坐标\n"
            "直接作为缺陷生成范围，产品识别对错都不影响落点。")
        self.cmb_posmode.currentIndexChanged.connect(self._on_posmode_changed)
        bar.addWidget(self.cmb_posmode)

        bar.addSpacing(10)

        # ---- 轮廓显示 ----
        self.cb_show_outline = QCheckBox("显示产品轮廓")
        self.cb_show_outline.setChecked(True)
        self.cb_show_outline.setToolTip("在预览图上用明显的线条描出识别到的产品轮廓")
        self.cb_show_outline.toggled.connect(lambda _v: self.overlayChanged.emit())
        bar.addWidget(self.cb_show_outline)

        self.lbl_outline_color = QLabel("轮廓色")
        bar.addWidget(self.lbl_outline_color)
        self.btn_outline_color = QPushButton("■")
        self.btn_outline_color.setFixedSize(46, 24)
        self.btn_outline_color.setToolTip("点击选择产品轮廓线的颜色")
        self._outline_color = (255, 80, 80)
        self.btn_outline_color.clicked.connect(self._pick_outline_color)
        bar.addWidget(self.btn_outline_color)

        self.lbl_outline_w = QLabel("线宽")
        bar.addWidget(self.lbl_outline_w)
        self.sp_outline_w = QSpinBox()
        self.sp_outline_w.setRange(1, 8)
        self.sp_outline_w.setValue(2)
        self.sp_outline_w.setSuffix(" px")
        self.sp_outline_w.setFixedWidth(70)
        self.sp_outline_w.setToolTip("产品轮廓线的粗细（像素）")
        self.sp_outline_w.valueChanged.connect(lambda _v: self.overlayChanged.emit())
        bar.addWidget(self.sp_outline_w)

        self.cb_show_drawarea = QCheckBox("显示限定范围")
        self.cb_show_drawarea.setChecked(True)
        self.cb_show_drawarea.setToolTip("用黄色虚线标出手绘限定范围，缺陷只能落在这个范围内")
        self.cb_show_drawarea.toggled.connect(lambda _v: self.overlayChanged.emit())
        bar.addWidget(self.cb_show_drawarea)

        bar.addStretch(1)
        return bar

    def _pick_outline_color(self):
        # ⚠️ QColorDialog 在 QtWidgets 里（不在 QtGui）。
        #    之前误写成 `from PyQt6.QtGui import QColorDialog`，点击按钮时抛
        #    ImportError —— 在 PyInstaller 打包的 exe 里该异常从 Qt 槽函数抛出
        #    会直接导致进程 abort，表现为「点一下轮廓色就闪退」。
        c = QColorDialog.getColor(
            QColor(*self._outline_color), self, "选择产品轮廓颜色")
        if c.isValid():
            self._outline_color = (c.red(), c.green(), c.blue())
            self.btn_outline_color.setStyleSheet(
                f"background:rgb({c.red()},{c.green()},{c.blue()});")
            self.overlayChanged.emit()

    def pos_mode(self) -> str:
        """当前缺陷位置方式（POS_MODE_PRODUCT / POS_MODE_PIXEL）。"""
        return self.cmb_posmode.currentData()

    def outline_color(self):
        return self._outline_color

    def outline_width(self) -> int:
        return int(self.sp_outline_w.value())

    def show_product_outline(self) -> bool:
        return self.cb_show_outline.isChecked()

    def show_draw_area(self) -> bool:
        return self.cb_show_drawarea.isChecked()

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
                                 "其它缺陷在手绘的矩形·画笔内）")
        self.cb_draw.toggled.connect(self._on_draw_toggle)
        bar.addWidget(self.cb_draw)

        # 绘制方式下拉：多边形已按需求删除；
        # 用 QStandardItemModel 以便按启用的缺陷类型禁用个别项
        # （划痕启用 → 只能画线条；其它类型启用 → 只能画矩形/画笔）
        self.cmb_mode = QComboBox()
        _m = QStandardItemModel(self.cmb_mode)
        for _txt, _data in (("浏览", ImageView.MODE_VIEW),
                            ("画线条（划痕）", ImageView.MODE_LINE),
                            ("画矩形区域", ImageView.MODE_RECT),
                            ("画笔涂抹", ImageView.MODE_BRUSH)):
            _it = QStandardItem(_txt)
            _it.setData(_data, Qt.ItemDataRole.UserRole)
            _m.appendRow(_it)
        self.cmb_mode.setModel(_m)
        self.cmb_mode.setFixedWidth(150)
        self.cmb_mode.setToolTip("线条：划痕专用（可拖动或逐点点击，起止点即划痕起止点）\n"
                                 "矩形/画笔：其它缺陷的区域限定\n"
                                 "可用方式随启用的缺陷类型自动限制：\n"
                                 "　· 启用划痕 → 只能画线条\n"
                                 "　· 启用其它类型 → 只能画矩形 / 画笔")
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

        # ROI 角度（紧跟「清除手绘」右侧）—— 与手绘操作同属一组，放在一起更顺手
        self._add_angle_widgets(bar)

        bar.addStretch(1)

        # 结束当前线条/多边形（点回起点、双击、右键、回车之外再给一个按钮）
        b_fin = QPushButton("结束当前线条")
        b_fin.setToolTip("把正在画的线条/多边形提交（等同双击或回车）")
        b_fin.clicked.connect(self.finish_pending)
        self.btn_finish = b_fin
        bar.addWidget(b_fin)
        return bar

    def _add_angle_widgets(self, bar: QHBoxLayout):
        """把「ROI角度 + ±90°」加入指定行（只在矩形模式显示）。"""
        self.lbl_angle = QLabel("ROI角度")
        self.lbl_angle.setToolTip("旋转矩形 ROI 的角度（度）。"
                                   "也可以在图上按住 Ctrl 左键拖动来旋转")
        bar.addWidget(self.lbl_angle)
        self.sp_angle = QSpinBox()
        self.sp_angle.setRange(-180, 180)
        self.sp_angle.setValue(0)
        self.sp_angle.setSuffix("°")
        self.sp_angle.setFixedWidth(112)
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
        # 存为实例属性：非矩形模式下要随 ROI角度 一起隐藏
        self._rot_l = b_rot_l
        self._rot_r = b_rot_r

    def _bar_zoom(self) -> QHBoxLayout:
        """缩放工具行 + 缺陷位置像素级微调按钮。"""
        bar = QHBoxLayout()
        bar.setSpacing(6)

        b_out = QPushButton("－")
        b_out.setFixedWidth(40)
        b_out.setToolTip("缩小")
        b_out.clicked.connect(lambda: self.zoom_all(1 / 1.25))
        bar.addWidget(b_out)
        b_1to1 = QPushButton("100%显示")
        b_1to1.setFixedWidth(88)
        b_1to1.setToolTip("按 1 图像像素 = 1 屏幕像素显示（100%）")
        b_1to1.clicked.connect(self.zoom_1to1)
        bar.addWidget(b_1to1)
        b_fit = QPushButton("适应窗口显示")
        b_fit.setFixedWidth(116)
        b_fit.setToolTip("三栏都缩放到刚好放下整图")
        b_fit.clicked.connect(self.fit_all)
        bar.addWidget(b_fit)
        b_in = QPushButton("＋")
        b_in.setFixedWidth(40)
        b_in.setToolTip("放大")
        b_in.clicked.connect(lambda: self.zoom_all(1.25))
        bar.addWidget(b_in)

        # ---- 缺陷位置微调（像素级，单击一次移动 1px，按住不放持续移动） ----
        bar.addSpacing(10)
        t_mv = QLabel("缺陷位置")
        t_mv.setProperty("role", "title")
        t_mv.setToolTip("把最近一次生成的缺陷整体平移（单位：像素 px）。\n"
                        "单击一次移动 1px；按住不松则连续移动，松开停止。\n"
                        "偏移会同步应用到「保存缺陷图」。")
        bar.addWidget(t_mv)
        self.btn_move = {}
        _dirs = (("←", "左", -1, 0), ("→", "右", 1, 0),
                 ("↑", "上", 0, -1), ("↓", "下", 0, 1))
        for label, cn, dx, dy in _dirs:
            b = QPushButton(label)
            b.setFixedWidth(40)
            b.setToolTip(f"缺陷位置{cn}移（每次 1px；按住不放持续移动，松开停止）")
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(350)      # 按住 350ms 后开始连续移动
            b.setAutoRepeatInterval(30)    # 连续移动间隔 30ms（约 33px/秒）
            b.clicked.connect(lambda _=False, d=dx, e=dy: self.defectMove.emit(d, e))
            self.btn_move[(dx, dy)] = b
            bar.addWidget(b)
        bar.addStretch(1)     # 行尾弹性：让微调按钮组紧跟「＋」而不是被撑到行尾
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
        # set_image 会清掉叠加层，这里把最近一次的轮廓重新贴回去，
        # 避免每次刷新预览后产品轮廓就消失了
        self._reapply_overlay()

    def set_source_image(self, arr):
        """只更新最左栏（原图 / 手绘区域），保留三栏缩放与叠加层设置。"""
        self.views[0].set_image(arr)
        self._reapply_overlay()

    def set_overlay(self, rgba):
        """把产品轮廓 / 限定范围叠加层画到最左栏（手绘那栏）。"""
        self._overlay_arr = rgba
        self.views[0].set_overlay(rgba)

    def _reapply_overlay(self):
        if getattr(self, "_overlay_arr", None) is not None:
            self.views[0].set_overlay(self._overlay_arr)

    def _on_posmode_changed(self, _idx: int):
        self.posModeChanged.emit(self.pos_mode())
        self.overlayChanged.emit()

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

    # ---------------- 模式可用性（随启用的缺陷类型联动） ----------------
    def _mode_item(self, idx: int):
        m = self.cmb_mode.model()
        return m.item(idx) if hasattr(m, "item") else None

    @staticmethod
    def allowed_modes(scratch_on: bool, any_on: bool) -> set:
        """按启用的缺陷类型计算可用绘制方式。

        · 划痕启用 → 只能画线条（其它方式不可选）
        · 其它类型启用 → 只能画矩形 / 画笔
        · 没有启用任何类型 → 线条 / 矩形 / 画笔都可用
        """
        allowed = {ImageView.MODE_VIEW}
        if not any_on:
            allowed |= {ImageView.MODE_LINE, ImageView.MODE_RECT,
                        ImageView.MODE_BRUSH}
        elif scratch_on:
            allowed |= {ImageView.MODE_LINE}
        else:
            allowed |= {ImageView.MODE_RECT, ImageView.MODE_BRUSH}
        return allowed

    def set_mode_availability(self, scratch_on: bool, any_on: bool):
        """禁用不可用的绘制方式，并把当前模式切到最近的可用项。"""
        allowed = self.allowed_modes(scratch_on, any_on)
        for i in range(self.cmb_mode.count()):
            it = self._mode_item(i)
            if it is not None:
                it.setEnabled(self.cmb_mode.itemData(i) in allowed)
        self.ensure_valid_mode()

    def _mode_allowed_now(self, mode: str) -> bool:
        m = self.cmb_mode.model()
        for i in range(self.cmb_mode.count()):
            if self.cmb_mode.itemData(i) == mode:
                it = m.item(i) if hasattr(m, "item") else None
                return bool(it is None or it.isEnabled())
        return False

    def ensure_valid_mode(self):
        """当前绘制方式被禁用时，自动切到第一个可用的绘制方式。"""
        if not self.cb_draw.isChecked():
            return
        if self._mode_allowed_now(self.cmb_mode.currentData()):
            return
        for i in range(self.cmb_mode.count()):
            data = self.cmb_mode.itemData(i)
            if data != ImageView.MODE_VIEW and self._mode_allowed_now(data):
                self.cmb_mode.setCurrentIndex(i)
                return

    # ---------------- 模式 ----------------
    def _on_mode(self, _idx: int):
        # 切换模式前先提交上一个模式未完成的线条，避免被丢弃
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
            self.ensure_valid_mode()
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
        for w in (self.lbl_angle, self.sp_angle,
                  getattr(self, "_rot_l", None), getattr(self, "_rot_r", None)):
            if w is not None:
                w.setVisible(rect)
        # 线条模式下才需要「结束当前线条」（多边形方式已删除）
        self.btn_finish.setVisible(mode == ImageView.MODE_LINE)
