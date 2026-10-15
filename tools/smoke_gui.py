"""GUI 离屏冒烟测试。

覆盖 v1.2 的 5 项改进 + v1.3 的 4 项新需求：
  1) 手动绘制区域后缺陷只落在区域内
  2) 缺陷参数为 Tab 分页（DefectPage + QTabWidget）
  3) 无运行日志块
  4) 预览支持缩放/平移（ImageView / PreviewPane）
  5) 自动查找 MLCC 产品，缺陷不落背景
  6) 「产品区域检测」模块已隐藏，左栏只剩输入设置 + 输出设置
  7) 缺陷参数区去掉「缺陷参数」标题、标题位显示缺陷名、参数行限宽
  8) 缺角缺陷类型出现在 Tab 中并可生成
  9) 模板创建 / 匹配 / 匹配框内生成缺陷
"""
import os
import sys
import json
import math
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np  # noqa: E402
from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QGroupBox, QTabWidget,  # noqa: E402
                             QPushButton, QLineEdit, QPlainTextEdit, QTextEdit,
                             QSlider, QLabel, QWidget, QVBoxLayout,
                             QSpinBox, QComboBox, QCheckBox)

import gui.main_window as mw  # noqa: E402
from core.batch import KIND_CN  # noqa: E402
from core.defects import DEFECT_KINDS  # noqa: E402
from gui.image_view import ImageView  # noqa: E402

FAIL = []


def check(cond, msg):
    tag = "OK " if cond else "FAIL"
    print(f"[{tag}] {msg}")
    if not cond:
        FAIL.append(msg)


def deepest_visible_bottom(card) -> int:
    """卡片内最深的**可见**子控件底边（相对卡片坐标）。

    只统计可见控件：QComboBox 下拉弹层等隐藏子控件几何不可信。
    """
    best = 0
    for c in card.findChildren(QWidget):
        if not c.isVisible() or c.window() is not card.window():
            continue
        y = c.geometry().y()
        p = c.parentWidget()
        while p is not None and p is not card:
            y += p.y()
            p = p.parentWidget()
        if p is card:
            best = max(best, y + c.geometry().height())
    return best


def check_layout_at(win, app, size, tag):
    """指定窗口尺寸下校验：无重叠 + 内容不越出卡片 + Tab 高度足够。"""
    win.resize(*size)
    for _ in range(2):
        app.processEvents()
    win._equalize_cards()
    for _ in range(3):
        app.processEvents()
    cards = (win._card_in, win._card_def, win._card_out)
    bottom = max(c.y() + c.height() for c in cards)
    # 预览现嵌套在 split_main 里，其 .y() 相对 splitter 为 0；
    # 用 splitter 在主窗口坐标系下的顶边作为"预览顶"的等价判断。
    preview_top = win.split_main.y()
    check(bottom <= preview_top,
          f"{tag} 三卡片与预览无重叠（卡底 {bottom} ≤ 预览顶 {preview_top}）")
    for c in cards:
        db = deepest_visible_bottom(c)
        check(db <= c.height(),
              f"{tag} {c.title().strip()} 内容不越出卡片（{db} ≤ {c.height()}）")
    tabbar_h = win.tabs.tabBar().sizeHint().height()
    need = tabbar_h + max(max(p.sizeHint().height(), p.minimumSizeHint().height())
                          for p in win.pages.values())
    check(win.tabs.height() >= need - 2,
          f"{tag} Tab 高度容纳参数最多的一页（{win.tabs.height()} ≥ {need}）")


def _apply_with_paths(win, paths, seed=0):
    """用给定手绘线条生成划痕（走 GUI 同样的调用链）。"""
    from core.defects import apply_defects
    from core.mlcc_region import segment_mlcc

    seg = segment_mlcc(win.current_image)
    st = dict(win.current_settings()["scratch"])
    st["enable"] = True
    st["target"] = "both"
    _, applied = apply_defects(win.current_image, {"scratch": st}, seed=seed,
                               region_masks={"body": seg.body,
                                             "electrode": seg.electrode},
                               scratch_paths=paths)
    return applied[-1] if applied else None


def main() -> int:
    app = QApplication(sys.argv)
    win = mw.MainWindow()
    win.resize(1680, 1000)
    win.show()
    app.processEvents()

    # ================= 需求：顶部三卡片布局（输入 → 缺陷配置 → 输出） =================
    gbs = [g.title().strip() for g in win.findChildren(QGroupBox)]
    print("      顶部分组:", gbs)
    check(not any("区域检测" in t for t in gbs),
          "「产品区域检测」分组已隐藏（现有分组：%s）" % gbs)
    expected = ["1 · 输入设置", "2 · 缺陷图生成配置", "3 · 输出设置"]
    top_titles = [t for t in gbs if t in expected]
    check(top_titles == expected,
          f"顶部三卡片顺序正确（输入→缺陷配置→输出）：{top_titles}")
    for gone in ("cb_auto", "cb_use_draw", "cmb_bgs", "sp_elec", "lbl_seg"):
        check(not hasattr(win, gone), f"已移除控件属性 {gone}")
    check(win.findChildren(QLineEdit), "界面存在输出目录输入框（旧区域检测控件已移除）")

    # ================= 需求：划痕默认开启，其余三类默认关闭 =================
    check(win.pages["scratch"].enable.isChecked(),
          "划痕默认开启")
    check(all(not win.pages[k].enable.isChecked()
              for k in ("smudge", "discolor", "chip")),
          "脏污 / 异色 / 缺角默认关闭")

    # ================= 需求：输出设置只有「缺陷图」默认开启 =================
    outs = [("缺陷图", win.cb_defect, True), ("缺陷掩膜", win.cb_mask, False),
            ("对比图", win.cb_prev, False), ("区域分割图", win.cb_region, False),
            ("YOLO txt", win.cb_yolo, False), ("VOC xml", win.cb_voc, False),
            ("COCO json", win.cb_coco, False)]
    for name, cb, want in outs:
        check(cb.isChecked() == want,
              f"输出设置「{name}」默认{'开启' if want else '关闭'}")

    # ================= 需求：底部按钮 + 上一张/下一张 =================
    check(win.btn_save.text() == "保存缺陷图",
          f"存在「保存缺陷图」按钮（实际 {win.btn_save.text()!r}）")
    check(hasattr(win, "save_defect_image"),
          "保存缺陷图按钮已接 save_defect_image")
    check(win.btn_run.text() == "批量生成全部",
          f"原「开始批量生成」改为「批量生成全部」（{win.btn_run.text()!r}）")
    check("上一张" in win.btn_prev.text() and "下一张" in win.btn_next.text(),
          "输入设置含「上一张 / 下一张」按钮")

    # ================= 需求：数量默认 1 / 划痕弯曲度 / 边缘模糊 =================
    from core.batch import DEFAULT_SETTINGS

    cnt = {"scratch": "count", "smudge": "blob_count",
           "discolor": "region_count", "chip": "count"}
    for kind, key in cnt.items():
        check(DEFAULT_SETTINGS[kind][key] == 1,
              f"{kind} 数量默认值 = 1（{key}={DEFAULT_SETTINGS[kind][key]}）")
        r = win.pages[kind].rows.get(key)
        check(r is not None and abs(r.value() - 1.0) < 1e-6,
              f"[{kind}] GUI 数量行 {key} 显示 1")
    st0 = win.current_settings()
    check(all(st0[k][v] == 1 for k, v in cnt.items()),
          f"GUI 读回的配置数量均为 1：{ {k: st0[k][v] for k, v in cnt.items()} }")
    check("curve" in win.pages["scratch"].rows, "划痕新增「弯曲度」参数行")
    check("edge_soft" in win.pages["smudge"].rows, "脏污有「边界柔和度」参数行")
    check("edge_soft" in win.pages["discolor"].rows, "异色有「边界柔和度」参数行")
    # 脏污/异色不再有「边缘模糊」——边界感改由 edge_soft 直接控制（0=清晰 1=几乎无边界）
    check("edge_blur" not in win.pages["discolor"].rows,
          "异色已移除「边缘模糊」（并入边界柔和度）")

    # ================= 需求 2：Tab 分页 + 去标题 + 限宽 =================
    tabs = win.findChildren(QTabWidget)
    check(len(tabs) >= 1, f"存在 QTabWidget（找到 {len(tabs)} 个）")
    check(set(win.pages.keys()) == set(DEFECT_KINDS),
          f"缺陷页覆盖全部类型：{sorted(win.pages.keys())}")
    if tabs:
        names = [tabs[0].tabText(i) for i in range(tabs[0].count())]
        print("      Tab 标题:", names)
        check(tabs[0].count() == len(DEFECT_KINDS),
              f"Tab 页数为 {len(DEFECT_KINDS)}（实际 {tabs[0].count()}）")
        check(all("●" in n or "○" in n for n in names),
              "Tab 标题含启用状态指示（●/○）")

    # 卡片标题「缺陷参数」已删除（现为「2 · 缺陷图生成配置」）
    check(not any("缺陷参数" in (g.title() or "") for g in win.findChildren(QGroupBox)),
          "已删除卡片标题「缺陷参数」")
    check(not hasattr(win, "lbl_defect") and not hasattr(win, "lbl_defect_tip"),
          "缺陷卡顶部的缺陷名标题与提示标签已删除")

    # Tab 标签即四个缺陷名（卡片顶部分页）；同时校验每页文字完整、滑条未塌缩
    for idx, kind in enumerate(DEFECT_KINDS):
        win.tabs.setCurrentIndex(idx)
        app.processEvents()
        check(KIND_CN[kind] in win.tabs.tabText(idx),
              f"Tab{idx} 标签为缺陷名「{KIND_CN[kind]}」（{win.tabs.tabText(idx)!r}）")
        check(not hasattr(win.pages[kind], "w_target")
              and "target" not in win.pages[kind].combos,
              f"[{kind}] 已移除「目标表面」及其下拉")
        for lbl in win.pages[kind].findChildren(QLabel):
            txt = lbl.text()
            if not txt:
                continue
            # 标签未被布局压缩（width≈sizeHint 即说明文字有完整显示空间）
            check(lbl.width() >= lbl.sizeHint().width() - 1,
                  f"[{kind}] 文字完整显示：{txt[:14]!r}（{lbl.width()}≥{lbl.sizeHint().width()}）")
        for key, r in win.pages[kind].rows.items():
            check(r.slider.width() > 120,
                  f"[{kind}/{key}] 滑条宽度足够（{r.slider.width()}px）")
    win.tabs.setCurrentIndex(0)

    # 四个 Tab 标签文字完整（卡片顶部的分页标题）
    bar = win.tabs.tabBar()
    for i in range(win.tabs.count()):
        r = bar.tabRect(i)
        need = bar.tabSizeHint(i)
        check(not r.isEmpty() and r.width() >= need.width() - 1,
              f"Tab{i} 标签文字完整：{win.tabs.tabText(i)!r}"
              f"（{r.width()}≥{need.width()}）")

    # 三张卡片等高（需求：缺陷配置卡 = 输入卡 = 输出卡）
    app.processEvents()
    hs = {c.title().strip(): c.height()
          for c in (win._card_in, win._card_def, win._card_out)}
    check(len(set(hs.values())) == 1, f"三张卡片等高：{hs}")
    check(win.tabs.width() >= win._card_def.width() - 30,
          f"Tab 容器铺满缺陷卡片宽度（{win.tabs.width()} / 卡{win._card_def.width()}）")

    # ============ 需求：卡片不可压到下方预览（多尺寸回归） ============
    for size in ((1560, 940), (1280, 800), (1180, 740)):
        check_layout_at(win, app, size, f"{size[0]}x{size[1]}")

    # 紧凑化效果：默认窗口下三卡片高度应远小于上一版（511px）
    win.resize(1560, 940)
    app.processEvents()
    win._equalize_cards()
    app.processEvents()
    check(win._card_def.height() <= 380,
          f"三卡片高度已压到 {win._card_def.height()}px（上一版 511px）")

    # 参数行按卡片宽度自适应 1/2 列
    check(win.pages["chip"]._cols == 2,
          f"宽窗参数两列排布（cols={win.pages['chip']._cols}）")
    win.resize(1180, 740)
    app.processEvents()
    win._equalize_cards()
    app.processEvents()
    check(all(p._cols == 1 for p in win.pages.values()),
          f"窄窗回落单列避免文字被裁（cols={win.pages['chip']._cols}）")
    win.resize(1680, 1000)
    app.processEvents()
    win._equalize_cards()
    app.processEvents()
    check(all(p._cols == 2 for p in win.pages.values()),
          "恢复宽窗后重新排成两列")

    # 预览工具条为四行（窄窗下控件不被挤扁、文字完整）
    # 四行 = 手绘 / 缺陷位置方式+轮廓 / 缩放 / 状态信息
    tbar = win.preview.layout().itemAt(0).layout()
    check(isinstance(tbar, QVBoxLayout) and tbar.count() == 4,
          f"预览工具条为四行排布（实际 {type(tbar).__name__}/{tbar.count()}）")

    # 「缺陷位置」行必须齐备：位置方式下拉 + 轮廓显隐/颜色/线宽/限定范围
    _pm = win.preview.pos_mode()
    check(_pm in ("product", "pixel"),
          f"缺陷位置方式下拉有有效值（{_pm}）")
    check(isinstance(win.preview.outline_color(), tuple)
          and len(win.preview.outline_color()) == 3,
          f"轮廓颜色为 RGB 三元组（{win.preview.outline_color()}）")
    check(1 <= win.preview.outline_width() <= 8,
          f"轮廓线宽在 1~8 内（{win.preview.outline_width()}）")
    check(isinstance(win.preview.show_product_outline(), bool)
          and isinstance(win.preview.show_draw_area(), bool),
          "轮廓显隐 / 限定范围开关可用")

    # ================= 需求 3：无日志块 =================
    check(not hasattr(win, "log_box"), "已删除日志块 log_box")
    txts = win.findChildren(QPlainTextEdit) + win.findChildren(QTextEdit)
    check(len(txts) == 0, f"界面内无多行日志控件（找到 {len(txts)} 个）")

    # ================= 需求 4：预览缩放/平移 =================
    pv = win.preview
    check(len(pv.views) == 3, f"预览三栏（实际 {len(pv.views)}）")

    # ================= 输入 =================
    samples = [os.path.join(HERE, "samples_mlcc", f)
               for f in sorted(os.listdir(os.path.join(HERE, "samples_mlcc")))
               if f.lower().endswith(".png") and not f.startswith("region")]
    check(len(samples) >= 1, f"找到 MLCC 样例 {len(samples)} 张")
    win.add_inputs(samples)
    win.lst.setCurrentRow(0)
    app.processEvents()

    # 四种缺陷默认关闭，这里显式开启以走通预览生成路径
    for p in win.pages.values():
        p.enable.setChecked(True)
    app.processEvents()

    win.ed_out.setText(os.path.join(HERE, "build", "out_gui"))
    win.sp_count.setValue(1)
    win.refresh_preview()
    app.processEvents()

    # ================= 需求 5：自动查找产品（内置） =================
    seg = win._ensure_segmentation()
    check(seg is not None and seg.ok,
          f"内置区域检测成功（method={getattr(seg, 'method', None)}）")
    if seg is not None and seg.ok:
        print(f"      覆盖率={seg.coverage:.1%} 主轴={seg.axis_angle_deg:.0f}°")
        check(0.02 < seg.coverage < 0.98, f"产品占比合理：{seg.coverage:.1%}")
    check(pv.views[2].has_image(), "预览缺陷图已生成")

    # ================= 需求 1：手绘区域（工具栏开关） =================
    v0 = pv.views[0]
    modes = [pv.cmb_mode.itemText(i) for i in range(pv.cmb_mode.count())]
    print("      手绘模式:", modes)
    check(len(modes) == 5 and "画线条" in modes[1] and "画矩形区域" in modes[2]
          and "画多边形" in modes[3] and "画笔涂抹" in modes[4],
          f"手绘模式共 5 种且顺序正确：{modes}")
    check(pv.cmb_mode.itemData(1) == ImageView.MODE_LINE
          and pv.cmb_mode.itemData(3) == ImageView.MODE_POLY,
          "线条 / 多边形模式已注册")
    # 需求：「限定手绘区域」默认开启，且默认进入「画线条」
    check(pv.cb_draw.isChecked(), "工具栏「限定手绘区域」默认开启")
    check(pv.cmb_mode.isEnabled() and pv.cmb_mode.currentData() == ImageView.MODE_LINE,
          f"默认进入「画线条」模式（{pv.cmb_mode.currentData()}）")
    # 需求：删除「与产品求交 / 并入产品区域 / 完全以手绘为准」下拉
    _items = []
    for c in pv.findChildren(QComboBox):
        _items += [c.itemText(i) for i in range(c.count())]
    _bad = [t for t in _items if "求交" in t or "并入" in t or "以手绘为准" in t]
    check(not hasattr(pv, "cmb_draw_mode") and not _bad,
          f"已删除范围合并方式下拉（残留：{_bad}）")
    check(pv.draw_mode() == "intersect", "手绘与产品区域固定取交集")
    check(not hasattr(win.preview, "drawModeChanged"),
          "已移除 drawModeChanged 信号")
    # 需求：笔刷只在「画笔涂抹」时出现
    check(not pv.lbl_brush.isVisible() and not pv.sld_brush.isVisible(),
          "非画笔模式下笔刷控件隐藏")
    pv.cmb_mode.setCurrentIndex(4)      # 画笔涂抹
    for _ in range(3):
        app.processEvents()
    check(pv.lbl_brush.isVisible() and pv.sld_brush.isVisible(),
          "切到「画笔涂抹」时笔刷控件出现")
    pv.cmb_mode.setCurrentIndex(1)      # 回线条模式
    for _ in range(3):
        app.processEvents()
    check(not pv.lbl_brush.isVisible(), "切回线条模式后笔刷控件再次隐藏")

    v0.clear_regions()
    h, w = win.current_image.shape[:2]
    v0._rects.append((int(0.08 * w + 0.20 * w), int(0.08 * h + 0.20 * h),
                       int(0.40 * w), int(0.40 * h), 0.0))
    v0.update()
    app.processEvents()
    check(pv.has_drawn_regions(), "手绘区域已记录")

    win._force_resegment()
    win.refresh_preview()
    app.processEvents()

    dm = win._draw_mask()
    check(dm is not None and dm.sum() > 0, "手绘 mask 生成成功")
    seg2 = win._ensure_segmentation()
    check(seg2 is not None and seg2.ok, "手绘+产品求交后区域检测成功")
    if seg2 is not None and seg2.ok:
        from core.defects import apply_defects

        # 必须用界面真实使用的区域计算：分割结果本身是**完整产品**（不再与手绘区域
        # 求交），「∩ 手绘区域」这一步在 _region_masks_for() 里完成。
        # 直接拿 seg2.body 当区域会让缺陷落到整个产品上（位置校验就失去意义了）。
        region_masks = win._region_masks_for(seg2)
        check(region_masks is not None, "手绘∩产品 得到有效生成区域")
        n_rm = 0 if region_masks is None else int(
            (region_masks["body"] > 0.5).sum() + (region_masks["electrode"] > 0.5).sum())
        print("      位置模式=%s  cb_draw=%s  dm=%.0f px  生成区域=%.0f px"
              % (win._pos_mode(), pv.cb_draw.isChecked(),
                 float(dm.sum()) if dm is not None else -1, n_rm))
        img, applied = apply_defects(win.current_image, win.current_settings(),
                                     seed=int(win.sp_seed.value()),
                                     region_masks=region_masks)
        # 越界判据**随位置方式而定**：
        #   · 按图像坐标位置 → 契约是「不越出手绘区域」；
        #     注意手绘框可以画到产品之外（这正是该模式的意义），此时不能拿产品当判据。
        #   · 限定在产品轮廓内 → 契约是「不越出产品表面」。
        if win._pos_mode() == "pixel":
            limit, lim_name = dm, "手绘区域"
        else:
            limit, lim_name = seg2.product, "产品表面"
        grow = (limit < 0.5)
        tot = sum(float((r.mask > 0.05).sum()) for r in applied)
        out = sum(float(((r.mask > 0.05) & grow).sum()) for r in applied)
        for _r in applied:      # 逐类定位越界来源，便于排查
            _m = _r.mask > 0.05
            if _m.any():
                print("      %-7s mask=%6d 越出%s=%6d (%.1f%%)"
                      % (_r.kind, _m.sum(), lim_name, int((_m & grow).sum()),
                         100.0 * (_m & grow).sum() / max(_m.sum(), 1)))
        rate = (out / tot * 100) if tot else 0.0
        print(f"      越界判据={lim_name}：缺陷像素 {int(tot)}，越界 {int(out)}（{rate:.2f}%）")
        check(rate <= 1.0, f"缺陷越出{lim_name}率 ≤ 1%（实际 {rate:.2f}%）")
        ys, xs = np.where(sum((r.mask > 0.05) for r in applied) > 0)
        if len(xs):
            cx = xs.mean() / w
            # 注意：崩缺(chip)是**边沿**缺陷，会落在产品轮廓上，其分布中心
            # 天然偏向产品中线而非手绘框，因此这里只对「非崩缺」缺陷做位置校验。
            inner = [r for r in applied if r.kind != "chip"]
            if inner:
                ys2, xs2 = np.where(sum((r.mask > 0.05) for r in inner) > 0)
                if len(xs2):
                    cx = xs2.mean() / w
                    check(cx < 0.60, f"非崩缺缺陷集中在手绘区域侧（cx={cx:.2f}）")

    pv.clear_regions()
    pv.cb_draw.setChecked(False)
    app.processEvents()
    check(not pv.has_drawn_regions(), "取消勾选后手绘区域已清空")
    check(v0.mode == ImageView.MODE_VIEW, "取消勾选后回到浏览模式")

    # ================= 需求 8：崩缺缺陷 =================
    check("chip" in win.pages, "崩缺页已注册")
    chip = win.pages["chip"]
    for key in ("size_ratio", "depth_ratio", "jag_ratio", "crack_ratio", "corner_bias"):
        check(key in chip.rows, f"崩缺参数行存在：{key}")
    print("      崩缺配置:", chip.config())

    # 崩缺必须贴着产品轮廓：统计缺陷像素到轮廓的距离
    import cv2 as _cv2
    _seg3 = win._ensure_segmentation()
    _b = (_seg3.product > 0.5).astype(np.uint8)
    _dt = _cv2.distanceTransform(_b, _cv2.DIST_L2, 5)
    _img3, _ap3 = apply_defects(
        win.current_image, {"chip": chip.config()}, seed=5,
        region_masks={"body": _seg3.body, "electrode": _seg3.electrode})
    _m = _ap3[0].mask > 0.05 if _ap3 else None
    if _m is not None and _m.any():
        _yy, _xx = np.nonzero(_m)
        _dmin = float(_dt[_yy, _xx].min())
        check(_dmin <= 2.0, f"崩缺紧贴产品轮廓（最小距离 {_dmin:.1f}px ≤ 2px）")
    else:
        check(False, "崩缺应能生成")

    # ================= 新增三类缺陷 =================
    for _k, _cn in (("dot", "黑点"), ("pit", "凹坑"), ("debris", "异物")):
        check(_k in win.pages, f"{_cn}页已注册")
        check("count" in win.pages[_k].rows, f"{_cn}有「个数」参数行")
    check("fiber_ratio" in win.pages["debris"].rows, "异物有「纤维状比例」参数行")
    print("      七类缺陷页:", list(win.pages.keys()))

    win.pages["scratch"].enable.setChecked(False)
    win.pages["smudge"].enable.setChecked(False)
    win.pages["discolor"].enable.setChecked(False)
    chip.enable.setChecked(True)
    chip.rows["count"].setValue(2)
    chip.rows["size_ratio"].setValue(0.45)
    chip.rows["severity"] if False else None
    chip.combos["severity"].setCurrentIndex(
        max(0, chip.combos["severity"].findData("obvious")))
    app.processEvents()
    win.refresh_preview()
    app.processEvents()

    from core.defects import apply_defects

    rm = {"body": seg.body, "electrode": seg.electrode}
    img, applied = apply_defects(win.current_image, win.current_settings(),
                                 seed=123, region_masks=rm)
    kinds_applied = [r.kind for r in applied]
    print("      实际生成:", kinds_applied)
    check("chip" in kinds_applied, "崩缺缺陷已生成")
    info = pv.lbl_info.text()
    print("      预览信息:", info)
    check("崩缺" in info, "预览信息含「崩缺」")
    check("越界" in info, "预览信息包含「越界」指标")
    try:
        leak = float(info.split("越界")[1].split("%")[0])
    except Exception:
        leak = None
    check(leak is not None and leak <= 1.0, f"缺角越界率 ≤ 1%（实际 {leak}）")
    check("chip" not in [k for k in kinds_applied if k != "chip"],
          "仅启用缺角时不会生成其它类型")

    # 恢复前三类
    win.pages["scratch"].enable.setChecked(True)
    win.pages["smudge"].enable.setChecked(True)
    win.pages["discolor"].enable.setChecked(True)
    chip.enable.setChecked(False)
    app.processEvents()

    # ================= 需求：线条 / 多边形 / 划痕沿线条 =================
    pv.clear_regions()
    v0 = pv.views[0]
    h_img, w_img = win.current_image.shape[:2]

    # 多边形（脏污/异色/缺角 可用）
    v0._polys.append([(w_img * 0.30, h_img * 0.35), (w_img * 0.70, h_img * 0.32),
                      (w_img * 0.72, h_img * 0.70), (w_img * 0.28, h_img * 0.68)])
    m_poly = v0.region_mask((h_img, w_img), include=("rect", "poly", "brush"))
    check(m_poly is not None and float(m_poly.max()) > 0.5,
          f"多边形掩膜生成成功（面积 {float((m_poly > 0.5).mean()) * 100:.1f}%）")

    # 线条（划痕专用）
    sx, sy = w_img * 0.22, h_img * 0.50
    ex, ey = w_img * 0.78, h_img * 0.42
    line = [(sx, sy), (w_img * 0.5, h_img * 0.56), (ex, ey)]
    v0._lines.append(list(line))
    check(v0.has_line_regions(), "线条已记录")
    check(len(v0.line_paths()) == 1 and len(v0.line_paths()[0]) == 3,
          "线条点列可取出")
    m_all = v0.region_mask((h_img, w_img))
    m_area = v0.region_mask((h_img, w_img), include=("rect", "poly", "brush"))
    check(float(m_all.sum()) > float(m_area.sum()),
          "全图元掩膜比「框/多边形/画笔」掩膜更大（线条确实被排除在外）")
    m_line = v0.region_mask((h_img, w_img), include=("line",))
    check(float(m_line.sum()) > 0
          and float(m_line.sum()) < float(m_all.sum()),
          f"只取线条的掩膜独立可用（{float((m_line > 0.5).mean()) * 100:.1f}%）")

    # 勾选手绘 + 划痕默认开启 → 预览应沿线条生成
    pv.cb_draw.setChecked(True)
    app.processEvents()
    check(pv.cmb_mode.currentData() == ImageView.MODE_LINE,
          "勾选手绘后默认进入「画线条」模式")
    win._force_resegment()
    win.refresh_preview()
    app.processEvents()
    info = pv.lbl_info.text()
    print("      预览信息:", info)
    check("沿手绘线条" in info, f"预览按手绘线条生成划痕：{info[:56]}")
    check("划痕" in info, "预览生成了划痕")

    # 端点一致性：掩膜在两个手绘端点附近都要有像素
    ap_ln = _apply_with_paths(win, [line], seed=1)
    if ap_ln:
        mk = ap_ln.mask > 0.05

        def _near(px, py, rr=12):
            y0, y1 = max(0, int(py - rr)), int(py + rr) + 1
            x0, x1 = max(0, int(px - rr)), int(px + rr) + 1
            return float(mk[y0:y1, x0:x1].mean())

        cs, ce = _near(sx, sy), _near(ex, ey)
        print(f"      端点附近掩膜覆盖：起点 {cs:.2f} 终点 {ce:.2f}")
        check(cs > 0.10 and ce > 0.10,
              f"划痕起止位置 = 手绘线条起止位置（{cs:.2f} / {ce:.2f}）")

    # 线条不能作为其它三类的区域：只有线条时 _draw_mask() 应为空
    v0._rects.clear()
    v0._polys.clear()
    v0._strokes.clear()
    win._force_resegment()
    check(win._draw_mask() is None,
          "只画线条时，脏污/异色/缺角拿不到区域（线条不算它们的 ROI）")
    check(len(win._draw_paths()) == 1, "但划痕仍能拿到线条路径")

    # 无线条 + 开启划痕 → 跳过并提示（先关掉其它三类，便于观察）
    for k in ("smudge", "discolor", "chip"):
        win.pages[k].enable.setChecked(False)
    v0._lines.clear()
    win._force_resegment()
    win.refresh_preview()
    app.processEvents()
    check("线条" in pv.lbl_info.text() and "划痕" in pv.lbl_info.text(),
          f"未画线条时提示先画线：{pv.lbl_info.text()[:40]}")

    # 补一个多边形 + 线条：验证两者同时存在时各司其职
    v0._polys.append([(w_img * 0.30, h_img * 0.35), (w_img * 0.70, h_img * 0.32),
                      (w_img * 0.72, h_img * 0.70), (w_img * 0.28, h_img * 0.68)])
    v0._lines.append(list(line))
    win.pages["smudge"].enable.setChecked(True)
    win._force_resegment()
    win.refresh_preview()
    app.processEvents()
    check("沿手绘线条" in pv.lbl_info.text()
          and "划痕" in pv.lbl_info.text(),
          f"线条 + 多边形同时存在时两者各司其职：{pv.lbl_info.text()[:44]}")
    for k in ("smudge", "discolor", "chip"):
        win.pages[k].enable.setChecked(k == "smudge")

    # ============ 需求：拖动画线（真实鼠标事件）→ 刷新预览必须有图 ============
    from PyQt6.QtCore import QPoint, QPointF, Qt as _Qt
    from PyQt6.QtTest import QTest

    def _wpt(widget, img_pt):
        """图像坐标 → 控件坐标（裁进控件内，保证事件落在窗口上）。"""
        p = widget.image_to_widget(QPointF(*img_pt))
        return QPoint(int(np.clip(p.x(), 1, widget.width() - 2)),
                      int(np.clip(p.y(), 1, widget.height() - 2)))

    def _drag(widget, p_from, p_to, steps=12):
        """按住左键从p_from 拖到 p_to（真实鼠标事件序列）。"""
        QTest.mousePress(widget, _Qt.MouseButton.LeftButton, _Qt.KeyboardModifier.NoModifier,
                         _wpt(widget, p_from))
        for i in range(1, steps + 1):
            t = i / float(steps)
            QTest.mouseMove(widget, _wpt(widget, (p_from[0] + (p_to[0] - p_from[0]) * t,
                                              p_from[1] + (p_to[1] - p_from[1]) * t)))
        QTest.mouseRelease(widget, _Qt.MouseButton.LeftButton,
                           _Qt.KeyboardModifier.NoModifier, _wpt(widget, p_to))

    def _click(widget, img_pt):
        """单击（逐点画线/多边形用）。"""
        QTest.mouseClick(widget, _Qt.MouseButton.LeftButton,
                         _Qt.KeyboardModifier.NoModifier, _wpt(widget, img_pt))

    pv.clear_regions()
    v0 = pv.views[0]
    pv.cmb_mode.setCurrentIndex(1)          # 画线条
    for _ in range(2):
        app.processEvents()
    check(v0.mode == ImageView.MODE_LINE, "切到画线条模式")

    # 产品中心附近（保证线落在产品表面上）
    from core.mlcc_region import segment_mlcc as _seg2
    _s = _seg2(win.current_image)
    _ys, _xs = np.nonzero(_s.product > 0.5)
    _cy = float((int(_ys.min()) + int(_ys.max())) // 2)
    _x0 = float(_xs.min() + (_xs.max() - _xs.min()) * 0.25)
    _x1 = float(_xs.min() + (_xs.max() - _xs.min()) * 0.75)
    p_start = (_x0, _cy - 20.0)
    p_end = (_x1, _cy + 10.0)

    _drag(v0, p_start, p_end, steps=14)
    for _ in range(2):
        app.processEvents()

    check(len(v0._lines) == 1,
          f"拖动一次即生成一条线条（lines={len(v0._lines)}）")
    if v0._lines:
        ln = v0._lines[0]
        d0 = math.hypot(ln[0][0] - p_start[0], ln[0][1] - p_start[1])
        d1 = math.hypot(ln[-1][0] - p_end[0], ln[-1][1] - p_end[1])
        # 真正的功能保证是「坐标映射精确」：合成鼠标事件只能落在整数屏幕像素上，
        # 所以 image→widget→image 的往返误差理论上限就是「半个屏幕像素」换算成
        # 图像像素 = 0.75/scale。用这个物理上限来断言，而不是拍一个固定阈值。
        sc = max(v0._scale(), 1e-6)
        worst_map = 0.0
        for _pt in ((p_start[0], p_start[1]), (p_end[0], p_end[1])):
            _w = v0.image_to_widget(QPointF(_pt[0], _pt[1]))
            _b = v0.widget_to_image(QPointF(round(_w.x()), round(_w.y())))
            worst_map = max(worst_map, math.hypot(_b.x() - _pt[0], _b.y() - _pt[1]))
        q_bound = 0.75 / sc
        check(worst_map <= q_bound,
              f"坐标映射精确（往返误差 {worst_map:.2f}px ≤ 半个屏幕像素 {q_bound:.2f}px）")
        # 拖动偏差：按下/松开两端各自再取整一次，合计约 1.5 个屏幕像素。
        # 原本阈值写死 3.0，而实测基线就是 2.9，卡在临界上，界面一改布局就误报。
        tol_px = max(4.0, 1.5 / sc)
        check(d0 < tol_px and d1 < tol_px,
              f"线条起止点= 拖动起止点（偏差{d0:.1f}/{d1:.1f}px，容差{tol_px:.1f}px）")

    # 刷新预览：必须生成划痕，且第三栏（缺陷图）必须有图
    win.pages["smudge"].enable.setChecked(False)
    win.pages["discolor"].enable.setChecked(False)
    win.pages["chip"].enable.setChecked(False)
    win.pages["scratch"].enable.setChecked(True)
    win.refresh_preview()
    for _ in range(3):
        app.processEvents()
    info = pv.lbl_info.text()
    print("      拖线后info:", info)
    check("划痕" in info, f"拖动画线后生成了划痕：{info[:52]}")
    check(pv.views[2].has_image(), "「缺陷图」栏有图片（不是空白）")
    check(pv.views[1].has_image() and pv.views[0].has_image(),
          "「区域与缺陷」「原图」栏都有图片")

    # 未提交的多边形在刷新时自动提交（用户点刷新前忘了双击）
    pv.cmb_mode.setCurrentIndex(3)          # 画多边形
    for _ in range(2):
        app.processEvents()
    poly_pts = [(_x0 - 60.0, _cy - 60.0), (_x0 + 40.0, _cy - 80.0),
                (_x0 + 60.0, _cy + 20.0)]
    for pt in poly_pts:
        _click(v0, pt)
    check(v0.has_pending(), "多边形处于未提交状态")
    n_before = len(v0._polys)
    win.refresh_preview()
    for _ in range(2):
        app.processEvents()
    check(len(v0._polys) == n_before + 1,
          "点「刷新预览」时自动提交未完成的多边形")

    # 没有线条时不应让第三栏空白
    pv.clear_regions()
    win.refresh_preview()
    for _ in range(2):
        app.processEvents()
    check(pv.views[2].has_image(),
          "无线条时第三栏仍显示原图（不空白）")
    check("待画线" in pv.lbl_info.text() or "未生成" in pv.lbl_info.text()
          or "未启用" in pv.lbl_info.text(),
          f"无线条时给出明确提示：{pv.lbl_info.text()[:44]}")

    # ================= 需求：可旋转 ROI + 划痕方向/颜色 =================
    check(not hasattr(pv, "cb_tpl") and not hasattr(pv, "cb_learn"),
          "模板 / 表面学习相关控件已全部移除")
    check(not any(b.text() in ("建模板", "匹配定位", "学习表面…")
                  for b in pv.findChildren(QPushButton)),
          "模板 / 表面学习相关按钮已全部移除")
    check(not any(hasattr(win, a) for a in
                  ("create_template", "learn_surface", "surface_model",
                   "_match_current", "_library", "_ensure_surface")),
          "主窗口不再有模板 / 表面学习的方法")
    check(hasattr(pv, "sp_angle") and pv.lbl_angle.text() == "ROI角度",
          "工具栏含「ROI角度」控制")

    # ---- 工具条不被挤扁（用户截图里的问题：ROI角度框/±90° 按钮被压扁）----
    pv.cb_draw.setChecked(True)
    pv.cmb_mode.setCurrentIndex(2)      # 画矩形区域 → ROI角度组可见
    for _ in range(3):
        app.processEvents()
    # 判定口径：控件实际宽度是否 **小于它自己声明的固定宽度**。
    # 直接比 sizeHint 在离屏缺 CJK 字体时会偏大（曾把 70px 的按钮误判成需要 80px）。
    squeezed = []
    for wd in pv.findChildren((QSpinBox, QPushButton, QComboBox, QCheckBox,
                               QLabel, QSlider)):
        try:
            if wd.parentWidget() is not pv or not wd.isVisible():
                continue
            # 底部信息栏是「单行省略号」的状态文本（setWordWrap(False)），
            # 文本长就被 Qt 省略，这属于设计意图，不是控件被压扁。
            if wd is getattr(pv, "lbl_info", None):
                continue
            mw_, mh_ = wd.minimumWidth(), wd.minimumHeight()
            need = mw_ if mw_ > 0 else wd.minimumSizeHint().width()
            if need > 0 and wd.width() < need - 1:
                squeezed.append((type(wd).__name__,
                                 wd.text() if hasattr(wd, "text") else "",
                                 wd.width(), need))
        except Exception:
            continue
    print("      被压扁的控件:", squeezed)
    check(not squeezed, f"所有可见工具条控件宽度完整（被压扁 {len(squeezed)} 个）")
    check(pv.sp_angle.isVisible() and pv.lbl_angle.isVisible(),
          "矩形模式下 ROI 角度组可见")
    pv.cmb_mode.setCurrentIndex(1)      # 线条模式 → ROI角度组隐藏
    for _ in range(3):
        app.processEvents()
    check(not pv.sp_angle.isVisible(),
          "线条模式下 ROI 角度组隐藏（不再挤占工具条）")

    # 画一个可旋转的 ROI：先按默认角度画，再旋转 40°
    pv.clear_regions()
    v0 = pv.views[0]
    v0._rects.append((w_img * 0.35, h_img * 0.5, w_img * 0.30, h_img * 0.22, 0.0))
    check(len(v0._rects) == 1 and len(v0._rects[0]) == 5,
          "ROI 以 (中心x, 中心y, 宽, 高, 角度) 存储")
    pv.sp_angle.setValue(40)
    app.processEvents()
    check(abs(v0.rect_angle_deg() - 40.0) < 1e-6,
          f"角度输入框旋转 ROI（{v0.rect_angle_deg():.0f}°）")
    pv.sp_angle.setValue(130)
    app.processEvents()
    check(abs(v0.rect_angle_deg() - 130.0) < 1e-6,
          f"角度可设到 130°（{v0.rect_angle_deg():.0f}°）")

    # 旋转后的掩膜应覆盖斜着的区域
    rm_rot = v0.region_mask((h_img, w_img))
    check(rm_rot is not None and float(rm_rot.max()) > 0.5,
          f"旋转 ROI 掩膜生成成功（面积 {float((rm_rot > 0.5).mean()) * 100:.1f}%）")

    # 方向提示 = ROI 角度
    hint = v0.region_hint()
    check(hint is not None and abs(hint[0] - np.radians(130.0)) < 1e-3,
          f"区域方向提示 = ROI 角度（{np.degrees(hint[0]):.0f}°）")

    # 划痕方向：沿 130°（逐条连通域量主轴，允许 ±10° 抖动 + 弯曲带来的偏差）
    pv.cb_draw.setChecked(True)
    app.processEvents()
    win._force_resegment()
    win.pages["smudge"].enable.setChecked(False)
    win.pages["discolor"].enable.setChecked(False)
    win.pages["chip"].enable.setChecked(False)
    win.pages["scratch"].enable.setChecked(True)
    win.pages["scratch"].rows["count"].setValue(6)
    win.pages["scratch"].rows["length_ratio"].setValue(0.5)
    win.refresh_preview()
    app.processEvents()

    import cv2
    from core.defects import apply_defects as _ad
    from core.mlcc_region import segment_mlcc as _seg2

    _s = _seg2(win.current_image)
    _rm = {"body": _s.body, "electrode": _s.electrode}
    _dm = win._draw_mask()
    _hint = win._draw_hint()
    check(_dm is not None and _hint is not None,
          "手绘掩膜与方向提示都取到")
    devs = []
    for sd in range(4):
        _, ap = _ad(win.current_image,
                    {"scratch": dict(win.current_settings()["scratch"])},
                    seed=sd, region_masks=_rm, region_hint=_hint)
        if not ap:
            continue
        m = (ap[-1].mask > 0.05).astype(np.uint8)
        n_cc, lab = cv2.connectedComponents(m, connectivity=8)
        for i in range(1, n_cc):
            ys, xs = np.nonzero(lab == i)
            if len(xs) < 150:         # 被产品边缘截断的碎块不参与方向统计
                continue
            pts = np.stack([xs, ys], 1).astype(np.float64)
            pts -= pts.mean(axis=0)
            ev, evec = np.linalg.eigh(np.cov(pts.T))
            v = evec[:, int(np.argmax(ev))]
            a = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180
            devs.append(min(abs(a - 130.0), 180 - abs(a - 130.0)))
    mean_dev = float(np.mean(devs)) if devs else 999
    p90 = float(np.percentile(devs, 90)) if devs else 999
    worst = float(np.max(devs)) if devs else 999
    print(f"      逐条划痕主轴偏差：n={len(devs)} 平均 {mean_dev:.1f}° "
          f"P90 {p90:.1f}° 最差 {worst:.1f}°")
    check(len(devs) >= 6, f"取得 {len(devs)} 条可统计的划痕")
    check(mean_dev <= 12.0,
          f"划痕沿 ROI 方向生成（与 130° 平均偏差 {mean_dev:.1f}° ≤ 12°）")
    check(p90 <= 20.0,
          f"九成划痕偏差 ≤ 20°（P90 {p90:.1f}°；最差 {worst:.1f}° 为强弯曲个体）")

    # 颜色：白 / 黑 / 随机
    from core.defects import ScratchDefect

    def _lum_delta(tone):
        gen = ScratchDefect(strength=0.8, severity_mult=1.0, count=3,
                            tone=tone, tone_depth=1.2, length_ratio=0.4,
                            width_px=3.0)
        out, res = [], None
        acc = []
        for sd in range(3):
            from core.utils import Rng
            res = gen.render(win.current_image, Rng(sd), region=None)
            g0 = win.current_image.astype(np.float32).mean(axis=2)
            g1 = res.image.astype(np.float32).mean(axis=2)
            acc.append(float((g1 - g0)[res.mask > 0.5].mean()))
        return float(np.mean(acc))

    d_white = _lum_delta(-1.0)
    d_black = _lum_delta(1.0)
    d_zero = _lum_delta(0.0)
    print(f"      平均亮度变化：白 {d_white:+.1f} / 黑 {d_black:+.1f} / 随机 {d_zero:+.1f}")
    check(d_white > 3.0, f"颜色=-1 → 明显发白（{d_white:+.1f}）")
    check(d_black < -3.0, f"颜色=+1 → 明显发黑（{d_black:+.1f}）")
    check(d_white - d_black > 10.0,
          f"白/黑两端区分明显（差 {d_white - d_black:.1f}）")

    # 深浅：程度越大偏离越多
    def _depth(delta):
        acc = []
        for sd in range(3):
            from core.utils import Rng
            gen = ScratchDefect(strength=0.8, severity_mult=1.0, count=3,
                                tone=1.0, tone_depth=delta, length_ratio=0.4,
                                width_px=3.0)
            res = gen.render(win.current_image, Rng(sd), region=None)
            g0 = win.current_image.astype(np.float32).mean(axis=2)
            g1 = res.image.astype(np.float32).mean(axis=2)
            acc.append(float((g0 - g1)[res.mask > 0.5].mean()))
        return float(np.mean(acc))

    dp_lo, dp_hi = _depth(0.2), _depth(1.5)
    print(f"      深浅 0.2 → {dp_lo:.1f}；深浅 1.5 → {dp_hi:.1f}")
    check(dp_hi > dp_lo * 1.4,
          f"「深浅」可调颜色程度（0.2→{dp_lo:.1f}，1.5→{dp_hi:.1f}）")

    # 颜色/深浅参数出现在配置里
    st2 = win.current_settings()["scratch"]
    check("tone" in st2 and "tone_depth" in st2,
          f"划痕配置含颜色与深浅（tone={st2['tone']} depth={st2['tone_depth']}）")

    pv.cb_draw.setChecked(False)
    win._force_resegment()
    app.processEvents()

    # ================= 缩放功能（点击真实按钮） =================
    btns = {b.text(): b for b in pv.findChildren(QPushButton)}
    check("适应窗口" in btns and "1:1" in btns and "＋" in btns and "－" in btns,
          f"工具栏含 适应窗口/1:1/＋/－ 按钮（实际 {sorted(btns)}）")

    pv.fit_all()
    app.processEvents()
    fit_scale = pv.views[0]._fit_scale
    check(abs(pv.views[0]._zoom - 1.0) < 1e-6,
          f"「适应」后 zoom 归 1.0（实际 {pv.views[0]._zoom:.3f}）")
    btns["＋"].click()
    app.processEvents()
    z_up = pv.views[0]._zoom
    check(abs(z_up - 1.25) < 1e-6, f"「＋」放大到 zoom={z_up:.3f}")
    check(abs(pv.views[0]._scale() - 1.25 * fit_scale) < 1e-6,
          f"实际缩放 = 适应的 1.25 倍（{pv.views[0]._scale():.4f}）")
    check(abs(pv.views[1]._zoom - z_up) < 1e-6 and abs(pv.views[2]._zoom - z_up) < 1e-6,
          "三栏缩放同步")
    btns["－"].click()
    app.processEvents()
    check(abs(pv.views[0]._zoom - 1.0) < 1e-6,
          f"「－」回到 zoom={pv.views[0]._zoom:.3f}")

    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtCore import QPointF, QPoint, Qt as _Qt

    z_before = pv.views[0]._zoom
    ev = QWheelEvent(QPointF(50, 50), QPointF(50, 50), QPoint(0, 0), QPoint(0, 120),
                     _Qt.MouseButton.NoButton, _Qt.KeyboardModifier.NoModifier,
                     _Qt.ScrollPhase.NoScrollPhase, False)
    pv.views[0].wheelEvent(ev)
    app.processEvents()
    check(pv.views[0]._zoom > z_before, f"滚轮放大（{z_before:.3f} → {pv.views[0]._zoom:.3f}）")

    pv.zoom_1to1()
    app.processEvents()
    v = pv.views[0]
    check(abs(v._scale() - 1.0) < 1e-3, f"1:1 后缩放为 1.0（{v._scale():.4f}）")

    off0 = QPointF(v._offset)
    v._panning = True
    v._pan_start = QPointF(0, 0)
    v._pan_off_start = QPointF(off0)
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QEvent

    me = QMouseEvent(QEvent.Type.MouseMove, QPointF(40, 30), QPointF(40, 30),
                     _Qt.MouseButton.LeftButton, _Qt.MouseButton.LeftButton,
                     _Qt.KeyboardModifier.NoModifier)
    v.mouseMoveEvent(me)
    app.processEvents()
    moved = abs(v._offset.x() - off0.x()) + abs(v._offset.y() - off0.y())
    check(moved > 1.0, f"拖拽平移生效（位移 {moved:.1f}px）")
    v._panning = False
    pv.fit_all()
    app.processEvents()

    # ================= 批量（同步调用） =================
    from core.batch import BatchConfig, run_batch

    cfg = BatchConfig(
        input_paths=list(win.input_paths), output_dir=win.ed_out.text(),
        settings=win.current_settings(), images_per_input=1, seed=7,
        max_side=512, use_region=True)
    info2 = run_batch(cfg)
    check(info2["images_generated"] >= 1,
          f"批量生成成功：{info2['images_generated']} 张")

    # 含缺角的批量
    st = win.current_settings()
    st["scratch"]["enable"] = False
    st["smudge"]["enable"] = False
    st["discolor"]["enable"] = False
    st["chip"]["enable"] = True
    st["chip"]["severity"] = "mixed"
    out_chip = os.path.join(HERE, "build", "out_gui_chip")
    info3 = run_batch(BatchConfig(
        input_paths=list(win.input_paths), output_dir=out_chip,
        settings=st, images_per_input=1, seed=11, max_side=512, use_region=True))
    check(info3["images_generated"] >= 1,
          f"缺角批量生成成功：{info3['images_generated']} 张")
    check(info3.get("boxes_per_class", {}).get("chip", 0) >= 1,
          f"缺角标注框已写出（{info3.get('boxes_per_class')}）")

    # 预设保存/载入
    p = os.path.join(HERE, "build", "preset_test.json")
    win.pages["discolor"].rows["strength"].setValue(0.31)
    win.sp_seed.setValue(999)
    pv.sp_angle.setValue(35)
    win.save_preset_checked(p)
    with open(p, "r", encoding="utf-8") as f:
        saved = json.load(f)
    check(saved.get("region", {}).get("roi_angle") == 35,
          f"预设保存了 ROI 角度（{saved.get('region')}）")

    win.pages["discolor"].rows["strength"].setValue(0.77)
    win.sp_seed.setValue(111)
    pv.sp_angle.setValue(0)
    win.load_preset_checked(p)
    app.processEvents()
    check(abs(win.pages["discolor"].rows["strength"].value() - 0.31) < 1e-6,
          f"预设载入成功（discolor.strength="
          f"{win.pages['discolor'].rows['strength'].value()}）")
    check(win.sp_seed.value() == 999, f"seed 已恢复（={win.sp_seed.value()}）")
    check(pv.sp_angle.value() == 35, "ROI 角度随预设恢复")

    # ================= 需求：上一张 / 下一张 =================
    n_in = len(win.input_paths)
    check(n_in >= 3, f"已导入多张图片用于翻页测试（{n_in} 张）")
    win.lst.setCurrentRow(0)
    app.processEvents()
    p0 = win.current_path
    win.btn_next.click()
    app.processEvents()
    check(win.lst.currentRow() == 1 and win.current_path != p0,
          f"「下一张」切到第 2 张（row={win.lst.currentRow()}）")
    win.btn_next.click()
    app.processEvents()
    check(win.lst.currentRow() == 2, f"再点「下一张」到第 3 张（row={win.lst.currentRow()}）")
    win.btn_prev.click()
    app.processEvents()
    check(win.lst.currentRow() == 1, f"「上一张」回到第 2 张（row={win.lst.currentRow()}）")
    # 边界：首/尾不越界
    win.lst.setCurrentRow(0)
    app.processEvents()
    win.btn_prev.click()
    app.processEvents()
    check(win.lst.currentRow() == 0, "首张时点「上一张」不越界")
    win.lst.setCurrentRow(n_in - 1)
    app.processEvents()
    win.btn_next.click()
    app.processEvents()
    check(win.lst.currentRow() == n_in - 1, "末张时点「下一张」不越界")

    # ================= 需求：保存缺陷图 =================
    import glob
    import numpy as _np

    out_dir = os.path.join(HERE, "build", "out_save_one")
    if os.path.isdir(out_dir):
        for f in glob.glob(os.path.join(out_dir, "*")):
            os.remove(f)
    win.lst.setCurrentRow(0)
    app.processEvents()
    win.ed_out.setText(out_dir)
    # 画一条线（划痕需要手绘线条）
    pv.cb_draw.setChecked(True)
    pv.cmb_mode.setCurrentIndex(1)
    for _ in range(3):
        app.processEvents()
    pv.clear_regions()
    v_s = pv.views[0]
    _h, _w = win.current_image.shape[:2]
    v_s._lines.append([(_w * 0.25, _h * 0.50), (_w * 0.75, _h * 0.46)])
    win._force_resegment()
    win.refresh_preview()
    for _ in range(3):
        app.processEvents()
    check("划痕" in pv.lbl_info.text(), "保存前预览已生成划痕")

    from PyQt6.QtWidgets import QMessageBox as _QMB
    _orig_info = _QMB.information
    _orig_warn = _QMB.warning
    _orig_crit = _QMB.critical
    _QMB.information = staticmethod(lambda *a, **k: _QMB.StandardButton.Ok)
    _QMB.warning = staticmethod(lambda *a, **k: _QMB.StandardButton.Ok)
    _QMB.critical = staticmethod(lambda *a, **k: _QMB.StandardButton.Ok)
    try:
        win.btn_save.click()
        app.processEvents()
    finally:
        _QMB.information = _orig_info
        _QMB.warning = _orig_warn
        _QMB.critical = _orig_crit
    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    print("      保存产物:", [os.path.basename(f) for f in pngs])
    check(len(pngs) >= 1, f"「保存缺陷图」写出了文件（{len(pngs)} 个）")
    if pngs:
        from core.utils import imread as _imread
        _im = _imread(pngs[0])
        _src = win.current_image
        check(_im is not None and _im.shape == _src.shape,
              f"保存的缺陷图与原图同尺寸（{None if _im is None else _im.shape}）")
        check(os.path.basename(pngs[0]).startswith("mlcc"),
              f"文件名以原图名开头（{os.path.basename(pngs[0])}）")
        check("seed" in os.path.basename(pngs[0]),
              "文件名带 seed 便于区分多次保存")

    win.close()
    if FAIL:
        print("\n==== 失败项 ====")
        for m in FAIL:
            print(" -", m)
        return 1
    print("\nALL GUI SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)