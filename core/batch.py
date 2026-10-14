"""批量缺陷图生成引擎：遍历输入、可控随机、多产物输出。"""
from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .defects import (DEFECT_KINDS, DefectResult, apply_augmentation,
                      apply_defects, make_comparison, merge_masks, _KIND_TO_COLOR)
from .mlcc_region import segment_mlcc, region_overlay
from .utils import imread, imwrite, list_images, safe_filename, to_u8, clip01


DEFAULT_SETTINGS: Dict[str, dict] = {
    # 数量类参数默认均为 1（数量即最终个数，不再随强度浮动）
    "scratch": {"enable": True, "strength": 0.55, "count": 1,
                "length_ratio": 0.55, "width_px": 2.2, "curve": 0.55,
                "tone": 0.0, "tone_depth": 0.85,
                "target": "both", "severity": "mixed"},
    "smudge": {"enable": False, "strength": 0.50, "blob_count": 1,
               "blob_scale": 0.30, "coverage": 0.35, "dirt_color": "auto",
               "target": "body", "severity": "mixed"},
    "discolor": {"enable": False, "strength": 0.50, "region_count": 1,
                 "region_scale": 0.35,
                 "color": "random", "color_rgb": [150, 130, 80],
                 "sat_gain": 1.25,
                 "target": "body", "severity": "mixed"},
    "chip": {"enable": False, "strength": 0.60, "count": 1,
             "size_ratio": 0.40, "depth_ratio": 0.85, "jag_ratio": 0.42,
             "crack_ratio": 0.35,
             "target": "both", "severity": "mixed"},
}

KIND_CN = {"scratch": "划痕", "smudge": "脏污", "discolor": "异色", "chip": "缺角"}
SEVERITY_CN = {"obvious": "明显", "medium": "中等", "subtle": "不明显", "mixed": "随机混合"}
TARGET_CN = {"body": "瓷体", "electrode": "端电极", "both": "瓷体+电极"}


@dataclass
class BatchConfig:
    input_paths: List[str] = field(default_factory=list)
    output_dir: str = ""
    settings: Dict[str, dict] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_SETTINGS)))
    images_per_input: int = 3          # 每张原图生成几张
    seed: Optional[int] = 42
    max_side: int = 0                  # >0 时按最长边缩放输入，加速处理
    use_region: bool = True            # 是否启用瓷体/电极区域约束
    bg_side: str = "auto"              # auto / bright_fg / dark_fg
    electrode_ratio: float = 0.0       # >0 时手动指定端电极宽度占比
    save_defect: bool = True
    save_mask: bool = True
    save_preview: bool = True
    save_region: bool = True           # 保存区域分割可视化与区域图
    save_yolo: bool = True
    save_voc: bool = True
    save_coco: bool = True
    mask_mode: str = "class"           # class: 每类一张掩膜  union: 合并一张
    jpeg_quality: int = 95
    # 缺陷样本增强模式（在用户手绘区域内随机生成多类缺陷）
    augment: bool = False
    augment_region: Optional[np.ndarray] = None   # 手绘区域掩膜（float32 [0,1]），与首图同尺寸
    augment_images: int = 6                       # 增强模式生成的缺陷图张数
    augment_severity: str = "mixed"


@dataclass
class ItemResult:
    src: str
    out_defect: Optional[str] = None
    out_masks: List[str] = field(default_factory=list)
    out_preview: Optional[str] = None
    boxes: List[dict] = field(default_factory=list)
    ok: bool = True
    error: str = ""


def _boxes_from_mask(mask: np.ndarray, kind: str, min_area_ratio: float = 3e-5):
    """从掩膜提取连通域外接框，返回 [{x,y,w,h,cls_name,score}]。"""
    boxes: List[dict] = []
    b = (mask > 0.35).astype(np.uint8)
    if b.sum() == 0:
        return boxes
    try:
        import cv2

        n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        H, W = mask.shape
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < max(12, int(mask.size * min_area_ratio)):
                continue
            boxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                          "cls_name": kind, "score": 1.0})
    except Exception:
        ys, xs = np.nonzero(b)
        if xs.size:
            boxes.append({"x": int(xs.min()), "y": int(ys.min()),
                          "w": int(xs.max() - xs.min() + 1),
                          "h": int(ys.max() - ys.min() + 1),
                          "cls_name": kind, "score": 1.0})
    return boxes


# --------------------------------------------------------------------------
# 标注写出
# --------------------------------------------------------------------------
def write_yolo(out_dir: str, stem: str, boxes: Sequence[dict], size, class_map: Dict[str, int]):
    h, w = size
    lines = []
    for b in boxes:
        cx = (b["x"] + b["w"] / 2.0) / w
        cy = (b["y"] + b["h"] / 2.0) / h
        bw = b["w"] / w
        bh = b["h"] / h
        cid = class_map.get(b["cls_name"], 0)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path = os.path.join(out_dir, "labels", f"{stem}.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return path


def write_voc(out_dir: str, stem: str, boxes: Sequence[dict], size,
              src_path: str, folder: str = "images"):
    h, w = size
    parts = [
        "<annotation>",
        f"  <folder>{folder}</folder>",
        f"  <filename>{stem}.jpg</filename>",
        f"  <path>{os.path.abspath(src_path)}</path>",
        "  <source><database>SyntheticDefectAI</database></source>",
        f"  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>",
        "  <segmented>0</segmented>",
    ]
    for b in boxes:
        surf = b.get("surface", "")
        surf_xml = f"\n      <surface>{surf}</surface>" if surf else ""
        parts += [
            "  <object>",
            f"    <name>{b['cls_name']}</name>",
            "    <pose>Unspecified</pose>",
            "    <truncated>0</truncated>",
            "    <difficult>0</difficult>",
            f"    <bndbox><xmin>{b['x']}</xmin><ymin>{b['y']}</ymin>"
            f"<xmax>{b['x'] + b['w']}</xmax><ymax>{b['y'] + b['h']}</ymax></bndbox>"
            + surf_xml,
            "  </object>",
        ]
    parts.append("</annotation>")
    path = os.path.join(out_dir, "annotations_voc", f"{stem}.xml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def write_coco(out_dir: str, records: Sequence[dict], class_map: Dict[str, int]):
    """records: [{file_name, width, height, boxes:[...]}]"""
    images, anns = [], []
    aid = 1
    for i, rec in enumerate(records, start=1):
        images.append({"id": i, "file_name": rec["file_name"],
                       "width": rec["width"], "height": rec["height"]})
        for b in rec["boxes"]:
            anns.append({
                "id": aid, "image_id": i,
                "category_id": class_map.get(b["cls_name"], 0),
                "bbox": [b["x"], b["y"], b["w"], b["h"]],
                "area": float(b["w"] * b["h"]),
                "iscrowd": 0, "score": float(b.get("score", 1.0)),
                "surface": b.get("surface", ""),
            })
            aid += 1
    cats = [{"id": cid, "name": name, "supercategory": "defect"}
            for name, cid in class_map.items()]
    data = {
        "info": {"description": "SyntheticDefectAI 合成缺陷数据集",
                 "version": "1.0", "year": 2026},
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": cats,
    }
    path = os.path.join(out_dir, "annotations_coco.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run_batch(cfg: BatchConfig,
              on_progress: Optional[Callable[[int, int, str], None]] = None,
              on_item: Optional[Callable[[ItemResult], None]] = None,
              should_stop: Optional[Callable[[], bool]] = None) -> Dict:
    """执行批量生成。返回统计摘要 dict。"""
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    use_augment = bool(cfg.augment)
    if use_augment:
        # 增强模式：所有四类缺陷都可能出现在手绘区域内，类表固定为全类
        class_map = {k: i for i, k in enumerate(DEFECT_KINDS)}
        enabled = list(DEFECT_KINDS)
    else:
        enabled = [k for k in DEFECT_KINDS
                   if cfg.settings.get(k, {}).get("enable")]
        class_map = {k: i for i, k in enumerate(enabled)} if enabled else {"scratch": 0}

    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    prev_dir = os.path.join(out_dir, "previews")

    base_seed = cfg.seed if cfg.seed is not None else 42
    per_src = (cfg.augment_images if use_augment
               else max(1, cfg.images_per_input))
    total = len(cfg.input_paths) * per_src
    done = 0
    ok_count = 0
    fail_count = 0
    coco_records: List[dict] = []
    summary_per_kind: Dict[str, int] = {k: 0 for k in enabled}

    for src in cfg.input_paths:
        try:
            rgb = imread(src)
        except Exception as e:  # 读图失败
            for _ in range(max(1, cfg.images_per_input)):
                done += 1
                if on_progress:
                    on_progress(done, total, os.path.basename(src))
            fail_count += 1
            if on_item:
                on_item(ItemResult(src=src, ok=False, error=f"读取失败: {e}"))
            continue

        if cfg.max_side and max(rgb.shape[:2]) > cfg.max_side:
            import cv2

            scale = cfg.max_side / float(max(rgb.shape[:2]))
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)

        stem_src = safe_filename(os.path.splitext(os.path.basename(src))[0])

        # ---- 区域分割（每图一次） ----
        seg_ok = False
        seg_res = None
        region_masks = None
        if cfg.use_region and not use_augment:
            try:
                seg_res = segment_mlcc(
                    rgb,
                    electrode_ratio=(cfg.electrode_ratio if cfg.electrode_ratio > 0 else None),
                    bg_side=cfg.bg_side)
                if seg_res.ok and seg_res.body.max() > 0.1:
                    region_masks = {"body": seg_res.body, "electrode": seg_res.electrode}
                    seg_ok = True
                if seg_res.notes:
                    pass  # 备注记录在 dataset_info 汇总
            except Exception as e:
                seg_res = None
                region_masks = None
            if not seg_ok:
                if on_item:
                    on_item(ItemResult(src=src, ok=False,
                                       error="区域分割失败，已跳过（可关闭区域约束重试）"))
                for _ in range(max(1, cfg.images_per_input)):
                    done += 1
                    if on_progress:
                        on_progress(done, total, os.path.basename(src))
                fail_count += 1
                continue

        # 增强模式：apply_augmentation 已一次产出全部张数，外层只跑一轮；
        # 普通模式：按 images_per_input 逐张产出
        n_out = 1 if use_augment else max(1, cfg.images_per_input)
        for k in range(n_out):
            if should_stop and should_stop():
                return _finalize(out_dir, cfg, coco_records, class_map, summary_per_kind,
                                 ok_count, fail_count, stopped=True)
            seed = base_seed + done * 7919
            stem = f"{stem_src}_syn{seed:08d}"
            try:
                if use_augment:
                    reg = _scale_mask(cfg.augment_region, rgb.shape[:2]) \
                        if cfg.augment_region is not None else None
                    items = apply_augmentation(
                        rgb, region=reg, n_images=cfg.augment_images,
                        seed=seed, settings=cfg.settings,
                        severity=cfg.augment_severity)
                else:
                    img, applied = apply_defects(rgb, cfg.settings, seed=seed,
                                                 region_masks=region_masks)
                    items = [(img, applied)]

                # 区域可视化：每张原图只保存一次（首轮迭代 k==0）
                if cfg.save_region and k == 0:
                    rdir = os.path.join(out_dir, "region")
                    os.makedirs(rdir, exist_ok=True)
                    if use_augment and cfg.augment_region is not None:
                        rmask = _scale_mask(cfg.augment_region, rgb.shape[:2])
                        imwrite(os.path.join(rdir, f"{stem_src}_augment_region.png"),
                                np.stack([to_u8(clip01(rmask) * 255.0)] * 3, axis=2))
                    elif seg_res is not None:
                        imwrite(os.path.join(rdir, f"{stem_src}_overlay.png"),
                                region_overlay(rgb, seg_res))
                        rmap = np.zeros(rgb.shape[:2], np.uint8)
                        rmap[seg_res.body > 0.5] = 127
                        rmap[seg_res.electrode > 0.5] = 255
                        imwrite(os.path.join(rdir, f"{stem_src}_map.png"),
                                np.stack([rmap] * 3, axis=2))

                for j, (img, applied) in enumerate(items):
                    if use_augment:
                        jseed = (seed + j * 7919) & 0x7FFFFFFF
                        stem = f"{stem_src}_aug{j:03d}_s{jseed:08d}"
                    else:
                        jseed = seed
                    union = merge_masks(applied, rgb.shape[:2])
                    rec = ItemResult(src=src)
                    if cfg.save_defect:
                        p = os.path.join(img_dir, f"{stem}.jpg")
                        _save_jpg(p, img, cfg.jpeg_quality)
                        rec.out_defect = p

                    if cfg.save_mask:
                        if cfg.mask_mode == "union":
                            mp = os.path.join(mask_dir, "union", f"{stem}.png")
                            os.makedirs(os.path.dirname(mp), exist_ok=True)
                            imwrite(mp, np.stack([to_u8(clip01(union))] * 3, axis=2))
                            rec.out_masks.append(mp)
                        else:
                            for r in applied:
                                mp = os.path.join(mask_dir, r.kind, f"{stem}.png")
                                os.makedirs(os.path.dirname(mp), exist_ok=True)
                                imwrite(mp, np.stack([to_u8(clip01(r.mask))] * 3, axis=2))
                                rec.out_masks.append(mp)

                    if cfg.save_preview and applied:
                        for r in applied:
                            pv = make_comparison(rgb, r)
                            _save_jpg(os.path.join(prev_dir, r.kind, f"{stem}.jpg"),
                                      pv, cfg.jpeg_quality)
                            rec.out_preview = os.path.join(prev_dir, applied[-1].kind)

                    # 逐类提取框（携带表面信息）
                    for r in applied:
                        boxes = _boxes_from_mask(r.mask, r.kind)
                        for b in boxes:
                            b["surface"] = r.surface
                        rec.boxes.extend(boxes)
                        summary_per_kind[r.kind] = summary_per_kind.get(r.kind, 0) + len(boxes)

                    H, W = rgb.shape[:2]
                    if cfg.save_yolo:
                        write_yolo(out_dir, stem, rec.boxes, (H, W), class_map)
                    if cfg.save_voc:
                        write_voc(out_dir, stem, rec.boxes, (H, W), src)
                    if cfg.save_coco:
                        coco_records.append({"file_name": f"images/{stem}.jpg",
                                             "width": W, "height": H,
                                             "boxes": rec.boxes})
                    ok_count += 1
                    if on_item:
                        on_item(rec)
            except Exception as e:
                fail_count += 1
                if on_item:
                    on_item(ItemResult(src=src, ok=False,
                                       error=f"{e}\n{traceback.format_exc()}"))
            finally:
                done += 1
                if on_progress:
                    on_progress(done, total, stem)

    return _finalize(out_dir, cfg, coco_records, class_map, summary_per_kind,
                     ok_count, fail_count, stopped=False)


def _finalize(out_dir, cfg, coco_records, class_map, per_kind, ok, fail, stopped):
    if cfg.save_coco and coco_records:
        write_coco(out_dir, coco_records, class_map)
    # 数据集说明文件
    info = {
        "tool": "SyntheticDefectAI",
        "version": "1.1",
        "stopped_early": bool(stopped),
        "images_generated": ok,
        "images_failed": fail,
        "classes": class_map,
        "boxes_per_class": per_kind,
        "settings": cfg.settings,
        "seed_base": cfg.seed,
        "images_per_input": cfg.images_per_input,
        "augment_mode": bool(cfg.augment),
        "region_constraint": {
            "enabled": bool(cfg.use_region),
            "bg_side": cfg.bg_side,
            "electrode_ratio": cfg.electrode_ratio,
            "augment": bool(cfg.augment),
            "augment_images": cfg.augment_images if cfg.augment else 0,
            "note": ("缺陷样本增强模式：缺陷随机生成在用户手绘区域（region/ 目录含区域图）内，"
                     "类型/位置/形状均随机" if cfg.augment else
                     "缺陷仅生成在目标表面（瓷体/端电极）内；"
                     "划痕方向沿目标区域主轴，region/ 目录含每图分割可视化"),
        },
        "layout": {
            "images/": "缺陷图（训练用）",
            "masks/<kind>/": "各缺陷类型掩膜（灰度，白色为缺陷区）",
            "region/": "区域分割可视化 overlay + 区域图 map（127=瓷体 255=端电极）",
            "previews/<kind>/": "原图|缺陷区域叠加|缺陷图 三联对比",
            "labels/": "YOLO 格式 txt",
            "annotations_voc/": "Pascal VOC 格式 xml（含 surface 字段）",
            "annotations_coco.json": "COCO 格式 json（含 surface 字段）",
        },
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def _scale_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """把一张区域掩膜缩放到目标图像尺寸（增强模式：把当前图的手绘区域复用到各图）。"""
    if mask is None:
        return None
    m = np.asarray(mask, dtype=np.float32)
    if m.shape[:2] == (int(shape[0]), int(shape[1])):
        return clip01(m)
    try:
        import cv2

        rs = cv2.resize(m, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image

        im = Image.fromarray((clip01(m) * 255).astype(np.uint8)).resize(
            (int(shape[1]), int(shape[0])), resample=Image.BILINEAR)
        rs = np.asarray(im, dtype=np.float32) / 255.0
    return clip01(rs)


def _save_jpg(path: str, rgb: np.ndarray, quality: int = 95):
    """JPEG 保存（兼容中文路径）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    try:
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            buf.tofile(path)
            return
    except Exception:
        pass
    from PIL import Image

    Image.fromarray(rgb, mode="RGB").save(path, quality=int(quality))
