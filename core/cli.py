"""命令行入口：便于集成进已有流水线。"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .batch import BatchConfig, DEFAULT_SETTINGS, KIND_CN, run_batch
from .defects import DEFECT_KINDS
from .utils import imread, list_images


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="SynthDefectAI",
        description="工业表面缺陷（划痕/脏污/异色/缺角）合成工具 — 命令行模式",
    )
    p.add_argument("-i", "--input", required=True,
                   help="输入图片或包含图片的文件夹")
    p.add_argument("-o", "--output", required=True, help="输出目录")
    p.add_argument("-n", "--num", type=int, default=3, help="每张原图生成张数（默认 3）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument("--max-side", type=int, default=0, help="长边缩放，0 表示原尺寸")
    p.add_argument("--kinds", default="scratch,smudge,discolor",
                   help="启用的缺陷类型，逗号分隔：scratch,smudge,discolor,chip,dot,pit,debris,bump")
    p.add_argument("--strength", type=float, default=None,
                   help="统一覆盖所有缺陷的强度 0~1")
    p.add_argument("--scratch-count", type=int, default=None, help="每张图划痕条数")
    p.add_argument("--scratch-curve", type=float, default=None,
                   help="划痕弯曲度 0=近直线，1=明显蜿蜒（默认 0.55）")
    p.add_argument("--scratch-tone", type=float, default=None,
                   help="划痕颜色 -1=白 … +1=黑，0=按 dark_ratio 随机")
    p.add_argument("--scratch-tone-depth", type=float, default=None,
                   help="划痕颜色深浅程度 0.1~1.5（默认 0.85）")
    p.add_argument("--smudge-count", type=int, default=None, help="每张图脏污斑块数")
    p.add_argument("--discolor-count", type=int, default=None, help="每张图异色区域数")
    p.add_argument("--discolor-color", choices=["random", "yellow", "blue", "green",
                   "red", "purple", "fade", "custom"], default=None,
                   help="异色色调：随机/泛黄/泛蓝/泛绿/泛红/泛紫/褪色/自定义")
    p.add_argument("--discolor-color-rgb", default=None,
                   help="异色自定义目标色，格式 r,g,b（0~255），需配合 --discolor-color custom")
    p.add_argument("--config", default=None, help="从 JSON 配置文件载入参数（命令行参数优先）")
    p.add_argument("--no-region", action="store_true",
                   help="关闭瓷体/电极区域约束（缺陷将生成在全图任意位置）")
    p.add_argument("--bg-side", choices=["auto", "bright_fg", "dark_fg"], default="auto",
                   help="前景亮度先验：auto/bright_fg（深色背景）/dark_fg（亮色背景）")
    p.add_argument("--electrode-ratio", type=float, default=0.0,
                   help="手动指定端电极宽度占比 0.03~0.45，0=自动剖面检测")
    p.add_argument("--target", choices=["body", "electrode", "both"], default=None,
                   help="统一覆盖所有缺陷的目标表面")
    p.add_argument("--severity", choices=["obvious", "medium", "subtle", "mixed"],
                   default=None, help="统一覆盖所有缺陷的明显度")
    p.add_argument("--chip-count", type=int, default=None, help="每张图崩缺个数")
    p.add_argument("--dot-count", type=int, default=None, help="每张图黑点个数")
    p.add_argument("--pit-count", type=int, default=None, help="每张图凹坑个数")
    p.add_argument("--debris-count", type=int, default=None, help="每张图异物个数")
    p.add_argument("--bump-count", type=int, default=None, help="每张图鼓包个数")
    p.add_argument("--no-mask", action="store_true", help="不输出掩膜")
    p.add_argument("--no-region-out", action="store_true", help="不输出区域分割可视化")
    p.add_argument("--no-preview", action="store_true", help="不输出可视化对比图")
    p.add_argument("--no-yolo", action="store_true", help="不输出 YOLO txt")
    p.add_argument("--no-voc", action="store_true", help="不输出 VOC xml")
    p.add_argument("--no-coco", action="store_true", help="不输出 COCO json")
    p.add_argument("--quiet", action="store_true", help="精简输出")
    return p


def cli_main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in (data.get("settings") or {}).items():
            if k in settings:
                settings[k].update(v)

    enabled = [s.strip() for s in args.kinds.split(",") if s.strip()]
    for k in DEFECT_KINDS:
        settings[k]["enable"] = k in enabled

    if args.strength is not None:
        for k in settings:
            settings[k]["strength"] = float(args.strength)
    if args.scratch_count is not None:
        settings["scratch"]["count"] = int(args.scratch_count)
    if args.scratch_curve is not None:
        settings["scratch"]["curve"] = float(args.scratch_curve)
    if args.scratch_tone is not None:
        settings["scratch"]["tone"] = float(max(-1.0, min(1.0, args.scratch_tone)))
    if args.scratch_tone_depth is not None:
        settings["scratch"]["tone_depth"] = float(max(0.1, min(1.5,
                                                             args.scratch_tone_depth)))
    if args.smudge_count is not None:
        settings["smudge"]["blob_count"] = int(args.smudge_count)
    if args.discolor_count is not None:
        settings["discolor"]["region_count"] = int(args.discolor_count)
    if args.discolor_color is not None:
        settings["discolor"]["color"] = args.discolor_color
    if args.discolor_color_rgb is not None:
        try:
            rgb = [int(x) for x in str(args.discolor_color_rgb).split(",")]
            if len(rgb) == 3 and all(0 <= v <= 255 for v in rgb):
                settings["discolor"]["color_rgb"] = rgb
                settings["discolor"]["color"] = "custom"
        except Exception:
            pass
    if args.chip_count is not None:
        settings["chip"]["count"] = int(args.chip_count)
    for _k, _v in (("dot", args.dot_count), ("pit", args.pit_count),
                   ("debris", args.debris_count), ("bump", args.bump_count)):
        if _v is not None and _k in settings:
            settings[_k]["count"] = int(_v)
    if args.target is not None:
        for k in settings:
            settings[k]["target"] = args.target
    if args.severity is not None:
        for k in settings:
            settings[k]["severity"] = args.severity

    src = args.input
    paths = list_images(src, recursive=True) if os.path.isdir(src) else [src]
    if not paths:
        print(f"[错误] 未找到图片：{src}", file=sys.stderr)
        return 2

    cfg = BatchConfig(
        input_paths=paths,
        output_dir=args.output,
        settings=settings,
        images_per_input=max(1, args.num),
        seed=args.seed,
        max_side=args.max_side,
        use_region=not args.no_region,
        bg_side=args.bg_side,
        electrode_ratio=max(0.0, args.electrode_ratio),
        save_mask=not args.no_mask,
        save_preview=not args.no_preview,
        save_region=not args.no_region_out,
        save_yolo=not args.no_yolo,
        save_voc=not args.no_voc,
        save_coco=not args.no_coco,
    )

    def prog(d, t, name):
        if not args.quiet:
            print(f"\r  [{d}/{t}] {name[:60]:<60}", end="", flush=True)

    info = run_batch(cfg, on_progress=prog)
    if not args.quiet:
        print()
    print(f"完成：成功 {info['images_generated']} 张，失败 {info['images_failed']} 张")
    for k, n in (info.get("boxes_per_class") or {}).items():
        print(f"  · {KIND_CN.get(k, k)}：{n} 个缺陷框")
    print(f"输出目录：{os.path.abspath(args.output)}")
    return 0 if info["images_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(cli_main())
