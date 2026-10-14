"""SyntheticDefectAI 核心包。

纯本地图像处理实现的工业表面缺陷（划痕 / 脏污 / 异色）合成引擎。
"""
from .defects import (BaseDefect, DefectResult, DiscolorDefect, ScratchDefect,
                      SmudgeDefect, apply_defects, make_comparison, merge_masks)
from .batch import (BatchConfig, DEFAULT_SETTINGS, ItemResult, KIND_CN, run_batch)
from .utils import Rng, imread, imwrite, list_images

__version__ = "1.0.0"
__all__ = [
    "BaseDefect", "DefectResult", "ScratchDefect", "SmudgeDefect",
    "DiscolorDefect", "apply_defects", "make_comparison", "merge_masks",
    "BatchConfig", "DEFAULT_SETTINGS", "ItemResult", "KIND_CN", "run_batch",
    "Rng", "imread", "imwrite", "list_images", "__version__",
]
