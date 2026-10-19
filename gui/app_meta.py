"""应用元信息：版本 / 作者 / 发布日期 / 标题。

版本规则：从 V1.0.0.0 开始；每发布一个小版本，第三位累加（1.0.1.0、1.0.2.0…），
满 10 进一位（1.0.10.0 → 1.1.0.0）。改动这里后需同步 version_info.txt
（exe 详细信息）再重新打包。
"""

APP_NAME = "SynthDefect AI"
# 窗口标题（用户指定文案）
APP_TITLE = "SynthDefect  AI —表面缺陷模拟生成工具"
APP_VERSION = "1.0.6.0"
APP_AUTHOR = "ZY.Shaw"
APP_RELEASE_DATE = "2026-10-19"
