# -*- coding: utf-8 -*-
"""
 shared_utils.py — 共享工具模块（纯 Python 优先）
 提供各 core 模块和 CLI 流水线共用的基础函数，消除重复代码。

 使用方式：
     from shared_utils import _HAS_ARCPY, log_arcpy
     （跨模块共享工具函数）

 注意：
     - 本模块默认纯 Python：不依赖任何商业 GIS，无 arcpy 环境可直接使用
     - 保留对可选专有 GIS 环境的兼容探测：默认走纯 Python 实现，
       日志自动改走 arcpy.AddMessage；其余情况一律 print，无 arcpy 时绝不报错
     - 不导入项目内其他模块（避免循环依赖）
"""

# ── arcpy 可用性检测（可选兼容分支，默认 False） ─────────────────────
_HAS_ARCPY = False
try:
    import arcpy as _arcpy
    _HAS_ARCPY = True
except ImportError:
    _arcpy = None


# ── 统一日志输出 ──────────────────────────────────────────────────────
def log_arcpy(msg):
    """
    统一的日志输出函数：默认 print；
    仅当运行在可选专有环境中时改走其原生消息接口。
    （函数名保留 log_arcpy 以兼容既有调用方与双布局导入）
    """
    if _HAS_ARCPY:
        _arcpy.AddMessage(msg)
    else:
        print(msg)
