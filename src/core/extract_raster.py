#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将多波段 TIF 的 6 维特征提取到渔网 shapefile 中（命令行 / 模块两用）
每个渔网格子 → 计算该格子范围内各波段的均值

此模块将核心处理逻辑封装为参数化的 run_extract_raster() 函数，
既可被流水线 CLI 调用，也可独立运行。
"""
import os
import sys
import geopandas as gpd
import rasterio
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ★ 使用共享工具模块的 arcpy 检测和日志函数（避免与 fishnet_cutting_core.py 重复）
try:
    from _shared_utils import _HAS_ARCPY, log_arcpy as _log  # 平铺布局兼容
except ImportError:
    from shared_utils import _HAS_ARCPY, log_arcpy as _log   # 仓库布局（src/utils）

try:
    from crs_tools import assert_same_axis as _assert_axis  # 坐标量级守卫
except ImportError:  # utils 不在 path 时跳过守卫；流水线入口总会带上
    _assert_axis = None

# ★ 延迟导入 rasterstats.zonal_stats（避免模块级 ImportError）
#   仅在 run_extract_raster() 被实际调用时才检查 rasterstats 是否已安装，
#   确保在缺少可选依赖时给出清晰提示，而不是直接崩溃。
_zonal_stats = None

def _get_zonal_stats():
    """延迟获取 zonal_stats，未安装时给出清晰的中文安装指引。"""
    global _zonal_stats
    if _zonal_stats is None:
        try:
            from rasterstats import zonal_stats as zs
            _zonal_stats = zs
        except ImportError:
            raise ImportError(
                "缺少 rasterstats 包，无法执行栅格分区统计。\n"
                "请在已安装依赖的 Python 环境中运行：\n"
                "    pip install rasterstats\n"
                "或使用项目附带的 requirements.txt 一键安装：\n"
                "    pip install -r requirements.txt"
            )
    return _zonal_stats

# 6 个波段默认字段名（顺序必须与 TIF 波段顺序一致）
# 注意：ESRI Shapefile 字段名最长 10 个字符，此处已满足要求
DEFAULT_BAND_NAMES = [
    'Decay_Idx',   # Band 1: Decay_Index
    'Build_Den',   # Band 2: Building_Density
    'Road_Den',    # Band 3: Road_Density
    'Txt_Compl',   # Band 4: Texture_Complexity
    'LandUseMx',   # Band 5: LandUse_Mix
    'Green_Cov',   # Band 6: Green_Coverage
]

# 注意：_log() 函数已由 _shared_utils.log_arcpy 提供（通过别名引入）

def _safe_field_name(name, max_len=10):
    """
    确保字段名符合 ESRI Shapefile 的 10 字符限制。
    若超长则截断并记录警告。

    参数
    ----
    name    : str  原始字段名
    max_len : int  最大允许长度（默认 10）

    返回
    ----
    str : 安全的字段名
    """
    if len(name) > max_len:
        safe = name[:max_len]
        _log(f"   ⚠ 字段名 '{name}' 超过 {max_len} 字符，已截断为 '{safe}'")
        return safe
    return name


def run_extract_raster(
    fishnet_shp,
    raster_tif,
    output_shp,
    band_names=None,
    output_csv=None,
):
    """
    将多波段栅格的波段均值提取到渔网要素的字段中，
    并将含属性数据的结果写入输出渔网要素类（SHP）。
    同时输出 CSV 属性表作为额外产出。

    参数
    ----
    fishnet_shp : str
        渔网矢量格网路径 (.shp)
    raster_tif  : str
        多波段栅格路径 (.tif)
    output_shp  : str
        输出要素类路径（含提取后的波段特征字段）
        ★ 此文件将包含完整属性数据，可直接用于后续聚类分析
    band_names  : list of str or None
        波段字段名列表（长度需与栅格波段数一致）。
        默认使用 DEFAULT_BAND_NAMES。
        每个字段名不超过 10 字符（ESRI Shapefile 限制）。
    output_csv  : str or None
        输出 CSV 副本路径（默认与 output_shp 同名的 .csv）

    返回
    ----
    dict : {
        "output_shp"   : 输出要素类绝对路径（含属性字段）,
        "output_csv"   : 输出 CSV 绝对路径,
        "band_fields"  : 实际写入 SHP 的波段字段名列表,
    }
    """
    if band_names is None:
        band_names = DEFAULT_BAND_NAMES

    # ── 0. 对字段名做安全处理（截断超长名称）──────────
    safe_band_names = [_safe_field_name(n) for n in band_names]

    # ── 1. 读取渔网 ──────────────────────────────────
    _log("1. 读取渔网 shapefile...")
    gdf = gpd.read_file(fishnet_shp)
    _log(f"   渔网共 {len(gdf)} 个格子")
    _log(f"   现有字段: {list(gdf.columns)}")

    # ── 2. 读取栅格信息 ──────────────────────────────
    _log("\n2. 读取栅格 TIF 信息...")
    with rasterio.open(raster_tif) as src:
        band_count = src.count
        _log(f"   波段数: {band_count}")
        _log(f"   尺寸:   {src.width} x {src.height}")
        _log(f"   CRS:    {src.crs}")
        raster_crs = src.crs

        if band_count != len(safe_band_names):
            _log(
                f"   ⚠ 警告: TIF 有 {band_count} 个波段，"
                f"但定义了 {len(safe_band_names)} 个字段名，"
                f"将只处理前 {min(band_count, len(safe_band_names))} 个波段"
            )

    # 取实际可处理的波段数（防止越界）
    n_bands = min(band_count, len(safe_band_names))
    safe_band_names = safe_band_names[:n_bands]

    # ── 3. 坐标系统一 ────────────────────────────────
    _log("\n3. 检查坐标系...")
    if gdf.crs != raster_crs:
        _log(f"   渔网 CRS: {gdf.crs}")
        _log(f"   栅格 CRS: {raster_crs}")
        _log("   → 将渔网投影转换为栅格坐标系...")
        gdf = gdf.to_crs(raster_crs)
        _log("   ✓ 投影转换完成")
    else:
        _log("   ✓ 坐标系一致，无需转换")

    # ★ 坐标量级守卫：渔网与栅格必须同为主基准带号量级
    if _assert_axis:
        _assert_axis(gdf, raster_tif, context="run_extract_raster")

    # ── 4. 逐波段提取均值并写入 GeoDataFrame ─────────
    _log("\n4. 逐波段提取分区统计（均值）...")

    for band_idx, band_name in enumerate(safe_band_names, start=1):
        _log(f"   提取 Band {band_idx}: {band_name} ...")

        stats = _get_zonal_stats()(
            gdf,
            raster_tif,
            band=band_idx,
            stats=['mean'],
            nodata=-9999,
            all_touched=True
        )

        # 将均值写入 GeoDataFrame 对应列
        # ★ 无值格子填 0.0，保证字段完整性
        gdf[band_name] = [
            float(s['mean']) if s['mean'] is not None else 0.0
            for s in stats
        ]

        non_null = sum(1 for s in stats if s['mean'] is not None)
        _log(f"   ✓ ({non_null}/{len(gdf)} 格子有有效值)")

    # ── 5. 结果预览与统计 ────────────────────────────
    _log("\n5. 提取结果预览:")
    preview_cols = [c for c in ['GRID_ID', 'vitality'] if c in gdf.columns]
    preview_cols += [b for b in safe_band_names if b in gdf.columns]
    if preview_cols:
        _log(gdf[preview_cols].head(10).to_string())

    _log("\n   各特征统计:")
    for col in safe_band_names:
        if col in gdf.columns:
            vals = gdf[col]
            _log(
                f"   {col:12s}  "
                f"min={vals.min():.6f}  "
                f"max={vals.max():.6f}  "
                f"mean={vals.mean():.6f}  "
                f"零值={(vals == 0).sum()} 个"
            )

    # ── 6. 创建输出目录（如不存在）───────────────────
    out_dir = os.path.dirname(os.path.abspath(output_shp))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        _log(f"\n   [提示] 已自动创建输出文件夹: {out_dir}")

    # ── 7. 写入输出渔网要素类（含波段属性字段）────────
    # ★ 核心修改：明确验证 GeoDataFrame 在写入前已含所有波段字段
    _log(f"\n6. 保存输出渔网要素类（含提取属性）...")
    _log(f"   待写入字段: {safe_band_names}")

    # 验证字段是否均已存在于 gdf
    missing = [f for f in safe_band_names if f not in gdf.columns]
    if missing:
        raise RuntimeError(
            f"以下波段字段在写入前丢失，请检查提取步骤: {missing}"
        )

    # 写入 SHP —— GeoDataFrame 包含 geometry + 所有原始字段 + 波段字段
    gdf.to_file(output_shp, driver='ESRI Shapefile', encoding='utf-8')
    _log(f"   ✓ 输出渔网要素类已保存: {output_shp}")

    # 写入后回读验证，确认字段真正落盘
    _log("   验证输出要素类字段...")
    verify_gdf = gpd.read_file(output_shp)
    verified_fields = [f for f in safe_band_names if f in verify_gdf.columns]
    missing_after = [f for f in safe_band_names if f not in verify_gdf.columns]

    if missing_after:
        _log(f"   ✗ 以下字段写入后未能读回，请检查路径权限或磁盘空间: {missing_after}")
    else:
        _log(f"   ✓ 验证通过，以下字段已成功写入输出要素类: {verified_fields}")

    # ── 8. 输出 CSV 属性表（额外产出）───────────────
    _log(f"\n7. 保存属性表 CSV（额外产出）...")
    if output_csv is None:
        output_csv = os.path.splitext(output_shp)[0] + '.csv'

    # CSV 不含 geometry 列，仅保留属性字段
    gdf.drop(columns='geometry').to_csv(
        output_csv, index=False, encoding='utf-8-sig'
    )
    _log(f"   ✓ CSV 属性表已保存: {output_csv}")

    # ── 9. 返回结果路径 ──────────────────────────────
    return {
        "output_shp"  : os.path.abspath(output_shp),   # 含属性的渔网要素类
        "output_csv"  : os.path.abspath(output_csv),   # CSV 属性表（额外产出）
        "band_fields" : safe_band_names,                # 实际写入的波段字段名
    }


# ============================================================
# 独立运行入口（调试用，路径由命令行参数显式指定）
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="将多波段栅格的波段均值提取到渔网要素字段中")
    parser.add_argument("--fishnet", required=True,
                        help="渔网 shapefile 路径")
    parser.add_argument("--raster", required=True,
                        help="多波段栅格 TIF 路径")
    parser.add_argument("--output", required=True,
                        help="输出要素类路径（含提取后的波段特征字段）")
    parser.add_argument("--bands", default=None,
                        help="波段字段名（逗号分隔，如 Decay_Idx,Build_Den）")
    parser.add_argument("--csv", default=None,
                        help="输出 CSV 副本路径（可选）")

    args = parser.parse_args()

    band_names = (
        [b.strip() for b in args.bands.split(",") if b.strip()]
        if args.bands else None
    )

    result = run_extract_raster(
        fishnet_shp=args.fishnet,
        raster_tif=args.raster,
        output_shp=args.output,
        band_names=band_names,
        output_csv=args.csv,
    )

    print("\n─── 运行结果 ───")
    for k, v in result.items():
        print(f"  {k}: {v}")