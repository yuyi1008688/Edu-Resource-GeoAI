# -*- coding: utf-8 -*-
"""
crs_tools.py — 坐标基准统一与坐标量级守卫

实测事实（2026-08-29，以 L0 基线为准）：
  主分析矢量（渔网/学校/路网，标签 EPSG:4526）东坐标均为 3850 万量级（带带号，
  false_easting=38500000）。部分栅格却是"去带号"表示（东坐标 ~5×10^5，即不带
  38000000 带号），其中：
    - ZhanggongQu_Physical_Features_V5.tif：标签 EPSG:4547（CM 114°E、FE 500000，
      与 4526 同带去号），实际坐标与主矢量相差整整 38000000 m；
    - Zhanggong_road_density.tif：完全无 CRS 标签，东坐标同为去带号量级；
  这类栅格**绝不能按标签 to_crs**（Physical_Features 的 4547 标签本身几何自洽，
  to_crs 4526 会做正确平移；但 Zhanggong_road_density 无标签，且历史上只要标签
  与实际量级不符，to_crs 就会静默生成一片空栅格）。

统一纪律（本模块固化）：
  1. 主基准 = EPSG:4526，东坐标带带号（3850 万量级）；
  2. 去带号栅格：像元坐标整体平移 +38,000,000 m 并 set_crs(4526)——**平移，非 to_crs**；
  3. 经纬度栅格（如 MNDWI EPSG:4326）：正常 to_crs(4526)（rasterio.warp 重投影）；
  4. assert_same_axis() 在每个 zonal / intersect / 距离运算前断言双方东坐标同量级，
     混入 50 万量级或经纬度立即报错。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# 主基准
MAIN_EPSG = 4526
ZONED_MIN = 10_000_000          # 东坐标 ≥ 1e7 视为"带带号"（主基准 ≈3.85e7）
GEOGRAPHIC_MAX = 720.0          # |坐标| ≤ 720 视为经纬度
DEZONE_SHIFT = 38_000_000       # 带号前缀宽度（3 度带 38 带：38500000-500000）


class CRSAxisError(AssertionError):
    """东坐标量级不一致（混入去带号/经纬度坐标）——立即报错，绝不静默。"""


def _bounds_of(obj):
    """支持 GeoDataFrame / rasterio Dataset / dataset 路径 / bounds 四元组。"""
    import geopandas as gpd  # 局部导入，避免循环依赖
    if isinstance(obj, (tuple, list)) and len(obj) == 4:
        return tuple(float(v) for v in obj)
    if isinstance(obj, gpd.GeoDataFrame):
        return tuple(float(v) for v in obj.total_bounds)
    if isinstance(obj, (str, Path)):
        with rasterio.open(obj) as src:
            b = src.bounds
            return (b.left, b.bottom, b.right, b.top)
    if isinstance(obj, rasterio.DatasetReader):
        b = obj.bounds
        return (b.left, b.bottom, b.right, b.top)
    raise TypeError(f"assert_same_axis 不支持的类型: {type(obj)}")


def axis_of(bounds) -> str:
    """按东坐标量级判定：'zoned'（主基准带号）/ 'dezoned'（去带号）/ 'geographic'。"""
    left, bottom, right, top = bounds
    if abs(left) <= GEOGRAPHIC_MAX and abs(right) <= GEOGRAPHIC_MAX \
            and abs(bottom) <= 180 and abs(top) <= 180:
        return "geographic"
    if abs(left) >= ZONED_MIN and abs(right) >= ZONED_MIN:
        return "zoned"
    if abs(left) < ZONED_MIN and abs(right) < ZONED_MIN:
        return "dezoned"
    return "mixed"


def assert_same_axis(*objs, context: str = "", allow_geographic: bool = False):
    """断言所有输入的东坐标同为主基准量级（3850 万带号）。

    参数
    ----
    objs : GeoDataFrame | rasterio.Dataset | 栅格路径 | bounds 四元组
    context : 报错时附带的业务上下文（哪个运算、哪一步）
    allow_geographic : 经纬度输入是否放行（仅限"即将被 to_crs"的输入，如 MNDWI 对齐前的源）；
        即便放行，也要求**全部**输入同为 geographic，不允许 geographic 与投影坐标混用。

    抛出
    ----
    CRSAxisError：量级不一致（zoned 混 dezoned、投影混经纬度）。
    """
    axes = {}
    for o in objs:
        axes.setdefault(axis_of(_bounds_of(o)), []).append(o)
    allowed = {"zoned"} if not allow_geographic else {"zoned", "geographic"}
    bad = set(axes) - allowed
    if not bad:
        return
    detail = "; ".join(f"{axis}×{len(v)}" for axis, v in axes.items())
    where = f"[{context}] " if context else ""
    raise CRSAxisError(
        f"{where}坐标量级不一致（{detail}）。"
        "主基准应为 EPSG:4526 带带号（东坐标≈3850 万）。"
        "检测到去带号（≈5×10^5）或经纬度坐标混入——请先用 "
        "crs_tools.normalize_raster / normalize_dataset 纠偏，不要直接 to_crs。"
    )


# ────────────────────────────────────────────────────────────────
# 栅格纠偏
# ────────────────────────────────────────────────────────────────
def raster_axis_action(src_path) -> str:
    """判定栅格纠偏动作：'copy'（已是主基准）/ 'shift_dezoned'（平移+带号）/ 'reproject'（经纬度）。"""
    with rasterio.open(src_path) as src:
        axis = axis_of((src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top))
    return {"zoned": "copy", "dezoned": "shift_dezoned", "geographic": "reproject"}[axis]


def normalize_raster(src_path, dst_path) -> dict:
    """把任一输入栅格规范到主基准（EPSG:4526，东坐标带号），返回动作记录。

    - copy           : 已是主基准，字节级复制（保留 .ovr 等侧车文件一并复制）；
    - shift_dezoned  : 全波段数据原样，仿射变换东坐标 +38,000,000，set_crs(4526)；
                       像元值零重采样零插值，与原流程 to_crs 后的相对几何严格等价；
    - reproject      : 经纬度 → 4526，rasterio.warp 双线性（与原 load_and_align_mndwi
                       的 Resampling.bilinear 一致），目标分辨率取 default_transform 建议。
    """
    src_path, dst_path = Path(src_path), Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    action = raster_axis_action(src_path)
    rec = {"src": str(src_path), "dst": str(dst_path), "action": action}

    if action == "copy":
        shutil.copyfile(src_path, dst_path)
        for side in (".ovr", ".aux.xml"):
            if Path(str(src_path) + side).exists():
                shutil.copyfile(str(src_path) + side, str(dst_path) + side)

    elif action == "shift_dezoned":
        with rasterio.open(src_path) as src:
            data = src.read()
            t = src.transform
            new_transform = rasterio.Affine(t.a, t.b, t.c + DEZONE_SHIFT,
                                            t.d, t.e, t.f + 0.0)
            profile = src.profile.copy()
            profile.update(transform=new_transform, crs=f"EPSG:{MAIN_EPSG}")
            dst_path.unlink(missing_ok=True)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data)
            b = dst.bounds
            rec["shift_m"] = DEZONE_SHIFT
            rec["bounds_after"] = [round(b.left, 1), round(b.bottom, 1),
                                   round(b.right, 1), round(b.top, 1)]

    elif action == "reproject":
        with rasterio.open(src_path) as src:
            dst_crs = rasterio.crs.CRS.from_epsg(MAIN_EPSG)
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            profile = src.profile.copy()
            profile.update(crs=dst_crs, transform=transform,
                           width=width, height=height)
            dst_path.unlink(missing_ok=True)
            with rasterio.open(dst_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=transform, dst_crs=dst_crs,
                        resampling=Resampling.bilinear)
            rec["reprojected_size"] = [int(width), int(height)]

    # 终检：输出必须落在主基准量级
    assert axis_of(_bounds_of(dst_path)) == "zoned", f"纠偏后仍非带号量级: {dst_path}"
    rec["verified"] = True
    return rec


def report_table(raster_paths) -> list:
    """生成『标签 EPSG vs 实际坐标量级 vs 纠偏动作』对照表（L0 用）。"""
    rows = []
    for p in raster_paths:
        with rasterio.open(p) as src:
            b = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            axis = axis_of(b)
            rows.append({
                "file": Path(p).name,
                "tag_epsg": src.crs.to_epsg() if src.crs else None,
                "easting_range": [round(b[0], 1), round(b[2], 1)],
                "axis": axis,
                "action": {"zoned": "copy", "dezoned": "shift_dezoned",
                           "geographic": "reproject", "mixed": "ERROR"}.get(axis, "ERROR"),
            })
    return rows
