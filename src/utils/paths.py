# -*- coding: utf-8 -*-
"""
paths.py — 纯 Python 复现的路径与随机种子集中管理

设计原则：
  1. 输入数据目录通过环境变量 B236_DATA_DIR 指定，源码中不出现开发机绝对路径；
  2. 输出目录通过 B236_OUTPUT_DIR 指定，缺省为仓库根下 output/（已被 .gitignore 忽略）；
  3. 全局随机种子 SEED 与 src/core/geoxgboost.py::RANDOM_SEED 保持一致（20260724），
     既有模块中已固定的 random_state=42（fishnet_cutting 的 KMeans/IsolationForest）保持原值不动，
     仅对原本未固定随机性的位置补 SEED，保证重跑结果一致。

使用方式：
    from paths import data_dir, output_dir, SEED
    gdb = data_dir() / "中间数据.gdb"
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 全局随机种子（与 core/geoxgboost.RANDOM_SEED 同值；既有 42 不改动）
SEED = 20260724

# 输入数据清单（L0 体检与 convert_gdb 共用；相对 B236_DATA_DIR）
# expected_crs：该文件在源数据中的真实坐标系（消费方使用前自行 to_crs(4526)）。
# 注意：river_full.shp 源数据即 EPSG:4490（地理坐标），非 4526——以真实数据为准。
VECTOR_INPUTS = {
    "Primary_school.shp": ("小学（99校体系）", 4526),
    "Middle_school.shp": ("初中（99校体系）", 4526),
    "Vitality_fishnet.shp": ("活力渔网（含 vitality 字段）", 4526),
    "Zhanggong_District_Fishing_Net.shp": ("250m 渔网格网", 4526),
    "Zhanggongluwang_Original.shp": ("原始路网", 4526),
    "building_footprint.shp": ("建筑轮廓", 4526),
    "river_full.shp": ("水系（源数据为地理坐标）", 4490),
    "zhanggong.shp": ("研究区行政边界（4.3 服务区裁剪用）", 4526),
}
RASTER_INPUTS = {
    # expected_crs：源数据的真实坐标系（原流程逐栅格 to_crs 容忍混合坐标系，
    # 重构必须保留该行为，勿"统一"成 4526——那会改变结果）
    "WorldPop_250m_EPSG4526.tif": ("WorldPop 人口栅格", 4526),
    "road_density_fixed_EPSG4526.tif": ("路网密度栅格（修正版）", 4526),
    "zhangong_buildings_density.tif": ("建筑密度栅格（WKT 形式存储，语义=4526）", 4526),
    "ZhanggongQu_Physical_Features_V5.tif": ("AlphaEarth 6 维特征栅格（源为 4547，大文件）", 4547),
    "zhanggong_mndwi_landsat.tif": ("MNDWI 水体指数栅格（源为 4326 地理坐标）", 4326),
}
CSV_INPUTS = {
    "school_data.csv": "99 校主数据（含 presure 压力列，经纬度坐标）",
    "POI_data.csv": "高德 POI（lng/lat/cat_id）",
    "L1_学校自述文本.csv": "4.6 语料 L1",
    "L2_媒体报道文本.csv": "4.6 语料 L2",
    "L3_教育局公示.csv": "4.6 语料 L3",
    "community_map.csv": "社区面积/类型映射",
    "99校主名单_shp名称.csv": "99 校主名单",
    "名称_POI_ID对照.csv": "校名→POI_ID 对照（源文件为 GBK 编码）",
}
GDB_NAME = "中间数据.gdb"


def data_dir() -> Path:
    """输入数据目录（环境变量 B236_DATA_DIR，未设置时给出明确指引）。"""
    v = os.environ.get("B236_DATA_DIR", "").strip()
    if not v:
        raise EnvironmentError(
            "未设置输入数据目录环境变量 B236_DATA_DIR。\n"
            "  设置示例（Git Bash）: export B236_DATA_DIR=\"/c/.../数据副本/输入数据\"\n"
            "  设置示例（CMD）     : set B236_DATA_DIR=D:\\...\\数据副本\\输入数据")
    p = Path(v)
    if not p.is_dir():
        raise FileNotFoundError(f"B236_DATA_DIR 指向的目录不存在: {p}")
    return p


def output_dir() -> Path:
    """输出根目录（默认 <仓库根>/output，已 gitignore）。"""
    v = os.environ.get("B236_OUTPUT_DIR", "").strip()
    p = Path(v) if v else REPO_ROOT / "output"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gdb_path() -> Path:
    return data_dir() / GDB_NAME


def converted_gpkg_path() -> Path:
    """convert_gdb.py 的输出 GeoPackage 路径。"""
    out = output_dir() / "gdb_converted" / "中间数据.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def normalized_raster(name: str) -> Path:
    """主基准（4526 带号）归一化栅格路径；缺失时自动生成。

    去带号/经纬度源栅格必须经此取用，下游 core 的 assert_same_axis 守卫
    会在量级混入时立即报错。生成记录见 output/normalized/normalize_report.json。
    """
    dst = output_dir() / "normalized" / name
    if not dst.exists():
        from crs_tools import normalize_raster  # src/utils 已在 sys.path
        rec = normalize_raster(data_dir() / name, dst)
        print(f"[normalize] {name}: {rec['action']}")
    return dst
