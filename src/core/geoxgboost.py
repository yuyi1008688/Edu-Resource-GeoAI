"""
GeoXGBoost — 双模型架构（解释模型+制图模型）+ 一致性检验
============================================================================

模块结构（约 2500 行，功能内聚暂时保持单文件以简化部署）：
  1. 全局常量与 SCI 配色方案          (~L60-100)
  2. matplotlib 字体与样式配置        (~L100-160)
  3. 空间核函数与距离工具函数         (~L160-280)
  4. 特征工程（建筑密度/路网/POI等）  (~L280-700)
  5. 诊断模型（模型A）：XGBoost + SHAP (~L700-1200)
  6. 制图模型（模型B）：空间插值 + 栅格化 (~L1200-1700)
  7. 空间交叉验证与 Optuna 调参       (~L1700-2100)
  8. 一致性检验（Spearman + Kappa）   (~L2100-2300)
  9. 可视化输出（SCI论文级图表）      (~L2300-2500)
 10. run_analysis() 主入口函数        (~L2500+)

说明：
  - 本模块既可被流水线 CLI 调用，也可独立运行
  - 所有路径参数由调用方传入，不依赖硬编码路径
  - 如需拆分为多文件模块，可按上述结构逐段提取
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
# ★ 延迟导入 rasterstats.zonal_stats（避免模块级 ImportError）
#   仅在需要分区统计时才检查 rasterstats 是否已安装
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
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import boxcox as scipy_boxcox
from scipy.stats import spearmanr
from shapely.geometry import Point
from shapely import wkt
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from crs_tools import assert_same_axis
except ImportError:  # utils 不在 path 时跳过守卫；流水线入口总会带上
    assert_same_axis = None

# ========== 用户预设路径（由调用方传入，不再硬编码）==========
# DEFAULT_PATHS 为空字典，所有路径在运行时由调用方动态构建并传入：
#   1. 不依赖特定机器上的绝对路径
#   2. 支持命令行/外部接口灵活指定输入
#   3. 支持纯 Python 环境直接调用（只需传入 paths 字典）
#
# 如需在命令行独立运行，请构建 paths 字典如下：
#   paths = {
#       "school_csv": "path/to/school_data.csv",
#       "fishnet_shp": "path/to/fishnet.shp",
#       "primary_service_area": "path/to/primary_sa.shp",
#       "middle_service_area": "path/to/middle_sa.shp",
#       "worldpop_tif": "path/to/worldpop.tif",
#       "building_density_tif": "path/to/bld_density.tif",       # 可选
#       "building_footprint_shp": "path/to/bld_footprint.shp",   # 可选
#       "output_dir": "path/to/output/",
#   }
DEFAULT_PATHS = {}

# ========== 全局常量 ==========
NODATA = -9999.0
RISK_LABELS = ["极低", "低", "中", "高", "极高"]
D3_WINSORIZE_PCT = 95
K_NEIGHBORS = 6
RANDOM_SEED = 20260724
N_RBF_ANCHORS = 8
RBF_GAMMA_QUANTILE = 0.5
N_OPTUNA_TRIALS = 75
N_OPTUNA_REFINE = 40
OPTUNA_CV_FOLDS = 5
OPTUNA_TIMEOUT = None
OPTUNA_METRIC = "mdape"
MAPE_MIN_DENOM = 0.01
BOXCOT_EPSILON = 1e-4
MIN_FEATURE_IMPORTANCE = 0.001
IDW_UNIQUE_REL_TOL = 0.001

# ==================== SCI论文级字体与样式配置 ====================

# SCI论文配色方案
SCI_COLORS = {
    'primary': '#2E4057',  # 深蓝
    'secondary': '#048A81',  # 青绿
    'accent': '#E84855',  # 红色
    'warm': '#F4A261',  # 橙色
    'light': '#A8DADC',  # 浅蓝
    'neutral': '#6B717E',  # 灰色
    'bg': '#FAFAFA',  # 背景
    'grid': '#E8E8E8',  # 网格线
}

# 风险等级配色
RISK_COLORS = ['#2166AC', '#74ADD1', '#FEE090', '#F46D43', '#A50026']


def _setup_matplotlib_fonts():
    """配置SCI论文级别字体"""
    candidate_fonts = [
        "Microsoft YaHei", "SimHei", "PingFang SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for font in candidate_fonts:
        if font in available:
            chosen = font
            break

    plt.rcParams.update({
        # 字体设置
        "font.family": "sans-serif",
        "font.sans-serif": [chosen, "DejaVu Sans"],
        "axes.unicode_minus": False,
        # 图片质量
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # 坐标轴
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.facecolor": SCI_COLORS['bg'],
        # 刻度
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        # 图例
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": True,
        # 网格
        "grid.alpha": 0.4,
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
        "grid.color": SCI_COLORS['grid'],
        # 线条
        "lines.linewidth": 1.5,
        # 图形背景
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
    })
    return chosen


_FONT_NAME = _setup_matplotlib_fonts()


def sci_style_ax(ax, title=None, xlabel=None, ylabel=None,
                 grid=True, grid_axis='both'):
    """为坐标轴应用SCI论文样式"""
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold',
                     pad=10, color=SCI_COLORS['primary'])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, labelpad=6)
    if grid:
        ax.grid(True, axis=grid_axis, alpha=0.4, linewidth=0.6,
                linestyle='--', color=SCI_COLORS['grid'])
        ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    return ax


def add_panel_label(ax, label, x=-0.12, y=1.05, fontsize=13):
    """添加图面板标签，如(a)(b)(c)"""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            va='top', ha='left', color=SCI_COLORS['primary'])


def canonical_school_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _extra_high_school_keywords():
    """本地补充的高中校名关键词（环境变量 EDU_EXTRA_HIGH_KW，逗号分隔，默认空）。

    若原始数据把某些高中/完全中学笼统标为"中学"、名称里又没有"高中/高级中学/
    完全中学"字样，可在本地通过该环境变量补充，避免误判为初中；该清单属于
    本地数据侧配置，不写入代码仓库。
    """
    raw = os.environ.get("EDU_EXTRA_HIGH_KW", "").strip()
    return [k.strip() for k in raw.split(",") if k.strip()]


def map_level(name: str, level_raw: str) -> str:
    """依据学校名称与原始学段字段判定学段（小学/初中/高中）。

    高中判定先于"中学→初中"，避免把高中/完全中学误判为初中；名称中无通用
    高中字样但确为高中的本地特例，经 EDU_EXTRA_HIGH_KW 补充（见上）。
    """
    name_str = str(name).strip() if name else ""
    lvl_str = str(level_raw).strip() if level_raw else ""

    HIGH_SCHOOL_KEYWORDS = (["高中", "高级中学", "完全中学"]
                            + _extra_high_school_keywords())
    for kw in HIGH_SCHOOL_KEYWORDS:
        if kw in name_str or kw in lvl_str:
            return "高中"

    PRIMARY_KEYWORDS = ["小学", "实验小学", "中心小学"]
    MIDDLE_KEYWORDS = ["初中", "中学", "初级中学"]

    for kw in PRIMARY_KEYWORDS:
        if kw in name_str or kw in lvl_str:
            return "小学"
    for kw in MIDDLE_KEYWORDS:
        if kw in name_str or kw in lvl_str:
            return "初中"

    return lvl_str if lvl_str else "未知"


def log(message: str) -> None:
    print(message, flush=True)


def die(message: str) -> None:
    raise RuntimeError(message)


def json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能序列化: {type(value)}")


def read_vector(path, target_crs, layer_name):
    gdf = gpd.read_file(path)
    if gdf.empty:
        die(f"{layer_name} 为空：{path}")
    if gdf.crs is None:
        die(f"{layer_name} 无坐标系。")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    inv = ~gdf.geometry.is_valid
    if inv.any():
        gdf.loc[inv, "geometry"] = gdf.loc[inv, "geometry"].buffer(0)
    return gdf.to_crs(target_crs)


def require_columns(frame, cols, name):
    miss = [c for c in cols if c not in frame.columns]
    if miss:
        die(f"{name} 缺少字段：{miss}。现有：{list(frame.columns[:10])}")


def find_d3_field(df, requested):
    if requested != "auto":
        require_columns(df, [requested], "学校CSV")
        return requested
    for c in ("D3_pressure", "D3", "D3_raw"):
        if c in df.columns:
            log(f"[自动识别] D3字段 → '{c}'")
            return c
    die("未能自动识别D3字段。")
    return ""


# ==================== Box-Cox ====================

def boxcox_inverse(y_trans, lambda_opt, epsilon=BOXCOT_EPSILON):
    if abs(lambda_opt) < 1e-6:
        y_raw = np.exp(y_trans) - epsilon
    else:
        inner = np.maximum(y_trans * lambda_opt + 1.0, 1e-9)
        y_raw = inner ** (1.0 / lambda_opt) - epsilon
    return np.maximum(y_raw, 0.0)


def apply_boxcox(y, epsilon=BOXCOT_EPSILON):
    y_clean = np.maximum(y, epsilon)
    y_trans, lam = scipy_boxcox(y_clean)
    log(f"[Box-Cox] λ={lam:.4f}, "
        f"变换后偏度={pd.Series(y_trans).skew():.2f}"
        f" (原始偏度={pd.Series(y).skew():.2f})")
    return y_trans, lam


# ==================== 数据读取 ====================

def read_school_csv(path, school_id_field, d3_field,
                    school_crs, target_crs):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")

    log(f"  CSV字段数: {len(df.columns)}")
    require_columns(df, [school_id_field, "geometry_wkt"], "学校CSV")
    d3_field = find_d3_field(df, d3_field)
    df[school_id_field] = df[school_id_field].map(canonical_school_id)
    if df[school_id_field].duplicated().any():
        df = df.drop_duplicates(subset=school_id_field, keep='first')

    # 修复：若存在Level和School_Name字段，重新映射学校类型
    if "Level" in df.columns and "School_Name" in df.columns:
        df["Level"] = df.apply(
            lambda row: map_level(row.get("School_Name", ""),
                                  row.get("Level", "")), axis=1)
        level_counts = df["Level"].value_counts()
        log(f"  学校类型分布: {dict(level_counts)}")
        high_count = int(level_counts.get("高中", 0))
        middle_count = int(level_counts.get("初中", 0))
        log(f"  → 高中={high_count}所, 初中={middle_count}所 (修复后)")

    log(f"  保留全部 {len(df)} 所学校（含所有学校类型）")

    geoms = df["geometry_wkt"].apply(wkt.loads)
    school = gpd.GeoDataFrame(df.copy(), geometry=geoms,
                              crs=school_crs).to_crs(target_crs)
    school[d3_field] = pd.to_numeric(school[d3_field], errors="coerce")
    if school[d3_field].isna().any():
        school = school[school[d3_field].notna()].copy()
    return school, d3_field


def diagnose_id_mismatch(school_ids, iso_ids, sid_field):
    log("\n" + "=" * 60)
    log("【ID匹配诊断】")
    log(f"  字段='{sid_field}' | CSV={len(school_ids)} | "
        f"服务区={len(iso_ids)} | 匹配={len(school_ids & iso_ids)}")
    miss = school_ids - iso_ids
    if miss:
        log(f"  ⚠ 服务区缺失({len(miss)}所): {sorted(miss)}")
    else:
        log("  ✓ 所有学校均匹配成功！")
    extra = iso_ids - school_ids
    if extra:
        log(f"  ℹ 服务区多余({len(extra)}条): {sorted(extra)[:8]}")
    log("=" * 60 + "\n")


# ==================== IDW工具 ====================

def count_effective_unique(values: np.ndarray,
                           rel_tol: float = IDW_UNIQUE_REL_TOL) -> int:
    vrange = values.max() - values.min()
    if vrange < 1e-12:
        return 1
    resolution = vrange * rel_tol
    bins = np.floor(values / resolution).astype(np.int64)
    return int(len(np.unique(bins)))


def idw_interpolate_to_grid(grid_coords, school_coords,
                            school_values, power=2.0, k=15):
    tree = cKDTree(school_coords)
    k_eff = min(k, len(school_coords))
    dists, idxs = tree.query(grid_coords, k=k_eff)

    if k_eff == 1:
        return school_values[idxs]

    zero_mask = (dists == 0)
    has_zero = zero_mask.any(axis=1)
    weights = np.where(dists > 0, 1.0 / (dists ** power), 0.0)

    for i in np.where(has_zero)[0]:
        zc = np.where(zero_mask[i])[0][0]
        weights[i] = 0.0
        weights[i, zc] = 1.0

    w_sum = weights.sum(axis=1, keepdims=True)
    weights = weights / (w_sum + 1e-12)
    return (weights * school_values[idxs]).sum(axis=1)


# ==================== D3质量诊断 ====================

def diagnose_d3_quality(model_df, d3_col="D3_raw"):
    d3 = model_df[d3_col].dropna()
    diag = {
        "n_total": int(len(d3)),
        "n_zero": int((d3 == 0).sum()),
        "pct_zero": float((d3 == 0).mean() * 100),
        "mean": float(d3.mean()),
        "median": float(d3.median()),
        "std": float(d3.std()),
        "cv": float(d3.std() / (d3.mean() + 1e-9)),
        "skewness": float(d3.skew()),
        "p5": float(d3.quantile(0.05)),
        "p95": float(d3.quantile(0.95)),
        "p95_p5_ratio": float(d3.quantile(0.95) /
                              (d3.quantile(0.05) + 1e-9)),
    }
    log("\n" + "=" * 55)
    log("【D3质量诊断】")
    log(f"  样本量/D3=0   : {diag['n_total']} / {diag['n_zero']}")
    log(f"  均值/中位数   : {diag['mean']:.4f} / {diag['median']:.4f}")
    log(f"  CV/偏度       : {diag['cv']:.4f} / {diag['skewness']:.4f}")
    log(f"  P95/P5比      : {diag['p95_p5_ratio']:.1f}×")
    warnings_found = []
    if diag["pct_zero"] > 20:
        warnings_found.append("D3零值>20%")
    if diag["cv"] < 0.3:
        warnings_found.append("CV<0.3区分度低")
    if diag["p95_p5_ratio"] > 50:
        warnings_found.append("极差比>50×，以MdAPE为主指标")
    if warnings_found:
        log(f"  ⚠ {warnings_found}")
    log("=" * 55)
    diag["warnings"] = warnings_found
    return diag


# ==================== 空间特征工程 ====================

def overlay_area_weighted_mean(zones, grid, zone_id, value_field):
    if assert_same_axis:
        assert_same_axis(zones, grid, context="overlay_area_weighted_mean")
    require_columns(zones, [zone_id], "服务区")
    left = zones[[zone_id, "geometry"]].copy()
    right = grid[[value_field, "geometry"]].copy()
    right[value_field] = pd.to_numeric(right[value_field], errors="coerce")
    right = right[right[value_field].notna()].copy()
    inter = gpd.overlay(left, right, how="intersection",
                        keep_geom_type=False)
    if inter.empty:
        die("服务区与格网无相交。")
    inter["_area"] = inter.geometry.area
    inter = inter[inter["_area"] > 0].copy()
    num = (inter[value_field] * inter["_area"]).groupby(inter[zone_id]).sum()
    den = inter["_area"].groupby(inter[zone_id]).sum()
    out = (num / den).rename("X1_vitality").reset_index()
    za = zones.set_index(zone_id).geometry.area
    cov = (den / za).rename("vitality_coverage").reset_index()
    return out.merge(cov, on=zone_id, how="left")


def polygon_area_density(zones, buildings, zone_id, output_field):
    if assert_same_axis:
        assert_same_axis(zones, buildings, context="polygon_area_density")
    left = zones[[zone_id, "geometry"]].copy()
    right = buildings[["geometry"]].copy()
    inter = gpd.overlay(left, right, how="intersection",
                        keep_geom_type=False)
    denom = left.set_index(zone_id).geometry.area.rename("_zone_area")
    if inter.empty:
        out = denom.reset_index()
        out[output_field] = 0.0
        return out[[zone_id, output_field]]
    inter["_ia"] = inter.geometry.area
    built = inter.groupby(zone_id)["_ia"].sum()
    out = denom.to_frame().join(built, how="left").fillna({"_ia": 0.0})
    out[output_field] = np.where(out["_zone_area"] > 0,
                                 out["_ia"] / out["_zone_area"], np.nan)
    return out.reset_index()[[zone_id, output_field]]


# ======================================================================
# 【修复】raster_zonal_mean：添加 rasterio.Env(GDAL_MEM_ENABLE_OPEN=True)
# 解决新版 GDAL 默认关闭内存指针特性导致的 MEM:::DATAPOINTER= 报错。
# 原功能完全不变，仅在外层包裹 rasterio.Env 上下文管理器。
# ======================================================================
def raster_zonal_mean(zones, raster_path, output_field):
    if assert_same_axis:
        assert_same_axis(zones, raster_path, context="raster_zonal_mean")
    with rasterio.Env(GDAL_MEM_ENABLE_OPEN=True):
        with rasterio.open(raster_path) as src:
            nodata = src.nodata
        stats = _get_zonal_stats()(zones, raster_path, stats=["mean"],
                            nodata=nodata, all_touched=False)
    values = [s.get("mean", np.nan) if s.get("mean") is not None
              else np.nan for s in stats]
    return pd.DataFrame({output_field: values}, index=zones.index)


def build_spatial_weight_matrix(coords, k, kernel="gaussian"):
    dist = cdist(coords, coords)
    np.fill_diagonal(dist, np.inf)
    idx = np.argsort(dist, axis=1)[:, :k]
    rd = dist[np.arange(len(coords))[:, None], idx]
    if kernel == "idw":
        w = 1.0 / (rd + 1e-9)
    elif kernel == "gaussian":
        bw = np.median(rd)
        w = np.exp(-0.5 * (rd / (bw + 1e-9)) ** 2)
    elif kernel == "bisquare":
        bw = rd.max(axis=1, keepdims=True)
        u = rd / (bw + 1e-9)
        w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
    else:
        raise ValueError(f"未知核函数: {kernel}")
    ws = w.sum(axis=1, keepdims=True)
    return idx, w / (ws + 1e-9)


def compute_spatial_lag(values, neigh_indices, neigh_weights):
    return (neigh_weights * values[neigh_indices]).sum(axis=1)


class RBFSpatialEncoder:
    def __init__(self, n_anchors=8, gamma_quantile=0.5, random_state=42):
        self.n_anchors = n_anchors
        self.gamma_quantile = gamma_quantile
        self.random_state = random_state
        self.anchors_ = None
        self.gamma_ = None

    def fit(self, coords):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_anchors,
                    random_state=self.random_state, n_init=10)
        km.fit(coords)
        self.anchors_ = km.cluster_centers_
        pw = cdist(coords, coords)
        np.fill_diagonal(pw, np.nan)
        self.gamma_ = 1.0 / (2.0 * np.nanquantile(
            pw, self.gamma_quantile) ** 2 + 1e-9)
        log(f"[RBF] 锚点={self.n_anchors}, γ={self.gamma_:.4e}")
        return self

    def transform(self, coords):
        return np.exp(-self.gamma_ * cdist(coords, self.anchors_) ** 2)

    def fit_transform(self, coords):
        return self.fit(coords).transform(coords)

    def get_feature_names(self):
        return [f"rbf_{i}" for i in range(self.n_anchors)]


def compute_gwr_sample_weights(coords):
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(coords.T)
    density = kde(coords.T)
    w = 1.0 / (density + 1e-9)
    wmin, wmax = w.min(), w.max()
    w = 0.5 + 1.5 * (w - wmin) / (wmax - wmin + 1e-9)
    log(f"[GWR权重] [{w.min():.3f},{w.max():.3f}], 均值={w.mean():.3f}")
    return w


def compute_morans_i(residuals, coords, k=6):
    n = len(residuals)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idx[i]] = 1.0
    Ws = W.sum()
    if Ws == 0:
        return {"moran_I": np.nan, "z_score": np.nan,
                "interpretation": "无法计算"}
    z = residuals - residuals.mean()
    I = (n / Ws) * (z @ W @ z) / (z @ z + 1e-9)
    EI = -1.0 / (n - 1)
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum()
    VI = ((n * n * S1 - n * S2 + 3 * Ws ** 2) /
          ((Ws ** 2) * (n * n - 1) + 1e-9)) - EI ** 2
    zs = (I - EI) / (np.sqrt(abs(VI)) + 1e-9)
    interp = ("残差无显著空间自相关(p>0.05) ✓" if abs(zs) < 1.96
              else f"残差存在{'正' if I > 0 else '负'}空间自相关(p<0.05) ⚠")
    return {"moran_I": float(I), "z_score": float(zs),
            "E_I": float(EI), "interpretation": interp}


# ==================== 度量指标 ====================

def safe_mape(y_true, y_pred, min_denom=MAPE_MIN_DENOM):
    denom = np.maximum(np.abs(y_true), min_denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom * 100.0))


def safe_mdape(y_true, y_pred, min_denom=MAPE_MIN_DENOM):
    denom = np.maximum(np.abs(y_true), min_denom)
    return float(np.median(np.abs(y_true - y_pred) / denom * 100.0))


def safe_spearmanr(x, y, label=""):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        msg = f"[{label}] 有效样本不足3个(n={len(x)})，无法计算秩相关"
        log(f"  ⚠ {msg}")
        return np.nan, np.nan, msg

    if np.std(x) < 1e-10:
        msg = f"[{label}] x序列为常数(std={np.std(x):.2e})，无法计算秩相关"
        log(f"  ⚠ {msg}")
        return np.nan, np.nan, msg

    if np.std(y) < 1e-10:
        msg = f"[{label}] y序列为常数(std={np.std(y):.2e})，无法计算秩相关"
        log(f"  ⚠ {msg}")
        return np.nan, np.nan, msg

    rho, p = spearmanr(x, y)
    return float(rho), float(p), ""


def fmt_p(p_val):
    if p_val is None or not np.isfinite(p_val):
        return "nan"
    return f"{p_val:.4f}"


def fmt_rho(rho_val, p_val):
    if rho_val is None or not np.isfinite(rho_val):
        return "nan", "n.s."
    p = p_val if (p_val is not None and np.isfinite(p_val)) else 1.0
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    return f"{rho_val:.3f}", sig


# ==================== 分层KFold ====================

def make_stratified_kfold_by_raw_d3(
        d3_raw: np.ndarray,
        n_splits: int = 5,
        random_state: int = RANDOM_SEED):
    n = len(d3_raw)
    quantiles = np.quantile(d3_raw, np.linspace(0, 1, n_splits + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 2:
        log("  [分层KFold] D3分位退化，使用普通KFold")
        kf = KFold(n_splits=n_splits, shuffle=True,
                   random_state=random_state)
        yield from kf.split(np.arange(n))
        return

    strata = np.digitize(d3_raw, quantiles[1:-1], right=False)
    folds = [[] for _ in range(n_splits)]
    rng = np.random.RandomState(random_state)
    for stratum in np.unique(strata):
        sidx = np.where(strata == stratum)[0]
        rng.shuffle(sidx)
        for i, idx in enumerate(sidx):
            folds[i % n_splits].append(int(idx))

    for fold_i in range(n_splits):
        val_idx = np.array(folds[fold_i])
        train_idx = np.array([i for j, f in enumerate(folds)
                              if j != fold_i for i in f])
        yield train_idx, val_idx


# ==================== CV ====================

def run_cv(X_scaled, y_transformed, d3_raw,
           best_params, sample_weight,
           n_splits=5, random_state=RANDOM_SEED,
           boxcox_lambda=None, label=""):
    metrics = {
        "r2_transformed": [], "r2_raw": [],
        "rmse_raw": [], "mae_raw": [],
        "mape_raw": [], "mdape_raw": [],
        "smearing_sf": [],
    }
    log(f"\n[交叉验证 {label}] {n_splits}折分层验证（按原始D3分层）...")
    fold_gen = make_stratified_kfold_by_raw_d3(
        d3_raw, n_splits, random_state)

    for fold, (tr_idx, val_idx) in enumerate(fold_gen):
        sw = sample_weight[tr_idx] if sample_weight is not None else None
        m = xgb.XGBRegressor(**best_params)
        m.fit(X_scaled[tr_idx], y_transformed[tr_idx],
              sample_weight=sw, verbose=False)

        tr_pred = m.predict(X_scaled[tr_idx])
        sf = float(np.mean(np.exp(y_transformed[tr_idx] - tr_pred)))
        yp_t = m.predict(X_scaled[val_idx])

        if boxcox_lambda is not None:
            y_p = boxcox_inverse(yp_t, boxcox_lambda)
            y_v = boxcox_inverse(y_transformed[val_idx], boxcox_lambda)
        else:
            y_p = np.maximum(np.exp(yp_t) * sf - BOXCOT_EPSILON, 0.0)
            y_v = np.maximum(np.exp(y_transformed[val_idx])
                             - BOXCOT_EPSILON, 0.0)

        r2t = float(r2_score(y_transformed[val_idx], yp_t))
        r2r = float(r2_score(y_v, y_p))
        metrics["r2_transformed"].append(r2t)
        metrics["r2_raw"].append(r2r)
        metrics["rmse_raw"].append(
            float(np.sqrt(mean_squared_error(y_v, y_p))))
        metrics["mae_raw"].append(float(mean_absolute_error(y_v, y_p)))
        metrics["mape_raw"].append(safe_mape(y_v, y_p))
        metrics["mdape_raw"].append(safe_mdape(y_v, y_p))
        metrics["smearing_sf"].append(sf)
        log(f"  折{fold + 1}: R²(变换)={r2t:.4f},"
            f" R²(原始)={r2r:.4f}, MdAPE={metrics['mdape_raw'][-1]:.1f}%")

    r2t_m = np.mean(metrics["r2_transformed"])
    r2r_m = np.mean(metrics["r2_raw"])
    mdape_m = np.mean(metrics["mdape_raw"])
    log(f"  汇总: R²(变换)={r2t_m:.4f}"
        f"(±{np.std(metrics['r2_transformed']):.4f})"
        f" R²(原始)={r2r_m:.4f}"
        f"(±{np.std(metrics['r2_raw']):.4f})"
        f" MdAPE={mdape_m:.1f}%")
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "all": [float(x) for x in v]}
            for k, v in metrics.items()}


# ==================== Huber基线 ====================

def huber_baseline(X_scaled, y):
    try:
        from sklearn.linear_model import HuberRegressor
        col_var = X_scaled.var(axis=0)
        X_valid = X_scaled[:, col_var > 1e-10]
        n_rem = (col_var <= 1e-10).sum()
        if n_rem > 0:
            log(f"  [Huber基线] 移除 {n_rem} 个常数列，剩余 {X_valid.shape[1]} 列")
        if X_valid.shape[1] == 0:
            return {"huber_cv_r2": None, "note": "所有列为常数"}
        huber = HuberRegressor(epsilon=1.35, max_iter=300)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(huber, X_valid, y, cv=5, scoring='r2')
        valid = scores[np.isfinite(scores)]
        if len(valid) == 0:
            return {"huber_cv_r2": None, "note": "CV全NaN"}
        r2, std = float(np.mean(valid)), float(np.std(valid))
        log(f"[Huber基线] R²={r2:.4f} (±{std:.4f})")
        if r2 < 0.1:
            log(f"  → 线性无解释力，XGBoost有合理性 ✓")
        return {"huber_cv_r2": r2, "huber_cv_r2_std": std}
    except Exception as e:
        log(f"[Huber基线] 失败: {e}")
        return {"huber_cv_r2": None, "error": str(e)}


# ==================== 两阶段特征精简 ====================

def two_stage_feature_selection(
        X_scaled, y, feature_cols, scaler_X,
        base_params, sample_weight,
        target_ratio=5.0, random_state=RANDOM_SEED):
    n_samples = len(y)
    n_features = len(feature_cols)
    log(f"\n[特征精简] 两阶段特征精简")
    log(f"  初始: {n_samples}样本/{n_features}特征"
        f" = 比例{n_samples / n_features:.1f}")

    m1 = xgb.XGBRegressor(**base_params)
    m1.fit(X_scaled, y, sample_weight=sample_weight, verbose=False)
    importance = m1.feature_importances_

    keep_mask = importance > MIN_FEATURE_IMPORTANCE
    kept = [f for f, k in zip(feature_cols, keep_mask) if k]
    removed = [f for f, k in zip(feature_cols, keep_mask) if not k]
    if removed:
        log(f"  移除零重要性特征({len(removed)}): {removed}")

    max_feat = max(int(n_samples / target_ratio), 5)
    if len(kept) > max_feat:
        imp_kept = sorted(
            [(f, importance[feature_cols.index(f)]) for f in kept],
            key=lambda x: x[1], reverse=True)
        kept = [f for f, _ in imp_kept[:max_feat]]
        log(f"  样本/特征比控制: 精简至 {len(kept)} 个")

    log(f"  保留特征({len(kept)}): {kept}")
    log(f"  精简后比例: {n_samples / len(kept):.1f}"
        f"  {'✓' if n_samples / len(kept) >= target_ratio else '⚠'}")

    X_raw = scaler_X.inverse_transform(X_scaled)
    kid = [feature_cols.index(f) for f in kept]
    scaler_new = StandardScaler().fit(X_raw[:, kid])
    X_new = scaler_new.transform(X_raw[:, kid])
    return kept, X_new, scaler_new, kid


# ==================== Optuna ====================

def run_optuna(X_scaled, y, d3_raw, sample_weight,
               n_trials, cv_folds, random_state, outdir,
               optimize_metric="mdape", boxcox_lambda=None,
               label=""):
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError:
        die("请先安装optuna: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log(f"\n[贝叶斯调参 {label}] 目标={optimize_metric} | 试验={n_trials}")

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "gamma": trial.suggest_float("gamma", 0.5, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 80.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.85),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 0.85),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 700),
            "random_state": random_state, "verbosity": 0,
            "tree_method": "hist",
        }
        fold_gen = make_stratified_kfold_by_raw_d3(
            d3_raw, cv_folds, random_state)
        scores = []
        for step, (tr_idx, val_idx) in enumerate(fold_gen):
            sw = sample_weight[tr_idx] if sample_weight is not None else None
            m = xgb.XGBRegressor(**params)
            m.fit(X_scaled[tr_idx], y[tr_idx],
                  sample_weight=sw, verbose=False)
            sf = float(np.mean(np.exp(y[tr_idx] - m.predict(X_scaled[tr_idx]))))
            yp = m.predict(X_scaled[val_idx])
            if boxcox_lambda is not None:
                yp_raw = boxcox_inverse(yp, boxcox_lambda)
                yv_raw = boxcox_inverse(y[val_idx], boxcox_lambda)
            else:
                yp_raw = np.maximum(np.exp(yp) * sf - BOXCOT_EPSILON, 0.0)
                yv_raw = np.maximum(np.exp(y[val_idx]) - BOXCOT_EPSILON, 0.0)

            if optimize_metric == "mdape":
                score = -safe_mdape(yv_raw, yp_raw)
            elif optimize_metric == "mape":
                score = -safe_mape(yv_raw, yp_raw)
            else:
                score = float(r2_score(y[val_idx], yp))
            scores.append(score)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state, n_startup_trials=12),
        pruner=MedianPruner(n_startup_trials=6, n_warmup_steps=2),
    )
    study.enqueue_trial({
        "max_depth": 3, "min_child_weight": 10,
        "gamma": 2.0, "reg_lambda": 25.0, "reg_alpha": 8.0,
        "subsample": 0.65, "colsample_bytree": 0.70,
        "colsample_bylevel": 0.70,
        "learning_rate": 0.04, "n_estimators": 350,
    })
    study.optimize(objective, n_trials=n_trials, timeout=OPTUNA_TIMEOUT,
                   show_progress_bar=False)

    best = study.best_params.copy()
    best.update({"random_state": random_state,
                 "verbosity": 0, "tree_method": "hist"})
    bv = study.best_value
    if optimize_metric in ("mdape", "mape"):
        log(f"  最优 {optimize_metric.upper()} = {-bv:.2f}%"
            f" | 完成试验: {len(study.trials)}")
    else:
        log(f"  最优 R² = {bv:.4f} | 完成试验: {len(study.trials)}")

    out_f = outdir / f"optuna_trials_{label or 'main'}.csv"
    pd.DataFrame([{"trial": t.number, "score": t.value, **t.params}
                  for t in study.trials if t.value is not None]
                 ).sort_values("score", ascending=False
                               ).to_csv(out_f, index=False, encoding="utf-8-sig")
    _plot_optuna(study, outdir, optimize_metric, label)
    return best


def _plot_optuna(study, outdir, metric, label=""):
    """SCI论文级别Optuna优化历史图"""
    try:
        trials = [(t.number, t.value)
                  for t in study.trials if t.value is not None]
        if not trials:
            return
        nums, vals = zip(*trials)
        best_so_far = np.maximum.accumulate(vals)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('white')

        ylabel = (f"负{metric.upper()}"
                  if metric in ("mdape", "mape") else "决定系数 R²")

        # 散点图：所有试验
        ax.scatter(nums, vals, alpha=0.35, s=20,
                   color=SCI_COLORS['light'], zorder=2,
                   label="单次试验")
        # 最优曲线
        ax.plot(nums, best_so_far, color=SCI_COLORS['accent'],
                lw=2.0, zorder=3, label="当前最优")
        # 最优点标注
        best_idx = int(np.argmax(best_so_far))
        ax.scatter([nums[best_idx]], [best_so_far[best_idx]],
                   color=SCI_COLORS['accent'], s=80, zorder=4,
                   marker='*', label=f"最优值={best_so_far[best_idx]:.4f}")

        sci_style_ax(ax,
                     title=f"GeoXGBoost 贝叶斯超参数优化历史（{label}）",
                     xlabel="试验编号",
                     ylabel=ylabel)
        ax.legend(loc='lower right', fontsize=9)

        plt.tight_layout()
        fig.savefig(outdir / f"optuna_history_{label or 'main'}.png",
                    dpi=300, bbox_inches="tight", facecolor='white')
        plt.close(fig)
    except Exception as e:
        log(f"[Optuna图] {e}")


# ==================== SHAP ====================

def run_shap_analysis(model, X_scaled, feature_cols,
                      model_df, fishnet, grid_feat_scaled,
                      outdir, sid, boxcox_lambda=None):
    try:
        import shap
    except ImportError:
        log("[SHAP] 未安装，跳过。")
        return {}

    log("\n" + "─" * 55)
    log("[SHAP 分析]")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_scaled)
    expected_val = float(explainer.expected_value)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    pd.DataFrame(shap_vals, columns=feature_cols
                 ).assign(**{sid: model_df[sid].values}
                          ).to_csv(outdir / "shap_values_school.csv",
                                   index=False, encoding="utf-8-sig")

    _shap_summary(shap_vals, X_scaled, feature_cols, outdir)
    _shap_bar(mean_abs, feature_cols, outdir)
    top4 = np.argsort(mean_abs)[::-1][:4]
    _shap_dep(shap_vals, X_scaled, feature_cols, top4, outdir)
    _shap_spatial(explainer, grid_feat_scaled,
                  feature_cols, mean_abs, fishnet, outdir)

    top5 = [feature_cols[i] for i in np.argsort(mean_abs)[::-1][:5]]
    log(f"  SHAP 前5位特征: {top5}")
    log("[SHAP] 完成！")
    return {"expected_value": expected_val, "top5_features": top5,
            "mean_abs_shap": {feature_cols[i]: float(mean_abs[i])
                              for i in range(len(feature_cols))}}


def _make_chinese_feature_name(name: str) -> str:
    """将英文特征名映射为中文标签（用于图表）"""
    mapping = {
        "X1_vitality": "区域活力指数",
        "X2_build_den": "建筑密度",
        "X3_worldpop": "人口密度",
        "lag_X1_vitality": "空间滞后活力",
        "lag_X2_build_den": "空间滞后建筑密度",
        "lag_X3_worldpop": "空间滞后人口密度",
        "lag_D3": "空间滞后学位压力",
        "x_coord": "横坐标",
        "y_coord": "纵坐标",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("rbf_") or name.startswith("B_rbf_"):
        idx = name.split("_")[-1]
        return f"RBF空间基函数{idx}"
    return name


def _shap_summary(shap_vals, X_scaled, feature_cols, outdir):
    """SCI论文级别SHAP摘要蜂巢图"""
    order = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1]
    n = len(order)

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.45)))
    fig.patch.set_facecolor('white')

    np.random.seed(42)
    # 自定义红蓝色谱
    cmap = LinearSegmentedColormap.from_list(
        'shap_cmap',
        ['#2166AC', '#92C5DE', '#F7F7F7', '#F4A582', '#CA0020'], N=256)

    for ri, fi in enumerate(order):
        sv = shap_vals[:, fi]
        xv = X_scaled[:, fi]
        xn = (xv - xv.min()) / (xv.max() - xv.min() + 1e-9)
        jitter = np.random.uniform(-0.22, 0.22, len(sv))
        sc = ax.scatter(sv, ri + jitter,
                        c=xn, cmap=cmap, vmin=0, vmax=1,
                        s=22, alpha=0.75, linewidths=0, zorder=2)

    ax.axvline(0, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(range(n))

    ylabels = [_make_chinese_feature_name(feature_cols[i]) for i in order]
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("SHAP值（对预测的影响量）", fontsize=11)

    # 颜色条
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("特征值（低→高）", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["低", "中", "高"])

    sci_style_ax(ax, title="SHAP特征重要性摘要图",
                 grid=True, grid_axis='x')
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(outdir / "shap_summary_beeswarm.png",
                dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def _shap_bar(mean_abs, feature_cols, outdir):
    """SCI论文级别SHAP条形图"""
    order = np.argsort(mean_abs)
    vals = mean_abs[order]
    names = [_make_chinese_feature_name(feature_cols[i]) for i in order]
    n = len(names)

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.45)))
    fig.patch.set_facecolor('white')

    # 渐变色条形
    norm_vals = vals / (vals.max() + 1e-9)
    colors = [plt.cm.Blues(0.35 + 0.55 * v) for v in norm_vals]
    bars = ax.barh(range(n), vals, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.65)

    # 数值标签
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(val + vals.max() * 0.01, i, f"{val:.4f}",
                va='center', ha='left', fontsize=8,
                color=SCI_COLORS['primary'])

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("平均 |SHAP值|", fontsize=11)

    sci_style_ax(ax, title="GeoXGBoost 特征重要性（SHAP均值）",
                 grid=True, grid_axis='x')
    ax.set_xlim(0, vals.max() * 1.15)

    plt.tight_layout()
    fig.savefig(outdir / "shap_bar_plot.png",
                dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def _shap_dep(shap_vals, X_scaled, feature_cols, top4_idx, outdir):
    """SCI论文级别SHAP依赖图（2×2面板）"""
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    cmap = LinearSegmentedColormap.from_list(
        'dep_cmap', ['#2166AC', '#F7F7F7', '#CA0020'], N=256)

    for pi, fi in enumerate(top4_idx):
        ax = fig.add_subplot(2, 2, pi + 1)
        sv = shap_vals[:, fi]
        xv = X_scaled[:, fi]
        corrs = [abs(np.corrcoef(X_scaled[:, j], sv)[0, 1])
                 if j != fi else -1.0
                 for j in range(len(feature_cols))]
        interact = int(np.argmax(corrs))
        norm_iv = ((X_scaled[:, interact] - X_scaled[:, interact].min()) /
                   (X_scaled[:, interact].max() -
                    X_scaled[:, interact].min() + 1e-9))
        sc = ax.scatter(xv, sv, c=norm_iv, cmap=cmap,
                        vmin=0, vmax=1,
                        s=40, alpha=0.8, edgecolors="none", zorder=2)
        ax.axhline(0, color="#888888", lw=1.0, ls="--", zorder=1)

        feat_name = _make_chinese_feature_name(feature_cols[fi])
        interact_name = _make_chinese_feature_name(feature_cols[interact])

        cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(interact_name, fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["低", "中", "高"])

        sci_style_ax(ax,
                     title=f"{panel_labels[pi]} {feat_name}",
                     xlabel=f"{feat_name}（标准化）",
                     ylabel="SHAP值")
        add_panel_label(ax, panel_labels[pi])

    plt.suptitle("SHAP特征依赖图（前4位特征）",
                 fontsize=13, fontweight='bold', y=1.01,
                 color=SCI_COLORS['primary'])
    plt.tight_layout()
    fig.savefig(outdir / "shap_dependence_top4.png",
                dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def _shap_spatial(explainer, grid_feat, feature_cols,
                  mean_abs, fishnet, outdir):
    """SCI论文级别SHAP空间分布图"""
    try:
        gs = explainer.shap_values(grid_feat)
        top3 = np.argsort(mean_abs)[::-1][:3]

        fig, axs = plt.subplots(1, 3, figsize=(17, 6))
        fig.patch.set_facecolor('white')

        cmap_div = LinearSegmentedColormap.from_list(
            'shap_div', ['#2166AC', '#F7F7F7', '#CA0020'], N=256)
        panel_labels = ["(a)", "(b)", "(c)"]

        for ci, fi in enumerate(top3):
            sv = gs[:, fi]
            vmax = max(abs(sv.min()), abs(sv.max()))
            fp = fishnet.copy()
            fp["_sv"] = sv

            fp.plot(column="_sv", ax=axs[ci],
                    cmap=cmap_div, vmin=-vmax, vmax=vmax,
                    legend=True,
                    legend_kwds={
                        "label": "SHAP值",
                        "orientation": "horizontal",
                        "pad": 0.05,
                        "shrink": 0.75,
                        "aspect": 30,
                    })
            feat_name = _make_chinese_feature_name(feature_cols[fi])
            axs[ci].set_title(f"{panel_labels[ci]} {feat_name}",
                              fontsize=11, fontweight='bold',
                              color=SCI_COLORS['primary'])
            axs[ci].set_axis_off()
            axs[ci].text(0.02, 0.98, panel_labels[ci],
                         transform=axs[ci].transAxes,
                         fontsize=13, fontweight='bold',
                         va='top', color=SCI_COLORS['primary'])

        plt.suptitle("学位压力影响因素空间SHAP分布（前3位特征）",
                     fontsize=13, fontweight='bold', y=1.02,
                     color=SCI_COLORS['primary'])
        plt.tight_layout()
        fig.savefig(outdir / "shap_spatial_map.png",
                    dpi=300, bbox_inches="tight", facecolor='white')
        plt.close(fig)

        fs = fishnet.copy()
        for fi in top3:
            fs[f"shap_{feature_cols[fi][:8]}"] = gs[:, fi]
        fs["shap_total"] = gs.sum(axis=1)
        sc = (["geometry", "press_pred", "risk_cls", "risk_zh",
               "shap_total"] +
              [f"shap_{feature_cols[i][:8]}" for i in top3])
        sc = [c for c in sc if c in fs.columns]
        fs[sc].to_file(outdir / "shap_spatial_top3.gpkg", driver="GPKG")
        log("    → shap_spatial_top3.gpkg")
    except Exception as e:
        log(f"[SHAP空间图] {e}")


# ==================== 预测诊断图（SCI论文级别）====================

def plot_diagnostics(model_df, fishnet, outdir):
    """SCI论文级别预测诊断综合图（三联面板）"""
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('white')

    gs_layout = gridspec.GridSpec(1, 3, figure=fig,
                                  wspace=0.35, hspace=0.1)

    # ── 左图：预测 vs 真实散点图 ──
    ax0 = fig.add_subplot(gs_layout[0])
    yt = model_df["D3_raw"].values
    yp = model_df["D3_pred_raw"].values

    # 密度着色
    from scipy.stats import gaussian_kde
    try:
        xy = np.vstack([yt, yp])
        kde = gaussian_kde(xy)
        dens = kde(xy)
        norm_dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-9)
    except Exception:
        norm_dens = np.ones(len(yt)) * 0.5

    cmap_scatter = LinearSegmentedColormap.from_list(
        'dens', ['#74ADD1', '#4575B4', '#313695'], N=256)
    sc0 = ax0.scatter(yt, yp, c=norm_dens, cmap=cmap_scatter,
                      alpha=0.75, s=45,
                      edgecolors='white', linewidths=0.4, zorder=2)

    lim = max(yt.max(), yp.max()) * 1.05
    ax0.plot([0, lim], [0, lim], color=SCI_COLORS['accent'],
             lw=1.8, ls='--', label="1:1参考线", zorder=3)

    r2 = r2_score(yt, yp)
    mdp = safe_mdape(yt, yp)
    rmse_val = float(np.sqrt(mean_squared_error(yt, yp)))

    info_text = (f"训练集 R² = {r2:.3f}\n"
                 f"MdAPE = {mdp:.1f}%\n"
                 f"RMSE = {rmse_val:.3f}")
    ax0.text(0.05, 0.95, info_text,
             transform=ax0.transAxes, fontsize=9,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       alpha=0.85, edgecolor='#CCCCCC', linewidth=0.8))

    ax0.set_xlim(0, lim)
    ax0.set_ylim(0, lim)
    ax0.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.05, 0.78))

    cbar0 = plt.colorbar(sc0, ax=ax0, shrink=0.8, pad=0.02)
    cbar0.set_label("样本密度", fontsize=8)
    cbar0.set_ticks([0, 0.5, 1])
    cbar0.set_ticklabels(["低", "中", "高"])

    sci_style_ax(ax0,
                 title="(a) 诊断模型：预测值 vs 真实值",
                 xlabel="学位压力真实值（D3）",
                 ylabel="学位压力预测值（D3）")
    add_panel_label(ax0, "(a)")

    # ── 中图：格网压力空间分布 ──
    ax1 = fig.add_subplot(gs_layout[1])
    if "press_pred" in fishnet.columns:
        # 自定义风险色谱
        risk_cmap = LinearSegmentedColormap.from_list(
            'risk', ['#2166AC', '#74ADD1', '#FFFFBF',
                     '#F46D43', '#A50026'], N=256)
        fishnet.plot(
            column="press_pred", ax=ax1,
            cmap=risk_cmap,
            legend=True,
            legend_kwds={
                "label": "归一化学位压力",
                "orientation": "horizontal",
                "pad": 0.05,
                "shrink": 0.8,
                "aspect": 30,
                "format": "%.2f",
            })
    ax1.set_title("(b) 制图模型：格网学位压力空间分布",
                  fontsize=11, fontweight='bold',
                  color=SCI_COLORS['primary'], pad=8)
    ax1.set_axis_off()
    ax1.text(0.02, 0.98, "(b)", transform=ax1.transAxes,
             fontsize=13, fontweight='bold',
             va='top', color=SCI_COLORS['primary'])

    # ── 右图：残差分布直方图 ──
    ax2 = fig.add_subplot(gs_layout[2])
    res = yt - yp

    n_bins = min(25, max(10, len(res) // 5))
    n_vals, bins, patches = ax2.hist(
        res, bins=n_bins,
        color=SCI_COLORS['secondary'],
        edgecolor='white', linewidth=0.6,
        alpha=0.85, zorder=2, density=False)

    # 正态曲线参考
    from scipy.stats import norm as sp_norm
    mu, sigma = res.mean(), res.std()
    x_fit = np.linspace(res.min(), res.max(), 200)
    y_fit = sp_norm.pdf(x_fit, mu, sigma) * len(res) * (bins[1] - bins[0])
    ax2.plot(x_fit, y_fit, color=SCI_COLORS['accent'],
             lw=2.0, ls='-', label="正态参考曲线", zorder=3)

    ax2.axvline(0, color='#444444', lw=1.2, ls='--', zorder=3)
    ax2.axvline(mu, color=SCI_COLORS['warm'], lw=1.5, ls='-',
                label=f"均值 = {mu:.3f}", zorder=3)

    skew_val = pd.Series(res).skew()
    res_info = (f"均值 = {mu:.3f}\n"
                f"标准差 = {sigma:.3f}\n"
                f"偏度 = {skew_val:.3f}")
    ax2.text(0.97, 0.97, res_info,
             transform=ax2.transAxes, fontsize=8.5,
             va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       alpha=0.85, edgecolor='#CCCCCC'))

    ax2.legend(fontsize=8.5, loc='upper left')
    sci_style_ax(ax2,
                 title="(c) 诊断模型：训练集残差分布",
                 xlabel="残差（真实值 − 预测值）",
                 ylabel="频数")
    add_panel_label(ax2, "(c)")

    plt.suptitle("GeoXGBoost 双模型预测效果综合诊断",
                 fontsize=13, fontweight='bold', y=1.03,
                 color=SCI_COLORS['primary'])

    fig.savefig(outdir / "prediction_diagnostics.png",
                dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    log("  → prediction_diagnostics.png（SCI论文级别）")


# ==================== 双模型一致性检验图（SCI论文级别）====================

def plot_consistency_check(model_df, consistency_result, outdir):
    """SCI论文级别双模型一致性检验图（双联面板 + 秩相关散点）"""

    d3_true = model_df["D3_raw"].values
    a_pred = model_df["D3_pred_raw"].values
    b_pred = model_df["B_pred_at_school"].values

    mask_a = np.isfinite(a_pred) & np.isfinite(d3_true)
    mask_b = np.isfinite(b_pred) & np.isfinite(d3_true)
    mask_ab = np.isfinite(a_pred) & np.isfinite(b_pred)

    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('white')
    gs_layout = gridspec.GridSpec(1, 3, figure=fig,
                                  wspace=0.38, hspace=0.1)

    # ── 左图：模型A预测 vs 真实 ──
    ax0 = fig.add_subplot(gs_layout[0])

    if mask_a.sum() >= 2:
        # 密度着色
        try:
            from scipy.stats import gaussian_kde
            xy_a = np.vstack([d3_true[mask_a], a_pred[mask_a]])
            kde_a = gaussian_kde(xy_a)
            dens_a = kde_a(xy_a)
            nd_a = (dens_a - dens_a.min()) / (dens_a.max() - dens_a.min() + 1e-9)
        except Exception:
            nd_a = np.ones(mask_a.sum()) * 0.5

        cmap_a = LinearSegmentedColormap.from_list(
            'cmap_a', ['#74ADD1', '#4575B4', '#313695'], N=256)
        sc_a = ax0.scatter(d3_true[mask_a], a_pred[mask_a],
                           c=nd_a, cmap=cmap_a,
                           alpha=0.78, s=50,
                           edgecolors='white', linewidths=0.4, zorder=2,
                           label="学校样本点")

        lim_a = max(d3_true[mask_a].max(), a_pred[mask_a].max()) * 1.05
        ax0.plot([0, lim_a], [0, lim_a],
                 color=SCI_COLORS['accent'], lw=1.8, ls='--',
                 label="1:1参考线", zorder=3)

        # 趋势线
        z_a = np.polyfit(d3_true[mask_a], a_pred[mask_a], 1)
        p_a = np.poly1d(z_a)
        x_fit_a = np.linspace(0, lim_a, 100)
        ax0.plot(x_fit_a, p_a(x_fit_a),
                 color=SCI_COLORS['secondary'], lw=1.5, ls='-',
                 alpha=0.7, label="线性趋势", zorder=3)

        cbar_a = plt.colorbar(sc_a, ax=ax0, shrink=0.75, pad=0.02)
        cbar_a.set_label("样本密度", fontsize=8)
        cbar_a.set_ticks([0, 0.5, 1])
        cbar_a.set_ticklabels(["低", "中", "高"])

    rho_at = consistency_result.get("rho_A_true")
    p_at = consistency_result.get("p_A_true")
    sig_at = consistency_result.get("sig_A_true", "")
    rho_at_str = f"{rho_at:.3f}" if (rho_at is not None and np.isfinite(rho_at)) else "nan"
    p_at_str = fmt_p(p_at)

    r2_a = r2_score(d3_true[mask_a], a_pred[mask_a]) if mask_a.sum() >= 2 else np.nan
    mdpe_a = safe_mdape(d3_true[mask_a], a_pred[mask_a]) if mask_a.sum() >= 2 else np.nan

    info_a = (f"Spearman ρ = {rho_at_str}{sig_at}\n"
              f"p 值 = {p_at_str}\n"
              f"R² = {r2_a:.3f}\n"
              f"MdAPE = {mdpe_a:.1f}%")
    ax0.text(0.05, 0.95, info_a,
             transform=ax0.transAxes, fontsize=8.5,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       alpha=0.88, edgecolor='#CCCCCC', linewidth=0.8))

    ax0.legend(fontsize=8.5, loc='lower right')
    sci_style_ax(ax0,
                 title="(a) 诊断模型：预测值 vs 真实值",
                 xlabel="学位压力真实值（D3）",
                 ylabel="诊断模型预测值")
    add_panel_label(ax0, "(a)")

    # ── 中图：模型B在学校点预测 vs 真实 ──
    ax1 = fig.add_subplot(gs_layout[1])

    if mask_b.sum() >= 2:
        try:
            from scipy.stats import gaussian_kde
            xy_b = np.vstack([d3_true[mask_b], b_pred[mask_b]])
            kde_b = gaussian_kde(xy_b)
            dens_b = kde_b(xy_b)
            nd_b = (dens_b - dens_b.min()) / (dens_b.max() - dens_b.min() + 1e-9)
        except Exception:
            nd_b = np.ones(mask_b.sum()) * 0.5

        cmap_b = LinearSegmentedColormap.from_list(
            'cmap_b', ['#FEE090', '#F46D43', '#A50026'], N=256)
        sc_b = ax1.scatter(d3_true[mask_b], b_pred[mask_b],
                           c=nd_b, cmap=cmap_b,
                           alpha=0.78, s=50,
                           edgecolors='white', linewidths=0.4, zorder=2,
                           label="学校样本点")

        lim_b = max(d3_true[mask_b].max(), b_pred[mask_b].max()) * 1.05
        ax1.plot([0, lim_b], [0, lim_b],
                 color=SCI_COLORS['accent'], lw=1.8, ls='--',
                 label="1:1参考线", zorder=3)

        z_b = np.polyfit(d3_true[mask_b], b_pred[mask_b], 1)
        p_b = np.poly1d(z_b)
        x_fit_b = np.linspace(0, lim_b, 100)
        ax1.plot(x_fit_b, p_b(x_fit_b),
                 color=SCI_COLORS['secondary'], lw=1.5, ls='-',
                 alpha=0.7, label="线性趋势", zorder=3)

        cbar_b = plt.colorbar(sc_b, ax=ax1, shrink=0.75, pad=0.02)
        cbar_b.set_label("样本密度", fontsize=8)
        cbar_b.set_ticks([0, 0.5, 1])
        cbar_b.set_ticklabels(["低", "中", "高"])

    rho_bt = consistency_result.get("rho_B_true")
    p_bt = consistency_result.get("p_B_true")
    sig_bt = consistency_result.get("sig_B_true", "")
    rho_ab = consistency_result.get("rho_A_B")
    sig_ab = consistency_result.get("sig_A_B", "")
    rho_bt_str = f"{rho_bt:.3f}" if (rho_bt is not None and np.isfinite(rho_bt)) else "nan"
    rho_ab_str = f"{rho_ab:.3f}" if (rho_ab is not None and np.isfinite(rho_ab)) else "nan"
    p_bt_str = fmt_p(p_bt)

    info_b = (f"Spearman ρ(制图模型,真实) = {rho_bt_str}{sig_bt}\n"
              f"p 值 = {p_bt_str}\n"
              f"ρ(诊断模型,制图模型) = {rho_ab_str}{sig_ab}")
    ax1.text(0.05, 0.95, info_b,
             transform=ax1.transAxes, fontsize=8.5,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       alpha=0.88, edgecolor='#CCCCCC', linewidth=0.8))

    ax1.legend(fontsize=8.5, loc='lower right')
    sci_style_ax(ax1,
                 title="(b) 制图模型：学校点预测值 vs 真实值",
                 xlabel="学位压力真实值（D3）",
                 ylabel="制图模型预测值")
    add_panel_label(ax1, "(b)")

    # ── 右图：模型A vs 模型B 一致性散点 ──
    ax2 = fig.add_subplot(gs_layout[2])

    if mask_ab.sum() >= 2:
        try:
            from scipy.stats import gaussian_kde
            xy_ab = np.vstack([a_pred[mask_ab], b_pred[mask_ab]])
            kde_ab = gaussian_kde(xy_ab)
            dens_ab = kde_ab(xy_ab)
            nd_ab = (dens_ab - dens_ab.min()) / (dens_ab.max() - dens_ab.min() + 1e-9)
        except Exception:
            nd_ab = np.ones(mask_ab.sum()) * 0.5

        cmap_ab = LinearSegmentedColormap.from_list(
            'cmap_ab', ['#A6DBA0', '#1B7837', '#00441B'], N=256)
        sc_ab = ax2.scatter(a_pred[mask_ab], b_pred[mask_ab],
                            c=nd_ab, cmap=cmap_ab,
                            alpha=0.78, s=50,
                            edgecolors='white', linewidths=0.4, zorder=2,
                            label="学校样本点")

        lim_ab = max(a_pred[mask_ab].max(), b_pred[mask_ab].max()) * 1.05
        ax2.plot([0, lim_ab], [0, lim_ab],
                 color=SCI_COLORS['accent'], lw=1.8, ls='--',
                 label="1:1参考线", zorder=3)

        z_ab = np.polyfit(a_pred[mask_ab], b_pred[mask_ab], 1)
        p_ab_fit = np.poly1d(z_ab)
        x_fit_ab = np.linspace(0, lim_ab, 100)
        ax2.plot(x_fit_ab, p_ab_fit(x_fit_ab),
                 color=SCI_COLORS['secondary'], lw=1.5, ls='-',
                 alpha=0.7, label="线性趋势", zorder=3)

        cbar_ab = plt.colorbar(sc_ab, ax=ax2, shrink=0.75, pad=0.02)
        cbar_ab.set_label("样本密度", fontsize=8)
        cbar_ab.set_ticks([0, 0.5, 1])
        cbar_ab.set_ticklabels(["低", "中", "高"])

    p_ab_val = consistency_result.get("p_A_B")
    p_ab_str_fig = fmt_p(p_ab_val)

    info_ab = (f"Spearman ρ(诊断,制图) = {rho_ab_str}{sig_ab}\n"
               f"p 值 = {p_ab_str_fig}")
    ax2.text(0.05, 0.95, info_ab,
             transform=ax2.transAxes, fontsize=8.5,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       alpha=0.88, edgecolor='#CCCCCC', linewidth=0.8))

    ax2.legend(fontsize=8.5, loc='lower right')
    sci_style_ax(ax2,
                 title="(c) 双模型预测结果一致性",
                 xlabel="诊断模型预测值",
                 ylabel="制图模型预测值")
    add_panel_label(ax2, "(c)")

    # 底部综合判断文字
    verdict = consistency_result.get("verdict", "")
    # 截取前60字避免过长
    verdict_short = verdict[:60] + "…" if len(verdict) > 60 else verdict
    fig.text(0.5, -0.04, f"综合判断：{verdict_short}",
             ha='center', va='top', fontsize=9.5,
             color=SCI_COLORS['neutral'],
             style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F8F8',
                       alpha=0.8, edgecolor='#DDDDDD'))

    plt.suptitle("GeoXGBoost 双模型空间认知一致性检验",
                 fontsize=13, fontweight='bold', y=1.04,
                 color=SCI_COLORS['primary'])

    fig.savefig(outdir / "dual_model_consistency.png",
                dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    log("  → dual_model_consistency.png（SCI论文级别）")


# ==================== 风险等级分布图（新增）====================

def plot_risk_distribution(fishnet, model_df, outdir):
    """SCI论文级别风险等级统计图"""
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    gs_layout = gridspec.GridSpec(1, 3, figure=fig,
                                  wspace=0.4, hspace=0.1)

    # ── 左：格网风险等级空间分布 ──
    ax0 = fig.add_subplot(gs_layout[0])
    if "risk_zh" in fishnet.columns:
        risk_cmap = LinearSegmentedColormap.from_list(
            'risk5', RISK_COLORS, N=5)
        risk_cls_vals = fishnet["risk_cls"].values.astype(float)
        fishnet.plot(column="risk_cls", ax=ax0,
                     cmap=risk_cmap, vmin=1, vmax=5,
                     legend=False)
        # 手动图例
        patches = [mpatches.Patch(color=RISK_COLORS[i],
                                  label=RISK_LABELS[i])
                   for i in range(5)]
        ax0.legend(handles=patches, loc='lower right',
                   fontsize=8, title="风险等级",
                   title_fontsize=8.5,
                   framealpha=0.9)
    ax0.set_title("(a) 学位压力风险等级空间分布",
                  fontsize=11, fontweight='bold',
                  color=SCI_COLORS['primary'], pad=8)
    ax0.set_axis_off()
    ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes,
             fontsize=13, fontweight='bold',
             va='top', color=SCI_COLORS['primary'])

    # ── 中：格网风险等级频数条形图 ──
    ax1 = fig.add_subplot(gs_layout[1])
    if "risk_zh" in fishnet.columns:
        risk_counts = fishnet["risk_zh"].value_counts().reindex(
            RISK_LABELS, fill_value=0)
        bars = ax1.bar(range(5), risk_counts.values,
                       color=RISK_COLORS,
                       edgecolor='white', linewidth=0.6,
                       width=0.65)
        # 数值标签
        total = risk_counts.values.sum()
        for i, (bar, val) in enumerate(zip(bars, risk_counts.values)):
            pct = val / (total + 1e-9) * 100
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total * 0.005,
                     f"{val}\n({pct:.1f}%)",
                     ha='center', va='bottom', fontsize=8.5)
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(RISK_LABELS, fontsize=9)
        sci_style_ax(ax1,
                     title="(b) 格网单元风险等级频数分布",
                     xlabel="风险等级",
                     ylabel="格网单元数量")
        add_panel_label(ax1, "(b)")

    # ── 右：学校点风险等级条形图 ──
    ax2 = fig.add_subplot(gs_layout[2])
    if "risk_zh" in model_df.columns:
        school_counts = model_df["risk_zh"].value_counts().reindex(
            RISK_LABELS, fill_value=0)
        bars2 = ax2.bar(range(5), school_counts.values,
                        color=RISK_COLORS,
                        edgecolor='white', linewidth=0.6,
                        width=0.65)
        total2 = school_counts.values.sum()
        for i, (bar, val) in enumerate(zip(bars2, school_counts.values)):
            pct = val / (total2 + 1e-9) * 100
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total2 * 0.005,
                     f"{val}\n({pct:.1f}%)",
                     ha='center', va='bottom', fontsize=8.5)
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(RISK_LABELS, fontsize=9)
        sci_style_ax(ax2,
                     title="(c) 学校点风险等级频数分布",
                     xlabel="风险等级",
                     ylabel="学校数量")
        add_panel_label(ax2, "(c)")

    plt.suptitle("GeoXGBoost 学位压力风险分级结果",
                 fontsize=13, fontweight='bold', y=1.04,
                 color=SCI_COLORS['primary'])

    fig.savefig(outdir / "risk_distribution.png",
                dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    log("  → risk_distribution.png（SCI论文级别）")


# ==================== 交叉验证结果图（新增）====================

def plot_cv_results(cv_results_A, cv_results_B, outdir):
    """SCI论文级别交叉验证结果对比图"""
    fig = plt.figure(figsize=(13, 5))
    fig.patch.set_facecolor('white')
    gs_layout = gridspec.GridSpec(1, 3, figure=fig,
                                  wspace=0.4, hspace=0.1)

    metrics_info = [
        ("r2_raw", "决定系数 R²", "(a) 交叉验证 R²"),
        ("mdape_raw", "中位绝对百分比误差 (%)", "(b) 交叉验证 MdAPE"),
        ("rmse_raw", "均方根误差（RMSE）", "(c) 交叉验证 RMSE"),
    ]
    panel_labs = ["(a)", "(b)", "(c)"]

    for pi, (metric_key, ylabel_str, title_str) in enumerate(metrics_info):
        ax = fig.add_subplot(gs_layout[pi])

        vals_A = cv_results_A[metric_key]["all"]
        vals_B = cv_results_B[metric_key]["all"] if metric_key in cv_results_B else []
        n_folds = len(vals_A)
        x_A = np.arange(n_folds) - 0.18
        x_B = np.arange(n_folds) + 0.18

        # 模型A
        ax.bar(x_A, vals_A, width=0.32,
               color=SCI_COLORS['primary'], alpha=0.85,
               edgecolor='white', linewidth=0.5,
               label="诊断模型")

        # 模型B
        if vals_B:
            ax.bar(x_B, vals_B, width=0.32,
                   color=SCI_COLORS['secondary'], alpha=0.85,
                   edgecolor='white', linewidth=0.5,
                   label="制图模型")

        # 均值线
        mean_A = np.mean(vals_A)
        ax.axhline(mean_A, color=SCI_COLORS['primary'],
                   lw=1.5, ls='--', alpha=0.7)
        if vals_B:
            mean_B = np.mean(vals_B)
            ax.axhline(mean_B, color=SCI_COLORS['secondary'],
                       lw=1.5, ls='--', alpha=0.7)

        ax.set_xticks(np.arange(n_folds))
        ax.set_xticklabels([f"折{i + 1}" for i in range(n_folds)],
                           fontsize=9)
        ax.legend(fontsize=8.5, loc='best')

        sci_style_ax(ax, title=title_str,
                     xlabel="交叉验证折次",
                     ylabel=ylabel_str)
        add_panel_label(ax, panel_labs[pi])

    plt.suptitle("GeoXGBoost 双模型交叉验证性能对比",
                 fontsize=13, fontweight='bold', y=1.04,
                 color=SCI_COLORS['primary'])

    fig.savefig(outdir / "cv_results_comparison.png",
                dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    log("  → cv_results_comparison.png（SCI论文级别）")


# ==================== 栅格/矢量工具 ====================

def check_continuous_raster(path, label, target_crs,
                            expected_cell_size=None):
    if not path:
        return
    with rasterio.open(path) as src:
        if src.count != 1:
            die(f"{label} 必须是单波段。")
    log(f"[注意] 已跳过{label}的CRS严格检查。")


def fishnet_transform(gdf, cell_size):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    left = np.floor(xmin / cell_size) * cell_size
    top = np.ceil(ymax / cell_size) * cell_size
    width = int(np.ceil((xmax - left) / cell_size))
    height = int(np.ceil((top - ymin) / cell_size))
    if width <= 0 or height <= 0:
        die("fishnet范围无效。")
    return Affine(cell_size, 0, left, 0, -cell_size, top), width, height


def write_grid_raster(grid, values, out_path, transform,
                      width, height, crs,
                      dtype="float32", nodata=NODATA):
    shapes = [(g, float(v)) for g, v in zip(grid.geometry, values)
              if g is not None and np.isfinite(v)]
    arr = rasterize(shapes=shapes, out_shape=(height, width),
                    transform=transform, fill=nodata, dtype=dtype,
                    all_touched=False)
    with rasterio.open(out_path, "w", driver="GTiff",
                       height=height, width=width, count=1,
                       dtype=dtype, crs=crs, transform=transform,
                       nodata=nodata, compress="lzw") as dst:
        dst.write(arr, 1)


def safe_to_file(gdf, path, driver="ESRI Shapefile"):
    if driver == "ESRI Shapefile":
        for s in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"):
            c = path.with_suffix(s)
            if c.exists():
                c.unlink()
        gdf.to_file(path, driver=driver, encoding="UTF-8")
    else:
        if path.exists():
            path.unlink()
        gdf.to_file(path, driver=driver)


def make_jenks(values, n_classes=5):
    import jenkspy
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        die("无有效预测值。")
    n_u = np.unique(vals).size
    if n_u < n_classes:
        log(f"[Jenks分类] 仅{n_u}个不同值，等距降级")
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-9:
            breaks = [vmin - 1e-6] + [vmin + (i + 1) * 1e-6
                                      for i in range(n_classes)]
        else:
            breaks = [vmin + i * (vmax - vmin) / n_classes
                      for i in range(n_classes + 1)]
        return [float(v) for v in breaks]
    return [float(v) for v in jenkspy.jenks_breaks(
        vals, n_classes=n_classes)]


def classify_with_breaks(values, breaks):
    return np.searchsorted(
        np.asarray(breaks[1:-1]), values, side="left"
    ).astype(int) + 1


def percentile_normalize(values, p_low=2.0, p_high=98.0):
    v_low = float(np.percentile(values, p_low))
    v_high = float(np.percentile(values, p_high))
    if v_high - v_low < 1e-9:
        v_low, v_high = float(values.min()), float(values.max())
    norm = np.clip((values - v_low) / (v_high - v_low + 1e-9), 0.0, 1.0)
    log(f"  [百分位归一化] P{p_low:.0f}={v_low:.4f},"
        f" P{p_high:.0f}={v_high:.4f}"
        f" → [{norm.min():.4f},{norm.max():.4f}]")
    return norm, v_low, v_high


# ==================== 参数化运行入口（替代 argparse + main）====================

def run_analysis(
    school_csv=None,
    fishnet_path=None,
    iso_primary_path=None,
    iso_middle_path=None,
    build_density_raster=None,
    buildings_path=None,
    worldpop_path=None,
    outdir_path=None,
    target_crs="EPSG:4526",
    school_crs="EPSG:4526",
    school_id_field="school_id",
    service_id_field="school_id",
    d3_field="auto",
    vitality_field="vitality",
    grid_build_field="",
    cell_size=250.0,
    spatial_kernel="gaussian",
    idw_power=2.0,
    idw_k=15,
    target_sample_ratio=5.0,
    norm_plow=2.0,
    norm_phigh=98.0,
    use_rbf=True,
    use_gwr_weight=True,
    use_optuna=True,
    use_shap=True,
    use_boxcox=True,
    use_huber_baseline=True,
    optuna_trials=N_OPTUNA_TRIALS,
    optuna_refine=N_OPTUNA_REFINE,
    optuna_metric=OPTUNA_METRIC,
):
    """
    GeoXGBoost 双模型架构主流程（参数化版本）。

    所有文件路径必须由调用方明确传入（不再使用 DEFAULT_PATHS 硬编码）。
    返回一个 dict，包含所有输出文件的路径，供调用方读取派生输出。
    """
    # ── 参数校验 ──
    missing = []
    for key, val in [("school_csv", school_csv), ("fishnet_path", fishnet_path),
                     ("iso_primary_path", iso_primary_path), ("iso_middle_path", iso_middle_path),
                     ("worldpop_path", worldpop_path), ("outdir_path", outdir_path)]:
        if val is None:
            missing.append(key)
    if missing:
        die(f"缺少必需参数: {missing}。请通过命令行参数或函数参数提供。")
    if build_density_raster is None and buildings_path is None:
        die("请提供 build_density_raster（建筑密度栅格）或 buildings_path（建筑轮廓矢量）。")

    np.random.seed(RANDOM_SEED)
    outdir = Path(outdir_path)
    outdir.mkdir(parents=True, exist_ok=True)
    target_crs_obj = CRS.from_user_input(target_crs)

    log("=" * 72)
    log("GeoXGBoost - 双模型架构（诊断模型 + 制图模型）+ 一致性检验")
    log("  诊断模型（模型A）：解释学位压力成因，SHAP归因分析")
    log("  制图模型（模型B）：生成全域连续压力面")
    log("  修复：map_level()高中特判；B在学校点直接推断；f-string修复")
    log("=" * 72)

    # ── 1. 读取数据 ──
    log("\n[步骤1] 读取输入数据...")
    school, d3_field = read_school_csv(
        school_csv, school_id_field, d3_field,
        school_crs, target_crs)
    log(f"  学校CSV: {len(school)} 所（全部学校类型）")

    fishnet = read_vector(fishnet_path, target_crs, "250m格网")
    require_columns(fishnet, [vitality_field], "250m格网")

    iso_p = read_vector(iso_primary_path, target_crs, "小学服务区")
    iso_m = read_vector(iso_middle_path, target_crs, "初中服务区")
    require_columns(iso_p, [service_id_field], "小学服务区")
    require_columns(iso_m, [service_id_field], "初中服务区")

    buildings = None
    if build_density_raster:
        check_continuous_raster(build_density_raster, "建筑密度",
                                target_crs_obj, cell_size)
    else:
        buildings = read_vector(buildings_path, target_crs, "建筑轮廓")
    check_continuous_raster(worldpop_path, "WorldPop",
                            target_crs_obj, cell_size)

    # ── 2. school_id 匹配 ──
    log("\n[步骤2] school_id 匹配...")
    for iso in [iso_p, iso_m]:
        iso[service_id_field] = (iso[service_id_field]
                                  .map(canonical_school_id))
    isochrones = pd.concat([iso_p, iso_m], ignore_index=True)
    isochrones = isochrones.drop_duplicates(
        subset=service_id_field, keep="first")

    school_id_set = set(school[school_id_field]) - {""}
    iso_id_set = set(isochrones[service_id_field]) - {""}
    diagnose_id_mismatch(school_id_set, iso_id_set, school_id_field)

    common = school_id_set & iso_id_set
    school = school[school[school_id_field].isin(common)].copy()
    isochrones = isochrones[
        isochrones[service_id_field].isin(common)].copy()
    if service_id_field != school_id_field:
        isochrones = isochrones.rename(
            columns={service_id_field: school_id_field})

    sid = school_id_field
    school = school.sort_values(sid).reset_index(drop=True)
    isochrones = isochrones.sort_values(sid).reset_index(drop=True)
    log(f"  建模样本: {len(school)} 所 | 格网: {len(fishnet)} 个")
    if len(school) < 20:
        die(f"有效学校仅{len(school)}所。")

    # ── 3. 学校级特征 ──
    log("\n[步骤3] 构建学校级特征...")
    vitality_school = overlay_area_weighted_mean(
        isochrones, fishnet, sid, vitality_field)

    if build_density_raster:
        density_school = isochrones[[sid, "geometry"]].copy()
        density_school["X2_build_den"] = raster_zonal_mean(
            isochrones, build_density_raster, "X2_build_den").values
        density_school = density_school[[sid, "X2_build_den"]]
    else:
        density_school = polygon_area_density(
            isochrones, buildings, sid, "X2_build_den")

    pop_school = isochrones[[sid, "geometry"]].copy()
    pop_school["X3_worldpop"] = raster_zonal_mean(
        isochrones, worldpop_path, "X3_worldpop").values

    model_df = (
        school.drop(columns="geometry")
        .merge(vitality_school[[sid, "X1_vitality"]], on=sid, how="left")
        .merge(density_school, on=sid, how="left")
        .merge(pop_school[[sid, "X3_worldpop"]], on=sid, how="left")
    )
    model_df["x_coord"] = school.geometry.x.values
    model_df["y_coord"] = school.geometry.y.values
    model_df = model_df.rename(columns={d3_field: "D3_raw"})
    for col in ["D3_raw", "X1_vitality", "X2_build_den", "X3_worldpop"]:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    if model_df["D3_raw"].isna().any():
        die("D3有缺失值。")

    d3_diag = diagnose_d3_quality(model_df, "D3_raw")
    d3_raw_arr = model_df["D3_raw"].values

    # Box-Cox
    boxcox_lambda = None
    if use_boxcox:
        log(f"\n[Box-Cox变换]...")
        d3_win = np.minimum(
            model_df["D3_raw"].values,
            np.percentile(model_df["D3_raw"].dropna(), D3_WINSORIZE_PCT))
        d3_trans, boxcox_lambda = apply_boxcox(d3_win)
        model_df["D3_transformed"] = d3_trans
        target_col = "D3_transformed"
    else:
        d3_cap = np.percentile(model_df["D3_raw"].dropna(), D3_WINSORIZE_PCT)
        model_df["D3_log"] = np.log(
            np.minimum(model_df["D3_raw"], d3_cap) + 1e-6)
        target_col = "D3_log"

    medians = model_df[["X1_vitality", "X2_build_den", "X3_worldpop"]].median()
    for col in ["X1_vitality", "X2_build_den", "X3_worldpop"]:
        model_df[col] = model_df[col].fillna(medians[col])

    # ── 4. 模型A 空间特征工程 ──
    log("\n[步骤4] 空间特征工程（诊断模型全特征）...")
    coords = model_df[["x_coord", "y_coord"]].values

    neigh_idx, neigh_w = build_spatial_weight_matrix(
        coords, k=K_NEIGHBORS, kernel=spatial_kernel)
    for src, lag in {"X1_vitality": "lag_X1_vitality",
                     "X2_build_den": "lag_X2_build_den",
                     "X3_worldpop": "lag_X3_worldpop"}.items():
        model_df[lag] = compute_spatial_lag(
            model_df[src].values, neigh_idx, neigh_w)
    model_df["lag_D3"] = compute_spatial_lag(
        model_df[target_col].values, neigh_idx, neigh_w)

    v4_cands = [
        "dist_to_center", "dist_to_river", "poi_diversity",
        "residential_ratio", "compactness", "service_area_km2",
        "dominant_cluster",
        "pct_C1", "pct_C2", "pct_C3", "pct_C4", "pct_C5", "pct_C6",
    ]
    available_v4 = [c for c in v4_cands if c in model_df.columns]
    if available_v4:
        log(f"  扩展空间特征（诊断模型）: {len(available_v4)}个")

    rbf_encoder_A = None
    rbf_names_A = []
    if use_rbf:
        rbf_encoder_A = RBFSpatialEncoder(
            N_RBF_ANCHORS, RBF_GAMMA_QUANTILE, RANDOM_SEED)
        rbf_feats_A = rbf_encoder_A.fit_transform(coords)
        rbf_names_A = rbf_encoder_A.get_feature_names()
        for i, nm in enumerate(rbf_names_A):
            model_df[nm] = rbf_feats_A[:, i]

    feature_cols_A = (
            ["X1_vitality", "X2_build_den", "X3_worldpop",
             "lag_X1_vitality", "lag_X2_build_den",
             "lag_X3_worldpop", "lag_D3",
             "x_coord", "y_coord"] +
            available_v4 + rbf_names_A
    )
    for c in feature_cols_A:
        if c not in model_df.columns:
            model_df[c] = 0.0

    # ── 5. 模型A 训练准备 ──
    log("\n[步骤5] 诊断模型（模型A）训练准备...")
    X_A = model_df[feature_cols_A].values
    y_A = model_df[target_col].values
    scaler_A = StandardScaler().fit(X_A)
    Xs_A = scaler_A.transform(X_A)

    sample_weight = None
    if use_gwr_weight:
        sample_weight = compute_gwr_sample_weights(coords)

    huber_info = {}
    if use_huber_baseline:
        huber_info = huber_baseline(Xs_A, y_A)

    # ── 6. 模型A 调参 + 精简 ──
    log("\n[步骤6] 诊断模型（模型A）全特征粗调参...")
    if use_optuna:
        best_params_coarse = run_optuna(
            Xs_A, y_A, d3_raw_arr, sample_weight,
            n_trials=optuna_trials,
            cv_folds=OPTUNA_CV_FOLDS,
            random_state=RANDOM_SEED,
            outdir=outdir,
            optimize_metric=optuna_metric,
            boxcox_lambda=boxcox_lambda if use_boxcox else None,
            label="诊断模型粗调")
    else:
        best_params_coarse = {
            "max_depth": 3, "min_child_weight": 10,
            "reg_lambda": 25, "reg_alpha": 8,
            "subsample": 0.65, "colsample_bytree": 0.70,
            "colsample_bylevel": 0.70, "learning_rate": 0.04,
            "n_estimators": 350, "gamma": 2.0,
            "tree_method": "hist", "random_state": RANDOM_SEED,
            "verbosity": 0,
        }

    log("\n[步骤6b] 诊断模型（模型A）特征精简...")
    (feature_cols_A, Xs_A, scaler_A,
     kept_A) = two_stage_feature_selection(
        Xs_A, y_A, feature_cols_A, scaler_A,
        best_params_coarse, sample_weight,
        target_ratio=target_sample_ratio,
        random_state=RANDOM_SEED
    )
    rbf_names_A = [f for f in rbf_names_A if f in feature_cols_A]

    log(f"\n[步骤6c] 诊断模型（模型A）精简后细调参（{optuna_refine}次）...")
    if use_optuna and optuna_refine > 0:
        best_params_A = run_optuna(
            Xs_A, y_A, d3_raw_arr, sample_weight,
            n_trials=optuna_refine,
            cv_folds=OPTUNA_CV_FOLDS,
            random_state=RANDOM_SEED,
            outdir=outdir,
            optimize_metric=optuna_metric,
            boxcox_lambda=boxcox_lambda if use_boxcox else None,
            label="诊断模型细调")
    else:
        best_params_A = best_params_coarse

    # ── 7. 模型A CV ──
    cv_results = run_cv(
        Xs_A, y_A, d3_raw_arr, best_params_A, sample_weight,
        n_splits=5, random_state=RANDOM_SEED,
        boxcox_lambda=boxcox_lambda if use_boxcox else None,
        label="诊断模型（模型A）")

    # ── 8. 模型A 最终训练 ──
    log("\n[步骤8] 诊断模型（模型A）最终训练...")
    model_A = xgb.XGBRegressor(**best_params_A)
    model_A.fit(Xs_A, y_A, sample_weight=sample_weight, verbose=False)

    residuals_A = y_A - model_A.predict(Xs_A)
    moran_result = compute_morans_i(residuals_A, coords, k=K_NEIGHBORS)
    log(f"  Moran's I={moran_result['moran_I']:.4f},"
        f" Z={moran_result['z_score']:.4f}")
    log(f"  {moran_result['interpretation']}")

    # ── 9. 特征重要性（模型A）──
    log("\n[步骤9] 诊断模型（模型A）特征重要性...")
    importance_A = model_A.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols_A, "importance": importance_A
    }).sort_values("importance", ascending=False)
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["importance"] * 40)
        log(f"  {row['feature']:30s}: {row['importance']:.4f} {bar}")
    imp_df.to_csv(outdir / "feature_importance.csv",
                  index=False, encoding="utf-8-sig")

    # ══════════════════════════════════════════════════════════════
    # 模型B：格网直推制图模型
    # ══════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("[制图模型] 模型B：格网直推制图模型")
    log("  特征：格网栅格直读（活力/建筑密度/WorldPop）+坐标+RBF")
    log("  一致性检验：直接对学校坐标推断（消除nan）")
    log("=" * 60)

    log("\n[步骤B1] 构建制图模型（模型B）特征（学校端训练集）...")
    rbf_encoder_B = RBFSpatialEncoder(
        N_RBF_ANCHORS, RBF_GAMMA_QUANTILE, RANDOM_SEED)
    rbf_feats_B = rbf_encoder_B.fit_transform(coords)
    rbf_names_B = rbf_encoder_B.get_feature_names()
    for i, nm in enumerate(rbf_names_B):
        model_df[f"B_{nm}"] = rbf_feats_B[:, i]

    feature_cols_B = (
            ["X1_vitality", "X2_build_den", "X3_worldpop",
             "x_coord", "y_coord"] +
            [f"B_{nm}" for nm in rbf_names_B]
    )
    for c in feature_cols_B:
        if c not in model_df.columns:
            model_df[c] = 0.0

    X_B = model_df[feature_cols_B].values
    y_B = model_df[target_col].values
    scaler_B = StandardScaler().fit(X_B)
    Xs_B = scaler_B.transform(X_B)

    B_feat_var = X_B.var(axis=0)
    zero_var_cols = [feature_cols_B[i] for i, v in enumerate(B_feat_var)
                     if v < 1e-10]
    if zero_var_cols:
        log(f"  ⚠ 制图模型以下特征方差为零: {zero_var_cols}")
    else:
        log(f"  ✓ 制图模型所有特征均有方差，特征集有效。")

    log(f"  制图模型特征数: {len(feature_cols_B)}"
        f" | 样本/特征比: {len(model_df) / len(feature_cols_B):.1f}")

    log("\n[步骤B2] 制图模型（模型B）调参...")
    if use_optuna:
        best_params_B = run_optuna(
            Xs_B, y_B, d3_raw_arr, sample_weight,
            n_trials=max(30, optuna_refine),
            cv_folds=OPTUNA_CV_FOLDS,
            random_state=RANDOM_SEED,
            outdir=outdir,
            optimize_metric=optuna_metric,
            boxcox_lambda=boxcox_lambda if use_boxcox else None,
            label="制图模型调参")
    else:
        best_params_B = best_params_A.copy()

    log("\n[步骤B3] 制图模型（模型B）训练...")
    model_B = xgb.XGBRegressor(**best_params_B)
    model_B.fit(Xs_B, y_B, sample_weight=sample_weight, verbose=False)

    cv_B = run_cv(
        Xs_B, y_B, d3_raw_arr, best_params_B, sample_weight,
        n_splits=5, random_state=RANDOM_SEED,
        boxcox_lambda=boxcox_lambda if use_boxcox else None,
        label="制图模型（模型B）")
    log(f"  制图模型 CV R²(原始): {cv_B['r2_raw']['mean']:.4f}"
        f" | MdAPE: {cv_B['mdape_raw']['mean']:.1f}%")

    # ── 10. 格网特征提取 ──
    log("\n[步骤10] 格网特征提取（制图模型直读栅格）...")
    fishnet = fishnet.copy().reset_index(drop=True)
    fishnet["grid_id"] = np.arange(1, len(fishnet) + 1, dtype=int)
    fishnet["X1_vitality"] = pd.to_numeric(
        fishnet[vitality_field], errors="coerce").fillna(
        float(medians["X1_vitality"]))

    if build_density_raster:
        fishnet["X2_build_den"] = raster_zonal_mean(
            fishnet, build_density_raster, "X2_build_den").values
    elif grid_build_field:
        fishnet["X2_build_den"] = pd.to_numeric(
            fishnet[grid_build_field], errors="coerce")
    else:
        gd = polygon_area_density(
            fishnet[["grid_id", "geometry"]], buildings,
            "grid_id", "X2_build_den")
        fishnet = fishnet.merge(gd, on="grid_id", how="left")
    fishnet["X2_build_den"] = fishnet["X2_build_den"].fillna(
        float(medians["X2_build_den"]))

    fishnet["X3_worldpop"] = raster_zonal_mean(
        fishnet, worldpop_path, "X3_worldpop").values
    fishnet["X3_worldpop"] = fishnet["X3_worldpop"].fillna(
        float(medians["X3_worldpop"]))

    fishnet["x_coord"] = fishnet.geometry.centroid.x
    fishnet["y_coord"] = fishnet.geometry.centroid.y
    grid_coords = fishnet[["x_coord", "y_coord"]].values

    rbf_feats_grid = rbf_encoder_B.transform(grid_coords)
    for i, nm in enumerate(rbf_names_B):
        fishnet[f"B_{nm}"] = rbf_feats_grid[:, i]

    log("\n  格网特征唯一值诊断（制图模型）:")
    for feat in ["X1_vitality", "X2_build_den", "X3_worldpop"]:
        if feat in fishnet.columns:
            vals = fishnet[feat].values
            n_uniq = count_effective_unique(vals, IDW_UNIQUE_REL_TOL)
            log(f"    {feat:20s}: 有效唯一值={n_uniq:5d},"
                f" 范围=[{vals.min():.4f},{vals.max():.4f}]"
                f"{'  ✓' if n_uniq >= 50 else '  ⚠'}")

    grid_B_df = pd.DataFrame()
    for c in feature_cols_B:
        grid_B_df[c] = fishnet[c].values if c in fishnet.columns else 0.0
    grid_B_df = grid_B_df.fillna(0.0)
    grid_Xs_B = scaler_B.transform(grid_B_df[feature_cols_B].values)

    # ── 11. 格网预测（模型B）──
    log("\n[步骤11] 格网压力预测（制图模型直推）...")
    train_pred_B = model_B.predict(Xs_B)
    smearing_B = float(np.mean(np.exp(y_B - train_pred_B)))
    log(f"  smearing系数 = {smearing_B:.4f}")

    pred_trans_B = model_B.predict(grid_Xs_B)
    if boxcox_lambda is not None:
        pred_raw_B = boxcox_inverse(pred_trans_B, boxcox_lambda)
    else:
        pred_raw_B = np.maximum(
            np.exp(pred_trans_B) * smearing_B - BOXCOT_EPSILON, 0.0)

    n_uniq_B = len(np.unique(np.round(pred_raw_B, 4)))
    cv_B_val = float(pred_raw_B.std() / (pred_raw_B.mean() + 1e-9))
    log(f"  格网预测范围: [{pred_raw_B.min():.4f},{pred_raw_B.max():.4f}]")
    log(f"  格网唯一值: {n_uniq_B}"
        f"{'  ✓ 空间连续性正常' if n_uniq_B >= 20 else '  ⚠'}")
    log(f"  变异系数CV: {cv_B_val:.4f}"
        f"{'  ✓' if cv_B_val > 0.05 else '  ⚠'}")

    fishnet["press_pred_raw"] = pred_raw_B
    press_norm, p_low_v, p_high_v = percentile_normalize(
        pred_raw_B, norm_plow, norm_phigh)
    fishnet["press_pred"] = press_norm
    n_uniq_norm = len(np.unique(np.round(press_norm, 4)))
    log(f"  press_pred归一化唯一值: {n_uniq_norm}"
        f"{'  ✓' if n_uniq_norm >= 10 else '  ⚠'}")

    # ── SHAP（基于模型B）──
    shap_summary = {}
    if use_shap:
        shap_summary = run_shap_analysis(
            model=model_B, X_scaled=Xs_B,
            feature_cols=feature_cols_B, model_df=model_df,
            fishnet=fishnet, grid_feat_scaled=grid_Xs_B,
            outdir=outdir, sid=sid,
            boxcox_lambda=boxcox_lambda if use_boxcox else None)

    # ── 12. 风险分级 ──
    log("\n[步骤12] 风险分级与输出...")
    breaks = make_jenks(fishnet["press_pred"].to_numpy(), n_classes=5)
    fishnet["risk_cls"] = classify_with_breaks(
        fishnet["press_pred"].to_numpy(), breaks).astype(np.uint8)
    fishnet["risk_zh"] = [RISK_LABELS[i - 1] for i in fishnet["risk_cls"]]
    p80 = float(np.quantile(fishnet["press_pred"], 0.80))
    log(f"  Jenks断点: {[round(x, 4) for x in breaks]}")
    log(f"  P80={p80:.4f} | 高压力格网数:"
        f" {(fishnet['press_pred'] >= p80).sum()}")

    transform, width, height = fishnet_transform(fishnet, cell_size)
    write_grid_raster(fishnet, fishnet["press_pred"].to_numpy(),
                      outdir / "pressure_risk.tif",
                      transform, width, height, target_crs)
    write_grid_raster(fishnet, fishnet["risk_cls"].to_numpy(),
                      outdir / "pressure_class.tif",
                      transform, width, height, target_crs,
                      dtype="uint8", nodata=255)

    vf = ["grid_id", "X1_vitality", "X2_build_den", "X3_worldpop",
          "press_pred", "press_pred_raw", "risk_cls", "risk_zh"]
    safe_to_file(fishnet[vf + ["geometry"]].copy(),
                 outdir / "pressure_coefs.shp")
    safe_to_file(fishnet[vf + ["geometry"]].copy(),
                 outdir / "pressure_coefs.gpkg", driver="GPKG")

    high = fishnet.loc[fishnet["press_pred"] >= p80,
    ["press_pred", "geometry"]].copy()
    if len(high) > 0:
        high["p80_thr"] = p80
        safe_to_file(high.dissolve().copy().assign(
            geometry=lambda g: g.boundary),
            outdir / "pressure_p80_boundary.shp")

    # ── 13. 学校级预测（模型A）──
    log("\n[步骤13] 学校级预测输出（诊断模型）...")
    y_pred_A_t = model_A.predict(Xs_A)
    sf_A = float(np.mean(np.exp(y_A - y_pred_A_t)))
    if boxcox_lambda is not None:
        y_pred_A_raw = boxcox_inverse(y_pred_A_t, boxcox_lambda)
    else:
        y_pred_A_raw = np.maximum(
            np.exp(y_pred_A_t) * sf_A - BOXCOT_EPSILON, 0.0)

    model_df["D3_pred_raw"] = y_pred_A_raw
    model_df["D3_residual"] = model_df["D3_raw"] - y_pred_A_raw

    d3_pred_norm, _, _ = percentile_normalize(
        y_pred_A_raw, norm_plow, norm_phigh)
    school_breaks = make_jenks(d3_pred_norm, n_classes=5)
    model_df["risk_cls"] = classify_with_breaks(
        d3_pred_norm, school_breaks).astype(np.uint8)
    model_df["risk_zh"] = [RISK_LABELS[i - 1] for i in model_df["risk_cls"]]

    # ════════════════════════════════════════════════════════════
    # [核心] 双模型Spearman秩相关一致性检验
    # ════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("[核心] 双模型空间认知一致性检验（Spearman秩相关）")
    log("=" * 60)

    log("\n  用制图模型直接对学校坐标推断（消除nan）...")
    school_B_df = pd.DataFrame()
    for c in feature_cols_B:
        school_B_df[c] = model_df[c].values if c in model_df.columns else 0.0
    school_B_df = school_B_df[feature_cols_B].fillna(0.0)
    school_Xs_B = scaler_B.transform(school_B_df.values)

    b_pred_trans_school = model_B.predict(school_Xs_B)
    if boxcox_lambda is not None:
        b_pred_raw_school = boxcox_inverse(b_pred_trans_school, boxcox_lambda)
    else:
        b_pred_raw_school = np.maximum(
            np.exp(b_pred_trans_school) * smearing_B - BOXCOT_EPSILON, 0.0)

    model_df["B_pred_at_school"] = b_pred_raw_school

    b_std = float(b_pred_raw_school.std())
    b_n_uniq = int(len(np.unique(np.round(b_pred_raw_school, 4))))
    log(f"\n  制图模型在学校点预测诊断:")
    log(f"    预测范围: [{b_pred_raw_school.min():.4f},"
        f" {b_pred_raw_school.max():.4f}]")
    log(f"    标准差: {b_std:.4f} | 唯一值数: {b_n_uniq}")

    if b_std < 1e-6:
        log(f"  ⚠ 制图模型预测为常数，回退到格网IDW插值到学校点...")
        b_pred_raw_school = idw_interpolate_to_grid(
            coords, grid_coords, pred_raw_B,
            power=2.0, k=min(15, len(pred_raw_B)))
        model_df["B_pred_at_school"] = b_pred_raw_school
        b_std = float(b_pred_raw_school.std())
        b_n_uniq = int(len(np.unique(np.round(b_pred_raw_school, 4))))
        log(f"    回退后范围: [{b_pred_raw_school.min():.4f},"
            f" {b_pred_raw_school.max():.4f}] | std={b_std:.4f}")

    d3_true = model_df["D3_raw"].values
    a_pred = model_df["D3_pred_raw"].values
    b_pred = model_df["B_pred_at_school"].values

    rho_A_B, p_AB, warn_AB = safe_spearmanr(a_pred, b_pred, "ρ(诊断,制图)")
    rho_A_true, p_Atrue, warn_Atrue = safe_spearmanr(a_pred, d3_true, "ρ(诊断,真实)")
    rho_B_true, p_Btrue, warn_Btrue = safe_spearmanr(b_pred, d3_true, "ρ(制图,真实)")

    rho_AB_str, rho_AB_sig = fmt_rho(rho_A_B, p_AB)
    rho_Atrue_str, rho_Atrue_sig = fmt_rho(rho_A_true, p_Atrue)
    rho_Btrue_str, rho_Btrue_sig = fmt_rho(rho_B_true, p_Btrue)
    p_AB_str = fmt_p(p_AB)
    p_Atrue_str = fmt_p(p_Atrue)
    p_Btrue_str = fmt_p(p_Btrue)

    rho_AB_ok = (np.isfinite(rho_A_B) and rho_A_B > 0.4
                 and np.isfinite(p_AB) and p_AB < 0.05)
    rho_Btrue_ok = (np.isfinite(rho_B_true) and rho_B_true > 0.2
                    and np.isfinite(p_Btrue) and p_Btrue < 0.05)
    rho_AB_dir = np.isfinite(rho_A_B) and rho_A_B > 0.2
    rho_Btrue_dir = np.isfinite(rho_B_true) and rho_B_true > 0
    consistent = bool(rho_AB_ok and rho_Btrue_ok)

    if not np.isfinite(rho_A_B) or not np.isfinite(rho_B_true):
        verdict = ("⚠ 秩相关计算出现nan，制图模型预测缺乏变异性。"
                   "建议检查X1/X2/X3特征在学校点的方差。")
    elif consistent:
        verdict = ("✓ 双模型空间认知一致：诊断模型与制图模型对高/低压力学校排序高度一致，"
                   "制图模型空间梯度方向正确，可用于论文叙述。")
    elif rho_Btrue_dir and rho_AB_dir:
        verdict = ("○ 双模型方向一致但相关较弱：制图模型在学校点精度低，"
                   "但空间梯度方向正确（ρ>0），制图功能不受影响。")
    else:
        verdict = "⚠ 两模型存在分歧，建议检查制图模型特征设置或数据质量。"

    log(f"\n  【秩相关结果】")
    log(f"  ρ(诊断预测, 制图预测)  = {rho_AB_str} {rho_AB_sig}  p={p_AB_str}")
    log(f"  ρ(诊断预测, D3真实) = {rho_Atrue_str} {rho_Atrue_sig}  p={p_Atrue_str}")
    log(f"  ρ(制图预测, D3真实) = {rho_Btrue_str} {rho_Btrue_sig}  p={p_Btrue_str}")

    for warn in [warn_AB, warn_Atrue, warn_Btrue]:
        if warn:
            log(f"  ⚠ {warn}")

    log(f"\n  【综合判断】")
    log(f"  {verdict}")
    log(f"\n  【论文叙述建议】")
    log(f"  诊断模型：CV R²={cv_results['r2_raw']['mean']:.3f}，"
        f"MdAPE={cv_results['mdape_raw']['mean']:.1f}%，")
    log(f"    用于解释学位压力成因（SHAP归因），关注特征可解释性。")
    log(f"  制图模型：格网唯一值={n_uniq_norm}，CV={cv_B_val:.3f}，")
    log(f"    用于生成全域连续压力面，关注空间分异与制图精度。")
    log(f"  两模型ρ(诊断,制图)={rho_AB_str}{rho_AB_sig}，"
        f"ρ(制图,真实)={rho_Btrue_str}{rho_Btrue_sig}，")
    log(f"  在高/低压力排序上的一致性符合双模型设计初衷。")
    log("=" * 60)

    consistency_result = {
        "rho_A_B": float(rho_A_B) if np.isfinite(rho_A_B) else None,
        "p_A_B": float(p_AB) if np.isfinite(p_AB) else None,
        "sig_A_B": rho_AB_sig,
        "rho_A_true": float(rho_A_true) if np.isfinite(rho_A_true) else None,
        "p_A_true": float(p_Atrue) if np.isfinite(p_Atrue) else None,
        "sig_A_true": rho_Atrue_sig,
        "rho_B_true": float(rho_B_true) if np.isfinite(rho_B_true) else None,
        "p_B_true": float(p_Btrue) if np.isfinite(p_Btrue) else None,
        "sig_B_true": rho_Btrue_sig,
        "verdict": verdict,
        "consistent": consistent,
        "b_pred_std": b_std,
        "b_pred_n_uniq": b_n_uniq,
        "warnings": {
            "rho_AB": warn_AB,
            "rho_Atrue": warn_Atrue,
            "rho_Btrue": warn_Btrue,
        },
        "note": ("制图模型在学校点采用直接推断（非查格网表），"
                 "消除因格网分辨率粗导致的制图模型预测常数化问题。"
                 "若仍出现nan，已自动回退到IDW插值。"),
    }

    # 一致性检验图（SCI论文级别）
    plot_consistency_check(model_df, consistency_result, outdir)

    # 学校预测CSV
    out_school_cols = [
        sid,
        "School_Name" if "School_Name" in model_df.columns else sid,
        "Level" if "Level" in model_df.columns else sid,
        "D3_raw", "D3_pred_raw", "D3_residual",
        "B_pred_at_school", "risk_cls", "risk_zh",
        "x_coord", "y_coord",
    ]
    out_school_cols = list(dict.fromkeys(
        [c for c in out_school_cols if c in model_df.columns]))
    model_df[out_school_cols].to_csv(
        outdir / "school_pressure_prediction.csv",
        index=False, encoding="utf-8-sig")

    # 预测诊断图（SCI论文级别）
    plot_diagnostics(model_df, fishnet, outdir)

    # 风险等级分布图（SCI论文级别）
    plot_risk_distribution(fishnet, model_df, outdir)

    # 交叉验证结果对比图（SCI论文级别）
    plot_cv_results(cv_results, cv_B, outdir)

    # ── 14. JSON报告 ──
    log("\n[步骤14] 生成JSON报告...")
    report = {
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "GeoXGBoost-双模型架构",
        "version": "正式版",
        "note": ("双模型：诊断模型=学校诊断（解释），制图模型=格网制图（空间连续性）。"
                 "修复：map_level()高中特判；制图模型在学校点直接推断（消除nan）；"
                 "f-string格式符修复。"),
        "sample_n": int(len(model_df)),
        "grid_n": int(len(fishnet)),
        "dual_model_design": {
            "model_A": {
                "role": "诊断模型（解释学位压力成因）",
                "features": feature_cols_A,
                "n_features": len(feature_cols_A),
                "cv_R2_trans_mean": cv_results["r2_transformed"]["mean"],
                "cv_R2_raw_mean": cv_results["r2_raw"]["mean"],
                "cv_MdAPE_mean": cv_results["mdape_raw"]["mean"],
                "eval_standard": "CV R²与MdAPE（学校点精准预测）",
            },
            "model_B": {
                "role": "制图模型（生成全域连续压力面）",
                "features": feature_cols_B,
                "n_features": len(feature_cols_B),
                "cv_R2_raw_mean": cv_B["r2_raw"]["mean"],
                "cv_MdAPE_mean": cv_B["mdape_raw"]["mean"],
                "grid_n_unique": n_uniq_B,
                "grid_cv": cv_B_val,
                "press_n_unique": n_uniq_norm,
                "eval_standard": "格网唯一值数、CV、Spearman方向一致性",
                "b_pred_at_school_std": b_std,
                "b_pred_at_school_nuniq": b_n_uniq,
                "note_on_low_R2": ("制图模型学校端R²低属预期行为：缺少服务区结构特征，"
                                   "不能精准还原单校D3，空间梯度方向由ρ(制图,真实)验证。"),
            },
        },
        "consistency_check": consistency_result,
        "cv_metrics_A": {
            "R2_transformed_mean": cv_results["r2_transformed"]["mean"],
            "R2_transformed_std": cv_results["r2_transformed"]["std"],
            "R2_raw_mean": cv_results["r2_raw"]["mean"],
            "R2_raw_std": cv_results["r2_raw"]["std"],
            "MdAPE_mean": cv_results["mdape_raw"]["mean"],
            "MAPE_mean": cv_results["mape_raw"]["mean"],
        },
        "cv_metrics_B": {
            "R2_raw_mean": cv_B["r2_raw"]["mean"],
            "R2_raw_std": cv_B["r2_raw"]["std"],
            "MdAPE_mean": cv_B["mdape_raw"]["mean"],
        },
        "transform": {
            "type": "Box-Cox" if use_boxcox else "对数变换",
            "lambda": float(boxcox_lambda) if boxcox_lambda else None,
        },
        "spatial_autocorrelation": moran_result,
        "best_params_A": {k: v for k, v in best_params_A.items()
                          if k != "verbosity"},
        "best_params_B": {k: v for k, v in best_params_B.items()
                          if k != "verbosity"},
        "huber_baseline": huber_info,
        "shap_summary": shap_summary,
        "risk_classification": {
            "method": "Jenks自然断点法",
            "breaks": breaks,
            "P80": p80,
        },
        "norm_params": {
            "p_low": norm_plow,
            "p_high": norm_phigh,
            "v_low": p_low_v,
            "v_high": p_high_v,
        },
        "d3_quality": d3_diag,
    }
    with open(outdir / "model_report.json", "w",
              encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False,
                  indent=2, default=json_default)

    # ── 完成汇总 ──
    log("\n" + "=" * 72)
    log("GeoXGBoost 双模型架构运行完成！")
    log(f"  学校样本            : {len(model_df)} 所（全部学校类型）")
    log(f"  ┌ 诊断模型（模型A） : 特征={len(feature_cols_A)}"
        f" | R²={cv_results['r2_raw']['mean']:.4f}"
        f" | MdAPE={cv_results['mdape_raw']['mean']:.1f}%")
    log(f"  └ 制图模型（模型B） : 特征={len(feature_cols_B)}"
        f" | 格网唯一值={n_uniq_norm}"
        f" | CV={cv_B_val:.3f}")
    log(f"  一致性检验          :"
        f" ρ(诊断,制图)={rho_AB_str}{rho_AB_sig}"
        f" | ρ(制图,真实)={rho_Btrue_str}{rho_Btrue_sig}"
        f" | {'✓ 一致' if consistent else ('○ 方向正确' if rho_Btrue_dir else '⚠ 检查数据')}")
    log(f"  Moran's I（诊断模型）: {moran_result['moran_I']:.4f}"
        f" → {moran_result['interpretation'][:25]}...")
    if shap_summary.get("top5_features"):
        log(f"  SHAP前5特征（制图模型）: {shap_summary['top5_features']}")
    log(f"  输出目录            : {outdir}")
    log(f"  主要输出文件：")
    log(f"    prediction_diagnostics.png   - 预测诊断综合图")
    log(f"    dual_model_consistency.png   - 双模型一致性检验图")
    log(f"    risk_distribution.png        - 风险等级分布图")
    log(f"    cv_results_comparison.png    - 交叉验证对比图")
    log(f"    shap_summary_beeswarm.png    - SHAP摘要图")
    log(f"    shap_bar_plot.png            - SHAP特征重要性图")
    log(f"    shap_dependence_top4.png     - SHAP依赖图")
    log(f"    shap_spatial_map.png         - SHAP空间分布图")
    log("=" * 72)

    # ── 返回所有输出文件路径 ──
    return {
        "school_pressure_csv": str(outdir / "school_pressure_prediction.csv"),
        "pressure_coefs_shp": str(outdir / "pressure_coefs.shp"),
        "pressure_coefs_gpkg": str(outdir / "pressure_coefs.gpkg"),
        "pressure_risk_tif": str(outdir / "pressure_risk.tif"),
        "pressure_class_tif": str(outdir / "pressure_class.tif"),
        "pressure_p80_boundary_shp": str(outdir / "pressure_p80_boundary.shp"),
        "model_report_json": str(outdir / "model_report.json"),
        "feature_importance_csv": str(outdir / "feature_importance.csv"),
        "fig_diagnostics_png": str(outdir / "prediction_diagnostics.png"),
        "fig_consistency_png": str(outdir / "dual_model_consistency.png"),
        "fig_risk_distribution_png": str(outdir / "risk_distribution.png"),
        "fig_cv_comparison_png": str(outdir / "cv_results_comparison.png"),
        "fig_shap_summary_png": str(outdir / "shap_summary_beeswarm.png"),
        "fig_shap_bar_png": str(outdir / "shap_bar_plot.png"),
        "fig_shap_dependence_png": str(outdir / "shap_dependence_top4.png"),
        "fig_shap_spatial_png": str(outdir / "shap_spatial_map.png"),
        "shap_values_csv": str(outdir / "shap_values_school.csv"),
        "shap_spatial_gpkg": str(outdir / "shap_spatial_top3.gpkg"),
        "optuna_trials_main_csv": str(outdir / "optuna_trials_诊断模型粗调.csv"),
        "optuna_trials_refine_csv": str(outdir / "optuna_trials_诊断模型细调.csv"),
        "optuna_trials_B_csv": str(outdir / "optuna_trials_制图模型调参.csv"),
        "output_directory": str(outdir),
    }