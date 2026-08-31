#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
章贡区渔网精准裁剪 V4.0
新增：MNDWI水体精确提取
      双通道分类：
        - MNDWI → 精确识别河流水体
        - 多波段评分 → 识别荒郊野岭

此模块将核心处理逻辑封装为参数化的 run_fishnet_cutting() 函数，
既可被流水线 CLI 调用，也可独立运行。
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免运行时弹窗
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from shapely.geometry import mapping
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ★ 使用共享工具模块的 arcpy 检测和日志函数（避免与 extract_raster_core.py 重复）
try:
    from _shared_utils import _HAS_ARCPY, log_arcpy as _log  # 平铺布局兼容
except ImportError:
    from shared_utils import _HAS_ARCPY, log_arcpy as _log   # 仓库布局（src/utils）

try:
    from crs_tools import assert_same_axis as _assert_axis  # 坐标量级守卫
except ImportError:  # utils 不在 path 时跳过守卫；流水线入口总会带上
    _assert_axis = None


# ============================================================
# 默认配置（可通过 run_fishnet_cutting() 参数覆盖）
# ============================================================
DEFAULT_CONFIG = {
    # ---- MNDWI水体判定阈值 ----
    "mndwi_water_threshold": 0.0,

    # ---- 波段配置 ----
    "bands": {
        "Decay_Index":          1,
        "Building_Density":     2,
        "Road_Density":         3,
        "Texture_Complexity":   4,
        "LandUse_Mix":          5,
        "Green_Coverage":       6,
    },

    # ---- 评分参数 ----
    "urban_score_threshold": 29,
    "fuzzy_band_low":        27,
    "fuzzy_band_high":       31,
    "fuzzy_building_threshold": 0.025,

    "score_weights": {
        "Building_Density": 0.50,
        "Road_Density":     0.25,
        "Green_Coverage":   0.15,
        "LandUse_Mix":      0.10,
    },

    # ---- 荒野判定 ----
    "wilderness_rules": {
        "green_coverage_min":    0.85,
        "building_density_max":  0.03,
        "road_density_max":      0.015,
    },

    # ---- 强制规则 ----
    "force_urban_building_min":    0.08,
    "force_nonurban_building_max": 0.001,
    "force_nonurban_green_min":    0.92,

    # ---- 高级参数 ----
    "use_spatial_smoothing":   True,
    "smoothing_iterations":    2,
    "use_ml_refinement":       True,
    "kmeans_clusters":         6,
}

# 注意：_log() 函数已由 _shared_utils.log_arcpy 提供（通过别名引入）


def create_output_dir(d):
    os.makedirs(d, exist_ok=True)
    _log(f"✅ 输出目录：{d}")


# ============================================================
# 以下所有处理函数与原版 Fishnet_cutting.py 完全一致
# ============================================================

def load_and_align_mndwi(config, reference_meta):
    """
    加载MNDWI栅格，自动对齐到参考栅格的坐标系和分辨率
    支持：
      1. 直接加载已计算的MNDWI
      2. 用Green+SWIR波段现算
    """
    _log("\n" + "="*60)
    _log("🌊 加载MNDWI数据")
    _log("="*60)

    # ---- 情况1：已有MNDWI文件 ----
    if config.get("mndwi_path") and os.path.exists(config["mndwi_path"]):
        _log("  使用已计算的MNDWI文件...")
        with rasterio.open(config["mndwi_path"]) as src:
            mndwi_data = src.read(1).astype(np.float32)
            mndwi_meta = src.meta.copy()
            mndwi_crs  = src.crs
            mndwi_transform = src.transform

    # ---- 情况2：用Green+SWIR现算 ----
    elif config.get("green_band_path") and config.get("swir_band_path"):
        _log("  从Green+SWIR波段计算MNDWI...")

        with rasterio.open(config["green_band_path"]) as g_src:
            green = g_src.read(1).astype(np.float32)
            g_meta = g_src.meta.copy()
            g_transform = g_src.transform
            g_crs = g_src.crs

        with rasterio.open(config["swir_band_path"]) as s_src:
            swir_raw  = s_src.read(1).astype(np.float32)
            s_meta    = s_src.meta.copy()
            s_transform = s_src.transform

            if swir_raw.shape != green.shape:
                _log(f"  SWIR尺寸{swir_raw.shape}≠Green尺寸{green.shape}，自动重采样...")
                swir = np.empty(green.shape, dtype=np.float32)
                reproject(
                    source=swir_raw,
                    destination=swir,
                    src_transform=s_transform,
                    src_crs=s_src.crs,
                    dst_transform=g_transform,
                    dst_crs=g_crs,
                    resampling=Resampling.bilinear
                )
            else:
                swir = swir_raw

        denom = green + swir
        denom[denom == 0] = 1e-9
        mndwi_data = (green - swir) / denom
        mndwi_meta = g_meta.copy()
        mndwi_crs  = g_crs
        mndwi_transform = g_transform

    else:
        raise FileNotFoundError(
            "❌ 未找到MNDWI文件！\n"
            "   请设置 mndwi_path 或同时设置 green_band_path + swir_band_path"
        )

    # ---- 对齐到参考栅格 ----
    ref_crs       = reference_meta['crs']
    ref_transform = reference_meta['transform']
    ref_height    = reference_meta['height']
    ref_width     = reference_meta['width']

    if mndwi_crs != ref_crs or mndwi_data.shape != (ref_height, ref_width):
        _log(f"  对齐MNDWI到参考坐标系...")
        mndwi_aligned = np.empty((ref_height, ref_width), dtype=np.float32)
        reproject(
            source=mndwi_data,
            destination=mndwi_aligned,
            src_transform=mndwi_transform,
            src_crs=mndwi_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
        mndwi_data = mndwi_aligned

    # 保存对齐后的MNDWI
    out_path = os.path.join(config["output_dir"], "mndwi_aligned.tif")
    aligned_meta = reference_meta.copy()
    aligned_meta.update({'count': 1, 'dtype': 'float32'})
    with rasterio.open(out_path, 'w', **aligned_meta) as dst:
        dst.write(mndwi_data, 1)

    water_ratio = np.sum(mndwi_data > config["mndwi_water_threshold"]) / mndwi_data.size
    _log(f"  MNDWI范围：[{mndwi_data.min():.3f}, {mndwi_data.max():.3f}]")
    _log(f"  水体像元比例（MNDWI>{config['mndwi_water_threshold']}）："
         f"{water_ratio*100:.1f}%")
    _log(f"  ✅ MNDWI已保存（可在GIS中查看）：{out_path}")

    if _assert_axis:
        _assert_axis(out_path, context="load_and_align_mndwi(对齐后栅格须为带号量级)")
    return mndwi_data


def extract_mndwi_per_grid(fishnet, mndwi_data, reference_meta, config):
    """为每个渔网格网提取MNDWI统计值"""
    _log("\n" + "="*60)
    _log("🔬 提取每格网MNDWI统计")
    _log("="*60)

    tmp_path = os.path.join(config["output_dir"], "mndwi_aligned.tif")

    mndwi_mean_list   = []
    mndwi_water_ratio_list = []
    thr = config["mndwi_water_threshold"]

    for _, row in tqdm(fishnet.iterrows(), total=len(fishnet),
                       desc="  提取MNDWI统计"):
        geom = [mapping(row.geometry)]
        try:
            with rasterio.open(tmp_path) as src:
                masked, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
            vals = masked[0].flatten()
            vals = vals[~np.isnan(vals)]

            if len(vals) == 0:
                mndwi_mean_list.append(np.nan)
                mndwi_water_ratio_list.append(np.nan)
            else:
                mndwi_mean_list.append(np.nanmean(vals))
                water_ratio = np.sum(vals > thr) / len(vals)
                mndwi_water_ratio_list.append(water_ratio)

        except Exception:
            mndwi_mean_list.append(np.nan)
            mndwi_water_ratio_list.append(np.nan)

    fishnet = fishnet.copy()
    fishnet['mndwi_mean']        = mndwi_mean_list
    fishnet['mndwi_water_ratio'] = mndwi_water_ratio_list

    water_grids = (np.array(mndwi_water_ratio_list) > 0.5).sum()
    _log(f"  水体格网（水体像元>50%）：{water_grids} 个")
    return fishnet


def load_data(config):
    _log("\n" + "="*60)
    _log("📂 加载主数据")
    _log("="*60)
    fishnet = gpd.read_file(config["fishnet_path"])
    _log(f"  渔网格网总数：{len(fishnet)}")
    with rasterio.open(config["raster_path"]) as src:
        raster_crs  = src.crs
        raster_meta = src.meta.copy()
        _log(f"  栅格尺寸：{src.width}×{src.height}，波段：{src.count}")
    if fishnet.crs != raster_crs:
        fishnet = fishnet.to_crs(raster_crs)
    if _assert_axis:
        _assert_axis(fishnet, config["raster_path"], context="fishnet_cutting.load_data")
    return fishnet, raster_meta


def extract_zonal_stats(fishnet, config):
    _log("\n" + "="*60)
    _log("📊 提取多波段分区统计")
    _log("="*60)
    band_names = list(config["bands"].keys())
    stats_dict = {f"{b}_{s}": []
                  for b in band_names
                  for s in ['mean','median','std','p10','p90']}
    nodata = None
    with rasterio.open(config["raster_path"]) as src:
        nodata = src.nodata
    if _assert_axis:
        _assert_axis(fishnet, config["raster_path"], context="extract_zonal_stats")

    for _, row in tqdm(fishnet.iterrows(), total=len(fishnet), desc="  提取统计"):
        geom = [mapping(row.geometry)]
        try:
            with rasterio.open(config["raster_path"]) as src:
                masked, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
            for i, bn in enumerate(band_names):
                vals = masked[i].flatten()
                if nodata is not None:
                    vals = vals[vals != nodata]
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    for s in ['mean','median','std','p10','p90']:
                        stats_dict[f"{bn}_{s}"].append(np.nan)
                else:
                    stats_dict[f"{bn}_mean"].append(np.nanmean(vals))
                    stats_dict[f"{bn}_median"].append(np.nanmedian(vals))
                    stats_dict[f"{bn}_std"].append(np.nanstd(vals))
                    stats_dict[f"{bn}_p10"].append(np.nanpercentile(vals, 10))
                    stats_dict[f"{bn}_p90"].append(np.nanpercentile(vals, 90))
        except Exception:
            for bn in band_names:
                for s in ['mean','median','std','p10','p90']:
                    stats_dict[f"{bn}_{s}"].append(np.nan)

    stats_df = pd.DataFrame(stats_dict)
    out = fishnet.copy().reset_index(drop=True)
    for col in stats_df.columns:
        out[col] = stats_df[col].values
    _log(f"  ✅ 完成：{len(out)} 个格网")
    return out


def classify_v4(fishnet_stats, config):
    """
    V4.0 双通道分类核心
    通道1：MNDWI → 精确水体
    通道2：多波段评分 → 城市 vs 荒野
    """
    _log("\n" + "="*60)
    _log("🎯 V4.0 双通道分类")
    _log("="*60)

    df = fishnet_stats.copy()
    w  = config["score_weights"]

    # 通道1：MNDWI水体提取
    _log("  [通道1] MNDWI水体提取...")

    mndwi_mean  = df['mndwi_mean'].fillna(-1)
    water_ratio = df['mndwi_water_ratio'].fillna(0)
    thr         = config["mndwi_water_threshold"]

    water_mask = (mndwi_mean > thr) | (water_ratio > 0.5)
    n_water    = water_mask.sum()
    _log(f"    MNDWI识别水体格网：{n_water} 个")

    # 通道2：多波段评分
    _log("  [通道2] 多波段城市化评分...")

    bd = df['Building_Density_mean'].fillna(0).clip(0, 1)
    rd = df['Road_Density_mean'].fillna(0).clip(0, 1)
    gc = df['Green_Coverage_mean'].fillna(1).clip(0, 1)
    lm = df['LandUse_Mix_mean'].fillna(0)

    bd_q95 = max(bd.quantile(0.95), 1e-9)
    rd_q95 = max(rd.quantile(0.95), 1e-9)

    bd_score = (bd / bd_q95).clip(0, 1) * 100
    rd_score = (rd / rd_q95).clip(0, 1) * 100
    gc_score = (1 - gc) * 100
    lm_score = ((lm - lm.min()) / (lm.max() - lm.min() + 1e-9)) * 100

    total_score = (
        w["Building_Density"] * bd_score +
        w["Road_Density"]     * rd_score +
        w["Green_Coverage"]   * gc_score +
        w["LandUse_Mix"]      * lm_score
    )
    df['urban_score'] = total_score

    f_low = config["fuzzy_band_low"]
    f_hi  = config["fuzzy_band_high"]
    f_bd  = config["fuzzy_building_threshold"]

    clear_urban    = total_score >= f_hi
    clear_nonurban = total_score <= f_low
    fuzzy          = (~clear_urban) & (~clear_nonurban)

    df['label'] = -1
    df.loc[clear_urban,                      'label'] = 0
    df.loc[fuzzy & (bd >= f_bd),             'label'] = 0
    df.loc[clear_nonurban | (fuzzy & (bd < f_bd)), 'label'] = 1

    # 融合双通道
    _log("  [融合] 双通道结果融合...")

    df.loc[water_mask, 'label'] = 2

    force_u = bd >= config["force_urban_building_min"]
    force_n = (bd <= config["force_nonurban_building_max"]) & \
              (gc >= config["force_nonurban_green_min"])

    df.loc[force_u & ~water_mask, 'label'] = 0
    df.loc[force_n & water_mask,  'label'] = 2
    df.loc[force_n & ~water_mask & (df['label'] == 0), 'label'] = 1

    n_u = (df['label'] == 0).sum()
    n_w = (df['label'] == 1).sum()
    n_r = (df['label'] == 2).sum()
    _log(f"\n  ▶ 融合结果 → 城市：{n_u} | 荒野：{n_w} | 水体：{n_r}")
    _log(f"    其中MNDWI直接识别水体：{n_water} 个")

    return df


def ml_refinement(df_in, config):
    _log("\n" + "="*60)
    _log("🤖 机器学习精修")
    _log("="*60)
    df = df_in.copy()

    feat_cols = [c for c in df.columns
                 if any(b in c for b in config["bands"].keys())
                 and '_mean' in c]
    if 'mndwi_mean' in df.columns:
        feat_cols += ['mndwi_mean', 'mndwi_water_ratio']

    X  = df[feat_cols].fillna(0).values
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=config['kmeans_clusters'], random_state=42, n_init=20)
    df['km_cluster'] = km.fit_predict(Xs)

    cluster_vote = {
        cl: df.loc[df['km_cluster'] == cl, 'label'].mode()[0]
        for cl in range(config['kmeans_clusters'])
        if (df['km_cluster'] == cl).sum() > 0
    }

    fuzzy = (
        (df['urban_score'] >= config["fuzzy_band_low"]) &
        (df['urban_score'] <= config["fuzzy_band_high"]) &
        (df['label'] != 2)
    )
    count_fix = sum(
        1 for idx in df[fuzzy].index
        if cluster_vote.get(df.loc[idx,'km_cluster'], df.loc[idx,'label'])
           != df.loc[idx,'label']
        and not (df.loc[idx,'mndwi_water_ratio'] > 0.5)
    )
    for idx in df[fuzzy].index:
        if df.loc[idx, 'mndwi_water_ratio'] > 0.5:
            continue
        cl  = df.loc[idx, 'km_cluster']
        df.loc[idx, 'label'] = cluster_vote.get(cl, df.loc[idx, 'label'])

    _log(f"  K-means修正：{count_fix} 个格网")

    urban_mask = df['label'] == 0
    if urban_mask.sum() > 20:
        iso = IsolationForest(contamination=0.04, random_state=42, n_estimators=200)
        iso_pred = iso.fit_predict(Xs[urban_mask])
        u_idx    = df[urban_mask].index
        anom_idx = u_idx[iso_pred == -1]
        bd_a     = df.loc[anom_idx, 'Building_Density_mean'].fillna(0)
        gc_a     = df.loc[anom_idx, 'Green_Coverage_mean'].fillna(1)
        mw_a     = df.loc[anom_idx, 'mndwi_water_ratio'].fillna(0)
        df.loc[anom_idx[(gc_a > 0.82) & (bd_a < 0.012) & (mw_a < 0.3)], 'label'] = 1
        df.loc[anom_idx[mw_a > 0.3], 'label'] = 2
        _log(f"  Isolation Forest修正：{len(anom_idx)} 个异常城市格网")

    n_u = (df['label'] == 0).sum()
    n_w = (df['label'] == 1).sum()
    n_r = (df['label'] == 2).sum()
    _log(f"  ML精修后 → 城市：{n_u} | 荒野：{n_w} | 水体：{n_r}")
    return df


def spatial_smoothing(df_in, config):
    _log("\n" + "="*60)
    _log("🗺️  空间平滑")
    _log("="*60)
    from shapely.strtree import STRtree
    df     = df_in.copy()
    geoms  = df.geometry.values
    tree   = STRtree(geoms)
    labels = df['label'].values.copy()
    mndwi_confirmed = (df['mndwi_water_ratio'].fillna(0) > 0.5).values

    for it in range(config["smoothing_iterations"]):
        new_labels = labels.copy()
        changed    = 0
        for i in range(len(df)):
            if mndwi_confirmed[i]:
                continue
            nb = [j for j in tree.query(geoms[i].buffer(1e-6)) if j != i]
            if not nb:
                continue
            nb_labels = labels[nb]
            cur = labels[i]

            if cur == 0 and len(nb_labels) >= 3 and np.all(nb_labels != 0):
                new_labels[i] = 1
                changed += 1
            elif cur == 1 and len(nb_labels) >= 4:
                if np.sum(nb_labels == 0) / len(nb_labels) >= 0.85:
                    new_labels[i] = 0
                    changed += 1
        labels = new_labels
        _log(f"  第{it+1}次平滑：变更 {changed} 个")

    df['label'] = labels
    return df


def export_and_visualize(df_final, config, show_plot=False):
    """
    导出分类结果 + 生成可视化

    参数
    ----
    df_final : GeoDataFrame
        分类完成的渔网数据
    config : dict
        配置字典（必须包含 output_dir 及各个输出文件路径的键）
    show_plot : bool
        是否调用 plt.show()（无显示环境应设为 False）

    返回
    ----
    dict : {输出简称: 绝对路径}
    """
    _log("\n" + "="*60)
    _log("💾 导出 + 可视化")
    _log("="*60)

    out_dir = config["output_dir"]
    urban   = df_final[df_final['label'] == 0].copy()
    wild    = df_final[df_final['label'] == 1].copy()
    river   = df_final[df_final['label'] == 2].copy()

    # 各输出路径：优先使用 config 中显式指定的路径，否则回退到默认命名
    path_urban = config.get("output_urban_shp",
                            os.path.join(out_dir, "fishnet_urban_only.shp"))
    path_wild  = config.get("output_wilderness_shp",
                            os.path.join(out_dir, "fishnet_wilderness.shp"))
    path_river = config.get("output_river_shp",
                            os.path.join(out_dir, "fishnet_river.shp"))
    path_all   = config.get("output_all_shp",
                            os.path.join(out_dir, "fishnet_all_v4.shp"))
    path_png   = config.get("output_result_png",
                            os.path.join(out_dir, "result_v4.png"))

    # 确保输出目录存在
    for p in [path_urban, path_wild, path_river, path_all]:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    urban.to_file(path_urban, encoding='utf-8')
    wild.to_file(path_wild,   encoding='utf-8')
    river.to_file(path_river, encoding='utf-8')
    df_final.to_file(path_all, encoding='utf-8')

    total = len(df_final)
    _log(f"  城市：{len(urban)}（{len(urban)/total*100:.1f}%）")
    _log(f"  荒野：{len(wild)}（{len(wild)/total*100:.1f}%）")
    _log(f"  水体：{len(river)}（{len(river)/total*100:.1f}%）")

    # 生成可视化（可选）
    out_paths = {
        "urban_fishnet":      os.path.abspath(path_urban),
        "wilderness_fishnet": os.path.abspath(path_wild),
        "river_fishnet":      os.path.abspath(path_river),
        "all_classified":     os.path.abspath(path_all),
        "result_image":       os.path.abspath(path_png),
    }

    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axes = plt.subplots(1, 4, figsize=(26, 7))
        cmap3 = ListedColormap(['#FF6B6B', '#51CF66', '#339AF0'])

        df_final.plot(ax=axes[0], column='label', cmap=cmap3,
                      edgecolor='white', linewidth=0.15)
        axes[0].legend(handles=[
            mpatches.Patch(color='#FF6B6B', label='城市建成区'),
            mpatches.Patch(color='#51CF66', label='荒郊野岭'),
            mpatches.Patch(color='#339AF0', label='河流水体'),
        ], loc='lower right', fontsize=9)
        axes[0].set_title('分类结果 V4.0（含MNDWI）', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        df_final.plot(ax=axes[1], column='mndwi_mean', cmap='RdBu',
                      edgecolor='none', vmin=-0.5, vmax=0.5, legend=True,
                      legend_kwds={'label':'MNDWI','shrink':0.7})
        axes[1].axhline(0, color='blue', linewidth=0)
        axes[1].set_title(f'MNDWI（蓝色=水体，阈值={config["mndwi_water_threshold"]}）',
                          fontsize=12, fontweight='bold')
        axes[1].axis('off')

        df_final.plot(ax=axes[2], column='mndwi_water_ratio', cmap='Blues',
                      edgecolor='none', vmin=0, vmax=1, legend=True,
                      legend_kwds={'label':'水体像元占比','shrink':0.7})
        axes[2].set_title('格网内水体像元占比', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        df_final.plot(ax=axes[3], column='Building_Density_mean', cmap='Oranges',
                      edgecolor='none', legend=True,
                      legend_kwds={'label':'Building Density','shrink':0.7})
        axes[3].set_title('建筑密度', fontsize=12, fontweight='bold')
        axes[3].axis('off')

        plt.suptitle('章贡区渔网精准分类 V4.0（MNDWI+多波段融合）',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path_png, dpi=200, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        _log(f"  ✅ 图像：{path_png}")
    except Exception as e:
        _log(f"  ⚠ 可视化生成失败（非致命）：{e}")

    return out_paths


# ============================================================
# 命令行入口函数
# ============================================================

def run_fishnet_cutting(
    raster_path,
    fishnet_path,
    output_dir,
    output_urban_shp=None,
    output_wilderness_shp=None,
    output_river_shp=None,
    output_all_shp=None,
    output_result_png=None,
    output_mndwi_tif=None,
    mndwi_path=None,
    green_band_path=None,
    swir_band_path=None,
    mndwi_water_threshold=0.0,
    urban_score_threshold=29,
    fuzzy_band_low=27,
    fuzzy_band_high=31,
    fuzzy_building_threshold=0.025,
    use_spatial_smoothing=True,
    use_ml_refinement=True,
    kmeans_clusters=6,
    generate_plot=True,
):
    """
    章贡区渔网精准裁剪 V4.0 —— 命令行入口

    所有数据处理逻辑与原版 Fishnet_cutting.py 完全一致。

    参数
    ----
    raster_path : str
        多波段物理特征栅格 (.tif)
    fishnet_path : str
        渔网矢量格网 (.shp)
    output_dir : str
        中间文件输出目录
    output_urban_shp : str or None
        输出城市建成区渔网路径（默认: output_dir/fishnet_urban_only.shp）
    output_wilderness_shp : str or None
        输出荒野渔网路径
    output_river_shp : str or None
        输出河流水体渔网路径
    output_all_shp : str or None
        输出全部分类渔网路径
    output_result_png : str or None
        输出可视化PNG路径
    output_mndwi_tif : str or None
        输出对齐后MNDWI栅格路径
    mndwi_path : str or None
        已有的MNDWI栅格（可选）
    green_band_path : str or None
        绿波段栅格（B03，用于计算MNDWI）
    swir_band_path : str or None
        SWIR波段栅格（B11，用于计算MNDWI）
    mndwi_water_threshold : float
        MNDWI水体判定阈值（默认 0.0）
    urban_score_threshold : int
        城市评分阈值（默认 29）
    fuzzy_band_low : int
        模糊带下限（默认 27）
    fuzzy_band_high : int
        模糊带上限（默认 31）
    fuzzy_building_threshold : float
        模糊带建筑密度阈值（默认 0.025）
    use_spatial_smoothing : bool
        是否使用空间平滑（默认 True）
    use_ml_refinement : bool
        是否使用ML精修（默认 True）
    kmeans_clusters : int
        KMeans聚类数（默认 6）
    generate_plot : bool
        是否生成可视化PNG（默认 True）

    返回
    ----
    dict : {
        "urban_fishnet":      城市建成区渔网路径,
        "wilderness_fishnet": 荒野渔网路径,
        "river_fishnet":      河流水体渔网路径,
        "all_classified":     全部分类渔网路径,
        "mndwi_aligned":      对齐后的MNDWI栅格路径,
        "result_image":       可视化PNG路径,
    }
    """
    _log("="*60)
    _log("  章贡区渔网精准裁剪 V4.0")
    _log("  新增：MNDWI精确水体提取")
    _log("="*60)

    # ---- 构建配置 ----
    config = dict(DEFAULT_CONFIG)
    config["raster_path"]  = raster_path
    config["fishnet_path"] = fishnet_path
    config["output_dir"]   = output_dir
    config["mndwi_path"]        = mndwi_path or ""
    config["green_band_path"]   = green_band_path or ""
    config["swir_band_path"]    = swir_band_path or ""
    config["mndwi_water_threshold"] = mndwi_water_threshold
    config["urban_score_threshold"] = urban_score_threshold
    config["fuzzy_band_low"]         = fuzzy_band_low
    config["fuzzy_band_high"]        = fuzzy_band_high
    config["fuzzy_building_threshold"] = fuzzy_building_threshold
    config["use_spatial_smoothing"] = use_spatial_smoothing
    config["use_ml_refinement"]     = use_ml_refinement
    config["kmeans_clusters"]       = kmeans_clusters

    # 显式输出路径（回退到默认命名）
    config["output_urban_shp"]   = output_urban_shp or os.path.join(output_dir, "fishnet_urban_only.shp")
    config["output_wilderness_shp"] = output_wilderness_shp or os.path.join(output_dir, "fishnet_wilderness.shp")
    config["output_river_shp"]      = output_river_shp or os.path.join(output_dir, "fishnet_river.shp")
    config["output_all_shp"]        = output_all_shp or os.path.join(output_dir, "fishnet_all_v4.shp")
    config["output_result_png"]     = output_result_png or os.path.join(output_dir, "result_v4.png")
    config["output_mndwi_tif"]      = output_mndwi_tif or os.path.join(output_dir, "mndwi_aligned.tif")

    # 确保 output_dir 存在
    create_output_dir(config["output_dir"])

    # ---- 执行处理流水线（与原版 main() 完全一致）----
    fishnet, raster_meta = load_data(config)

    mndwi_data = load_and_align_mndwi(config, raster_meta)

    fishnet_stats = extract_zonal_stats(fishnet, config)

    fishnet_stats = extract_mndwi_per_grid(
        fishnet_stats, mndwi_data, raster_meta, config
    )

    fishnet_classified = classify_v4(fishnet_stats, config)

    if config["use_ml_refinement"]:
        fishnet_classified = ml_refinement(fishnet_classified, config)

    if config["use_spatial_smoothing"]:
        fishnet_classified = spatial_smoothing(fishnet_classified, config)

    out_paths = export_and_visualize(
        fishnet_classified, config, show_plot=generate_plot
    )

    # 补充 MNDWI 对齐栅格路径
    out_paths["mndwi_aligned"] = os.path.abspath(
        os.path.join(config["output_dir"], "mndwi_aligned.tif")
    )

    _log("\n🎉 V4.0 完成！")
    return out_paths


# ============================================================
# 独立运行入口（调试用，路径由命令行参数显式指定）
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="章贡区渔网精准裁剪 V4.0")
    parser.add_argument("--raster", required=True,
                        help="多波段物理特征栅格路径")
    parser.add_argument("--fishnet", required=True,
                        help="渔网矢量格网路径")
    parser.add_argument("--output", required=True,
                        help="输出工作空间目录")
    parser.add_argument("--mndwi", default=None,
                        help="MNDWI 水体指数栅格（可选）")
    parser.add_argument("--green", default=None,
                        help="绿波段栅格 B03（可选）")
    parser.add_argument("--swir", default=None,
                        help="SWIR 波段栅格 B11（可选）")
    parser.add_argument("--mndwi-threshold", type=float, default=0.0,
                        help="MNDWI 水体判定阈值 (默认 0.0)")
    parser.add_argument("--score-threshold", type=int, default=29,
                        help="城市评分阈值 (默认 29)")
    parser.add_argument("--fuzzy-low", type=int, default=27,
                        help="模糊带下限 (默认 27)")
    parser.add_argument("--fuzzy-high", type=int, default=31,
                        help="模糊带上限 (默认 31)")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="禁用空间平滑")
    parser.add_argument("--no-ml", action="store_true",
                        help="禁用机器学习精修")
    parser.add_argument("--no-plot", action="store_true",
                        help="禁用可视化图表生成")

    args = parser.parse_args()

    result = run_fishnet_cutting(
        raster_path=args.raster,
        fishnet_path=args.fishnet,
        output_dir=args.output,
        mndwi_path=args.mndwi,
        green_band_path=args.green,
        swir_band_path=args.swir,
        mndwi_water_threshold=args.mndwi_threshold,
        urban_score_threshold=args.score_threshold,
        fuzzy_band_low=args.fuzzy_low,
        fuzzy_band_high=args.fuzzy_high,
        use_spatial_smoothing=not args.no_smoothing,
        use_ml_refinement=not args.no_ml,
        generate_plot=not args.no_plot,
    )
    for k, v in result.items():
        print(f"  {k}: {v}")
