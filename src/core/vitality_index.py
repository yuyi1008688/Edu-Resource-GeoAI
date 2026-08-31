# -*- coding: utf-8 -*-
"""
vitality_index_core.py  —  城市活力指数计算核心模块

从 Vitality_index_tool.py 提取的参数化计算引擎。
支持两种模式：
  1. 从 CSV 读取预计算的 POI 数据（推荐，流水线使用此方式）
  2. 直接传入 DataFrame（用于 Python 脚本调用）

核心算法：
  A. KDE 高斯核密度估计（城区活力密度）
  B. 香农熵 POI 多样性（城区功能混合度）
  C. Z-score 标准化 + 0.5:0.5 融合 → vitality ∈ [0, 1]

原版脚本：Vitality_index_tool.py V5.3.1
提取日期：2026-08-08
仓库文件名：vitality_index.py（逻辑保留自早期单文件版本）
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import KernelDensity


class VitalityEngine:
    """城市活力指数计算引擎（参数化版本）"""

    def __init__(self, fishnet_path, proj_crs="EPSG:4547", kde_bandwidth=260):
        """
        Parameters
        ----------
        fishnet_path : str
            渔网 shapefile 路径
        proj_crs : str
            投影坐标系（用于距离计算）
        kde_bandwidth : float
            KDE 高斯核带宽（米），默认 260
        """
        self.fishnet = gpd.read_file(fishnet_path).to_crs(proj_crs)
        self.proj_crs = proj_crs
        self.kde_bandwidth = kde_bandwidth
        self.fishnet["fid"] = range(len(self.fishnet))
        print(f"[VitalityEngine] 渔网加载: {len(self.fishnet)} 个格网单元, "
              f"KDE带宽={kde_bandwidth}m")

    def calculate(self, df_poi):
        """
        计算活力指数。

        Parameters
        ----------
        df_poi : pd.DataFrame
            POI 数据，必须包含列: lng, lat, cat_id
            - lng/lat: WGS84 经纬度
            - cat_id: POI 类别 ID（整数，用于多样性计算）

        Returns
        -------
        gpd.GeoDataFrame
            原渔网 + density / diversity / vitality / is_empty 列
        """
        if df_poi.empty:
            print("  ⚠ 活力POI为空，跳过计算，全部置零")
            self.fishnet["density"] = 0.0
            self.fishnet["diversity"] = 0.0
            self.fishnet["vitality"] = 0.0
            self.fishnet["is_empty"] = 1
            return self.fishnet

        # ── POI → 投影坐标 ──
        gdf_poi = gpd.GeoDataFrame(
            df_poi,
            geometry=gpd.points_from_xy(df_poi["lng"], df_poi["lat"]),
            crs="EPSG:4326"
        ).to_crs(self.proj_crs)

        # ── A. KDE 密度 ──
        print(f"\n [步骤A] KDE密度 (带宽={self.kde_bandwidth}m)...")
        poi_xy = np.vstack([gdf_poi.geometry.x, gdf_poi.geometry.y]).T
        grid_xy = np.vstack([self.fishnet.centroid.x,
                             self.fishnet.centroid.y]).T
        kde = KernelDensity(bandwidth=self.kde_bandwidth,
                            kernel='gaussian').fit(poi_xy)
        density = np.exp(kde.score_samples(grid_xy)) * len(poi_xy)

        # ── B. 香农熵多样性 ──
        print(" [步骤B] 香农熵多样性...")
        joined = gpd.sjoin(gdf_poi, self.fishnet[["fid", "geometry"]],
                           how="inner", predicate="within")
        diversity = np.zeros(len(self.fishnet))
        for fid, grp in joined.groupby("fid"):
            p_i = grp["cat_id"].value_counts(normalize=True)
            diversity[int(fid)] = float(-np.sum(p_i * np.log(p_i + 1e-12)))

        # ── C. Z-score 标准化 + 融合 ──
        print(" [步骤C] Z-score标准化 + 融合...")
        def zsw(arr):
            return np.clip((arr - arr.mean()) / (arr.std() + 1e-9), -3.5, 3.5)

        pv = 0.5 * zsw(density) + 0.5 * zsw(diversity)
        self.fishnet["density"] = density
        self.fishnet["diversity"] = diversity
        self.fishnet["vitality"] = (
            (pv - pv.min()) / (pv.max() - pv.min() + 1e-9)
        )
        self.fishnet["is_empty"] = (density == 0).astype(int)
        return self.fishnet

    def calculate_from_csv(self, poi_csv_path, output_path=None):
        """
        从 CSV 文件读取 POI 数据并计算活力指数。

        Parameters
        ----------
        poi_csv_path : str
            POI CSV 文件路径，需包含 lng, lat, cat_id 列
        output_path : str, optional
            输出 GPKG 路径；为 None 则不保存

        Returns
        -------
        gpd.GeoDataFrame
        """
        df_poi = pd.read_csv(poi_csv_path, encoding="utf-8-sig")
        print(f"[VitalityEngine] 从CSV读取POI: {len(df_poi)} 条")

        result = self.calculate(df_poi)

        if output_path:
            result.to_file(output_path, driver="GPKG")
            print(f"  结果已保存: {output_path}")

        return result


# ============================================================================
# 独立运行入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="城市活力指数计算（从预计算POI CSV → 活力渔网GPKG）")
    parser.add_argument("--fishnet", required=True,
                        help="渔网 shapefile 路径")
    parser.add_argument("--poi", required=True,
                        help="POI CSV 路径 (需含 lng, lat, cat_id)")
    parser.add_argument("--output", required=True,
                        help="输出 GPKG 路径")
    parser.add_argument("--crs", default="EPSG:4547",
                        help="投影坐标系 (默认 EPSG:4547)")
    parser.add_argument("--bandwidth", type=float, default=260,
                        help="KDE 带宽 (默认 260m)")

    args = parser.parse_args()

    engine = VitalityEngine(args.fishnet, proj_crs=args.crs,
                            kde_bandwidth=args.bandwidth)
    engine.calculate_from_csv(args.poi, args.output)
    print("✅ 活力指数计算完成！")
