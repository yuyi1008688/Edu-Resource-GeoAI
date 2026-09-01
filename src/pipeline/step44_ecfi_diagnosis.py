# -*- coding: utf-8 -*-
"""
step44_ecfi_diagnosis.py — 4.4 ECFI 三维教育压力诊断（纯 Python CLI）

纯 Python CLI，实现 ECFI 三维教育压力诊断的全部算法（safe_zonal_stats、
  winsorize_d3、entropy_weight、calc_d1_weighted、d1_nearest_fallback、
  calc_cluster_proportions、calc_location_features、
  calc_poi_features_with_spatial_fallback、calculate_supply、diagnose_d3_quality、
  三级回退等）；命令行参数默认值与原始实现一致。
  - 阈值（BUFFER_DIST、winsorize 0.95、POI 三级回退等）保持不变。

上游依赖：小学/中学服务区由 step43b 生成，Fishnet 聚类层由 step42 生成。

示例：
  python src/pipeline/step44_ecfi_diagnosis.py --isochrone-elem <shp> \
      --isochrone-mid <shp> --fishnet <shp> --out-dir output/step44
"""
import argparse
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "core"), str(_SRC / "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import paths  # noqa: E402


# ── 消息层：统一 print ──
def _log(msg):
    print(msg)


def _logw(msg):
    print("[警告]", msg)


def _loge(msg):
    print("[错误]", msg)


class _MsgShim:
    def addErrorMessage(self, m):
        print("[错误]", m)


class _Param:
    def __init__(self, v=None):
        self.value = v

    @property
    def valueAsText(self):
        return None if self.value is None else str(self.value)


# -*- coding: utf-8 -*-

import os





def tool_44_execute(params, messages=None):

        # ── 导入依赖 ──────────────────────────────
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        try:
            from rasterstats import zonal_stats
        except ImportError:
            _loge(
                "缺少 rasterstats 包，无法执行栅格分区统计。\n"
                "请在已安装依赖的 Python 环境中运行：\n"
                "    pip install rasterstats\n"
                "或使用项目附带的 requirements.txt 一键安装：\n"
                "    pip install -r requirements.txt"
            )
            raise
        import rasterio
        from rasterio.warp import (calculate_default_transform,
                                   reproject, Resampling)
        from shapely.geometry import Point
        from scipy.spatial import cKDTree
        import warnings
        import re
        warnings.filterwarnings('ignore')

        # ── 读取参数 ──────────────────────────────
        SCHOOL_CSV            = params[0].valueAsText
        POI_CSV               = params[1].valueAsText
        ISOCHRONE_ELEM        = params[2].valueAsText
        ISOCHRONE_MID         = params[3].valueAsText
        FISHNET_SHP           = params[4].valueAsText
        ROAD_DENSITY_TIF_4526 = params[5].valueAsText or ""
        ROAD_DENSITY_TIF_RAW  = params[6].valueAsText or ""
        WORLDPOP_TIF          = params[7].valueAsText
        BUILDING_DEN_TIF      = params[8].valueAsText
        STUDY_AREA_SHP        = params[9].valueAsText  or None
        BUILDINGS_SHP         = params[10].valueAsText or None
        RIVER_SHP             = params[11].valueAsText or None
        city_lng              = float(params[12].value)
        city_lat              = float(params[13].value)
        RATIO_6_14            = float(params[14].value)
        S_PER_PRIMARY         = float(params[15].value)
        S_PER_MIDDLE          = float(params[16].value)
        AVG_FLOORS            = float(params[17].value)
        WALK_BUFFER           = int(params[18].value)
        D3_WINSORIZE          = bool(params[19].value)
        D3_WINSORIZE_UPPER    = float(params[20].value)
        K_NEIGHBORS           = int(params[21].value)
        OUTPUT_DIR            = params[22].valueAsText
        output_csv_name       = params[23].valueAsText

        CITY_CENTER = (city_lng, city_lat)
        OUTPUT_CSV  = os.path.join(OUTPUT_DIR, output_csv_name)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ── 全局常量 ──────────────────────────────
        TARGET_CRS             = "EPSG:4526"
        BUFFER_DIST            = 100
        # 勘误（复现基准核对）：历史脚本写的是“学校点 buffer(100m)”叠加建筑，
        # 但历史基准 4.4_school_profile 中 98/99 所 Supply_final 恰为钳制上限
        # 10000；实测点缓冲口径仅 8 所过万，而服务区面口径 97 所过万，故基准
        # 真实口径为“服务区面叠加建筑”。默认 service_area；point_buffer 备查。
        SUPPLY_GEOM_MODE       = os.environ.get(
            "SUPPLY_GEOM_MODE", "service_area")
        FALLBACK_PRIMARY_SMALL = 1080
        FALLBACK_PRIMARY_LARGE = 1620
        FALLBACK_MIDDLE_SMALL  = 1200
        FALLBACK_MIDDLE_LARGE  = 1800
        WORLDPOP_NODATA        = -9999.0
        ROAD_NODATA            = -9999.0
        BLD_NODATA             = -9999.0

        COL_FID       = 'FID'
        COL_NAME      = '名称'
        COL_SCHOOL_ID = 'school_id'
        COL_LEVEL     = '类型'
        COL_STUDENTS  = '学生数'
        COL_LNG       = '经度'
        COL_LAT       = '纬度'
        COL_X_COORD   = 'x_coord'
        COL_Y_COORD   = 'y_coord'

        def msg(text):
            _log(str(text))

        # ══════════════════════════════════════════
        # 关键修复：safe_zonal_stats
        # 修复1：GDAL_MEM_ENABLE_OPEN="YES"
        #        解决新版 GDAL 安全限制
        # 修复2：只传纯 geometry 列
        #        解决 Pandas 混合类型报错
        # ══════════════════════════════════════════
        def safe_zonal_stats(geometries, raster_path,
                              stats, nodata_val):
            if nodata_val is None or (
                    isinstance(nodata_val, float)
                    and np.isnan(nodata_val)):
                nodata_val = -9999.0
            with rasterio.Env(GDAL_MEM_ENABLE_OPEN="YES"):
                if isinstance(geometries, gpd.GeoDataFrame):
                    geom_only = gpd.GeoDataFrame(
                        geometry=geometries.geometry.values,
                        crs=geometries.crs)
                    return zonal_stats(
                        geom_only, raster_path,
                        stats=stats, nodata=nodata_val)
                return zonal_stats(
                    geometries, raster_path,
                    stats=stats, nodata=nodata_val)

        # ══════════════════════════════════════════
        # 工具函数（与原 py 完全一致）
        # ══════════════════════════════════════════

        def force_clean(name):
            if pd.isna(name):
                return ""
            name = str(name)
            name = name.replace('（', '(').replace('）', ')')
            name = name.replace('　', '').replace(' ', '')
            name = re.sub(r'[^一-龥a-zA-Z0-9()]', '', name)
            return name.strip()

        def get_s_per(level):
            level = str(level)
            if '小学' in level:
                return S_PER_PRIMARY
            if ('初中' in level
                    or '中学' in level
                    or '高中' in level):
                return S_PER_MIDDLE
            return S_PER_PRIMARY

        def map_level(name, orig_level=''):
            name = str(name)
            # 通用高中关键词；名称无通用高中字样但确为高中的本地完全中学，
            # 经环境变量 EDU_EXTRA_HIGH_KW（逗号分隔）补充，清单不入库
            _extra = [k.strip() for k in
                      os.environ.get('EDU_EXTRA_HIGH_KW', '').split(',')
                      if k.strip()]
            HIGH_KW = ['高级中学', '高中', '完全中学'] + _extra
            for kw in HIGH_KW:
                if kw in name:
                    return '高中'
            if ('九年' in name
                    or '一贯制' in name):
                return '九年一贯制'
            if '小学' in name:
                return '小学'
            if '初中' in name:
                return '初中'
            if '中学' in name:
                return '初中'
            # 名称无“中学”但原始类型为中学（如“XX学校”），统一归为初中
            if '中学' in str(orig_level) or '初中' in str(orig_level):
                return '初中'
            if '小学' in str(orig_level):
                return '小学'
            return str(orig_level) if orig_level else str(name)

        def entropy_weight(df, cols):
            X      = df[cols].values.astype(float)
            X_min  = X.min(axis=0)
            X_max  = X.max(axis=0)
            X_norm = (X - X_min) / (X_max - X_min + 1e-9)
            n      = X_norm.shape[0]
            col_sums = np.where(
                X_norm.sum(axis=0) == 0,
                1e-9, X_norm.sum(axis=0))
            p = X_norm / col_sums
            p = np.where(p == 0, 1e-9, p)
            e = (-(1.0 / np.log(n))
                 * (p * np.log(p)).sum(axis=0))
            d = 1 - e
            return d / (d.sum() + 1e-9)

        def winsorize_d3(series, upper_quantile=0.95):
            upper_bound = series.quantile(upper_quantile)
            clipped     = series.clip(upper=upper_bound)
            n_clipped   = int((series > upper_bound).sum())
            return clipped, upper_bound, n_clipped

        # ── [P2-3] 栅格分辨率 ─────────────────────

        def get_raster_resolution_info(raster_path,
                                        label=""):
            info = {"label": label, "path": raster_path,
                    "res_x_m": None, "res_y_m": None,
                    "crs_epsg": None,
                    "width": None, "height": None}
            try:
                with rasterio.open(raster_path) as src:
                    info["res_x_m"]  = abs(src.transform.a)
                    info["res_y_m"]  = abs(src.transform.e)
                    info["crs_epsg"] = (src.crs.to_epsg()
                                        if src.crs else None)
                    info["width"]    = src.width
                    info["height"]   = src.height
                msg("  [%s] %.2fm x %.2fm | EPSG:%s"
                    " | %dx%d"
                    % (label,
                       info["res_x_m"], info["res_y_m"],
                       info["crs_epsg"],
                       info["width"], info["height"]))
            except Exception as ex:
                msg("  [%s] 分辨率读取失败: %s" % (label, ex))
            return info

        def print_raster_resolution_summary(res_info_list):
            msg("=" * 65)
            msg("[论文注明] 各栅格因子实际分辨率汇总")
            msg("=" * 65)
            for info in res_info_list:
                if info["res_x_m"] is not None:
                    note = ""
                    if abs(info["res_x_m"]
                           - info["res_y_m"]) > 0.01:
                        note = " (非正方形像元)"
                    if info["res_x_m"] not in [
                            10.0, 30.0, 100.0,
                            250.0, 500.0, 1000.0]:
                        note += " (非标准分辨率)"
                    msg("  %-22s %.2fx%.2fm  EPSG:%-10s%s"
                        % (info["label"],
                           info["res_x_m"],
                           info["res_y_m"],
                           str(info["crs_epsg"]),
                           note))
                else:
                    msg("  %-22s 读取失败"
                        % info["label"])
            msg("=" * 65)

        # ── 栅格预处理 ────────────────────────────

        def fix_raster_nodata(input_path, output_path,
                               src_nodata_value):
            with rasterio.open(input_path) as src:
                data    = src.read(1).astype(np.float64)
                profile = src.profile.copy()
            data[data == src_nodata_value] = -9999.0
            data[data < -1e10]             = -9999.0
            nan_count = int(np.isnan(data).sum())
            if nan_count > 0:
                msg("    [NaN修复] %d 个 -> -9999"
                    % nan_count)
                data[np.isnan(data)] = -9999.0
            profile.update(dtype='float64', nodata=-9999.0)
            with rasterio.open(output_path, 'w',
                               **profile) as dst:
                dst.write(data, 1)
            msg("    [写出] %s" % output_path)
            return output_path

        def ensure_raster_crs(input_path,
                               target_crs=TARGET_CRS):
            with rasterio.open(input_path) as src:
                src_crs  = src.crs
                src_epsg = (src_crs.to_epsg()
                             if src_crs else None)
            if src_epsg is None:
                msg("  栅格无 CRS: %s" % input_path)
                return input_path
            target_epsg = int(target_crs.split(':')[1])
            if src_epsg == target_epsg:
                msg("  [CRS] EPSG:%d 无需重投影"
                    % src_epsg)
                return input_path
            output_path = input_path.replace(
                '.tif',
                '_%s.tif' % target_crs.replace(':', ''))
            msg("  [CRS] EPSG:%d -> %s 重投影中..."
                % (src_epsg, target_crs))
            with rasterio.open(input_path) as src:
                transform, width, height = (
                    calculate_default_transform(
                        src.crs, target_crs,
                        src.width, src.height,
                        *src.bounds))
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'nodata': -9999.0})
                with rasterio.open(
                        output_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(
                                dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear)
            msg("  [CRS] 已保存: %s" % output_path)
            return output_path

        def fix_building_density(input_path, output_path,
                                  study_area_shp=None,
                                  target_crs=TARGET_CRS):
            msg("\n[建筑密度黑边修复]")
            with rasterio.open(input_path) as src:
                data      = src.read(1).astype(np.float64)
                profile   = src.profile.copy()
                transform = src.transform
                crs       = src.crs
            total    = data.size
            zero_cnt = int((data == 0).sum())
            pos_cnt  = int((data > 0).sum())
            msg("  总像元:%d  值为0:%d  值>0:%d"
                % (total, zero_cnt, pos_cnt))

            if (study_area_shp is not None
                    and os.path.exists(study_area_shp)):
                msg("  [方案A] 研究区掩膜")
                sa = gpd.read_file(
                    study_area_shp).to_crs(crs)
                from rasterio.features import geometry_mask
                # 关键修复：GDAL_MEM_ENABLE_OPEN="YES"
                with rasterio.Env(
                        GDAL_MEM_ENABLE_OPEN="YES"):
                    mask = geometry_mask(
                        [g for g in sa.geometry],
                        transform=transform,
                        invert=True,
                        out_shape=data.shape)
                outside_zero = (~mask) & (data == 0)
                data[outside_zero] = -9999.0
                msg("  区外黑边:%d -> nodata | 区内0:%d"
                    % (int(outside_zero.sum()),
                       int((mask & (data == 0)).sum())))
            else:
                msg("  [方案B] 连通域边缘检测")
                from scipy import ndimage
                zero_mask  = (data == 0)
                labeled, _ = ndimage.label(zero_mask)
                edge_labels = set()
                for arr in [labeled[0, :], labeled[-1, :],
                             labeled[:, 0], labeled[:, -1]]:
                    edge_labels.update(
                        np.unique(arr).tolist())
                edge_labels.discard(0)
                bm = np.isin(labeled, list(edge_labels))
                data[bm] = -9999.0
                msg("  黑边:%d -> nodata | 内部0:%d"
                    % (int(bm.sum()),
                       int((zero_mask & ~bm).sum())))

            data[np.isnan(data)] = -9999.0
            profile.update(dtype='float64', nodata=-9999.0)
            with rasterio.open(output_path, 'w',
                               **profile) as dst:
                dst.write(data, 1)
            valid_after = int((data > -9999.0).sum())
            msg("  修复后有效像元:%d (%.1f%%)"
                % (valid_after,
                   valid_after / total * 100))
            return output_path

        # ── D1 面积加权 ───────────────────────────

        def calc_d1_weighted(isochrone_gdf, fishnet_gdf,
                              vitality_field,
                              id_field='school_id',
                              name_field=None):
            msg("  服务区:%d  格网:%d"
                % (len(isochrone_gdf), len(fishnet_gdf)))
            joined = gpd.sjoin(
                fishnet_gdf[[vitality_field, 'geometry']]
                    .copy()
                    .reset_index(drop=False)
                    .rename(columns={'index': 'f_idx'}),
                isochrone_gdf[[id_field, 'geometry']].copy(),
                how='inner', predicate='intersects')
            if len(joined) == 0:
                msg("  intersects 无命中")
                return pd.DataFrame(
                    columns=[id_field, 'D1_vitality'])

            msg("  命中:%d 所 %d 条"
                % (joined[id_field].nunique(), len(joined)))
            iso_index = (isochrone_gdf
                         .set_index(id_field)['geometry'])
            fn_re     = fishnet_gdf[
                [vitality_field, 'geometry']].copy()
            results   = []
            for sid, group in joined.groupby(id_field):
                iso_geom = iso_index[sid]
                sub      = fn_re.loc[
                    group.index.tolist()].copy()
                sub['intersect_area'] = sub.geometry.apply(
                    lambda g: g.intersection(iso_geom).area)
                sub = sub[sub['intersect_area'] > 0.01]
                if len(sub) == 0:
                    continue
                tw = sub['intersect_area'].sum()
                wv = ((sub[vitality_field]
                       * sub['intersect_area']).sum() / tw)
                results.append({
                    id_field:      sid,
                    'D1_vitality': wv,
                    '_grid_cnt':   len(sub),
                    '_cover_m2':   tw})
            if not results:
                msg("  面积加权结果为空")
                return pd.DataFrame(
                    columns=[id_field, 'D1_vitality'])

            result_df  = pd.DataFrame(results)
            all_ids    = set(isochrone_gdf[id_field].unique())
            missed_ids = (all_ids
                          - set(result_df[id_field].unique()))
            msg("  面积加权命中:%d/%d"
                % (len(result_df), len(isochrone_gdf)))
            if missed_ids:
                msg("  未命中 %d 所:" % len(missed_ids))
                for sid in sorted(missed_ids):
                    if (name_field
                            and name_field
                            in isochrone_gdf.columns):
                        nm = isochrone_gdf.loc[
                            isochrone_gdf[id_field] == sid,
                            name_field].values[0]
                        msg("    - [%s] %s" % (sid, nm))
                    else:
                        msg("    - [%s]" % sid)
            return result_df[[id_field, 'D1_vitality']]

        def d1_nearest_fallback(missed_ids, isochrone_gdf,
                                 fishnet_gdf, vitality_field,
                                 id_field='school_id'):
            rows = []
            for sid in missed_ids:
                iso_row = isochrone_gdf[
                    isochrone_gdf[id_field] == sid]
                if len(iso_row) == 0:
                    continue
                cent  = iso_row.geometry.centroid.values[0]
                dists = fishnet_gdf.geometry.distance(cent)
                nv    = fishnet_gdf.loc[
                    dists.idxmin(), vitality_field]
                msg("    [%s] 最近格网=%.1fm v=%.4f"
                    % (sid, dists.min(), nv))
                rows.append({id_field:      sid,
                              'D1_vitality': nv})
            return pd.DataFrame(rows)

        # ── 社区类型占比 ──────────────────────────

        def calc_cluster_proportions(isochrone_gdf,
                                      fishnet_gdf,
                                      id_field='school_id'):
            if 'CLUSTER_ID' not in fishnet_gdf.columns:
                msg("  Fishnet 无 CLUSTER_ID，跳过")
                return pd.DataFrame(columns=[id_field])
            joined = gpd.sjoin(
                fishnet_gdf[['CLUSTER_ID',
                             'geometry']].copy(),
                isochrone_gdf[[id_field,
                               'geometry']].copy(),
                how='inner', predicate='intersects')
            if len(joined) == 0:
                return pd.DataFrame(columns=[id_field])
            joined['intersect_area'] = joined.geometry.area
            joined = joined[joined['intersect_area'] > 0.01]
            result_rows = []
            for sid, group in joined.groupby(id_field):
                total_area = group['intersect_area'].sum()
                row = {id_field: sid}
                for cid in range(1, 7):
                    ca = group.loc[
                        group['CLUSTER_ID'] == cid,
                        'intersect_area'].sum()
                    row['pct_C%d' % cid] = (
                        ca / (total_area + 1e-9))
                pct_cols = ['pct_C%d' % c
                            for c in range(1, 7)]
                row['dominant_cluster'] = int(
                    np.argmax(
                        [row[c] for c in pct_cols]) + 1)
                result_rows.append(row)
            result_df = pd.DataFrame(result_rows)
            msg("  社区类型命中:%d/%d"
                % (len(result_df), len(isochrone_gdf)))
            return result_df

        # ── 区位距离特征 ──────────────────────────

        def calc_location_features(school_gdf,
                                    river_shp=None,
                                    city_center=None):
            results = pd.DataFrame(
                {COL_SCHOOL_ID:
                     school_gdf[COL_SCHOOL_ID].values})
            if city_center is not None:
                center_pt = gpd.GeoSeries(
                    [Point(city_center[0],
                           city_center[1])],
                    crs="EPSG:4326"
                ).to_crs(TARGET_CRS).iloc[0]
                results['dist_to_center'] = (
                    school_gdf.geometry.distance(center_pt))
                msg("  dist_to_center: [%.0f, %.0f] m"
                    % (results['dist_to_center'].min(),
                       results['dist_to_center'].max()))
            else:
                results['dist_to_center'] = 0.0

            if river_shp and os.path.exists(river_shp):
                try:
                    rivers = gpd.read_file(
                        river_shp).to_crs(TARGET_CRS)
                    from shapely.geometry import (
                        LineString, MultiLineString,
                        Polygon, MultiPolygon)
                    lines = []
                    for geom in rivers.geometry:
                        if isinstance(
                                geom,
                                (Polygon, MultiPolygon)):
                            lines.append(geom.boundary)
                        elif isinstance(
                                geom,
                                (LineString,
                                 MultiLineString)):
                            lines.append(geom)
                    if lines:
                        river_union = (
                            gpd.GeoSeries(lines).union_all())
                        results['dist_to_river'] = (
                            school_gdf.geometry
                            .distance(river_union))
                        msg("  dist_to_river:"
                            " [%.0f, %.0f] m"
                            % (results['dist_to_river'].min(),
                               results['dist_to_river'].max()))
                    else:
                        results['dist_to_river'] = 0.0
                except Exception as ex:
                    msg("  河流距离失败: %s" % ex)
                    results['dist_to_river'] = 0.0
            else:
                results['dist_to_river'] = 0.0
            return results

        # ── [P2-1] POI 地理学空间兜底 ─────────────

        def calc_poi_features_with_spatial_fallback(
                isochrone_gdf, poi_gdf, school_gdf,
                id_field='school_id',
                buffer_expand_factors=(1.5, 2.0, 3.0),
                k_neighbors=3):
            if 'cat_name' not in poi_gdf.columns:
                msg("  POI 无 cat_name 字段，跳过")
                return pd.DataFrame(columns=[id_field])
            msg("  [POI] 开始计算（含地理学空间兜底）...")

            RESIDENTIAL_KW = (
                '居住小区'
                '|住宅小区'
                '|社区|居民区'
                '|花园|家园'
                '|里|蘓|公馆'
                '|庄园|别墅'
                '|公寓')

            def _poi_metrics(geom, poi_local):
                try:
                    sub = poi_local[
                        poi_local.geometry.within(geom)]
                except Exception:
                    sub = poi_local[
                        poi_local.geometry.within(
                            geom.buffer(0))]
                total = len(sub)
                if total == 0:
                    return None
                probs     = (sub['cat_name']
                             .value_counts() / total)
                diversity = float(
                    -np.sum(probs * np.log(probs + 1e-9)))
                res_ratio = float(
                    sub['cat_name'].str.contains(
                        RESIDENTIAL_KW,
                        na=False).sum() / total)
                return {'poi_diversity':     diversity,
                        'residential_ratio': res_ratio,
                        'poi_count':         total}

            # Step1：正常计算
            results  = {}
            iso_idx  = isochrone_gdf.set_index(id_field)
            for sid, row in iso_idx.iterrows():
                m = _poi_metrics(row.geometry, poi_gdf)
                if m is not None:
                    m['fill_method'] = 'normal'
                    results[sid]     = m

            all_ids    = set(isochrone_gdf[id_field].unique())
            missed_ids = all_ids - set(results.keys())
            msg("  正常命中:%d/%d"
                % (len(results), len(all_ids)))

            # Step2：扩大缓冲区
            if missed_ids:
                msg("  [兜底1] 扩大缓冲区 (%d 所)"
                    % len(missed_ids))
                iso_areas = iso_idx.geometry.area
                for sid in list(missed_ids):
                    if sid not in iso_idx.index:
                        continue
                    base_geom = iso_idx.loc[sid, 'geometry']
                    eq_r = np.sqrt(iso_areas[sid] / np.pi)
                    for factor in buffer_expand_factors:
                        ed = eq_r * (factor - 1.0)
                        m  = _poi_metrics(
                            base_geom.buffer(ed), poi_gdf)
                        if m is not None:
                            m['fill_method'] = (
                                'buffer_expand_%.1fx' % factor)
                            m['expand_dist_m'] = round(ed, 1)
                            results[sid] = m
                            missed_ids.discard(sid)
                            msg("    [%s] x%.1f poi=%d"
                                % (sid, factor,
                                   m['poi_count']))
                            break
                    else:
                        msg("    [%s] 仍无POI->KNN" % sid)

            # Step3：KNN IDW
            if missed_ids:
                msg("  [兜底2] KNN IDW K=%d (%d 所)"
                    % (k_neighbors, len(missed_ids)))
                school_idx = school_gdf.set_index(id_field)
                known_sids = [s for s in results
                              if s in school_idx.index]
                if len(known_sids) >= k_neighbors:
                    kc = np.array([
                        [school_idx.loc[s, 'geometry'].x,
                         school_idx.loc[s, 'geometry'].y]
                        for s in known_sids])
                    kd = np.array(
                        [results[s]['poi_diversity']
                         for s in known_sids])
                    kr = np.array(
                        [results[s]['residential_ratio']
                         for s in known_sids])
                    tree = cKDTree(kc)
                    for sid in list(missed_ids):
                        if sid not in school_idx.index:
                            continue
                        g     = school_idx.loc[
                            sid, 'geometry']
                        k_eff = min(k_neighbors,
                                    len(known_sids))
                        dists, idxs = tree.query(
                            [[g.x, g.y]], k=k_eff)
                        dists = dists[0]
                        idxs  = idxs[0]
                        if np.min(dists) < 1.0:
                            w = np.zeros(k_eff)
                            w[np.argmin(dists)] = 1.0
                        else:
                            w = 1.0 / (dists ** 2)
                        w = w / w.sum()
                        div_knn   = float(
                            np.dot(w, kd[idxs]))
                        ratio_knn = float(
                            np.dot(w, kr[idxs]))
                        near_sid = (
                            known_sids[int(idxs[0])]
                            if hasattr(idxs, '__len__')
                            else known_sids[int(idxs)])
                        near_dist = (
                            float(dists[0])
                            if hasattr(dists, '__len__')
                            else float(dists))
                        results[sid] = {
                            'poi_diversity':      div_knn,
                            'residential_ratio':  ratio_knn,
                            'poi_count':          0,
                            'fill_method': (
                                'spatial_knn_%d' % k_eff),
                            'knn_nearest_id':     near_sid,
                            'knn_nearest_dist_m': round(
                                near_dist, 1)}
                        missed_ids.discard(sid)
                        msg("    [%s] KNN div=%.4f"
                            " res_ratio=%.4f"
                            % (sid, div_knn, ratio_knn))
                else:
                    msg("    已知点不足 %d，跳过 KNN"
                        % k_neighbors)

            # Step4：全局中位数
            if missed_ids:
                msg("  [兜底3] 全局中位数 (%d 所)"
                    % len(missed_ids))
                med_div = float(np.median(
                    [v['poi_diversity']
                     for v in results.values()]))
                med_ratio = float(np.median(
                    [v['residential_ratio']
                     for v in results.values()]))
                for sid in missed_ids:
                    results[sid] = {
                        'poi_diversity':     med_div,
                        'residential_ratio': med_ratio,
                        'poi_count':         0,
                        'fill_method': (
                            'global_median_fallback')}
                    msg("    [%s] 中位数兜底"
                        " div=%.4f res=%.4f"
                        % (sid, med_div, med_ratio))

            result_df = pd.DataFrame(
                [{id_field: sid, **v}
                 for sid, v in results.items()])
            msg("\n  [POI] 填充方法汇总:")
            for method, cnt in (result_df['fill_method']
                                 .value_counts().items()):
                icon = "ok" if method == 'normal' else "-"
                msg("    %s %s: %d 所"
                    % (icon, method, cnt))

            keep_cols = [id_field, 'poi_diversity',
                         'residential_ratio',
                         'poi_count', 'fill_method']
            for opt in ['knn_nearest_id',
                        'knn_nearest_dist_m',
                        'expand_dist_m']:
                if opt in result_df.columns:
                    keep_cols.append(opt)
            result_df = result_df[keep_cols].copy()
            for col in ['poi_diversity',
                        'residential_ratio']:
                n_null = result_df[col].isna().sum()
                if n_null > 0:
                    fb = (result_df[col].median()
                          if result_df[col].notna().any()
                          else 0.0)
                    result_df[col] = (
                        result_df[col].fillna(fb))
                    msg("  %s 最终%d个空值已兜底"
                        % (col, n_null))

            msg("\n  [POI] 最终覆盖:%d/%d"
                % (len(result_df), len(all_ids)))
            msg("  poi_diversity: [%.4f, %.4f]"
                % (result_df['poi_diversity'].min(),
                   result_df['poi_diversity'].max()))
            msg("  residential_ratio: [%.4f, %.4f]"
                % (result_df['residential_ratio'].min(),
                   result_df['residential_ratio'].max()))
            return result_df

        # ── D3 质量诊断 ───────────────────────────

        def diagnose_d3_quality(d3_series,
                                 label="D3_pressure"):
            d3 = d3_series.dropna()
            diag = {
                "n_total":  int(len(d3)),
                "n_zero":   int((d3 == 0).sum()),
                "pct_zero": float((d3 == 0).mean() * 100),
                "mean":     float(d3.mean()),
                "median":   float(d3.median()),
                "std":      float(d3.std()),
                "cv":       float(d3.std()
                                   / (d3.mean() + 1e-9)),
                "skewness": float(d3.skew()),
                "p5":       float(d3.quantile(0.05)),
                "p95":      float(d3.quantile(0.95)),
                "p95_p5_ratio": float(
                    d3.quantile(0.95)
                    / (d3.quantile(0.05) + 1e-9))}
            msg("=" * 60)
            msg("[D3质量诊断] %s" % label)
            msg("  样本量:%d  零值:%d (%.1f%%)"
                % (diag['n_total'], diag['n_zero'],
                   diag['pct_zero']))
            msg("  均值/中位数:%.4f / %.4f"
                % (diag['mean'], diag['median']))
            cv_flag = ("区分度低"
                       if diag['cv'] < 0.3 else "OK")
            msg("  CV:%.4f (%s)  偏度:%.4f"
                % (diag['cv'], cv_flag, diag['skewness']))
            msg("  P5~P95:[%.4f, %.4f]  P95/P5:%.1fx"
                % (diag['p5'], diag['p95'],
                   diag['p95_p5_ratio']))
            warnings_found = []
            if diag["cv"] < 0.3:
                warnings_found.append(
                    "D3 CV<0.3，区分度不足")
            if abs(diag["skewness"]) > 2.5:
                warnings_found.append(
                    "D3 偏度%.2f，建议 Box-Cox"
                    % diag["skewness"])
            if diag["p95_p5_ratio"] > 50:
                warnings_found.append(
                    "D3 极差比%.0fx，建议稳健回归"
                    % diag["p95_p5_ratio"])
            if warnings_found:
                msg("  警告:")
                for w in warnings_found:
                    msg("    -> %s" % w)
            else:
                msg("  D3 质量检查通过")
            msg("=" * 60)
            diag["warnings"] = warnings_found
            return diag

        # ── Supply 计算（三层回退）────────────────

        def calculate_supply(school_gdf, isochrone_all,
                              bld_fixed,
                              buildings_shp=None,
                              avg_floors=AVG_FLOORS):
            msg("\n[Supply 计算] 三层回退策略...")
            scheme_used = "C"

            if (buildings_shp is not None
                    and os.path.exists(buildings_shp)):
                msg("  [方案A] 建筑轮廓矢量叠加...")
                try:
                    bld_gdf = gpd.read_file(
                        buildings_shp).to_crs(TARGET_CRS)
                    if 'area' not in bld_gdf.columns:
                        bld_gdf['area'] = (
                            bld_gdf.geometry.area)
                    if 'floors' not in bld_gdf.columns:
                        bld_gdf['floors'] = avg_floors
                    bld_gdf['floors'] = (
                        bld_gdf['floors']
                        .fillna(avg_floors).clip(1, 30))
                    bld_gdf['floor_area'] = (
                        bld_gdf['area'] * bld_gdf['floors'])
                    if SUPPLY_GEOM_MODE == "service_area":
                        # 基准口径：整个服务区面与建筑叠加
                        sb = isochrone_all[
                            [COL_SCHOOL_ID, 'geometry']].copy()
                    else:
                        # 历史脚本字面口径：学校点外扩 100m
                        sb = school_gdf[
                            [COL_SCHOOL_ID, 'geometry']].copy()
                        sb['geometry'] = sb.geometry.buffer(
                            BUFFER_DIST)
                    inter = gpd.overlay(
                        bld_gdf[['geometry', 'floor_area']],
                        sb, how='intersection')
                    if len(inter) > 0:
                        ss = (
                            inter.groupby(COL_SCHOOL_ID)
                            ['floor_area'].sum()
                            .reset_index())
                        ss.columns = [COL_SCHOOL_ID,
                                      'total_floor_area']
                        school_gdf = school_gdf.merge(
                            ss, on=COL_SCHOOL_ID,
                            how='left')
                        school_gdf[
                            'total_floor_area'] = (
                            school_gdf['total_floor_area']
                            .fillna(0))
                        school_gdf['Supply'] = (
                            school_gdf['total_floor_area']
                            / school_gdf['S_per'])
                        nv = int(
                            (school_gdf[
                                 'total_floor_area']
                             > 0).sum())
                        msg("  [方案A] 成功! %d/%d"
                            % (nv, len(school_gdf)))
                        if nv > len(school_gdf) * 0.5:
                            scheme_used = "A"
                    else:
                        msg("  [方案A] 叠加空->降级B")
                except Exception as ex:
                    msg("  [方案A] 失败:%s->降级B" % ex)

            if scheme_used != "A":
                msg("  [方案B] 建筑密度x服务区x楼层...")
                school_gdf = school_gdf.merge(
                    isochrone_all[
                        [COL_SCHOOL_ID, 'geometry']]
                    .rename(columns={
                        'geometry': 'service_geom'}),
                    on=COL_SCHOOL_ID, how='left')
                # 关键修复：只传纯 geometry
                svc_geom_only = gpd.GeoDataFrame(
                    geometry=(
                        school_gdf['service_geom'].values),
                    crs=TARGET_CRS)
                bld_stats = safe_zonal_stats(
                    svc_geom_only, bld_fixed,
                    stats=["mean"],
                    nodata_val=BLD_NODATA)
                school_gdf[
                    'build_density_service'] = [
                    float(s['mean'])
                    if (s['mean'] is not None
                        and not np.isnan(s['mean'])
                        and s['mean'] > 0) else 0.0
                    for s in bld_stats]
                school_gdf['service_area_m2'] = (
                    school_gdf['service_geom'].area)
                school_gdf['Area_bld'] = (
                    school_gdf['build_density_service']
                    * school_gdf['service_area_m2']
                    * avg_floors)
                school_gdf['Supply'] = (
                    school_gdf['Area_bld']
                    / school_gdf['S_per'])
                msg("  [方案B] Supply:[%.0f, %.0f]"
                    % (school_gdf['Supply'].min(),
                       school_gdf['Supply'].max()))
                scheme_used = "B"

            def smart_fallback(row):
                supply   = row['Supply']
                level    = str(row[COL_LEVEL])
                is_p     = '小学' in level
                area_km2 = (
                    isochrone_all.loc[
                        isochrone_all[COL_SCHOOL_ID]
                        == row[COL_SCHOOL_ID],
                        'geometry'].area.sum() / 1e6
                    if 'service_geom' not in row else 0)
                is_small = area_km2 < 1.0
                if supply <= 0:
                    if is_p and is_small:
                        return FALLBACK_PRIMARY_SMALL
                    elif is_p:
                        return FALLBACK_PRIMARY_LARGE
                    elif is_small:
                        return FALLBACK_MIDDLE_SMALL
                    else:
                        return FALLBACK_MIDDLE_LARGE
                elif supply < 50:
                    return supply * 3.0
                elif supply > 10000:
                    return 10000
                return supply

            school_gdf['Supply_final'] = (
                school_gdf.apply(smart_fallback, axis=1))
            fb_count = int((
                (school_gdf['Supply_final']
                 == FALLBACK_PRIMARY_SMALL)
                | (school_gdf['Supply_final']
                   == FALLBACK_PRIMARY_LARGE)
                | (school_gdf['Supply_final']
                   == FALLBACK_MIDDLE_SMALL)
                | (school_gdf['Supply_final']
                   == FALLBACK_MIDDLE_LARGE)).sum())
            msg("  [Supply] 方案=%s fallback=%d/%d"
                " 范围=[%.0f,%.0f]"
                % (scheme_used, fb_count,
                   len(school_gdf),
                   school_gdf['Supply_final'].min(),
                   school_gdf['Supply_final'].max()))

            drop_cols = ['service_geom', 'service_area_m2',
                         'build_density_service',
                         'Area_bld', 'Supply',
                         'total_floor_area']
            for c in drop_cols:
                if c in school_gdf.columns:
                    school_gdf = school_gdf.drop(columns=c)
            return school_gdf

        # ══════════════════════════════════════════
        # 数据读取
        # ══════════════════════════════════════════
        msg("=" * 60)
        msg("===== 学区供需诊断 数据读取 =====")
        msg("=" * 60)

        school_df = pd.read_csv(SCHOOL_CSV,
                                 encoding='utf-8-sig')
        msg("学校数据:%d 行  字段:%s"
            % (len(school_df), list(school_df.columns)))

        required_cols = [COL_SCHOOL_ID, COL_NAME,
                         COL_LEVEL, COL_LNG, COL_LAT,
                         COL_X_COORD, COL_Y_COORD]
        missing = [c for c in required_cols
                   if c not in school_df.columns]
        if missing:
            _loge(
                "school_data.csv 缺少字段:%s" % missing)
            return

        school_df[COL_SCHOOL_ID] = (
            school_df[COL_SCHOOL_ID]
            .astype(str).str.strip())

        # [期刊版修复] 历史 x_coord/y_coord 误填为经纬度副本，若直接当 EPSG:4526
        # 投影坐标会令距离特征出现 ~3.87e7 m 的天文值。统一以经纬度为唯一权威坐标源，
        # 先按 EPSG:4326 构造几何再重投影到 TARGET_CRS，并把正确投影米坐标回填 x/y_coord。
        school_df['geometry'] = school_df.apply(
            lambda r: Point(r[COL_LNG], r[COL_LAT]), axis=1)
        school_gdf = gpd.GeoDataFrame(
            school_df, geometry='geometry',
            crs="EPSG:4326").to_crs(TARGET_CRS)
        school_gdf[COL_X_COORD] = school_gdf.geometry.x
        school_gdf[COL_Y_COORD] = school_gdf.geometry.y
        msg("  统一由经纬度(EPSG:4326)重投影至 %s，并回填 x/y_coord" % TARGET_CRS)

        if COL_FID not in school_gdf.columns:
            school_gdf[COL_FID] = range(len(school_gdf))

        if COL_STUDENTS in school_gdf.columns:
            school_gdf[COL_STUDENTS] = pd.to_numeric(
                school_gdf[COL_STUDENTS], errors='coerce')
            msg("  学生数:[%.0f, %.0f]  空值:%d"
                % (school_gdf[COL_STUDENTS].min(),
                   school_gdf[COL_STUDENTS].max(),
                   int(school_gdf[COL_STUDENTS]
                       .isna().sum())))
        else:
            school_gdf[COL_STUDENTS] = np.nan

        poi_df = pd.read_csv(POI_CSV, encoding='utf-8-sig')
        msg("POI 数据:%d 行" % len(poi_df))
        poi_gdf = gpd.GeoDataFrame(
            poi_df,
            geometry=gpd.points_from_xy(
                poi_df['lng'], poi_df['lat']),
            crs="EPSG:4326").to_crs(TARGET_CRS)

        isochrone_elem = gpd.read_file(
            ISOCHRONE_ELEM).to_crs(TARGET_CRS)
        isochrone_mid  = gpd.read_file(
            ISOCHRONE_MID).to_crs(TARGET_CRS)
        isochrone_all  = pd.concat(
            [isochrone_elem, isochrone_mid],
            ignore_index=True)
        msg("服务区总数:%d 条" % len(isochrone_all))

        if COL_SCHOOL_ID not in isochrone_all.columns:
            _loge(
                "服务区 shp 缺少 school_id 字段")
            return

        isochrone_all[COL_SCHOOL_ID] = (
            isochrone_all[COL_SCHOOL_ID]
            .astype(str).str.strip())
        isochrone_all = isochrone_all.drop_duplicates(
            subset=COL_SCHOOL_ID, keep='first')

        fishnet        = gpd.read_file(
            FISHNET_SHP).to_crs(TARGET_CRS)
        vitality_field = 'vitality'

        # school_id 配对诊断
        msg("\n" + "=" * 60)
        msg("===== school_id 配对诊断 =====")
        msg("=" * 60)
        s_ids = set(school_gdf[COL_SCHOOL_ID].unique())
        v_ids = set(isochrone_all[COL_SCHOOL_ID].unique())
        msg("学校:%d  服务区:%d  匹配:%d"
            % (len(s_ids), len(v_ids),
               len(s_ids & v_ids)))
        if s_ids - v_ids:
            msg("未匹配 %d 所:" % len(s_ids - v_ids))
            for sid in sorted(s_ids - v_ids):
                nm = school_gdf.loc[
                    school_gdf[COL_SCHOOL_ID] == sid,
                    COL_NAME].values[0]
                msg("  - [%s] %s" % (sid, nm))

        # ══════════════════════════════════════════
        # 栅格预处理
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== 栅格预处理 =====")
        msg("=" * 60)
        raster_res_info_list = []

        msg("\n[WorldPop] 修复 nodata...")
        wp_fixed_path = os.path.join(OUTPUT_DIR,
                                      "WorldPop_fixed.tif")
        fix_raster_nodata(WORLDPOP_TIF,
                           wp_fixed_path, -99999)
        worldpop_fixed = ensure_raster_crs(
            wp_fixed_path, TARGET_CRS)
        raster_res_info_list.append(
            get_raster_resolution_info(
                worldpop_fixed, "WorldPop人口密度"))

        msg("\n[路网密度] 优先使用 EPSG:4526 版本...")
        if (ROAD_DENSITY_TIF_4526
                and os.path.exists(ROAD_DENSITY_TIF_4526)):
            with rasterio.open(
                    ROAD_DENSITY_TIF_4526) as src:
                exist_epsg = (src.crs.to_epsg()
                               if src.crs else None)
            if exist_epsg == 4526:
                road_fixed = ROAD_DENSITY_TIF_4526
                msg("  EPSG:4526 版本直接使用")
            else:
                msg("  文件 EPSG 为%s，重新重投影..."
                    % exist_epsg)
                road_fp = os.path.join(
                    OUTPUT_DIR,
                    "road_density_fixed.tif")
                fix_raster_nodata(
                    ROAD_DENSITY_TIF_4526,
                    road_fp, -1.797693e+308)
                road_fixed = ensure_raster_crs(
                    road_fp, TARGET_CRS)
        elif (ROAD_DENSITY_TIF_RAW
              and os.path.exists(ROAD_DENSITY_TIF_RAW)):
            msg("  4526 版本不存在，从原始文件处理...")
            road_fp = os.path.join(
                OUTPUT_DIR, "road_density_fixed.tif")
            fix_raster_nodata(
                ROAD_DENSITY_TIF_RAW,
                road_fp, -1.797693e+308)
            road_fixed = ensure_raster_crs(
                road_fp, TARGET_CRS)
        else:
            _loge("未提供任何路网密度栅格")
            return
        raster_res_info_list.append(
            get_raster_resolution_info(road_fixed,
                                       "路网密度"))

        msg("\n[建筑密度] 黑边修复...")
        bld_fixed_path = os.path.join(
            OUTPUT_DIR, "building_density_fixed.tif")
        bld_fixed = fix_building_density(
            BUILDING_DEN_TIF, bld_fixed_path,
            study_area_shp=STUDY_AREA_SHP,
            target_crs=TARGET_CRS)
        bld_fixed = ensure_raster_crs(
            bld_fixed, TARGET_CRS)
        raster_res_info_list.append(
            get_raster_resolution_info(bld_fixed,
                                       "建筑密度"))
        print_raster_resolution_summary(
            raster_res_info_list)

        # ══════════════════════════════════════════
        # Step1: D1 活力指数
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== Step1: D1 活力指数（面积加权）=====")
        msg("=" * 60)
        D1_df = calc_d1_weighted(
            isochrone_gdf=isochrone_all,
            fishnet_gdf=fishnet,
            vitality_field=vitality_field,
            id_field=COL_SCHOOL_ID,
            name_field=(
                'School_Nam'
                if 'School_Nam' in isochrone_all.columns
                else None))

        missed_d1 = (set(isochrone_all[COL_SCHOOL_ID])
                     - set(D1_df[COL_SCHOOL_ID]))
        if missed_d1:
            msg("\n  最近邻兜底 %d 所..."
                % len(missed_d1))
            fb_df = d1_nearest_fallback(
                missed_d1, isochrone_all, fishnet,
                vitality_field, COL_SCHOOL_ID)
            D1_df = pd.concat(
                [D1_df, fb_df], ignore_index=True)
        msg("D1最终覆盖:%d/%d  [%.4f, %.4f]"
            % (len(D1_df), len(isochrone_all),
               D1_df['D1_vitality'].min(),
               D1_df['D1_vitality'].max()))

        # ══════════════════════════════════════════
        # Step2: D2 承载力指数
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== Step2: D2 承载力指数 =====")
        msg("=" * 60)

        RES_KW = (
            '居住小区'
            '|住宅小区'
            '|社区|居民区'
            '|花园|家园'
            '|里|蘓|公馆'
            '|庄园|别墅|公寓')
        poi_residential = poi_gdf[
            poi_gdf['cat_name'].str.contains(
                RES_KW, na=False)]
        joined_res = gpd.sjoin(
            poi_residential,
            isochrone_all[[COL_SCHOOL_ID, 'geometry']],
            how='inner', predicate='within')
        res_count = joined_res.groupby(
            COL_SCHOOL_ID).size().reset_index(
            name='res_poi_count')

        EDU_KW = (
            '科研|教育|学校'
            '|培训|中学|小学'
            '|幼儿园|大学'
            '|学院|研究院'
            '|科技馆|图书馆')
        poi_edu = (
            poi_gdf[poi_gdf['is_edu'] == 1]
            if 'is_edu' in poi_df.columns
            else poi_gdf[poi_gdf['cat_name'].str.contains(
                EDU_KW, na=False)])
        joined_edu = gpd.sjoin(
            poi_edu,
            isochrone_all[[COL_SCHOOL_ID, 'geometry']],
            how='inner', predicate='within')
        edu_count = joined_edu.groupby(
            COL_SCHOOL_ID).size().reset_index(
            name='edu_poi_count')

        msg("提取路网密度...")
        # 关键修复：只传纯 geometry 列
        iso_geom_only = gpd.GeoDataFrame(
            geometry=isochrone_all.geometry.values,
            crs=TARGET_CRS)
        road_stats = safe_zonal_stats(
            iso_geom_only, road_fixed,
            stats=["mean"], nodata_val=ROAD_NODATA)
        road_mean = pd.DataFrame({
            COL_SCHOOL_ID: (
                isochrone_all[COL_SCHOOL_ID].values),
            'road_density_mean': [
                float(s['mean'])
                if (s['mean'] is not None
                    and not np.isinf(s['mean']))
                else 0.0
                for s in road_stats]})

        isochrone_all['area_km2'] = (
            isochrone_all.geometry.area / 1e6)
        D2_df = (
            isochrone_all[[COL_SCHOOL_ID, 'area_km2']]
            .merge(res_count, on=COL_SCHOOL_ID,
                   how='left')
            .merge(edu_count, on=COL_SCHOOL_ID,
                   how='left')
            .merge(road_mean, on=COL_SCHOOL_ID,
                   how='left')
            .fillna(0))
        D2_df['sub1_res_density'] = (
            D2_df['res_poi_count']
            / (D2_df['area_km2'] + 1e-9))
        D2_df['sub2_edu_density'] = (
            D2_df['edu_poi_count']
            / (D2_df['area_km2'] + 1e-9))
        D2_df['sub3_road_density'] = (
            D2_df['road_density_mean'])
        sub_cols   = ['sub1_res_density',
                      'sub2_edu_density',
                      'sub3_road_density']
        weights_D2 = entropy_weight(D2_df, sub_cols)
        msg("D2权重 -> 居住=%.4f 教育=%.4f 路网=%.4f"
            % (weights_D2[0], weights_D2[1],
               weights_D2[2]))
        X_sub      = D2_df[sub_cols].values
        X_sub_norm = (
            (X_sub - X_sub.min(axis=0))
            / (X_sub.max(axis=0)
               - X_sub.min(axis=0) + 1e-9))
        D2_df['D2_capacity'] = (
            (X_sub_norm * weights_D2).sum(axis=1))
        msg("D2覆盖:%d  [%.4f, %.4f]"
            % (len(D2_df),
               D2_df['D2_capacity'].min(),
               D2_df['D2_capacity'].max()))

        # ══════════════════════════════════════════
        # Step3: D3 供需压力指数
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== Step3: D3 供需压力指数 =====")
        msg("=" * 60)

        msg("[3-A] 提取服务区人口...")
        # 关键修复：只传纯 geometry 列
        iso_geom_only2 = gpd.GeoDataFrame(
            geometry=isochrone_all.geometry.values,
            crs=TARGET_CRS)
        pop_stats = safe_zonal_stats(
            iso_geom_only2, worldpop_fixed,
            stats=["sum"], nodata_val=WORLDPOP_NODATA)
        pop_df = pd.DataFrame({
            COL_SCHOOL_ID: (
                isochrone_all[COL_SCHOOL_ID].values),
            'pop_total': [
                float(s['sum'])
                if (s['sum'] is not None
                    and s['sum'] > 0) else 0.0
                for s in pop_stats]})
        pop_df['Demand'] = (
            pop_df['pop_total'] * RATIO_6_14)
        msg("  pop_total:[%.1f, %.1f]"
            % (pop_df['pop_total'].min(),
               pop_df['pop_total'].max()))

        msg("\n[3-B] Supply 估算...")
        school_gdf['S_per'] = (
            school_gdf[COL_LEVEL].apply(get_s_per))
        bld_shp_ok = (BUILDINGS_SHP is not None
                      and os.path.exists(BUILDINGS_SHP))
        school_gdf = calculate_supply(
            school_gdf=school_gdf,
            isochrone_all=isochrone_all,
            bld_fixed=bld_fixed,
            buildings_shp=(BUILDINGS_SHP
                            if bld_shp_ok else None),
            avg_floors=AVG_FLOORS)

        msg("\n[3-C] 合并需求，计算 D3...")
        school_gdf = school_gdf.merge(
            pop_df[[COL_SCHOOL_ID,
                    'pop_total', 'Demand']],
            on=COL_SCHOOL_ID, how='left')
        unmatched_demand = int(
            school_gdf['Demand'].isna().sum())
        if unmatched_demand > 0:
            med_demand = pop_df[
                pop_df['Demand'] > 0]['Demand'].median()
            school_gdf['Demand'] = (
                school_gdf['Demand'].fillna(med_demand))
            school_gdf['pop_total'] = (
                school_gdf['pop_total']
                .fillna(med_demand / RATIO_6_14))
            msg("  中位数填补 %d 所需求缺失"
                % unmatched_demand)

        school_gdf['D3_pressure'] = (
            school_gdf['Demand']
            / (school_gdf['Supply_final'] + 1e-9))
        msg("  D3:[%.4f, %.4f]  偏度:%.2f"
            % (school_gdf['D3_pressure'].min(),
               school_gdf['D3_pressure'].max(),
               school_gdf['D3_pressure'].skew()))
        d3_diag_raw = diagnose_d3_quality(
            school_gdf['D3_pressure'], "D3(原始)")

        # D3 Winsorization
        msg("\n" + "=" * 60)
        msg("===== D3 Winsorization 缩尾处理 =====")
        msg("=" * 60)
        school_gdf['D3_pressure_raw'] = (
            school_gdf['D3_pressure'].copy())
        upper_bound = None
        if D3_WINSORIZE:
            d3w, upper_bound, n_clipped = winsorize_d3(
                school_gdf['D3_pressure'],
                D3_WINSORIZE_UPPER)
            school_gdf['D3_pressure'] = d3w
            msg("  P%d上限:%.4f  截断:%d 所"
                % (int(D3_WINSORIZE_UPPER * 100),
                   upper_bound, n_clipped))
            if n_clipped > 0:
                cm = (school_gdf['D3_pressure_raw']
                      > upper_bound)
                for _, r in school_gdf[cm].iterrows():
                    msg("    [%s] %s: %.4f -> %.4f"
                        % (r[COL_SCHOOL_ID],
                           r[COL_NAME],
                           r['D3_pressure_raw'],
                           upper_bound))
            msg("  处理后 均值=%.4f 偏度=%.4f"
                % (school_gdf['D3_pressure'].mean(),
                   school_gdf['D3_pressure'].skew()))
        else:
            msg("  D3_WINSORIZE=False，保留原始值")
        msg("  总学校数:%d 所" % len(school_gdf))
        d3_diag_final = diagnose_d3_quality(
            school_gdf['D3_pressure'],
            "D3(Winsorization后)")

        # ══════════════════════════════════════════
        # Step4: ECFI 综合指数
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== Step4: ECFI 综合指数 =====")
        msg("=" * 60)
        ecfi_df = (
            D1_df
            .merge(D2_df[[COL_SCHOOL_ID,
                           'D2_capacity']],
                   on=COL_SCHOOL_ID, how='left')
            .merge(
                school_gdf[[COL_SCHOOL_ID,
                             'D3_pressure']]
                .drop_duplicates(COL_SCHOOL_ID),
                on=COL_SCHOOL_ID, how='left'))

        ecfi_df['D1_norm'] = (
            (ecfi_df['D1_vitality']
             - ecfi_df['D1_vitality'].min())
            / (ecfi_df['D1_vitality'].max()
               - ecfi_df['D1_vitality'].min() + 1e-9))
        ecfi_df['D2_norm'] = (
            (ecfi_df['D2_capacity']
             - ecfi_df['D2_capacity'].min())
            / (ecfi_df['D2_capacity'].max()
               - ecfi_df['D2_capacity'].min() + 1e-9))
        ecfi_df['D3_norm'] = (
            (ecfi_df['D3_pressure'].max()
             - ecfi_df['D3_pressure'])
            / (ecfi_df['D3_pressure'].max()
               - ecfi_df['D3_pressure'].min() + 1e-9))
        ecfi_cols    = ['D1_norm', 'D2_norm', 'D3_norm']
        weights_ecfi = entropy_weight(ecfi_df, ecfi_cols)
        msg("ECFI权重 -> D1=%.4f D2=%.4f D3=%.4f"
            % (weights_ecfi[0], weights_ecfi[1],
               weights_ecfi[2]))
        ecfi_df['ECFI'] = (
            (ecfi_df[ecfi_cols].values
             * weights_ecfi).sum(axis=1))
        msg("ECFI:[%.4f, %.4f]  有效:%d/%d"
            % (ecfi_df['ECFI'].min(),
               ecfi_df['ECFI'].max(),
               int(ecfi_df['ECFI'].notna().sum()),
               len(ecfi_df)))

        # Step5: Priority
        msg("\n===== Step5: 优先级 =====")
        ecfi_df['priority_score'] = (
            0.5 * ecfi_df['D3_norm']
            + 0.3 * ecfi_df['D2_norm']
            + 0.2 * (1 - ecfi_df['ECFI']))
        ecfi_df['priority_score'] = (
            ecfi_df['priority_score']
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0))
        ps_min = ecfi_df['priority_score'].min()
        ps_max = ecfi_df['priority_score'].max()
        ecfi_df['priority_score'] = (
            (ecfi_df['priority_score'] - ps_min)
            / (ps_max - ps_min + 1e-9))
        ecfi_df['priority_rank'] = (
            ecfi_df['priority_score']
            .rank(ascending=False, method='min')
            .fillna(0).astype(int))

        # Step6: 分类
        msg("\n===== Step6: 学校分类 =====")
        q75 = ecfi_df['ECFI'].quantile(0.75)
        q50 = ecfi_df['ECFI'].quantile(0.50)
        q25 = ecfi_df['ECFI'].quantile(0.25)
        msg("ECFI分位数 Q25=%.4f Q50=%.4f Q75=%.4f"
            % (q25, q50, q75))

        def ecfi_class(v):
            if pd.isna(v):
                return '低'
            if v >= q75:
                return '极高'
            elif v >= q50:
                return '高'
            elif v >= q25:
                return '中'
            return '低'

        def classify_school(row):
            if pd.isna(row['ECFI']):
                return 'IV_待提升型'
            if row['ECFI'] >= q75:
                base = 'I_综合较优型'
            elif row['ECFI'] >= q50:
                base = 'II_中等偏上型'
            elif row['ECFI'] >= q25:
                base = ('III_中等'
                        '偏下型')
            else:
                base = 'IV_待提升型'
            sc = row.get('student_count', 0)
            if pd.isna(sc) or sc is None:
                sc = 0
            if (base == 'IV_待提升型'
                    and sc > 2000):
                base = ('III_中等'
                        '偏下型')
            if (base == 'II_中等'
                    '偏上型'
                    and 0 < sc < 500):
                base = ('III_中等'
                        '偏下型')
            return base

        strategy_map = {
            'I_综合较优型':   'S1',
            'II_中等偏上型':  'S2',
            'III_中等偏下型': 'S3',
            'IV_待提升型':         'S4'}
        ecfi_df['strategy_new'] = ecfi_df.apply(
            classify_school, axis=1)

        # ══════════════════════════════════════════
        # 扩展特征计算
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== 扩展特征计算 =====")
        msg("=" * 60)
        cluster_pct_df = calc_cluster_proportions(
            isochrone_all, fishnet, COL_SCHOOL_ID)

        location_df = calc_location_features(
            school_gdf=school_gdf,
            river_shp=(RIVER_SHP
                        if RIVER_SHP
                        and os.path.exists(RIVER_SHP)
                        else None),
            city_center=CITY_CENTER)

        poi_feat_df = (
            calc_poi_features_with_spatial_fallback(
                isochrone_gdf=isochrone_all,
                poi_gdf=poi_gdf,
                school_gdf=school_gdf,
                id_field=COL_SCHOOL_ID,
                buffer_expand_factors=(1.5, 2.0, 3.0),
                k_neighbors=K_NEIGHBORS))

        isochrone_all['perimeter'] = (
            isochrone_all.geometry.length)
        isochrone_all['compactness'] = (
            4 * np.pi * isochrone_all.geometry.area
            / (isochrone_all['perimeter'] ** 2 + 1e-9))
        compact_df = isochrone_all[
            [COL_SCHOOL_ID, 'compactness']].copy()
        msg("  紧凑度:[%.4f, %.4f]"
            % (compact_df['compactness'].min(),
               compact_df['compactness'].max()))

        # ══════════════════════════════════════════
        # Step7: 构建输出表
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== Step7: 构建输出表 =====")
        msg("=" * 60)

        output_df = school_gdf[[
            COL_SCHOOL_ID, COL_FID, COL_NAME,
            COL_LEVEL, COL_LNG, COL_LAT]].copy()
        output_df.columns = [
            'school_id', 'School_ID', 'School_Name',
            'Level', 'longitude', 'latitude']
        output_df['Level'] = output_df.apply(
            lambda r: map_level(
                r['School_Name'], r['Level']), axis=1)

        output_df = output_df.merge(
            ecfi_df[[COL_SCHOOL_ID, 'D1_vitality',
                     'D2_capacity', 'D3_pressure',
                     'ECFI', 'priority_score',
                     'priority_rank', 'strategy_new']],
            on=COL_SCHOOL_ID, how='left')

        d3_raw_df = (
            school_gdf[[COL_SCHOOL_ID,
                         'D3_pressure_raw']]
            .drop_duplicates(COL_SCHOOL_ID))
        output_df = output_df.merge(
            d3_raw_df, on=COL_SCHOOL_ID, how='left')

        output_df['geometry_wkt'] = output_df.apply(
            lambda r: "POINT (%s %s)"
                      % (r['longitude'],
                         r['latitude']), axis=1)

        output_df = output_df.merge(
            isochrone_all[[COL_SCHOOL_ID,
                           'area_km2', 'perimeter']],
            on=COL_SCHOOL_ID, how='left')
        output_df.rename(
            columns={'area_km2': 'service_area_km2'},
            inplace=True)

        if (len(cluster_pct_df) > 0
                and 'pct_C1' in cluster_pct_df.columns):
            output_df = output_df.merge(
                cluster_pct_df,
                on=COL_SCHOOL_ID, how='left')
            msg("  社区类型占比已合并")
        else:
            for cid in range(1, 7):
                output_df['pct_C%d' % cid] = 0.0
            output_df['dominant_cluster'] = -1

        if len(location_df) > 0:
            output_df = output_df.merge(
                location_df,
                on=COL_SCHOOL_ID, how='left')
            msg("  区位特征已合并")
        else:
            output_df['dist_to_center'] = 0.0
            output_df['dist_to_river']  = 0.0

        if len(poi_feat_df) > 0:
            output_df = output_df.merge(
                poi_feat_df,
                on=COL_SCHOOL_ID, how='left')
            msg("  POI 特征已合并")
        else:
            output_df['poi_diversity']     = 0.0
            output_df['residential_ratio'] = 0.0
            output_df['fill_method']       = 'missing'

        if len(compact_df) > 0:
            output_df = output_df.merge(
                compact_df,
                on=COL_SCHOOL_ID, how='left')
        else:
            output_df['compactness'] = 0.0

        output_df['ECFI_class'] = (
            output_df['ECFI'].apply(ecfi_class))
        output_df['strategy_new_code'] = (
            output_df['strategy_new'].map(strategy_map))

        if COL_STUDENTS in school_df.columns:
            output_df = output_df.merge(
                school_df[[COL_SCHOOL_ID, COL_STUDENTS]],
                on=COL_SCHOOL_ID, how='left')
            output_df.rename(
                columns={COL_STUDENTS: 'student_count'},
                inplace=True)
        else:
            output_df['student_count'] = None

        for col in ['poi_diversity', 'residential_ratio']:
            n_null = int(output_df[col].isna().sum())
            if n_null > 0:
                fb = (output_df[col].median()
                      if output_df[col].notna().any()
                      else 0.0)
                output_df[col] = output_df[col].fillna(fb)
                msg("  [POI兜底] %s 补%d个=%.4f"
                    % (col, n_null, fb))

        msg("  poi_diversity空值:%d"
            % int(output_df['poi_diversity'].isna().sum()))
        msg("  residential_ratio空值:%d"
            % int(output_df[
                'residential_ratio'].isna().sum()))

        final_cols = [
            'school_id', 'School_ID', 'School_Name',
            'Level', 'geometry_wkt',
            'ECFI', 'ECFI_class',
            'D1_vitality', 'D2_capacity', 'D3_pressure',
            'D3_pressure_raw',
            'priority_score', 'priority_rank',
            'strategy_new', 'strategy_new_code',
            'service_area_km2', 'student_count',
            'pct_C1', 'pct_C2', 'pct_C3',
            'pct_C4', 'pct_C5', 'pct_C6',
            'dominant_cluster',
            'dist_to_center', 'dist_to_river',
            'poi_diversity', 'residential_ratio',
            'compactness', 'fill_method']
        output_final = output_df[final_cols].copy()

        # ══════════════════════════════════════════
        # 空值学校空间邻近补充
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== 空值学校空间邻近补充 =====")
        msg("=" * 60)
        empty_mask = output_final['ECFI'].isna()
        empty_ids  = (output_final[empty_mask]
                      [COL_SCHOOL_ID].tolist())
        empty_nms  = (output_final[empty_mask]
                      ['School_Name'].tolist())
        msg("需补充:%d 所" % len(empty_ids))
        for sid, nm in zip(empty_ids, empty_nms):
            msg("  [%s] %s" % (sid, nm))

        if len(empty_ids) > 0:
            empty_school_gdf = school_gdf[
                school_gdf[COL_SCHOOL_ID].isin(
                    empty_ids)
            ].copy().reset_index(drop=True)

            empty_school_gdf['service_geom'] = (
                empty_school_gdf.geometry.buffer(
                    WALK_BUFFER))
            service_gdf = (
                empty_school_gdf.copy()
                .set_geometry('service_geom'))
            service_gdf['area_km2'] = (
                service_gdf['service_geom'].area / 1e6)
            # 只保留 school_id + geometry
            service_gdf_geom = gpd.GeoDataFrame(
                {COL_SCHOOL_ID:
                     service_gdf[COL_SCHOOL_ID].values},
                geometry=(
                    service_gdf['service_geom'].values),
                crs=TARGET_CRS)
            msg("  生成 %dm 替代服务区" % WALK_BUFFER)

            msg("\n  重算 D1...")
            try:
                d1_new = calc_d1_weighted(
                    service_gdf_geom, fishnet,
                    vitality_field, COL_SCHOOL_ID)
                miss_d1n = (
                    set(service_gdf[COL_SCHOOL_ID])
                    - set(d1_new[COL_SCHOOL_ID]))
                if miss_d1n:
                    fb2 = d1_nearest_fallback(
                        miss_d1n, service_gdf_geom,
                        fishnet, vitality_field,
                        COL_SCHOOL_ID)
                    d1_new = pd.concat(
                        [d1_new, fb2],
                        ignore_index=True)
                d1_new.columns = [COL_SCHOOL_ID,
                                   'D1_vitality_new']
            except Exception as ex:
                msg("  D1 失败:%s" % ex)
                d1_new = pd.DataFrame(columns=[
                    COL_SCHOOL_ID, 'D1_vitality_new'])

            msg("  重算 D2...")
            try:
                j_res2 = gpd.sjoin(
                    poi_residential,
                    service_gdf_geom,
                    how='inner', predicate='within')
                res_count2 = j_res2.groupby(
                    COL_SCHOOL_ID).size().reset_index(
                    name='res_poi_count')
            except Exception:
                res_count2 = pd.DataFrame(columns=[
                    COL_SCHOOL_ID, 'res_poi_count'])
            try:
                j_edu2 = gpd.sjoin(
                    poi_edu, service_gdf_geom,
                    how='inner', predicate='within')
                edu_count2 = j_edu2.groupby(
                    COL_SCHOOL_ID).size().reset_index(
                    name='edu_poi_count')
            except Exception:
                edu_count2 = pd.DataFrame(columns=[
                    COL_SCHOOL_ID, 'edu_poi_count'])

            # 关键修复：只传纯 geometry
            sgg_geom = gpd.GeoDataFrame(
                geometry=(
                    service_gdf_geom.geometry.values),
                crs=TARGET_CRS)
            road_stats2 = safe_zonal_stats(
                sgg_geom, road_fixed,
                stats=["mean"], nodata_val=ROAD_NODATA)
            road_mean2 = pd.DataFrame({
                COL_SCHOOL_ID: (
                    service_gdf[COL_SCHOOL_ID].values),
                'road_density_mean': [
                    float(s['mean'])
                    if (s['mean'] is not None
                        and not np.isinf(s['mean']))
                    else 0.0
                    for s in road_stats2]})

            d2_new = (
                service_gdf[[COL_SCHOOL_ID,
                              'area_km2']].copy()
                .merge(res_count2, on=COL_SCHOOL_ID,
                       how='left')
                .merge(edu_count2, on=COL_SCHOOL_ID,
                       how='left')
                .merge(road_mean2, on=COL_SCHOOL_ID,
                       how='left')
                .fillna(0))
            d2_new['sub1_res_density'] = (
                d2_new['res_poi_count']
                / (d2_new['area_km2'] + 1e-9))
            d2_new['sub2_edu_density'] = (
                d2_new['edu_poi_count']
                / (d2_new['area_km2'] + 1e-9))
            d2_new['sub3_road_density'] = (
                d2_new['road_density_mean'])
            ref_min = D2_df[sub_cols].min().values
            ref_max = D2_df[sub_cols].max().values
            X_new_norm = (
                (d2_new[sub_cols].values - ref_min)
                / (ref_max - ref_min + 1e-9))
            d2_new['D2_capacity_new'] = (
                (X_new_norm * weights_D2).sum(axis=1))

            msg("  重算 D3...")
            # 关键修复：只传纯 geometry
            pop_stats2 = safe_zonal_stats(
                sgg_geom, worldpop_fixed,
                stats=["sum"],
                nodata_val=WORLDPOP_NODATA)
            pop_new = pd.DataFrame({
                COL_SCHOOL_ID: (
                    service_gdf[COL_SCHOOL_ID].values),
                'pop_total': [
                    float(s['sum'])
                    if (s['sum'] is not None
                        and s['sum'] > 0) else 0.0
                    for s in pop_stats2]})
            pop_new['Demand'] = (
                pop_new['pop_total'] * RATIO_6_14)
            supply_ref = (
                school_gdf[[COL_SCHOOL_ID,
                             'Supply_final']]
                .drop_duplicates(COL_SCHOOL_ID))
            pop_new = pop_new.merge(
                supply_ref, on=COL_SCHOOL_ID,
                how='left')
            pop_new['D3_pressure_new'] = (
                pop_new['Demand']
                / (pop_new['Supply_final'] + 1e-9))
            if D3_WINSORIZE and upper_bound is not None:
                pop_new['D3_pressure_new'] = (
                    pop_new['D3_pressure_new'].clip(
                        upper=upper_bound))

            msg("\n  写入补充值...")
            for _, row in empty_school_gdf.iterrows():
                sid  = row[COL_SCHOOL_ID]
                mask = (output_final[COL_SCHOOL_ID]
                        == sid)
                r1 = d1_new[
                    d1_new[COL_SCHOOL_ID] == sid]
                if len(r1) > 0:
                    output_final.loc[
                        mask, 'D1_vitality'] = (
                        r1['D1_vitality_new'].values[0])
                r2 = d2_new[
                    d2_new[COL_SCHOOL_ID] == sid]
                if len(r2) > 0:
                    output_final.loc[
                        mask, 'D2_capacity'] = (
                        r2['D2_capacity_new'].values[0])
                r3 = pop_new[
                    pop_new[COL_SCHOOL_ID] == sid]
                if len(r3) > 0:
                    output_final.loc[
                        mask, 'D3_pressure'] = (
                        r3['D3_pressure_new'].values[0])
                ra = service_gdf[
                    service_gdf[COL_SCHOOL_ID] == sid]
                if len(ra) > 0:
                    output_final.loc[
                        mask, 'service_area_km2'] = (
                        ra['area_km2'].values[0])

                d1v = output_final.loc[
                    mask, 'D1_vitality'].values[0]
                d2v = output_final.loc[
                    mask, 'D2_capacity'].values[0]
                d3v = output_final.loc[
                    mask, 'D3_pressure'].values[0]
                msg("  [%s]%s -> D1=%.4f D2=%.4f"
                    " D3=%.4f"
                    % (sid, row[COL_NAME],
                       d1v, d2v, d3v))

            for col in ['D1_vitality', 'D2_capacity',
                        'D3_pressure']:
                null_n = int(
                    output_final[col].isna().sum())
                if null_n > 0:
                    msg("\n  %s 仍%d个空值，最近邻填充..."
                        % (col, null_n))
                    for idx in output_final[
                            output_final[col]
                            .isna()].index:
                        sid = output_final.loc[
                            idx, COL_SCHOOL_ID]
                        sp  = school_gdf[
                            school_gdf[COL_SCHOOL_ID]
                            == sid]
                        if len(sp) == 0:
                            continue
                        pt  = sp.geometry.values[0]
                        vm  = output_final[col].notna()
                        vs  = school_gdf[
                            school_gdf[COL_SCHOOL_ID]
                            .isin(output_final[vm][
                                COL_SCHOOL_ID])]
                        if len(vs) == 0:
                            continue
                        nidx = (vs.geometry
                                .distance(pt).idxmin())
                        nsid = vs.loc[
                            nidx, COL_SCHOOL_ID]
                        nval = output_final.loc[
                            output_final[COL_SCHOOL_ID]
                            == nsid, col].values[0]
                        output_final.loc[idx, col] = nval
                        msg("    [%s]%s->[%s]=%.4f"
                            % (sid, col, nsid, nval))

        # ══════════════════════════════════════════
        # 重新计算全部学校 ECFI 与排名
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== 重新计算全部学校 ECFI =====")
        msg("=" * 60)
        d1_min = output_final['D1_vitality'].min()
        d1_max = output_final['D1_vitality'].max()
        d2_min = output_final['D2_capacity'].min()
        d2_max = output_final['D2_capacity'].max()
        d3_min = output_final['D3_pressure'].min()
        d3_max = output_final['D3_pressure'].max()

        D1_nf = ((output_final['D1_vitality'] - d1_min)
                  / (d1_max - d1_min + 1e-9))
        D2_nf = ((output_final['D2_capacity'] - d2_min)
                  / (d2_max - d2_min + 1e-9))
        D3_nf = ((d3_max - output_final['D3_pressure'])
                  / (d3_max - d3_min + 1e-9))

        output_final['ECFI'] = (
            weights_ecfi[0] * D1_nf
            + weights_ecfi[1] * D2_nf
            + weights_ecfi[2] * D3_nf)
        output_final['priority_score'] = (
            0.5 * D3_nf + 0.3 * D2_nf
            + 0.2 * (1 - output_final['ECFI']))
        output_final['priority_score'] = (
            output_final['priority_score']
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0))
        ps_min = output_final['priority_score'].min()
        ps_max = output_final['priority_score'].max()
        output_final['priority_score'] = (
            (output_final['priority_score'] - ps_min)
            / (ps_max - ps_min + 1e-9))
        output_final['priority_rank'] = (
            output_final['priority_score']
            .rank(ascending=False, method='min')
            .fillna(0).astype(int))
        output_final['ECFI_class'] = (
            output_final['ECFI'].apply(ecfi_class))
        output_final['strategy_new'] = (
            output_final.apply(classify_school, axis=1))
        output_final['strategy_new_code'] = (
            output_final['strategy_new'].map(
                strategy_map))

        # ══════════════════════════════════════════
        # 最终输出
        # ══════════════════════════════════════════
        msg("\n" + "=" * 60)
        msg("===== 最终输出 =====")
        msg("=" * 60)
        msg("最终学校数:%d 所" % len(output_final))
        if len(output_final) != 99:
            _logw(
                "期望99所，实际%d所！"
                % len(output_final))

        msg("\n空值统计:")
        check_cols = ['D1_vitality', 'D2_capacity',
                      'D3_pressure', 'ECFI',
                      'poi_diversity',
                      'residential_ratio']
        msg(output_final[check_cols].isnull()
            .sum().to_string())

        if 'fill_method' in output_final.columns:
            msg("\nPOI 填充方法分布:")
            for m, c in (output_final['fill_method']
                         .value_counts().items()):
                icon = "ok" if m == 'normal' else "-"
                msg("  %s %s: %d 所"
                    % (icon, m, c))

        msg("\nD3统计: [%.4f,%.4f] 均值=%.4f 偏度=%.4f"
            % (output_final['D3_pressure'].min(),
               output_final['D3_pressure'].max(),
               output_final['D3_pressure'].mean(),
               output_final['D3_pressure'].skew()))

        output_final.to_csv(
            OUTPUT_CSV, index=False,
            encoding='utf-8-sig')
        msg("\n已保存:%s  共%d行"
            % (OUTPUT_CSV, len(output_final)))

        if output_final['ECFI'].isna().any():
            still = output_final[
                output_final['ECFI'].isna()][
                [COL_SCHOOL_ID,
                 'School_Name']].values.tolist()
            _logw(
                "仍有 ECFI 空值:%s" % still)
        else:
            msg("全部99所学校均有完整数据！")

        msg("\n" + "=" * 60)
        msg("===== 运行总结 =====")
        msg("=" * 60)
        msg("[字段适配] 已正确映射实际字段")
        msg("[P2-1] POI空值地理学空间兜底已启用")
        msg("[P2-2] road density 使用 EPSG:4526 版本")
        msg("[P2-3] 各栅格因子分辨率已输出")
        msg("[GDAL] GDAL_MEM_ENABLE_OPEN 已开启")
        msg("[Pandas] 纯 geometry 传参已启用")
        msg("[输出] %s" % OUTPUT_CSV)
        msg("[学校数] %d 所" % len(output_final))
        msg("完成！")

        # 写回输出参数（模型构建器连线）
        params[24].value = OUTPUT_CSV
        params[25].value = worldpop_fixed
        params[26].value = road_fixed
        params[27].value = bld_fixed
        return


# ══════════════════════════════════════════════════════════════════
# CLI 接线（28 个参数；默认值与原始实现一致）
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="4.4 ECFI 三维教育压力诊断（纯 Python）")
    ap.add_argument("--school-csv", default=str(paths.data_dir() / "school_data.csv"))
    ap.add_argument("--poi-csv", default=str(paths.data_dir() / "POI_data.csv"))
    ap.add_argument("--isochrone-elem", required=True, help="小学服务区面（step43b 生成）")
    ap.add_argument("--isochrone-mid", required=True, help="中学服务区面（step43b 生成）")
    ap.add_argument("--fishnet", required=True, help="社区聚类渔网（step42 生成）")
    ap.add_argument("--road-density-4526", default=None,
                    help="路网密度栅格 4526 版（优先）")
    ap.add_argument("--road-density-raw", default=None, help="路网密度栅格 原始版（备用）")
    ap.add_argument("--worldpop", default=str(paths.data_dir() / "WorldPop_250m_EPSG4526.tif"))
    ap.add_argument("--building-den", default=str(paths.data_dir() / "zhangong_buildings_density.tif"))
    ap.add_argument("--study-area", default=None, help="研究区面（黑边修复用，可选）")
    ap.add_argument("--buildings", default=str(paths.data_dir() / "building_footprint.shp"))
    ap.add_argument("--river", default=str(paths.data_dir() / "river_full.shp"))
    ap.add_argument("--city-center-lng", type=float, default=114.935)
    ap.add_argument("--city-center-lat", type=float, default=25.831)
    ap.add_argument("--ratio-6-14", type=float, default=0.1174)
    ap.add_argument("--s-per-primary", type=float, default=6.5)
    ap.add_argument("--s-per-middle", type=float, default=7.5)
    ap.add_argument("--avg-floors", type=float, default=3.0)
    ap.add_argument("--walk-buffer", type=int, default=800)
    ap.add_argument("--no-d3-winsorize", dest="d3_winsorize", action="store_false")
    ap.add_argument("--d3-winsorize-upper", type=float, default=0.95)
    ap.add_argument("--k-neighbors", type=int, default=3)
    ap.add_argument("--out-dir", default=str(paths.output_dir() / "step44"))
    ap.add_argument("--output-csv-name", default="4.4_school_profile.csv")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [
        _Param(a.school_csv), _Param(a.poi_csv),
        _Param(a.isochrone_elem), _Param(a.isochrone_mid), _Param(a.fishnet),
        _Param(a.road_density_4526), _Param(a.road_density_raw),
        _Param(a.worldpop), _Param(a.building_den),
        _Param(a.study_area), _Param(a.buildings), _Param(a.river),
        _Param(a.city_center_lng), _Param(a.city_center_lat),
        _Param(a.ratio_6_14), _Param(a.s_per_primary), _Param(a.s_per_middle),
        _Param(a.avg_floors), _Param(a.walk_buffer),
        _Param(a.d3_winsorize), _Param(a.d3_winsorize_upper),
        _Param(a.k_neighbors), _Param(str(out_dir)), _Param(a.output_csv_name),
        # 派生输出（updateParameters 命名规则）
        _Param(str(out_dir / a.output_csv_name)),
        _Param(str(out_dir / "WorldPop_fixed.tif")),
        _Param(str(out_dir / "road_density_fixed.tif")),
        _Param(str(out_dir / "building_density_fixed.tif")),
    ]
    tool_44_execute(params, _MsgShim())


if __name__ == "__main__":
    main()
