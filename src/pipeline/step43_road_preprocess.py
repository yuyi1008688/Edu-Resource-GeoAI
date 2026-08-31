# -*- coding: utf-8 -*-
"""
step43_road_preprocess.py — 4.3 路网预处理 + 服务区后处理（纯 Python CLI）

纯 Python CLI：
  - 路网预处理：逐行计算改写为等价向量化列运算；
    权重表（LTS+OSRM 终审 20 类）、时间公式（WalkTime=len/75、Bike_Time=len×Bike/240）、
    禁行规则保持不变；
  - 服务区后处理：geopandas 实现（属性连接→Dissolve→buffer(0)→字段构建）。

数据血缘：
  输入原始路网 shp（44102 要素）
  → 本步骤输出 Prepare（+Walk_Wt/Bike/Bike_Re/Walk_Re/WalkTime/Bike_Time），
    字段口径可与历史基准逐字段比对。

服务区子命令对 step43b 生成的等时圈面做属性连接、融合与字段构建。

示例：
  python src/pipeline/step43_road_preprocess.py preprocess \
      --in src输入路网.shp --out output/step43/Zhanggongluwang_Prepare.shp
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

# ══════════════════════════════════════════════════════════════════
# 路网权重表（LTS 分级口径）
# ══════════════════════════════════════════════════════════════════

WEIGHT_TABLE = {
    'footway'        : (1.0, 0.3),
    'path'           : (1.0, 0.3),
    'pedestrian'     : (1.0, 0.3),
    'steps'          : (1.0, 0.3),
    'residential'    : (1.0, 1.0),
    'living_street'  : (1.0, 1.0),
    'unclassified'   : (1.0, 1.0),
    'service'        : (1.0, 1.0),
    'tertiary'       : (0.8, 0.9),
    'tertiary_link'  : (0.8, 0.9),
    'track'          : (0.8, 0.9),
    'secondary'      : (0.6, 0.8),
    'secondary_link' : (0.6, 0.8),
    'cycleway'       : (0.6, 0.8),
    'primary'        : (0.4, 0.6),
    'primary_link'   : (0.4, 0.6),
    'trunk'          : (0.3, 0.3),
    'trunk_link'     : (0.3, 0.3),
    'motorway'       : (0.0, 0.0),
    'motorway_link'  : (0.0, 0.0),
}
BIKE_RESTRICT = {'motorway', 'motorway_link', 'trunk', 'trunk_link'}
WALK_RESTRICT = {'motorway', 'motorway_link'}


def _parse_highway(hw_str):
    if hw_str is None:
        return []
    hw_str = str(hw_str).strip().strip("[]'\"")
    types = [t.strip().strip("'\"") for t in hw_str.split(',')]
    return [t for t in types if t in WEIGHT_TABLE]


def _get_weight(hw_str, col_idx):
    types = _parse_highway(hw_str)
    if not types:
        return None
    return min(WEIGHT_TABLE[t][col_idx] for t in types)


def _get_bike_re(hw_str):
    raw = str(hw_str) if hw_str else ''
    return 1.0 if any(kw in raw for kw in BIKE_RESTRICT) else 0.0


def _get_walk_re(hw_str):
    raw = str(hw_str) if hw_str else ''
    return 1.0 if any(kw in raw for kw in WALK_RESTRICT) else 0.0


# ══════════════════════════════════════════════════════════════════
# 子命令：preprocess（= RoadNetworkPreprocess.execute）
# ══════════════════════════════════════════════════════════════════

def run_preprocess(input_fc, output_fc, highway_field="highway",
                   length_field="length") -> dict:
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    gdf = gpd.read_file(input_fc, engine="pyogrio")
    print("► Step 1  加载要素 ...")
    print(f"  输入: {input_fc}")
    print(f"  要素: {len(gdf)} 条 | CRS={gdf.crs}")

    # Step 2 字段（gdf 直接赋列，等价 AddField）
    print("► Step 2  检查并添加字段 ...")
    for f in ('Walk_Wt', 'Bike', 'Bike_Re', 'Walk_Re', 'WalkTime', 'Bike_Time'):
        if f not in gdf.columns:
            gdf[f] = np.nan

    # Step 3 权重修正与时间重算（向量化，公式与 UpdateCursor 逐行逻辑一致）
    print("► Step 3  权重修正与时间重算 ...")
    hw = gdf[highway_field]
    gdf["Walk_Wt"] = hw.map(lambda s: _get_weight(s, 0))
    gdf["Bike"] = hw.map(lambda s: _get_weight(s, 1))
    gdf["Bike_Re"] = hw.map(_get_bike_re)
    gdf["Walk_Re"] = hw.map(_get_walk_re)

    length = pd.to_numeric(gdf[length_field], errors="coerce")
    walk_time = length / 75.0
    bike_time = np.where((gdf["Bike"].notna()) & (gdf["Bike"] > 0),
                         length * gdf["Bike"].astype(float) / 240.0, 0.0)
    gdf["WalkTime"] = walk_time.where(length.notna(), other=np.nan)
    gdf["Bike_Time"] = np.where(length.notna(), bike_time, np.nan)

    # Step 4 统计报告（口径与原工具一致）
    total = len(gdf)
    walk_wt_dist = gdf["Walk_Wt"].value_counts().to_dict()
    nan_hw = int(gdf["Walk_Wt"].isna().sum())
    bike_dist = gdf["Bike"].value_counts().to_dict()
    print("\n" + "=" * 55)
    print("权重修正统计报告")
    print("=" * 55)
    print(f"总记录数：{total} 条")
    print("\n【Walk_Wt 分布】")
    for k in sorted(walk_wt_dist):
        print(f"  {k:.1f} : {walk_wt_dist[k]} 条")
    print(f"  NaN  : {nan_hw} 条")
    print("\n【Bike 分布】")
    for k in sorted(bike_dist):
        print(f"  {k:.1f} : {bike_dist[k]} 条")
    wt = gdf["WalkTime"].dropna()
    bt = gdf["Bike_Time"].dropna()
    if len(wt):
        print("\n【时间统计（分钟）】")
        print(f"  WalkTime : min={wt.min():.4f}  max={wt.max():.2f}  mean={wt.mean():.2f}")
        print(f"  Bike_Time: min={bt.min():.4f}  max={bt.max():.2f}  mean={bt.mean():.2f}"
              if len(bt) else "  Bike_Time: 无数据")
    print(f"\n  禁止骑行路段 (Bike=0)   : {int((gdf['Bike'] == 0.0).sum())} 条")
    print(f"  禁止步行路段 (Walk_Wt=0): {int((gdf['Walk_Wt'] == 0.0).sum())} 条")
    print("=" * 55)

    out_dir = os.path.dirname(os.path.abspath(output_fc))
    os.makedirs(out_dir, exist_ok=True)
    gdf.to_file(output_fc, encoding="utf-8")
    print(f"\n✔ 处理完成，结果已保存至：\n  {output_fc}")
    return {"output": output_fc, "rows": total,
            "walk_wt_dist": walk_wt_dist, "bike_dist": bike_dist,
            "walk_time_mean": float(wt.mean()) if len(wt) else None,
            "bike_time_mean": float(bt.mean()) if len(bt) else None,
            "bike_zero": int((gdf['Bike'] == 0.0).sum()),
            "walk_zero": int((gdf['Walk_Wt'] == 0.0).sum())}


# ══════════════════════════════════════════════════════════════════
# 子命令：service-area（= ServiceAreaProcess.execute，原实现即 geopandas）
# ══════════════════════════════════════════════════════════════════

def run_service_area(face_path, facilities_path, output_path,
                     reference_path=None) -> dict:
    import geopandas as gpd

    facilities = gpd.read_file(facilities_path, engine="pyogrio")
    print(f"[1/6] 加载设施点: {len(facilities)} 条, CRS={facilities.crs}")
    if "OBJECTID_1" in facilities.columns:
        facilities["FacilityID"] = facilities["OBJECTID_1"].astype(int)
    elif "FacilityID" not in facilities.columns:
        facilities["FacilityID"] = range(1, len(facilities) + 1)

    face = gpd.read_file(face_path, engine="pyogrio")
    print(f"[2/6] 加载面要素: {len(face)} 条, CRS={face.crs}")
    if face.crs and str(face.crs) != "EPSG:4526":
        face = face.to_crs("EPSG:4526")

    # Step 3 属性连接
    school_fields = ["FacilityID", "POI_ID", "名称", "类型", "学生数",
                     "经度", "纬度", "地址", "电话", "区县", "school_id"]
    available = [f for f in school_fields if f in facilities.columns]
    fac_subset = facilities[available].drop_duplicates(subset=["FacilityID"])
    merged = face.merge(fac_subset, on="FacilityID", how="left")
    matched = merged["POI_ID"].notna().sum() if "POI_ID" in merged.columns else 0
    print(f"[3/6] 属性连接: {matched}/{len(merged)} 条匹配成功")

    # Step 4 Dissolve + buffer(0)
    dissolved = merged.dissolve(by="FacilityID", as_index=False)
    print(f"[4/6] Dissolve: {len(merged)}条 → {len(dissolved)}个多边形")
    invalid_before = int((~dissolved.geometry.is_valid).sum())
    dissolved["geometry"] = dissolved.buffer(0)
    print(f"      buffer(0): 修复 {invalid_before} 个无效几何")

    # Step 5-6 CRS + 字段
    if dissolved.crs is None or str(dissolved.crs) != "EPSG:4526":
        dissolved = dissolved.set_crs("EPSG:4526", allow_override=True)
    dissolved = dissolved.reset_index(drop=True)
    dissolved["FacilityID"] = range(1, len(dissolved) + 1)
    dissolved["Name"] = dissolved["FacilityID"].apply(
        lambda x: "地点 {} : 10 - 15".format(x))
    dissolved["FromBreak"] = 10.0
    dissolved["ToBreak"] = 15.0
    dissolved["Shape_Leng"] = dissolved.geometry.length
    dissolved["Shape_Area"] = dissolved.geometry.area
    output_fields = ["FacilityID", "Name", "FromBreak", "ToBreak",
                     "Shape_Leng", "Shape_Area", "POI_ID", "名称", "类型",
                     "学生数", "经度", "纬度", "地址", "电话", "区县",
                     "school_id", "geometry"]
    result = dissolved[[f for f in output_fields if f in dissolved.columns]].copy()
    print(f"[6/6] 输出字段: {list(result.columns)} | 记录数: {len(result)}")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    result.to_file(output_path, encoding="utf-8")
    areas_km2 = result.geometry.area / 1e6
    print(f"  记录数: {len(result)} | 总面积: {areas_km2.sum():.2f} km² | "
          f"均面积: {areas_km2.mean():.2f} km²")

    if reference_path and os.path.exists(reference_path):
        ref = gpd.read_file(reference_path, engine="pyogrio")
        print("\n========== 验证对比 ==========")
        checks = [len(result) == len(ref),
                  str(result.crs) == str(ref.crs),
                  abs(result.geometry.area.sum() / 1e6
                      - ref.geometry.area.sum() / 1e6) < 0.1]
        if "POI_ID" in result.columns and "POI_ID" in ref.columns:
            checks.append(set(result["POI_ID"].dropna().unique())
                          == set(ref["POI_ID"].dropna().unique()))
        print(f"验证结果: {sum(checks)}/{len(checks)} PASS")
    return {"output": output_path, "rows": len(result),
            "total_area_km2": float(areas_km2.sum())}


def main():
    ap = argparse.ArgumentParser(description="4.3 路网预处理与服务区后处理（纯 Python）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("preprocess", help="路网预处理（权重修正与时间重算）")
    p1.add_argument("--in", dest="input_fc",
                    default=str(paths.data_dir() / "Zhanggongluwang_Original.shp"))
    p1.add_argument("--out", dest="output_fc",
                    default=str(paths.output_dir() / "step43" / "Zhanggongluwang_Prepare.shp"))
    p1.add_argument("--highway-field", default="highway")
    p1.add_argument("--length-field", default="length")

    p2 = sub.add_parser("service-area", help="服务区后处理（对 step43b 等时圈面做属性连接/融合）")
    p2.add_argument("--face", required=True)
    p2.add_argument("--facilities", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--reference", default=None)

    args = ap.parse_args()
    if args.cmd == "preprocess":
        run_preprocess(args.input_fc, args.output_fc,
                       args.highway_field, args.length_field)
    else:
        run_service_area(args.face, args.facilities, args.out, args.reference)


if __name__ == "__main__":
    main()
