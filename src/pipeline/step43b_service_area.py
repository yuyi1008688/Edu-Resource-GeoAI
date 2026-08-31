# -*- coding: utf-8 -*-
"""
step43b_service_area.py — 4.3 等时圈服务区【从零生成】（纯 Python CLI）

纯 Python 实现路网可达「服务区(Service Area)」分析：
  - 小学=步行：边权 WalkTime=length/75（75 m/min），剔除 Walk_Wt=0（motorway 禁步）；
  - 中学=骑行：边权 Bike_Time=length×Bike/240（240 m/min×骑行系数），剔除 Bike=0
    （trunk/motorway 禁骑）；
  - 尊重 OSM 单向（oneway/reversed）建【有向图】（实测比无向更贴近历史结果）；
  - 每校从最近路网节点 Dijkstra，cutoff=15min（5/10/15 断点取最外环）；
  - 可达边按剩余时间在 15min 边界线性截断 → 投影 EPSG:4526 → 沿线缓冲(buffer)
    融合 → 行政区边界裁剪。

诚实口径：专有 GIS 软件用三角剖分把可达路网"填"成整片面，开源以
  "可达路段缓冲融合"近似。中学骑行对参考面 IoU 中位≈0.62、面积比≈1.35；
  下游 4.4 以面积加权 zonal 统计为主（权重内部归一化、供给多被钳顶），
  对覆盖范围(IoU)敏感、对绝对面积不敏感，故按 IoU 最优取 buffer=90m。

示例：
  # 中学骑行
  python src/pipeline/step43b_service_area.py bike \
      --roads output/step43/Zhanggongluwang_Prepare.shp \
      --facilities <输入>/Middle_school.shp --boundary <输入>/zhanggong.shp \
      --out output/service/iso_middle.shp
  # 小学步行
  python src/pipeline/step43b_service_area.py walk \
      --roads output/step43/Zhanggongluwang_Prepare.shp \
      --facilities <输入>/Primary_school.shp --boundary <输入>/zhanggong.shp \
      --out output/service/iso_primary.shp
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

TARGET_CRS = "EPSG:4526"

# ─────────────────────────────────────────────────────────────────────
# 路网阻抗权重（LTS + OSRM 标定的 20 类道路）
#
# ⚠ 关键坑：项目里存在【两套】骑行/步行权重，切勿混用——
#   (1) LTS 终审分级表（step43_road_preprocess.WEIGHT_TABLE / fix_road_weights.py，
#       trunk=0.3、motorway=0），用于复现 GDB Zhanggongluwang_Prepare；
#   (2) 用于生成等时圈的 OSRM 细系数表（本处）：
#       trunk/motorway 骑行=0（城市快速路禁骑）、其 _link 匝道=0.6，
#       多标签复合段 Bike=0.6 / Walk_Wt=0。
#   若误用 (1)，骑行者会沿 trunk 以 1/0.3 倍时间"飞奔"，中学 15min 圈暴涨 2.25 倍。
#   等时圈一律以 highway 字段按本表重算阻抗，不读输入路网的 Bike/Walk_Wt 列。
#   数值：(Bike 骑行系数, Walk_Wt 步行系数)；时间 Bike_Time=len*Bike/240、WalkTime=len/75。
NET_WEIGHT = {
    "footway": (0.9, 1.0), "path": (0.6, 1.0), "pedestrian": (0.6, 1.0),
    "steps": (0.6, 1.0), "residential": (1.0, 1.0), "living_street": (0.6, 1.0),
    "unclassified": (0.6, 1.0), "service": (0.6, 1.0), "tertiary": (0.9, 0.8),
    "tertiary_link": (0.6, 0.8), "track": (0.6, 1.0), "secondary": (0.7, 0.6),
    "secondary_link": (0.6, 0.6), "cycleway": (0.6, 0.8), "primary": (0.5, 0.4),
    "primary_link": (0.6, 0.4), "trunk": (0.0, 0.2), "trunk_link": (0.6, 0.2),
    "motorway": (0.0, 0.2), "motorway_link": (0.6, 0.2),
}


def _parse_highway(h):
    s = str(h).strip().strip("[]'\"")
    return [t.strip().strip("'\"") for t in s.split(",") if t.strip()]


def _net_coef(hw_str):
    """返回 (Bike, Walk_Wt)；单类型查表，多标签复合段=(0.6,0)，无法解析=(nan,nan)。"""
    import numpy as np
    ts = _parse_highway(hw_str)
    if len(ts) == 1 and ts[0] in NET_WEIGHT:
        return NET_WEIGHT[ts[0]]
    if len(ts) > 1:
        return 0.6, 0.0
    return np.nan, np.nan


def _build_graph(roads, mode):
    """按模式建有向图（阻抗以 highway 按 OSRM 网络分析表重算）。返回 (DiGraph, node_xy)。"""
    import networkx as nx
    import numpy as np

    coef = roads["highway"].map(_net_coef)
    roads = roads.copy()
    roads["_bike"] = [c[0] for c in coef]
    roads["_walk"] = [c[1] for c in coef]
    length = roads["length"].astype(float)
    if mode == "walk":
        keep = (roads["_walk"] > 0)
        roads["_t"] = length / 75.0          # WalkTime=length/75（75 m/min）
    else:
        keep = (roads["_bike"] > 0)
        roads["_t"] = length * roads["_bike"] / 240.0  # Bike_Time=len*Bike/240
    use = roads[keep & (roads["_t"] > 0)].copy()
    print(f"  {mode}: 可用路段 {len(use)}（OSRM 网络分析权重）")

    node_xy = {}
    for _, r in use.iterrows():
        cs = list(r.geometry.coords)
        node_xy.setdefault(int(r.u), cs[0])
        node_xy.setdefault(int(r.v), cs[-1])

    G = nx.DiGraph()
    for _, r in use.iterrows():
        u, v, w = int(r.u), int(r.v), float(r["_t"])
        if w <= 0:
            continue
        oneway = int(r.oneway) if "oneway" in use.columns else 0
        reversed_ = str(r.reversed) == "True" if "reversed" in use.columns else False
        if oneway == 1:
            # OSM 单向：reversed 表示存储几何相对通行方向被反转
            if reversed_:
                G.add_edge(v, u, w=w, geom=r.geometry.reverse())
            else:
                G.add_edge(u, v, w=w, geom=r.geometry)
        else:
            G.add_edge(u, v, w=w, geom=r.geometry)
            G.add_edge(v, u, w=w, geom=r.geometry.reverse())
    print(f"  有向图: {G.number_of_nodes()} 节点 / {G.number_of_edges()} 有向边 / "
          f"{nx.number_weakly_connected_components(G)} 弱连通分量")
    return G, node_xy


def _reachable_segments(G, src, cutoff):
    """单校 Dijkstra，返回 15min 可达（含边界截断）的线段几何列表（源 CRS）。"""
    import networkx as nx
    from shapely.geometry import LineString

    dist = nx.single_source_dijkstra_path_length(G, src, weight="w", cutoff=cutoff)
    segs = []
    for u, v, d in G.edges(data=True):
        du, dv = dist.get(u), dist.get(v)
        geom, w = d["geom"], d["w"]
        cs = list(geom.coords)
        if du is not None and dv is not None and du <= cutoff and dv <= cutoff:
            segs.append(geom)
        elif du is not None and du <= cutoff and (dv is None or dv > cutoff):
            f = min(max((cutoff - du) / w, 0.0), 1.0)
            e = geom.line_interpolate_point(f, normalized=True)
            segs.append(LineString([cs[0], (e.x, e.y)]))
        elif dv is not None and dv <= cutoff and (du is None or du > cutoff):
            f = min(max((cutoff - dv) / w, 0.0), 1.0)
            e = geom.line_interpolate_point(f, normalized=True)
            segs.append(LineString([cs[-1], (e.x, e.y)]))
    return segs


def run(mode, roads_path, facilities_path, boundary_path, output_path,
        cutoff=15.0, buffer_m=90.0, golden_path=None) -> dict:
    import numpy as np
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.ops import unary_union

    roads = gpd.read_file(roads_path)
    print(f"[1/5] 路网 {len(roads)} 段，CRS={roads.crs}")
    G, node_xy = _build_graph(roads, mode)

    fac = gpd.read_file(facilities_path).to_crs(roads.crs)
    print(f"[2/5] 设施点 {len(fac)}，已统一到路网 CRS")
    nodes = list(node_xy)
    kdt = cKDTree(np.array([node_xy[n] for n in nodes]))

    if boundary_path and os.path.exists(boundary_path):
        boundary = unary_union(
            gpd.read_file(boundary_path).to_crs(TARGET_CRS).geometry)
        print(f"  行政区裁剪边界: {boundary_path}")
    else:
        boundary = None
        print("  [警告] 未提供行政区边界，输出未裁剪等时圈")

    # ── 3/4. 逐校等时圈 ───────────────────────────────────────────
    out_rows = []
    for _, p in fac.iterrows():
        _, i = kdt.query([p.geometry.x, p.geometry.y])
        src = nodes[i]
        segs = _reachable_segments(G, src, cutoff)
        if segs:
            seg_gdf = gpd.GeoDataFrame(geometry=segs, crs=roads.crs).to_crs(TARGET_CRS)
            poly = unary_union(list(seg_gdf.geometry.buffer(buffer_m)))
        else:
            # 路网孤岛兜底：设施点小缓冲，保证有几何
            pt = gpd.GeoSeries([p.geometry], crs=roads.crs).to_crs(TARGET_CRS)
            poly = pt.geometry.iloc[0].buffer(buffer_m)
        if boundary is not None:
            poly = poly.intersection(boundary)
        row = {"geometry": poly}
        for f in ("school_id", "POI_ID", "School_Nam", "名称", "类型", "学生数"):
            if f in fac.columns:
                row[f] = p[f]
        out_rows.append(row)
    result = gpd.GeoDataFrame(out_rows, crs=TARGET_CRS)
    result = result[result.geometry.area > 0].reset_index(drop=True)
    print(f"[3/5] 生成服务区 {len(result)} 个，总面积 {result.geometry.area.sum()/1e6:.2f} km²")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.to_file(output_path, encoding="utf-8")
    print(f"[4/5] ✅ 输出: {output_path}")

    report = {"mode": mode, "n": len(result), "buffer_m": buffer_m,
              "total_km2": float(result.geometry.area.sum() / 1e6)}

    # ── 5. 可选：对参考面逐校验证面积比 + IoU ──────────────────────
    if golden_path and os.path.exists(golden_path):
        gold = gpd.read_file(golden_path).to_crs(TARGET_CRS)
        key = "POI_ID" if "POI_ID" in gold.columns and "POI_ID" in result.columns else None
        metrics = []
        rg = result.set_index(key) if key else result
        gg = gold.set_index(key) if key else gold
        for idx in gg.index if key else range(len(gold)):
            if idx not in rg.index:
                continue
            a, b = rg.loc[idx, "geometry"], gg.loc[idx, "geometry"]
            ga = b.area
            if ga <= 0 or a.area <= 0:
                continue
            iou = a.intersection(b).area / a.union(b).area
            metrics.append((a.area / ga, iou))
        arr = np.array(metrics)
        report["golden"] = {"matched": len(arr),
                            "area_ratio_median": float(np.nanmedian(arr[:, 0])),
                            "iou_median": float(np.nanmedian(arr[:, 1])),
                            "iou_mean": float(np.nanmean(arr[:, 1])),
                            "gold_total_km2": float(gold.geometry.area.sum() / 1e6)}
        print(f"[5/5] 对参考基准 {len(arr)} 校：面积比中位 {np.nanmedian(arr[:,0]):.3f}，"
              f"IoU 中位 {np.nanmedian(arr[:,1]):.3f} / 均值 {np.nanmean(arr[:,1]):.3f}")
    return report


def main():
    ap = argparse.ArgumentParser(description="4.3 等时圈服务区从零生成（纯 Python）")
    ap.add_argument("mode", choices=["walk", "bike"], help="walk=小学步行 / bike=中学骑行")
    ap.add_argument("--roads", required=True, help="step43 Prepare 路网（含 u/v/oneway/时间字段）")
    ap.add_argument("--facilities", required=True, help="学校点（Primary_school/Middle_school）")
    ap.add_argument("--boundary", default=None,
                    help="行政区裁剪边界（默认 B236_DATA_DIR/zhanggong.shp）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=float, default=15.0, help="时间断点（分钟）")
    ap.add_argument("--buffer", type=float, default=90.0, help="可达路段缓冲半径（米）")
    ap.add_argument("--golden", default=None, help="参考服务区面（可选一致性验证）")
    args = ap.parse_args()
    boundary = args.boundary or str(paths.data_dir() / "zhanggong.shp")
    run(args.mode, args.roads, args.facilities, boundary, args.out,
        args.cutoff, args.buffer, args.golden)


if __name__ == "__main__":
    main()
