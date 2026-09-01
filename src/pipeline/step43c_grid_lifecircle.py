# -*- coding: utf-8 -*-
"""4.3d 格网 15 分钟生活圈批量生成（纯 Python，多进程）。

用途（模型B重构 / 路线丙）：把每个 250m 格网质心当作“源点”，在 OSM 路网上用
Dijkstra 求 15min 可达路段并缓冲融合，得到与学校等时圈**同口径**的生活圈面。
模型B据此在格网端聚合与学校端同尺度的特征，消除“训练用服务区、预测用单格”的
MAUP 尺度失配。

复用 step43b_service_area 的 _build_graph / _reachable_segments，保证权重、cutoff、
缓冲半径与学校等时圈完全一致。Windows 多进程用 spawn：每个 worker 进程在
initializer 中各自建图一次，任务只传递源点坐标、回传生活圈 WKB。
"""
import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from shapely import wkb

# 保证 spawn 子进程 import 本模块时也能导入同目录 step43b（Windows 无 fork）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TARGET_CRS = "EPSG:4526"

# ── 多进程 worker 的进程内全局图（initializer 建立一次）─────────────────
_G = None
_NODES = None
_KDT = None
_CUTOFF = None
_BUFFER = None


def _init_worker(roads_path, mode, cutoff, buffer_m):
    """每个 worker 进程启动时建一次图（避免反复 pickle 网络图）。"""
    import geopandas as gpd
    from scipy.spatial import cKDTree
    # 复用同目录 step43b 的建图/可达段逻辑，保证口径一致
    from step43b_service_area import _build_graph
    roads = gpd.read_file(roads_path)
    G, node_xy = _build_graph(roads, mode)
    global _G, _NODES, _KDT, _CUTOFF, _BUFFER
    _G = G
    _NODES = list(node_xy)
    _KDT = cKDTree(np.array([node_xy[n] for n in _NODES]))
    _CUTOFF = cutoff
    _BUFFER = buffer_m


def _one_lifecircle(xy):
    """单个源点 → 生活圈面 WKB（bytes）；无路网时返回 None。"""
    from shapely.ops import unary_union
    from step43b_service_area import _reachable_segments
    x, y = float(xy[0]), float(xy[1])
    _, i = _KDT.query([x, y])
    src = _NODES[i]
    segs = _reachable_segments(_G, src, _CUTOFF)
    if segs:
        poly = unary_union([s.buffer(_BUFFER) for s in segs])
    else:
        from shapely.geometry import Point
        poly = Point(x, y).buffer(_BUFFER)
    if poly is None or poly.is_empty or poly.area <= 0:
        return None
    return wkb.dumps(poly)


def run(roads_path, grid_path, boundary_path, output_path,
        mode="walk", cutoff=15.0, buffer_m=90.0, n_jobs=None,
        grid_id_field="grid_id"):
    import geopandas as gpd
    from shapely import wkb
    from shapely.ops import unary_union

    t0 = time.time()
    roads = gpd.read_file(roads_path)
    grid = gpd.read_file(grid_path).to_crs(roads.crs)
    # 保证有稳定的格网编号
    if grid_id_field not in grid.columns:
        grid[grid_id_field] = np.arange(1, len(grid) + 1, dtype=int)
    cen = grid.geometry.centroid
    xy_list = np.column_stack([cen.x.values, cen.y.values])
    print(f"[1/3] 路网 {len(roads)} 段；格网 {len(grid)} 个；mode={mode} cutoff={cutoff}min")

    boundary = None
    if boundary_path and os.path.exists(boundary_path):
        boundary = unary_union(
            gpd.read_file(boundary_path).to_crs(TARGET_CRS).geometry)
        print(f"  行政区裁剪边界: {boundary_path}")

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 2)
    print(f"[2/3] 多进程生成生活圈（{n_jobs} 进程）...")
    rows = []
    t1 = time.time()
    # 单进程顺序路径（便于调试 / 确定性）
    if n_jobs <= 1:
        _init_worker(roads_path, mode, cutoff, buffer_m)
        for k, xy in enumerate(xy_list):
            rows.append((grid[grid_id_field].iloc[k], _one_lifecircle(xy)))
            if (k + 1) % 200 == 0:
                print(f"    {k+1}/{len(xy_list)}  用时 {time.time()-t1:.0f}s")
    else:
        with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_worker,
                initargs=(roads_path, mode, cutoff, buffer_m)) as ex:
            wkbs = list(ex.map(_one_lifecircle, xy_list, chunksize=8))
        rows = list(zip(grid[grid_id_field].values, wkbs))
    print(f"  等时圈计算用时 {time.time()-t1:.1f}s")

    # 组装 + 边界裁剪
    geoms, ids = [], []
    n_empty = 0
    for gid, wb in rows:
        if wb is None:
            n_empty += 1
            continue
        geom = wkb.loads(wb)
        geoms.append(geom); ids.append(gid)
    result = gpd.GeoDataFrame(
        {grid_id_field: ids, "geometry": geoms}, crs=roads.crs
    ).to_crs(TARGET_CRS)
    if boundary is not None:
        result["geometry"] = result.geometry.intersection(boundary)
        result = result[~result.geometry.is_empty]
    result = result[result.geometry.area > 0].reset_index(drop=True)
    result["lc_area_km2"] = result.geometry.area / 1e6
    print(f"[3/3] 有效生活圈 {len(result)}/{len(grid)}（空 {n_empty}）；"
          f"总面积 {result.geometry.area.sum()/1e6:.2f} km²；总用时 {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.lower().endswith(".gpkg"):
        result.to_file(output_path, layer="grid_lifecircle", driver="GPKG")
    else:
        result.to_file(output_path, encoding="utf-8")
    print(f"✅ 输出: {output_path}")
    return {"n_grid": len(grid), "n_valid": len(result),
            "n_empty": n_empty, "n_jobs": n_jobs,
            "total_km2": float(result.geometry.area.sum() / 1e6),
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description="4.3d 格网15分钟生活圈批量生成（多进程）")
    ap.add_argument("--roads", required=True, help="step43 Prepare 路网")
    ap.add_argument("--grid", required=True, help="格网 shp（含 CLUSTER_ID/vitality）")
    ap.add_argument("--boundary", default=None, help="行政区边界（可选裁剪）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="walk", choices=["walk", "bike"])
    ap.add_argument("--cutoff", type=float, default=15.0)
    ap.add_argument("--buffer", type=float, default=90.0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--grid-id-field", default="grid_id")
    a = ap.parse_args()
    rep = run(a.roads, a.grid, a.boundary, a.out, a.mode,
              a.cutoff, a.buffer, a.jobs, a.grid_id_field)
    print("REPORT", rep)


if __name__ == "__main__":
    # 允许 `python -m pipeline.step43c...` 与直接运行两种方式导入同目录 step43b
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
