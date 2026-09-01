# -*- coding: utf-8 -*-
"""期刊定稿跨运行确定性比对。"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd

print("=" * 66)
print("① 格网生活圈：两次独立生成(output_paper vs output_journal)几何面积比对")
a = gpd.read_file("output_paper/service/grid_lifecircle.gpkg").sort_values("grid_id").reset_index(drop=True)
b = gpd.read_file("output_journal/service/grid_lifecircle.gpkg").sort_values("grid_id").reset_index(drop=True)
print(f"  格数 {len(a)} vs {len(b)}；grid_id集合相等: {set(a.grid_id)==set(b.grid_id)}")
area_a = a.geometry.area.values; area_b = b.geometry.area.values
print(f"  生活圈面积 max|Δ|={np.abs(area_a-area_b).max():.6e} m²  相关={np.corrcoef(area_a,area_b)[0,1]:.8f}")
if "lc_area_km2" in a:
    d = np.abs(a.lc_area_km2.values - b.lc_area_km2.values)
    print(f"  lc_area_km2 max|Δ|={d.max():.3e}")

print("=" * 66)
print("② 4.5 学校预测：全链(output_journal) vs 单步(_smokeLIFE2,不同上游目录)")
ca = pd.read_csv("output_journal/step45/school_pressure_prediction.csv")
cb = pd.read_csv("_smokeLIFE2/school_pressure_prediction.csv")
key = "school_id" if "school_id" in ca.columns else ca.columns[0]
m = ca.merge(cb, on=key, suffixes=("_full", "_smoke"))
print(f"  学校数 {len(ca)} vs {len(cb)}，匹配 {len(m)}")
numcols = [c[:-5] for c in m.columns if c.endswith("_full")]
for base in numcols:
    x = pd.to_numeric(m[f"{base}_full"], errors="coerce")
    y = pd.to_numeric(m[f"{base}_smoke"], errors="coerce")
    if x.notna().sum() >= 50 and y.notna().sum() >= 50:
        d = (x - y).abs().max()
        if np.isfinite(d):
            print(f"  {base:24s} max|Δ|={d:.3e}")

print("=" * 66)
print("③ 格网压力面：output_journal 内部唯一性/连续性")
pc = gpd.read_file("output_journal/step45/pressure_coefs.gpkg") if __import__("os").path.exists("output_journal/step45/pressure_coefs.gpkg") else None
if pc is not None:
    for c in pc.columns:
        if pc[c].dtype.kind in "fiu" and pc[c].nunique() > 5:
            print(f"  {c:20s} 唯一值={pc[c].nunique():5d} 范围=[{pc[c].min():.3f},{pc[c].max():.3f}]")
print("确定性比对完成")
