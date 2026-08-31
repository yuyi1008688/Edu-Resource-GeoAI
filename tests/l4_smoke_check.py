# -*- coding: utf-8 -*-
"""L4 干净虚拟环境隔离体检：
1) 动态导入 src/pipeline 全部 CLI + src/core + src/utils，断言 sys.modules 零 arcpy；
2) 在干净环境实跑两个新模块（step42 聚类走 spopt/libpysal、step43b 步行等时圈走 networkx），
   证明 requirements.txt 自足、不依赖主虚拟环境与任何商业 GIS。

数据通过环境变量 B236_DATA_DIR 提供；第 2 步实跑还需要先跑过全链（output_l2fresh 上游）。
缺少数据/上游时自动跳过第 2 步（不计为失败），第 1 步导入与 arcpy 隔离始终执行。
仅用于环境复验，产物落 output_l4_smoke（可删）。
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["B236_DATA_DIR"]) if os.environ.get("B236_DATA_DIR") else None
L2 = ROOT / "output_l2fresh"
OUT = ROOT / "output_l4_smoke"
OUT.mkdir(exist_ok=True)

fail = []

# ---------- 1) 全模块导入 + arcpy 隔离 ----------
print("=" * 70)
print("[1] 导入 src/pipeline 全部 CLI 模块")
mods = sorted((ROOT / "src" / "pipeline").glob("step*.py"))
loaded = 0
for mf in mods:
    spec = importlib.util.spec_from_file_location(f"pipe_{mf.stem}", mf)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    loaded += 1
print(f"    已导入 pipeline 模块 {loaded} 个")
for pkg in ["spopt", "libpysal", "networkx", "xgboost", "shap", "pulp", "optuna",
            "geopandas", "rasterio", "sklearn", "scipy", "jenkspy"]:
    __import__(pkg)
print(f"    关键第三方依赖全部可导入：spopt/libpysal/networkx/xgboost/shap/pulp 等")
arc = [k for k in sys.modules if k.split(".")[0] == "arcpy"]
print(f"    sys.modules 中 arcpy* 模块数 = {len(arc)}（应为 0）")
if arc:
    fail.append(f"检测到 arcpy 泄漏: {arc[:5]}")

# 第 2 步实跑的前置条件：数据目录 + 全链上游产物
can_run = DATA is not None and (L2 / "extract" / "fishnet_features.shp").exists()
if not can_run:
    print("=" * 70)
    print("[2] 跳过实跑：未设置 B236_DATA_DIR 或缺少 output_l2fresh 上游（第 1 步已完成）")

# ---------- 2a) step42 聚类实跑（spopt Skater） ----------
if can_run:
    print("=" * 70)
    print("[2a] step42_cluster 实跑（Delaunay + spopt.Skater）")
    py = sys.executable
    c42 = [py, str(ROOT / "src/pipeline/step42_cluster.py"),
           "--features", str(L2 / "extract/fishnet_features.shp"),
           "--vitality", str(L2 / "vitality/vitality_fishnet.gpkg"),
           "--out", str(OUT / "Fishnet_Cluster_l4.shp")]
    r = subprocess.run(c42, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "B236_DATA_DIR": str(DATA)})
    print(r.stdout[-1200:])
    if r.returncode != 0:
        fail.append("step42 实跑失败: " + r.stderr[-800:])
    else:
        import geopandas as gpd
        g = gpd.read_file(OUT / "Fishnet_Cluster_l4.shp")
        print(f"    -> 聚类输出 {len(g)} 格，类别 {sorted(g['cluster'].unique()) if 'cluster' in g else g.columns.tolist()}")

# ---------- 2b) step43b 小学步行等时圈实跑（networkx Dijkstra） ----------
if can_run:
    print("=" * 70)
    print("[2b] step43b walk 实跑（networkx 有向 Dijkstra + buffer 拼面）")
    py = sys.executable
    c43 = [py, str(ROOT / "src/pipeline/step43b_service_area.py"), "walk",
           "--roads", str(L2 / "step43/Zhanggongluwang_Prepare.shp"),
           "--facilities", str(DATA / "Primary_school.shp"),
           "--boundary", str(DATA / "zhanggong.shp"),
           "--out", str(OUT / "iso_primary_l4.shp")]
    r = subprocess.run(c43, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "B236_DATA_DIR": str(DATA)})
    print(r.stdout[-1000:])
    if r.returncode != 0:
        fail.append("step43b 实跑失败: " + r.stderr[-800:])
    else:
        g = gpd.read_file(OUT / "iso_primary_l4.shp")
        print(f"    -> 小学等时圈 {len(g)} 校，总面积 {g.to_crs(4526).area.sum()/1e6:.2f} km2")

print("=" * 70)
if fail:
    print("L4 体检【未通过】：")
    for x in fail:
        print("  -", x)
    sys.exit(1)
print("L4 体检【通过】：全模块可导入、零 arcpy" + ("、step42/step43b 干净环境实跑成功" if can_run else "（实跑已跳过）"))
