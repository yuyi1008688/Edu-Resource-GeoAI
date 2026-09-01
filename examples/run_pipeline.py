# -*- coding: utf-8 -*-
"""
run_pipeline.py — 全流程纯 Python 编排器

按 4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6 → 4.7/5.0 的业务顺序串联各 step CLI。
每个 step 的算法、阈值、权重与原始实现一致（详见 src/pipeline/）。

全链 14 个注册步骤（其中 4.3 服务区拆小学步行/中学骑行两次）：
  gdb_convert → normalize → cut → extract → vitality → cluster42 →
  road43 → sa_primary → sa_middle → ecfi44 → geoxgboost45 → sms46 →
  opt47 → eval50

用法：
  python examples/run_pipeline.py --list                      # 查看步骤与数据依赖
  python examples/run_pipeline.py --data-dir <输入数据目录>    # 跑完整全链
  python examples/run_pipeline.py --only cluster42,sa_middle  # 只跑指定步骤

数据准备：
  设置环境变量 B236_DATA_DIR 指向输入数据目录（或用 --data-dir 传入），
  目录内容清单见 data/README.md；中间数据.gdb 首次运行自动转 GPKG。
  4.3 服务区裁剪需要研究区边界 zhanggong.shp（缺失时自动跳过裁剪并告警）。

可复现性说明：
  全链均可纯开源端到端运行：空间约束聚类用 spopt.Skater、路网服务区用 networkx
  有向图 Dijkstra + 缓冲拼面；三次独立运行的最终决策指标（覆盖率、新增学位、
  低可达格网数、扩容校数）逐次一致，Optuna / KFold / KMeans 随机种子均已固定。
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# 步骤注册表：名称 → (脚本相对路径, 说明, 是否可复现)
STEPS = {
    "gdb_convert":    ("src/utils/convert_gdb.py", "中间数据.gdb → GPKG 开源转换", True),
    "normalize":      ("src/utils/normalize_rasters.py", "栅格坐标基准归一化（主基准 4526 带号）", True),
    "cut":            ("src/pipeline/step41_fishnet_extract.py", "4.1a 渔网精准裁剪（城区/荒野/水体）", True),
    "extract":        ("src/pipeline/step41_fishnet_extract.py", "4.1b 栅格特征提取（6 波段均值）", True),
    "vitality":       ("src/pipeline/step41_vitality.py", "4.1c 城市活力指数（KDE+Shannon熵）", True),
    "cluster42":      ("src/pipeline/step42_cluster.py", "4.2 社区空间约束聚类（spopt.Skater）", True),
    "road43":         ("src/pipeline/step43_road_preprocess.py", "4.3a 路网预处理（LTS+OSRM 权重）", True),
    "sa_primary":     ("src/pipeline/step43b_service_area.py", "4.3b 小学步行 15min 等时圈（networkx）", True),
    "sa_middle":      ("src/pipeline/step43b_service_area.py", "4.3c 中学骑行 15min 等时圈（networkx）", True),
    "grid_lc":        ("src/pipeline/step43c_grid_lifecircle.py", "4.3d 格网15min生活圈（模型B同口径，多进程）", True),
    "ecfi44":         ("src/pipeline/step44_ecfi_diagnosis.py", "4.4 ECFI 三维诊断", True),
    "geoxgboost45":   ("src/pipeline/step45_geoxgboost.py", "4.5 GeoXGBoost 双模型", True),
    "sms46":          ("src/pipeline/step46_pipeline.py", "4.6 软实力标签与 SMS 主引擎", True),
    "opt47":          ("src/pipeline/step47_50_optimization.py", "4.7 MILP 优化配置", True),
    "eval50":         ("src/pipeline/step47_50_optimization.py", "5.0 效益评估（依赖 4.7 中间文件）", True),
}

DEFAULT_ORDER = ["gdb_convert", "normalize", "cut", "extract", "vitality",
                 "cluster42", "road43", "sa_primary", "sa_middle", "grid_lc",
                 "ecfi44", "geoxgboost45", "sms46", "opt47", "eval50"]


def step_cmds(name, data_dir, out_dir):
    """构造每个步骤的命令行列表（一个步骤可对应多条命令，如无则空）。"""
    out = Path(out_dir)
    dd = Path(data_dir)
    script = STEPS[name][0]
    base = [PY, str(ROOT / script)]

    if name == "gdb_convert":
        return [base]
    if name == "normalize":
        return [base]
    if name == "cut":
        return [base + ["cut", "--out", str(out / "fishnet"), "--no-plot"]]
    if name == "extract":
        return [base + ["extract",
                        "--fishnet", str(out / "fishnet" / "fishnet_urban_only.shp"),
                        "--out", str(out / "extract" / "fishnet_features.shp")]]
    if name == "vitality":
        return [base + ["--fishnet", str(out / "fishnet" / "fishnet_urban_only.shp"),
                        "--out", str(out / "vitality" / "vitality_fishnet.gpkg")]]
    if name == "cluster42":
        return [base + ["--features", str(out / "extract" / "fishnet_features.shp"),
                        "--vitality", str(out / "vitality" / "vitality_fishnet.gpkg"),
                        "--out", str(out / "cluster" / "Fishnet_Cluster.shp")]]
    if name == "road43":
        return [base + ["preprocess"]]
    if name == "sa_primary":
        return [base + ["walk",
                        "--roads", str(out / "step43" / "Zhanggongluwang_Prepare.shp"),
                        "--facilities", str(dd / "Primary_school.shp"),
                        "--boundary", str(dd / "zhanggong.shp"),
                        "--out", str(out / "service" / "iso_primary.shp")]]
    if name == "sa_middle":
        return [base + ["bike",
                        "--roads", str(out / "step43" / "Zhanggongluwang_Prepare.shp"),
                        "--facilities", str(dd / "Middle_school.shp"),
                        "--boundary", str(dd / "zhanggong.shp"),
                        "--out", str(out / "service" / "iso_middle.shp")]]
    if name == "grid_lc":
        return [base + ["--roads", str(out / "step43" / "Zhanggongluwang_Prepare.shp"),
                        "--grid", str(out / "cluster" / "Fishnet_Cluster.shp"),
                        "--boundary", str(dd / "zhanggong.shp"),
                        "--mode", "walk",
                        "--out", str(out / "service" / "grid_lifecircle.gpkg")]]
    if name == "ecfi44":
        return [base + [
            "--school-csv", str(dd / "school_data.csv"),
            "--poi-csv", str(dd / "POI_data.csv"),
            "--isochrone-elem", str(out / "service" / "iso_primary.shp"),
            "--isochrone-mid", str(out / "service" / "iso_middle.shp"),
            "--fishnet", str(out / "cluster" / "Fishnet_Cluster.shp"),
            "--road-density-4526", str(dd / "road_density_fixed_EPSG4526.tif"),
            "--out-dir", str(out / "step44")]]
    if name == "geoxgboost45":
        return [base + [
            "--school-csv", str(out / "step44" / "4.4_school_profile.csv"),
            "--d3-field", "D3_pressure",
            "--fishnet", str(out / "cluster" / "Fishnet_Cluster.shp"),
            "--iso-primary", str(out / "service" / "iso_primary.shp"),
            "--iso-middle", str(out / "service" / "iso_middle.shp"),
            "--worldpop", str(dd / "WorldPop_250m_EPSG4526.tif"),
            "--build-density-raster", str(dd / "zhangong_buildings_density.tif"),
            "--buildings", str(dd / "building_footprint.shp"),
            # 期刊版修正：4.4输出WKT为经纬度，须按4326读；模型B用格网生活圈同口径特征
            "--school-crs", "EPSG:4326",
            "--grid-lifecircle", str(out / "service" / "grid_lifecircle.gpkg"),
            # 期刊版：充分超参搜索（默认75/40未收敛；100/120后两模型均提升）
            "--optuna-trials", "100", "--optuna-refine", "120",
            "--outdir", str(out / "step45")]]
    if name == "sms46":
        return [base + ["pipeline"]]
    if name == "opt47":
        return [base + ["opt47",
                        "--school-profile-csv", str(out / "step44" / "4.4_school_profile.csv"),
                        "--school-pressure-csv", str(out / "step45" / "school_pressure_prediction.csv"),
                        "--road-network", str(out / "step43" / "Zhanggongluwang_Prepare.shp"),
                        "--worldpop-raster", str(dd / "WorldPop_250m_EPSG4526.tif"),
                        "--out-dir", str(out / "step47")]]
    if name == "eval50":
        s47 = out / "step47"
        return [base + ["eval50",
                        "--school-before", str(s47 / "F_school_before.csv"),
                        "--school-after", str(s47 / "F_school_after.csv"),
                        "--grid-result", str(s47 / "F_grid_result.csv"),
                        "--assignment", str(s47 / "F_assignment.csv"),
                        "--opt-meta", str(s47 / "F_opt_meta.json"),
                        "--sfca-pri", str(s47 / "F_sfca_A_pri_init.npy"),
                        "--sfca-mid", str(s47 / "F_sfca_A_mid_init.npy"),
                        "--sfca-comb", str(s47 / "F_sfca_A_combined_init.npy"),
                        "--mask", str(s47 / "F_low_access_mask.npy"),
                        "--time-mat", str(s47 / "E_time_matrix.npy"),
                        "--out-dir", str(out / "step50")]]
    return [base]


def main():
    ap = argparse.ArgumentParser(description="纯 Python 全流程编排器")
    ap.add_argument("--data-dir", default=os.environ.get("B236_DATA_DIR", ""),
                    help="输入数据目录（也可用环境变量 B236_DATA_DIR）")
    ap.add_argument("--out-dir", default=str(ROOT / "output"),
                    help="输出根目录（默认 <仓库>/output）")
    ap.add_argument("--only", default="", help="逗号分隔的步骤名子集")
    ap.add_argument("--list", action="store_true", help="列出步骤与数据依赖后退出")
    args = ap.parse_args()

    if args.list:
        print(f"{'步骤':14s} {'可复现':6s} 说明")
        print("-" * 86)
        for name, (_, desc, ok) in STEPS.items():
            print(f"{name:14s} {'是' if ok else '否':6s} {desc}")
        print()
        print(f"默认全链顺序：{' -> '.join(DEFAULT_ORDER)}")
        return 0

    if not args.data_dir:
        ap.error("必须提供 --data-dir 或设置环境变量 B236_DATA_DIR")
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"数据目录不存在：{data_dir}")

    order = [s.strip() for s in args.only.split(",") if s.strip()] or DEFAULT_ORDER
    unknown = [s for s in order if s not in STEPS]
    if unknown:
        sys.exit(f"未知步骤：{unknown}（可选：{list(STEPS)}）")

    t0 = time.time()
    failed = []
    for name in order:
        desc, ok = STEPS[name][1], STEPS[name][2]
        env = {**os.environ, "B236_DATA_DIR": str(data_dir),
               "B236_OUTPUT_DIR": str(args.out_dir)}
        cmds = step_cmds(name, data_dir, args.out_dir)
        for ci, cmd in enumerate(cmds):
            tag = name if len(cmds) == 1 else f"{name}({ci+1}/{len(cmds)})"
            print()
            print("=" * 78)
            print(f"▶ {tag} — {desc}")
            print("=" * 78)
            t1 = time.time()
            r = subprocess.run(cmd, env=env, cwd=str(ROOT))
            dt = time.time() - t1
            if r.returncode != 0:
                print(f"X {tag} 失败（exit {r.returncode}），耗时 {dt:.1f}s —— 中止后续步骤")
                failed.append(tag)
                break
            print(f"OK {tag} 完成，耗时 {dt:.1f}s")
        if failed:
            break
    total = time.time() - t0
    print()
    print("=" * 78)
    if failed:
        print(f"全链中止于 {failed}，总耗时 {total:.1f}s")
        return 1
    print(f"所选步骤全部完成，总耗时 {total:.1f}s；产物见 {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
