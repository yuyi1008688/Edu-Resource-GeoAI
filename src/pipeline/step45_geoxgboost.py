# -*- coding: utf-8 -*-
"""
step45_geoxgboost.py — 4.5 GeoXGBoost 双模型学位压力分析（纯 Python CLI）

纯 Python CLI：调用 core/geoxgboost
core/geoxgboost.run_analysis（诊断模型 A + 制图模型 B，均为回归器；
风险五级为 Jenks 后处理；一致性检验 = Spearman 秩相关）。

输入依赖：
  school_csv 需含 geometry_wkt 列（4.4 ECFI 的衍生产物）与压力列；
  渔网需含活力字段；小学/初中服务区由 step43b 生成。

示例：
  python src/pipeline/step45_geoxgboost.py --school-csv <csv> --fishnet <gpkg> \
      --iso-primary <shp> --iso-middle <shp> --worldpop <tif> --outdir output/step45
"""
import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "core"), str(_SRC / "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run(**kw):
    """调用 geoxgboost.run_analysis。"""
    os.environ.setdefault("GDAL_MEM_ENABLE_OPEN", "YES")  # 进程级设置
    from geoxgboost import run_analysis
    return run_analysis(**kw)


import os  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="4.5 GeoXGBoost 双模型学位压力分析（纯 Python）")
    # 第1组：必需输入
    ap.add_argument("--school-csv", required=True)
    ap.add_argument("--fishnet", required=True)
    ap.add_argument("--iso-primary", required=True)
    ap.add_argument("--iso-middle", required=True)
    ap.add_argument("--worldpop", required=True)
    ap.add_argument("--outdir", required=True)
    # 第2组：可选输入（建筑二选一）
    ap.add_argument("--build-density-raster", default=None)
    ap.add_argument("--buildings", default=None)
    # 路线丙：格网15min生活圈（模型B同口径特征，消除训练/预测尺度失配）
    ap.add_argument("--grid-lifecircle", default=None,
                    help="step43c 格网生活圈面；提供后模型B按生活圈同口径聚合特征")
    # 第3组：字段与坐标系
    ap.add_argument("--target-crs", default="EPSG:4526")
    ap.add_argument("--school-crs", default="EPSG:4526")
    ap.add_argument("--school-id-field", default="school_id")
    ap.add_argument("--service-id-field", default="school_id")
    ap.add_argument("--d3-field", default="presure",
                    help="学位压力字段（源数据列名为 presure，勘误勘定；auto 仅认 D3_pressure/D3/D3_raw）")
    ap.add_argument("--vitality-field", default="vitality")
    ap.add_argument("--grid-build-field", default="")
    # 第4组：空间分析参数
    ap.add_argument("--cell-size", type=float, default=250.0)
    ap.add_argument("--spatial-kernel", default="gaussian", choices=["gaussian", "idw", "bisquare"])
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument("--idw-k", type=int, default=15)
    ap.add_argument("--target-sample-ratio", type=float, default=5.0)
    ap.add_argument("--norm-plow", type=float, default=2.0)
    ap.add_argument("--norm-phigh", type=float, default=98.0)
    # 第5组：功能开关
    ap.add_argument("--no-rbf", dest="use_rbf", action="store_false")
    ap.add_argument("--no-gwr-weight", dest="use_gwr_weight", action="store_false")
    ap.add_argument("--no-optuna", dest="use_optuna", action="store_false")
    ap.add_argument("--no-shap", dest="use_shap", action="store_false")
    ap.add_argument("--no-boxcox", dest="use_boxcox", action="store_false")
    ap.add_argument("--no-huber", dest="use_huber_baseline", action="store_false")
    # 第6组：Optuna
    ap.add_argument("--optuna-trials", type=int, default=75)
    ap.add_argument("--optuna-refine", type=int, default=40)
    ap.add_argument("--optuna-metric", default="mdape", choices=["mdape", "mape", "r2"])
    args = ap.parse_args()

    outputs = run(
        school_csv=args.school_csv, fishnet_path=args.fishnet,
        iso_primary_path=args.iso_primary, iso_middle_path=args.iso_middle,
        build_density_raster=args.build_density_raster, buildings_path=args.buildings,
        worldpop_path=args.worldpop, grid_lifecircle_path=args.grid_lifecircle,
        outdir_path=args.outdir,
        target_crs=args.target_crs, school_crs=args.school_crs,
        school_id_field=args.school_id_field, service_id_field=args.service_id_field,
        d3_field=args.d3_field, vitality_field=args.vitality_field,
        grid_build_field=args.grid_build_field, cell_size=args.cell_size,
        spatial_kernel=args.spatial_kernel, idw_power=args.idw_power, idw_k=args.idw_k,
        target_sample_ratio=args.target_sample_ratio,
        norm_plow=args.norm_plow, norm_phigh=args.norm_phigh,
        use_rbf=args.use_rbf, use_gwr_weight=args.use_gwr_weight,
        use_optuna=args.use_optuna, use_shap=args.use_shap,
        use_boxcox=args.use_boxcox, use_huber_baseline=args.use_huber_baseline,
        optuna_trials=args.optuna_trials, optuna_refine=args.optuna_refine,
        optuna_metric=args.optuna_metric,
    )
    print("\n─── 4.5 输出 ───")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
