# -*- coding: utf-8 -*-
"""
step41_fishnet_extract.py — 4.1/4.2 渔网精准裁剪 + 栅格特征提取（纯 Python CLI）

纯 Python CLI：
  - 封装为命令行，业务逻辑调用 core 模块（fishnet_cutting / extract_raster）；
  - 算法、阈值、权重零改动；输入栅格自动取主基准归一化版本（坐标专项）。

子命令：
  cut      渔网精准裁剪：城区/荒野/水体三分（MNDWI + 多波段评分双通道 + ML 精修）
  extract  栅格特征提取：6 波段均值 → 渔网属性字段（rasterstats 分区统计）

示例：
  python src/pipeline/step41_fishnet_extract.py cut   --out output/fishnet
  python src/pipeline/step41_fishnet_extract.py extract --fishnet output/fishnet/fishnet_urban_only.shp --out output/extract/fishnet_features.shp
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


def _default(name):
    """输入数据目录下的文件路径（B236_DATA_DIR）。"""
    return str(paths.data_dir() / name)


# ──────────────────────────── 子命令：cut ────────────────────────────
def run_cut(raster_path, fishnet_path, output_dir, mndwi_path=None,
            green_band_path=None, swir_band_path=None,
            mndwi_water_threshold=0.0, urban_score_threshold=29,
            fuzzy_band_low=27, fuzzy_band_high=31,
            use_spatial_smoothing=True, use_ml_refinement=True,
            generate_plot=True) -> dict:
    """渔网精准裁剪（调用 core.run_fishnet_cutting）。"""
    import fishnet_cutting as fishnet_cutting_core
    os.makedirs(output_dir, exist_ok=True)
    return fishnet_cutting_core.run_fishnet_cutting(
        raster_path=raster_path, fishnet_path=fishnet_path,
        output_dir=output_dir, mndwi_path=mndwi_path,
        green_band_path=green_band_path, swir_band_path=swir_band_path,
        mndwi_water_threshold=mndwi_water_threshold,
        urban_score_threshold=urban_score_threshold,
        fuzzy_band_low=fuzzy_band_low, fuzzy_band_high=fuzzy_band_high,
        use_spatial_smoothing=use_spatial_smoothing,
        use_ml_refinement=use_ml_refinement, generate_plot=generate_plot)


# ────────────────────────── 子命令：extract ──────────────────────────
def run_extract(fishnet_shp, raster_tif, output_shp, band_names=None,
                output_csv=None) -> dict:
    """栅格特征提取（调用 core.run_extract_raster）。"""
    import extract_raster as extract_raster_core
    out_dir = os.path.dirname(os.path.abspath(output_shp))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return extract_raster_core.run_extract_raster(
        fishnet_shp=fishnet_shp, raster_tif=raster_tif,
        output_shp=output_shp, band_names=band_names, output_csv=output_csv)


def main():
    ap = argparse.ArgumentParser(description="4.1/4.2 渔网裁剪与栅格特征提取（纯 Python）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cut = sub.add_parser("cut", help="渔网精准裁剪（城区/荒野/水体）")
    p_cut.add_argument("--raster", default=None,
                       help="多波段特征栅格（默认自动取归一化 Physical_Features_V5）")
    p_cut.add_argument("--fishnet", default=_default("Zhanggong_District_Fishing_Net.shp"))
    p_cut.add_argument("--mndwi", default=None,
                       help="MNDWI 栅格（默认自动取归一化版本）")
    p_cut.add_argument("--out", default=str(paths.output_dir() / "fishnet"))
    p_cut.add_argument("--mndwi-threshold", type=float, default=0.0)
    p_cut.add_argument("--score-threshold", type=int, default=29)
    p_cut.add_argument("--fuzzy-low", type=int, default=27)
    p_cut.add_argument("--fuzzy-high", type=int, default=31)
    p_cut.add_argument("--no-smoothing", action="store_true")
    p_cut.add_argument("--no-ml", action="store_true")
    p_cut.add_argument("--no-plot", action="store_true")

    p_ext = sub.add_parser("extract", help="栅格特征提取到渔网")
    p_ext.add_argument("--fishnet", required=True, help="输入渔网（通常为 cut 产出的城区渔网）")
    p_ext.add_argument("--raster", default=None,
                       help="多波段栅格（默认自动取归一化 Physical_Features_V5）")
    p_ext.add_argument("--out", required=True, help="输出渔网要素类（.shp）")
    p_ext.add_argument("--bands", default="Decay_Idx,Build_Den,Road_Den,Txt_Compl,LandUseMx,Green_Cov",
                       help="波段字段名（逗号分隔，每个 ≤10 字符）")
    p_ext.add_argument("--csv", default=None, help="输出属性表 CSV（默认与 --out 同名）")

    args = ap.parse_args()

    if args.cmd == "cut":
        raster = args.raster or str(paths.normalized_raster("ZhanggongQu_Physical_Features_V5.tif"))
        mndwi = args.mndwi or str(paths.normalized_raster("zhanggong_mndwi_landsat.tif"))
        print(f"[cut] 栅格(归一化): {raster}")
        print(f"[cut] 渔网: {args.fishnet}")
        print(f"[cut] MNDWI(归一化): {mndwi}")
        result = run_cut(raster_path=raster, fishnet_path=args.fishnet,
                         output_dir=args.out, mndwi_path=mndwi,
                         mndwi_water_threshold=args.mndwi_threshold,
                         urban_score_threshold=args.score_threshold,
                         fuzzy_band_low=args.fuzzy_low, fuzzy_band_high=args.fuzzy_high,
                         use_spatial_smoothing=not args.no_smoothing,
                         use_ml_refinement=not args.no_ml,
                         generate_plot=not args.no_plot)
        print("\n─── cut 输出 ───")
        for k, v in result.items():
            print(f"  {k}: {v}")

    elif args.cmd == "extract":
        raster = args.raster or str(paths.normalized_raster("ZhanggongQu_Physical_Features_V5.tif"))
        band_names = [b.strip() for b in args.bands.split(",") if b.strip()] or None
        print(f"[extract] 渔网: {args.fishnet}")
        print(f"[extract] 栅格(归一化): {raster}")
        result = run_extract(fishnet_shp=args.fishnet, raster_tif=raster,
                             output_shp=args.out, band_names=band_names,
                             output_csv=args.csv)
        print("\n─── extract 输出 ───")
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
