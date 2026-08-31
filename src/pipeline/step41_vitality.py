# -*- coding: utf-8 -*-
"""
step41_vitality.py — 4.1c 城市活力指数计算（纯 Python CLI）

纯 Python CLI：调用 core/vitality_index.VitalityEngine
（KDE 高斯核密度 + Shannon 熵多样性 → Z-score 0.5:0.5 融合 → [0,1]）。

示例：
  python src/pipeline/step41_vitality.py \
      --fishnet output/fishnet/fishnet_urban_only.shp \
      --out output/vitality/vitality_fishnet.gpkg
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


def run(fishnet_path, poi_csv_path, proj_crs="EPSG:4526",
        kde_bandwidth=260.0, output_path=None):
    """城市活力指数（调用 VitalityEngine）。"""
    from vitality_index import VitalityEngine
    engine = VitalityEngine(fishnet_path, proj_crs=proj_crs,
                            kde_bandwidth=kde_bandwidth)
    return engine.calculate_from_csv(poi_csv_path, output_path)


def main():
    ap = argparse.ArgumentParser(description="4.1c 城市活力指数（纯 Python）")
    ap.add_argument("--fishnet", required=True,
                    help="渔网 Shapefile（通常为 cut 产出的城区渔网）")
    ap.add_argument("--poi", default=str(paths.data_dir() / "POI_data.csv"),
                    help="POI CSV（含 lng, lat, cat_id）")
    ap.add_argument("--crs", default="EPSG:4526", help="投影坐标系")
    ap.add_argument("--bandwidth", type=float, default=260.0, help="KDE 带宽（米）")
    ap.add_argument("--out", default=str(paths.output_dir() / "vitality" / "vitality_fishnet.gpkg"),
                    help="输出 GeoPackage")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    result = run(args.fishnet, args.poi, args.crs, args.bandwidth, args.out)
    print(f"  格网数: {len(result)}")
    print(f"  活力均值: {result['vitality'].mean():.4f}")
    print(f"  活力范围: [{result['vitality'].min():.4f}, {result['vitality'].max():.4f}]")
    print(f"  空网格: {result['is_empty'].sum()}")
    print(f"✅ 输出: {args.out}")


if __name__ == "__main__":
    main()
