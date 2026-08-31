# -*- coding: utf-8 -*-
"""
convert_gdb.py — FileGDB → GeoPackage 开源格式转换

用法：
    set B236_DATA_DIR=<输入数据目录>
    python src/utils/convert_gdb.py [--gdb 中间数据.gdb] [--out <gpkg路径>]

功能：
  1. 用 pyogrio（内置 OpenFileGDB 只读驱动）列出 GDB 全部层；
  2. 逐层导出为单一 GeoPackage（矢量层带几何，纯属性表转 CSV 备份）；
  3. 打印每层的层名 / 要素数 / 字段清单 / CRS，并生成 inventory JSON 留档。

设计要点：
  - 跳过 *.gdb_tmp 之类的临时目录；
  - 单层读取失败不中断整体转换，逐层记录错误（L0 体检会复核）；
  - 输出默认写到 <B236_OUTPUT_DIR>/gdb_converted/（已 gitignore）。
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import paths  # noqa: E402


def iter_gdb_dirs(data_dir: Path):
    """列出数据目录下所有正式 FileGDB（排除 *_tmp 临时目录）。"""
    found = sorted(p for p in data_dir.glob("*.gdb") if p.is_dir() and not p.name.endswith("_tmp"))
    if not found:
        raise FileNotFoundError(f"{data_dir} 下未找到 FileGDB 目录（*.gdb）")
    return found


def convert_gdb(gdb: Path, out_gpkg: Path, csv_backup: bool = True) -> dict:
    """把单个 FileGDB 的全部层转换到 out_gpkg，返回 inventory 字典。"""
    if out_gpkg.exists():
        out_gpkg.unlink()
        print(f"  已删除旧输出: {out_gpkg.name}")

    layers = pyogrio.list_layers(gdb)  # ndarray: [[layer_name, geom_type], ...]
    print(f"  OpenFileGDB 共 {len(layers)} 个层")

    inventory = {"gdb": str(gdb), "gpkg": str(out_gpkg), "layers": [], "errors": []}
    for i, (name, geom_type) in enumerate(layers, 1):
        rec = {"layer": str(name), "geom_type": str(geom_type)}
        try:
            # pyogrio.read_dataframe：有几何→GeoDataFrame；纯属性表（如网络数据集
            # 的 N_* 系统表）→普通 DataFrame，两者都能读
            obj = pyogrio.read_dataframe(gdb, layer=str(name))
            rec["features"] = int(len(obj))
            rec["fields"] = [c for c in obj.columns if c != "geometry"]

            is_geo = isinstance(obj, gpd.GeoDataFrame) and "geometry" in obj.columns
            has_geom = bool(
                is_geo and obj.geometry.notna().any() and (~obj.geometry.is_empty).any()
            ) if len(obj) else False
            if has_geom:
                gdf = obj
                rec["crs"] = str(gdf.crs) if gdf.crs else None
                # 追加写入 GPKG（首层自动建库）
                gdf.to_file(out_gpkg, layer=str(name), driver="GPKG",
                            mode="a" if out_gpkg.exists() else "w",
                            engine="pyogrio")
                rec["written_to"] = "gpkg"
                # 几何有效性留档（供 L0 对比）
                rec["invalid_geom"] = int((~gdf.geometry.is_valid).sum())
            else:
                rec["crs"] = None
                rec["written_to"] = "csv_only"

            if csv_backup:
                csv_path = out_gpkg.parent / f"gdb_layer_{name}.csv"
                (obj.drop(columns=["geometry"], errors="ignore")
                    .to_csv(csv_path, index=False, encoding="utf-8-sig"))
                rec["csv_backup"] = str(csv_path)

            print(f"  [{i}/{len(layers)}] {name}: {rec['features']} 行 | "
                  f"CRS={rec['crs']} | 几何={geom_type} | → {rec['written_to']}"
                  + (f" | 无效几何={rec['invalid_geom']}" if has_geom else ""))
        except Exception as e:  # 单层失败不中断
            rec["error"] = repr(e)
            inventory["errors"].append(rec)
            print(f"  [{i}/{len(layers)}] {name}: ✗ 转换失败 → {e!r}")
        inventory["layers"].append(rec)

    return inventory


def main():
    ap = argparse.ArgumentParser(description="FileGDB → GeoPackage 开源转换")
    ap.add_argument("--gdb", default=paths.GDB_NAME, help="GDB 目录名（默认 中间数据.gdb）")
    ap.add_argument("--out", default=None, help="输出 GPKG 路径（默认 output/gdb_converted/）")
    ap.add_argument("--no-csv", action="store_true", help="不生成每层 CSV 备份")
    args = ap.parse_args()

    gdb = paths.data_dir() / args.gdb
    if not gdb.is_dir():
        raise FileNotFoundError(f"GDB 不存在: {gdb}")
    out_gpkg = Path(args.out) if args.out else paths.converted_gpkg_path()
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FileGDB → GPKG 开源转换 | {gdb.name}")
    print("=" * 70)
    inventory = convert_gdb(gdb, out_gpkg, csv_backup=not args.no_csv)
    inventory["finished_at"] = datetime.now().isoformat(timespec="seconds")

    inv_path = out_gpkg.parent / "gdb_inventory.json"
    inv_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for r in inventory["layers"] if "error" not in r)
    print("-" * 70)
    print(f"完成：成功 {ok}/{len(inventory['layers'])} 层 → {out_gpkg}")
    print(f"层清单留档: {inv_path}")
    if inventory["errors"]:
        print(f"⚠ {len(inventory['errors'])} 层失败，详见 inventory 的 errors 字段")
    return 0 if not inventory["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
