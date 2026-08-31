# -*- coding: utf-8 -*-
"""
L0_data_check.py — 数据层体检（L0）

对全部输入做"可读性 + 基准数字"体检并留档：
  1. 矢量 SHP：要素数 / CRS（投影层必须 EPSG:4526）/ 逐列非空率 /
     无效几何计数与 buffer(0) 修复计数；
  2. CSV：utf-8-sig 读取 / 行列数 / 逐列非空率；school_data.csv 坐标落在
     研究区经纬度范围（lon≈[114,115.5], lat≈[25,26.5]）做合理性断言；
  3. 栅格 TIF：尺寸 / CRS / nodata / dtype / 波段数（只读元数据，不载入数据）；
  4. GDB→GPKG 转换复核：逐层要素数一致、字段一致、
     线层总长度 / 面层总面积相对误差 < 0.1%。

输出：
  - 控制台摘要表；
  - tests/L0_baseline.json（全部基准数字，供 L1/L2 引用与日后回归对比）。

失败语义：
  - 文件不可读 / 投影层 CRS 不是 4526 / GPKG 与 GDB 要素数不一致 → 硬失败（非零退出）；
  - 几何修复、CRS 缺失的纯属性表等仅记录计数，不算失败。
"""
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "utils"))
from utils import paths  # noqa: E402

BASELINE_PATH = Path(__file__).parent / "L0_baseline.json"
HARD_ERRORS = []


def nonnull_rates(df: pd.DataFrame) -> dict:
    return {c: round(float(df[c].notna().mean()), 4) for c in df.columns}


def check_vector(data_dir: Path, name: str, desc: str) -> dict:
    p = data_dir / name
    gdf = gpd.read_file(p, engine="pyogrio")
    crs = gdf.crs
    rec = {
        "file": name, "kind": "vector", "desc": desc,
        "features": int(len(gdf)),
        "crs": str(crs) if crs else None,
        "columns": list(gdf.columns),
        "nonnull_rates": nonnull_rates(gdf.drop(columns="geometry", errors="ignore")),
    }
    expected_epsg = desc[1] if isinstance(desc, tuple) else 4526
    desc = desc[0] if isinstance(desc, tuple) else desc
    if crs is None:
        HARD_ERRORS.append(f"{name}: 无 CRS")
    elif crs.to_epsg() != expected_epsg:
        HARD_ERRORS.append(f"{name}: CRS={crs}，预期 EPSG:{expected_epsg}")
    if "geometry" in gdf.columns and len(gdf):
        invalid = int((~gdf.geometry.is_valid).sum())
        empty = int(gdf.geometry.is_empty.sum())
        rec["invalid_geom"] = invalid
        rec["empty_geom"] = empty
        if invalid:
            fixed = gdf[gdf.geometry.is_valid == False].copy()  # noqa: E712
            fixed["geometry"] = fixed.geometry.buffer(0)
            rec["invalid_after_buffer0"] = int((~fixed.geometry.is_valid).sum())
        # 线/面总量基准（L1 复用）
        try:
            if str(gdf.geometry.geom_type.iloc[0]) in ("LineString", "MultiLineString"):
                rec["total_length_m"] = round(float(gdf.geometry.length.sum()), 3)
            elif str(gdf.geometry.geom_type.iloc[0]) in ("Polygon", "MultiPolygon"):
                rec["total_area_m2"] = round(float(gdf.geometry.area.sum()), 3)
        except Exception:
            pass
    print(f"  [VEC] {name:44s} {rec['features']:>7} 要素  CRS={rec['crs']}"
          f"  无效几何={rec.get('invalid_geom', 0)}")
    return rec


def read_csv_robust(p: Path):
    """utf-8-sig 优先，失败回退 GBK 系（名称_POI_ID对照.csv 即 GBK——沿用原 4.6 的容错策略）。"""
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc), enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeDecodeError(f"{p} 无法以 utf-8/gbk 系解码")


def check_csv(data_dir: Path, name: str, desc: str) -> dict:
    p = data_dir / name
    df, enc = read_csv_robust(p)
    rec = {
        "file": name, "kind": "csv", "desc": desc, "encoding": enc,
        "rows": int(len(df)), "columns": list(df.columns),
        "nonnull_rates": nonnull_rates(df),
    }
    if name == "school_data.csv":
        lon = pd.to_numeric(df["经度"], errors="coerce")
        lat = pd.to_numeric(df["纬度"], errors="coerce")
        ok = lon.between(113.5, 115.5).all() and lat.between(25.0, 26.5).all()
        rec["lonlat_range"] = [float(lon.min()), float(lon.max()),
                               float(lat.min()), float(lat.max())]
        if not ok:
            HARD_ERRORS.append(f"{name}: 经纬度超出研究区合理范围 {rec['lonlat_range']}")
        print(f"  [CSV] {name:44s} {rec['rows']:>7} 行   经纬度范围 {rec['lonlat_range']}")
    else:
        print(f"  [CSV] {name:44s} {rec['rows']:>7} 行   {len(df.columns)} 列")
    return rec


def _proj4_keyparams(crs) -> dict:
    """取 proj4 数值参数（展开 +ellps 为 a/rf），用于坐标变换语义层面的等价比较。"""
    import pyproj
    d = {}
    for item in crs.to_proj4().split()[1:]:
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = v
    d.pop("no_defs", None)
    ellps = d.pop("ellps", None)
    if ellps:
        a, rf = pyproj.database.get_ellps_map()[ellps]
        d["a"], d["rf"] = str(a), str(rf)
    return d


def _crs_params_equal(crs_a, crs_b) -> bool:
    da, db = _proj4_keyparams(crs_a), _proj4_keyparams(crs_b)
    if set(da) != set(db):
        return False
    for k in da:
        try:
            if abs(float(da[k]) - float(db[k])) > 1e-9:
                return False
        except ValueError:
            if str(da[k]) != str(db[k]):
                return False
    return True


def check_raster(data_dir: Path, name: str, desc) -> dict:
    """expected：desc=(说明, epsg)。CRS 用 proj4 关键参数语义比较（源数据可能是 WKT 而非 EPSG 代码）。"""
    expected = desc[1] if isinstance(desc, tuple) else 4526
    desc = desc[0] if isinstance(desc, tuple) else desc
    p = data_dir / name
    with rasterio.open(p) as src:
        rec = {
            "file": name, "kind": "raster", "desc": desc, "expected_epsg": expected,
            "width": src.width, "height": src.height, "count": src.count,
            "crs": str(src.crs), "nodata": src.nodata, "dtype": src.dtypes[0],
            "res": [round(src.res[0], 4), round(src.res[1], 4)],
            "bounds": [round(v, 1) for v in src.bounds],
        }
        if src.crs is None:
            HARD_ERRORS.append(f"{name}: 无 CRS")
        else:
            same = _crs_params_equal(src.crs, rasterio.crs.CRS.from_epsg(expected))
            rec["crs_matches_expected"] = bool(same)
            if not same:
                HARD_ERRORS.append(f"{name}: 栅格 CRS 与预期 EPSG:{expected} 不一致（实际 {src.crs.to_string()[:80]}）")
    print(f"  [TIF] {name:44s} {rec['width']}x{rec['height']}  预期EPSG:{expected}"
          f"  匹配={rec.get('crs_matches_expected', '无CRS')}  nodata={rec['nodata']}")
    return rec


def check_gpkg_vs_gdb() -> dict:
    gpkg = paths.converted_gpkg_path()
    gdb = paths.gdb_path()
    if not gpkg.exists():
        HARD_ERRORS.append("GPKG 不存在——请先运行 src/utils/convert_gdb.py")
        return {"skipped": True}
    gpkg_layers = {str(n): g for n, g in pyogrio.list_layers(gpkg)}
    gdb_layers = {str(n): g for n, g in pyogrio.list_layers(gdb)}
    rec = {"gpkg_layers": sorted(gpkg_layers), "comparisons": []}
    for name, info in json.loads(
            (gpkg.parent / "gdb_inventory.json").read_text(encoding="utf-8"))["layers"].items() if False else []:
        pass  # 占位（inventory 为 list，见下）
    inv = json.loads((gpkg.parent / "gdb_inventory.json").read_text(encoding="utf-8"))
    for item in inv["layers"]:
        lname = item["layer"]
        if item.get("written_to") != "gpkg":
            continue
        a = gpd.read_file(gdb, layer=lname, engine="pyogrio")
        b = gpd.read_file(gpkg, layer=lname, engine="pyogrio")
        cmprec = {"layer": lname, "gdb_rows": int(len(a)), "gpkg_rows": int(len(b)),
                  "fields_equal": list(a.columns) == list(b.columns)}
        if len(a) != len(b):
            HARD_ERRORS.append(f"GPKG 层 {lname}: 要素数 {len(b)} != GDB {len(a)}")
        if not cmprec["fields_equal"]:
            HARD_ERRORS.append(f"GPKG 层 {lname}: 字段不一致")
        geom_type = str(a.geometry.geom_type.iloc[0]) if len(a) else ""
        if "LineString" in geom_type:
            sa, sb = float(a.geometry.length.sum()), float(b.geometry.length.sum())
            cmprec["total_length_gdb"] = round(sa, 3)
            cmprec["total_length_gpkg"] = round(sb, 3)
            cmprec["rel_diff"] = abs(sa - sb) / sa if sa else 0.0
        elif "Polygon" in geom_type:
            sa, sb = float(a.geometry.area.sum()), float(b.geometry.area.sum())
            cmprec["total_area_gdb"] = round(sa, 3)
            cmprec["total_area_gpkg"] = round(sb, 3)
            cmprec["rel_diff"] = abs(sa - sb) / sa if sa else 0.0
        if cmprec.get("rel_diff", 0) > 0.001:
            HARD_ERRORS.append(f"GPKG 层 {lname}: 几何总量相对误差 {cmprec['rel_diff']:.2%} > 0.1%")
        rec["comparisons"].append(cmprec)
        print(f"  [GPK] {lname:36s} 行数一致={cmprec['gdb_rows'] == cmprec['gpkg_rows']}"
              f"  字段一致={cmprec['fields_equal']}"
              + (f"  几何总量相对误差={cmprec['rel_diff']:.2e}" if "rel_diff" in cmprec else ""))
    return rec


def main():
    data_dir = paths.data_dir()
    print("=" * 78)
    print(f"L0 数据体检 | 数据目录 = {data_dir}")
    print("=" * 78)

    baseline = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                "data_dir": str(data_dir), "vectors": [], "csvs": [],
                "rasters": [], "gpkg_check": {}}

    print("\n── 矢量 ──")
    for name, desc in paths.VECTOR_INPUTS.items():
        baseline["vectors"].append(check_vector(data_dir, name, desc))
    print("\n── CSV ──")
    for name, desc in paths.CSV_INPUTS.items():
        baseline["csvs"].append(check_csv(data_dir, name, desc))
    print("\n── 栅格 ──")
    for name, desc in paths.RASTER_INPUTS.items():
        baseline["rasters"].append(check_raster(data_dir, name, desc))
    print("\n── GDB→GPKG 转换复核 ──")
    print()
    print("── 坐标量级对照表（标签 EPSG vs 实际量级 vs 纠偏动作）──")
    import crs_tools
    aux = data_dir / "Zhanggong_road_density.tif"
    raster_paths = [data_dir / n for n in paths.RASTER_INPUTS]
    if aux.exists():
        raster_paths.append(aux)
    baseline["crs_axis_table"] = crs_tools.report_table(raster_paths)
    for row in baseline["crs_axis_table"]:
        print("  [CRS] {:44s} 标签={}  量级={:10s} 动作={}".format(
            row["file"], row["tag_epsg"], row["axis"], row["action"]))
    baseline["gpkg_check"] = check_gpkg_vs_gdb()

    baseline["hard_errors"] = HARD_ERRORS
    BASELINE_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print("-" * 78)
    if HARD_ERRORS:
        print(f"L0 FAIL：{len(HARD_ERRORS)} 项硬错误（详见 {BASELINE_PATH}）")
        for e in HARD_ERRORS:
            print("  ✗", e)
        return 1
    print(f"L0 PASS：全部输入可读、CRS 达标、GPKG 复核通过。基线已留档 → {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
