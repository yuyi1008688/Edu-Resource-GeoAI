# -*- coding: utf-8 -*-
"""
normalize_rasters.py — 输入栅格坐标基准归一化运行器

把 5 个正式输入栅格 + 1 个无标签辅助栅格统一规范到主基准
（EPSG:4526，东坐标带号 3850 万），输出到 <B236_OUTPUT_DIR>/normalized/：

  动作映射（详见 crs_tools.py）：
    WorldPop / road_density_fixed / buildings_density  → copy（已是主基准）
    ZhanggongQu_Physical_Features_V5（标签4547，实际去带号）→ 东坐标平移 +38,000,000 + set_crs(4526)
    zhangong_mndwi_landsat（4326 地理坐标）             → 双线性重投影到 4526
    Zhanggong_road_density（无标签，去带号）            → 东坐标平移 +38,000,000 + set_crs(4526)

  去带号平移不做任何重采样/插值，像元值与相对几何和原流程"渔网 to_crs(4547)"
  完全等价——这是对历史行为的忠实保持，不是修正。

用法：
    set B236_DATA_DIR=... && python src/utils/normalize_rasters.py
下游 CLI / run_pipeline 一律读取 normalized/ 下的栅格；core 内的
assert_same_axis 守卫会在任何量级混入时立即报错。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import paths, crs_tools  # noqa: E402


def main():
    data_dir = paths.data_dir()
    out_root = paths.output_dir() / "normalized"
    out_root.mkdir(parents=True, exist_ok=True)

    targets = [data_dir / n for n in paths.RASTER_INPUTS]
    aux = data_dir / "Zhanggong_road_density.tif"
    if aux.exists():
        targets.append(aux)

    print("=" * 78)
    print("栅格坐标基准归一化 → 主基准 EPSG:4526（东坐标带号）")
    print("=" * 78)

    rows, report = [], {"generated_at": datetime.now().isoformat(timespec="seconds"),
                        "normalized": []}
    for src in targets:
        rec = crs_tools.normalize_raster(src, out_root / src.name)
        rows.append(crs_tools.report_table([rec["dst"]])[0])
        rows[-1]["src_file"] = src.name
        rows[-1]["action"] = rec["action"]
        if rec["action"] == "shift_dezoned":
            rows[-1]["shift_m"] = rec.get("shift_m")
            rows[-1]["bounds_after"] = rec.get("bounds_after")
        report["normalized"].append(rec)
        print(f"  {src.name[:44]:46s} 动作={rec['action']:14s} → {Path(rec['dst']).name}")

    (out_root / "normalize_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 78)
    print("标签 EPSG vs 实际量级 vs 纠偏动作（归一化后）：")
    for r in rows:
        print("  {:44s} 标签={}  量级={}  动作={}".format(
            r["file"], r["tag_epsg"], r["axis"], r["action"]))
    print(f"\n报告留档: {out_root / 'normalize_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
