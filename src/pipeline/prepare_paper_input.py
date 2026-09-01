# -*- coding: utf-8 -*-
"""
prepare_paper_input.py — 期刊版输入数据准备（可审计、不修改原始数据）

清洗规则（义务教育口径）：
  研究对象限定为九年义务教育学校（小学 + 初中/含初中部的完中），剔除"普通高中"。
  原始 99 所学校中类型=='高中'的恰好 4 所，且其中 2 所同时是重复爬取点：
    - XBAZE 赣州四中-云鹰楼   : 与 WWDPG 赣州四中仅距 28.9m，是同校园内一栋楼被 POI 误作独立学校
    - IJWKX 赣州中学(赣州外国语): 与 LSVEY 江西省赣州中学同地址(赣康路20号)、同电话，同校双 POI
    - ETRVO 赣州市星苑高级中学 : 独立纯高中，非义务教育
    - QCHZS 赣州博雅高级中学   : 独立纯高中，非义务教育
  剔除后 99 -> 95 所（小学 53 + 中学 42）。

说明：爬虫所得"学生数/教师数"精度低、教育局未提供官方值，且本项目 D3 压力、GeoXGBoost、
MILP、2SFCA 全链路均不使用学生数（需求=WorldPop人口×0.1174学龄比，供给=建筑物理容量），
故学生数仅作输出表展示列，不对其做任何插补/修正。

用法：
  python src/pipeline/prepare_paper_input.py --src-dir <原始输入> --out-dir <期刊版输入>
"""
import argparse
import os
import shutil
import sys

import pandas as pd
import geopandas as gpd

# 义务教育保留类型；其余（高中）剔除
KEEP_TYPES = {"小学", "中学"}
# 显式剔除清单（双保险：与"类型==高中"规则互相校验）
EXCLUDE_IDS = {
    "XBAZE": "赣州四中-云鹰楼｜与赣州四中仅距28.9m的同校园楼栋，POI误作独立学校",
    "IJWKX": "赣州中学(赣州外国语学校)｜与江西省赣州中学同址同电话，同校双POI",
    "ETRVO": "赣州市星苑高级中学｜独立普通高中，非九年义务教育",
    "QCHZS": "赣州博雅高级中学｜独立普通高中，非九年义务教育",
}
SCHOOL_CSV = "school_data.csv"
MIDDLE_SHP = "Middle_school.shp"
PRIMARY_SHP = "Primary_school.shp"
SHP_SIDECAR = [".shp", ".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx",
               ".qmd", ".qix"]


def log(m):
    print(m, flush=True)


def clean_school_csv(src, dst):
    df = pd.read_csv(src, encoding="utf-8-sig")
    before = len(df)
    hs = df[df["类型"] == "高中"]
    hs_ids = set(hs["school_id"])
    # 双保险校验：类型==高中 的集合必须恰好等于显式清单
    assert hs_ids == EXCLUDE_IDS.keys(), (
        "类型为高中的集合 %s 与显式剔除清单 %s 不一致，请人工复核"
        % (hs_ids, set(EXCLUDE_IDS)))
    kept = df[df["类型"].isin(KEEP_TYPES)].copy()
    kept.to_csv(dst, index=False, encoding="utf-8-sig")
    log("[school_data.csv] %d -> %d（剔除 %d 所高中）"
        % (before, len(kept), before - len(kept)))
    for _, r in hs.iterrows():
        log("    剔除 %s | %s | %s"
            % (r["school_id"], r["名称"], EXCLUDE_IDS[r["school_id"]]))
    return kept


def clean_middle_shp(src_dir, dst_dir):
    src = os.path.join(src_dir, MIDDLE_SHP)
    g = gpd.read_file(src)
    before = len(g)
    kept = g[~g["school_id"].isin(EXCLUDE_IDS)].copy()
    out = os.path.join(dst_dir, MIDDLE_SHP)
    kept.to_file(out, driver="ESRI Shapefile", encoding="utf-8")
    log("[Middle_school.shp] %d -> %d（剔除 %d 所高中设施，CRS=%s）"
        % (before, len(kept), before - len(kept), kept.crs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    # 1) 全量复制原始文件（学校CSV/中学SHP族除外，它们走清洗逻辑）
    skip_names = {SCHOOL_CSV}
    skip_stems = {os.path.splitext(MIDDLE_SHP)[0]}
    n_copy = 0
    for f in sorted(os.listdir(a.src_dir)):
        sp = os.path.join(a.src_dir, f)
        if not os.path.isfile(sp):
            continue
        stem, ext = os.path.splitext(f)
        if f == SCHOOL_CSV or stem in skip_stems:
            continue  # 单独清洗生成
        shutil.copy2(sp, os.path.join(a.out_dir, f))
        n_copy += 1
    log("已原样复制 %d 个其他输入文件" % n_copy)

    # 1b) 复制需要的子目录（FileGDB 是目录形态；gdb_tmp 为重复临时副本，不复制）
    for d in ["中间数据.gdb"]:
        sp = os.path.join(a.src_dir, d)
        if os.path.isdir(sp):
            dp = os.path.join(a.out_dir, d)
            shutil.copytree(sp, dp, dirs_exist_ok=True)
            log("已复制目录 %s" % d)
        else:
            raise FileNotFoundError("缺少必需的输入目录：%s" % sp)

    # 2) 清洗学校表
    clean_school_csv(
        os.path.join(a.src_dir, SCHOOL_CSV),
        os.path.join(a.out_dir, SCHOOL_CSV))

    # 3) 清洗中学设施 shp（小学不含高中，随步骤1原样复制）
    clean_middle_shp(a.src_dir, a.out_dir)

    # 4) 清洗说明
    note = os.path.join(a.out_dir, "CLEANING_NOTE.md")
    with open(note, "w", encoding="utf-8") as fp:
        fp.write(
            "# 期刊版输入数据清洗说明\n\n"
            "本目录由 `src/pipeline/prepare_paper_input.py` 从原始输入自动生成，原始数据未改动。\n\n"
            "## 清洗规则\n研究对象限定九年义务教育（小学+中学），剔除普通高中，99 -> 95 所"
            "（小学53 + 中学42）。\n\n## 剔除清单（4 所高中）\n\n"
            "| school_id | 名称 | 原因 |\n|---|---|---|\n")
        for sid, reason in EXCLUDE_IDS.items():
            name, why = reason.split("｜")
            fp.write(f"| {sid} | {name} | {why} |\n")
        fp.write(
            "\n## 学生数/教师数说明\n爬虫学生数精度低且未获教育局官方值；全链路"
            "（D3/GeoXGBoost/MILP/2SFCA）均不使用学生数（需求=WorldPop×0.1174学龄比，"
            "供给=建筑物理容量），学生数仅作展示列，不做插补。\n")
    log("已写出清洗说明 %s" % note)
    log("期刊版输入准备完成 -> %s" % a.out_dir)


if __name__ == "__main__":
    sys.exit(main())
