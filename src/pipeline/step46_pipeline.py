# -*- coding: utf-8 -*-
"""
step46_pipeline.py — 4.6 学校软实力标签与 SMS 主流水线（纯 Python CLI）

纯 Python CLI：
  - HealthCheck / MainPipeline / ReviewStats → 委托 src/core/sms_engine
    （本 CLI 规范导入 sms_engine，输入输出路径由 B236_DATA_DIR/B236_OUTPUT_DIR 驱动）；
  - CommunityMapBuilder（服务区∩聚类→community_map.csv）→ gpd.overlay 实现，
    上游小学/中学服务区由 step43b 生成、聚类面由 step42 生成；
  - 地图出图统一由 sms_engine 的 matplotlib 完成（fig_4_6_ab_spatial 等 PNG）。

示例：
  python src/pipeline/step46_pipeline.py health
  python src/pipeline/step46_pipeline.py pipeline
  python src/pipeline/step46_pipeline.py pipeline --sweep
  python src/pipeline/step46_pipeline.py review 甲.csv 乙.csv
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


def _bridge_master_name_style(eng):
    """主名单名称风格桥接：SHP 名称风格 → 语料名称风格（统一口径）。

    实测：L1/L2/community_map/POI对照 的校名与语料风格精确一致（99/99），
    而工作区的 99校主名单_shp名称.csv 是 SHP 名称变体（22 处空格差异，
    如“XX中学 (校区)” vs “XX中学(校区)”）。统一后体检通过，
    说明当年引擎吃到的是语料风格 master。此桥接按归一化键（统一全/半角
    括号、去空格）1:1 映射——不动原始数据、不改匹配算法。
    """
    import pandas as pd
    try:
        m = pd.read_csv(eng.PATH_MASTER, encoding="utf-8-sig")
    except UnicodeDecodeError:
        m = pd.read_csv(eng.PATH_MASTER, encoding="gbk")
    l1 = pd.read_csv(eng.PATH_L1, encoding="utf-8-sig")

    def norm(s):
        return str(s).replace("（", "(").replace("）", ")").replace(" ", "")

    l1_by_norm = {}
    for n in l1["学校名称"]:
        l1_by_norm.setdefault(norm(n), []).append(n)

    if set(m["School_Name"]) <= set(l1["学校名称"]):
        return 0  # 已同风格，无需桥接

    mapping, conflicts = {}, []
    for n in m["School_Name"]:
        cands = l1_by_norm.get(norm(n))
        if not cands:
            continue
        if len(cands) > 1:
            conflicts.append(n)
            continue
        if cands[0] != n:
            mapping[n] = cands[0]
    if conflicts:
        raise ValueError(f"master 桥接冲突（归一化后重名，请人工核对）: {conflicts}")

    m["School_Name"] = m["School_Name"].map(lambda x: mapping.get(x, x))
    out = os.path.join(eng.DIR_OUTPUT, "master_bridge.csv")
    m.to_csv(out, index=False, encoding="utf-8-sig")
    eng.PATH_MASTER = out
    print(f"[桥接] 主名单名称风格桥接 {len(mapping)} 处（SHP名 → 语料名）→ {out}")
    return len(mapping)


def configure_engine():
    """把 B236_DATA_DIR/B236_OUTPUT_DIR 注入 sms_engine 的路径配置区。"""
    import sms_engine as eng
    d = str(paths.data_dir())
    o = str(paths.output_dir() / "step46")
    eng.PATH_MASTER = os.path.join(d, "99校主名单_shp名称.csv")
    eng.PATH_L1 = os.path.join(d, "L1_学校自述文本.csv")
    eng.PATH_L2 = os.path.join(d, "L2_媒体报道文本.csv")
    eng.PATH_L3 = os.path.join(d, "L3_教育局公示.csv")
    eng.PATH_POI_MAP = os.path.join(d, "名称_POI_ID对照.csv")
    eng.PATH_COMMUNITY = os.path.join(d, "community_map.csv")
    eng.DIR_INPUT = d
    eng.DIR_OUTPUT = o
    os.makedirs(o, exist_ok=True)
    _bridge_master_name_style(eng)
    return eng


# ══════════════════════════════════════════════════════════════════
# 社区面积制表（= CommunityMapBuilder.execute，Intersect → gpd.overlay 等价）
# ══════════════════════════════════════════════════════════════════
THESIS = {"1": "C3_产业工人区", "2": "C1_新城高知区", "3": "C4_成熟居住区",
          "4": "C5_城乡过渡带", "5": "C6_隐性收缩区", "6": "C2_老城退休区"}
AREA_COLS = [THESIS[str(i)] for i in range(1, 7)]


def _canon_cluster(v):
    """纯数字编码优先（防'1'被名称匹配抢先），名称兜底（原样）。"""
    s = str(v).strip()
    if s.isdigit():
        return THESIS.get(s)
    for k, name in THESIS.items():
        if name in s or ("C" + k) == s or ("Cluster" + k) == s:
            return name
    return None


def _detect_column(cols, cands):
    for cand in cands:
        for n in cols:
            if cand.lower() == n.lower():
                return n
    for cand in cands:
        for n in cols:
            if cand in n:
                return n
    return None


def run_community_map(es_sa, jh_sa, cluster_fc, out_csv, unit="km2") -> dict:
    import geopandas as gpd
    import statistics
    try:
        from crs_tools import assert_same_axis
    except ImportError:
        assert_same_axis = None

    factor = 1e-6 if unit != "m2" else 1.0
    cluster = gpd.read_file(cluster_fc, engine="pyogrio")
    cfield = _detect_column(list(cluster.columns), ["CLUSTER_ID", "CLUSTER", "社区", "类型"])
    if not cfield:
        raise ValueError("聚类层无 CLUSTER_ID/社区 字段，实际：" + str(list(cluster.columns)))
    print("聚类类字段：" + cfield)

    book, order = {}, []
    for fc, stage in ((es_sa, "小学"), (jh_sa, "中学")):
        g = gpd.read_file(fc, engine="pyogrio")
        nfield = _detect_column(list(g.columns), ["名称", "学校", "NAME", "School"])
        if not nfield:
            raise ValueError(stage + "服务区找不到校名字段，实际：" + str(list(g.columns)))
        if str(g.crs) != str(cluster.crs):
            g = g.to_crs(cluster.crs)
        if assert_same_axis:
            assert_same_axis(g, cluster, context=f"community_map[{stage}]")
        print(f"交集计算：{stage} ∩ 聚类层 …")
        inter = gpd.overlay(g[[nfield, "geometry"]], cluster[[cfield, "geometry"]],
                            how="intersection", keep_geom_type=False)
        inter = inter[inter.geometry.notna() & (~inter.geometry.is_empty)].copy()
        inter["_a"] = inter.geometry.area * factor
        per_school = {}
        for sch, cl, a in zip(inter[nfield], inter[cfield], inter["_a"]):
            cn = _canon_cluster(cl)
            if not cn:
                continue
            d = per_school.setdefault(str(sch).strip(), {})
            d[cn] = d.get(cn, 0.0) + float(a)
        for sch, d in per_school.items():
            book.setdefault(sch, {})
            for cn, a in d.items():
                book[sch][cn] = book[sch].get(cn, 0.0) + a
            if sch not in order:
                order.append(sch)
        totals = [sum(d.values()) for d in per_school.values()] or [0.0]
        mean_v = statistics.mean(totals)
        cv = (statistics.pstdev(totals) / mean_v) if mean_v else 0.0
        print(f"  {stage}段：{len(per_school)} 校，总 {sum(totals):.2f}，"
              f"单校均值 {mean_v:.2f}，CV={cv:.2f}")
        if cv < 0.15 and len(totals) > 3:
            print(f"  ⚠️ 同质巨面指纹（CV={cv:.2f}<0.15）：疑似旧错叠灾难版服务区，请核对面层！")

    import csv
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["学校名称"] + AREA_COLS)
        for sch in order:
            d = book[sch]
            w.writerow([sch] + [round(d.get(cn, 0.0), 6) for cn in AREA_COLS])
    idx_total = sum(sum(book[s].values()) for s in order)
    print("=" * 50)
    print(f"✅ 写出 {out_csv} | 校数={len(order)} 六类合计={idx_total:.2f}")
    print(f"对照基准（真值）：99 校 / 合计 911.6436 km²")
    if len(order) != 99:
        print(f"⚠️ 校数≠99（={len(order)}），请检查服务区与聚类层覆盖/名称归一")
    return {"output": out_csv, "schools": len(order), "total": idx_total}


def main():
    ap = argparse.ArgumentParser(description="4.6 软实力标签与 SMS 主流水线（纯 Python）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="数据就位体检")
    p_pipe = sub.add_parser("pipeline", help="主引擎")
    p_pipe.add_argument("--scope", choices=["all99", "edu95"], default="all99")
    p_pipe.add_argument("--sweep", action="store_true")
    p_pipe.add_argument("--conf-threshold", type=float, default=0.70)
    p_pipe.add_argument("--window", type=int, default=60)
    p_pipe.add_argument("--alpha", type=float, default=0.15)

    p_rev = sub.add_parser("review", help="人工复核统计")
    p_rev.add_argument("file_a", nargs="?", default="人工复核_甲.csv")
    p_rev.add_argument("file_b", nargs="?", default="人工复核_乙.csv")

    p_cm = sub.add_parser("community-map",
                          help="服务区∩聚类面 → community_map.csv")
    p_cm.add_argument("--es-sa", required=True, help="小学服务区面")
    p_cm.add_argument("--jh-sa", required=True, help="中学服务区面")
    p_cm.add_argument("--cluster", required=True, help="社区聚类面（CLUSTER_ID=1~6）")
    p_cm.add_argument("--out", required=True, help="输出 community_map.csv")
    p_cm.add_argument("--unit", choices=["km2", "m2"], default="km2")

    args = ap.parse_args()
    eng = configure_engine()

    if args.cmd == "health":
        eng.run_health_check()
    elif args.cmd == "pipeline":
        eng.run_pipeline(scope=args.scope, sweep=args.sweep,
                         conf_threshold=args.conf_threshold,
                         window=args.window, alpha=args.alpha)
    elif args.cmd == "review":
        eng.run_review_stats(args.file_a, args.file_b)
    else:
        run_community_map(args.es_sa, args.jh_sa, args.cluster, args.out, args.unit)


if __name__ == "__main__":
    main()
