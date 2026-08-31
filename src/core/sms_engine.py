# -*- coding: utf-8 -*-
"""
================================================================================
 sms_engine.py— 4.6 学校软实力标签体系 · 单文件整合版
 整合自: data_health_check.py / main_pipeline.py / review_stats.py
================================================================================

【运行方式】
  python sms_engine.py health                        # 数据就位体检
  python sms_engine.py pipeline                      # 主引擎（默认模式）
  python sms_engine.py pipeline --sweep               # 主引擎 + 稳健性检验
  python sms_engine.py pipeline --scope edu95         # 95校历史口径
  python sms_engine.py review 甲.csv 乙.csv           # 人工复核统计

【依赖】pandas, numpy, matplotlib（标准 Anaconda 环境即可）
       chardet 为可选依赖，未安装时自动回退到编码探测序列

【编码】读取时自动检测 UTF-8 / GBK / GB2312；写入统一 UTF-8-SIG

【路径说明】
  以本脚本所在目录为基准自动定位输入/输出目录。
  输入数据子文件夹：<脚本所在目录>/输入数据/
  输出数据子文件夹：<脚本所在目录>/输出数据/
  如需自定义路径，修改下方"文件路径配置区"即可。

【主名单CSV字段说明（新版）】
  school_id   : 随机字母ID（如 BNMQX）
  School_ID   : 数字序号（如 0, 1, 2, ...）
  School_Name : 学校全称（如 XX小学）
  Level       : 学校类型（如 小学 / 初中 / 高中 / 中学）
  经度        : 经度坐标
  纬度        : 纬度坐标
================================================================================
"""
import os, sys, csv, re, io, argparse
from collections import Counter

# ============================================================================
# ★ 宿主环境 stdout 兼容补丁
#   某些宿主环境的 sys.stdout 对象没有 reconfigure() 方法，
#   必须在调用前做防御性处理，否则被 importlib 加载时模块级代码会立即崩溃。
# ============================================================================
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # 该类环境下忽略该调用，不影响任何功能
        pass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ============================================================================
# ★★★  文件路径配置区  ★★★
# 默认：以本脚本（.py）所在目录为基准，自动推算输入/输出目录。
# 默认以本脚本所在目录为基准，无需手动修改路径。
# 如需自定义，直接将下方各变量改为绝对路径字符串即可，例如：
#   PATH_MASTER = r"D:\MyData\99校主名单_shp名称.csv"
# ============================================================================

# 本脚本所在目录（自动定位）
_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- 输入文件 ----------

# [1] 99校主名单（含 school_id / School_ID / School_Name / Level / 经度 / 纬度）
PATH_MASTER     = os.path.join(_HERE, "输入数据", "99校主名单_shp名称.csv")

# [2] L1 学校自述文本（含列: 学校名称 / L1_text / L1_source / L1_date）
PATH_L1         = os.path.join(_HERE, "输入数据", "L1_学校自述文本.csv")

# [3] L2 媒体报道文本（含列: 学校名称 / L2_title / L2_summary / L2_source / L2_date）
PATH_L2         = os.path.join(_HERE, "输入数据", "L2_媒体报道文本.csv")

# [4] L3 教育局公示（含列: 学校名称 / L3_tag / L3_document / L3_date / L3_url）
PATH_L3         = os.path.join(_HERE, "输入数据", "L3_教育局公示.csv")

# [5] 学校名称→POI_ID 对照表（含列: 名称 / POI_ID）
PATH_POI_MAP    = os.path.join(_HERE, "输入数据", "名称_POI_ID对照.csv")

# [6] 社区面积/类型映射表（含列: 学校名称 + C1~C6面积列 或 社区类型列）
PATH_COMMUNITY  = os.path.join(_HERE, "输入数据", "community_map.csv")

# ---------- 输出目录 ----------

# 所有输出文件（CSV / PNG）统一写入此目录，不存在时自动创建
DIR_OUTPUT      = os.path.join(_HERE, "输出数据")

# ---------- 体检时的输入目录（health 子命令用）----------

# health 子命令扫描的目录，与上方各 PATH_* 文件所在目录保持一致
DIR_INPUT       = os.path.join(_HERE, "输入数据")

# ============================================================================
# （路径配置结束，以下无需修改）
# ============================================================================

os.makedirs(DIR_OUTPUT, exist_ok=True)


# ============================================================================
# 编码自动检测辅助函数
# ============================================================================
def _detect_encoding_bytes(raw_bytes):
    """从字节数据自动检测编码（UTF-8 / GBK / GB2312 兼容）。

    优先使用 chardet 自动探测，未安装时回退到依次尝试常见编码序列。
    """
    try:
        import chardet
        result = chardet.detect(raw_bytes)
        enc  = result.get("encoding")
        conf = result.get("confidence", 0)
        if enc and conf > 0.7:
            try:
                raw_bytes.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                pass
    except ImportError:
        pass
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "gb2312", "latin-1"]:
        try:
            raw_bytes.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8-sig"


def _read_csv(filepath):
    """自动检测编码并返回 csv.DictReader（去除 BOM）。"""
    with open(filepath, "rb") as f:
        raw = f.read()
    enc  = _detect_encoding_bytes(raw)
    text = raw.decode(enc).lstrip("\ufeff")
    return csv.DictReader(io.StringIO(text))


def _read_csv_pd(filepath):
    """自动检测编码并用 pandas 读取 CSV（去除 BOM）。"""
    with open(filepath, "rb") as f:
        raw = f.read()
    enc  = _detect_encoding_bytes(raw)
    text = raw.decode(enc).lstrip("\ufeff")
    return pd.read_csv(io.StringIO(text))


# ============================================================================
# 共享工具函数
# ============================================================================
def load_csv(path):
    """读取 CSV，自动检测编码，返回 list[dict]。"""
    return list(_read_csv(path))


def kappa_stats(pairs):
    """Kappa / Po / F1 统计。"""
    n  = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    tp = sum(1 for a, b in pairs if a and b)
    fp = sum(1 for a, b in pairs if a and not b)
    fn = sum(1 for a, b in pairs if b and not a)
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    return (
        po,
        (po - pe) / (1 - pe) if pe < 1 else 1.0,
        2 * prec * rec / (prec + rec) if prec + rec else 0
    )


# ============================================================================
# 中文字体
# ============================================================================
def _get_chinese_font():
    cands = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei',
             'Microsoft YaHei', 'PingFang SC', 'STHeiti', 'STKaiti',
             'SimSun', 'Droid Sans Fallback', 'Arial Unicode MS']
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in cands:
        if c in avail:
            return c
    for a in avail:
        if any(k in a for k in ('CJK', 'Hei', 'Song', 'Kai')):
            return a
    return 'sans-serif'


CJK      = _get_chinese_font()
plt.rcParams['font.sans-serif']    = [CJK, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
CJK_PROP = fm.FontProperties(family=CJK)


# ============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  子模块 A：数据就位体检                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ============================================================================

EXPECTED = {
    "L1_学校自述文本.csv": {
        "min_rows":     99,
        "bad_markers":  ["待人工补充", "未找到"],
        "warn_markers": [],
        "key":    "L1_text",
        "expect": "99行/99校全覆盖"
    },
    "L2_媒体报道文本.csv": {
        "min_rows":     100,
        "bad_markers":  ["待人工补充"],
        "warn_markers": [],
        "key":    "L2_summary",
        "expect": "111行/94校"
    },
    "L3_教育局公示.csv": {
        "min_rows":     60,
        "bad_markers":  ["待核实"],
        "warn_markers": ["待核验"],
        "key":    "L3_tag",
        "expect": "60行/39校/30类(1条已知待核验保留)"
    },
}


def run_health_check(data_dir=None):
    """执行数据就位体检。data_dir 为 None 时使用路径配置区的 DIR_INPUT。"""
    data_dir = data_dir or DIR_INPUT
    print("=" * 66)
    print("4.6 数据体检 | 数据目录 = {}".format(data_dir))
    print("=" * 66)

    ok_all = True
    for fname, spec in EXPECTED.items():
        p = os.path.join(data_dir, fname)
        if not os.path.exists(p):
            print("⛔ {}: 文件不存在（基准：{}）".format(fname, spec["expect"]))
            ok_all = False
            continue

        rows = list(_read_csv(p))
        n    = len(rows)
        bad  = sum(
            1 for r in rows
            if any(m in (r.get(spec["key"]) or "") for m in spec["bad_markers"])
        )
        warn = sum(
            1 for r in rows
            if any(m in (r.get(spec["key"]) or "") for m in spec.get("warn_markers", []))
        )

        # 校名一致性检查（仅 L1；主名单现在用 School_Name 列）
        master_ok = True
        if fname == "L1_学校自述文本.csv":
            if os.path.exists(PATH_MASTER):
                master = {r["School_Name"] for r in _read_csv(PATH_MASTER)}
                cur    = {r["学校名称"]    for r in rows}
                diff   = cur - master
                if diff:
                    master_ok = False
                    print("   ⚠ 校名与SHP主名单不一致 {} 个: {}".format(
                        len(diff), sorted(diff)[:5]))

        tag = "OK" if (n >= spec["min_rows"] and bad == 0 and master_ok) else \
              ("WARN" if n >= spec["min_rows"] and bad < n else "FAIL")

        if bad > 0 or n < spec["min_rows"] or not master_ok:
            ok_all = False

        print("[{}] {}: {}行（基准>={}），致命占位 {} 行，待核验保留 {} 行〔基准:{}〕".format(
            tag, fname, n, spec["min_rows"], bad, warn, spec["expect"]))

    print("-" * 66)
    if ok_all:
        print("体检通过：可运行  python sms_engine.py pipeline")
        print("   预期控制台（99口径）：有标签约74/99（覆盖率74.7%）；真值错配=19校")
    else:
        print("数据未就位：请确认路径配置区的各 PATH_* 变量指向正确文件")
        sys.exit(1)


# ============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  子模块 B：主引擎                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ============================================================================

# --- B.1 POI_ID 映射加载 ---
def _load_poi_map():
    """学校名称 → 真实 POI_ID 映射。"""
    m = {}
    if os.path.isfile(PATH_POI_MAP):
        for r in _read_csv(PATH_POI_MAP):
            nm  = (r.get("名称")   or "").strip()
            pid = (r.get("POI_ID") or "").strip()
            if nm and pid and pid.lower() != "nan":
                m[nm] = pid
        print("[POI] 加载真实POI_ID映射 {} 条：{}".format(len(m), PATH_POI_MAP))
    if not m:
        print("[POI] ⚠ 未找到 POI_ID 对照表 → 退回占位（禁止入论文数据）")
    return m


# --- B.2 4×8 标签词典 ---
# 从共享模块导入权威常量（单一来源：constants.py，原 _4_6_constants.py）
# 修改标签/权重时只需更新 constants.py，本文件自动同步
# ★ 仓库布局导入引导：src/utils 加入 sys.path（平铺布局下原模块名仍优先生效）
_UTILS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "utils"))
if os.path.isdir(_UTILS_DIR) and _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
try:
    from _4_6_constants import (      # 平铺布局兼容
        ENTITY_COMMON, GENERIC, NEGATION,
        DICT, TKEYS, SDIMS, ECONF, DOMMAP,
        AHBP_THESIS, ALIAS_42, DEBUG_DEMAND
    )
except ImportError:
    from constants import (           # 仓库布局（src/utils/constants.py）
        ENTITY_COMMON, GENERIC, NEGATION,
        DICT, TKEYS, SDIMS, ECONF, DOMMAP,
        AHBP_THESIS, ALIAS_42, DEBUG_DEMAND
    )


def _canon_community(label):
    s = str(label).strip()
    if s in AHBP_THESIS:
        return s
    if s in ALIAS_42:
        return ALIAS_42[s]
    for c, v in ALIAS_42.items():
        if s and (s in c or c in s):
            return v
    return None


# --- B.4 数据加载 ---
def _load_all():
    """加载全部输入数据，并校验主名单字段完整性。"""
    l1     = load_csv(PATH_L1)
    l2     = load_csv(PATH_L2)
    l3     = load_csv(PATH_L3)
    master = load_csv(PATH_MASTER)

    required = {"school_id", "School_ID", "School_Name", "Level", "经度", "纬度"}
    if master:
        actual  = set(master[0].keys())
        missing = required - actual
        if missing:
            raise KeyError(
                "[主名单] 缺少必要字段: {}\n"
                "  实际字段 : {}\n"
                "  文件路径 : {}".format(missing, sorted(actual), PATH_MASTER)
            )
    return l1, l2, l3, master


def _build_corpus(school, l1, l2):
    out = []
    for r in l1:
        if r["学校名称"] == school and (r.get("L1_text") or "").strip():
            out.append(("L1", r.get("L1_source", ""), r.get("L1_date", ""), r["L1_text"]))
    for r in l2:
        if r["学校名称"] == school:
            out.append(("L2", r.get("L2_source", ""), r.get("L2_date", ""),
                        (r.get("L2_title") or "") + "。 " + (r.get("L2_summary") or "")))
    return out


# --- B.5 E0-E5 证据规则引擎 ---
def _cooccur(corpus, anchors, entities, window):
    for tier, src, date, text in corpus:
        for a in anchors:
            for m in re.finditer(re.escape(a), text):
                seg = text[max(0, m.start() - window): m.end() + window]
                if any(e in seg and e != a for e in entities):
                    if any(neg in seg for neg in NEGATION):
                        continue
                    s  = text.rfind("。", 0, m.start()) + 1
                    en = text.find("。", m.end())
                    return True, "[{}|{}|{}] {}".format(
                        tier, src, date,
                        text[s: en if en != -1 else len(text)][:120]
                    )
    return False, ""


def _self_entity(corpus, self_anchors):
    for tier, src, date, text in corpus:
        for a in self_anchors:
            for m in re.finditer(re.escape(a), text):
                seg = text[max(0, m.start() - 20): m.end() + 20]
                if any(neg in seg for neg in NEGATION):
                    continue
                return True, "[{}|{}|{}] 实体型锚点直接命中：{}".format(tier, src, date, a)
    return False, ""


def _l3_honor(l3rows, honors):
    ev = []
    for r in l3rows:
        blob = (r.get("L3_tag", "") + "|" + r.get("L3_document", ""))
        for h in honors:
            if h in blob:
                ev.append((r.get("L3_tag", ""), (r.get("L3_date") or "")[:10],
                           "（无官方URL" in (r.get("L3_url") or "")))
                break
    return ev


def _judge(school, corpus, l3rows, window):
    res = {}
    for sd, kw in SDIMS:
        honor    = _l3_honor(l3rows, kw["honor"])
        co, ev   = _cooccur(corpus, kw["anchor"], set(kw["extra"]) | set(ENTITY_COMMON), window)
        if not co and kw.get("self"):
            co, ev = _self_entity(corpus, kw["self"])
        hits = {a for _, _, _, t in corpus for a in kw["anchor"] if a in t}
        if honor and co:
            E = "E1"
        elif honor:
            E = "E2"
        elif co:
            E = "E3"
        elif len(hits) >= 2:
            E = "E4"
            ev = "关键词命中：" + ", ".join(sorted(hits)[:6])
        elif hits:
            E = "E5"
            ev = "泛化/单词命中：" + ", ".join(sorted(hits))
        else:
            E = "E0"
            ev = ""
        if honor:
            ev = (ev + "｜" if ev else "") + "L3认定：" + "；".join(
                "{}({})".format(t, d) for t, d, _w in honor
            ) + ("〔含待核验源〕" if any(w for *_x, w in honor) else "")
        res[sd] = (E, ECONF[E], ev, bool(honor))
    return res


def _rollup(sdres, th):
    perT = {}
    for T in TKEYS:
        best = None
        for sd in DICT[T]:
            E, conf, ev, hl = sdres[sd]
            if conf >= th and (best is None or conf > best[1]
                               or (conf == best[1] and hl and not best[3])):
                best = (sd, conf, ev, hl)
        if best:
            perT[T] = best
    return perT


def _school_type(perT):
    n = len(perT)
    if n == 0:
        return "无特色均衡发展型"
    if n >= 3:
        return "全面综合型"
    if n == 2:
        return "复合协同发展型"
    return {
        "T1_科创素养": "STEM引领型",
        "T2_审美素养": "艺体特色型",
        "T3_身心素养": "身心关爱型",
        "T4_实践素养": "实践创新型"
    }[next(iter(perT))]


def _dominant(perT):
    if not perT:
        return "无特色"
    b = max(perT.items(), key=lambda kv: (kv[1][1], kv[1][3]))
    return DOMMAP[b[0]]


# --- B.6 SMS ---
def _load_demand(cmap_path, schools, debug=False):
    Dbar = {}
    if cmap_path and os.path.exists(cmap_path):
        rows = list(_read_csv(cmap_path))
        if not rows:
            return None, "PENDING"
        cols      = rows[0].keys()
        area_cols = [c for c in cols if _canon_community(c)]
        if area_cols:
            for r in rows:
                areas = {_canon_community(c): float(r.get(c) or 0) for c in area_cols}
                tot   = sum(areas.values()) or 1.0
                Dbar[r["学校名称"]] = {
                    k: sum(areas[c] / tot * AHBP_THESIS[c][k] for c in areas)
                    for k in ("T1", "T2", "T3", "T4")
                }
        elif "社区类型" in cols:
            for r in rows:
                c = _canon_community(r["社区类型"])
                if c:
                    Dbar[r["学校名称"]] = dict(AHBP_THESIS[c])
        return Dbar, "REAL"
    if debug:
        return {s: dict(DEBUG_DEMAND) for s in schools}, "DEBUG"
    return None, "PENDING"


def _calc_sms(res, schools, Dbar, alpha):
    if not Dbar:
        return {}
    tmap = {"T1_科创素养": "T1", "T2_审美素养": "T2",
            "T3_身心素养": "T3", "T4_实践素养": "T4"}
    raw  = {}
    for s in schools:
        d      = Dbar.get(s) or dict(DEBUG_DEMAND)
        base   = sum(res[s]["perT"][T][1] * d[tmap[T]] for T in res[s]["perT"])
        raw[s] = base * (1 + alpha * res[s]["vr"])
    mx = max(raw.values()) or 1.0
    return {s: v / mx for s, v in raw.items()}


def _calc_mismatch(res, sms, schools):
    elig = sorted(sms[s] for s in schools if res[s]["perT"])
    if not elig:
        return {s: 0 for s in schools}
    q1 = elig[max(0, len(elig) // 4)]
    return {s: 1 if (res[s]["perT"] and sms[s] <= q1) else 0 for s in schools}


# --- B.7 Spearman ---
def _spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rk(x), rk(y)
    n   = len(x)
    mx  = sum(rx) / n
    my  = sum(ry) / n
    sxy = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx  = sum((v - mx) ** 2 for v in rx) ** 0.5
    sy  = sum((v - my) ** 2 for v in ry) ** 0.5
    return sxy / (sx * sy) if sx * sy else 1.0


# --- B.8 图件 ---
MARKERS = {
    "T1_STEM":     ("o", "#d62728"),
    "T2_艺体":     ("s", "#1f77b4"),
    "T3_心理德育": ("^", "#2ca02c"),
    "T4_劳动实践": ("D", "#ff7f0e"),
    "无特色":      ("x", "#7f7f7f")
}


def _fig_ab(smdf, tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ax = axes[0]
    for dt, (mk, c) in MARKERS.items():
        sub = smdf[smdf.dominant_type == dt]
        ax.scatter(sub.lon, sub.lat, marker=mk, c=c, s=45,
                   label="{} ({})".format(dt, len(sub)), alpha=.85,
                   edgecolors="white", linewidths=.4)
    ax.set_title("图4-6-a 学校软实力主导类型空间分布", fontproperties=CJK_PROP)
    ax.legend(prop=CJK_PROP, loc="best", fontsize=8)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax = axes[1]
    sc = ax.scatter(smdf.lon, smdf.lat, c=smdf.SMS_score, cmap="RdYlGn",
                    vmin=0, vmax=1, s=55, edgecolors="k", linewidths=.3)
    plt.colorbar(sc, ax=ax, label="SMS（归一化）")
    mm = smdf[smdf.mismatch_flag == 1]
    ax.scatter(mm.lon, mm.lat, facecolors="none", edgecolors="red", s=120,
               linewidths=1.2, label="错配候选 {}校".format(len(mm)))
    ax.set_title("图4-6-b SMS空间分布", fontproperties=CJK_PROP)
    ax.legend(prop=CJK_PROP, loc="best")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    plt.tight_layout()
    p = os.path.join(DIR_OUTPUT, "fig_4_6_ab_spatial{}.png".format(tag))
    plt.savefig(p, dpi=300)
    plt.close()
    return p


def _fig_types(smdf):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    c = smdf.school_type_label.value_counts()
    axes[0].barh(range(len(c)), c.values, color="#1f77b4")
    axes[0].set_yticks(range(len(c)))
    axes[0].set_yticklabels(c.index, fontproperties=CJK_PROP)
    axes[0].set_title("7类学校综合类型分布", fontproperties=CJK_PROP)
    axes[0].set_xlabel("学校数")
    sub = smdf[smdf.eligible_tag_count > 0]
    axes[1].hist(sub.l3_verification_rate, bins=10, color="#2ca02c", edgecolor="white")
    axes[1].axvline(0.6, color="red", ls="--", label="60%参考线")
    axes[1].set_title("L3验证率分布（有eligible标签的学校）", fontproperties=CJK_PROP)
    axes[1].set_xlabel("L3验证率", fontproperties=CJK_PROP)
    axes[1].set_ylabel("学校数",   fontproperties=CJK_PROP)
    axes[1].legend(prop=CJK_PROP)
    plt.tight_layout()
    p = os.path.join(DIR_OUTPUT, "fig_4_6_ce_type_L3.png")
    plt.savefig(p, dpi=300)
    plt.close()
    return p


def _fig_heat(detail_rows):
    el   = [r for r in detail_rows if r["eligible_tag_count"] > 0]
    el   = sorted(el, key=lambda r: (r["dominant_type"], -r["l3_verification_rate"]))
    data = np.array([[r["{}_conf".format(sd)] for sd, _ in SDIMS] for r in el])
    fig, ax = plt.subplots(figsize=(11, max(6, len(el) * 0.16 + 2)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(SDIMS)))
    ax.set_xticklabels([sd for sd, _ in SDIMS], rotation=30, ha="right",
                       fontproperties=CJK_PROP)
    ax.set_yticks(range(len(el)))
    ax.set_yticklabels([r["school_name"] for r in el],
                       fontproperties=CJK_PROP, fontsize=6)
    ax.set_title("8个二级维度置信度热力图（{}所有标签学校）".format(len(el)),
                 fontproperties=CJK_PROP)
    plt.colorbar(im, ax=ax, label="置信度")
    plt.tight_layout()
    p = os.path.join(DIR_OUTPUT, "fig_4_6_d_confidence_heatmap.png")
    plt.savefig(p, dpi=300)
    plt.close()
    return p


# --- B.9 主流程 ---
def run_pipeline(scope="all99", debug_demand=False, sweep=False,
                 conf_threshold=0.70, window=60, alpha=0.15):
    """执行 4.6 学校软实力标签与SMS 主引擎。"""

    print("=" * 72)
    print("4.6 主引擎  ·  路径配置一览")
    print("=" * 72)
    print("  [输入1] 99校主名单      : {}".format(PATH_MASTER))
    print("  [输入2] L1学校自述文本  : {}".format(PATH_L1))
    print("  [输入3] L2媒体报道文本  : {}".format(PATH_L2))
    print("  [输入4] L3教育局公示    : {}".format(PATH_L3))
    print("  [输入5] POI_ID对照表    : {}".format(PATH_POI_MAP))
    print("  [输入6] 社区映射表      : {}".format(PATH_COMMUNITY))
    print("  [输出]  输出目录        : {}".format(DIR_OUTPUT))
    print("  scope={}  conf_th={}  window={}  alpha={}".format(
        scope, conf_threshold, window, alpha))
    print("=" * 72)

    l1, l2, l3, master = _load_all()

    schools    = [r["School_Name"] for r in master]
    stype      = {r["School_Name"]: r["Level"]     for r in master}
    geo        = {r["School_Name"]: (float(r["经度"]), float(r["纬度"])) for r in master}
    school_id  = {r["School_Name"]: r["school_id"] for r in master}
    school_num = {r["School_Name"]: r["School_ID"] for r in master}

    print("[加载] 主名单 {} 校  |  字体: {}".format(len(schools), CJK))

    comps = (schools[:]
             if scope == "all99"
             else [s for s in schools if stype[s] not in ("高中",)])
    print("[口径] scope={} → 输出 {} 校".format(scope, len(comps)))

    # E级判定
    detail = {}
    for s in schools:
        l3rows    = [r for r in l3 if r["学校名称"] == s]
        detail[s] = _judge(s, _build_corpus(s, l1, l2), l3rows, window)

    # 输出1：学校标签_E级明细.csv
    path_e_detail = os.path.join(DIR_OUTPUT, "学校标签_E级明细.csv")
    with open(path_e_detail, "w", encoding="utf-8-sig", newline="") as f:
        w   = csv.writer(f)
        hdr = ["school_id", "School_ID", "School_Name", "Level"]
        for sd, _ in SDIMS:
            hdr += ["{}_E级".format(sd), "{}_置信度".format(sd), "{}_证据".format(sd)]
        w.writerow(hdr)
        for s in schools:
            row = [school_id[s], school_num[s], s, stype[s]]
            for sd, _ in SDIMS:
                E, conf, ev, _ = detail[s][sd]
                row += [E, conf, ev]
            w.writerow(row)
    print("[输出1] {}".format(path_e_detail))

    # 聚合
    res = {}
    for s in schools:
        perT = _rollup(detail[s], conf_threshold)
        eli  = len(perT)
        vr   = (sum(v[3] for v in perT.values()) / eli) if eli else 0.0
        res[s] = {
            "perT":     perT,
            "dominant": _dominant(perT),
            "stl":      _school_type(perT),
            "vr":       vr,
            "tags":     ";".join(
                "{}[{},E={}]".format(T, v[0], v[1]) for T, v in sorted(perT.items())
            )
        }

    # SMS
    Dbar, mode = _load_demand(PATH_COMMUNITY, schools, debug_demand)
    sms        = _calc_sms(res, schools, Dbar, alpha)
    mismatch   = _calc_mismatch(res, sms, schools) if sms else {}

    # 输出2：soft_match_results.csv
    suffix    = {"REAL": "", "DEBUG": "_DEBUG", "PENDING": "_pending"}[mode]
    path_soft = os.path.join(DIR_OUTPUT, "soft_match_results{}.csv".format(suffix))
    with open(path_soft, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "school_id", "School_ID", "School_Name", "Level",
            "dominant_type", "school_type_label",
            "mismatch_flag", "SMS_score",
            "l3_verification_rate", "eligible_tag_count", "active_tags_eligible"
        ])
        for s in comps:
            r = res[s]
            w.writerow([
                school_id[s], school_num[s], s, stype[s],
                r["dominant"], r["stl"],
                mismatch.get(s, "") if sms else "",
                "{:.4f}".format(sms[s]) if sms else "",
                "{:.3f}".format(r["vr"]),
                len(r["perT"]),
                r["tags"]
            ])
    print("[输出2] {}  (SMS mode={})".format(path_soft, mode))

    # 输出3：interface_46_to_47.csv
    path_i47 = os.path.join(DIR_OUTPUT, "interface_46_to_47.csv")
    pd.DataFrame([{
        "school_id":            school_id[s],
        "School_ID":            school_num[s],
        "School_Name":          s,
        "Level":                stype[s],
        "dominant_type":        res[s]["dominant"],
        "school_type_label":    res[s]["stl"],
        "mismatch_flag":        mismatch.get(s, "") if sms else "",
        "SMS_score":            "{:.4f}".format(sms[s]) if sms else "",
        "l3_verification_rate": "{:.3f}".format(res[s]["vr"]),
        "eligible_tag_count":   len(res[s]["perT"]),
        "active_tags_eligible": res[s]["tags"]
    } for s in comps]).to_csv(path_i47, index=False, encoding="utf-8-sig")
    print("[输出3] {}".format(path_i47))

    # 输出4：图4-6制图数据.csv
    rows_plot = []
    for s in comps:
        lon, lat = geo[s]
        rows_plot.append({
            "school_id":            school_id[s],
            "School_ID":            school_num[s],
            "School_Name":          s,
            "Level":                stype[s],
            "lon":                  lon,
            "lat":                  lat,
            "dominant_type":        res[s]["dominant"],
            "school_type_label":    res[s]["stl"],
            "eligible_tag_count":   len(res[s]["perT"]),
            "l3_verification_rate": res[s]["vr"],
            "SMS_score":            sms.get(s, np.nan),
            "mismatch_flag":        mismatch.get(s, 0)
        })
    smdf           = pd.DataFrame(rows_plot)
    path_plot_data = os.path.join(DIR_OUTPUT, "图4-6制图数据{}.csv".format(suffix))
    smdf.to_csv(path_plot_data, index=False, encoding="utf-8-sig")
    print("[输出4] {}".format(path_plot_data))

    # 验证统计
    n     = len(comps)
    cover = sum(1 for s in comps if res[s]["perT"]) / n
    elig  = sum(len(res[s]["perT"]) for s in comps)
    l3b   = sum(v[3] for s in comps for v in res[s]["perT"].values())
    pairs = [(detail[s][sd][0] in ("E1", "E3"), detail[s][sd][3])
             for s in comps for sd, _ in SDIMS]
    po, kappa, f1 = kappa_stats(pairs)
    print("[验证] 覆盖率={:.1f}%  L3交叉率={:.1f}% (elig={})  "
          "Kappa={:.3f}  Po={:.3f}  F1={:.3f}".format(
              cover * 100, l3b / elig * 100, elig, kappa, po, f1))
    print("[分布] dominant={}".format(
        dict(Counter(res[s]["dominant"] for s in comps))))

    # 稳健性检验
    if sweep:
        t070 = {s: set(_rollup(detail[s], 0.70).keys()) for s in comps}
        for th in (0.65, 0.75):
            tx = {s: set(_rollup(detail[s], th).keys()) for s in comps}
            print("[稳健] 阈值0.70→{}: 标签集合变化 {:.1f}% (<15)".format(
                th, sum(1 for s in comps if tx[s] != t070[s]) / n * 100))
        for wnd in (40, 80):
            d2  = {s: _judge(s, _build_corpus(s, l1, l2),
                             [r for r in l3 if r["学校名称"] == s], wnd)
                   for s in comps}
            e3b = {s: set(sd for sd, _ in SDIMS if detail[s][sd][0] == "E3")
                   for s in comps}
            e3x = {s: set(sd for sd, _ in SDIMS if d2[s][sd][0] == "E3")
                   for s in comps}
            print("[稳健] 窗口60→{}: E3集合变化 {:.1f}% (<10)".format(
                wnd, sum(1 for s in comps if e3b[s] != e3x[s]) / n * 100))
        if sms:
            for al_ in (0.10, 0.20):
                s2 = _calc_sms(res, schools, Dbar, al_)
                print("[稳健] alpha0.15→{}: SMS Spearman rho={:.4f} (>0.90)".format(
                    al_, _spearman([sms[s] for s in comps],
                                   [s2[s] for s in comps])))

    # 图件
    figs = [_fig_types(smdf)]
    if sms:
        figs.append(_fig_ab(smdf, suffix))
    det_rows = []
    for s in comps:
        one = {
            "school_name":          s,
            "dominant_type":        res[s]["dominant"],
            "l3_verification_rate": res[s]["vr"],
            "eligible_tag_count":   len(res[s]["perT"])
        }
        for sd, _ in SDIMS:
            one["{}_conf".format(sd)] = detail[s][sd][1]
        det_rows.append(one)
    figs.append(_fig_heat(det_rows))
    for p in figs:
        print("[图件] {}".format(p))

    print("=" * 72)
    print("完成。全部输出 → {}".format(DIR_OUTPUT))
    print("=" * 72)


# ============================================================================
# 子模块 C：人工复核统计
# ============================================================================

VALID = {"E1", "E2", "E3", "E4", "E5", "E0"}


def _to_bin(e):
    return 1 if str(e).strip() in ("E1", "E2", "E3") else 0


def _majority(df):
    """
    多数决：两人一致取一致值，不一致取更保守值。
    依赖列：_r1 / _r2（已在 run_review_stats 中赋值）
    """
    out = []
    for _, r in df.iterrows():
        if r["_r1"] == r["_r2"]:
            out.append(r["_r1"])
        else:
            # 不一致时取 eligible 倾向更低的（更保守）
            out.append(r["_r1"] if _to_bin(r["_r1"]) <= _to_bin(r["_r2"])
                       else r["_r2"])
    return out


def _load_review(path, rater_num):
    """
    加载评审员复核文件。

    rater_num=1 → 读「人工复核员1结论(E级)」（甲文件）
    rater_num=2 → 读「人工复核员2结论(E级)」（乙文件）

    实际列名（已确认）：
      学校名称 / 二级维度 / 规则引擎E级 / 置信度 /
      证据（来源|日期|句）/
      人工复核员1结论(E级) / 人工复核员2结论(E级)
    """
    df = _read_csv_pd(path)

    print("  文件: {}".format(os.path.basename(path)))
    print("  全部列名: {}".format(list(df.columns)))

    # 目标列名（精确）
    target = "人工复核员{}结论(E级)".format(rater_num)

    if target in df.columns:
        col = target
    else:
        # 模糊匹配：含「复核员N」且含「结论」或「E级」
        candidates = [
            c for c in df.columns
            if "复核员{}".format(rater_num) in c
            and ("结论" in c or "E级" in c)
        ]
        assert candidates, (
            "找不到第{}位评审员的结论列！\n"
            "  期望列名: 「{}」\n"
            "  实际列名: {}\n"
            "  请确认CSV列名是否正确".format(
                rater_num, target, list(df.columns))
        )
        col = candidates[0]
        print("  ⚠ 未找到精确列名「{}」，改用「{}」".format(target, col))

    print("  → 采用列「{}」".format(col))

    v = df[col].astype(str).str.strip()

    # 校验合法值
    bad = sorted(set(v) - VALID - {"nan", ""})
    assert not bad, (
        "列「{}」含非法E级取值: {}\n"
        "  合法值仅限: E0 E1 E2 E3 E4 E5".format(col, bad)
    )

    # 校验无空值
    unfilled = v.isin(["nan", ""]).sum()
    assert unfilled == 0, (
        "列「{}」还有 {} 行未填写！\n"
        "  请填写完整后重新运行".format(col, unfilled)
    )

    df["_rater"] = v
    return df


def run_review_stats(file_a, file_b):
    """
    执行人工复核结果统计。

    输入：
      file_a —— 人工复核_甲.csv（含「人工复核员1结论(E级)」）
      file_b —— 人工复核_乙.csv（含「人工复核员2结论(E级)」）

    输出（写入当前工作目录）：
      rater_stats_summary.csv
      rater_full_with_flags.csv
      rater_disagreements.csv（有分歧时生成）
    """
    print("=" * 64)
    print("人工复核统计")
    print("  评审员甲(1号): {}".format(file_a))
    print("  评审员乙(2号): {}".format(file_b))
    print("=" * 64)

    # ── 加载 ──
    a = _load_review(file_a, 1)
    b = _load_review(file_b, 2)

    # ── 行数校验 ──
    assert len(a) == len(b), (
        "两文件行数不一致: 甲={} 行，乙={} 行\n"
        "请确认是否为同一批抽样记录".format(len(a), len(b))
    )
    n_rows = len(a)
    print("\n共 {} 条记录".format(n_rows))
    if n_rows != 35:
        print("  ⚠ 提示：原设计35行，当前{}行，继续计算".format(n_rows))

    # ── 拼合 ──
    key_cols = [c for c in a.columns if c in (
        "学校名称",
        "二级维度",
        "规则引擎E级",
        "置信度",
        "证据（来源|日期|句）"
    )]

    df = a[key_cols].copy()

    # 赋值评审员结论（_r1/_r2 供 _majority() 使用）
    df["_r1"] = a["_rater"].values
    df["_r2"] = b["_rater"].values
    df["_eng"] = df["规则引擎E级"].astype(str).str.strip()

    # 多数决（调用全局 _majority，依赖 _r1/_r2）
    df["_maj"] = _majority(df)

    # 一致性标记
    df["评审是否一致"] = (df["_r1"] == df["_r2"]).map(
        {True: "一致", False: "不一致"})

    # ── 二值化 ──
    eng_bin = df["_eng"].map(_to_bin).tolist()
    r1_bin  = df["_r1"].map(_to_bin).tolist()
    r2_bin  = df["_r2"].map(_to_bin).tolist()
    maj_bin = df["_maj"].map(_to_bin).tolist()

    # ── 统计指标 ──
    acc_exact    = (df["_eng"] == df["_maj"]).mean()
    acc_exact_r1 = (df["_eng"] == df["_r1"]).mean()
    acc_exact_r2 = (df["_eng"] == df["_r2"]).mean()

    po_m, k_m, f1_m = kappa_stats(list(zip(eng_bin, maj_bin)))
    po_r, k_r, f1_r = kappa_stats(list(zip(r1_bin,  r2_bin)))
    inter_same       = (df["_r1"] == df["_r2"]).mean()

    # ── 打印 ──
    print("\n" + "=" * 64)
    print("统计结果")
    print("=" * 64)
    print("【准确率（E级精确一致）】")
    print("  机器 vs 多数决  : {:.1f}%  目标 >85%".format(acc_exact    * 100))
    print("  机器 vs 评审员1 : {:.1f}%".format(acc_exact_r1 * 100))
    print("  机器 vs 评审员2 : {:.1f}%".format(acc_exact_r2 * 100))
    print("【人机一致性（二值 eligible/not）】")
    print("  Po={:.3f}  Kappa={:.3f}  F1={:.3f}".format(po_m, k_m, f1_m))
    print("【评审员间一致性】")
    print("  精确一致率={:.1f}%  Po={:.3f}  Kappa={:.3f}  F1={:.3f}".format(
        inter_same * 100, po_r, k_r, f1_r))

    dis = df[df["_r1"] != df["_r2"]]
    print("【分歧条目】{} 条".format(len(dis)))

    if k_m < 0.6:
        print("\n⚠ 人机Kappa={:.3f} < 0.60，按既定口径改报 Po/F1".format(k_m))

    # ── 输出（重命名_r1/_r2/_maj为可读列名）──
    df_out = df[key_cols].copy()
    df_out["评审员1结论"] = df["_r1"]
    df_out["评审员2结论"] = df["_r2"]
    df_out["多数决结论"]  = df["_maj"]
    df_out["评审是否一致"] = df["评审是否一致"]

    # [1] 汇总统计
    pd.DataFrame([{
        "总条数":                       n_rows,
        "accuracy_exact_vs_majority_%": round(acc_exact    * 100, 1),
        "accuracy_vs_r1_%":             round(acc_exact_r1 * 100, 1),
        "accuracy_vs_r2_%":             round(acc_exact_r2 * 100, 1),
        "Po_machine_human":             round(po_m,        3),
        "Kappa_machine_human":          round(k_m,         3),
        "F1_machine_human":             round(f1_m,        3),
        "inter_rater_agree_%":          round(inter_same   * 100, 1),
        "inter_rater_Po":               round(po_r,        3),
        "inter_rater_Kappa":            round(k_r,         3),
        "inter_rater_F1":               round(f1_r,        3),
        "disagreements":                len(dis),
    }]).to_csv("rater_stats_summary.csv", index=False, encoding="utf-8-sig")

    # [2] 全量明细
    df_out.to_csv("rater_full_with_flags.csv", index=False, encoding="utf-8-sig")

    # [3] 分歧清单
    if len(dis):
        df_out[df["_r1"] != df["_r2"]].to_csv(
            "rater_disagreements.csv", index=False, encoding="utf-8-sig")

    print("\n已输出:")
    print("  ✓ rater_stats_summary.csv")
    print("  ✓ rater_full_with_flags.csv")
    if len(dis):
        print("  ✓ rater_disagreements.csv（{}条分歧）".format(len(dis)))
        
# ============================================================================
# 命令行入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4.6 学校软实力标签体系 · 整合版")
    sub    = parser.add_subparsers(dest="command")

    p_health = sub.add_parser("health", help="数据就位体检")
    p_health.add_argument("--data-dir", default=None)

    p_pipe = sub.add_parser("pipeline", help="主引擎")
    p_pipe.add_argument("--scope",          choices=["all99", "edu95"], default="all99")
    p_pipe.add_argument("--debug-demand",   action="store_true")
    p_pipe.add_argument("--sweep",          action="store_true")
    p_pipe.add_argument("--conf-threshold", type=float, default=0.70)
    p_pipe.add_argument("--window",         type=int,   default=60)
    p_pipe.add_argument("--alpha",          type=float, default=0.15)

    p_review = sub.add_parser("review", help="人工复核统计")
    p_review.add_argument("file_a", nargs="?", default="人工复核_甲.csv")
    p_review.add_argument("file_b", nargs="?", default="人工复核_乙.csv")

    args = parser.parse_args()

    if args.command == "health":
        run_health_check(data_dir=args.data_dir)
    elif args.command == "review":
        run_review_stats(args.file_a, args.file_b)
    else:
        run_pipeline(
            scope          = getattr(args, "scope",          "all99"),
            debug_demand   = getattr(args, "debug_demand",   False),
            sweep          = getattr(args, "sweep",          False),
            conf_threshold = getattr(args, "conf_threshold", 0.70),
            window         = getattr(args, "window",         60),
            alpha          = getattr(args, "alpha",          0.15),
        )