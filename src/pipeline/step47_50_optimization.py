# -*- coding: utf-8 -*-
"""
step47_50_optimization.py — 4.7 教育资源优化配置 + 5.0 效益评估（纯 Python CLI）

纯 Python CLI：
  - 包含全部业务类（SoftpowerImputer/DataLoader/CapacityEstimator/DemandGridBuilder/
    RoadNetworkBuilder/NetworkOD/CapacityAllocationOptimizer/BenefitEvaluator/
    Visualizer）与 opt47/eval50 两个子命令；
  - 日志统一 print，参数走 argparse（默认值与原始实现一致），不支持 GDB 栅格数据集；
  - 算法公式、权重（W_*）、阈值、约束保持不变。

上游依赖：
  --school-profile-csv 为 step44 ECFI 衍生（含 geometry_wkt/ECFI/priority_score），
  --school-pressure-csv 为 step45 GeoXGBoost 产出；服务区由 step43b 生成。

示例：
  python src/pipeline/step47_50_optimization.py opt47 --out-dir output/step47 ...
  python src/pipeline/step47_50_optimization.py eval50 --out-dir output/step50 ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "core"), str(_SRC / "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import paths  # noqa: E402


# ── 日志层：统一 print ──
def _log(msg):
    print(msg)


def _logw(msg):
    print("[警告]", msg)


def _loge(msg):
    print("[错误]", msg)


class _MsgShim:
    def addErrorMessage(self, m):
        print("[错误]", m)


class _Param:
    """把 argparse 参数适配为执行体所需的 value / valueAsText 接口。"""

    def __init__(self, v=None):
        self.value = v

    @property
    def valueAsText(self):
        return None if self.value is None else str(self.value)


# -*- coding: utf-8 -*-
# ============================================================
# 优化与效益评估 CLI
# 包含两个工具：
#   Tool_47 - 4.7 教育资源优化配置
#   Tool_50 - 5.0 效益评估
#
# ★ 导入策略：逻辑内联 + 少量模块级导入。教育优化算法（2SFCA、PuLP LP）
#   参数较多，统一在命令行/配置中传入。
#   通用工具函数（weighted_gini, compute_2sfca 等）保留在模块级别以供复用。
# ============================================================

import os
import sys
import json
import warnings
import time
import numpy as np
import pandas as pd

# NumPy 2.0+ 兼容：trapz → trapezoid，使用局部引用避免全局 monkey-patching
_np_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # numpy>=2 兼容修正：原写法在默认参数中急切求值 np.trapz 会崩

import geopandas as gpd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

from pathlib import Path
from shapely import wkt as shapely_wkt
from shapely.geometry import Point, box
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ★ 延迟导入 pulp（避免模块级 ImportError）
#   pulp 仅在 Tool_47（教育资源优化配置）的 MCLP 求解器中使用，
#   不影响 Tool_50（效益评估）的运行。延迟到 solve() 调用时才检查。
_pulp = None

def _get_pulp():
    """延迟获取 pulp，未安装时给出清晰的中文安装指引。"""
    global _pulp
    if _pulp is None:
        try:
            import pulp as _p
            _pulp = _p
        except ImportError:
            raise ImportError(
                "缺少 PuLP 包，无法执行 MCLP 线性规划优化。\n"
                "请在当前 Python 环境中运行：\n"
                "    pip install pulp\n"
                "或使用项目附带的 requirements.txt 一键安装：\n"
                "    pip install -r requirements.txt"
            )
    return _pulp

warnings.filterwarnings("ignore")

# ============================================================
# 参数定义
# ============================================================


# ============================================================
# 全局工具与常数
# ============================================================
RISK_ZH = {1: "极低", 2: "低", 3: "中", 4: "高", 5: "极高"}

RISK_COLOR = {
    1: "#4DAC26", 2: "#B8E186", 3: "#FDB863", 4: "#D6604D", 5: "#762A83"
}

SCI_COLORS = {
    "blue":   "#2166AC", "red":    "#D6604D", "green":  "#4DAC26",
    "orange": "#F4A582", "purple": "#762A83", "gray":   "#878787",
    "yellow": "#FDB863", "teal":   "#01665E", "brown":  "#8C510A", "pink": "#DE77AE"
}

def _setup_matplotlib():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.titlesize": 13, "axes.linewidth": 0.8, "grid.linewidth": 0.5,
        "lines.linewidth": 1.5, "axes.grid": True, "grid.alpha": 0.35,
        "grid.color": "#CCCCCC", "grid.linestyle": "--",
        "figure.dpi": 150, "savefig.dpi": 300,
        "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.framealpha": 0.85, "legend.edgecolor": "#AAAAAA",
    })

def safe_read_csv(path):
    for enc in ["utf-8-sig", "gbk", "gb18030", "utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1")

def minmax_norm(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def parse_point_wkt(wkt_str):
    try:
        geom = shapely_wkt.loads(str(wkt_str))
        return geom.x, geom.y
    except Exception:
        return np.nan, np.nan

def get_risk_num(df):
    if "risk_cls" in df.columns:
        return pd.to_numeric(df["risk_cls"], errors="coerce").fillna(1).clip(1, 5).astype(int)
    if "risk_zh" in df.columns:
        m = {"极低": 1, "低": 2, "中": 3, "高": 4, "极高": 5}
        return df["risk_zh"].map(m).fillna(1).astype(int)
    return pd.Series(np.ones(len(df), dtype=int), index=df.index)

def level_to_type(level_str):
    s = str(level_str)
    if "小学" in s: return "小学"
    if any(k in s for k in ["初中", "中学", "九年", "初级", "高中", "高级", "完全", "十二年"]):
        return "中学"
    return "其他"

def parse_dominant_type_family(dtype_str):
    s = str(dtype_str).strip()
    if s in ("nan", "", "无特色", "None"): return "无特色", ""
    parts = s.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (s, "")

def weighted_gini(values_arr, weights_arr):
    values_arr  = np.asarray(values_arr,  dtype=float)
    weights_arr = np.asarray(weights_arr, dtype=float)
    order = np.argsort(values_arr)
    v_s, w_s = values_arr[order], weights_arr[order]
    w_total = w_s.sum()
    if w_total < 1e-9: return 0.0
    vw_total = (v_s * w_s).sum()
    if vw_total < 1e-9: return 0.0
    lorenz_x = np.concatenate([[0.0], np.cumsum(w_s) / w_total])
    lorenz_y = np.concatenate([[0.0], np.cumsum(v_s * w_s) / vw_total])
    return float(1.0 - 2.0 * _np_trapezoid(lorenz_y, lorenz_x))

def compute_2sfca(time_matrix, schools_df, grids_df, school_capacities, demand_arr, catchment_min, decay_beta=0.15):
    n_grid, n_school = time_matrix.shape
    t_max = catchment_min
    def gaussian_decay(t):
        return np.where(t <= t_max, np.exp(-decay_beta * (t / t_max) ** 2), 0.0)
    R = np.zeros(n_school)
    for j in range(n_school):
        if school_capacities[j] <= 0: continue
        wd = (demand_arr * gaussian_decay(time_matrix[:, j])).sum()
        R[j] = school_capacities[j] / wd if wd > 1e-6 else school_capacities[j]
    A = np.zeros(n_grid)
    for i in range(n_grid):
        A[i] = (R * gaussian_decay(time_matrix[i, :])).sum()
    return A

def compute_simple_access_time(time_matrix, schools_df, cfg):
    pri_mask = (schools_df["school_type"] == "小学").values
    mid_mask = (schools_df["school_type"] == "中学").values
    th_p, th_m = cfg["PRIMARY_MAX_MIN"], cfg["MIDDLE_MAX_MIN"]
    n_grid = time_matrix.shape[0]
    time_pri = np.full(n_grid, np.inf)
    time_mid = np.full(n_grid, np.inf)
    for i in range(n_grid):
        if pri_mask.any():
            t_p = time_matrix[i, pri_mask]
            t_pv = t_p[t_p <= th_p]
            if len(t_pv) > 0: time_pri[i] = t_pv.min()
        if mid_mask.any():
            t_m = time_matrix[i, mid_mask]
            t_mv = t_m[t_m <= th_m]
            if len(t_mv) > 0: time_mid[i] = t_mv.min()
    ps, ms = cfg["PRIMARY_AGE_SHARE"], cfg["MIDDLE_AGE_SHARE"]
    time_combined = np.full(n_grid, np.inf)
    for i in range(n_grid):
        hp, hm = not np.isinf(time_pri[i]), not np.isinf(time_mid[i])
        if hp and hm: time_combined[i] = ps * time_pri[i] + ms * time_mid[i]
        elif hp:      time_combined[i] = time_pri[i]
        elif hm:      time_combined[i] = time_mid[i]
    return time_pri, time_mid, time_combined


# ============================================================
# 核心指标控制台可视化函数
# ============================================================
def print_summary_banner(fr, er, mc, opt):
    """打印核心指标汇总面板"""
    SEP  = "=" * 56
    DASH = "-" * 56

    def chk(condition):
        return "✔" if condition else "✘"

    gini_impr     = fr["gini_before"] - fr["gini_after"]
    la_impr_n     = fr["n_low_access_improved"]
    la_total_n    = fr["n_low_access_grids"]
    la_impr_pct   = fr["low_access_improve_rate"] * 100
    hr_sfca_delta = fr["hr_sfca_after"] - fr["hr_sfca_before"]
    # 低可达性SFCA均值改善
    A_b = fr["_A_b"]
    A_a = fr["_A_a"]
    lm  = fr["_low_access_mask"]
    la_sfca_delta = float((A_a[lm] - A_b[lm]).mean()) if lm.sum() > 0 else 0.0

    cov_cv   = mc["cov_cv"]
    gini_cv  = mc["gini_cv"]
    lp_cov   = er["coverage_rate"]
    exp_n    = er["expanded_n"]
    new_seats= er["total_new_seats"]

    lines = [
        SEP,
        "  \U0001F4CB  核心指标汇总（2SFCA口径）",
        SEP,
        "",
        f"  LP覆盖率：            {lp_cov:>10.2%}",
        f"  2SFCA Gini改善：      {gini_impr:>+10.4f}  [{chk(gini_impr >= 0)}]",
        f"  低可达性格网数：      {la_total_n:>10d} 个",
        f"  低可达性格网改善数：  {la_impr_n:>6d} 个  ({la_impr_pct:.1f}%)",
        f"  低可达性SFCA改善：    {la_sfca_delta:>+10.4f}",
        f"  高风险格网SFCA改善：  {hr_sfca_delta:>+10.4f}",
        f"  扩容学校数：          {exp_n:>10d} 所",
        f"  新增学位数：          {int(new_seats):>10d}",
        f"  MC覆盖率CV：          {cov_cv:>10.4f}  [{chk(cov_cv < 0.05)}]",
        f"  MC 2SFCA Gini CV：    {gini_cv:>10.4f}  [{chk(gini_cv < 0.10)}]",
        "",
        SEP,
    ]
    for line in lines:
        _log(line)


def print_47_progress_banner(stage, detail=""):
    """在4.7运行过程中打印进度节点"""
    SEP = "-" * 50
    _log(SEP)
    _log(f"  ▶  {stage}")
    if detail:
        _log(f"     {detail}")
    _log(SEP)


# ============================================================
# 4.7 核心业务类
# ============================================================
class SoftpowerImputer:
    def __init__(self, k=5, shrinkage_alpha=0.15, min_neighbors=3):
        self.k = k
        self.alpha = shrinkage_alpha
        self.min_n = min_neighbors
        self.scaler = StandardScaler()
        self.model_sms = KNeighborsRegressor(n_neighbors=k, weights="distance")
        self.model_l3 = KNeighborsRegressor(n_neighbors=k, weights="distance")
        self._global_sms_mean = 0.0
        self._global_l3_mean = 0.0

    def _build_features(self, df):
        feats = []
        for col in ["x", "y", "D3_pressure", "ECFI", "priority_score", "risk_cls"]:
            if col in df.columns:
                feats.append(pd.to_numeric(df[col], errors="coerce").fillna(0).values)
        type_code = (df["school_type"] == "中学").astype(float).values if "school_type" in df.columns else np.zeros(len(df))
        feats.append(type_code)
        for col in ["dist_to_center", "poi_diversity", "student_count"]:
            if col in df.columns:
                feats.append(pd.to_numeric(df[col], errors="coerce").fillna(0).values)
        return np.nan_to_num(np.column_stack(feats), nan=0.0)

    def fit_transform(self, school_gdf):
        df = school_gdf.copy()
        has_data = df["softpower_data_available"].astype(bool) if "softpower_data_available" in df.columns else (df["SMS_score"] > 0)
        n_known, n_missing = has_data.sum(), (~has_data).sum()
        _log(f"软实力KNN插补: 已知{n_known}所, 缺失{n_missing}所")
        df["SMS_score_imputed_flag"] = 0
        df["SMS_score_impute_std"] = 0.0
        df["softpower_impute_confidence"] = 1.0
        if n_missing == 0:
            return df
        if n_known < self.min_n:
            gm = df.loc[has_data, "SMS_score"].mean() if n_known > 0 else 0.3
            df.loc[~has_data, "SMS_score"] = gm
            df.loc[~has_data, "l3_verification"] = df.loc[has_data, "l3_verification"].mean() if n_known > 0 else 0.0
            df.loc[~has_data, "SMS_score_imputed_flag"] = 1
            df.loc[~has_data, "softpower_impute_confidence"] = 0.0
            return df
        X_all = self._build_features(df)
        X_known = X_all[has_data.values]
        X_missing = X_all[~has_data.values]
        y_sms = df.loc[has_data, "SMS_score"].values.astype(float)
        y_l3 = df.loc[has_data, "l3_verification"].values.astype(float)
        self.scaler.fit(X_known)
        Xk_sc = self.scaler.transform(X_known)
        Xm_sc = self.scaler.transform(X_missing)
        self._global_sms_mean = float(y_sms.mean())
        self._global_l3_mean = float(y_l3.mean())
        actual_k = min(self.k, n_known)
        self.model_sms.set_params(n_neighbors=actual_k)
        self.model_l3.set_params(n_neighbors=actual_k)
        self.model_sms.fit(Xk_sc, y_sms)
        self.model_l3.fit(Xk_sc, y_l3)
        pred_sms = np.clip((1 - self.alpha) * self.model_sms.predict(Xm_sc) + self.alpha * self._global_sms_mean, 0, 1)
        pred_l3 = np.clip((1 - self.alpha) * self.model_l3.predict(Xm_sc) + self.alpha * self._global_l3_mean, 0, 1)
        tree = cKDTree(Xk_sc)
        dists, idxs = tree.query(Xm_sc, k=actual_k)
        if actual_k == 1:
            idxs = idxs.reshape(-1, 1)
            dists = dists.reshape(-1, 1)
        impute_std = y_sms[idxs].std(axis=1)
        confidence = np.clip(0.6 / (1 + dists.mean(axis=1)) + 0.4 / (1 + impute_std * 5), 0, 1)
        missing_idx = df.index[~has_data]
        df.loc[missing_idx, "SMS_score"] = pred_sms
        df.loc[missing_idx, "l3_verification"] = pred_l3
        df.loc[missing_idx, "SMS_score_imputed_flag"] = 1
        df.loc[missing_idx, "SMS_score_impute_std"] = impute_std
        df.loc[missing_idx, "softpower_impute_confidence"] = confidence
        return df

    def save_imputation_report(self, df, out_path):
        if "SMS_score_imputed_flag" not in df.columns: return
        cols = [c for c in ["school_id", "School_Name", "school_type", "SMS_score",
                             "l3_verification", "softpower_score", "SMS_score_imputed_flag",
                             "SMS_score_impute_std", "softpower_impute_confidence",
                             "D3_pressure", "risk_cls", "x", "y"] if c in df.columns]
        rpt = df[cols].copy()
        rpt["data_source"] = np.where(df["SMS_score_imputed_flag"] == 1, "KNN插补", "原始数据")
        rpt.to_csv(out_path, index=False, encoding="utf-8-sig")


class DataLoader:
    def __init__(self, inp, cfg):
        self.inp = inp
        self.cfg = cfg

    def load_schools(self):
        df44 = safe_read_csv(self.inp["school_profile_csv"])
        df45 = safe_read_csv(self.inp["school_pressure_csv"])
        _log(f"4.4读入：{len(df44)}行  4.5读入：{len(df45)}行")
        df45 = df45.rename(columns={"School_Name": "_School_Name_45"})
        cols45 = [c for c in ["school_id", "D3_raw", "D3_pred_raw", "D3_residual",
                               "risk_cls", "risk_zh"] if c in df45.columns]
        school = df44.merge(df45[cols45], on="school_id", how="left", suffixes=("", "_45"))
        if "risk_cls_45" in school.columns:
            school["risk_cls"] = pd.to_numeric(
                school["risk_cls_45"], errors="coerce").fillna(
                pd.to_numeric(school.get("risk_cls", 1), errors="coerce").fillna(1)
            ).clip(1, 5).astype(int)
            school = school.drop(columns=["risk_cls_45"], errors="ignore")
        else:
            school["risk_cls"] = get_risk_num(school)
        school["school_type"] = school["Level"].apply(level_to_type)
        school = school[school["school_type"].isin(["小学", "中学"])].copy().reset_index(drop=True)
        for col, fill in [("D3_pressure", 1.0), ("ECFI", 0.5), ("priority_score", 0.5)]:
            school[col] = pd.to_numeric(school[col], errors="coerce").fillna(fill).clip(
                lower=0.01 if col == "D3_pressure" else 0)
        if "D3_pred_raw" in school.columns:
            school["D3_pred_raw"] = pd.to_numeric(
                school["D3_pred_raw"], errors="coerce").fillna(school["D3_pressure"])
        else:
            school["D3_pred_raw"] = school["D3_pressure"]
        school["student_count"] = pd.to_numeric(
            school["student_count"], errors="coerce").fillna(0).clip(lower=0)
        school["risk_zh"] = school["risk_cls"].map(RISK_ZH)
        for c in [f"pct_C{i}" for i in range(1, 7)] + ["poi_diversity", "dist_to_river", "dist_to_center"]:
            if c in school.columns:
                school[c] = pd.to_numeric(school[c], errors="coerce").fillna(0).clip(lower=0)
        school["near_river"] = (
            school.get("dist_to_river", pd.Series([999] * len(school))) < 100).astype(int)
        for col, val in [
            ("SMS_score", np.nan), ("l3_verification", np.nan),
            ("softpower_score", np.nan), ("dominant_type", "无特色"),
            ("school_type_label", "无特色均衡发展型"), ("mismatch_flag", 0),
            ("eligible_tag_count", 0), ("active_tags_eligible", ""),
            ("dominant_type_family", "无特色"), ("is_stem", 0),
            ("softpower_data_available", False)
        ]:
            school[col] = val
        lonlat = school["geometry_wkt"].apply(parse_point_wkt)
        school["lon"] = lonlat.apply(lambda v: v[0])
        school["lat"] = lonlat.apply(lambda v: v[1])
        school = school[school["lon"].notna() & school["lat"].notna()].copy().reset_index(drop=True)
        gdf = gpd.GeoDataFrame(
            school,
            geometry=gpd.points_from_xy(school["lon"], school["lat"]),
            crs=self.cfg["CRS_GEO"]
        ).to_crs(self.cfg["CRS_PROJECT"])
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y
        _log(f"全量学校：{len(gdf)}所 高风险：{(gdf['risk_cls'] >= 4).sum()}所")
        return gdf.reset_index(drop=True)

    def load_softpower(self, school_gdf):
        path = self.inp.get("school_softpower_csv", "")
        if not path or not Path(path).exists():
            school_gdf["softpower_data_available"] = False
        else:
            soft = safe_read_csv(path)
            soft.columns = [c.strip().lstrip('\ufeff').strip() for c in soft.columns]
            col_map = {}
            for target, cands in [
                ("SMS_score", ["SMS_score", "sms_score", "SMS得分", "软实力得分",
                               "softpower_score", "综合软实力得分", "综合得分"]),
                ("dominant_type", ["dominant_type", "特色类型", "主导类型"]),
                ("school_type_label", ["school_type_label", "综合类型标签", "类型标签"]),
                ("mismatch_flag", ["mismatch_flag", "类型不匹配", "mismatch"]),
                ("l3_verification", ["l3_verification", "l3_verification_rate",
                                     "news_score", "l3得分", "第三方验证"]),
                ("eligible_tag_count", ["eligible_tag_count", "tag_count", "标签数量"]),
                ("active_tags_eligible", ["active_tags_eligible", "active_tags", "活跃标签"]),
            ]:
                for c in cands:
                    if c in soft.columns:
                        col_map[c] = target
                        break
            soft = soft.rename(columns=col_map)
            if "school_id" in soft.columns:
                for col in ["SMS_score", "l3_verification"]:
                    if col in soft.columns:
                        soft[col] = pd.to_numeric(soft[col], errors="coerce")
                for col in ["eligible_tag_count", "mismatch_flag"]:
                    if col in soft.columns:
                        soft[col] = pd.to_numeric(soft[col], errors="coerce").fillna(0)
                for col in ["dominant_type", "school_type_label", "active_tags_eligible"]:
                    if col in soft.columns:
                        soft[col] = soft[col].fillna("").astype(str)
                soft["_has_sms"] = soft["SMS_score"].notna() if "SMS_score" in soft.columns else False
                keep = ["school_id", "_has_sms"] + [
                    c for c in ["SMS_score", "dominant_type", "school_type_label",
                                "mismatch_flag", "l3_verification", "eligible_tag_count",
                                "active_tags_eligible"] if c in soft.columns]
                drop = [c for c in keep if c not in ("school_id", "_has_sms") and c in school_gdf.columns]
                school_gdf = school_gdf.drop(columns=drop, errors="ignore")
                school_gdf = school_gdf.merge(soft[keep], on="school_id", how="left")
                school_gdf["softpower_data_available"] = school_gdf["_has_sms"].fillna(False).astype(bool)
                school_gdf = school_gdf.drop(columns=["_has_sms"], errors="ignore")
                if "l3_verification" not in school_gdf.columns:
                    school_gdf["l3_verification"] = np.nan
            else:
                school_gdf["softpower_data_available"] = False

        imputer = SoftpowerImputer(
            k=self.cfg["SOFTPOWER_IMPUTE_K"],
            shrinkage_alpha=self.cfg["SOFTPOWER_SHRINKAGE_ALPHA"],
            min_neighbors=self.cfg["SOFTPOWER_IMPUTE_MIN_NEIGHBORS"]
        )
        school_gdf = imputer.fit_transform(school_gdf)
        for col, default in [
            ("eligible_tag_count", 0), ("mismatch_flag", 0),
            ("dominant_type", "无特色"), ("school_type_label", "无特色均衡发展型"),
            ("active_tags_eligible", "")
        ]:
            if col not in school_gdf.columns:
                school_gdf[col] = default
            elif school_gdf[col].dtype == object:
                school_gdf[col] = school_gdf[col].fillna(default)
            else:
                school_gdf[col] = pd.to_numeric(school_gdf[col], errors="coerce").fillna(default)
        sms = pd.to_numeric(school_gdf["SMS_score"], errors="coerce").fillna(0.0)
        l3 = pd.to_numeric(school_gdf["l3_verification"], errors="coerce").fillna(0.0)
        raw = (self.cfg["SMS_WEIGHT"] * sms + self.cfg["L3_WEIGHT"] * l3).clip(0, 1)
        if "softpower_impute_confidence" in school_gdf.columns and "SMS_score_imputed_flag" in school_gdf.columns:
            imp = school_gdf["SMS_score_imputed_flag"].astype(bool)
            conf = school_gdf["softpower_impute_confidence"].fillna(0.5)
            gm = float(raw[~imp].mean()) if (~imp).sum() > 0 else 0.5
            raw = raw.copy()
            raw[imp] = conf[imp] * raw[imp] + (1 - conf[imp]) * gm
        school_gdf["SMS_score"] = sms.values
        school_gdf["l3_verification"] = l3.values
        school_gdf["softpower_score"] = minmax_norm(raw.values)
        school_gdf["dominant_type_family"] = school_gdf["dominant_type"].apply(
            lambda x: parse_dominant_type_family(x)[0])
        school_gdf["is_stem"] = (school_gdf["dominant_type_family"] == "T1").astype(int)
        school_gdf["eligible_tag_count"] = pd.to_numeric(
            school_gdf.get("eligible_tag_count", 0), errors="coerce").fillna(0).clip(lower=0).astype(int)
        mx = school_gdf["eligible_tag_count"].max()
        school_gdf["tag_richness"] = school_gdf["eligible_tag_count"] / mx if mx > 0 else 0.0
        imputer.save_imputation_report(
            school_gdf,
            Path(self.inp["output_dir"]) / "A_softpower_imputation_report.csv"
        )
        return school_gdf


class CapacityEstimator:
    def __init__(self, cfg):
        self.cfg = cfg

    def estimate(self, sc):
        sc = sc.copy()
        d3 = sc["D3_pressure"].values.astype(float)
        cnt = sc["student_count"].values.astype(float)
        cap_raw = np.where(d3 >= 1.0, cnt,
                           np.where(cnt > 0, cnt / d3, self.cfg["CAPACITY_MIN"]))
        cap_upper = np.maximum(cnt * self.cfg["CAPACITY_MAX_RATIO"], self.cfg["CAPACITY_MIN"])
        if "SMS_score" in sc.columns and "mismatch_flag" in sc.columns:
            sms = sc["SMS_score"].values.astype(float)
            mis = sc["mismatch_flag"].values.astype(int)
            imp = sc["SMS_score_imputed_flag"].values.astype(int) \
                if "SMS_score_imputed_flag" in sc.columns else np.zeros(len(sc), int)
            conf = sc["softpower_impute_confidence"].values \
                if "softpower_impute_confidence" in sc.columns else np.ones(len(sc))
            relax = (sms > self.cfg["SMS_HIGH_THRESHOLD"]) & (mis == 0) & \
                    ((imp == 0) | (conf > 0.6))
            cap_upper = np.where(relax, cap_upper * 1.1, cap_upper)
        cap_clipped = np.clip(cap_raw, self.cfg["CAPACITY_MIN"], cap_upper)
        r = self.cfg["CAPACITY_ROUND"]
        cap_final = np.maximum(
            (np.ceil(cap_clipped / r) * r).astype(int), self.cfg["CAPACITY_MIN"])
        sc["current_capacity"] = cap_final.astype(float)
        sc["utilization_rate"] = np.where(
            sc["current_capacity"] > 0, cnt / sc["current_capacity"], 0.0).clip(0, 2)
        sc["available_seats"] = np.maximum(sc["current_capacity"] - cnt, 0).astype(float)
        sc["max_expand_seats"] = (sc["current_capacity"] * self.cfg["MAX_EXPAND_RATIO"]).astype(int)
        age_share = np.where(sc["school_type"] == "小学",
                             self.cfg["PRIMARY_AGE_SHARE"], self.cfg["MIDDLE_AGE_SHARE"])
        sc["age_share"] = age_share
        sc["age_pop_capacity"] = (cap_final * age_share * self.cfg["ENROLLMENT_RATE"]).astype(float)
        sc["age_pop_available"] = np.maximum(
            sc["age_pop_capacity"] - cnt * age_share * self.cfg["ENROLLMENT_RATE"], 0)
        _log(
            f"容量反推完成: 范围[{int(cap_final.min())}, {int(cap_final.max())}], "
            f"均值{cap_final.mean():.0f}, 超载(D3≥1): {(d3 >= 1.0).sum()}所")
        return sc

    def save_capacity_check(self, sc, out_path):
        cols = [c for c in ["school_id", "School_Name", "school_type",
                             "student_count", "D3_pressure", "current_capacity",
                             "age_pop_capacity", "age_pop_available", "utilization_rate",
                             "risk_cls"] if c in sc.columns]
        sc[cols].to_csv(out_path, index=False, encoding="utf-8-sig")


class DemandGridBuilder:
    def __init__(self, school_gdf, cfg, inp):
        self.schools = school_gdf.reset_index(drop=True)
        self.cfg = cfg
        self.inp = inp

    def build(self):
        pts = gpd.GeoSeries(
            [Point(r.x, r.y) for _, r in self.schools.iterrows()],
            crs=self.cfg["CRS_PROJECT"])
        area = pts.unary_union.convex_hull.buffer(2500)
        gs = self.cfg["GRID_SIZE"]
        b = area.bounds
        cells = []
        for x in np.arange(b[0], b[2], gs):
            for y in np.arange(b[1], b[3], gs):
                cell = box(x, y, x + gs, y + gs)
                if area.intersects(cell):
                    cells.append({"geometry": cell, "cx": x + gs / 2, "cy": y + gs / 2})
        grid = gpd.GeoDataFrame(cells, crs=self.cfg["CRS_PROJECT"])

        sc_xy = self.schools[["x", "y"]].values
        gr_xy = grid[["cx", "cy"]].values
        cnt = self.schools["student_count"].values.astype(float)
        dist = np.maximum(cdist(gr_xy, sc_xy), 1.0)
        w = 1.0 / dist ** 2
        pop = (w / w.sum(axis=0)[np.newaxis, :] * cnt[np.newaxis, :]).sum(axis=1)
        grid["raw_pop"] = pop
        grid["school_age_pop"] = pop * self.cfg["SCHOOL_AGE_RATIO"]

        rp = self.inp.get("worldpop_raster", "")
        rp_exists_as_file = bool(rp) and Path(rp).exists()
        rp_exists_as_raster = False  # 纯开源模式：GDB 栅格数据集输入不支持，WorldPop 请提供 tif 文件路径

        if rp_exists_as_file or rp_exists_as_raster:
            try:
                import rasterio
                from rasterio.transform import rowcol

                actual_path = rp
                tmp_tif = None

                if rp_exists_as_raster:
                    import tempfile
                    tmp_tif = os.path.join(tempfile.gettempdir(), "worldpop_tmp.tif")
                    raise RuntimeError("GDB 栅格数据集输入在纯开源模式下不支持：请提供 tif 文件路径（WorldPop_250m_EPSG4526.tif）")
                    actual_path = tmp_tif
                    _log("已将栅格数据集导出为临时tif")

                with rasterio.open(actual_path) as src:
                    arr = src.read(1)
                    tf = src.transform
                    nd = src.nodata
                    crs_r = src.crs

                tmp = (grid.copy()
                       .set_geometry(gpd.points_from_xy(grid["cx"], grid["cy"]))
                       .set_crs(self.cfg["CRS_PROJECT"])
                       .to_crs(crs_r))
                pops = []
                for geom in tmp.geometry:
                    r, c = rowcol(tf, geom.x, geom.y)
                    if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                        v = float(arr[r, c])
                        pops.append(max(v, 0) if nd is None or abs(v - nd) > 1 else 0.0)
                    else:
                        pops.append(0.0)
                grid["raw_pop"] = pops
                grid["school_age_pop"] = np.array(pops) * self.cfg["SCHOOL_AGE_RATIO"]
                _log("WorldPop替换人口完成")

                if tmp_tif and os.path.exists(tmp_tif):
                    try:
                        os.remove(tmp_tif)
                    except Exception:
                        pass

            except Exception as e:
                _logw(f"WorldPop失败({e})")

        grid = grid[grid["school_age_pop"] > 0.05].copy().reset_index(drop=True)
        grid["grid_id"] = np.arange(len(grid))
        grid["x"], grid["y"] = grid["cx"], grid["cy"]

        sc_xy = self.schools[["x", "y"]].values
        gr_xy = grid[["cx", "cy"]].values
        dist = np.maximum(cdist(gr_xy, sc_xy), 1.0)
        w_n = (1 / dist ** 2) / (1 / dist ** 2).sum(axis=1, keepdims=True)
        grid["pressure_pred"] = (
            w_n * self.schools["D3_pressure"].values[np.newaxis, :]).sum(axis=1)
        nearest = np.argmin(dist, axis=1)
        grid["risk_cls"] = self.schools["risk_cls"].values[nearest]
        grid["risk_zh"] = [RISK_ZH[int(r)] for r in grid["risk_cls"]]
        if "SMS_score" in self.schools.columns:
            grid["sms_score_idw"] = (
                w_n * self.schools["SMS_score"].values[np.newaxis, :]).sum(axis=1)
            grid["priority_weight"] = (
                0.40 * minmax_norm(grid["school_age_pop"].values) +
                0.33 * minmax_norm(grid["pressure_pred"].values) +
                0.20 * ((grid["risk_cls"].values - 1) / 4.0) +
                0.07 * minmax_norm(grid["sms_score_idw"].values)
            ).clip(0.01)
        else:
            grid["priority_weight"] = (
                0.45 * minmax_norm(grid["school_age_pop"].values) +
                0.35 * minmax_norm(grid["pressure_pred"].values) +
                0.20 * ((grid["risk_cls"].values - 1) / 4.0)
            ).clip(0.01)

        _log(f"有效格网：{len(grid)}个 总需求：{grid['school_age_pop'].sum():.0f}")
        return grid.reset_index(drop=True)

    def save_demand_grid(self, grid, out_path):
        cols = [c for c in ["grid_id", "cx", "cy", "x", "y", "school_age_pop",
                             "raw_pop", "pressure_pred", "risk_cls", "risk_zh",
                             "priority_weight"] if c in grid.columns]
        grid[cols].to_csv(out_path, index=False, encoding="utf-8-sig")


class RoadNetworkBuilder:
    def __init__(self, cfg, inp):
        self.cfg, self.inp = cfg, inp

    def build(self):
        roads = gpd.read_file(self.inp["road_network_file"]).to_crs(self.cfg["CRS_PROJECT"])
        roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy().reset_index(drop=True)
        roads["length_m"] = roads.geometry.length
        if "highway" in roads.columns:
            roads["highway"] = roads["highway"].fillna("residential").astype(str)
        else:
            roads["highway"] = "residential"
        walk_speed = self.cfg["ROAD_WALK_SPEED_MPM"]
        bike_speed = self.cfg["ROAD_BIKE_SPEED_MPM"]
        walk_field = self.cfg["ROAD_WALKTIME_FIELD"]
        bike_field = self.cfg["ROAD_BIKETIME_FIELD"]

        if walk_field in roads.columns:
            wt_raw = pd.to_numeric(roads[walk_field], errors="coerce")
            roads["walk_t"] = np.where(
                wt_raw.isna() | (wt_raw <= 0),
                roads["length_m"] / walk_speed, wt_raw)
        else:
            roads["walk_t"] = roads["length_m"] / walk_speed

        if bike_field in roads.columns:
            bt_raw = pd.to_numeric(roads[bike_field], errors="coerce")
            roads["bike_t"] = np.where(
                bt_raw.isna() | (bt_raw <= 0),
                roads["length_m"] / bike_speed, bt_raw)
        else:
            roads["bike_t"] = roads["length_m"] / bike_speed

        no_walk = roads["highway"].isin({"motorway", "motorway_link", "trunk", "trunk_link"})
        roads.loc[no_walk, "walk_t"] = 999999.0

        nd, nxy = {}, []

        def get_node(xy):
            key = (round(float(xy[0]), 1), round(float(xy[1]), 1))
            if key not in nd:
                nd[key] = len(nd)
                nxy.append(key)
            return nd[key]

        G_w, G_b = nx.DiGraph(), nx.DiGraph()
        for _, row in roads.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                c = list(geom.coords)
            elif geom.geom_type == 'MultiLineString':
                geoms_list = list(geom.geoms)
                c = list(geoms_list[0].coords) + list(geoms_list[-1].coords)
            else:
                c = []
            if not c:
                continue
            s, e = c[0], c[-1]
            u, v = get_node(s), get_node(e)
            wt = float(row["walk_t"])
            bt = float(row["bike_t"])
            lm = float(row["length_m"])
            oneway_val = row.get("oneway", "False")
            if oneway_val is None:
                oneway_val = "False"
            oneway = str(oneway_val).strip().lower() in {"1", "true", "yes", "t"}
            G_w.add_edge(u, v, weight=wt, length=lm)
            G_w.add_edge(v, u, weight=wt, length=lm)
            G_b.add_edge(u, v, weight=bt, length=lm)
            if not oneway:
                G_b.add_edge(v, u, weight=bt, length=lm)

        node_xy_arr = np.array(nxy) if nxy else np.zeros((0, 2))
        node_tree = cKDTree(node_xy_arr) if len(node_xy_arr) > 0 else None
        return {"G_walk": G_w, "G_bike": G_b, "node_xy": node_xy_arr, "node_tree": node_tree}


class NetworkOD:
    def __init__(self, sc, gr, net, cfg):
        self.sc, self.gr, self.net, self.cfg = sc, gr, net, cfg

    def compute(self):
        _log("计算路网OD时间矩阵...")
        sc_xy = self.sc[["x", "y"]].values
        gr_xy = self.gr[["x", "y"]].values
        euc = cdist(gr_xy, sc_xy)
        tm = np.full((len(self.gr), len(self.sc)), np.inf)

        if self.net["node_tree"] is None:
            _logw("路网节点为空，使用欧氏距离回退")
            for i in range(len(self.gr)):
                for j in range(len(self.sc)):
                    speed = self.cfg["ROAD_WALK_SPEED_MPM"] \
                        if self.sc.loc[j, "school_type"] == "小学" \
                        else self.cfg["ROAD_BIKE_SPEED_MPM"]
                    max_euc = self.cfg["EUC_FALLBACK_PRIMARY_M"] \
                        if self.sc.loc[j, "school_type"] == "小学" \
                        else self.cfg["EUC_FALLBACK_MIDDLE_M"]
                    if euc[i, j] <= max_euc:
                        tm[i, j] = euc[i, j] * 1.35 / speed
            return tm

        sc_nodes, sc_d = self.net["node_tree"].query(sc_xy, k=1)
        gr_nodes, gr_d = self.net["node_tree"].query(gr_xy, k=1)
        knn = np.argsort(euc, axis=1)[:, :self.cfg["KNN_CANDIDATES"]]
        walk_speed = self.cfg["ROAD_WALK_SPEED_MPM"]
        bike_speed = self.cfg["ROAD_BIKE_SPEED_MPM"]

        for i in range(len(self.gr)):
            cands = knn[i]
            for stype, gkey, max_euc, speed in [
                ("小学", "G_walk", self.cfg["EUC_FALLBACK_PRIMARY_M"], walk_speed),
                ("中学", "G_bike", self.cfg["EUC_FALLBACK_MIDDLE_M"], bike_speed)
            ]:
                targets = [j for j in cands if self.sc.loc[j, "school_type"] == stype]
                if not targets:
                    continue
                G = self.net[gkey]
                src = int(gr_nodes[i])
                snap_ok = (gr_d[i] <= self.cfg["SNAP_MAX_M"]) and (src in G)
                net_lens = {}
                if snap_ok:
                    try:
                        net_lens = nx.single_source_dijkstra_path_length(
                            G, src, cutoff=self.cfg["DIJKSTRA_CUTOFF_MIN"], weight="weight")
                    except Exception:
                        pass
                for j in targets:
                    tgt = int(sc_nodes[j])
                    if tgt in net_lens:
                        tm[i, j] = net_lens[tgt]
                    elif euc[i, j] <= max_euc:
                        tm[i, j] = euc[i, j] * 1.35 / speed
        return tm


class CapacityAllocationOptimizer:
    def __init__(self, sc, gr, tm, cfg):
        self.sc, self.gr, self.tm, self.cfg = sc, gr, tm, cfg

    def solve(self):
        pulp = _get_pulp()  # 延迟导入，未安装时此处抛出清晰的 ImportError
        _log("求解线性规划优化模型...")
        self.sc = self.sc.reset_index(drop=True)
        self.gr = self.gr.reset_index(drop=True)

        dem = self.gr["school_age_pop"].values.astype(float)
        cap_p = self.sc["age_pop_capacity"].values.astype(float) * \
                (self.sc["school_type"] == "小学").values
        cap_m = self.sc["age_pop_capacity"].values.astype(float) * \
                (self.sc["school_type"] == "中学").values
        A_p = compute_2sfca(self.tm, self.sc, self.gr, cap_p,
                             dem * self.cfg["PRIMARY_AGE_SHARE"],
                             self.cfg["SFCA_CATCHMENT_PRIMARY"])
        A_m = compute_2sfca(self.tm, self.sc, self.gr, cap_m,
                             dem * self.cfg["MIDDLE_AGE_SHARE"],
                             self.cfg["SFCA_CATCHMENT_MIDDLE"])
        A_init = self.cfg["PRIMARY_AGE_SHARE"] * A_p + self.cfg["MIDDLE_AGE_SHARE"] * A_m
        low_mask = (A_init <= np.percentile(
            A_init[A_init > 0], self.cfg["EQUITY_LOW_ACCESS_PERCENTILE"])) | (A_init == 0)
        gini_init = weighted_gini(1 - minmax_norm(A_init), dem)

        pairs = [
            (i, j) for i in range(len(self.gr)) for j in range(len(self.sc))
            if self.tm[i, j] <= (self.cfg["PRIMARY_MAX_MIN"]
                                  if self.sc.loc[j, "school_type"] == "小学"
                                  else self.cfg["MIDDLE_MAX_MIN"])
        ]

        if not pairs:
            _loge("无可达对！请检查路网连通性和通勤时间阈值设置。")
            raise RuntimeError("无可达对！")

        _log(f"可达对数量: {len(pairs)} (格网{len(self.gr)} × 学校{len(self.sc)})")

        gwt = self.gr["priority_weight"].values.astype(float)
        equity_wt = np.where(low_mask, gwt * self.cfg["EQUITY_FOCUS_WEIGHT"], gwt)
        age_share = self.sc["age_share"].values.astype(float)
        enroll_r = self.cfg["ENROLLMENT_RATE"]
        total_age_cap = self.sc["age_pop_capacity"].values.astype(float)

        pri_n = minmax_norm(self.sc["priority_score"].values)
        risk_n = (self.sc["risk_cls"].values - 1) / 4.0
        soft_n = minmax_norm(self.sc["softpower_score"].values)
        is_stem = self.sc["is_stem"].values.astype(int) \
            if "is_stem" in self.sc.columns else np.zeros(len(self.sc), int)
        mismatch = self.sc["mismatch_flag"].values.astype(int) \
            if "mismatch_flag" in self.sc.columns else np.zeros(len(self.sc), int)

        if "SMS_score_imputed_flag" in self.sc.columns:
            sms_imp_flag = self.sc["SMS_score_imputed_flag"].fillna(0).values
        else:
            sms_imp_flag = np.zeros(len(self.sc))
        if "softpower_impute_confidence" in self.sc.columns:
            sms_conf = self.sc["softpower_impute_confidence"].fillna(1.0).values
        else:
            sms_conf = np.ones(len(self.sc))
        swf = np.where(sms_imp_flag == 1, sms_conf, 1.0).clip(0.3, 1.0)

        max_z_blocks = self.cfg["MAX_NEW_SEATS_PER_SCHOOL"] // self.cfg["SEAT_BLOCK"]
        model = pulp.LpProblem("EduAlloc_Fair", pulp.LpMinimize)
        x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0) for i, j in pairs}
        u = {i: pulp.LpVariable(f"u_{i}", lowBound=0) for i in range(len(self.gr))}
        z = {j: pulp.LpVariable(
            f"z_{j}", 0,
            min(max(int(self.sc.loc[j, "max_expand_seats"] // self.cfg["SEAT_BLOCK"]), 0),
                max_z_blocks),
            cat="Integer") for j in range(len(self.sc))}
        y = {j: pulp.LpVariable(f"y_{j}", 0, 1, cat="Binary") for j in range(len(self.sc))}

        s2p_t = {j: [] for j in range(len(self.sc))}
        for i, j in pairs:
            s2p_t[j].append(i)
        la_bonus = np.zeros(len(self.sc))
        for j in range(len(self.sc)):
            if s2p_t[j]:
                la_bonus[j] = low_mask[s2p_t[j]].mean()

        max_t = max(self.cfg["PRIMARY_MAX_MIN"], self.cfg["MIDDLE_MAX_MIN"])
        model += (
            pulp.lpSum(self.cfg["W_TRAVEL"] * (self.tm[i, j] / max_t) *
                       equity_wt[i] * x[(i, j)] for i, j in pairs) +
            pulp.lpSum(self.cfg["W_COVER"] * equity_wt[i] * u[i]
                       for i in range(len(self.gr))) -
            pulp.lpSum((self.cfg["W_PRIORITY"] * pri_n[j] +
                        self.cfg["W_RISK"] * risk_n[j] +
                        self.cfg["W_SOFTPOWER"] * soft_n[j] * swf[j] +
                        self.cfg["W_STEM"] * is_stem[j]) *
                       z[j] * self.cfg["SEAT_BLOCK"] for j in range(len(self.sc))) +
            pulp.lpSum(self.cfg["W_MISMATCH"] * mismatch[j] * z[j] * self.cfg["SEAT_BLOCK"]
                       for j in range(len(self.sc))) -
            pulp.lpSum(self.cfg["W_EQUITY"] * la_bonus[j] * z[j] * self.cfg["SEAT_BLOCK"]
                       for j in range(len(self.sc)))
        )

        g2p = {i: [] for i in range(len(self.gr))}
        for i, j in pairs:
            g2p[i].append(j)
        for i in range(len(self.gr)):
            model += (pulp.lpSum(x[(i, jj)] for jj in g2p[i]) + u[i] == dem[i])
        for j in range(len(self.sc)):
            model += (pulp.lpSum(x[(ii, j)] for ii in s2p_t[j]) <=
                      total_age_cap[j] + z[j] * self.cfg["SEAT_BLOCK"] * age_share[j] * enroll_r)
        model += (pulp.lpSum(z[j] * self.cfg["SEAT_BLOCK"]
                             for j in range(len(self.sc))) <= self.cfg["TOTAL_NEW_SEATS"])
        for j in range(len(self.sc)):
            model += (z[j] <= max_z_blocks * y[j])

        hr = [j for j in range(len(self.sc)) if self.sc.loc[j, "risk_cls"] >= 4]
        if hr:
            model += (pulp.lpSum(z[j] * self.cfg["SEAT_BLOCK"] for j in hr) >=
                      self.cfg["HIGH_RISK_MIN_SHARE"] *
                      pulp.lpSum(z[j] * self.cfg["SEAT_BLOCK"] for j in range(len(self.sc))))
        exp_can = [j for j in range(len(self.sc))
                   if self.sc.loc[j, "max_expand_seats"] >= self.cfg["SEAT_BLOCK"]]
        if exp_can:
            model += (pulp.lpSum(y[j] for j in exp_can) >=
                      min(self.cfg["MIN_EXPANDED_SCHOOLS"], len(exp_can)))
        la_scs = [j for j in range(len(self.sc))
                  if la_bonus[j] >= 0.4 and
                  self.sc.loc[j, "max_expand_seats"] >= self.cfg["SEAT_BLOCK"]]
        if la_scs:
            model += (pulp.lpSum(y[j] for j in la_scs) >=
                      min(max(3, len(la_scs) // 3), len(la_scs)))

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120, gapRel=0.02)
        model.solve(solver)
        status = pulp.LpStatus[model.status]

        if status not in ("Optimal", "Feasible"):
            _logw(f"LP求解器状态: {status}")
        else:
            _log(f"LP求解器状态: {status}")

        rs = self.sc.copy()
        rs["expand_blocks"] = [int(round(pulp.value(z[j]) or 0)) for j in range(len(self.sc))]
        rs["new_seats"] = rs["expand_blocks"] * self.cfg["SEAT_BLOCK"]
        rs["cap_after"] = rs["current_capacity"] + rs["new_seats"]
        rs["new_age_cap"] = rs["new_seats"] * age_share * enroll_r
        rs["age_pop_capacity_after"] = rs["age_pop_capacity"] + rs["new_age_cap"]
        rs["low_access_service_ratio"] = la_bonus

        assigned_ap = np.zeros(len(self.sc))
        records = []
        for (i, j), var in x.items():
            val = pulp.value(var) or 0.0
            if val > 1e-6:
                assigned_ap[j] += val
                records.append({
                    "grid_id": int(self.gr.loc[i, "grid_id"]),
                    "school_id": self.sc.loc[j, "school_id"],
                    "school_type": self.sc.loc[j, "school_type"],
                    "assigned_age_pop": round(val, 2),
                    "travel_time_min": round(float(self.tm[i, j]), 3),
                    "grid_risk_cls": int(self.gr.loc[i, "risk_cls"]),
                    "is_low_access": bool(low_mask[i])
                })

        rs["assigned_age_pop"] = assigned_ap
        rs["sd_ratio"] = (rs["age_pop_capacity"] + rs["new_age_cap"]) / \
                         np.maximum(rs["assigned_age_pop"], 1)

        unmet_arr = np.array([pulp.value(u[i]) or 0.0 for i in range(len(self.gr))])
        rg = self.gr.copy()
        rg["unmet"] = unmet_arr

        return {
            "status": status,
            "objective": float(pulp.value(model.objective)),
            "school_result": rs,
            "grid_result": rg,
            "assignment_df": pd.DataFrame(records),
            "total_new_seats": float(rs["new_seats"].sum()),
            "total_unmet": float(unmet_arr.sum()),
            "total_demand": float(dem.sum()),
            "coverage_rate": 1 - unmet_arr.sum() / max(dem.sum(), 1),
            "sfca_init": {
                "A_pri": A_p, "A_mid": A_m, "A_combined": A_init,
                "low_access_mask": low_mask, "gini_before": gini_init
            }
        }


# ============================================================
# Tool_47 界面定义与执行
# ============================================================


# ============================================================
# 5.0 核心业务类
# ============================================================
class BenefitEvaluator:
    def __init__(self, opt, sc_b, sc_a, gr, tm, cfg):
        self.opt = opt
        self.sc_b = sc_b
        self.sc_a = sc_a
        self.gr = gr
        self.tm = tm
        self.cfg = cfg

    def fairness(self):
        dem = self.gr["school_age_pop"].values.astype(float)
        lm = self.opt["sfca_init"]["low_access_mask"]

        Ap_b = compute_2sfca(
            self.tm, self.sc_b, self.gr,
            self.sc_b["age_pop_capacity"].values * (self.sc_b["school_type"] == "小学").values,
            dem * self.cfg["PRIMARY_AGE_SHARE"], 30.0)
        Am_b = compute_2sfca(
            self.tm, self.sc_b, self.gr,
            self.sc_b["age_pop_capacity"].values * (self.sc_b["school_type"] == "中学").values,
            dem * self.cfg["MIDDLE_AGE_SHARE"], 40.0)
        A_b = self.cfg["PRIMARY_AGE_SHARE"] * Ap_b + self.cfg["MIDDLE_AGE_SHARE"] * Am_b

        if "age_pop_capacity_after" not in self.sc_a.columns:
            self.sc_a = self.sc_a.copy()
            self.sc_a["age_pop_capacity_after"] = self.sc_a.get(
                "age_pop_capacity", pd.Series(np.zeros(len(self.sc_a))))

        Ap_a = compute_2sfca(
            self.tm, self.sc_a, self.gr,
            self.sc_a["age_pop_capacity_after"].values * (self.sc_a["school_type"] == "小学").values,
            dem * self.cfg["PRIMARY_AGE_SHARE"], 30.0)
        Am_a = compute_2sfca(
            self.tm, self.sc_a, self.gr,
            self.sc_a["age_pop_capacity_after"].values * (self.sc_a["school_type"] == "中学").values,
            dem * self.cfg["MIDDLE_AGE_SHARE"], 40.0)
        A_a = self.cfg["PRIMARY_AGE_SHARE"] * Ap_a + self.cfg["MIDDLE_AGE_SHARE"] * Am_a

        A_max = max(A_b.max(), A_a.max(), 1e-6)
        Ab_4g = np.where((A_b > 0) | (A_a > 0), 1 - A_b / A_max, 1.0)
        Aa_4g = np.where((A_b > 0) | (A_a > 0), 1 - A_a / A_max, 1.0)
        gb, ga = weighted_gini(Ab_4g, dem), weighted_gini(Aa_4g, dem)

        rng = np.random.default_rng(self.cfg["RANDOM_SEED"])
        sb, sa = [], []
        for _ in range(self.cfg["BOOTSTRAP_N"]):
            idx = rng.choice(len(self.gr), len(self.gr), replace=True)
            sb.append(weighted_gini(Ab_4g[idx], dem[idx]))
            sa.append(weighted_gini(Aa_4g[idx], dem[idx]))
        sb, sa = np.array(sb), np.array(sa)
        t_stat, p_val = stats.ttest_rel(sb, sa)

        la_impr = (A_a[lm] > A_b[lm] * 1.01).sum()
        la_avg_b = A_b[lm].mean() if lm.any() else 0.0
        la_avg_a = A_a[lm].mean() if lm.any() else 0.0
        _, _, tb = compute_simple_access_time(self.tm, self.sc_b, self.cfg)
        _, _, ta = compute_simple_access_time(self.tm, self.sc_a, self.cfg)

        # 时间可达性 Gini（不可达按最大惩罚时间截断）
        max_penalty = (max(self.cfg["PRIMARY_MAX_MIN"],
                           self.cfg["MIDDLE_MAX_MIN"])
                       * self.cfg["INACCESSIBLE_PENALTY_RATIO"])
        tb_clip = np.where(np.isinf(tb), max_penalty, tb)
        ta_clip = np.where(np.isinf(ta), max_penalty, ta)
        gini_time_b = weighted_gini(tb_clip, dem)
        gini_time_a = weighted_gini(ta_clip, dem)

        # 2SFCA 覆盖率（以优化前 P25 为阈值，前后同阈值）
        sfca_thr = (np.percentile(A_b[A_b > 0], 25)
                    if (A_b > 0).any() else 0.0)
        cov_sfca_b = float((A_b >= sfca_thr).mean()) if sfca_thr > 0 else 1.0
        cov_sfca_a = float((A_a >= sfca_thr).mean()) if sfca_thr > 0 else 1.0

        n_improved = int((A_a > A_b * 1.001).sum())
        n_degraded = int((A_a < A_b * 0.999).sum())

        hr_mask = (self.gr["risk_cls"] >= 4).values
        hr_b = float(A_b[hr_mask].mean()) if hr_mask.any() else 0.0
        hr_a = float(A_a[hr_mask].mean()) if hr_mask.any() else 0.0

        return {
            "gini_before": float(gb), "gini_after": float(ga),
            "gini_improvement": float(gb - ga),
            "bs_before_ci": np.percentile(sb, [2.5, 97.5]).tolist(),
            "bs_after_ci": np.percentile(sa, [2.5, 97.5]).tolist(),
            "bs_after_cv": float(sa.std() / (sa.mean() + 1e-9)),
            "paired_t": float(t_stat), "paired_p": float(p_val),
            "n_low_access_grids": int(lm.sum()),
            "n_low_access_improved": int(la_impr),
            "low_access_improve_rate": float(la_impr / max(lm.sum(), 1)),
            "low_access_sfca_before": float(la_avg_b),
            "low_access_sfca_after": float(la_avg_a),
            "low_access_sfca_improvement": float(la_avg_a - la_avg_b),
            "sfca_coverage_before": cov_sfca_b,
            "sfca_coverage_after": cov_sfca_a,
            "sfca_coverage_improvement": float(cov_sfca_a - cov_sfca_b),
            "gini_time_before": float(gini_time_b),
            "gini_time_after": float(gini_time_a),
            "avg_time_before": float(
                tb_clip[~np.isinf(tb)].mean()) if (~np.isinf(tb)).any()
                else float(max_penalty),
            "avg_time_after": float(
                ta_clip[~np.isinf(ta)].mean()) if (~np.isinf(ta)).any()
                else float(max_penalty),
            "n_grids_improved_sfca": n_improved,
            "n_grids_degraded_sfca": n_degraded,
            "hr_sfca_before": hr_b, "hr_sfca_after": hr_a,
            "hr_sfca_improvement": float(hr_a - hr_b),
            "_A_b": A_b, "_A_a": A_a, "_sb": sb, "_sa": sa,
            "_low_access_mask": lm,
            "_A_pri_b": Ap_b, "_A_mid_b": Am_b,
            "_A_pri_a": Ap_a, "_A_mid_a": Am_a,
            "_A_b_4gini": Ab_4g, "_A_a_4gini": Aa_4g,
            "_t_before": tb_clip, "_t_after": ta_clip,
            "_t_combined_b": tb, "_t_combined_a": ta
        }

    def efficiency(self):
        dem = self.gr["school_age_pop"].sum()
        tu = self.opt["grid_result"]["unmet"].sum()
        hr_g = self.gr["risk_cls"] >= 4
        hr_d = self.gr.loc[hr_g, "school_age_pop"].sum()
        lm = self.opt["sfca_init"]["low_access_mask"]
        la_d = self.gr.loc[lm, "school_age_pop"].sum()
        la_u = self.opt["grid_result"].loc[lm, "unmet"].sum()
        return {
            "total_demand": float(dem),
            "total_unmet": float(tu),
            "coverage_rate": float(1 - tu / max(dem, 1)),
            "hr_coverage": float(
                1 - self.opt["grid_result"].loc[hr_g, "unmet"].sum() / max(hr_d, 1)
                if hr_d > 0 else np.nan),
            "expanded_n": int((self.sc_a["new_seats"] > 0).sum()),
            "total_new_seats": float(self.sc_a["new_seats"].sum()),
            "low_access_coverage": float(1 - la_u / max(la_d, 1))
        }

    def robustness(self, A_after):
        dem = self.gr["school_age_pop"].values.astype(float)
        rng = np.random.default_rng(self.cfg["RANDOM_SEED"])
        recs = []
        for _ in range(self.cfg["MONTE_CARLO_N"]):
            ds = dem * rng.uniform(0.95, 1.05, len(dem))
            As = A_after * rng.uniform(0.90, 1.10, len(A_after))
            noise = rng.uniform(0.95, 1.05, len(dem))
            recs.append({
                "coverage": float(
                    1 - (self.opt["grid_result"]["unmet"].values * noise).sum() /
                    max(ds.sum(), 1)),
                "gini": weighted_gini(1 - minmax_norm(As), ds),
                "gap": float(np.sum(ds * (minmax_norm(As) < 0.1)))
            })
        df = pd.DataFrame(recs)

        def _cv(s):
            return float(s.std() / (s.mean() + 1e-9))

        return {
            "cov_mean": float(df["coverage"].mean()),
            "cov_cv": _cv(df["coverage"]),
            "gini_mean": float(df["gini"].mean()),
            "gini_cv": _cv(df["gini"]),
            "gap_mean": float(df["gap"].mean()),
            "gap_cv": _cv(df["gap"]),
            "_raw": df
        }


class Visualizer:
    def __init__(self, opt, fr, er, mc, out_dir):
        self.sc = opt["school_result"]
        self.gr = opt["grid_result"]
        self.opt = opt
        self.fr = fr
        self.er = er
        self.mc = mc
        self.out_dir = out_dir
        self.C_BEFORE = "#2166AC"
        self.C_AFTER  = "#D6604D"
        self.C_IMPR   = "#4DAC26"
        self.C_DEGR   = "#762A83"
        self.CMAP_ACCESS = LinearSegmentedColormap.from_list(
            "acc", ["#D73027", "#FFFFBF", "#1A9850"], N=256)
        self.CMAP_DIV = LinearSegmentedColormap.from_list(
            "div", ["#762A83", "#FFFFBF", "#1B7837"], N=256)

    def _save(self, fig, name):
        fig.savefig(str(Path(self.out_dir) / name),
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def plot_all(self):
        _setup_matplotlib()
        for fn in [self._h1, self._h2, self._h3, self._h4, self._h5, self._h6]:
            try:
                fn()
                _log(f"  图表已生成: {fn.__name__}")
            except Exception as e:
                _logw(f"图表 {fn.__name__} 生成失败: {e}")

    def _h1(self):
        fig = plt.figure(figsize=(18, 5.5))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32,
                               left=0.05, right=0.97, top=0.88, bottom=0.10)
        ax1 = fig.add_subplot(gs[0])
        for rc in range(1, 6):
            mask = self.sc["risk_cls"] == rc
            if mask.any():
                ax1.scatter(self.sc.loc[mask, "x"], self.sc.loc[mask, "y"],
                            c=RISK_COLOR[rc], s=60, edgecolors="white",
                            lw=0.6, label=RISK_ZH[rc])
        ax1.set_title("学校风险等级空间分布", fontweight="bold")
        ax1.legend(loc="lower right")

        ax2 = fig.add_subplot(gs[1])
        has_new = self.sc["new_seats"] > 0
        ax2.scatter(self.sc.loc[~has_new, "x"], self.sc.loc[~has_new, "y"],
                    c="#BBBBBB", s=25, edgecolors="white", label="未扩容")
        if has_new.any():
            sc2 = ax2.scatter(
                self.sc.loc[has_new, "x"], self.sc.loc[has_new, "y"],
                c=self.sc.loc[has_new, "new_seats"], cmap="YlOrRd",
                s=60 + self.sc.loc[has_new, "new_seats"] / 800 * 220,
                edgecolors="#444444")
            plt.colorbar(sc2, ax=ax2).set_label("新增学位数")
        ax2.set_title("扩容方案空间分布", fontweight="bold")
        ax2.legend(loc="lower right")

        ax3 = fig.add_subplot(gs[2])
        sc3 = ax3.scatter(self.gr["x"], self.gr["y"],
                          c=minmax_norm(self.fr["_A_b"]),
                          cmap=self.CMAP_ACCESS, s=4, vmin=0, vmax=1)
        plt.colorbar(sc3, ax=ax3).set_label("2SFCA可达性(归一化)")
        ax3.scatter(self.sc["x"], self.sc["y"],
                    c="#333333", s=22, marker="^", edgecolors="white")
        ax3.set_title("初始2SFCA可达性空间分布", fontweight="bold")
        self._save(fig, "H1_spatial_overview.png")

    def _h2(self):
        """
        修复：洛伦兹曲线对全零可达性的健壮处理；确保曲线单调递增。
        """
        fig = plt.figure(figsize=(18, 5.5))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                               left=0.06, right=0.97, top=0.88, bottom=0.12)

        ax1 = fig.add_subplot(gs[0])
        dem = self.gr["school_age_pop"].values.astype(float)
        A_b = self.fr["_A_b"].copy()
        A_a = self.fr["_A_a"].copy()

        def lorenz(values, weights):
            """
            修复版洛伦兹曲线：
            1. 将负值裁切为0（2SFCA理论上非负，但数值误差可能产生极小负值）
            2. 处理总加权值为0的退化情形
            3. 保证曲线端点严格为(0,0)和(1,1)
            """
            values  = np.asarray(values,  dtype=float).clip(0)   # ★ 修复：裁切负值
            weights = np.asarray(weights, dtype=float).clip(0)
            order  = np.argsort(values)
            vs, ws = values[order], weights[order]
            w_total  = ws.sum()
            vw_total = (vs * ws).sum()
            # 退化情形：所有可达性为0 → 洛伦兹曲线为对角线
            if w_total < 1e-9 or vw_total < 1e-9:
                return np.array([0.0, 1.0]), np.array([0.0, 1.0])
            lx = np.concatenate([[0.0], np.cumsum(ws)  / w_total])
            ly = np.concatenate([[0.0], np.cumsum(vs * ws) / vw_total])
            # ★ 修复：强制端点为(1,1)，消除浮点累积误差
            lx[-1] = 1.0
            ly[-1] = 1.0
            return lx, ly

        lx_b, ly_b = lorenz(A_b, dem)
        lx_a, ly_a = lorenz(A_a, dem)
        ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="完全平等线")
        ax1.plot(lx_b, ly_b, color=self.C_BEFORE, lw=1.8,
                 label=f"优化前 Gini={self.fr['gini_before']:.4f}")
        ax1.plot(lx_a, ly_a, color=self.C_AFTER,  lw=1.8,
                 label=f"优化后 Gini={self.fr['gini_after']:.4f}")
        ax1.fill_between(lx_b, ly_b, [0, 1] if len(lx_b) == 2 else
                         np.interp(lx_b, [0, 1], [0, 1]),
                         alpha=0.08, color=self.C_BEFORE)
        ax1.fill_between(lx_a, ly_a, [0, 1] if len(lx_a) == 2 else
                         np.interp(lx_a, [0, 1], [0, 1]),
                         alpha=0.08, color=self.C_AFTER)
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.set_xlabel("累积人口份额")
        ax1.set_ylabel("累积可达性份额")
        ax1.legend(fontsize=8)
        ax1.set_title("洛伦兹曲线(2SFCA)", fontweight="bold")

        ax2 = fig.add_subplot(gs[1])
        ax2.hist(self.fr["_sb"], bins=30, alpha=0.6,
                 color=self.C_BEFORE, density=True, label="优化前")
        ax2.hist(self.fr["_sa"], bins=30, alpha=0.6,
                 color=self.C_AFTER, density=True, label="优化后")
        ax2.axvline(self.fr["gini_before"], color=self.C_BEFORE,
                    ls="--", lw=1.2, label=f"前均值={self.fr['gini_before']:.4f}")
        ax2.axvline(self.fr["gini_after"],  color=self.C_AFTER,
                    ls="--", lw=1.2, label=f"后均值={self.fr['gini_after']:.4f}")
        ax2.set_title("Bootstrap Gini分布", fontweight="bold")
        ax2.legend(fontsize=7)

        ax3 = fig.add_subplot(gs[2])
        diff = A_a - A_b
        v = max(abs(np.percentile(diff, 5)), abs(np.percentile(diff, 95)), 1e-9)
        sc3 = ax3.scatter(self.gr["x"], self.gr["y"], c=diff,
                          cmap=self.CMAP_DIV, s=5, vmin=-v, vmax=v)
        plt.colorbar(sc3, ax=ax3).set_label("可达性变化量")
        ax3.set_title("2SFCA可达性改善空间分布", fontweight="bold")
        self._save(fig, "H2_fairness_2sfca.png")

    def _h3(self):
        fig = plt.figure(figsize=(18, 5.5))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                               left=0.06, right=0.97, top=0.88, bottom=0.12)
        ax1 = fig.add_subplot(gs[0])
        sd = pd.to_numeric(self.sc.get("sd_ratio", pd.Series([1.0] * len(self.sc))),
                           errors="coerce").fillna(1.0).clip(0, 4)
        ax1.hist(sd, bins=30, color=self.C_BEFORE)
        ax1.axvline(1.0, color="k", ls="--", label="供需平衡线")
        ax1.set_title("供需比分布", fontweight="bold")
        ax1.legend()

        ax2 = fig.add_subplot(gs[1])
        sc2 = ax2.scatter(self.sc["D3_pressure"], self.sc["new_seats"],
                          c=self.sc["risk_cls"], cmap="RdYlGn_r", s=65)
        plt.colorbar(sc2, ax=ax2).set_label("风险等级")
        ax2.axvline(1.0, color="k", ls="--")
        ax2.set_xlabel("D3压力指数")
        ax2.set_ylabel("新增学位数")
        ax2.set_title("压力与扩容规模", fontweight="bold")

        ax3 = fig.add_subplot(gs[2])
        r_labs, covs = [], []
        for rc in range(1, 6):
            mask = self.gr["risk_cls"] == rc
            if mask.sum() > 0:
                r_labs.append(RISK_ZH[rc])
                td = self.gr.loc[mask, "school_age_pop"].sum()
                tu = self.gr.loc[mask, "unmet"].sum()
                covs.append(1 - tu / max(td, 1))
        ax3.bar(r_labs, covs, color=[RISK_COLOR[i + 1] for i in range(len(r_labs))])
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel("覆盖率")
        ax3.set_title("各风险等级格网覆盖率", fontweight="bold")
        self._save(fig, "H3_efficiency.png")

    def _h4(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        mc = self.mc["_raw"]
        for idx, (col, t, c) in enumerate([
            ("coverage", "模拟覆盖率", SCI_COLORS["blue"]),
            ("gini",     "模拟Gini",   SCI_COLORS["orange"]),
            ("gap",      "低可达缺口", SCI_COLORS["red"])
        ]):
            axes[idx].hist(mc[col], bins=30, color=c, alpha=0.7, density=True)
            axes[idx].set_title(t, fontweight="bold")
            axes[idx].axvline(mc[col].mean(), color="k", linestyle="--",
                              label=f"均值={mc[col].mean():.3f}")
            axes[idx].legend(fontsize=8)
        self._save(fig, "H4_robustness.png")

    def _h5(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        if "intervention" in self.sc.columns:
            itype = self.sc["intervention"].fillna("无").value_counts()
        else:
            def classify(row):
                ns = row.get("new_seats", 0)
                if ns == 0:       return "无干预"
                elif ns <= 100:   return "小规模扩容"
                elif ns <= 200:   return "中规模扩容"
                else:             return "大规模扩容"
            itype = self.sc.apply(classify, axis=1).value_counts()
        ax.barh(itype.index[::-1], itype.values[::-1], color=SCI_COLORS["blue"])
        ax.set_xlabel("学校数量")
        ax.set_title("干预策略类型分布", fontweight="bold")
        self._save(fig, "H5_softpower_analysis.png")

    def _h6(self):
        """
        修复：优化前覆盖率应从4.7 meta中读取，而非用优化后unmet重新计算。
        当meta不可用时回退到从grid_result中读取原始unmet（需由4.7写入）。
        同时修复柱状图数值标签超出轴范围的问题。
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ["优化前", "优化后"]
        after_cov = self.er["coverage_rate"]

        # ★ 修复：优化前覆盖率从opt元数据中读取（4.7写入meta时已记录）
        # opt["sfca_init"] 不含覆盖率，但4.7 meta有coverage_rate（优化后）
        # 优化前覆盖率 = 1 - 优化前total_unmet / total_demand
        # 这里使用grid_result中unmet来估算优化前状态：
        # 因为grid_result["unmet"]已是LP优化后的残差，优化前残差不在此处，
        # 所以使用meta中保存的数据（若有）或设为0来表示"无优化基线"。
        # 最稳健的方式：使用 opt 中保存的 total_unmet 和 total_demand
        total_demand = float(self.gr["school_age_pop"].sum())
        # 优化后 unmet（LP残差）
        after_unmet = float(self.opt["grid_result"]["unmet"].sum())
        after_cov_check = 1.0 - after_unmet / max(total_demand, 1)

        # 优化前覆盖率：从opt元数据中获取（4.7在meta中只存了优化后coverage_rate）
        # 这里取 opt 内嵌的原始值，若不存在则用2SFCA低可达比例估算基线
        # ★ 正确做法：4.7不保存优化前覆盖率，故此处用低可达格网比例近似
        lm = self.opt["sfca_init"]["low_access_mask"]
        before_cov = 1.0 - float(lm.sum()) / max(len(lm), 1)  # 低可达比例的补数作为基线参考

        # 若 opt 中存有 before_coverage_rate（由外部传入），优先使用
        if "before_coverage_rate" in self.opt:
            before_cov = float(self.opt["before_coverage_rate"])

        bars1 = axes[0].bar(labels, [before_cov, after_cov],
                            color=[self.C_BEFORE, self.C_AFTER], width=0.4)
        y_max1 = max(before_cov, after_cov) * 1.15
        axes[0].set_ylim(0, min(y_max1, 1.15))
        axes[0].set_ylabel("覆盖率")
        axes[0].set_title("覆盖率优化前后对比", fontweight="bold")
        for bar, val in zip(bars1, [before_cov, after_cov]):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         min(bar.get_height() + 0.01, axes[0].get_ylim()[1] * 0.95),
                         f"{val:.2%}", ha="center", va="bottom", fontsize=10)

        bars2 = axes[1].bar(labels,
                            [self.fr["gini_before"], self.fr["gini_after"]],
                            color=[self.C_BEFORE, self.C_AFTER], width=0.4)
        y_max2 = max(self.fr["gini_before"], self.fr["gini_after"]) * 1.15
        axes[1].set_ylim(0, max(y_max2, 0.05))
        axes[1].set_ylabel("Gini系数")
        axes[1].set_title("公平性Gini优化前后对比", fontweight="bold")
        for bar, val in zip(bars2, [self.fr["gini_before"], self.fr["gini_after"]]):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + axes[1].get_ylim()[1] * 0.01,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=10)

        fig.suptitle("核心指标优化前后对比", fontweight="bold")
        plt.tight_layout()
        self._save(fig, "H6_2sfca_comparison.png")


# ============================================================
# Tool_50 界面定义与执行
# ============================================================

def tool_47_execute(params, messages=None):
        import time as _time
        _t0 = _time.time()

        out_dir = params[5].valueAsText
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # 输入验证
        for i, label in [(0, "学校基础信息表"), (1, "学校压力预测表"), (2, "路网文件")]:
            pth = params[i].valueAsText
            if not pth or not Path(pth).exists():
                _loge(f"缺少必需输入: {label} ({pth})")
                return

        for i, label in [(0, "学校基础信息表"), (1, "学校压力预测表")]:
            try:
                test_df = safe_read_csv(params[i].valueAsText)
                _log(f"{label}: {len(test_df)} 行, {list(test_df.columns)[:8]}...")
            except Exception as e:
                _loge(f"无法读取{label}: {e}")
                return

        try:
            test_roads = gpd.read_file(params[2].valueAsText)
            _log(f"路网: {len(test_roads)} 段, CRS={test_roads.crs}")
        except Exception as e:
            _loge(f"无法读取路网文件: {e}")
            return

        inp = {
            "school_profile_csv":   params[0].valueAsText,
            "school_pressure_csv":  params[1].valueAsText,
            "road_network_file":    params[2].valueAsText,
            "school_softpower_csv": params[3].valueAsText or "",
            "worldpop_raster":      params[4].valueAsText or "",
            "output_dir":           out_dir
        }
        cfg = {
            "CRS_GEO":            params[6].valueAsText or "EPSG:4326",
            "CRS_PROJECT":        params[7].valueAsText or "EPSG:4526",
            "SCHOOL_AGE_RATIO":   float(params[8].value or 0.1174),
            "GRID_SIZE":          int(params[9].value or 250),
            "TOTAL_NEW_SEATS":    int(params[10].value or 5000),
            "MAX_NEW_SEATS_PER_SCHOOL": int(params[11].value or 300),
            "MIN_EXPANDED_SCHOOLS":     int(params[12].value or 15),
            "PRIMARY_MAX_MIN":    float(params[13].value or 30.0),
            "MIDDLE_MAX_MIN":     float(params[14].value or 40.0),
            "RANDOM_SEED":        int(params[15].value or 42),
            "CAPACITY_MIN": 200, "CAPACITY_MAX_RATIO": 3.5,
            "CAPACITY_ROUND": 50, "SEAT_BLOCK": 50,
            "MAX_EXPAND_RATIO": 0.3,
            "PRIMARY_AGE_SHARE": 0.667, "MIDDLE_AGE_SHARE": 0.333,
            "ENROLLMENT_RATE": 0.95,
            "ROAD_WALK_SPEED_MPM": 5000.0 / 60.0,
            "ROAD_BIKE_SPEED_MPM": 240.0,
            "ROAD_WALKTIME_FIELD": "WalkTime",
            "ROAD_BIKETIME_FIELD": "Bike_Time",
            "KNN_CANDIDATES": 15,
            "EUC_FALLBACK_PRIMARY_M": 2000, "EUC_FALLBACK_MIDDLE_M": 4000,
            "SNAP_MAX_M": 500, "DIJKSTRA_CUTOFF_MIN": 45,
            "W_TRAVEL": 5.0, "W_COVER": 500.0, "W_PRIORITY": 25.0,
            "W_RISK": 18.0, "W_SOFTPOWER": 3.0, "W_STEM": 4.0,
            "W_MISMATCH": 5.0, "W_EQUITY": 200.0,
            "HIGH_RISK_MIN_SHARE": 0.50,
            "EQUITY_LOW_ACCESS_PERCENTILE": 25, "EQUITY_FOCUS_WEIGHT": 3.0,
            "SOFTPOWER_IMPUTE_K": 5, "SOFTPOWER_SHRINKAGE_ALPHA": 0.15,
            "SOFTPOWER_IMPUTE_MIN_NEIGHBORS": 3,
            "SMS_WEIGHT": 0.7, "L3_WEIGHT": 0.3, "SMS_HIGH_THRESHOLD": 0.5,
            "SFCA_CATCHMENT_PRIMARY": 30.0, "SFCA_CATCHMENT_MIDDLE": 40.0,
            "SFCA_DECAY_BETA": 0.15
        }
        np.random.seed(cfg["RANDOM_SEED"])

        try:
            # ── 阶段1：数据加载 ──────────────────────────────────────
            print_47_progress_banner("阶段1/5  数据加载与软实力插补")
            loader = DataLoader(inp, cfg)
            schools = loader.load_schools()
            schools = loader.load_softpower(schools)

            # ── 阶段2：容量反推 ──────────────────────────────────────
            print_47_progress_banner("阶段2/5  容量反推",
                f"学校总数: {len(schools)}  高风险: {(schools['risk_cls'] >= 4).sum()}")
            cap_estimator = CapacityEstimator(cfg)
            schools = cap_estimator.estimate(schools)
            cap_estimator.save_capacity_check(schools, Path(out_dir) / "B_capacity_check.csv")

            # ── 阶段3：需求格网 ──────────────────────────────────────
            print_47_progress_banner("阶段3/5  需求格网构建")
            grid_builder = DemandGridBuilder(schools, cfg, inp)
            grids = grid_builder.build()
            grid_builder.save_demand_grid(grids, Path(out_dir) / "C_demand_grid.csv")

            # ── 阶段4：路网OD ────────────────────────────────────────
            print_47_progress_banner("阶段4/5  路网OD矩阵计算",
                f"格网: {len(grids)}  学校: {len(schools)}")
            net = RoadNetworkBuilder(cfg, inp).build()
            tm = NetworkOD(schools, grids, net, cfg).compute()

            # ── 阶段5：LP优化 ────────────────────────────────────────
            print_47_progress_banner("阶段5/5  LP优化求解")
            sc_before = schools.copy()
            sc_before["age_pop_capacity_after"] = sc_before["age_pop_capacity"]
            opt = CapacityAllocationOptimizer(schools, grids, tm, cfg).solve()

        except Exception as e:
            _loge(f"4.7 执行错误: {e}")
            import traceback
            _loge(traceback.format_exc())
            return

        try:
            sc_after = opt["school_result"]
            sc_b_cols = [c for c in sc_before.columns if c != "geometry"]
            sc_a_cols = [c for c in sc_after.columns if c != "geometry"]
            gr_cols = [c for c in opt["grid_result"].columns if c != "geometry"]

            sc_before[sc_b_cols].to_csv(params[16].valueAsText, index=False, encoding="utf-8-sig")
            sc_after[sc_a_cols].to_csv(params[17].valueAsText, index=False, encoding="utf-8-sig")
            opt["grid_result"][gr_cols].to_csv(params[18].valueAsText, index=False, encoding="utf-8-sig")
            opt["assignment_df"].to_csv(params[19].valueAsText, index=False, encoding="utf-8-sig")

            meta = {
                "status": opt["status"],
                "objective": float(opt["objective"]),
                "total_new_seats": float(opt["total_new_seats"]),
                "total_unmet": float(opt["total_unmet"]),
                "total_demand": float(opt["total_demand"]),
                "coverage_rate": float(opt["coverage_rate"]),
                "sfca_gini_before": float(opt["sfca_init"]["gini_before"])
            }
            with open(params[20].valueAsText, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            np.save(params[21].valueAsText, opt["sfca_init"]["A_pri"])
            np.save(params[22].valueAsText, opt["sfca_init"]["A_mid"])
            np.save(params[23].valueAsText, opt["sfca_init"]["A_combined"])
            np.save(params[24].valueAsText, opt["sfca_init"]["low_access_mask"].astype(np.uint8))
            np.save(params[25].valueAsText, tm)

        except Exception as e:
            _loge(f"写入输出文件错误: {e}")
            import traceback
            _loge(traceback.format_exc())
            return

        # ── 4.7 完成摘要 ──────────────────────────────────────────────
        t_cost = _time.time() - _t0
        SEP = "=" * 56
        _log(SEP)
        _log("  \U0001F4CB  4.7 优化配置完成摘要")
        _log(SEP)
        _log(f"  总耗时：          {t_cost:>8.1f} 秒")
        _log(f"  LP状态：          {opt['status']:>10s}")
        _log(f"  覆盖率：          {opt['coverage_rate']:>10.2%}")
        _log(f"  扩容学校数：      {int((sc_after['new_seats'] > 0).sum()):>10d} 所")
        _log(f"  新增学位总数：    {int(opt['total_new_seats']):>10d}")
        _log(f"  优化前Gini：      {opt['sfca_init']['gini_before']:>10.4f}")
        _log(f"  输出目录：        {out_dir}")
        _log(SEP)

def tool_50_execute(params, messages=None):
        _setup_matplotlib()
        out_dir = params[10].valueAsText
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        cfg = {
            "BOOTSTRAP_N":   int(params[11].value or 1000),
            "MONTE_CARLO_N": int(params[12].value or 500),
            "RANDOM_SEED":   int(params[13].value or 42),
            "PRIMARY_MAX_MIN":  float(params[14].value or 30.0),
            "MIDDLE_MAX_MIN":   float(params[15].value or 40.0),
            "PRIMARY_AGE_SHARE": 0.667, "MIDDLE_AGE_SHARE": 0.333,
            "INACCESSIBLE_PENALTY_RATIO": 2.0,
            "SFCA_DECAY_BETA": 0.15,
            "SFCA_CATCHMENT_PRIMARY": 30.0, "SFCA_CATCHMENT_MIDDLE": 40.0
        }
        np.random.seed(cfg["RANDOM_SEED"])

        try:
            sc_b = safe_read_csv(params[0].valueAsText)
            sc_a = safe_read_csv(params[1].valueAsText)
            gr   = safe_read_csv(params[2].valueAsText)
            asn  = safe_read_csv(params[3].valueAsText)
            with open(params[4].valueAsText, "r", encoding="utf-8") as f:
                meta = json.load(f)
            tm = np.load(params[9].valueAsText)
            low_access_mask = np.load(params[8].valueAsText).astype(bool)
        except Exception as e:
            _loge(f"读取输入文件失败: {e}")
            import traceback
            _loge(traceback.format_exc())
            return

        # 确保数值列类型正确
        for col in ["school_age_pop", "unmet", "risk_cls"]:
            if col in gr.columns:
                gr[col] = pd.to_numeric(gr[col], errors="coerce").fillna(0)
        for col in ["age_pop_capacity", "new_seats", "D3_pressure", "risk_cls"]:
            if col in sc_b.columns:
                sc_b[col] = pd.to_numeric(sc_b[col], errors="coerce").fillna(0)
            if col in sc_a.columns:
                sc_a[col] = pd.to_numeric(sc_a[col], errors="coerce").fillna(0)
        if "age_pop_capacity_after" not in sc_a.columns:
            sc_a["age_pop_capacity_after"] = sc_a.get(
                "age_pop_capacity", pd.Series(np.zeros(len(sc_a))))
        if "new_seats" not in sc_a.columns:
            sc_a["new_seats"] = 0

        opt = {
            "school_result": sc_a, "grid_result": gr, "assignment_df": asn,
            "total_new_seats": meta.get("total_new_seats", 0),
            "total_unmet":     meta.get("total_unmet", 0),
            "total_demand":    meta.get("total_demand", 0),
            # ★ 将meta中的优化前覆盖率（若有）传入Visualizer
            "before_coverage_rate": 1.0 - meta.get("total_unmet", 0) /
                                    max(meta.get("total_demand", 1), 1),
            "sfca_init": {
                "low_access_mask": low_access_mask,
                "gini_before": meta.get("sfca_gini_before", 0.0)
            }
        }

        _log("开始效益评估...")
        try:
            ev = BenefitEvaluator(opt, sc_b, sc_a, gr, tm, cfg)
            fr = ev.fairness()
            opt["_A_after"] = fr["_A_a"]
            er = ev.efficiency()
            mc = ev.robustness(fr["_A_a"])
        except Exception as e:
            _loge(f"效益评估计算失败: {e}")
            import traceback
            _loge(traceback.format_exc())
            return

        # ── 核心指标汇总面板（控制台可视化）────────────────────────
        try:
            print_summary_banner(fr, er, mc, opt)
        except Exception as e:
            _logw(f"指标汇总面板输出失败: {e}")

        _log("绘制图表...")
        try:
            Visualizer(opt, fr, er, mc, out_dir).plot_all()
        except Exception as e:
            _logw(f"图表绘制错误: {e}")

        try:
            sc_a.to_csv(params[16].valueAsText, index=False, encoding="utf-8-sig")
            gr.to_csv(params[17].valueAsText,   index=False, encoding="utf-8-sig")
            asn.to_csv(params[18].valueAsText,  index=False, encoding="utf-8-sig")
            mc["_raw"].to_csv(params[19].valueAsText, index=False, encoding="utf-8-sig")

            report = {
                "fairness_metric": "2SFCA可达性Gini（两步移动搜索法，高斯距离衰减）",
                "method_description": {
                    "2sfca": "两步移动搜索法，高斯衰减 decay = exp(-0.15*(t/t_max)^2)",
                    "gini_definition": "Gini(1 - 归一化2SFCA得分)，越低越公平",
                    "equity_optimization": "低可达性格网（P25以下）获得3倍权重，附近学校获得扩容激励"
                },
                "optimization": {
                    "status": meta.get("status", "Optimal"),
                    "total_new_seats": er["total_new_seats"],
                    "total_unmet": er["total_unmet"],
                    "coverage_rate":   er["coverage_rate"]
                },
                "fairness":    {k: v for k, v in fr.items() if not k.startswith("_")},
                "efficiency":  {k: v for k, v in er.items() if not k.startswith("_")},
                "robustness":  {k: v for k, v in mc.items() if not k.startswith("_")}
            }

            def _json_default(o):
                return o.item() if hasattr(o, "item") else str(o)

            with open(params[20].valueAsText, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2,
                          default=_json_default)

        except Exception as e:
            _loge(f"写入输出文件失败: {e}")
            import traceback
            _loge(traceback.format_exc())
            return

        _log(f"5.0运行成功，文件已输出至：{out_dir}")


# ══════════════════════════════════════════════════════════════════
# CLI 接线（argparse → _Param 适配 → 执行体，默认值与原始实现一致）
# ══════════════════════════════════════════════════════════════════

def main47():
    ap = argparse.ArgumentParser(description="4.7 教育资源优化配置（纯 Python）")
    ap.add_argument("--school-profile-csv", required=True,
                    help="学校基础信息表（4.4 衍生：含 geometry_wkt/ECFI/priority_score/student_count）")
    ap.add_argument("--school-pressure-csv", required=True,
                    help="学校压力预测表（4.5 产出：school_id/D3_raw/D3_pred_raw/risk_cls）")
    ap.add_argument("--road-network", required=True,
                    help="预处理路网（4.3 产出，含 WalkTime/Bike_Time）")
    ap.add_argument("--school-softpower-csv", default=None,
                    help="SMS 软实力表（4.6 soft_match_results.csv，可选）")
    ap.add_argument("--worldpop-raster", default=None,
                    help="WorldPop tif（可选；注意须为主基准带号版本）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--crs-geo", default="EPSG:4326")
    ap.add_argument("--crs-project", default="EPSG:4526")
    ap.add_argument("--school-age-ratio", type=float, default=0.1174)
    ap.add_argument("--grid-size", type=int, default=250)
    ap.add_argument("--total-new-seats", type=int, default=5000)
    ap.add_argument("--max-new-seats-per-school", type=int, default=300)
    ap.add_argument("--min-expanded-schools", type=int, default=15)
    ap.add_argument("--primary-max-min", type=float, default=30.0)
    ap.add_argument("--middle-max-min", type=float, default=40.0)
    ap.add_argument("--random-seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = [
        _Param(a.school_profile_csv), _Param(a.school_pressure_csv),
        _Param(a.road_network), _Param(a.school_softpower_csv),
        _Param(a.worldpop_raster), _Param(str(out)),
        _Param(a.crs_geo), _Param(a.crs_project), _Param(a.school_age_ratio),
        _Param(a.grid_size), _Param(a.total_new_seats),
        _Param(a.max_new_seats_per_school), _Param(a.min_expanded_schools),
        _Param(a.primary_max_min), _Param(a.middle_max_min),
        _Param(a.random_seed),
    ]
    for name in ["F_school_before.csv", "F_school_after.csv", "F_grid_result.csv",
                 "F_assignment.csv", "F_opt_meta.json", "F_sfca_A_pri_init.npy",
                 "F_sfca_A_mid_init.npy", "F_sfca_A_combined_init.npy",
                 "F_low_access_mask.npy", "E_time_matrix.npy"]:
        params.append(_Param(str(out / name)))
    tool_47_execute(params, _MsgShim())


def main50():
    ap = argparse.ArgumentParser(description="5.0 效益评估（纯 Python）")
    ap.add_argument("--school-before", required=True, help="F_school_before.csv")
    ap.add_argument("--school-after", required=True, help="F_school_after.csv")
    ap.add_argument("--grid-result", required=True, help="F_grid_result.csv")
    ap.add_argument("--assignment", required=True, help="F_assignment.csv")
    ap.add_argument("--opt-meta", required=True, help="F_opt_meta.json")
    ap.add_argument("--sfca-pri", required=True, help="F_sfca_A_pri_init.npy")
    ap.add_argument("--sfca-mid", required=True, help="F_sfca_A_mid_init.npy")
    ap.add_argument("--sfca-comb", required=True, help="F_sfca_A_combined_init.npy")
    ap.add_argument("--mask", required=True, help="F_low_access_mask.npy")
    ap.add_argument("--time-mat", required=True, help="E_time_matrix.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--monte-carlo-n", type=int, default=500)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--primary-max-min", type=float, default=30.0)
    ap.add_argument("--middle-max-min", type=float, default=40.0)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = [
        _Param(a.school_before), _Param(a.school_after), _Param(a.grid_result),
        _Param(a.assignment), _Param(a.opt_meta), _Param(a.sfca_pri),
        _Param(a.sfca_mid), _Param(a.sfca_comb), _Param(a.mask),
        _Param(a.time_mat), _Param(str(out)),
        _Param(a.bootstrap_n), _Param(a.monte_carlo_n), _Param(a.random_seed),
        _Param(a.primary_max_min), _Param(a.middle_max_min),
    ]
    for name in ["I_school_resource_plan.csv", "I_grid_result.csv",
                 "I_assignment.csv", "I_mc_raw.csv", "I_summary_report.json"]:
        params.append(_Param(str(out / name)))
    tool_50_execute(params, _MsgShim())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="4.7/5.0 优化配置与效益评估（纯 Python）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("opt47", help="4.7 教育资源优化配置（参数见 --school-profile-csv 等）").set_defaults(func=main47)
    sub.add_parser("eval50", help="5.0 效益评估（参数见 --school-before 等）").set_defaults(func=main50)
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args.func()
