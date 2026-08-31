# -*- coding: utf-8 -*-
"""
step42_cluster.py — 4.2 活力诊断与社区聚类（纯 Python CLI）

纯 Python 实现空间约束多元聚类（Spatially Constrained Multivariate
Clustering，SKATER 族）」：
  - 邻接：格网质心 Delaunay 三角网（与论文 4.2.2 一致）；
  - 算法：spopt.Skater（Assunção et al. 2006 最小生成树剪边，专有 GIS 软件的同类工具
    的同源开源实现），K=6；
  - 特征：6 维（剔除纹理 Txt_Compl）vitality / Decay_Idx / Build_Den /
    Road_Den / LandUseMx / Green_Cov，先 StandardScaler；
  - 编号：用论文表 4-AB 公布的 6 类特征剖面作"语义锚点"做最近邻编号，
    使 CLUSTER_ID 的语义（C1..C6 社区类型）跨运行稳定对齐。

数据血缘：4.1 extract（5 物理特征）+ 4.1 vitality（活力）→ 本步骤
  → Fishnet_Cluster.shp（CLUSTER_ID 1..6）→ 4.4。

诚实口径：专有 GIS 软件为另一 SKATER 实现，开源 spopt 同算法不同实现，
  以封存 Complete_fishnet 为相同输入时 ARI≈0.69、格网约 81% 一致
  （6 类中 C2/C4/C5 规模高度吻合，C1/C3/C6 有实现性差异）。

示例：
  python src/pipeline/step42_cluster.py \
      --features output/extract/fishnet_features.shp \
      --vitality output/vitality/vitality_fishnet.gpkg \
      --out output/cluster/Fishnet_Cluster.shp
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

FEATS = ["vitality", "Decay_Idx", "Build_Den",
         "Road_Den", "LandUseMx", "Green_Cov"]
K = 6

# 论文表 4-AB：6 类社区在【原始尺度】下的特征剖面（语义锚点，仅用于编号对齐）
ANCHOR_RAW = {
    1: [0.151, 0.061, 0.196, 0.144, 0.209, 0.366],  # C1 产业交通枢纽区
    2: [0.692, 0.033, 0.361, 0.124, 0.107, 0.276],  # C2 中央活力核心区
    3: [0.441, 0.047, 0.241, 0.138, 0.177, 0.366],  # C3 成熟综合城区
    4: [0.076, 0.051, 0.160, 0.074, 0.252, 0.620],  # C4 城乡生态过渡区
    5: [0.160, 0.056, 0.171, 0.098, 0.216, 0.461],  # C5 产城融合拓展区
    6: [0.137, 0.073, 0.116, 0.126, 0.212, 0.503],  # C6 科教文化更新区
}
CLUSTER_CN = {
    1: "产业交通枢纽区", 2: "中央活力核心区", 3: "成熟综合城区",
    4: "城乡生态过渡区", 5: "产城融合拓展区", 6: "科教文化更新区",
}


def _centroid_key(gdf):
    gdf = gdf.copy()
    gdf["_cx"] = gdf.geometry.centroid.x.round(2)
    gdf["_cy"] = gdf.geometry.centroid.y.round(2)
    return gdf


def run(features_path, vitality_path, output_path, k=K,
        golden_path=None) -> dict:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from scipy.spatial import Delaunay
    from scipy import sparse
    from scipy.optimize import linear_sum_assignment
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score
    from libpysal.weights import W
    from spopt.region import Skater

    # ── 1. 读入 4.1 物理特征 + 活力，按质心对齐到同一张格网 ──────────
    feat = gpd.read_file(features_path)
    vit = gpd.read_file(vitality_path)
    print(f"[1/5] 物理特征格网 {len(feat)}，活力格网 {len(vit)}，CRS={feat.crs}")
    feat = _centroid_key(feat)
    vit = _centroid_key(vit)
    if "vitality" not in feat.columns:
        vit_keep = vit[["_cx", "_cy", "vitality"]].copy()
        gdf = feat.merge(vit_keep, on=["_cx", "_cy"], how="inner")
        print(f"      按质心合并活力后 {len(gdf)} 格")
    else:
        gdf = feat
    missing = [c for c in FEATS if c not in gdf.columns]
    if missing:
        raise KeyError(f"缺聚类特征列: {missing}; 现有列 {list(gdf.columns)}")

    # 与 v3 一致：LandUseMx 负值清零
    n_neg = int((gdf["LandUseMx"] < 0).sum())
    if n_neg:
        gdf.loc[gdf["LandUseMx"] < 0, "LandUseMx"] = 0.0
        print(f"      LandUseMx 负值清零 {n_neg} 格")

    # ── 2. 标准化 + Delaunay 邻接（质心） ──────────────────────────
    X_raw = gdf[FEATS].astype(float).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    coords = gdf[["_cx", "_cy"]].values
    n = len(gdf)
    tri = Delaunay(coords)
    rr, cc = [], []
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                rr += [simplex[a], simplex[b]]
                cc += [simplex[b], simplex[a]]
    adj = sparse.coo_matrix((np.ones(len(rr)), (rr, cc)),
                            shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()
    neighbors = {i: adj.getrow(i).indices.tolist() for i in range(n)}
    w = W(neighbors, silence_warnings=True)
    print(f"[2/5] Delaunay 邻接 {adj.nnz} 条，连通分量 {w.n_components}")

    # ── 3. SKATER 空间约束聚类 ─────────────────────────────────────
    zdf = pd.DataFrame(X, columns=[f"z_{c}" for c in FEATS])
    model = Skater(zdf, w, attrs_name=list(zdf.columns),
                   n_clusters=k, islands="increase")
    model.solve()
    raw_labels = np.array(model.labels_)
    print(f"[3/5] Skater 聚类完成，规模 "
          f"{[int((raw_labels == t).sum()) for t in sorted(set(raw_labels))]}")

    # ── 4. 语义锚点编号（把任意 label 对齐到 C1..C6 语义） ─────────
    # 在【原始尺度】比较，并用锚点各维极差归一化（固定尺度，不随本次格网
    # 子集的标准化 mean/std 漂移，比标准化空间欧氏更稳）。
    anchor_raw = np.array([ANCHOR_RAW[c] for c in range(1, k + 1)])
    dim_scale = np.ptp(anchor_raw, axis=0)
    dim_scale[dim_scale == 0] = 1.0
    cent_raw = np.array([X_raw[raw_labels == t].mean(axis=0) for t in range(k)])
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost[i, j] = np.linalg.norm(
                (cent_raw[i] - anchor_raw[j]) / dim_scale)
    row_i, col_j = linear_sum_assignment(cost)  # row=Skater类, col=锚点C编号-1
    to_semantic = {int(r): int(c) + 1 for r, c in zip(row_i, col_j)}
    gdf["CLUSTER_ID"] = [to_semantic[int(v)] for v in raw_labels]
    gdf["Cluster_CN"] = gdf["CLUSTER_ID"].map(CLUSTER_CN)
    print("[4/5] 语义编号对齐：",
          {CLUSTER_CN[c]: int((gdf['CLUSTER_ID'] == c).sum())
           for c in range(1, k + 1)})

    # ── 5. 输出（Fishnet_Cluster 同构） ────────────────────────────
    keep = ["CLUSTER_ID", "Cluster_CN"] + FEATS + ["geometry"]
    keep = [c for c in keep if c in gdf.columns]
    out = gdf[keep].copy()
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    out.to_file(output_path, encoding="utf-8")
    print(f"[5/5] ✅ 输出: {output_path}（{len(out)} 格，EPSG:{out.crs.to_epsg()}）")

    report = {"output": output_path, "n": len(out), "k": k,
              "sizes": {int(c): int((out['CLUSTER_ID'] == c).sum())
                        for c in range(1, k + 1)}}

    # 可选：对参考聚类结果验证（--golden，按质心对齐算 ARI/一致率）
    if golden_path and os.path.exists(golden_path):
        gold = _centroid_key(gpd.read_file(golden_path))
        cmp_ = out.copy()
        cmp_["_cx"] = cmp_.geometry.centroid.x.round(2)
        cmp_["_cy"] = cmp_.geometry.centroid.y.round(2)
        merged = cmp_[["_cx", "_cy", "CLUSTER_ID"]].merge(
            gold[["_cx", "_cy", "CLUSTER_ID"]], on=["_cx", "_cy"],
            suffixes=("_pred", "_gold"))
        yp = merged["CLUSTER_ID_pred"].values
        yg = merged["CLUSTER_ID_gold"].values
        ari = adjusted_rand_score(yg, yp)
        # 最优标签匹配一致率（不依赖语义编号；标签为 1..6，需用实际标签值映射）
        from sklearn.metrics import confusion_matrix
        lab = sorted(set(yg.tolist()) | set(yp.tolist()))
        cm = confusion_matrix(yg, yp, labels=lab)
        ri, ci = linear_sum_assignment(-cm)
        mp = {int(lab[p]): int(lab[t]) for t, p in zip(ri, ci)}
        acc = float((np.array([mp[int(v)] for v in yp]) == yg).mean())
        report["golden"] = {"matched_grids": int(len(merged)),
                            "ARI": round(float(ari), 4),
                            "best_match_accuracy": round(acc, 4),
                            "gold_sizes": {int(c): int((yg == c).sum())
                                           for c in sorted(set(yg))}}
        print(f"      对参考基准：共有格 {len(merged)}，ARI={ari:.4f}，"
              f"最优匹配一致率={acc:.4f}")
    return report


def main():
    ap = argparse.ArgumentParser(description="4.2 社区空间约束聚类（纯 Python，SKATER）")
    ap.add_argument("--features", required=True, help="4.1 extract 产出的物理特征格网")
    ap.add_argument("--vitality", required=True, help="4.1 vitality 产出的活力格网")
    ap.add_argument("--out", default=str(paths.output_dir() / "cluster" / "Fishnet_Cluster.shp"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--golden", default=None, help="参考聚类结果（可选，用于一致性验证）")
    args = ap.parse_args()
    run(args.features, args.vitality, args.out, args.k, args.golden)


if __name__ == "__main__":
    main()
