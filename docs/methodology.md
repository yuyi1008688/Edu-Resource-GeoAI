# 技术方法详解（Methodology）

> 本文对应研究技术路线 4.1–5.0 章节，全部参数与逻辑均取自本仓库实际代码，可按图索骥复现。
> 代码位置标注格式：`src/core/xxx.py` / `src/pipeline/stepxx.py`。

---

## 0. 总体框架

技术路线：**大模型特征提取 → 空间约束聚类 → 路网可达服务区 → ECFI 空间代理诊断 → GeoXGBoost 双模型压力预测 → MILP 优化配置 → 三维效益评估**，另有一条并行的 **4.6 学校软实力（SMS）文本挖掘** 支线在优化阶段汇入。

| 章节 | 模块 | 主要代码 |
|------|------|----------|
| 4.1 | 渔网精准裁剪 / 栅格特征提取 / 城市活力指数 | `fishnet_cutting.py` · `extract_raster.py` · `vitality_index.py` |
| 4.2 | 空间约束聚类 → 六类社区 | `step42_cluster.py`（spopt.Skater） |
| 4.3 | 路网预处理 + 多级服务区 | `step43_road_preprocess.py` · `step43b_service_area.py` |
| 4.4 | ECFI 三维空间代理诊断 | `step44_ecfi_diagnosis.py` |
| 4.5 | GeoXGBoost 双模型 | `geoxgboost.py` + `step45_geoxgboost.py` |
| 4.6 | 学校软实力标签与 SMS | `sms_engine.py` + `constants.py` + `step46_pipeline.py` |
| 4.7 | 教育资源 MILP 优化配置 | `step47_50_optimization.py`（opt47） |
| 5.0 | 三维效益评估 | `step47_50_optimization.py`（eval50） |

---

## 1. 数据底座（4.1–4.2）

### 1.1 AlphaEarth 6 维特征栅格

10 m 分辨率、EPSG:4526 投影的 6 波段栅格，波段语义与字段名（受 Shapefile 10 字符限制截断）：

| 波段 | 字段名 | 含义 |
|------|--------|------|
| 1 | `Decay_Idx` | 衰减指数 Decay_Index |
| 2 | `Build_Den` | 建筑密度 Building_Density |
| 3 | `Road_Den` | 路网密度 Road_Density |
| 4 | `Txt_Compl` | 纹理复杂度 Texture_Complexity |
| 5 | `LandUseMx` | 土地利用混合度 LandUse_Mix |
| 6 | `Green_Cov` | 植被覆盖 Green_Coverage |

### 1.2 渔网精准裁剪 V4.0（`fishnet_cutting.py`）

目标：把 250 m 渔网的每个像元分类为 **城区(0) / 荒野(1) / 水体(2)**，剔除非建成区干扰。流水线：

1. **MNDWI 水体通道**：加载现成 MNDWI 或由 Green(B03)+SWIR(B11) 现算 `(G−S)/(G+S)`，重投影对齐到参考栅格；`MNDWI > 0` 或格网内水体像元占比 > 0.5 判为水体。
2. **多波段评分通道**：建筑/道路密度按 95 分位归一化 ×100，植被取 `(1−GC)×100`，土地利用混合度 min-max 归一化；加权合成城市化评分——
   `score = 0.50·BD + 0.25·RD + 0.15·(1−GC) + 0.10·LM`
3. **模糊带**：评分 ∈ [27, 31] 为模糊带，以建筑密度 0.025 为界二次判定；清晰带阈值 ≤27 / ≥31。
4. **强制规则**：BD ≥ 0.08 强制城区；BD ≤ 0.001 且 GC ≥ 0.92 强制非城区（与水体冲突时水体优先）。
5. **K-Means 精修**：6 波段均值 + MNDWI 两列标准化后 K=6 聚类（`n_init=20`，`random_state=42`），仅对模糊带格网按簇众数投票改判（水体格网豁免）。
6. **Isolation Forest 异常清洗**：对城区格网 `contamination=0.04`，异常且高植被/低建筑/低水体者改判荒野，高水体者改判水体。
7. **空间平滑**：STRtree 邻接查询，2 轮迭代——孤立城区（≥3 个邻居全非城区）降为荒野；被城区包围（≥85% 邻居为城区）的荒野升为城区。

### 1.3 栅格特征提取（`extract_raster.py`）

`rasterstats.zonal_stats` 逐波段提取每个格网的均值（`all_touched=True`，nodata=−9999，无值补 0），写入渔网属性并同步导出 CSV，供聚类与建模使用。

### 1.4 城市活力指数（`vitality_index.py`）

- **A. KDE 密度**：18,637 条高德 POI 投影后以高斯核（带宽 260 m）估计格网质心处密度；
- **B. Shannon 熵多样性**：格网内 POI 类别分布 `H = −Σ p·ln p` 衡量功能混合度；
- **C. 融合**：两者各自 Z-score 标准化（截断 ±3.5σ）后 **0.5 : 0.5** 加权，再 min-max 归一化到 [0,1] 得 `vitality`。

### 1.5 空间约束聚类 → 六类社区（4.2）

以 6 维特征 + 活力指数做空间约束聚类（spopt.Skater），产出六类社区并映射为语义名称（映射表 `constants.py::ALIAS_42`）：

| 聚类名 | 语义标签 |
|--------|----------|
| 中央活力核心区 | C1_新城高知区 |
| 科教文化更新区 | C2_老城退休区 |
| 产业交通枢纽区 | C3_产业工人区 |
| 成熟综合城区 | C4_成熟居住区 |
| 城乡生态过渡区 | C5_城乡过渡带 |
| 产城融合拓展区 | C6_隐性收缩区 |

---

## 2. 路网可达服务区（4.3，`step43_road_preprocess.py` / `step43b_service_area.py`）

- **路网预处理**：依据 **LTS（Level of Traffic Stress）理论 + OSRM 本地化标定**得到的终审权重表，更新 OSM 路网 17 类道路的步行/骑行权重字段并计算通行时间，构建可计算的路网权重表。
- **服务区生成**：

| 参数 | 值 |
|------|-----|
| 小学 | 步行 4.5 km/h，5 / 10 / 15 min 三级等时圈 |
| 中学 | 骑行 14.4 km/h，20 min 等时圈 |
| 跨江约束 | 桥梁连通；无桥处 ×5 阻抗惩罚 |

- **输出**：99 校 Dissolve 后的服务区多边形，挂接学校信息，是 ECFI 诊断与 GeoXGBoost 特征构建的空间单元。

---

## 3. ECFI 三维空间代理诊断（4.4，`step44_ecfi_diagnosis.py`）

ECFI（Education Capacity-Facility Index）在**学校服务区**尺度合成三个空间代理维度：

| 维度 | 权重 | 空间代理 |
|------|------|----------|
| D1 活力 | 0.3224 | 服务区内格网活力均值 |
| D2 支撑力 | 0.4214 | 居住 + 教育 + 路网熵权 |
| D3 压力 | 0.2563 | 需求−供给压力 |

工程细节：

- **Winsorization**：D3 在 P95 截尾，抑制极端值对后续回归的杠杆效应；
- **POI 三级空间回退**：服务区 → 格网 → 缓冲区，保证每校三个维度均可落值；
- 命令行提供 28 个参数（默认值与原始实现一致），算法逻辑在 step44 执行体中（含 `rasterio.Env(GDAL_MEM_ENABLE_OPEN="YES")` 兼容补丁与三层建筑数据回退）。

---

## 4. GeoXGBoost 双模型（4.5，`geoxgboost.py`）

### 4.1 架构设计

| | 模型 A · 诊断模型 | 模型 B · 制图模型 |
|---|---|---|
| 角色 | 解释学位压力成因 | 生成全域连续压力面 |
| 样本 | 学校点（服务区尺度特征） | 学校训练 → 2762 格网直推 |
| 特征 | 全特征：X1–X3 + 空间滞后 lag_X1–lag_X3 + lag_D3 + 坐标 + 扩展空间特征（到中心/河流距离、POI 多样性、居住比、紧凑度、服务区面积、社区占比 pct_C1–C6 等）+ RBF | 精简特征：X1_vitality、X2_build_den、X3_worldpop + 坐标 + RBF |
| 评价标准 | 5 折分层 CV 的 R² / MdAPE | 格网唯一值数、变异系数 CV、Spearman 方向一致性 |

### 4.2 关键技术

- **目标变换**：D3 经 P95 Winsorization 后 **Box-Cox 变换**（本案例 λ = 0.0743），预测后逆变换回原尺度；无 Box-Cox 时回退对数变换 + smearing 修正。
- **RBF 空间编码**：`RBFSpatialEncoder` 以 8 个 KMeans 锚点、γ 取距离中位数将坐标编码为非线性空间基函数，让树模型隐式获得空间趋势项。
- **空间滞后**：cKDTree K=6 高斯核空间权重矩阵，计算 X1/X2/X3 与 D3 的空间滞后项（空间计量思想的特征化）。
- **GWR 式样本权重**：`compute_gwr_sample_weights` 按局部密度给样本加权，缓解学校空间分布不均。
- **Optuna 两阶段调参**：粗调 75 次 → 两阶段特征精简（`two_stage_feature_selection`，目标样本/特征比 5:1）→ 精调 40 次；CV 指标 MdAPE；模型 B 单独调参（≥30 次）。
- **5 折分层 CV**：按原始 D3 分层（`make_stratified_kfold_by_raw_d3`），同时报告变换尺度与逆变换尺度的 R²/RMSE/MAE/MAPE/MdAPE。
- **Huber 基线**：`huber_baseline` 提供线性稳健回归对照。
- **残差空间自相关**：K=6 近邻 Moran's I 检验残差是否仍有未解释的空间结构。

### 4.3 一致性检验（双模型灵魂）

用模型 B **直接对学校坐标推断**（避免查格网表的分辨率损失），计算三组 Spearman 秩相关：

- ρ(诊断预测, 制图预测)：两模型对学校压力排序的一致性（要求 > 0.4 且 p < 0.05）；
- ρ(诊断预测, D3 真实) 与 ρ(制图预测, D3 真实)：各自与真值的方向一致性；
- 若模型 B 在学校点退化为常数（std < 1e-6），自动回退 **IDW 插值**（k=15，幂 2）。

### 4.4 风险分级与输出

- 格网预测值经 P2–P98 百分位归一化后，用 **Jenks 自然断点**分 5 级（极低/低/中/高/极高），另记 P80 阈值并输出高压力边界线；
- 输出：连续压力面 `pressure_risk.tif`、分级栅格 `pressure_class.tif`、格网矢量 `pressure_coefs.shp/gpkg`、学校级预测 CSV、`model_report.json` 全量报告，以及预测诊断、双模型一致性、风险分布、CV 对比、SHAP 蜂群/柱状/依赖/空间共 8 组 SCI 风格图件。

---

## 5. 学校软实力标签体系与 SMS（4.6，`sms_engine.py` + `constants.py`）

### 5.1 三级语料与证据分级

| 层级 | 来源 | 角色 |
|------|------|------|
| L1 | 学校自述文本 | 主体证据 |
| L2 | 媒体报道 | 旁证 |
| L3 | 教育局公示 | 官方认定（最高可信） |

**E0–E5 证据规则引擎**（`_judge()`）：

- E1（1.0）：L3 官方认定 + 语料共现双确认；
- E2（0.90）：仅 L3 认定；
- E3（0.70）：锚点词与实体词在 ±60 字符窗口共现（排除"拟建/筹备中"等否定短语）；
- E4（0.40）：≥2 个锚点关键词命中；
- E5（0.20）：单关键词泛化命中；
- E0（0.0）：无证据。

### 5.2 8 维标签体系

4 个一级维度（T1 科创 / T2 审美 / T3 身心 / T4 实践）× 2 个二级维度（如 T1a_STEM教育、T1b_科学探究……T4b_研学实践），每维维护 `anchor / self / extra / honor` 四类词表（`constants.py::DICT`）。

### 5.3 SMS 与错配判定

1. 二级维度按置信度 ≥ 0.70 汇总到一级维度（`_rollup`），推导学校类型（7 类：全面综合型 / 复合协同发展型 / 四类单维引领型 / 无特色均衡发展型）与主导类型；
2. **社区需求**：`AHBP_THESIS` AHP 矩阵给出 6 类社区 × 4 维度的需求偏好权重；学校服务区与社区相交面积加权得到该校的需求向量（`community_map.csv` 面积制表，由 `step46_pipeline.py community-map` 生成）；
3. **SMS** = Σ(维度置信度 × 社区需求权重) × (1 + 0.15 × L3验证率)，全域 max 归一化；
4. **错配判定**：有标签学校中 SMS 落入最低四分位（Q1）者标记为错配候选（供给-需求错配）；
5. **人机一致性**：双人人工复核 35 条抽样，机器 vs 多数决精确一致率、二值化 Po / Kappa / F1、评审员间一致性全套统计（`review` 子命令）。

### 5.6 稳健性检验（`--sweep`）

置信度阈值 0.70→0.65/0.75 的标签集合变化（<15%）、共现窗口 60→40/80 的 E3 集合变化（<10%）、α 0.15→0.10/0.20 的 SMS Spearman ρ（>0.90）。

---

## 6. MILP 优化配置（4.7，`step47_50_optimization.py::opt47`）

5 阶段流水线：**容量估算（CapacityEstimator）→ 需求格网（DemandGridBuilder）→ 路网 OD 矩阵（RoadNetworkBuilder + NetworkOD）→ PuLP/CBC 线性规划（CapacityAllocationOptimizer）→ 优化方案输出**。学校软实力得分（SoftpowerImputer）作为配置优先级的输入之一。

---

## 7. 三维效益评估（5.0，`step47_50_optimization.py::eval50`）

| 维度 | 指标 | 方法 |
|------|------|------|
| 公平性 | 2SFCA 可达性 + Gini + Lorenz 曲线 | Bootstrap 95% CI |
| 效率 | 覆盖率（按学校类型 / 社区类型分组） | 优化前后对比 |
| 稳健性 | 优化方案稳定性 | Monte Carlo 500 次模拟 |

---

## 8. 可复现性锚点

- 随机种子：`RANDOM_SEED = 20260724`（`geoxgboost.py`）；
- K-Means/IsolationForest：`random_state=42`（`fishnet_cutting.py`）；
- 全部阈值、权重、词表集中于 `src/utils/constants.py` 与各模块 `DEFAULT_CONFIG`，修改即生效、无需动算法主体。

---

*本文档由实际代码整理生成；如与代码不符，以代码为准。*
