# 成果图集与解读（Results Gallery）

> 以下图件均来自实际运行产出（`results/maps/` 与 `results/charts/`），
> 每图一句话说明；方法细节见 [methodology.md](./methodology.md)。

---

## 一、总体架构

### AI + GIS 双轮驱动技术总架构
![AI+GIS双轮驱动架构](../results/charts/A2_AI_GIS双轮驱动架构图.png)
左侧 AI 引擎（AlphaEarth 特征提取、GeoXGBoost 预测、SMS 文本挖掘）与右侧 GIS 引擎（路网可达分析、空间聚类、ECFI、2SFCA）经统一数据流耦合，构成全链路 6 模块。

### ECFI 三维框架与三级回退机制
![ECFI三维框架](../results/charts/E1_ECFI三维框架_三级回退机制.png)
D1 活力 + D2 支撑力 + D3 压力三维空间代理及熵权（0.3224 / 0.4214 / 0.2563），右侧为 POI"服务区 → 格网 → 缓冲区"三级空间回退逻辑。

### GeoXGBoost 双模型架构
![GeoXGBoost双模型架构](../results/charts/E2_GeoXGBoost双模型架构图.png)
Box-Cox（λ=0.0743）→ RBF 空间编码（8 锚点）→ Optuna 两阶段调参（粗调 75 + 精调 40）→ 诊断模型 A / 制图模型 B → SHAP 归因 → 风险栅格。

---

## 二、空间底座

### 研究区区位与数据概览
![研究区区位](../results/maps/B1_研究区区位与数据概览图.png)
章贡区 427.8 km²，2762 个 250 m 格网 + 99 所学校分层（小学/初中/高中）+ 路网河流。

### 六类社区空间分布
![六类社区](../results/maps/C1_六类社区空间分布图.png)
空间约束聚类将格网划为 C1 新城高知区 ~ C6 隐性收缩区六类社区，作为需求偏好与错配诊断的空间单元。

### 学校多级服务区分布
![多级服务区](../results/maps/C3_学校多级服务区分布图_修复版.png)
小学步行 5/10/15 min 等时圈 + 中学骑行 20 min 等时圈，LTS+OSRM 标定的 17 类道路权重、跨江无桥 ×5 阻抗惩罚。

---

## 三、诊断与预测

### ECFI 教育压力指数分布
![ECFI压力指数](../results/maps/C4_ECFI教育压力指数分布图.png)
99 校按三维加权综合压力分级着色，高压力学校集中于老城高密度与城乡过渡地带。

### SHAP 特征重要性
![SHAP特征重要性](../results/charts/D2_SHAP特征重要性.png)
模型预测力的来源排序——空间结构特征（活力、建筑密度、人口）与 RBF 空间编码项共同主导学位压力。

### SHAP 蜂群图
![SHAP蜂群图](../results/charts/shap_summary_beeswarm.png)
每个样本一个点，横轴为 SHAP 贡献、颜色为特征取值，展示特征-压力的非线性作用方向。

### 双模型一致性检验
![双模型一致性](../results/charts/dual_model_consistency.png)
诊断模型 A 与制图模型 B 的预测秩相关（Spearman ρ）与差异分布，验证两套模型空间认知一致。

### 模型性能验证图组
![模型性能验证](../results/charts/D4_模型性能验证图组.png)
实测 vs 预测散点（含 1:1 线）、残差图、Q-Q 图与残差直方图四联诊断。

### 5 折交叉验证对比
![CV对比](../results/charts/cv_results_comparison.png)
模型 A / 模型 B 在 5 折分层 CV 下的 R²、RMSE、MAE 分组对比（误差棒 = ±1σ）。

---

## 四、软实力与优化

### 六类社区 × 四维软实力需求雷达图
![六维雷达图](../results/charts/六维雷达图.png)
AHP 权重矩阵的可视化：六类社区对 T1 科创 / T2 艺体 / T3 心理德育 / T4 劳动实践四维软实力的差异化需求。

### 2SFCA 可达性优化前后对比
![2SFCA对比](../results/maps/C6_2SFCA优化前后对比图_优化版.png)
优化前后 2SFCA 可达性格网对比，可达性薄弱区显著收敛。

### 公平性评估：Lorenz 曲线 + Gini
![Lorenz与Gini](../results/charts/H2_fairness_2sfca.png)
优化前后 Lorenz 曲线与 Gini 系数（Bootstrap 95% CI），量化教育资源空间公平性改善。

### Monte Carlo 稳健性验证
![MonteCarlo稳健性](../results/charts/D7_MonteCarlo稳健性验证.png)
500 次模拟下目标函数与 Gini 的分布及各校被选为扩容校的频率，验证方案不依赖单次需求扰动。

---

## 五、可迁移性

### 可迁移架构示意
![可迁移架构](../results/charts/F3_可迁移架构示意图.png)
数据替换方案（AlphaEarth 全球覆盖 → 高德 POI → OSM 路网 → 参数本地化），方法可推广到任意城市。

---

*图件均由仓库代码（`geoxgboost.py`、`step47_50_optimization.py`、matplotlib 等）自动生成。*
