# 数据说明（Data）

> 本仓库**不包含**原始数据（体量大、部分含使用许可限制）。以下说明数据构成、关键参数与获取方式，
> 配齐后通过环境变量 `B236_DATA_DIR` 指向该目录，即可复现全流程。

## 1. 数据总览

| 数据 | 规格 | 来源 | 用途 |
|------|------|------|------|
| AlphaEarth 6 维特征栅格 | 10 m，6 波段，EPSG:4526 | AlphaEarth 地理大模型 | 渔网裁剪 / 聚类 / 建模特征 |
| 渔网格网 | 250 m × 250 m，2762 个 | step41 渔网裁剪生成 | 全流程空间单元 |
| 学校名单 | 99 所（小学 53 / 初中 36 / 高中 10），含坐标与类型 | 教育局公开信息整理 | 诊断与优化对象 |
| 高德 POI | 18,637 条（lng/lat/cat_id） | 高德开放平台 | 城市活力指数 |
| WorldPop 人口栅格 | 250 m，EPSG:4526 | WorldPop | 需求侧人口 |
| OSM 路网 | 17 类道路 + LTS/OSRM 标定权重 | OpenStreetMap | 路网可达服务区 |
| 河流 / 桥梁 | 河流中线与桥梁点 | OSM + 实测核对 | 跨江阻抗约束 |
| 建筑轮廓（可选） | 矢量 | OSM | 建筑密度回退方案 |
| L1 学校自述文本 | 99 行 × 99 校全覆盖 | 学校官网/公众号 | 4.6 软实力语料 |
| L2 媒体报道 | 111 行 / 94 校 | 新闻检索 | 4.6 软实力语料 |
| L3 教育局公示 | 60 行 / 39 校 / 30 类 | 教育局官网 | 4.6 官方认定 |

各表字段名以代码读取处为准（见 `src/core`、`src/pipeline` 对应模块的列名常量与 `--help`）。

> **学段校准**：学校名单的"类型"列若把个别高中/完全中学笼统记为"中学"，默认按"中学→初中"归类；可通过环境变量 `EDU_EXTRA_HIGH_KW`（逗号分隔的本地校名关键词）将其判为高中，使学段构成与实际一致（53/36/10），代码本身不内置任何具体校名。

## 2. 栅格波段定义（与代码一致）

`src/core/extract_raster.py::DEFAULT_BAND_NAMES`：

```
Band 1  Decay_Idx    衰减指数
Band 2  Build_Den    建筑密度
Band 3  Road_Den     路网密度
Band 4  Txt_Compl    纹理复杂度
Band 5  LandUseMx    土地利用混合度
Band 6  Green_Cov    植被覆盖
```

## 3. 4.6 软实力引擎输入清单

放在 `B236_DATA_DIR` 指向的目录（也可修改 `sms_engine.py` 顶部"文件路径配置区"）：

| 文件 | 必要列 |
|------|--------|
| `99校主名单_shp名称.csv` | school_id, School_ID, School_Name, Level, 经度, 纬度 |
| `L1_学校自述文本.csv` | 学校名称, L1_text, L1_source, L1_date |
| `L2_媒体报道文本.csv` | 学校名称, L2_title, L2_summary, L2_source, L2_date |
| `L3_教育局公示.csv` | 学校名称, L3_tag, L3_document, L3_date, L3_url |
| `名称_POI_ID对照.csv`（可选） | 名称, POI_ID |
| `community_map.csv` | 学校名称 + C1~C6 面积列（或 社区类型 列） |

运行 `python src/core/sms_engine.py health` 可自动体检上述文件是否就位。

## 4. GeoXGBoost（4.5）输入清单

`run_analysis()` 所需 `paths`（详见 `src/core/geoxgboost.py` 文件头示例）：

| 参数 | 内容 |
|------|------|
| `school_csv` | 学校表（含 school_id、D3 压力列，由 step44 ECFI 产出） |
| `fishnet_shp` | 250 m 格网（含 vitality 字段，由 step41 活力指数产出） |
| `iso_primary_path` / `iso_middle_path` | 小学 / 初中服务区（由 step43b 生成） |
| `worldpop_tif` | WorldPop 人口栅格 |
| `build_density_raster` 或 `buildings_path` | 建筑密度栅格或建筑轮廓（二选一） |
| `output_dir` | 输出目录 |

## 5. 数据获取方式

- **AlphaEarth**：Google Research 开源的 AlphaEarth Fields 特征（通过 Earth Engine 或官方发布渠道获取对应时相与区域），或替换为任意同语义 6 波段特征栅格；
- **高德 POI**：高德开放平台 Web 服务 API（搜索 POI，按类别抓取）；
- **WorldPop**：[worldpop.org](https://www.worldpop.org/) 免费下载（Constrained/Unconstrained 100m/250m 产品重投影到 EPSG:4526）；
- **OSM 路网 / 建筑轮廓**：[geofabrik.de](https://download.geofabrik.de/asia/china.html) 或 Overpass API 提取研究区裁剪；
- **学校名单与语料**：教育局官网、学校官网/公众号公开信息人工整理。

## 6. 坐标系约定

全流程统一 **EPSG:4526**（CGCS2000 / 3-degree Gauss-Kruger zone 38，中央经线 114°E），距离单位为米。
唯一例外：`vitality_index.py` 独立运行时默认 EPSG:4547，可由 `--crs` 参数覆盖；流水线调用时由 step41_vitality 统一传入 EPSG:4526。
