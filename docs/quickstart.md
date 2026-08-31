# 快速上手指南（Quickstart）

本项目为**纯开源 Python** 实现，无需任何商业 GIS 软件，普通 Python 虚拟环境即可端到端跑通 14 步全链。

## 0. 一键运行（推荐）

```bash
# 1) 新建虚拟环境并安装依赖（Python 3.10–3.12）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt

# 2) 指定输入数据目录（内容清单见 data/README.md）
#    Windows PowerShell：  $env:B236_DATA_DIR="D:\path\to\input"
#    Windows CMD：         set B236_DATA_DIR=D:\path\to\input
#    macOS / Linux：       export B236_DATA_DIR=/path/to/input
#    （可选）指定输出目录：  B236_OUTPUT_DIR，默认 ./output

# 3) 查看 14 步顺序与上游依赖，然后顺序跑全链
python examples/run_pipeline.py --list
python examples/run_pipeline.py

# 各步骤也可单独运行（参数见各 CLI 的 --help）：
python src/pipeline/step41_fishnet_extract.py cut --out output/fishnet
python src/pipeline/step43_road_preprocess.py preprocess
python src/pipeline/step46_pipeline.py pipeline
```

14 步依次为：格式转换 → 栅格归一化 → 渔网裁剪 → 栅格特征提取 → 城市活力指数 → 空间约束聚类 → 路网预处理 → 小学/中学等时圈服务区 → ECFI 诊断 → GeoXGBoost 双模型 → SMS 软实力 → MILP 优化 → 效益评估。全新空目录从零运行即可得到最终结果，无需任何历史中间件。

---

## 1. 环境配置

| 软件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10–3.12（推荐 3.11） | 全流程运行环境 |

核心依赖（完整列表见 `requirements.txt`）：GeoPandas / rasterio / pyogrio（空间读写）、scikit-learn / XGBoost / SHAP（建模与解释）、spopt（空间约束聚类）、networkx（路网最短路服务区）、PuLP（MILP 线性规划）、SciPy / pandas / numpy / matplotlib。

```bash
pip install -r requirements.txt
```

## 2. 数据准备

大数据（栅格、路网、POI、语料表）不入库，请按 [data/README.md](../data/README.md) 准备，并通过环境变量 `B236_DATA_DIR` 指向该目录。

## 3. 三种运行方式

### 方式一 · 一键流水线（推荐）

见上方第 0 节。编排器 `examples/run_pipeline.py` 按依赖顺序串联 14 步，上游输出自动作为下游输入；`--list` 可查看每一步的输入、输出与状态。

### 方式二 · 单步 CLI

`src/pipeline/` 下每个 `step*.py` 都是带 argparse 的独立命令，可用 `--help` 查看全部参数与默认值，例如：

```bash
python src/pipeline/step42_cluster.py --help
python src/pipeline/step43b_service_area.py --help
python src/pipeline/step45_geoxgboost.py --help
python src/pipeline/step47_50_optimization.py --help
```

### 方式三 · 独立核心引擎（4.6 软实力）

```bash
python src/core/sms_engine.py health            # 数据体检：检查 L1/L2/L3 语料完整性
python src/core/sms_engine.py pipeline          # 主引擎：标签判定 + SMS + 错配 + 图件
python src/core/sms_engine.py pipeline --sweep  # 主引擎 + 稳健性检验
python src/core/sms_engine.py review 甲.csv 乙.csv  # 人工复核统计（Kappa/Po/F1）
```

其它 core 模块同样带 argparse 独立入口，例如：

```bash
python src/core/fishnet_cutting.py --raster features.tif --fishnet grid.shp --output ./out
python src/core/extract_raster.py  --fishnet grid.shp --raster features.tif --output out.shp
python src/core/vitality_index.py  --fishnet grid.shp --poi poi.csv --output vitality.gpkg
```

## 4. 分阶段串联示例

见 [examples/run_pipeline.py](../examples/run_pipeline.py)：从格式转换到效益评估的完整调用序列模板，把数据目录替换为本地路径即可。

## 5. 环境变量一览

| 变量 | 作用 | 默认 |
|------|------|------|
| `B236_DATA_DIR` | 输入数据目录 | 脚本内相对路径兜底 |
| `B236_OUTPUT_DIR` | 输出根目录 | `./output` |
| `EDU_EXTRA_HIGH_KW` | 本地补充的高中校名关键词（逗号分隔），用于学段校准，代码不内置具体校名 | 空 |
| `GDAL_MEM_ENABLE_OPEN` | 新版 GDAL 内存指针开关，代码已自动 setdefault | YES |

## 6. 常见问题

**Q1 · 运行提示找不到模块 / 缺少依赖？**
先确认已激活虚拟环境并执行 `pip install -r requirements.txt`；单步 CLI 会自动把 `src/core`、`src/utils` 加入搜索路径，无需手动设置 `PYTHONPATH`。

**Q2 · sms_engine.py 报路径错误？**
优先用 `B236_DATA_DIR` 指定数据目录；也可在脚本顶部"文件路径配置区"改为绝对路径。可用 `python src/core/sms_engine.py health` 先做数据就位体检。

**Q3 · 报 `MEM:::DATAPOINTER=` 错误？**
新版 GDAL 默认关闭内存指针特性，代码已内置 `rasterio.Env(GDAL_MEM_ENABLE_OPEN="YES")` 上下文管理器（4.4 / geoxgboost），如自行修改代码请保留该上下文。

**Q4 · 为什么个别学校学段被归到初中？**
原始数据若把某些高中/完全中学笼统标为"中学"、校名又不含"高中/高级中学/完全中学"字样，默认会按"中学→初中"归类。设置 `EDU_EXTRA_HIGH_KW` 补充本地校名关键词即可校准，代码本身不写死任何具体校名。

**Q5 · 只想复用某一个算法（比如 GeoXGBoost 双模型）？**
`src/core/geoxgboost.py::run_analysis()` 是参数化入口，传入 `paths` 字典（学校 CSV、渔网、服务区、WorldPop、建筑密度、输出目录）即可独立运行，文件头部有完整参数示例。
