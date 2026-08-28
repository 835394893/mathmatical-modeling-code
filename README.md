# Mathematical Modeling Code

数学建模校赛代码仓库，主要研究新高考背景下的成绩分布变化、跨省成绩转换与高考竞争难度评价。

## 项目结构

```text
源程序代码/
├── README.md
├── 问题一/
│   ├── src/
│   │   └── problem1.py              # 完整分析脚本，原生读取 data/、写入 outputs/
│   ├── notebooks/
│   │   └── problem1.ipynb           # Notebook 运行入口
│   ├── data/                         # 输入 CSV
│   └── outputs/                      # 图片与运行结果
└── 问题二/
    ├── src/
    │   ├── problem2.py              # 推荐运行入口
    │   └── problem2_analysis.py     # 原始完整分析实现
    ├── notebooks/
    │   └── problem2.ipynb           # Notebook 运行入口
    ├── data/                         # 输入 CSV
    └── outputs/                      # 换分表、TOPSIS 结果等
```

## 研究内容

### 问题一

分析山西新旧高考成绩分布变化，包括描述统计、高分段比例、Wasserstein 距离、KS 距离，以及基于截断偏态正态分布的反事实预测。

问题一的 `src/problem1.py` 已直接适配整理后的目录结构，不依赖当前工作目录，也不再通过兼容层访问 CSV。

### 问题二

比较山西、河北、浙江高考成绩分布，构建基于百分位保持的跨省换分模型，并使用熵权法 + TOPSIS 对省际高考竞争难度进行综合评价。

## 环境

项目基于 Python 3.13 开发，主要依赖：

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

## 运行

问题一：

```bash
python "源程序代码/问题一/src/problem1.py"
```

问题二：

```bash
python "源程序代码/问题二/src/problem2.py"
```

也可以分别打开：

- `源程序代码/问题一/notebooks/problem1.ipynb`
- `源程序代码/问题二/notebooks/problem2.ipynb`

整理后的入口会自动从 `data/` 读取输入，并把运行生成的文件集中放到 `outputs/`，不再要求 CSV、脚本和 Notebook 平铺在同一目录。

## 数据说明

山西相关统计采用公开的一分一段数据。河北、浙江部分分科人数属于建模估算数据，仅用于模型仿真与竞赛研究，不应作为官方统计数据引用。
