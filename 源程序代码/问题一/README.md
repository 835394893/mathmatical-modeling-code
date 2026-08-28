# 问题一

## 目录

- `src/problem1.py`：完整建模分析脚本，直接从 `data/` 读取输入，并把生成图片写入 `outputs/`。
- `notebooks/problem1.ipynb`：Notebook 运行入口，直接执行 `src/problem1.py`。
- `data/`：山西各年份一分一段输入数据。
- `outputs/`：运行生成的图表。

问题一已经不再使用路径兼容层，也不再保留额外的 `problem1_analysis.py`。

## 运行

从仓库根目录执行：

```bash
python "源程序代码/问题一/src/problem1.py"
```

脚本使用基于自身位置计算出的绝对项目路径，因此无论当前终端位于仓库根目录还是其他目录，都能稳定访问：

```text
问题一/
├── src/problem1.py
├── notebooks/problem1.ipynb
├── data/
└── outputs/
```

主要分析内容包括新旧高考成绩分布、描述统计、高分段比例、Wasserstein/KS 距离，以及基于截断偏态正态分布的反事实预测。
