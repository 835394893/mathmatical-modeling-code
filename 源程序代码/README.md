# 源程序运行说明

本目录包含数学建模校赛问题一、问题二两组相互独立的程序。目录已按“代码 / Notebook / 输入数据 / 输出结果”重新整理。

## 目录约定

每个问题均采用以下结构：

```text
问题X/
├── src/         # Python 代码
├── notebooks/   # Jupyter Notebook
├── data/        # 输入数据
└── outputs/     # 输出结果
```

## 推荐运行方式

在仓库根目录执行：

```bash
python "源程序代码/问题一/src/problem1.py"
python "源程序代码/问题二/src/problem2.py"
```

也可以在 VSCode / Jupyter 中打开 `notebooks/` 下的对应 Notebook。

## 依赖

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

项目按 Python 3.13 环境整理。

## 路径处理

整理后的 `src/problem1.py` 与 `src/problem2.py` 是统一运行入口：

- 输入 CSV 自动从对应 `data/` 目录读取；
- 运行生成的 CSV、PNG 等文件统一写入 `outputs/`；
- 原完整分析逻辑保留在 `src/*_analysis.py` 中；
- Notebook 调用统一入口，因此移动目录后不需要手动修改工作目录。

## 注意事项

1. 不要随意修改 `data/` 中的文件名，否则旧分析逻辑可能无法识别输入文件。
2. `outputs/` 中的结果允许被后续运行覆盖。
3. 中文图形字体取决于本机字体环境；字体缺失可能导致方框，但不会影响数值结果。
4. 山西相关统计采用公开数据；河北、浙江部分数据为建模估算值，仅用于竞赛模型仿真。
