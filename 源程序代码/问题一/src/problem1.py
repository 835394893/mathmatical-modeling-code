"""问题一统一运行入口。

目录约定：
- data/：输入 CSV
- outputs/：运行生成的图片和结果
- src/problem1_analysis.py：原始分析实现

这个入口在不改动原分析逻辑的前提下，把旧代码对“同目录文件”的依赖
兼容到新的项目目录结构中。
"""

from pathlib import Path
import os
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
IMPLEMENTATION = Path(__file__).resolve().with_name("problem1_analysis.py")


def _run_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    original_read_csv = pd.read_csv

    def read_csv_with_data_fallback(path, *args, **kwargs):
        """优先读取代码指定路径；若不存在，则到 data/ 查找同名文件。"""
        if isinstance(path, (str, os.PathLike)):
            candidate = Path(path)
            if not candidate.exists():
                fallback = DATA_DIR / candidate.name
                if fallback.exists():
                    path = fallback
        return original_read_csv(path, *args, **kwargs)

    pd.read_csv = read_csv_with_data_fallback

    try:
        # 旧代码中的相对输出路径统一落到 outputs/。
        os.chdir(OUTPUT_DIR)
        source = IMPLEMENTATION.read_text(encoding="utf-8-sig")

        # 让旧代码中的 BASE_DIR 指向 outputs/；输入 CSV 若不存在会由
        # read_csv_with_data_fallback 自动转到 data/，生成结果则留在 outputs/。
        runtime_globals = {
            "__name__": "__main__",
            "__file__": str(OUTPUT_DIR / "problem1.py"),
        }
        exec(compile(source, str(IMPLEMENTATION), "exec"), runtime_globals)
    finally:
        pd.read_csv = original_read_csv
        os.chdir(original_cwd)


if __name__ == "__main__":
    _run_analysis()
