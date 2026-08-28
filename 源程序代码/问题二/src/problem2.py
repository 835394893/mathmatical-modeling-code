"""问题二统一运行入口。

目录约定：
- data/：输入 CSV
- outputs/：运行生成的 CSV 和图片
- src/problem2_analysis.py：原始分析实现

该入口通过统一路径适配保留原有建模逻辑，同时保证整理目录后仍能读取输入数据、
继续读取本次运行刚生成的中间结果，并把输出集中写入 outputs/。
"""

from pathlib import Path
import os
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
IMPLEMENTATION = Path(__file__).resolve().with_name("problem2_analysis.py")


def _run_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    original_read_csv = pd.read_csv

    def read_csv_with_data_fallback(path, *args, **kwargs):
        """先按原路径读取；若文件不存在，则到 data/ 查找同名输入文件。"""
        if isinstance(path, (str, os.PathLike)):
            candidate = Path(path)
            if not candidate.exists():
                fallback = DATA_DIR / candidate.name
                if fallback.exists():
                    path = fallback
        return original_read_csv(path, *args, **kwargs)

    pd.read_csv = read_csv_with_data_fallback

    try:
        # 旧代码中的相对输出路径和 BASE_DIR 统一指向 outputs/。
        # 输入文件若不存在，read_csv_with_data_fallback 会自动到 data/ 查找；
        # 本次运行生成的 TOPSIS 等中间文件则会直接从 outputs/ 继续读取。
        os.chdir(OUTPUT_DIR)
        source = IMPLEMENTATION.read_text(encoding="utf-8-sig")
        runtime_globals = {
            "__name__": "__main__",
            "__file__": str(OUTPUT_DIR / "problem2.py"),
        }
        exec(compile(source, str(IMPLEMENTATION), "exec"), runtime_globals)
    finally:
        pd.read_csv = original_read_csv
        os.chdir(original_cwd)


if __name__ == "__main__":
    _run_analysis()
