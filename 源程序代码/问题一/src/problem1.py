"""问题一：山西新旧高考成绩分布变化分析。

目录约定：
- ../data/：输入的一分一段 CSV 数据
- ../outputs/：脚本生成的图片

该脚本直接使用新目录结构，不依赖工作目录，也不需要路径兼容层。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import (
    kurtosis,
    ks_2samp,
    skew,
    skewnorm,
    wasserstein_distance,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)


# ======================================================
# 统一目录
# ======================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def data_file(filename: str) -> Path:
    """返回 data/ 中的输入文件，并在缺失时给出清晰报错。"""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"缺少输入数据：{path}")
    return path


def read_table(filename: str, columns=None) -> pd.DataFrame:
    """读取一分一段表；可选地限制字段。"""
    df = pd.read_csv(data_file(filename))
    if columns is not None:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"{filename} 缺少必要字段：{missing}")
        df = df[columns].copy()
    return df


def show_and_close() -> None:
    plt.show()
    plt.close()


# ======================================================
# 1. 新旧高考成绩概率分布
# ======================================================
comparison_files = {
    "2024理科": "shanxi_2024_science.csv",
    "2025物理组": "shanxi_2025_physics.csv",
    "2026物理组": "shanxi_2026_physics.csv",
    "2024文科": "shanxi_2024_arts.csv",
    "2025历史组": "shanxi_2025_history.csv",
    "2026历史组": "shanxi_2026_history.csv",
}


def load_probability_data(filename: str) -> pd.DataFrame:
    df = read_table(filename, ["分数", "本分人数"])
    total = df["本分人数"].sum()
    df["概率"] = df["本分人数"] / total
    return df.sort_values("分数")


probability_data = {
    name: load_probability_data(filename)
    for name, filename in comparison_files.items()
}


def plot_distribution(names, title):
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ["#2E86C1", "#E67E22", "#27AE60"]
    for color, name in zip(colors, names):
        df = probability_data[name]
        plt.plot(
            df["分数"],
            df["概率"],
            label=name,
            linewidth=2.2,
            color=color,
        )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("高考分数", fontsize=12)
    plt.ylabel("概率", fontsize=12)
    plt.grid(linestyle="--", alpha=0.35)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{title}.png", dpi=300, bbox_inches="tight")
    show_and_close()


plot_distribution(
    ["2024理科", "2025物理组", "2026物理组"],
    "山西高考物理组成绩概率分布",
)
plot_distribution(
    ["2024文科", "2025历史组", "2026历史组"],
    "山西高考历史组成绩概率分布",
)


# ======================================================
# 2. 一分一段表描述性统计
# ======================================================
def load_score_table(filename: str) -> pd.DataFrame:
    required = ["分数", "本分人数", "累计人数"]
    df = read_table(filename, required).dropna()
    for column in required:
        df[column] = pd.to_numeric(df[column])
    return df.sort_values("分数").reset_index(drop=True)


score_tables = {
    name: load_score_table(filename)
    for name, filename in comparison_files.items()
}

check_rows = []
for name, df in score_tables.items():
    check_rows.append([
        name,
        len(df),
        int(df["本分人数"].sum()),
        int(df["分数"].min()),
        int(df["分数"].max()),
        int(df["本分人数"].isna().sum()),
        int(df["累计人数"].isna().sum()),
    ])

check_table = pd.DataFrame(
    check_rows,
    columns=[
        "年份组别",
        "有效分数段",
        "考生人数",
        "最低分",
        "最高分",
        "本分人数缺失",
        "累计人数缺失",
    ],
).set_index("年份组别")
print(check_table.to_string())


def statistics(df: pd.DataFrame):
    n = df["本分人数"].sum()
    x = df["分数"]
    w = df["本分人数"]
    mean = (x * w).sum() / n
    var = (((x - mean) ** 2) * w).sum() / n
    std = np.sqrt(var)
    values = np.repeat(x.astype(int), w.astype(int))
    return [
        mean,
        np.median(values),
        std,
        skew(values),
        kurtosis(values),
        x.min(),
        x.max(),
    ]


summary_rows = []
for name, df in score_tables.items():
    summary_rows.append([name] + statistics(df))

summary = pd.DataFrame(
    summary_rows,
    columns=[
        "年份组别",
        "平均分",
        "中位数",
        "标准差",
        "偏度",
        "峰度",
        "最低分",
        "最高分",
    ],
).set_index("年份组别")
print(summary.round(3))

thresholds = [600, 650, 680]
high_rows = []
for name, df in score_tables.items():
    total = df["本分人数"].sum()
    row = [name]
    for threshold in thresholds:
        count = df.loc[df["分数"] >= threshold, "本分人数"].sum()
        row.append(count / total * 100)
    high_rows.append(row)

high_table = pd.DataFrame(
    high_rows,
    columns=[
        "年份组别",
        "600分以上(%)",
        "650分以上(%)",
        "680分以上(%)",
    ],
).set_index("年份组别")
print(high_table.round(3))


# 成绩分布箱线图
box_data = []
labels = []
for name, df in score_tables.items():
    box_data.append(
        np.repeat(df["分数"].astype(int), df["本分人数"].astype(int))
    )
    labels.append(name)

plt.figure(figsize=(11, 6), dpi=300)
plt.boxplot(box_data, showfliers=False, patch_artist=True)
plt.xticks(range(1, len(labels) + 1), labels, rotation=25)
plt.title("山西高考成绩分布箱线图", fontsize=16, fontweight="bold")
plt.ylabel("高考分数", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
show_and_close()


# 统计量变化趋势
plot_data = summary.loc[
    [
        "2024理科",
        "2025物理组",
        "2026物理组",
        "2024文科",
        "2025历史组",
        "2026历史组",
    ]
]
plt.figure(figsize=(10, 5.5), dpi=300)
x = np.arange(len(plot_data))
plt.plot(x, plot_data["平均分"], marker="o", linewidth=2.2, label="平均分")
plt.plot(x, plot_data["中位数"], marker="s", linewidth=2.2, label="中位数")
plt.xticks(x, plot_data.index, rotation=25)
plt.ylabel("分数")
plt.title("不同年份考生成绩中心位置变化", fontsize=16, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
show_and_close()


# ======================================================
# 3. Wasserstein 距离与 KS 距离
# ======================================================
def expand_scores(df: pd.DataFrame) -> np.ndarray:
    return np.repeat(df["分数"].values, df["本分人数"].values.astype(int))


score_samples = {
    name: expand_scores(df)
    for name, df in score_tables.items()
}
names = list(score_samples.keys())

wasserstein_matrix = pd.DataFrame(
    np.zeros((len(names), len(names))), index=names, columns=names
)
ks_matrix = pd.DataFrame(
    np.zeros((len(names), len(names))), index=names, columns=names
)

for i, name_i in enumerate(names):
    for j, name_j in enumerate(names):
        wasserstein_matrix.iloc[i, j] = wasserstein_distance(
            score_samples[name_i], score_samples[name_j]
        )
        ks_stat, _ = ks_2samp(score_samples[name_i], score_samples[name_j])
        ks_matrix.iloc[i, j] = ks_stat

wasserstein_matrix = wasserstein_matrix.round(2)
ks_matrix = ks_matrix.round(3)
print("Wasserstein距离矩阵：")
display(wasserstein_matrix)
print("KS距离矩阵：")
display(ks_matrix)

plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(wasserstein_matrix, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("山西高考不同年份及选科组分数分布Wasserstein距离", fontsize=15, fontweight="bold")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
show_and_close()

plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(ks_matrix, annot=True, fmt=".3f", cmap="Blues")
plt.title("山西高考不同年份及选科组成绩分布KS距离", fontsize=15, fontweight="bold")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
show_and_close()


# ======================================================
# 4. 截断偏态正态反事实预测
# ======================================================
def truncated_skewnorm_nll(params, data, lower):
    alpha, loc, scale = params
    if scale <= 0:
        return np.inf
    pdf = skewnorm.pdf(data, alpha, loc, scale)
    survival = 1 - skewnorm.cdf(lower, alpha, loc, scale)
    if survival <= 0:
        return np.inf
    likelihood = pdf / survival
    likelihood[likelihood <= 0] = 1e-12
    return -np.sum(np.log(likelihood))


def fit_truncated_skew(data, lower):
    init = [0, np.mean(data), np.std(data)]
    result = minimize(
        truncated_skewnorm_nll,
        init,
        args=(data, lower),
        method="Nelder-Mead",
    )
    alpha, loc, scale = result.x
    return {
        "alpha": alpha,
        "loc": loc,
        "scale": scale,
        "loss": result.fun,
    }


old_files = {
    "2022理科": "shanxi_2022_science.csv",
    "2023理科": "shanxi_2023_science.csv",
    "2024理科": "shanxi_2024_science.csv",
    "2022文科": "shanxi_2022_arts.csv",
    "2023文科": "shanxi_2023_arts.csv",
    "2024文科": "shanxi_2024_arts.csv",
}
old_scores = {
    name: expand_scores(load_score_table(filename))
    for name, filename in old_files.items()
}

lower_score = 400
params_rows = []
for name, values in old_scores.items():
    fitted = fit_truncated_skew(values, lower_score)
    fitted["year"] = name
    params_rows.append(fitted)

params_df = pd.DataFrame(params_rows)
print("2022-2024截断偏态模型参数")
display(params_df)


def predict_parameter(values):
    weights = np.array([0.2, 0.3, 0.5])
    return np.sum(np.asarray(values) * weights)


predict_result = {}
for group in ["理科", "文科"]:
    temp = params_df[params_df["year"].str.contains(group)]
    predict_result[group] = {
        column: predict_parameter(temp[column].values)
        for column in ["alpha", "loc", "scale"]
    }

predict_df = pd.DataFrame(predict_result).T
print("2025旧高考反事实模型参数预测")
display(predict_df)


def generate_distribution(params, n=100000):
    sample = skewnorm.rvs(
        params["alpha"],
        loc=params["loc"],
        scale=params["scale"],
        size=n * 3,
    )
    sample = sample[sample >= lower_score]
    return sample[:n]


counterfactual = {
    group: generate_distribution(predict_result[group])
    for group in ["理科", "文科"]
}
print("2025旧高考反事实分布生成完成")


def truncated_skew_density(x, alpha, loc, scale, lower):
    pdf = skewnorm.pdf(x, alpha, loc, scale)
    cdf_lower = skewnorm.cdf(lower, alpha, loc, scale)
    return pdf / (1 - cdf_lower)


def plot_three_lines(filename, counter_params, title, lower=400):
    df = read_table(filename, ["分数", "本分人数"]).sort_values("分数")
    score = df["分数"].values
    real_prob = df["本分人数"].values / df["本分人数"].sum()

    x = np.arange(lower, score.max() + 1)
    counter_density = truncated_skew_density(
        x,
        counter_params["alpha"],
        counter_params["loc"],
        counter_params["scale"],
        lower,
    )
    counter_density = counter_density / counter_density.sum()

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(score, real_prob, linewidth=1.5, label="2025实际分布")
    plt.plot(x, counter_density, linewidth=2.5, label="2025旧高考反事实预测")
    plt.xlabel("高考成绩", fontsize=13)
    plt.ylabel("考生比例", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    show_and_close()


plot_three_lines(
    "shanxi_2025_physics.csv",
    predict_result["理科"],
    "2025山西物理组实际分布与旧高考反事实预测",
)
plot_three_lines(
    "shanxi_2025_history.csv",
    predict_result["文科"],
    "2025山西历史组实际分布与旧高考反事实预测",
)


# ======================================================
# 5. 拟合与反事实预测评价
# ======================================================
def evaluate_fit(data, params, lower):
    x = np.arange(lower, data.max() + 1)
    real_counts = np.array([np.sum(data == score) for score in x])
    real_prob = real_counts / real_counts.sum()
    pred_prob = truncated_skew_density(
        x, params["alpha"], params["loc"], params["scale"], lower
    )
    pred_prob = pred_prob / pred_prob.sum()
    return {
        "RMSE": np.sqrt(mean_squared_error(real_prob, pred_prob)),
        "MAE": mean_absolute_error(real_prob, pred_prob),
        "Wasserstein": wasserstein_distance(
            x, x, u_weights=real_prob, v_weights=pred_prob
        ),
    }


fit_rows = []
for _, row in params_df.iterrows():
    name = row["year"]
    fit = evaluate_fit(old_scores[name], row, lower_score)
    fit["年份"] = name
    fit_rows.append(fit)
fit_eval = pd.DataFrame(fit_rows)
print("2022-2024历史拟合效果评价")
display(fit_eval)


def evaluate_prediction(filename, params, lower=400):
    df = read_table(filename, ["分数", "本分人数"]).sort_values("分数")
    score = df["分数"].values
    real_prob = df["本分人数"].values / df["本分人数"].sum()

    x = np.arange(lower, score.max() + 1)
    pred_prob = truncated_skew_density(
        x, params["alpha"], params["loc"], params["scale"], lower
    )
    pred_prob = pred_prob / pred_prob.sum()

    scores = np.arange(max(score.min(), x.min()), min(score.max(), x.max()) + 1)
    real_dict = dict(zip(score, real_prob))
    pred_dict = dict(zip(x, pred_prob))
    p = np.array([real_dict.get(value, 0) for value in scores])
    q = np.array([pred_dict.get(value, 0) for value in scores])
    p /= p.sum()
    q /= q.sum()

    wd = wasserstein_distance(scores, scores, u_weights=p, v_weights=q)
    ks = np.max(np.abs(np.cumsum(p) - np.cumsum(q)))
    return wd, ks


physics = evaluate_prediction("shanxi_2025_physics.csv", predict_result["理科"])
history = evaluate_prediction("shanxi_2025_history.csv", predict_result["文科"])
prediction_eval = pd.DataFrame(
    [physics, history],
    columns=["Wasserstein距离", "KS距离"],
    index=["2025物理组", "2025历史组"],
)
print("2025反事实预测评价")
display(prediction_eval)


# ======================================================
# 6. 新高考效应分解
# ======================================================
def read_real_distribution(filename):
    df = read_table(filename, ["分数", "本分人数"]).sort_values("分数")
    score = df["分数"].values
    prob = df["本分人数"].values / df["本分人数"].sum()
    return score, prob


def generate_predict_distribution(params, lower=400, upper=750):
    x = np.arange(lower, upper + 1)
    density = truncated_skew_density(
        x, params["alpha"], params["loc"], params["scale"], lower
    )
    return x, density / density.sum()


def w_real_predict(filename, params, lower=400):
    df = read_table(filename, ["分数", "本分人数"]).sort_values("分数")
    s1 = df["分数"].values
    p1 = df["本分人数"].values / df["本分人数"].sum()

    s2 = np.arange(lower, s1.max() + 1)
    p2 = truncated_skew_density(
        s2, params["alpha"], params["loc"], params["scale"], lower
    )
    p2 = p2 / p2.sum()

    low = max(s1.min(), s2.min())
    high = min(s1.max(), s2.max())
    x = np.arange(low, high + 1)
    p1_map = dict(zip(s1, p1))
    p2_map = dict(zip(s2, p2))
    p = np.array([p1_map.get(value, 0) for value in x])
    q = np.array([p2_map.get(value, 0) for value in x])
    p /= p.sum()
    q /= q.sum()
    return wasserstein_distance(x, x, u_weights=p, v_weights=q)


physics_time = w_real_predict("shanxi_2024_science.csv", predict_result["理科"])
physics_policy = w_real_predict("shanxi_2025_physics.csv", predict_result["理科"])
history_time = w_real_predict("shanxi_2024_arts.csv", predict_result["文科"])
history_policy = w_real_predict("shanxi_2025_history.csv", predict_result["文科"])

effect = pd.DataFrame(
    {
        "自然年际波动": [physics_time, history_time],
        "制度变革效应": [physics_policy, history_policy],
    },
    index=["2025物理组", "2025历史组"],
)
effect["自然贡献%"] = (
    effect["自然年际波动"]
    / (effect["自然年际波动"] + effect["制度变革效应"])
    * 100
)
effect["制度贡献%"] = (
    effect["制度变革效应"]
    / (effect["自然年际波动"] + effect["制度变革效应"])
    * 100
)
print("新高考效应分解")
display(effect)
