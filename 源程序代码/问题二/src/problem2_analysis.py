try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

import os
# 获取本py文件所在的文件夹路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ==========================================
# 问题二：
# 跨省高考成绩分布比较
# 描述统计 + KDE辅助 + Wasserstein距离
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import (
    wasserstein_distance,
    skew,
    kurtosis
)
from sklearn.neighbors import KernelDensity


# ===============================
# 中文字体
# ===============================
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False


# ===============================
# 1. 文件
# ===============================
files={
"山西2025物理":"shanxi_2025_physics.csv",
"山西2025历史":"shanxi_2025_history.csv",
"山西2026物理":"shanxi_2026_physics.csv",
"山西2026历史":"shanxi_2026_history.csv",
"河北2025物理":"hebei_2025_physics.csv",
"河北2025历史":"hebei_2025_history.csv",
"河北2026物理":"hebei_2026_physics.csv",
"河北2026历史":"hebei_2026_history.csv",
"浙江2025":"zhejiang_2025.csv",
"浙江2026":"zhejiang_2026.csv"
}

# ===============================
# 2. 本科线
# ===============================
line_df=pd.read_csv(os.path.join(BASE_DIR,"undergraduate_line.csv"))
cutoff=dict(zip(files.keys(),line_df["本科线"]))


# ===============================
# 3. 读取成绩
# ===============================
def load_score(file,name):
    df=pd.read_csv(file)
    df=df[["分数","本分人数"]]
    # 本科线上
    df=df[df["分数"]>=cutoff[name]]
    score=np.repeat(df["分数"].values,df["本分人数"].astype(int).values)
    return score
samples={}
for name,file in files.items():
    samples[name]=load_score(os.path.join(BASE_DIR,file),name)
print("数据读取完成")


# ===============================
# 4. 描述统计
# ===============================
result=[]
for name,data in samples.items():
    result.append({"样本":name,"本科线上线人数":len(data),"平均分":np.mean(data),
                   "标准差":np.std(data),"偏度":skew(data),"峰度":
        kurtosis(data,fisher=False)})
stat_df=pd.DataFrame(result)


# ==========================================
# 4.1 追加本科上线率、特控上线率
# ==========================================
line_df=pd.read_csv(os.path.join(BASE_DIR,"undergraduate_lines.csv"),encoding="utf-8")
# 构造样本名称，与stat_df保持一致
line_df["样本"]=(line_df["省份"]+line_df["年份"].astype(str)+line_df["方向"])
# 浙江名称特殊处理
line_df.loc[line_df["省份"]=="浙江","样本"]=("浙江"+line_df["年份"].astype(str))
# 计算本科上线率
line_df["本科上线率"]=(line_df["本科线对应位次"]/line_df["报考人数"])
# 计算特控上线率
line_df["特控上线率"]=(line_df["特控线对应位次"]/line_df["报考人数"])
# 合并
stat_df=stat_df.merge(line_df[["样本","本科上线率","特控上线率"]],on="样本",how="left")
print("\n==========扩展描述统计==========")
display(stat_df.round(4))


# ===============================
# 5. KDE拟合
# （只计算，不绘图）
# ===============================
kde_result={}
for name,data in samples.items():
    kde=KernelDensity(kernel="gaussian",bandwidth=3)
    kde.fit(data.reshape(-1,1))
    kde_result[name]=kde
print("KDE拟合完成")


# ===============================
# 6. Wasserstein距离
# ===============================
names=list(samples.keys())
W=pd.DataFrame(np.zeros((len(names),len(names))),
    index=names,
    columns=names)
for i in range(len(names)):
    for j in range(len(names)):
        W.iloc[i,j]=wasserstein_distance(samples[names[i]],samples[names[j]])
print("\n==========Wasserstein距离==========")
display(W.round(3))


# ==========================================
# 10. KDE分布特征提取
# ==========================================
kde_features=[]
# 选择观察分数点
score_points=[450,500,550,600]
for name,kde in kde_result.items():
    # 分数范围
    x=np.arange(400,701)
    # KDE密度
    density=np.exp(kde.score_samples(x.reshape(-1,1)))
    # 离散归一化
    density=density/density.sum()
    # 累计分布
    cdf=np.cumsum(density)
    row={"样本":name}
    # --------------------------
    # 密度值
    # --------------------------
    for s in score_points:
        row[f"{s}分密度"]=density[s-400]
    # --------------------------
    # 高分比例
    # --------------------------
    for s in [500,550,600]:
        row[f"{s}分以上比例"]=1-cdf[s-400]
    kde_features.append(row)
kde_df=pd.DataFrame(kde_features)
print("\n==========KDE分布特征==========")
display(kde_df.round(4))


# ==========================================
# 平均分 + 标准差综合比较图
# 物理/历史合并
# ==========================================
def mean_std_compare_plot(df):
    # -------------------------
    # 数据整理
    # -------------------------
    plot_df=df.copy()
    # 排序
    order=[
        "山西2025物理",
        "山西2026物理",
        "山西2025历史",
        "山西2026历史",
        "河北2025物理",
        "河北2026物理",
        "河北2025历史",
        "河北2026历史",
        "浙江2025",
        "浙江2026"
    ]
    plot_df["样本"]=pd.Categorical(plot_df["样本"],categories=order,ordered=True)
    plot_df=plot_df.sort_values("样本")
    x=np.arange(len(plot_df))
    # -------------------------
    # 绘图
    # -------------------------
    fig,ax1=plt.subplots(figsize=(12,6),dpi=300)
    # -------------------------
    # 平均分柱状图
    # -------------------------
    # 2025与2026不同颜色
    colors=[]
    for name in plot_df["样本"]:
        if "2025" in name:
            colors.append("#4C72B0")
        else:
            colors.append("#DD8452")
    bars=ax1.bar(x,plot_df["平均分"],width=0.65,color=colors)
    ax1.set_ylabel("平均分",fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["样本"],rotation=45,ha="right")
    # -------------------------
    # 标准差折线
    # -------------------------
    ax2=ax1.twinx()
    ax2.plot(x,plot_df["标准差"],marker="o",linewidth=2.5,color="red",label="标准差")
    ax2.set_ylabel("标准差",fontsize=12)
    # -------------------------
    # 平均分数字标注
    # -------------------------
    for bar,value in zip(bars,plot_df["平均分"]):
        ax1.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+1,
            f"{value:.1f}",
            ha="center",
            fontsize=8
        )
    # -------------------------
    # 图例
    # -------------------------
    from matplotlib.patches import Patch
    legend_elements=[
        Patch(facecolor="#4C72B0",label="2025"),
        Patch(facecolor="#DD8452",label="2026")]
    ax1.legend(handles=legend_elements,loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(axis="y",alpha=0.3)
    ax1.set_title("2025-2026山西、河北、浙江高考成绩分布比较",fontsize=15,fontweight="bold")
    plt.tight_layout()
    plt.show()


# ==========================================
# 调用
# ==========================================
mean_std_compare_plot(stat_df)


# =====================================================
# 问题二：
# 基于百分位保持的跨省成绩转换模型
# 山西基准评价体系
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# =====================================================
# 中文显示
# =====================================================
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False


# =====================================================
# 1. 文件配置
# =====================================================
files={
"山西2025物理":
"shanxi_2025_physics.csv",
"山西2025历史":
"shanxi_2025_history.csv",
"山西2026物理":
"shanxi_2026_physics.csv",
"山西2026历史":
"shanxi_2026_history.csv",
"河北2025物理":
"hebei_2025_physics.csv",
"河北2025历史":
"hebei_2025_history.csv",
"河北2026物理":
"hebei_2026_physics.csv",
"河北2026历史":
"hebei_2026_history.csv",
"浙江2025":
"zhejiang_2025.csv",
"浙江2026":
"zhejiang_2026.csv"
}


# =====================================================
# 2. 本科线
# =====================================================
line_df=pd.read_csv(os.path.join(BASE_DIR,"undergraduate_line.csv"))
cutoff=dict(zip(files.keys(),line_df["本科线"]))


# =====================================================
# 3. 读取一分一段表
# =====================================================
def load_table(file,name):
    df=pd.read_csv(file)
    df=df[["分数","本分人数"]]
    # 本科线上截断
    df=df[df["分数"]>=cutoff[name]]
    df=df.sort_values("分数",ascending=False)
    return df.reset_index(drop=True)
tables={}
for name,file in files.items():
    tables[name]=load_table(os.path.join(BASE_DIR,file),name)
print("数据读取完成")


# =====================================================
# 4. 构造百分位函数
# =====================================================
def build_percentile(df):
    df=df.sort_values("分数",ascending=False)
    total=df["本分人数"].sum()
    # 高分累计人数
    cum=df["本分人数"].cumsum()
    percentile=cum/total*100
    return (df["分数"].values,percentile.values)
curves={}
for name,df in tables.items():
    curves[name]=build_percentile(df)


# =====================================================
# 5. 百分位 -> 山西分数
# =====================================================
def percentile_to_score(percentile,target_curve):
    score,p=target_curve
    # 百分位递增
    order=np.argsort(p)
    p=p[order]
    score=score[order]
    f=interp1d(p,score,bounds_error=False,fill_value=(score[-1],score[0]))
    return f(percentile)


# =====================================================
# 6. 单个省份映射
# =====================================================
def convert_source_to_shanxi(source_df,source_name,target_name):
    score,p=curves[source_name]
    target_curve=curves[target_name]
    result=[]
    # 每5分一个节点
    nodes=np.arange(source_df["分数"].max(),source_df["分数"].min()-1,-5)
    for s in nodes:
        # 找最近分数
        idx=(np.abs(score-s)).argmin()
        real_score=score[idx]
        percentile=p[idx]
        shanxi_score=percentile_to_score(percentile,target_curve)
        result.append({
            "原省份":source_name,
            "原始分":real_score,
            "百分位(%)":percentile,
            "对应山西分":shanxi_score,
            "Delta":shanxi_score-real_score
        })
    return pd.DataFrame(result)


# =====================================================
# 7. 四组转换
# =====================================================
groups={"2025物理":("山西2025物理",["河北2025物理","浙江2025"]),
        "2025历史":("山西2025历史",["河北2025历史","浙江2025"]),
        "2026物理":("山西2026物理",["河北2026物理","浙江2026"]),
        "2026历史":("山西2026历史",["河北2026历史","浙江2026"])}
all_tables=[]
plot_data={}
for year,(shanxi,target_list) in groups.items():
    plot_data[year]={}
    # 山西基准线
    sx=tables[shanxi]
    plot_data[year]["山西"]=(sx["分数"].values,np.zeros(len(sx)))
    for province in target_list:
        temp=convert_source_to_shanxi(tables[province],province,shanxi)
        all_tables.append(temp)
        plot_data[year][province]=(temp["原始分"].values,temp["Delta"].values)


# =====================================================
# 10. 六对象代表性分数转换展示表
# =====================================================
target_objects=[
    "河北2025物理",
    "河北2026物理",
    "河北2025历史",
    "河北2026历史",
    "浙江2025",
    "浙江2026"]
display_tables=[]
for obj in target_objects:
    # 确定山西基准
    if "物理" in obj:
        if "2025" in obj:
            shanxi="山西2025物理"
        else:
            shanxi="山西2026物理"
    else:
        if "2025" in obj:
            shanxi="山西2025历史"
        else:
            shanxi="山西2026历史"
    # 生成完整换分结果
    temp=convert_source_to_shanxi(tables[obj],obj,shanxi)
    # 三个代表性分数
    sample_index=[
        0,          # 高分
        len(temp)//2,   # 中间
        len(temp)-1     # 本科线附近
        ]
    sample=temp.iloc[sample_index].copy()
    # 字段重新命名
    sample=sample.rename(columns={"原省份":"来源对象","原始分":"代表分数","对应山西分":"转换山西分"})
    display_tables.append(sample)
# 合并18行
final_table=pd.concat(display_tables,ignore_index=True)
# 调整字段顺序
final_table=final_table[["来源对象","代表分数","百分位(%)","转换山西分","Delta"]]
print("\n================================")
print("六对象典型分数跨省转换结果")
print("================================")
display(final_table.round(3))
# 保存
final_table.to_csv("六对象典型分数转换表.csv",index=False,encoding="utf-8-sig")
print("已生成：六对象典型分数转换表.csv")


# =====================================================
# 9. 绘制Delta曲线
# =====================================================
for year,data in plot_data.items():
    plt.figure(figsize=(10,6),dpi=300)
    for name,(x,y) in data.items():
        # 删除最高分点
        if len(x)>1:
            x=x[1:]
            y=y[1:]
        if name=="山西":
            plt.plot(x,y,color="black",linewidth=2,linestyle="--",label="山西基准")
        else:
            plt.plot(x,y,linewidth=2,label=name)
    plt.axhline(0,color="gray",linestyle="--")
    plt.xlabel("原省高考成绩")
    plt.ylabel("转换至山西后的分数差 Δ")
    plt.title(f"{year}跨省成绩山西基准转换")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# 问题二：
# 山西、河北、浙江高考竞争环境综合评价
# 5个评价对象 × 6个指标
# 熵权-TOPSIS前置矩阵构造
# =========================================================
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance


# =========================================================
# 1. 文件配置
# =========================================================
files = {
    "山西物理":"shanxi_2025_physics.csv",
    "山西历史":"shanxi_2025_history.csv",
    "河北物理":"hebei_2025_physics.csv",
    "河北历史":"hebei_2025_history.csv",
    "浙江综合":"zhejiang_2025.csv"}


# =========================================================
# 2. 读取一分一段表
# =========================================================
def load_distribution(file):
    df=pd.read_csv(file,encoding="utf-8")
    df=df[["分数","本分人数"]]
    df=df.sort_values("分数")
    score=np.repeat(df["分数"].values,df["本分人数"].astype(int).values)
    return score
samples={}
for name,file in files.items():
    samples[name]=load_distribution(os.path.join(BASE_DIR,file))
print("一分一段表读取完成")


# =========================================================
# 3. X2 成绩竞争差异
# 平均 Wasserstein距离
# =========================================================
names=list(samples.keys())
W=pd.DataFrame(np.zeros((len(names),len(names))),
    index=names,
    columns=names)
for i in range(len(names)):
    for j in range(len(names)):
        W.iloc[i,j]=wasserstein_distance(samples[names[i]],samples[names[j]])
X2={}
for name in names:
    X2[name]=(W.loc[name].sum()/(len(names)-1))


# =========================================================
# 4. X3 前1%分数
# =========================================================
def top1_score(data):
    return np.percentile(data,99)
X3={}
for name,data in samples.items():
    X3[name]=top1_score(data)


# =========================================================
# 5. 读取宏观数据
# =========================================================
basic=pd.read_csv(os.path.join(BASE_DIR,"province_competition_data.csv"),encoding="utf-8")
basic.columns=basic.columns.str.strip()


# =========================================================
# 6. 读取本科线、特控线位次
# 新版 undergraduate_lines.csv
# 年份
# 省份
# 方向
# 本科线
# 本科线对应位次
# 特控线
# 特控线对应位次
# 报考人数
# =========================================================
lines=pd.read_csv(os.path.join(BASE_DIR,"undergraduate_lines.csv"),encoding="utf-8")
lines.columns=lines.columns.str.strip()
# 构造对象名称
lines["对象"]=(lines["省份"]+lines["方向"])
# 浙江特殊处理
lines.loc[lines["省份"]=="浙江","对象"]="浙江综合"


# =========================================================
# 合并位次数据
# =========================================================
basic=basic.merge(
    lines[["年份","对象","本科线对应位次","特控线对应位次"]],
    on=["年份","对象"],
    how="left")


# =========================================================
# 7. X4 本科上线率
# 原公式：
# 本科线对应位次 / 报名人数
# =========================================================
basic["X4本科上线率"]=(basic["本科线对应位次"]/basic["报名人数"])


# =========================================================
# 8. X5 特控上线率
# 原公式：
# 特控线对应位次 / 报名人数
# =========================================================
basic["X5特控上线率"]=(basic["特控线对应位次"]/basic["报名人数"])


# =========================================================
# 9. X6 本科资源供给
# 本科招生计划 / 报名人数
# =========================================================
basic["X6本科资源供给"]=(basic["本科招生计划"]/basic["报名人数"])


# =========================================================
# 10. X1报名人数
# =========================================================
basic["X1报名人数"]=basic["报名人数"]


# =========================================================
# 11. 合并X2、X3
# =========================================================
basic["X2成绩竞争差异"]=(basic["对象"].map(X2))
basic["X3前1%分数"]=(basic["对象"].map(X3))


# =========================================================
# 12. 输出5×6指标矩阵
# =========================================================
result=basic[["年份","对象","X1报名人数","X2成绩竞争差异","X3前1%分数","X4本科上线率","X5特控上线率","X6本科资源供给"]]
print("\n==============================")
print("最终评价指标矩阵")
print("==============================")
display(result.round(4))
# 保存
result.to_csv("TOPSIS_input_matrix.csv",index=False,encoding="utf-8-sig")
print("已生成 TOPSIS_input_matrix.csv")


# ======================================================
# 新高考背景下省际高考竞争难度评价
# 熵权法 + TOPSIS
# 输入:
# TOPSIS_input_matrix.csv
# 输出:
# TOPSIS_result.csv
# 指标权重
# 物理方向排名
# 历史方向排名
# 两张比较图
# ======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ======================================================
# 中文显示
# ======================================================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ======================================================
# 1.读取指标矩阵
# ======================================================
df = pd.read_csv(os.path.join(BASE_DIR,"TOPSIS_input_matrix.csv"),encoding="utf-8-sig")
df.columns = df.columns.str.strip()


# ======================================================
# 2. 指标定义
# ======================================================
indicators=["X1报名人数","X2成绩竞争差异","X3前1%分数","X4本科上线率","X5特控上线率","X6本科资源供给"]
X=df[indicators].values


# ======================================================
# 3. 指标方向处理
# ======================================================
# 难度评价：
# 正向：
# X1 报名人数
# X2 成绩竞争差异
# X3 前1%分数
# 负向：
# X4 本科上线率
# X5 特控上线率
# X6 本科资源供给
negative_index=[3,4,5]
X_adj=X.copy()
for i in negative_index:
    X_adj[:,i]=(X_adj[:,i].max()-X_adj[:,i])


# ======================================================
# 4. 标准化
# ======================================================
Z=np.zeros_like(X_adj,dtype=float)
for j in range(X_adj.shape[1]):
    Z[:,j]=(X_adj[:,j]-X_adj[:,j].min())/(X_adj[:,j].max()-X_adj[:,j].min()+1e-12)


# ======================================================
# 5. 熵权法
# ======================================================
P=Z/(Z.sum(axis=0)+1e-12)
n=len(Z)
entropy=[]
for j in range(Z.shape[1]):
    e=-(1/np.log(n))*np.sum(P[:,j]*np.log(P[:,j]+1e-12))
    entropy.append(e)
entropy=np.array(entropy)
difference=1-entropy
weight=difference/difference.sum()
weight_df=pd.DataFrame({"指标":indicators,"权重":weight})
print("\n==========熵权结果==========")
display(weight_df.round(4))


# ======================================================
# 6. TOPSIS计算
# ======================================================
V=Z*weight
ideal_best=V.max(axis=0)
ideal_worst=V.min(axis=0)
D_best=np.sqrt(((V-ideal_best)**2).sum(axis=1))
D_worst=np.sqrt(((V-ideal_worst)**2).sum(axis=1))
score=(D_worst/(D_best+D_worst))


# ======================================================
# 7. 保存TOPSIS结果
# ======================================================
result=pd.DataFrame({"年份":df["年份"],"对象":df["对象"],"TOPSIS难度指数":score})
result["排名"]=result["TOPSIS难度指数"].rank(ascending=False).astype(int)
result.to_csv("TOPSIS_result.csv",
index=False,encoding="utf-8-sig")


# ======================================================
# 8. 输出物理方向排名
# 包含浙江综合
# ======================================================
physics_rank=result[result["对象"].isin(["山西物理","河北物理","浙江综合"])].copy()
physics_rank=physics_rank.sort_values(["年份","TOPSIS难度指数"],ascending=[True,False])
physics_rank.insert(1,"方向","物理方向")
print("\n==========物理方向高考难度排名==========")
display(physics_rank.round(4))


# ======================================================
# 9. 输出历史方向排名
# 包含浙江综合
# ======================================================
history_rank=result[result["对象"].isin(["山西历史","河北历史","浙江综合"])].copy()
history_rank=history_rank.sort_values(["年份","TOPSIS难度指数"],ascending=[True,False])
history_rank.insert(1,"方向","历史方向")
print("\n==========历史方向高考难度排名==========")
display(history_rank.round(4))


# ======================================================
# 10. 年度变化
# ======================================================
change=result.pivot(index="对象",columns="年份",values="TOPSIS难度指数")
change["变化量"]=change[2026]-change[2025]
print("\n==========2025-2026变化==========")
display(change.round(4))


# ======================================================
# 11. 物理方向柱状图
# ======================================================
physics_plot=result[result["对象"].isin(["山西物理","河北物理","浙江综合"])]
plt.figure(figsize=(10,5),dpi=200)
plt.bar(physics_plot["年份"].astype(str)+"_"+physics_plot["对象"],physics_plot["TOPSIS难度指数"])
plt.ylabel("TOPSIS高考难度指数")
plt.title("物理方向及浙江综合组高考竞争难度比较")
plt.xticks(rotation=35)
plt.grid(axis="y",alpha=0.3)
plt.tight_layout()
plt.show()


# ======================================================
# 12. 历史方向柱状图
# ======================================================
history_plot=result[result["对象"].isin(["山西历史","河北历史","浙江综合"])]
plt.figure(figsize=(10,5),dpi=200)
plt.bar(history_plot["年份"].astype(str)+"_"+history_plot["对象"],history_plot["TOPSIS难度指数"])
plt.ylabel("TOPSIS高考难度指数")
plt.title("历史方向及浙江综合组高考竞争难度比较")
plt.xticks(rotation=35)
plt.grid(axis="y",alpha=0.3)
plt.tight_layout()
plt.show()


# ======================================================
# 5.5 六指标竞争特征雷达图
# ======================================================
import matplotlib.pyplot as plt
import numpy as np
# 合并年份，比较省份整体特征
radar_df = pd.DataFrame(Z,columns=indicators)
radar_df["对象"]=df["对象"]
# 2025年
radar_2025 = radar_df.iloc[:5]
# 三省名称
province_names=["山西","河北","浙江"]
# 选择方向代表
# 山西物理
# 河北物理
# 浙江综合
select=["山西物理","河北物理","浙江综合"]
plot_data=[]
for name in select:
    row=radar_df[radar_df["对象"]==name][indicators].values[0]
    plot_data.append(row)


