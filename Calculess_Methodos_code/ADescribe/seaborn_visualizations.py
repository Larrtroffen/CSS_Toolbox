import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ==============================================================================
# 接口化绘图函数 (API-like Plotting Functions)
# ==============================================================================

OUTPUT_DIR = "seaborn_charts_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_histplot(data_series, title='直方图与核密度估计', xlabel=None, ylabel='密度', bins='auto', kde=True, color=None, save_path=None, **kwargs):
    """
    生成并（可选）保存直方图，可附带核密度估计(KDE)。

    参数:
    data_series (pd.Series): 用于绘图的数据序列。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为data_series的name属性。
    ylabel (str, optional): Y轴标签。
    bins (str, int, sequence, optional): 直方图的箱数或规格。
    kde (bool, optional): 是否绘制核密度估计曲线。默认为True。
    color (str, optional): 图表颜色。
    save_path (str, optional): 保存图表的路径。如果为None，则显示图表。
    **kwargs: 传递给 sns.histplot 的其他参数。
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data_series, bins=bins, kde=kde, color=color, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else data_series.name)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"直方图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_kdeplot(data_frame, x_var, hue_var=None, title='核密度估计图', xlabel=None, ylabel='密度', fill=True, multiple="layer", save_path=None, **kwargs):
    """
    生成并（可选）保存核密度估计图。

    参数:
    data_frame (pd.DataFrame): 输入的数据框。
    x_var (str): DataFrame中用于绘制KDE的列名。
    hue_var (str, optional): DataFrame中用于颜色编码的类别列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_var。
    ylabel (str, optional): Y轴标签。
    fill (bool, optional): 是否填充KDE曲线下的区域。
    multiple (str, optional): 当使用hue时，多系列的处理方式 ('layer', 'stack', 'fill')。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.kdeplot 的其他参数。
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=data_frame, x=x_var, hue=hue_var, fill=fill, multiple=multiple, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_var)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"KDE图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_ecdfplot(data_frame, x_var, hue_var=None, title='经验累积分布函数图', xlabel=None, ylabel='累积概率', save_path=None, **kwargs):
    """
    生成并（可选）保存经验累积分布函数(ECDF)图。

    参数:
    data_frame (pd.DataFrame): 输入的数据框。
    x_var (str): DataFrame中用于绘制ECDF的列名。
    hue_var (str, optional): DataFrame中用于颜色编码的类别列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_var。
    ylabel (str, optional): Y轴标签。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.ecdfplot 的其他参数。
    """
    plt.figure(figsize=(8, 6))
    sns.ecdfplot(data=data_frame, x=x_var, hue=hue_var, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_var)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"ECDF图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_scatterplot(data_frame, x_var, y_var, hue_var=None, style_var=None, size_var=None, title='散点图', xlabel=None, ylabel=None, save_path=None, **kwargs):
    """
    生成并（可选）保存散点图。

    参数:
    data_frame (pd.DataFrame): 输入的数据框。
    x_var (str): X轴对应的列名。
    y_var (str): Y轴对应的列名。
    hue_var (str, optional): 用于颜色编码的类别列名。
    style_var (str, optional): 用于标记样式编码的类别列名。
    size_var (str, optional): 用于标记大小编码的数值或类别列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_var。
    ylabel (str, optional): Y轴标签。默认为y_var。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.scatterplot 的其他参数。
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=data_frame, x=x_var, y=y_var, hue=hue_var, style=style_var, size=size_var, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_var)
    plt.ylabel(ylabel if ylabel else y_var)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"散点图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_lineplot(data_frame, x_var, y_var, hue_var=None, style_var=None, title='折线图', xlabel=None, ylabel=None, markers=True, sort=True, save_path=None, **kwargs):
    """
    生成并（可选）保存折线图。

    参数:
    data_frame (pd.DataFrame): 输入的数据框。
    x_var (str): X轴对应的列名。
    y_var (str): Y轴对应的列名。
    hue_var (str, optional): 用于颜色编码的类别列名。
    style_var (str, optional): 用于线条样式编码的类别列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_var。
    ylabel (str, optional): Y轴标签。默认为y_var。
    markers (bool or list, optional): 是否显示数据点标记。
    sort (bool, optional): 是否在绘图前按x_var排序数据。默认为True。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.lineplot 的其他参数。
    """
    plt.figure(figsize=(10, 6))
    if sort:
        data_frame_sorted = data_frame.sort_values(by=x_var)
    else:
        data_frame_sorted = data_frame
    sns.lineplot(data=data_frame_sorted, x=x_var, y=y_var, hue=hue_var, style=style_var, markers=markers, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_var)
    plt.ylabel(ylabel if ylabel else y_var)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"折线图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_stripplot(data_frame, x_cat_var, y_num_var, hue_var=None, title='带状图/抖动图', xlabel=None, ylabel=None, jitter=True, dodge=False, save_path=None, **kwargs):
    """
    生成并（可选）保存带状图 (stripplot)。

    参数:
    data_frame (pd.DataFrame): 输入的数据框。
    x_cat_var (str): X轴对应的类别列名。
    y_num_var (str): Y轴对应的数值列名。
    hue_var (str, optional): 用于颜色编码的类别列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_cat_var。
    ylabel (str, optional): Y轴标签。默认为y_num_var。
    jitter (bool or float, optional): 是否添加抖动以避免点重叠。
    dodge (bool, optional): 当使用hue时，是否分离不同级别的点。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.stripplot 的其他参数。
    """
    plt.figure(figsize=(10, 7))
    sns.stripplot(data=data_frame, x=x_cat_var, y=y_num_var, hue=hue_var, jitter=jitter, dodge=dodge, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_cat_var)
    plt.ylabel(ylabel if ylabel else y_num_var)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"带状图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_swarmplot(data_frame, x_cat_var, y_num_var, hue_var=None, title='蜂群图', xlabel=None, ylabel=None, dodge=False, save_path=None, **kwargs):
    """
    生成并（可选）保存蜂群图 (swarmplot)。

    参数:
    data_frame (pd.DataFrame): 输入的数据框。
    x_cat_var (str): X轴对应的类别列名。
    y_num_var (str): Y轴对应的数值列名。
    hue_var (str, optional): 用于颜色编码的类别列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_cat_var。
    ylabel (str, optional): Y轴标签。默认为y_num_var。
    dodge (bool, optional): 当使用hue时，是否分离不同级别的点。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.swarmplot 的其他参数。
    """
    plt.figure(figsize=(10, 7))
    sns.swarmplot(data=data_frame, x=x_cat_var, y=y_num_var, hue=hue_var, dodge=dodge, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_cat_var)
    plt.ylabel(ylabel if ylabel else y_num_var)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"蜂群图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_violinplot(data_frame, x_cat_var, y_num_var, hue_var=None, title='小提琴图', xlabel=None, ylabel=None, split=False, inner="box", save_path=None, **kwargs):
    """
    生成并（可选）保存小提琴图 (violinplot)。

    参数:
    data_frame (pd.DataFrame): 输入的数据框。
    x_cat_var (str): X轴对应的类别列名。
    y_num_var (str): Y轴对应的数值列名。
    hue_var (str, optional): 用于颜色编码的类别列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_cat_var。
    ylabel (str, optional): Y轴标签。默认为y_num_var。
    split (bool, optional): 当hue有两个级别时，是否绘制拆分的小提琴图。
    inner (str, optional): 小提琴内部的表示 ('box', 'quartile', 'point', 'stick', None)。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.violinplot 的其他参数。
    """
    plt.figure(figsize=(10, 7))
    sns.violinplot(data=data_frame, x=x_cat_var, y=y_num_var, hue=hue_var, split=split, inner=inner, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_cat_var)
    plt.ylabel(ylabel if ylabel else y_num_var)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"小提琴图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_boxplot_seaborn(data_frame, x_cat_var, y_num_var, hue_var=None, title='箱线图 (Seaborn)', xlabel=None, ylabel=None, save_path=None, **kwargs):
    """
    使用Seaborn生成并（可选）保存箱线图。

    参数:
    data_frame (pd.DataFrame): 输入的数据框。
    x_cat_var (str): X轴对应的类别列名。
    y_num_var (str): Y轴对应的数值列名。
    hue_var (str, optional): 用于颜色编码的类别列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_cat_var。
    ylabel (str, optional): Y轴标签。默认为y_num_var。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.boxplot 的其他参数。
    """
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=data_frame, x=x_cat_var, y=y_num_var, hue=hue_var, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_cat_var)
    plt.ylabel(ylabel if ylabel else y_num_var)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"箱线图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_regplot(data_frame, x_var, y_var, title='回归模型图 (regplot)', xlabel=None, ylabel=None, scatter_kws=None, line_kws=None, save_path=None, **kwargs):
    """
    生成并（可选）保存回归模型图 (regplot)。

    参数:
    data_frame (pd.DataFrame): 输入的数据框。
    x_var (str): X轴对应的列名。
    y_var (str): Y轴对应的列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_var。
    ylabel (str, optional): Y轴标签。默认为y_var。
    scatter_kws (dict, optional): 传递给底层散点图的参数。
    line_kws (dict, optional): 传递给底层回归线的参数。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.regplot 的其他参数。
    """
    plt.figure(figsize=(8, 6))
    sns.regplot(data=data_frame, x=x_var, y=y_var, scatter_kws=scatter_kws, line_kws=line_kws, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_var)
    plt.ylabel(ylabel if ylabel else y_var)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"Regplot已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_heatmap(data_pivot_or_corr, title='热力图', annot=True, fmt=".2f", cmap="viridis", save_path=None, **kwargs):
    """
    生成并（可选）保存热力图，常用于显示透视表或相关矩阵。

    参数:
    data_pivot_or_corr (pd.DataFrame): 用于绘制热力图的二维数据 (例如，透视表或相关系数矩阵)。
    title (str, optional): 图表标题。
    annot (bool, optional): 是否在单元格中标注数值。
    fmt (str, optional): 标注数值的格式字符串。
    cmap (str or Colormap, optional): 颜色映射。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.heatmap 的其他参数。
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_pivot_or_corr, annot=annot, fmt=fmt, cmap=cmap, **kwargs)
    plt.title(title)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"热力图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_clustermap(data_corr, title='聚类热力图', annot=True, cmap='coolwarm', figsize=(8,8), save_path=None, **kwargs):
    """
    生成并（可选）保存聚类热力图。

    参数:
    data_corr (pd.DataFrame): 通常是相关系数矩阵或相似性矩阵。
    title (str, optional): 图表标题 (注意: clustermap的标题通常用fig.suptitle设置)。
    annot (bool, optional): 是否在单元格中标注数值。
    cmap (str or Colormap, optional): 颜色映射。
    figsize (tuple, optional): 图形大小。
    save_path (str, optional): 保存图表的路径。
    **kwargs: 传递给 sns.clustermap 的其他参数。
    """
    g = sns.clustermap(data_corr, annot=annot, cmap=cmap, figsize=figsize, **kwargs)
    g.fig.suptitle(title, y=1.02) # Clustermap标题设置方式略有不同
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"聚类热力图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================
def run_seaborn_demos():
    """运行所有Seaborn演示函数（使用接口化函数和示例数据）。"""
    print(f"--- Seaborn 可视化接口化演示 (图表将保存到 '{OUTPUT_DIR}' 目录) ---")

    # 加载Seaborn内置数据集作为演示数据
    tips = sns.load_dataset("tips")
    iris = sns.load_dataset("iris")
    flights = sns.load_dataset("flights")
    flights_pivot = flights.pivot_table(index="month", columns="year", values="passengers")
    
    # 自定义一些数据用于特定图表
    np.random.seed(101)
    demo_df = pd.DataFrame({
        '身高cm': np.random.normal(170, 10, 100),
        '体重kg': np.random.normal(65, 15, 100) + (np.random.normal(170, 10, 100) - 170) * 0.5, # 体重与身高轻微正相关
        '性别': np.random.choice(['男', '女'], 100, p=[0.55, 0.45]),
        '运动习惯': np.random.choice(['规律运动', '偶尔运动', '从不运动'], 100, p=[0.3, 0.4, 0.3])
    })

    print("\n--- 分布型图表 (Distribution Plots) ---")
    create_histplot(tips['total_bill'], title='总消费金额分布 (Histplot)', xlabel='总消费金额($)', save_path="demo_histplot_tips.png")
    create_kdeplot(tips, x_var='tip', hue_var='sex', title='小费金额核密度估计 (按性别)', xlabel='小费($)', save_path="demo_kdeplot_tips_sex.png")
    create_ecdfplot(iris, x_var='sepal_length', hue_var='species', title='鸢尾花萼片长度ECDF (按品种)', xlabel='萼片长度(cm)', save_path="demo_ecdfplot_iris_species.png")

    print("\n---关系型图表 (Relational Plots) ---")
    create_scatterplot(tips, x_var="total_bill", y_var="tip", hue_var="time", style_var="smoker", size_var="size", 
                       title='消费金额与小费关系 (按就餐时间/是否吸烟/人数)', xlabel='总消费金额($)', ylabel='小费($)',
                       sizes=(20, 200), alpha=0.7, save_path="demo_scatterplot_tips_complex.png")
    create_lineplot(flights, x_var="year", y_var="passengers", hue_var="month", title='航班乘客数量年度变化 (按月份)', 
                    xlabel='年份', ylabel='乘客数量', sort=False, # 年份本身有序，但月份作为hue，按年份聚合
                    estimator=np.sum, # 显示总乘客数而非平均
                    save_path="demo_lineplot_flights_monthly.png")

    print("\n--- 类别型图表 (Categorical Plots) ---")
    create_stripplot(tips, x_cat_var="day", y_num_var="total_bill", hue_var="sex", 
                     title='每日消费金额带状图 (按性别)', xlabel='星期', ylabel='总消费金额($)', 
                     save_path="demo_stripplot_tips_day_sex.png")
    create_swarmplot(tips, x_cat_var="day", y_num_var="tip", hue_var="time", 
                     title='每日小费蜂群图 (按就餐时间)', xlabel='星期', ylabel='小费($)', dodge=True,
                     save_path="demo_swarmplot_tips_day_time.png")
    create_violinplot(demo_df, x_cat_var="运动习惯", y_num_var="身高cm", hue_var="性别", 
                      title='身高按运动习惯和性别的小提琴图', xlabel='运动习惯', ylabel='身高(cm)', 
                      split=True, inner='quartile', save_path="demo_violinplot_height_exercise_gender.png")
    create_boxplot_seaborn(tips, x_cat_var="time", y_num_var="total_bill", hue_var="smoker", 
                           title='就餐时间消费金额箱线图 (按是否吸烟)', xlabel='就餐时间', ylabel='总消费金额($)',
                           save_path="demo_boxplot_tips_time_smoker.png")

    print("\n--- 回归模型可视化 (Regression Model Visualizations) ---")
    create_regplot(tips, x_var="total_bill", y_var="tip", 
                   title='消费金额与小费的线性回归模型图', xlabel='总消费金额($)', ylabel='小费($)',
                   scatter_kws={'s': 50, 'alpha':0.5}, line_kws={'color':'red'},
                   save_path="demo_regplot_tips.png")

    print("\n--- 矩阵型图表 (Matrix Plots) ---")
    # 对于热力图，flights_pivot已经准备好了
    create_heatmap(flights_pivot, title='航班乘客数量热力图 (月份 vs 年份)', annot=True, fmt="d", cmap="BuPu",
                   save_path="demo_heatmap_flights.png")
    
    # 对于聚类热力图，使用iris数据集的相关系数矩阵
    iris_numeric_cols = iris.select_dtypes(include=np.number)
    iris_corr = iris_numeric_cols.corr()
    create_clustermap(iris_corr, title='鸢尾花数值特征相关性聚类热力图', annot=True, fmt=".2f", cmap="viridis",
                      save_path="demo_clustermap_iris_corr.png")

    print(f"--- Seaborn 可视化接口化演示完成 ---")

if __name__ == '__main__':
    run_seaborn_demos() 