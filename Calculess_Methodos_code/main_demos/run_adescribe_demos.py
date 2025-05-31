import pandas as pd
import numpy as np
import geopandas
from shapely.geometry import Point, Polygon

# Import ADescribe module functions
from ADescribe.pandas_operations import (
    create_dataframe, read_csv_data, write_csv_data, 
    get_descriptive_statistics, get_dataframe_info, get_value_counts,
    group_and_aggregate, create_pivot_table, create_crosstab,
    create_example_dataframe_for_demo as create_pandas_sample_df # alias to avoid clash
)
from ADescribe.matplotlib_visualizations import (
    plot_line_chart_api, plot_bar_chart_api, plot_histogram_api,
    plot_scatter_plot_api, plot_pie_chart_api, plot_boxplot_api,
    plot_heatmap_api, plot_multiple_subplots_api,
    create_matplotlib_sample_data
)
from ADescribe.seaborn_visualizations import (
    plot_seaborn_relational_api, plot_seaborn_categorical_api,
    plot_seaborn_distribution_api, plot_seaborn_regression_api,
    plot_seaborn_matrix_api, plot_seaborn_facetgrid_api,
    plot_seaborn_pairplot_api, plot_seaborn_jointplot_api,
    create_seaborn_sample_data
)
from ADescribe.plotly_visualizations import (
    create_plotly_scatter_api, create_plotly_line_api, create_plotly_bar_api,
    create_plotly_histogram_api, create_plotly_boxplot_api, create_plotly_heatmap_api,
    create_plotly_3d_scatter_api, create_plotly_choropleth_map_api,
    create_plotly_sample_data, create_plotly_geo_sample_data
)
from ADescribe.bokeh_visualizations import (
    create_bokeh_line_plot_api, create_bokeh_scatter_plot_api, create_bokeh_bar_chart_api,
    create_bokeh_histogram_api, create_bokeh_boxplot_api, create_bokeh_heatmap_api,
    create_bokeh_linked_plots_api, create_bokeh_hover_tools_example_api,
    create_bokeh_sample_data
)
from ADescribe.altair_visualizations import (
    create_altair_scatter_plot_api, create_altair_bar_chart_api, create_altair_line_chart_api,
    create_altair_histogram_api, create_altair_area_chart_api, create_altair_interactive_chart_api,
    create_altair_sample_data
)
from ADescribe.geospatial_visualizations import (
    create_geopandas_dataframe_api, plot_geopandas_basic_api,
    plot_geopandas_choropleth_api, create_folium_map_api,
    add_markers_to_folium_map_api, add_polygons_to_folium_map_api,
    load_sample_geospatial_data_api, create_sample_points_for_folium_api
)
from ADescribe.network_analysis_visualizations import (
    create_networkx_graph_api, draw_networkx_graph_api,
    calculate_network_metrics_api, visualize_node_attributes_api,
    visualize_edge_attributes_api, create_sample_edge_list_api,
    generate_example_graph_api
)
from ADescribe.text_data_analysis import (
    preprocess_text_nltk_api, preprocess_text_spacy_api,
    generate_wordcloud_api, vectorize_text_sklearn_api,
    get_sample_text_data_api
)
from ADescribe.dimensionality_reduction_visualizations import (
    perform_pca_sklearn_api, visualize_pca_results_api,
    perform_tsne_sklearn_api, visualize_tsne_results_api,
    create_dr_sample_data_api
)
from ADescribe.ydata_profiling_example import generate_pandas_profiling_report_api, create_profiling_sample_df_api
from ADescribe.sweetviz_example import generate_sweetviz_report_api, compare_dataframes_sweetviz_api
from ADescribe.lux_conceptual_example import explain_lux_philosophy, explain_lux_typical_workflow, explain_lux_limitations_and_considerations


def run_all_pandas_demos():
    """运行所有Pandas演示函数（使用接口化函数和示例数据）。"""
    print("--- Pandas 核心操作接口化演示 ---")
    
    # 1. 创建DataFrame示例
    print("\n--- 1. DataFrame 创建与基本信息 ---")
    sample_data_dict = {
        '姓名': ['张三', '李四', '王五', '赵六'],
        '年龄': [23, 34, 28, 45],
        '城市': ['北京', '上海', '深圳', '广州'],
        '薪资': [15000, 22000, 18000, 25000]
    }
    my_df = create_dataframe(sample_data_dict)
    print("创建的DataFrame:\n", my_df)
    get_dataframe_info(my_df)

    # 2. 描述性统计示例
    print("\n--- 2. 描述性统计 --- ")
    desc_stats_all = get_descriptive_statistics(my_df)
    print("\n所有列的描述性统计:\n", desc_stats_all)
    desc_stats_numeric = get_descriptive_statistics(my_df, include_types=['number'])
    print("\n数值列的描述性统计:\n", desc_stats_numeric)
    desc_stats_age = get_descriptive_statistics(my_df, specific_columns=['年龄'], percentiles=[0.1, 0.5, 0.9])
    print("\n'年龄'列的描述性统计 (指定百分位数):\n", desc_stats_age)

    # 3. 值计数示例
    print("\n--- 3. 值计数 --- ")
    if '城市' in my_df.columns:
        city_counts = get_value_counts(my_df['城市'])
        print("\n'城市'列的值计数:\n", city_counts)
        city_freq = get_value_counts(my_df['城市'], normalize=True)
        print("\n'城市'列的相对频率:\n", city_freq)

    # 4. 分组与聚合示例
    print("\n--- 4. 分组与聚合 --- ")
    demo_df_pd = create_pandas_sample_df() # Using alias
    print("用于分组聚合的演示数据:\n", demo_df_pd.head())
    
    agg_rules_major_gender = {
        '平均成绩': ('Score', 'mean'),
        '最高出勤率': ('Attendance', 'max'),
        '总学习小时数': ('StudyHours', 'sum'),
        '学生人数': ('StudentID', 'count')
    }
    grouped_data = group_and_aggregate(demo_df_pd, group_by_columns=['Major', 'Gender'], aggregations=agg_rules_major_gender)
    print("\n按'专业'和'性别'分组聚合的结果:\n", grouped_data)

    agg_rules_major = {
        '中位成绩': ('Score', 'median')
    }
    grouped_major_median_score = group_and_aggregate(demo_df_pd, group_by_columns='Major', aggregations=agg_rules_major)
    print("\n按'专业'分组计算成绩中位数:\n", grouped_major_median_score)

    # 5. 透视表示例
    print("\n--- 5. 透视表 --- ")
    pivot_score_by_major_gender = create_pivot_table(demo_df_pd, values='Score', index='Major', columns='Gender', aggfunc='mean', margins=True)
    print("\n按'专业'和'性别'统计的平均成绩透视表:\n", pivot_score_by_major_gender)

    pivot_studyhours_by_date_major = create_pivot_table(demo_df_pd, 
                                                       values='StudyHours', 
                                                       index=demo_df_pd['EnrollDate'].dt.month_name(), 
                                                       columns='Major', 
                                                       aggfunc='sum',
                                                       fill_value=0)
    print("\n按'入学月份'和'专业'统计的总学习小时透视表:\n", pivot_studyhours_by_date_major)

    # 6. 交叉表示例
    print("\n--- 6. 交叉表 --- ")
    crosstab_major_gender = create_crosstab(index=demo_df_pd['Major'], columns=demo_df_pd['Gender'], margins=True)
    print("\n'专业'与'性别'的交叉表:\n", crosstab_major_gender)
    
    crosstab_major_gender_scores = create_crosstab(index=demo_df_pd['Major'], columns=demo_df_pd['Gender'], 
                                                  values=demo_df_pd['Score'], aggfunc='mean', normalize='index')
    print("\n'专业'与'性别'的平均成绩交叉表 (按行归一化):\n", crosstab_major_gender_scores)
    
    # 7. 读写CSV示例 (注意：这些操作会实际创建文件)
    print("\n--- 7. CSV文件读写 (示例，将创建临时文件) ---")
    temp_csv_path = "temp_demo_data.csv"
    write_csv_data(my_df, temp_csv_path)
    reloaded_df = read_csv_data(temp_csv_path)
    print(f"从 {temp_csv_path} 重新加载的DataFrame (前5行):\n", reloaded_df.head())
    # 清理临时文件 (可选)
    try:
        import os
        os.remove(temp_csv_path)
        print(f"临时文件 {temp_csv_path} 已删除。")
    except OSError as e:
        print(f"删除临时文件 {temp_csv_path} 失败: {e}")

    print("\n--- Pandas 核心操作接口化演示结束 ---\n")


def demo_matplotlib_visuals(user_df_dict=None):
    """演示 ADescribe.matplotlib_visualizations 中的核心可视化功能。"""
    print("\n--- Matplotlib 可视化接口化演示 ---")
    
    # 准备数据
    if user_df_dict:
        df_mpl = pd.DataFrame(user_df_dict)
        print("使用用户提供的数据进行Matplotlib演示。")
    else:
        df_mpl = create_matplotlib_sample_data(n_samples=100, random_seed=42)
        print("使用自动生成的样本数据进行Matplotlib演示。")

    x_col = 'X_Value'
    y_col_line = 'Y_Value_Line'
    y_col_bar = 'Y_Value_Bar'
    hist_col = 'Distribution_Value'
    scatter_y_col = 'Y_Value_Scatter'
    pie_labels_col = 'Category_Labels' # 假设df_mpl中已有或创建此列
    pie_values_col = 'Category_Values'
    boxplot_cols = ['Box_Group1', 'Box_Group2', 'Box_Group3'] # 假设已有或创建
    
    # 为饼图和箱线图准备/检查数据
    if pie_labels_col not in df_mpl.columns or pie_values_col not in df_mpl.columns:
        df_mpl[pie_labels_col] = [f'Cat{i}' for i in range(5)] * (len(df_mpl)//5 +1)
        df_mpl[pie_labels_col] = df_mpl[pie_labels_col][:len(df_mpl)]
        df_mpl[pie_values_col] = np.random.randint(10, 50, len(df_mpl))

    for col in boxplot_cols:
        if col not in df_mpl.columns:
            df_mpl[col] = np.random.rand(len(df_mpl)) * 100

    # 1. 线图
    print("\n1. 绘制线图...")
    plot_line_chart_api(df_mpl, x_col=x_col, y_cols=[y_col_line, scatter_y_col], 
                        title='示例线图 (API)', xlabel='X轴', ylabel='Y轴',
                        save_path='matplotlib_outputs/line_chart_api.png')

    # 2. 条形图
    print("\n2. 绘制条形图...")
    # 为条形图准备类别数据，如果x_col是连续的
    bar_x_labels = [f'Item {i}' for i in range(min(10, len(df_mpl)))] # 取前10个或全部
    bar_y_values = df_mpl[y_col_bar].head(len(bar_x_labels)).abs() # 取对应数量的值
    
    plot_bar_chart_api(categories=bar_x_labels, values=bar_y_values, 
                       title='示例条形图 (API)', xlabel='类别', ylabel='值',
                       save_path='matplotlib_outputs/bar_chart_api.png', color='skyblue', orientation='vertical')

    # 3. 直方图
    print("\n3. 绘制直方图...")
    plot_histogram_api(df_mpl, column_name=hist_col, bins=15, 
                         title='示例直方图 (API)', xlabel='值', ylabel='频率',
                         save_path='matplotlib_outputs/histogram_api.png', color='lightgreen')

    # 4. 散点图
    print("\n4. 绘制散点图...")
    plot_scatter_plot_api(df_mpl, x_col=x_col, y_col=scatter_y_col, 
                          title='示例散点图 (API)', xlabel='X轴', ylabel='Y轴',
                          save_path='matplotlib_outputs/scatter_plot_api.png', color='red', marker='o', s=30)

    # 5. 饼图
    print("\n5. 绘制饼图...")
    # 准备饼图数据 (汇总)
    pie_data_agg = df_mpl.groupby(pie_labels_col)[pie_values_col].sum()
    plot_pie_chart_api(sizes=pie_data_agg.values, labels=pie_data_agg.index, 
                       title='示例饼图 (API)', autopct='%1.1f%%',
                       save_path='matplotlib_outputs/pie_chart_api.png')

    # 6. 箱线图
    print("\n6. 绘制箱线图...")
    plot_boxplot_api(df_mpl, columns=boxplot_cols, 
                     title='示例箱线图 (API)', notch=True,
                     save_path='matplotlib_outputs/boxplot_api.png')

    # 7. 热力图
    print("\n7. 绘制热力图...")
    # 准备热力图数据 (通常是相关性矩阵或类似数据)
    heatmap_data = df_mpl[[x_col, y_col_line, hist_col, scatter_y_col] + boxplot_cols].corr()
    plot_heatmap_api(heatmap_data, title='示例相关性热力图 (API)', cmap='viridis',
                     save_path='matplotlib_outputs/heatmap_api.png', show_annot=True)
    
    # 8. 多子图
    print("\n8. 绘制多子图...")
    # 为多子图准备数据和绘图函数列表
    # 确保函数签名与 plot_multiple_subplots_api 期望的一致
    def plot_func1(ax, data): # 示例绘图函数1 (线图)
        ax.plot(data[x_col], data[y_col_line], label='子图线图')
        ax.set_title('子图1: 线图')
        ax.legend()

    def plot_func2(ax, data): # 示例绘图函数2 (直方图)
        ax.hist(data[hist_col], bins=10, color='orange', alpha=0.7)
        ax.set_title('子图2: 直方图')

    plot_functions_with_data = [
        (plot_func1, df_mpl),
        (plot_func2, df_mpl)
    ]
    plot_multiple_subplots_api(plot_functions_with_data, nrows=1, ncols=2, 
                               main_title='示例多子图 (API)',
                               save_path='matplotlib_outputs/multiple_subplots_api.png')

    print("\n--- Matplotlib 可视化接口化演示结束 ---\n")


def demo_seaborn_visuals(user_df=None):
    """演示 ADescribe.seaborn_visualizations 中的核心可视化功能。"""
    print("\n--- Seaborn 可视化接口化演示 ---")

    if user_df is not None:
        df_sns = user_df
        print("使用用户提供的数据进行Seaborn演示。")
    else:
        df_sns = create_seaborn_sample_data(n_samples=150, random_seed=123)
        print("使用自动生成的样本数据进行Seaborn演示。")

    # 通用列名 (确保这些列存在于 df_sns 中)
    x_numeric = 'numeric_x'
    y_numeric = 'numeric_y'
    hue_categorical = 'category_hue'
    size_numeric = 'numeric_size'
    style_categorical = 'category_style' # 可能与hue_categorical相同或不同
    cat_x = 'category_x' # 用于类别图的X轴
    dist_col = 'distribution_data'
    
    # 确保必要的列存在
    if hue_categorical not in df_sns: df_sns[hue_categorical] = np.random.choice(['A','B','C'], size=len(df_sns))
    if cat_x not in df_sns: df_sns[cat_x] = np.random.choice(['X1','X2','X3', 'X4'], size=len(df_sns))
    if dist_col not in df_sns: df_sns[dist_col] = np.random.randn(len(df_sns)) * 10 + 50
    if style_categorical not in df_sns: df_sns[style_categorical] = np.random.choice(['S1','S2'], size=len(df_sns))
    if size_numeric not in df_sns: df_sns[size_numeric] = np.random.rand(len(df_sns)) * 100 + 20


    # 1. 关系图 (Relational Plots)
    print("\n1. 绘制Seaborn关系图 (散点图)...")
    plot_seaborn_relational_api(df_sns, x=x_numeric, y=y_numeric, hue=hue_categorical, 
                                kind='scatter', title='Seaborn 散点图 (API)',
                                xlabel='数值特征X', ylabel='数值特征Y',
                                save_path='seaborn_outputs/relplot_scatter_api.png',
                                size=size_numeric, style=style_categorical)
    
    print("\n1.1 绘制Seaborn关系图 (线图)...")
    # 线图通常需要排序后的X轴以获得更清晰的视觉效果
    df_sns_sorted_line = df_sns.sort_values(by=x_numeric)
    plot_seaborn_relational_api(df_sns_sorted_line, x=x_numeric, y=y_numeric, hue=hue_categorical,
                                kind='line', title='Seaborn 线图 (API)',
                                save_path='seaborn_outputs/relplot_line_api.png', style=style_categorical, markers=True)

    # 2. 类别图 (Categorical Plots)
    print("\n2. 绘制Seaborn类别图 (箱线图)...")
    plot_seaborn_categorical_api(df_sns, x=cat_x, y=y_numeric, hue=hue_categorical,
                                 kind='box', title='Seaborn 箱线图 (API)',
                                 save_path='seaborn_outputs/catplot_box_api.png')
    
    print("\n2.1 绘制Seaborn类别图 (小提琴图)...")
    plot_seaborn_categorical_api(df_sns, x=cat_x, y=y_numeric, hue=hue_categorical,
                                 kind='violin', title='Seaborn 小提琴图 (API)',
                                 save_path='seaborn_outputs/catplot_violin_api.png', split=True) # split 仅对二分类hue有效

    print("\n2.2 绘制Seaborn类别图 (条形图)...")
    plot_seaborn_categorical_api(df_sns, x=cat_x, y=y_numeric, hue=hue_categorical,
                                 kind='bar', title='Seaborn 条形图 (API)',
                                 save_path='seaborn_outputs/catplot_bar_api.png', dodge=True)
                                 
    # 3. 分布图 (Distribution Plots)
    print("\n3. 绘制Seaborn分布图 (直方图/KDE)...")
    plot_seaborn_distribution_api(df_sns, x=dist_col, hue=hue_categorical,
                                  kind='hist', kde=True, title='Seaborn 直方图与KDE (API)',
                                  save_path='seaborn_outputs/displot_hist_kde_api.png')

    print("\n3.1 绘制Seaborn分布图 (ECDF)...")
    plot_seaborn_distribution_api(df_sns, x=dist_col, hue=hue_categorical,
                                  kind='ecdf', title='Seaborn ECDF图 (API)',
                                  save_path='seaborn_outputs/displot_ecdf_api.png')

    # 4. 回归图 (Regression Plots) - lmplot
    print("\n4. 绘制Seaborn回归图 (lmplot)...")
    plot_seaborn_regression_api(df_sns, x=x_numeric, y=y_numeric, hue=hue_categorical,
                                title='Seaborn 回归图 (lmplot) (API)',
                                save_path='seaborn_outputs/regplot_lmplot_api.png',
                                col=style_categorical if style_categorical in df_sns.columns else None, # Facet by style
                                fit_reg=True, scatter_kws={'alpha':0.5})

    # 5. 矩阵图 (Matrix Plots) - heatmap
    print("\n5. 绘制Seaborn矩阵图 (热力图)...")
    # 热力图通常用于显示相关性矩阵等
    numeric_cols_for_corr = df_sns.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols_for_corr) >=2:
        corr_matrix = df_sns[numeric_cols_for_corr].corr()
        plot_seaborn_matrix_api(corr_matrix, kind='heatmap', annot=True, cmap='coolwarm',
                                title='Seaborn 相关性热力图 (API)',
                                save_path='seaborn_outputs/matrix_heatmap_api.png')
    else:
        print("  跳过热力图：数值列不足2个。")


    # 6. FacetGrid (使用relplot, displot, catplot的高级接口)
    print("\n6. 绘制Seaborn FacetGrid (通过relplot)...")
    # FacetGrid本身是一个对象，relplot/displot/catplot是更高级的接口来创建它们
    # 这里用relplot的col参数来演示分面
    plot_seaborn_facetgrid_api(df_sns, x=x_numeric, y=y_numeric, 
                               col=hue_categorical, # 分面依据
                               plot_kind='scatter', # 底层绘图类型
                               title_prefix='FacetGrid (Scatter) by ',
                               save_path='seaborn_outputs/facetgrid_relplot_api.png',
                               sharex=False, sharey=False)
    
    # 7. PairGrid / pairplot
    print("\n7. 绘制Seaborn Pairplot...")
    cols_for_pairplot = [x_numeric, y_numeric, dist_col] # 选择一些数值列
    if hue_categorical in df_sns.columns:
        cols_for_pairplot_safe = [col for col in cols_for_pairplot if col in df_sns.columns]
        if len(cols_for_pairplot_safe) >= 2:
             plot_seaborn_pairplot_api(df_sns, vars_list=cols_for_pairplot_safe, hue=hue_categorical,
                                       title_prefix='Seaborn Pairplot by ',
                                       save_path='seaborn_outputs/pairplot_api.png',
                                       diag_kind='kde', kind='scatter')
        else:
            print("  跳过Pairplot：合适的数值列不足2个。")
    else:
        print("  跳过Pairplot：hue列不存在。")


    # 8. JointGrid / jointplot
    print("\n8. 绘制Seaborn Jointplot...")
    if x_numeric in df_sns.columns and y_numeric in df_sns.columns:
        plot_seaborn_jointplot_api(df_sns, x=x_numeric, y=y_numeric, hue=hue_categorical if hue_categorical in df_sns.columns else None,
                                   kind='scatter', # 'scatter', 'kde', 'hist', 'reg'
                                   title_prefix=f'Seaborn Jointplot of {x_numeric} and {y_numeric} by ',
                                   save_path='seaborn_outputs/jointplot_api.png',
                                   marginal_ticks=True, color='skyblue', marginal_kws=dict(bins=15, fill=True))
    else:
        print(f"  跳过Jointplot：列 {x_numeric} 或 {y_numeric} 不存在。")

    print("\n--- Seaborn 可视化接口化演示结束 ---\n")

def demo_plotly_visuals(user_df=None):
    """演示 ADescribe.plotly_visualizations 中的核心可视化功能。"""
    print("\n--- Plotly 可视化接口化演示 ---")

    if user_df is not None:
        df_plotly = user_df
        print("使用用户提供的数据进行Plotly演示。")
    else:
        df_plotly = create_plotly_sample_data(n_samples=100, random_seed=777)
        geo_df_plotly = create_plotly_geo_sample_data()
        print("使用自动生成的样本数据进行Plotly演示。")

    # 列名 (确保这些列存在于 df_plotly 中)
    x_col = 'feature_x'
    y_col = 'feature_y'
    z_col = 'feature_z' # 用于3D图
    color_col = 'category_color'
    size_col = 'size_magnitude'
    text_col = 'hover_text' # 悬停文本
    hist_col = 'distribution_data'
    box_col_prefix = 'boxplot_group_' # 用于箱线图的多列

    # 确保基础列存在
    if color_col not in df_plotly: df_plotly[color_col] = np.random.choice(['TypeA','TypeB','TypeC'], size=len(df_plotly))
    if size_col not in df_plotly: df_plotly[size_col] = np.random.rand(len(df_plotly)) * 50 + 10
    if text_col not in df_plotly: df_plotly[text_col] = [f"Point_{i}" for i in range(len(df_plotly))]
    if hist_col not in df_plotly: df_plotly[hist_col] = np.random.randn(len(df_plotly)) * 15 + 60
    if z_col not in df_plotly: df_plotly[z_col] = np.random.rand(len(df_plotly)) * 100

    # 1. 散点图 (Scatter Plot)
    print("\n1. 创建Plotly散点图...")
    fig_scatter = create_plotly_scatter_api(df_plotly, x_col=x_col, y_col=y_col, 
                                            color_col=color_col, size_col=size_col, text_col=text_col,
                                            title='Plotly 散点图 (API)', labels={'feature_x':'X轴标签', 'feature_y':'Y轴标签'})
    fig_scatter.write_html("plotly_outputs/scatter_plot_api.html")
    print("  散点图已保存为 scatter_plot_api.html")

    # 2. 线图 (Line Plot)
    print("\n2. 创建Plotly线图...")
    # 线图通常需要X轴排序
    df_plotly_sorted = df_plotly.sort_values(by=x_col)
    fig_line = create_plotly_line_api(df_plotly_sorted, x_col=x_col, y_cols=[y_col, z_col], # 可以绘制多条线
                                      color_col=None, # 或者为每条线指定颜色
                                      title='Plotly 线图 (API)', labels={'value':'Y轴值', 'variable':'图例'})
    fig_line.write_html("plotly_outputs/line_plot_api.html")
    print("  线图已保存为 line_plot_api.html")

    # 3. 条形图 (Bar Chart)
    print("\n3. 创建Plotly条形图...")
    # 条形图通常使用类别X轴和数值Y轴
    bar_df = df_plotly.groupby(color_col)[y_col].mean().reset_index()
    fig_bar = create_plotly_bar_api(bar_df, x_col=color_col, y_cols=[y_col],
                                    color_col=None, barmode='group',
                                    title='Plotly 条形图 (按类别均值) (API)')
    fig_bar.write_html("plotly_outputs/bar_chart_api.html")
    print("  条形图已保存为 bar_chart_api.html")

    # 4. 直方图 (Histogram)
    print("\n4. 创建Plotly直方图...")
    fig_hist = create_plotly_histogram_api(df_plotly, x_col=hist_col, color_col=color_col,
                                           title='Plotly 直方图 (API)', nbins=20, marginal='rug') # marginal可以是 'rug', 'box', 'violin'
    fig_hist.write_html("plotly_outputs/histogram_api.html")
    print("  直方图已保存为 histogram_api.html")

    # 5. 箱线图 (Box Plot)
    print("\n5. 创建Plotly箱线图...")
    # 为箱线图创建几组数据
    for i in range(1,4):
        if f'{box_col_prefix}{i}' not in df_plotly:
            df_plotly[f'{box_col_prefix}{i}'] = np.random.normal(loc=i*10, scale=5, size=len(df_plotly))
            
    box_cols_to_plot = [col for col in df_plotly.columns if col.startswith(box_col_prefix)]
    fig_box = create_plotly_boxplot_api(df_plotly, y_cols=box_cols_to_plot, # 可以传入多列，或一列并用x_col分组
                                        # x_col=color_col, y_cols=[hist_col], # 另一种用法：按类别分组显示单一数值列的箱线图
                                        title='Plotly 箱线图 (API)', points='all') # points: 'all', 'outliers', 'suspectedoutliers', False
    fig_box.write_html("plotly_outputs/boxplot_api.html")
    print("  箱线图已保存为 boxplot_api.html")

    # 6. 热力图 (Heatmap)
    print("\n6. 创建Plotly热力图...")
    numeric_cols_for_corr_plotly = df_plotly.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols_for_corr_plotly) >=2:
        corr_matrix_plotly = df_plotly[numeric_cols_for_corr_plotly].corr()
        fig_heatmap = create_plotly_heatmap_api(z_data=corr_matrix_plotly.values, 
                                                x_labels=corr_matrix_plotly.columns.tolist(), 
                                                y_labels=corr_matrix_plotly.index.tolist(),
                                                title='Plotly 相关性热力图 (API)', colorscale='Viridis')
        fig_heatmap.write_html("plotly_outputs/heatmap_api.html")
        print("  热力图已保存为 heatmap_api.html")
    else:
        print("  跳过Plotly热力图：数值列不足2个。")


    # 7. 3D散点图 (3D Scatter Plot)
    print("\n7. 创建Plotly 3D散点图...")
    fig_3d_scatter = create_plotly_3d_scatter_api(df_plotly, x_col=x_col, y_col=y_col, z_col=z_col,
                                                  color_col=color_col, size_col=size_col,
                                                  title='Plotly 3D散点图 (API)')
    fig_3d_scatter.write_html("plotly_outputs/3d_scatter_plot_api.html")
    print("  3D散点图已保存为 3d_scatter_plot_api.html")

    # 8. 地理空间图 (Choropleth Map) - 需要地理数据
    print("\n8. 创建Plotly Choropleth地图...")
    if 'geo_df_plotly' in locals(): # 检查地理数据是否已创建
        fig_choropleth = create_plotly_choropleth_map_api(geo_df_plotly, locations_col='iso_alpha', 
                                                          color_col='value', hover_name_col='country',
                                                          title='Plotly Choropleth 地图示例 (API)', 
                                                          projection_type='natural earth',
                                                          color_continuous_scale='Plasma')
        fig_choropleth.write_html("plotly_outputs/choropleth_map_api.html")
        print("  Choropleth地图已保存为 choropleth_map_api.html")
    else:
        print("  跳过Choropleth地图：未找到地理样本数据。")

    print("\n--- Plotly 可视化接口化演示结束 ---\n")

def demo_bokeh_visuals(user_df=None):
    """演示 ADescribe.bokeh_visualizations 中的核心可视化功能。"""
    print("\n--- Bokeh 可视化接口化演示 ---")

    if user_df is not None:
        df_bokeh = user_df
        print("使用用户提供的数据进行Bokeh演示。")
    else:
        df_bokeh = create_bokeh_sample_data(n_points=100, random_seed=888)
        print("使用自动生成的样本数据进行Bokeh演示。")
    
    # 列名
    x_col = 'x_values'
    y_col1 = 'y_values_A'
    y_col2 = 'y_values_B'
    cat_col = 'categories' # 用于条形图或颜色编码
    val_col = 'numeric_values' # 用于直方图、箱线图
    
    # 确保基础列存在
    if cat_col not in df_bokeh: df_bokeh[cat_col] = np.random.choice(['P','Q','R','S'], size=len(df_bokeh))
    if val_col not in df_bokeh: df_bokeh[val_col] = np.random.randn(len(df_bokeh)) * 20 + 70


    # 1. 线图 (Line Plot)
    print("\n1. 创建Bokeh线图...")
    p_line = create_bokeh_line_plot_api(df_bokeh, x_col=x_col, y_cols=[y_col1, y_col2],
                                        title='Bokeh 线图 (API)', xlabel='X轴', ylabel='Y轴',
                                        legend_labels=['系列A', '系列B'], colors=['blue', 'green'])
    # bokeh.io.save(p_line, filename="bokeh_outputs/line_plot_api.html", title="Bokeh Line Plot")
    # 由于接口函数内部处理保存，这里仅作演示说明
    print("  线图通常由 create_bokeh_line_plot_api 内部保存/返回。")


    # 2. 散点图 (Scatter Plot)
    print("\n2. 创建Bokeh散点图...")
    p_scatter = create_bokeh_scatter_plot_api(df_bokeh, x_col=x_col, y_col=y_col1,
                                              category_col=cat_col, # 用于颜色编码
                                              title='Bokeh 散点图 (API)', xlabel='X轴', ylabel='Y轴 A',
                                              legend_group_name='类型')
    print("  散点图通常由 create_bokeh_scatter_plot_api 内部保存/返回。")

    # 3. 条形图 (Bar Chart)
    print("\n3. 创建Bokeh条形图...")
    # 条形图通常需要类别X轴和数值Y轴 (或反之，如果水平)
    # 聚合数据以用于条形图
    bar_data_bokeh = df_bokeh.groupby(cat_col)[val_col].mean().reset_index()
    p_bar = create_bokeh_bar_chart_api(bar_data_bokeh, categories_col=cat_col, values_col=val_col,
                                       title='Bokeh 条形图 (按类别均值) (API)', xlabel='类别', ylabel='平均值',
                                       legend_label='平均值', color='purple')
    print("  条形图通常由 create_bokeh_bar_chart_api 内部保存/返回。")
    
    # 4. 直方图 (Histogram)
    print("\n4. 创建Bokeh直方图...")
    p_hist = create_bokeh_histogram_api(df_bokeh, column_to_bin=val_col, bins=10,
                                        title='Bokeh 直方图 (API)', xlabel='数值', ylabel='频数',
                                        fill_color='orange', line_color='black')
    print("  直方图通常由 create_bokeh_histogram_api 内部保存/返回。")

    # 5. 箱线图 (Box Plot)
    print("\n5. 创建Bokeh箱线图...")
    # Bokeh的箱线图通常需要按类别对数值数据进行分组
    p_box = create_bokeh_boxplot_api(df_bokeh, category_col=cat_col, value_col=val_col,
                                     title='Bokeh 箱线图 (按类别) (API)', xlabel='类别', ylabel='值')
    print("  箱线图通常由 create_bokeh_boxplot_api 内部保存/返回。")

    # 6. 热力图 (Heatmap)
    print("\n6. 创建Bokeh热力图...")
    # Bokeh热力图需要特定格式的数据
    # 示例：创建一个简单的相关性矩阵用于热力图
    numeric_cols_bokeh = df_bokeh.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols_bokeh) >= 2:
        corr_matrix_bokeh = df_bokeh[numeric_cols_bokeh].corr()
        # 将corr_matrix转换为Bokeh热力图所需的格式 (通常是堆叠的DataFrame)
        corr_df_stacked = corr_matrix_bokeh.stack().reset_index()
        corr_df_stacked.columns = ['x_name', 'y_name', 'rate']
        
        p_heatmap = create_bokeh_heatmap_api(corr_df_stacked, x_col='x_name', y_col='y_name', value_col='rate',
                                             palette_name='Inferno256', # Bokeh调色板名称
                                             title='Bokeh 相关性热力图 (API)',
                                             xlabel='特征1', ylabel='特征2')
        print("  热力图通常由 create_bokeh_heatmap_api 内部保存/返回。")

    else:
        print("  跳过Bokeh热力图：数值列不足2个。")

    # 7. 联动图 (Linked Plots) - 通常包含多个图
    print("\n7. 创建Bokeh联动图...")
    # 这个API函数会创建并返回一个包含联动图的布局对象
    p_linked = create_bokeh_linked_plots_api(df_bokeh, x_col_scatter=x_col, y_col_scatter=y_col1,
                                             x_col_line=x_col, y_col_line=y_col2,
                                             common_tool_types=['pan', 'wheel_zoom', 'box_select', 'reset', 'save'],
                                             scatter_title='联动散点图', line_title='联动线图')
    print("  联动图布局通常由 create_bokeh_linked_plots_api 内部保存/返回。")

    # 8. 悬停工具示例 (Hover Tools)
    print("\n8. 创建带高级悬停工具的Bokeh图...")
    custom_tooltips = [
        ("索引", "@index"),
        (f"({x_col}, {y_col1})", "(@{"+x_col+"}, @{"+y_col1+"})"), # 引用列名
        ("类别", f"@{cat_col}"),
        ("额外信息", "固定文本或特定值")
    ]
    p_hover = create_bokeh_hover_tools_example_api(df_bokeh, x_col=x_col, y_col=y_col1,
                                                   tooltips_config=custom_tooltips,
                                                   title='Bokeh 高级悬停工具示例 (API)',
                                                   color_by_col=cat_col) # 可选颜色编码
    print("  带悬停工具的图通常由 create_bokeh_hover_tools_example_api 内部保存/返回。")

    print("\n--- Bokeh 可视化接口化演示结束 ---
")


def demo_altair_visuals(user_df=None):
    """演示 ADescribe.altair_visualizations 中的核心可视化功能。"""
    print("\n--- Altair 可视化接口化演示 ---")

    if user_df is not None:
        df_altair = user_df
        print("使用用户提供的数据进行Altair演示。")
    else:
        df_altair = create_altair_sample_data(n_rows=100, random_seed=999)
        print("使用自动生成的样本数据进行Altair演示。")

    # 列名 (确保这些列存在于 df_altair 中)
    x_quant = 'quantitative_x'
    y_quant = 'quantitative_y'
    color_nominal = 'nominal_color'
    shape_ordinal = 'ordinal_shape' # 假设已创建并有序
    size_quant = 'quantitative_size'
    
    # 确保列存在
    if color_nominal not in df_altair: df_altair[color_nominal] = np.random.choice(['GroupX','GroupY','GroupZ'], size=len(df_altair))
    if shape_ordinal not in df_altair: # 创建一个有序类别
        shape_cats = ['Low', 'Medium', 'High']
        df_altair[shape_ordinal] = pd.Categorical(np.random.choice(shape_cats, size=len(df_altair)), categories=shape_cats, ordered=True)
    if size_quant not in df_altair: df_altair[size_quant] = np.random.rand(len(df_altair)) * 200 + 50


    # 1. 散点图 (Scatter Plot)
    print("\n1. 创建Altair散点图...")
    chart_scatter = create_altair_scatter_plot_api(df_altair, x_col=x_quant, y_col=y_quant,
                                                   color_col=color_nominal, size_col=size_quant, shape_col=shape_ordinal,
                                                   tooltip_cols=[x_quant, y_quant, color_nominal, 'id'],
                                                   title='Altair 散点图 (API)')
    chart_scatter.save("altair_outputs/scatter_plot_api.html")
    print("  散点图已保存为 scatter_plot_api.html")

    # 2. 条形图 (Bar Chart)
    print("\n2. 创建Altair条形图...")
    # 条形图通常需要一个名义/序数轴和一个定量轴
    # 聚合数据，例如按颜色类别计算y_quant的平均值
    agg_data_bar = df_altair.groupby(color_nominal)[y_quant].mean().reset_index()
    chart_bar = create_altair_bar_chart_api(agg_data_bar, x_col=color_nominal, y_col=y_quant,
                                            color_col=None, # 或者也按 color_nominal 编码颜色
                                            tooltip_cols=[color_nominal, y_quant],
                                            title='Altair 条形图 (按类别均值) (API)')
    chart_bar.save("altair_outputs/bar_chart_api.html")
    print("  条形图已保存为 bar_chart_api.html")

    # 3. 线图 (Line Chart)
    print("\n3. 创建Altair线图...")
    # 线图通常需要X轴排序
    df_altair_sorted = df_altair.sort_values(by=x_quant)
    chart_line = create_altair_line_chart_api(df_altair_sorted, x_col=x_quant, y_col=y_quant,
                                              color_col=color_nominal, stroke_dash_col=shape_ordinal, # 按形状类别使用不同线条样式
                                              tooltip_cols=[x_quant, y_quant, color_nominal],
                                              title='Altair 线图 (API)')
    chart_line.save("altair_outputs/line_chart_api.html")
    print("  线图已保存为 line_chart_api.html")

    # 4. 直方图 (Histogram)
    print("\n4. 创建Altair直方图...")
    chart_hist = create_altair_histogram_api(df_altair, field_to_bin=x_quant,
                                             color_col=color_nominal, # 可以按类别堆叠或分面
                                             title='Altair 直方图 (API)', maxbins=20)
    chart_hist.save("altair_outputs/histogram_api.html")
    print("  直方图已保存为 histogram_api.html")

    # 5. 面积图 (Area Chart)
    print("\n5. 创建Altair面积图...")
    # 面积图类似于线图，但填充线下区域
    chart_area = create_altair_area_chart_api(df_altair_sorted, x_col=x_quant, y_col=y_quant,
                                              color_col=color_nominal,
                                              tooltip_cols=[x_quant, y_quant, color_nominal],
                                              title='Altair 面积图 (API)', opacity=0.7)
    chart_area.save("altair_outputs/area_chart_api.html")
    print("  面积图已保存为 area_chart_api.html")

    # 6. 交互式图表 (Interactive Chart with Selection)
    print("\n6. 创建Altair交互式图表 (联动选择)...")
    # 此函数通常会创建更复杂的图表，例如散点图矩阵或带选择的图
    # 这里的示例将简单化，演示一个带interval selection的图
    chart_interactive = create_altair_interactive_chart_api(
        df_altair,
        x_quant_col=x_quant,
        y_quant_col=y_quant,
        color_nominal_col=color_nominal,
        selection_type='interval', # 'interval' or 'single'
        title='Altair 交互式散点图 (区间选择高亮)',
        plot_type='scatter' # 'scatter', 'bar', etc.
    )
    # 对于更复杂的联动（如散点图矩阵），API函数内部会处理
    if chart_interactive: # API可能返回None或单个图表
        chart_interactive.save("altair_outputs/interactive_chart_api.html")
        print("  交互式图表已保存为 interactive_chart_api.html")
    else:
        print("  交互式图表生成被跳过或未返回单个图表对象。")

    print("\n--- Altair 可视化接口化演示结束 ---\n")


def demo_geospatial_visuals(user_points_df=None, user_polygons_df=None, user_latlon_df=None, points_lon_col=None, points_lat_col=None):
    """演示 ADescribe.geospatial_visualizations 中的核心功能。"""
    print("\n--- 地理空间可视化接口化演示 (GeoPandas & Folium) ---")

    # 1. GeoPandas 演示
    print("\n--- 1. GeoPandas 功能演示 ---")
    # 1.1 创建GeoDataFrame
    if user_points_df is not None and points_lon_col and points_lat_col:
        gdf_points = create_geopandas_dataframe_api(user_points_df, geometry_type='points', 
                                                    lon_col=points_lon_col, lat_col=points_lat_col, crs="EPSG:4326")
        print("  已从用户提供的点数据创建GeoDataFrame。")
    else:
        # 使用模块内建的样本数据
        sample_cities_data = {
            'city': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'],
            'latitude': [40.7128, 51.5074, 35.6895, 48.8566, -33.8688],
            'longitude': [-74.0060, -0.1278, 139.6917, 2.3522, 151.2093],
            'population': [8399000, 8982000, 13960000, 2141000, 5312000]
        }
        df_cities = pd.DataFrame(sample_cities_data)
        gdf_points = create_geopandas_dataframe_api(df_cities, geometry_type='points', 
                                                    lon_col='longitude', lat_col='latitude', crs="EPSG:4326")
        print("  已从内建城市样本数据创建GeoDataFrame。")
    print("  GeoDataFrame (点) 预览:\n", gdf_points.head())

    # 1.2 基础绘图
    print("\n  1.2 绘制基础GeoPandas地图...")
    plot_geopandas_basic_api(gdf_points, title='GeoPandas 基础点图 (API)', 
                             save_path='geospatial_outputs/geopandas_basic_points.png',
                             column_to_plot_color='population', cmap='viridis', legend=True)
    print("    基础点图已保存。")

    # 1.3 Choropleth 图 (需要多边形数据和世界地图)
    print("\n  1.3 绘制GeoPandas Choropleth地图...")
    # 使用GeoPandas自带的世界地图
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    # 为了演示，给世界地图添加一些随机数据
    world['random_value'] = np.random.rand(len(world)) * 100
    
    plot_geopandas_choropleth_api(world, column_to_plot='random_value', cmap='OrRd', 
                                  title='GeoPandas Choropleth 示例 (API)', 
                                  legend_label='随机值',
                                  save_path='geospatial_outputs/geopandas_choropleth_world.png')
    print("    Choropleth地图已保存。")


    # 2. Folium 演示
    print("\n\n--- 2. Folium 功能演示 ---")
    # 2.1 创建基础Folium地图
    print("\n  2.1 创建基础Folium地图...")
    m_base = create_folium_map_api(location=[20,0], zoom_start=2, map_tiles='OpenStreetMap')
    m_base.save("geospatial_outputs/folium_base_map.html")
    print("    基础Folium地图已保存。")

    # 2.2 添加标记到地图
    print("\n  2.2 向Folium地图添加标记...")
    m_with_markers = create_folium_map_api(location=[39.9, 116.4], zoom_start=5) # 北京附近
    if user_latlon_df is not None:
        # 假设user_latlon_df包含 'latitude', 'longitude', 'popup_info', 'tooltip_info' 列
        points_for_folium = user_latlon_df
    else:
        points_for_folium = create_sample_points_for_folium_api(num_points=5)
        
    m_with_markers = add_markers_to_folium_map_api(m_with_markers, points_for_folium, 
                                                   lat_col='latitude', lon_col='longitude', 
                                                   popup_col='popup_info', tooltip_col='tooltip_info')
    m_with_markers.save("geospatial_outputs/folium_map_with_markers.html")
    print("    带标记的Folium地图已保存。")

    # 2.3 添加多边形到地图 (需要多边形数据)
    print("\n  2.3 向Folium地图添加多边形...")
    m_with_polygons = create_folium_map_api(location=[0,0], zoom_start=1)
    if user_polygons_df is not None and 'geometry' in user_polygons_df.columns:
        # 假设user_polygons_df是一个GeoDataFrame
        gdf_polygons_folium = user_polygons_df
    else:
        # 创建一些示例多边形数据 (GeoDataFrame)
        poly_data = {
            'name': ['Area1', 'Area2'],
            'geometry': [
                Polygon([(-10, -10), (-10, 10), (10, 10), (10, -10)]),
                Polygon([(20, 20), (20, 40), (40, 40), (40, 20)])
            ],
            'value': [100, 150]
        }
        gdf_polygons_folium = geopandas.GeoDataFrame(poly_data, crs="EPSG:4326")
    
    # Folium通常使用GeoJSON格式，GeoPandas可以直接转换
    # 如果gdf_polygons_folium很大，转换可能耗时
    if not gdf_polygons_folium.empty:
        m_with_polygons = add_polygons_to_folium_map_api(m_with_polygons, gdf_polygons_folium, 
                                                         popup_col='name', tooltip_col='name',
                                                         style_function=lambda x: {'fillColor': 'blue', 'color': 'black', 'weight':1, 'fillOpacity':0.5},
                                                         highlight_function=lambda x: {'fillColor':'green', 'fillOpacity':0.7})
        m_with_polygons.save("geospatial_outputs/folium_map_with_polygons.html")
        print("    带多边形的Folium地图已保存。")
    else:
        print("    跳过添加多边形：无多边形数据提供或生成。")
        
    # 2.4 Folium Choropleth (也可以用GeoPandas数据)
    print("\n  2.4 创建Folium Choropleth地图...")
    # 需要一个GeoJSON文件路径或GeoDataFrame，以及数据进行关联
    # 使用GeoPandas自带的世界地图数据，转换为GeoJSON给Folium用
    world_geojson = world.to_json() # world是之前加载的GeoPandas世界地图
    
    # 准备要关联到GeoJSON的数据 (确保key_on能够匹配)
    # world GeoDataFrame中通常有 'iso_a3'或'name'等列可用于匹配
    # 我们用之前添加的 'random_value'
    data_for_choropleth = world[['name', 'random_value']] # name 作为 id, random_value 作为值

    # 为了让key_on (e.g., feature.properties.name) 能够匹配data_for_choropleth的列
    # 需要确保GeoJSON中的属性名和data_for_choropleth中的列名一致
    # GeoPandas的to_json()通常会保留列名作为属性

    m_choropleth_folium = create_folium_map_api(location=[0,0], zoom_start=1.5, map_tiles='CartoDB positron')
    
    # folium.Choropleth 调用通常在 API 函数内部
    # 这里假设 API 能够正确处理
    # add_choropleth_to_folium_map_api (假设有这样一个API函数)
    # 由于当前 geospatial_visualizations.py 没有这个API，我们这里仅作概念性说明
    # 实际应用时会调用类似下面的函数
    try:
        import folium # 确保folium被导入
        folium.Choropleth(
            geo_data=world_geojson, # GeoJSON数据
            name='choropleth_world',
            data=data_for_choropleth, # DataFrame数据
            columns=['name', 'random_value'], # DataFrame中的列：[ID列, 值列]
            key_on='feature.properties.name', # GeoJSON中用于匹配的属性路径
            fill_color='YlGnBu',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='随机值 (Folium)',
            nan_fill_color='grey' # 填充无匹配数据的区域
        ).add_to(m_choropleth_folium)
        m_choropleth_folium.save("geospatial_outputs/folium_choropleth_map.html")
        print("    Folium Choropleth地图已保存。")
    except Exception as e:
        print(f"    创建Folium Choropleth地图失败: {e} (这部分演示可能依赖未完全实现的API)")


    print("\n--- 地理空间可视化接口化演示结束 ---\n")


def demo_network_visuals(user_edgelist=None, graph_gen_type=None, graph_gen_params=None):
    """演示 ADescribe.network_analysis_visualizations 中的核心功能。"""
    print("\n--- 网络分析与可视化接口化演示 (NetworkX) ---")

    # 1. 创建图
    print("\n--- 1. 创建NetworkX图 ---")
    if user_edgelist is not None:
        G = create_networkx_graph_api(edge_list=user_edgelist)
        print("  已从用户提供的边列表创建图。")
    elif graph_gen_type:
        G = generate_example_graph_api(graph_type=graph_gen_type, params=graph_gen_params or {})
        print(f"  已生成示例图: {graph_gen_type}")
    else:
        # 默认创建一个简单的示例图
        sample_edges = create_sample_edge_list_api(num_edges=15, num_nodes=10)
        G = create_networkx_graph_api(edge_list=sample_edges, add_random_attributes=True)
        print("  已从内建样本边列表创建图，并添加了随机属性。")
    
    if G is None:
        print("错误：图对象未能创建。演示终止。")
        return
    print(f"  图信息: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边。")

    # 2. 绘制基础图
    print("\n--- 2. 绘制基础NetworkX图 ---")
    draw_networkx_graph_api(G, title='基础NetworkX图 (API)', 
                            save_path='network_outputs/basic_graph.png',
                            layout_type='spring', show_labels=True)
    print("  基础图已保存。")

    # 3. 计算网络指标
    print("\n--- 3. 计算网络指标 ---")
    metrics = calculate_network_metrics_api(G, metrics_to_calculate=['degree_centrality', 'betweenness_centrality', 'density', 'avg_clustering'])
    if metrics:
        print("  计算的网络指标:")
        for metric_name, value in metrics.items():
            if isinstance(value, dict): # 对于节点级别的指标
                print(f"    {metric_name} (前5个节点): {list(value.items())[:5]}")
            else:
                print(f"    {metric_name}: {value:.4f}" if isinstance(value, float) else f"    {metric_name}: {value}")
    else:
        print("  未能计算网络指标。")

    # 4. 可视化节点属性
    print("\n--- 4. 可视化节点属性 ---")
    # 确保图中有 'category' 节点属性用于颜色编码，'size' 用于大小编码
    if not all(attr in G.nodes[list(G.nodes())[0]] for attr in ['category', 'size']):
         print("  为演示目的，正在为节点添加 'category' 和 'size' 属性...")
         import random
         categories = ['A', 'B', 'C']
         for node in G.nodes():
             if 'category' not in G.nodes[node]:
                 G.nodes[node]['category'] = random.choice(categories)
             if 'size' not in G.nodes[node]:
                 G.nodes[node]['size'] = random.randint(50, 200)
    
    visualize_node_attributes_api(G, attribute_name_for_color='category', 
                                  attribute_name_for_size='size', # 使用节点属性 'size'
                                  size_multiplier=1.0, # 调整节点大小的乘数
                                  title='节点属性可视化 (颜色按类别, 大小按size属性) (API)',
                                  save_path='network_outputs/node_attributes_visualization.png',
                                  cmap_name='viridis', show_colorbar_for_size=False) # colorbar for size usually for continuous values
    print("  节点属性可视化图已保存。")


    # 5. 可视化边属性
    print("\n--- 5. 可视化边属性 ---")
    # 确保图中有 'weight' 边属性用于宽度/颜色编码
    if not any('weight' in G.edges[edge] for edge in G.edges()):
        print("  为演示目的，正在为边添加 'weight' 属性...")
        import random
        for u, v in G.edges():
            G.edges[u,v]['weight'] = random.uniform(0.5, 5.0)

    visualize_edge_attributes_api(G, attribute_name_for_width='weight',
                                  attribute_name_for_color='weight', # 也可以用weight来调色
                                  width_multiplier=1.5,
                                  title='边属性可视化 (宽度和颜色按权重) (API)',
                                  save_path='network_outputs/edge_attributes_visualization.png',
                                  edge_cmap_name='coolwarm', show_edge_colorbar=True)
    print("  边属性可视化图已保存。")
    
    # 演示一个特定类型的图生成 (例如：Barabasi-Albert)
    print("\n--- 6. 生成并绘制特定类型图 (Barabasi-Albert) ---")
    G_ba = generate_example_graph_api(graph_type='barabasi_albert', params={'n': 30, 'm': 2, 'seed':42})
    if G_ba:
        draw_networkx_graph_api(G_ba, title='Barabasi-Albert 图 (API)', 
                                save_path='network_outputs/barabasi_albert_graph.png')
        print("  Barabasi-Albert 图已保存。")
    else:
        print("  未能生成Barabasi-Albert图。")

    print("\n--- 网络分析与可视化接口化演示结束 ---\n")

def demo_text_analysis(sample_corpus=None, spacy_model_name='en_core_web_sm'):
    """演示 ADescribe.text_data_analysis 中的核心文本分析功能。"""
    print("\n--- 文本数据分析与可视化接口化演示 ---")

    if sample_corpus is None:
        corpus = get_sample_text_data_api(num_documents=3)
        print("使用自动生成的样本语料库进行文本分析演示。")
    else:
        corpus = sample_corpus
        print("使用用户提供的语料库进行文本分析演示。")
    
    print("示例文本 (第一篇):\n", corpus[0][:200] + "..." if corpus else "无")

    # 1. NLTK 预处理
    print("\n--- 1. NLTK 文本预处理 ---")
    if corpus:
        processed_nltk = preprocess_text_nltk_api(corpus[0], language='english', 
                                                  lemmatize=True, remove_stopwords=True)
        print("  NLTK预处理结果 (第一篇文档，前20个词):\n ", processed_nltk[:20])
        
        # 对整个语料库进行处理
        all_processed_nltk = [preprocess_text_nltk_api(doc, lemmatize=True) for doc in corpus]
        print(f"  NLTK处理了 {len(all_processed_nltk)} 篇文档。")
    else:
        print("  跳过NLTK预处理：语料库为空。")

    # 2. spaCy 预处理
    print("\n--- 2. spaCy 文本预处理 ---")
    if corpus:
        try:
            processed_spacy_doc = preprocess_text_spacy_api(corpus[0], model_name=spacy_model_name,
                                                        lemmatize=True, remove_stopwords=True,
                                                        extract_entities=True, extract_pos=True)
            if processed_spacy_doc:
                print("  spaCy预处理结果 (第一篇文档):")
                print("    Tokens (前10): ", processed_spacy_doc['tokens'][:10])
                print("    Lemmas (前10): ", processed_spacy_doc['lemmas'][:10])
                if processed_spacy_doc.get('entities'):
                     print("    Entities (前3): ", processed_spacy_doc['entities'][:3])
                if processed_spacy_doc.get('pos_tags'):
                     print("    POS Tags (前5): ", processed_spacy_doc['pos_tags'][:5])
            
            # 对整个语料库进行处理 (只提取词元)
            all_processed_spacy_tokens = []
            for doc_text in corpus:
                spacy_result = preprocess_text_spacy_api(doc_text, model_name=spacy_model_name, lemmatize=False, remove_stopwords=False, extract_entities=False, extract_pos=False)
                if spacy_result and 'tokens' in spacy_result:
                    all_processed_spacy_tokens.append(spacy_result['tokens'])
            print(f"  spaCy (仅tokens) 处理了 {len(all_processed_spacy_tokens)} 篇文档。")

        except Exception as e: # spaCy模型可能需要下载
            print(f"  spaCy预处理失败: {e}")
            print(f"  请确保已安装spaCy并下载了模型: python -m spacy download {spacy_model_name}")
            all_processed_spacy_tokens = None # 标记为失败
    else:
        print("  跳过spaCy预处理：语料库为空。")
        all_processed_spacy_tokens = None

    # 3. 生成词云
    print("\n--- 3. 生成词云 ---")
    # 使用NLTK处理后的文本（如果可用）来生成词云
    text_for_wordcloud = ""
    if corpus:
        # 将所有NLTK处理后的文档（词元列表）合并成一个长字符串
        if 'all_processed_nltk' in locals() and all_processed_nltk:
            text_for_wordcloud = " ".join([" ".join(doc_tokens) for doc_tokens in all_processed_nltk])
        elif corpus: # 如果NLTK失败，则使用原始语料库
            text_for_wordcloud = " ".join(corpus)

    if text_for_wordcloud:
        generate_wordcloud_api(text_for_wordcloud, 
                               save_path='text_analysis_outputs/wordcloud_api.png',
                               stopwords_list=None, # NLTK预处理时已移除停用词
                               width=800, height=400, background_color='white')
        print("  词云图已保存。")
    else:
        print("  跳过词云生成：无可用文本。")

    # 4. 文本向量化 (Scikit-learn TF-IDF)
    print("\n--- 4. 文本向量化 (TF-IDF) ---")
    # 使用原始语料库或经过基本处理（如小写转换）的语料库
    if corpus:
        vectorizer, feature_matrix = vectorize_text_sklearn_api(
            corpus, method='tfidf', 
            max_features=1000, ngram_range=(1,2), stop_words='english'
        )
        if feature_matrix is not None:
            print(f"  TF-IDF特征矩阵形状: {feature_matrix.shape}")
            print("  特征名称 (前10个): ", vectorizer.get_feature_names_out()[:10])
            # 可以进一步展示矩阵内容，但对于大型矩阵可能不直观
            # print("  TF-IDF 矩阵 (部分):
", feature_matrix[:2, :5].toarray())
        else:
            print("  TF-IDF向量化失败。")
            
        # 演示 CountVectorizer
        vectorizer_count, feature_matrix_count = vectorize_text_sklearn_api(
            corpus, method='count', 
            max_features=500, ngram_range=(1,1) 
        )
        if feature_matrix_count is not None:
            print(f"  CountVectorizer特征矩阵形状: {feature_matrix_count.shape}")
            print("  特征名称 (前10个): ", vectorizer_count.get_feature_names_out()[:10])
    else:
        print("  跳过文本向量化：语料库为空。")


    print("\n--- 文本数据分析与可视化接口化演示结束 ---\n")


def demo_dimensionality_reduction(n_samples=250, n_features=30, n_classes=3):
    """演示 ADescribe.dimensionality_reduction_visualizations 中的降维技术。"""
    print("\n--- 降维技术与可视化接口化演示 (PCA & t-SNE) ---")

    # 1. 生成样本数据
    print("\n--- 1. 生成高维样本数据 ---")
    X, y_labels, feature_names = create_dr_sample_data_api(
        n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=42
    )
    print(f"  生成的样本数据形状: X={X.shape}, y_labels={y_labels.shape}")
    print(f"  特征名称 (前5个): {feature_names[:5]}")

    # 2. PCA (主成分分析)
    print("\n--- 2. 执行PCA并可视化 ---")
    # 2.1 执行PCA
    pca_model, X_pca = perform_pca_sklearn_api(X, n_components=2, random_state=42)
    if pca_model and X_pca is not None:
        print(f"  PCA降维后数据形状: {X_pca.shape}")
        print(f"  PCA解释的方差比例: {pca_model.explained_variance_ratio_}")
        
        # 2.2 可视化PCA结果
        visualize_pca_results_api(X_pca, y_labels, pca_model,
                                  title='PCA降维结果 (前2个主成分) (API)',
                                  save_path_scatter='dimensionality_reduction_outputs/pca_scatter_2d.png',
                                  save_path_scree='dimensionality_reduction_outputs/pca_scree_plot.png',
                                  target_names=[f'Class {i}' for i in range(n_classes)]) # 可选的类别名称
        print("  PCA散点图和碎石图已保存。")

        # 演示PCA到更多成分并可视化解释方差
        pca_model_full, _ = perform_pca_sklearn_api(X, n_components=None) # 保留所有成分
        if pca_model_full:
            visualize_pca_results_api(None, None, pca_model_full, plot_type='scree_only',
                                      title='PCA 完整碎石图 (API)',
                                      save_path_scree='dimensionality_reduction_outputs/pca_scree_plot_full.png')
            print("  PCA完整碎石图已保存。")

    else:
        print("  PCA执行失败。")

    # 3. t-SNE (t-分布随机邻域嵌入)
    print("\n--- 3. 执行t-SNE并可视化 ---")
    # 3.1 执行t-SNE
    # t-SNE通常在PCA降维后的数据上执行，以减少计算量和噪声
    X_for_tsne = X
    if X_pca is not None and X.shape[1] > 10: # 如果原始特征很多，且PCA成功
        print("  在PCA降维后的数据 (前10个主成分，如果可用) 上执行t-SNE...")
        pca_for_tsne, X_pca_for_tsne = perform_pca_sklearn_api(X, n_components=min(10, X.shape[1]))
        if X_pca_for_tsne is not None:
            X_for_tsne = X_pca_for_tsne
        else:
            print("  PCA for t-SNE 失败，将使用原始数据。")
            
    tsne_model, X_tsne = perform_tsne_sklearn_api(X_for_tsne, n_components=2, random_state=42, 
                                                  perplexity=min(30.0, X_for_tsne.shape[0]-1), # perplexity应小于样本数
                                                  learning_rate=200.0, n_iter=300) # 减少迭代次数以加快演示
    if tsne_model and X_tsne is not None:
        print(f"  t-SNE降维后数据形状: {X_tsne.shape}")
        print(f"  t-SNE KL散度: {tsne_model.kl_divergence_:.2f}")

        # 3.2 可视化t-SNE结果
        visualize_tsne_results_api(X_tsne, y_labels,
                                   title='t-SNE降维结果 (2个成分) (API)',
                                   save_path='dimensionality_reduction_outputs/tsne_scatter_2d.png',
                                   target_names=[f'类别 {i}' for i in range(n_classes)])
        print("  t-SNE散点图已保存。")
    else:
        print("  t-SNE执行失败。")

    print("\n--- 降维技术与可视化接口化演示结束 ---\n")

def demo_ydata_profiling_eda(use_custom_df=False, custom_df_params=None):
    """演示 ADescribe.ydata_profiling_example 中的EDA报告生成。"""
    print("\n--- ydata-profiling (Pandas Profiling) EDA报告演示 ---")

    if use_custom_df and custom_df_params:
        df_profile = create_profiling_sample_df_api(**custom_df_params)
        print("使用用户定义的参数创建DataFrame进行ydata-profiling。")
    else:
        df_profile = create_profiling_sample_df_api(n_samples=150, n_num_features=5, n_cat_features=3, n_bool_features=2, date_col=True, nan_frac=0.1)
        print("使用默认参数创建DataFrame进行ydata-profiling。")
    
    print(f"  DataFrame形状: {df_profile.shape}")
    print("  DataFrame列类型:\n", df_profile.dtypes)

    # 1. 生成报告
    print("\n--- 1. 生成ydata-profiling报告 ---")
    report_path = "ydata_profiling_outputs/ydata_profile_report_api.html"
    profile = generate_pandas_profiling_report_api(df_profile, 
                                                   report_title="ydata-profiling EDA报告 (API)",
                                                   output_path=report_path,
                                                   minimal_mode=False, # 生成完整报告
                                                   config_overrides=None, # 可以传入配置字典
                                                   silent=True) # 减少控制台输出
    
    if profile: # API返回ProfileReport对象或None
        print(f"  ydata-profiling报告已生成并尝试保存至: {report_path}")
        # 如果API内部不处理保存，可以在这里保存：
        # if not os.path.exists(report_path): profile.to_file(report_path)
    else:
        print("  ydata-profiling报告生成失败或未返回ProfileReport对象。")

    print("\n--- ydata-profiling (Pandas Profiling) EDA报告演示结束 ---\n")

def demo_sweetviz_eda(custom_data_params=None):
    """演示 ADescribe.sweetviz_example 中的Sweetviz EDA报告功能。"""
    print("\n--- Sweetviz EDA报告接口化演示 ---")

    # 1. 准备数据
    print("\n--- 1. 准备Sweetviz演示数据 ---")
    if custom_data_params and isinstance(custom_data_params, dict):
        df_sv1 = create_profiling_sample_df_api(**custom_data_params.get('df1', {}))
        print("  已为Sweetviz df1 创建自定义数据。")
        if 'df2' in custom_data_params:
            df_sv2 = create_profiling_sample_df_api(**custom_data_params.get('df2', {}))
            print("  已为Sweetviz df2 创建自定义数据 (用于比较)。")
        else:
            df_sv2 = None
    else:
        # 使用 ydata_profiling 的数据生成函数创建一些数据
        df_sv1 = create_profiling_sample_df_api(n_samples=200, n_num_features=6, n_cat_features=4, nan_frac=0.05, random_seed=101)
        df_sv2 = create_profiling_sample_df_api(n_samples=180, n_num_features=6, n_cat_features=4, nan_frac=0.08, random_seed=202) # 略有不同的数据集
        print("  已为Sweetviz创建默认样本数据 (df1 和 df2)。")

    print(f"  df_sv1 形状: {df_sv1.shape}")
    if df_sv2 is not None:
        print(f"  df_sv2 形状: {df_sv2.shape}")
        
    target_feature_name = None
    # 尝试找到一个合适的数值或类别特征作为目标特征
    # Sweetviz 可以很好地处理有目标特征的情况
    if 'numeric_feat_0' in df_sv1.columns: target_feature_name = 'numeric_feat_0'
    elif 'cat_feat_0' in df_sv1.columns: target_feature_name = 'cat_feat_0'
    elif 'bool_feat_0' in df_sv1.columns: target_feature_name = 'bool_feat_0'
    
    if target_feature_name:
        print(f"  将使用 '{target_feature_name}' 作为Sweetviz的目标特征 (如果存在)。")


    # 2. 生成单个DataFrame的Sweetviz报告
    print("\n--- 2. 生成单个DataFrame的Sweetviz报告 ---")
    report_single_path = "sweetviz_outputs/sweetviz_report_single_api.html"
    sv_report_single = generate_sweetviz_report_api(
        df_sv1, 
        report_title="Sweetviz 单数据集分析 (API)",
        target_feature_name=target_feature_name, # 可选
        output_path=report_single_path,
        pairwise_analysis='on' # 'on', 'off', 'auto'
    )
    if sv_report_single: # API返回DataframeReport对象或None
        print(f"  Sweetviz单数据集报告已生成并尝试保存至: {report_single_path}")
        # sv_report_single.show_html(report_single_path, open_browser=False) # 如果API不保存
    else:
        print("  Sweetviz单数据集报告生成失败或未返回对象。")


    # 3. 比较两个DataFrame的Sweetviz报告
    print("\n--- 3. 比较两个DataFrame的Sweetviz报告 ---")
    if df_sv2 is not None:
        report_compare_path = "sweetviz_outputs/sweetviz_report_compare_api.html"
        sv_report_compare = compare_dataframes_sweetviz_api(
            df_sv1, df_sv2, 
            dataframe1_name="训练集 (模拟)", dataframe2_name="测试集 (模拟)",
            report_title="Sweetviz 数据集比较 (API)",
            target_feature_name=target_feature_name, # 可选
            output_path=report_compare_path
        )
        if sv_report_compare:
            print(f"  Sweetviz数据集比较报告已生成并尝试保存至: {report_compare_path}")
            # sv_report_compare.show_html(report_compare_path, open_browser=False)
        else:
            print("  Sweetviz数据集比较报告生成失败或未返回对象。")
    else:
        print("  跳过Sweetviz数据集比较：df_sv2 未提供。")

    print("\n--- Sweetviz EDA报告接口化演示结束 ---\n")


def demo_lux_conceptual():
    """演示 ADescribe.lux_conceptual_example 中的Lux概念解释。"""
    print("\n--- Lux 概念解释演示 ---")
    
    print("\n1. Lux的设计理念与目标:")
    explain_lux_philosophy()
    
    print("\n2. Lux的典型工作流程:")
    explain_lux_typical_workflow()
    
    print("\n3. Lux的局限性与注意事项:")
    explain_lux_limitations_and_considerations()
    
    print("\n--- Lux 概念解释演示结束 ---\n")

def run_all_adescribe_demos():
    """运行 ADescribe 部分的所有演示函数。"""
    print("========== 开始 A: 描述性分析与EDA 演示 ==========")
    
    # 确保输出目录存在
    output_dirs = [
        "pandas_outputs", "matplotlib_outputs", "seaborn_outputs", 
        "plotly_outputs", "bokeh_outputs", "altair_outputs",
        "geospatial_outputs", "network_outputs", "text_analysis_outputs",
        "dimensionality_reduction_outputs", "ydata_profiling_outputs", "sweetviz_outputs"
    ]
    import os
    for out_dir in output_dirs:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f"创建输出目录: {out_dir}")

    run_all_pandas_demos()
    demo_matplotlib_visuals()
    demo_seaborn_visuals()
    demo_plotly_visuals()
    demo_bokeh_visuals() # Bokeh 可能需要手动安装 chromedriver for png/svg export
    demo_altair_visuals() # Altair 可能需要 altair_viewer 和其他依赖
    demo_geospatial_visuals() # GeoPandas 和 Folium 有其自身依赖
    demo_network_visuals()
    demo_text_analysis() # NLTK/spaCy可能需要下载数据/模型
    demo_dimensionality_reduction()
    demo_ydata_profiling_eda()
    demo_sweetviz_eda()
    demo_lux_conceptual() # 仅概念解释，无实际代码运行

    print("========== A: 描述性分析与EDA 演示结束 ==========

")

if __name__ == '__main__':
    run_all_adescribe_demos() 