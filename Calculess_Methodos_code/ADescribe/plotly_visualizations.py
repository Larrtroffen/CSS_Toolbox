import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# ==============================================================================
# 接口化绘图函数 (API-like Plotting Functions)
# ==============================================================================

OUTPUT_DIR = "plotly_charts_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_px_scatter(df, x_col, y_col, color_col=None, symbol_col=None, size_col=None, 
                      hover_name_col=None, hover_data_cols=None, facet_row_col=None, facet_col_col=None,
                      title='交互式散点图 (Plotly Express)', xlabel=None, ylabel=None,
                      log_x=False, log_y=False, trendline=None, # e.g., 'ols', 'lowess'
                      save_path=None, **kwargs):
    """
    使用 Plotly Express 创建交互式散点图。

    参数:
    df (pd.DataFrame): 输入的数据框。
    x_col (str): X轴对应的列名。
    y_col (str): Y轴对应的列名。
    color_col (str, optional): 用于颜色编码的列名。
    symbol_col (str, optional): 用于标记形状编码的列名。
    size_col (str, optional): 用于标记大小编码的列名 (应为数值列)。
    hover_name_col (str, optional): 在悬停工具提示中显示的列名。
    hover_data_cols (list of str, optional): 在悬停工具提示中额外显示的列名列表。
    facet_row_col (str, optional): 用于创建分面行子图的列名。
    facet_col_col (str, optional): 用于创建分面列子图的列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_col。
    ylabel (str, optional): Y轴标签。默认为y_col。
    log_x (bool, optional): 是否使用对数X轴。默认为False。
    log_y (bool, optional): 是否使用对数Y轴。默认为False。
    trendline (str, optional): 添加趋势线类型 (例如 'ols', 'lowess', 'expanding')。
    save_path (str, optional): HTML文件的保存路径 (不含扩展名)。如果为None，则显示图表。
    **kwargs: 传递给 px.scatter 的其他参数。
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, symbol=symbol_col, size=size_col,
                     hover_name=hover_name_col, hover_data=hover_data_cols,
                     facet_row=facet_row_col, facet_col=facet_col_col,
                     log_x=log_x, log_y=log_y, trendline=trendline,
                     title=title, labels={x_col: xlabel if xlabel else x_col, y_col: ylabel if ylabel else y_col},
                     **kwargs)
    if save_path:
        full_save_path = os.path.join(OUTPUT_DIR, f"{save_path}_scatter.html")
        fig.write_html(full_save_path)
        print(f"Plotly Express 散点图已保存至: {full_save_path}")
    else:
        fig.show()

def create_px_line(df, x_col, y_col, color_col=None, line_group_col=None, symbol_col=None,
                   hover_name_col=None, facet_row_col=None, facet_col_col=None,
                   title='交互式折线图 (Plotly Express)', xlabel=None, ylabel=None,
                   log_x=False, log_y=False, markers=True, save_path=None, **kwargs):
    """
    使用 Plotly Express 创建交互式折线图。

    参数:
    df (pd.DataFrame): 输入的数据框，建议按x_col排序。
    x_col (str): X轴对应的列名。
    y_col (str): Y轴对应的列名。
    color_col (str, optional): 用于颜色编码的列名。
    line_group_col (str, optional): 用于定义不同线条组的列名。
    symbol_col (str, optional): 用于标记形状编码的列名。
    hover_name_col (str, optional): 在悬停工具提示中显示的列名。
    facet_row_col (str, optional): 用于创建分面行子图的列名。
    facet_col_col (str, optional): 用于创建分面列子图的列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。默认为x_col。
    ylabel (str, optional): Y轴标签。默认为y_col。
    log_x (bool, optional): 是否使用对数X轴。
    log_y (bool, optional): 是否使用对数Y轴。
    markers (bool, optional): 是否在折线上显示标记点。
    save_path (str, optional): HTML文件的保存路径 (不含扩展名)。
    **kwargs: 传递给 px.line 的其他参数。
    """
    fig = px.line(df, x=x_col, y=y_col, color=color_col, line_group=line_group_col, symbol=symbol_col,
                  hover_name=hover_name_col, facet_row=facet_row_col, facet_col=facet_col_col,
                  log_x=log_x, log_y=log_y, markers=markers,
                  title=title, labels={x_col: xlabel if xlabel else x_col, y_col: ylabel if ylabel else y_col},
                  **kwargs)
    if save_path:
        full_save_path = os.path.join(OUTPUT_DIR, f"{save_path}_line.html")
        fig.write_html(full_save_path)
        print(f"Plotly Express 折线图已保存至: {full_save_path}")
    else:
        fig.show()

def create_px_bar(df, x_col, y_col, color_col=None, barmode='relative', # 'group', 'overlay', 'relative'
                  hover_name_col=None, facet_row_col=None, facet_col_col=None, text_auto=False,
                  title='交互式条形图 (Plotly Express)', xlabel=None, ylabel=None, orientation='v', # 'v' or 'h'
                  save_path=None, **kwargs):
    """
    使用 Plotly Express 创建交互式条形图。

    参数:
    df (pd.DataFrame): 输入的数据框。
    x_col (str): X轴对应的列名 (或Y轴，如果orientation='h')。
    y_col (str): Y轴对应的列名 (或X轴，如果orientation='h')。
    color_col (str, optional): 用于颜色编码的列名。
    barmode (str, optional): 条形图模式 ('group', 'overlay', 'relative')。
    hover_name_col (str, optional): 在悬停工具提示中显示的列名。
    facet_row_col (str, optional): 用于创建分面行子图的列名。
    facet_col_col (str, optional): 用于创建分面列子图的列名。
    text_auto (bool or str, optional): 是否在条形上显示文本值。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。
    ylabel (str, optional): Y轴标签。
    orientation (str, optional): 条形图方向 ('v'垂直, 'h'水平)。
    save_path (str, optional): HTML文件的保存路径 (不含扩展名)。
    **kwargs: 传递给 px.bar 的其他参数。
    """
    fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode=barmode, text_auto=text_auto,
                 hover_name=hover_name_col, facet_row=facet_row_col, facet_col=facet_col_col,
                 orientation=orientation, title=title, 
                 labels={x_col: xlabel if xlabel else x_col, y_col: ylabel if ylabel else y_col},
                 **kwargs)
    if save_path:
        full_save_path = os.path.join(OUTPUT_DIR, f"{save_path}_bar.html")
        fig.write_html(full_save_path)
        print(f"Plotly Express 条形图已保存至: {full_save_path}")
    else:
        fig.show()

def create_px_histogram(df, x_col, color_col=None, marginal=None, # 'rug', 'box', 'violin'
                        cumulative=False, histnorm=None, # 'percent', 'probability', 'density', 'probability density'
                        title='交互式直方图 (Plotly Express)', xlabel=None, ylabel='计数',
                        save_path=None, **kwargs):
    """
    使用 Plotly Express 创建交互式直方图。

    参数:
    df (pd.DataFrame): 输入的数据框。
    x_col (str): 用于绘制直方图的列名。
    color_col (str, optional): 用于颜色编码的列名。
    marginal (str, optional): 在边缘添加的图表类型 ('rug', 'box', 'violin')。
    cumulative (bool, optional): 是否绘制累积直方图。
    histnorm (str, optional): 直方图的归一化类型。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。
    ylabel (str, optional): Y轴标签。
    save_path (str, optional): HTML文件的保存路径 (不含扩展名)。
    **kwargs: 传递给 px.histogram 的其他参数。
    """
    fig = px.histogram(df, x=x_col, color=color_col, marginal=marginal, cumulative=cumulative, histnorm=histnorm,
                       title=title, labels={x_col: xlabel if xlabel else x_col},
                       **kwargs)
    fig.update_layout(yaxis_title=ylabel)
    if save_path:
        full_save_path = os.path.join(OUTPUT_DIR, f"{save_path}_histogram.html")
        fig.write_html(full_save_path)
        print(f"Plotly Express 直方图已保存至: {full_save_path}")
    else:
        fig.show()

def create_px_box(df, x_col, y_col, color_col=None, points='outliers', # False, 'all', 'outliers', 'suspectedoutliers'
                  notched=False, title='交互式箱线图 (Plotly Express)', 
                  xlabel=None, ylabel=None, orientation='v', save_path=None, **kwargs):
    """
    使用 Plotly Express 创建交互式箱线图。

    参数:
    df (pd.DataFrame): 输入的数据框。
    x_col (str): X轴对应的列名 (通常是类别型)。
    y_col (str): Y轴对应的列名 (通常是数值型)。
    color_col (str, optional): 用于颜色编码的列名。
    points (str or bool, optional): 如何显示数据点。
    notched (bool, optional): 是否使用缺口箱线图。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。
    ylabel (str, optional): Y轴标签。
    orientation (str, optional): 箱线图方向 ('v'垂直, 'h'水平)。
    save_path (str, optional): HTML文件的保存路径 (不含扩展名)。
    **kwargs: 传递给 px.box 的其他参数。
    """
    fig = px.box(df, x=x_col, y=y_col, color=color_col, points=points, notched=notched, 
                 orientation=orientation, title=title, 
                 labels={x_col: xlabel if xlabel else x_col, y_col: ylabel if ylabel else y_col},
                 **kwargs)
    if save_path:
        full_save_path = os.path.join(OUTPUT_DIR, f"{save_path}_boxplot.html")
        fig.write_html(full_save_path)
        print(f"Plotly Express 箱线图已保存至: {full_save_path}")
    else:
        fig.show()

def create_px_scatter_3d(df, x_col, y_col, z_col, color_col=None, symbol_col=None, size_col=None,
                         title='交互式3D散点图 (Plotly Express)', 
                         xlabel=None, ylabel=None, zlabel=None, save_path=None, **kwargs):
    """
    使用 Plotly Express 创建交互式3D散点图。

    参数:
    df (pd.DataFrame): 输入的数据框。
    x_col (str): X轴对应的列名。
    y_col (str): Y轴对应的列名。
    z_col (str): Z轴对应的列名。
    color_col (str, optional): 用于颜色编码的列名。
    symbol_col (str, optional): 用于标记形状编码的列名。
    size_col (str, optional): 用于标记大小编码的列名。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。
    ylabel (str, optional): Y轴标签。
    zlabel (str, optional): Z轴标签。
    save_path (str, optional): HTML文件的保存路径 (不含扩展名)。
    **kwargs: 传递给 px.scatter_3d 的其他参数。
    """
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, symbol=symbol_col, size=size_col,
                         title=title, 
                         labels={x_col: xlabel if xlabel else x_col, 
                                 y_col: ylabel if ylabel else y_col, 
                                 z_col: zlabel if zlabel else z_col},
                         **kwargs)
    if save_path:
        full_save_path = os.path.join(OUTPUT_DIR, f"{save_path}_scatter3d.html")
        fig.write_html(full_save_path)
        print(f"Plotly Express 3D散点图已保存至: {full_save_path}")
    else:
        fig.show()


def create_custom_go_plot(traces_data, layout_options, title='自定义图表 (Plotly Graph Objects)', save_path=None):
    """
    使用 Plotly Graph Objects 创建更灵活的自定义图表。

    参数:
    traces_data (list of dicts): 描述每个trace的字典列表。
                                 每个字典应包含trace类型 (例如 go.Scatter, go.Bar) 及其参数。
                                 示例: [{'type': go.Scatter, 'x': [1,2,3], 'y': [4,5,6], 'mode':'lines'}, ...]
    layout_options (dict): 描述图表布局的字典，传递给 fig.update_layout()。
    title (str, optional): 图表标题 (也可以在layout_options中指定)。
    save_path (str, optional): HTML文件的保存路径 (不含扩展名)。
    """
    fig = go.Figure()
    for trace_info in traces_data:
        trace_type = trace_info.pop('type', go.Scatter) # 默认为Scatter
        fig.add_trace(trace_type(**trace_info))
    
    final_layout = {'title_text': title}
    if layout_options:
        final_layout.update(layout_options)
    fig.update_layout(**final_layout)
    
    if save_path:
        full_save_path = os.path.join(OUTPUT_DIR, f"{save_path}_custom_go.html")
        fig.write_html(full_save_path)
        print(f"Plotly Graph Objects 自定义图表已保存至: {full_save_path}")
    else:
        fig.show()

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================
def run_plotly_demos():
    """运行所有Plotly演示函数（使用接口化函数和示例数据）。"""
    print(f"--- Plotly 交互式可视化接口化演示 (图表将保存到 '{OUTPUT_DIR}' 目录或在浏览器中打开) ---")

    # 准备一些通用的演示数据
    np.random.seed(420)
    demo_df = pd.DataFrame({
        '数值特征A': np.random.rand(100) * 20 + 10,       # 范围 10-30
        '数值特征B': np.random.randn(100) * 50 + 200,    # 均值200，标准差50
        '类别特征X': np.random.choice(['类别1', '类别2', '类别3', '类别4'], 100, p=[0.25,0.25,0.3,0.2]),
        '类别特征Y': np.random.choice(['类型Alpha', '类型Beta', '类型Gamma'], 100, p=[0.4,0.3,0.3]),
        '点大小': np.random.rand(100) * 30 + 5,          # 范围 5-35，用于散点图大小
        '时间戳': pd.date_range(start='2023-01-01', periods=100, freq='W-MON') # 每周一
    })
    demo_df['数值特征C_计算'] = demo_df['数值特征A'] * np.log1p(demo_df['点大小']) + np.random.normal(0,5,100)

    # Plotly Express 示例
    print("\n--- Plotly Express 示例 ---")
    create_px_scatter(demo_df, x_col='数值特征A', y_col='数值特征B', color_col='类别特征X', size_col='点大小',
                      title='散点图: A vs B (按X类和点大小着色)', xlabel='特征A的值', ylabel='特征B的值',
                      hover_data_cols=['类别特征Y', '时间戳'], save_path="demo_px_scatter")

    # 为折线图准备聚合数据 (例如按时间戳和类别X聚合特征A的均值)
    line_df_agg = demo_df.groupby(['时间戳', '类别特征X'])['数值特征A'].mean().reset_index()
    create_px_line(line_df_agg.sort_values('时间戳'), x_col='时间戳', y_col='数值特征A', color_col='类别特征X',
                   title='折线图: 特征A均值随时间变化 (按X类)', xlabel='日期', ylabel='特征A均值',
                   markers=True, save_path="demo_px_line_agg")

    # 条形图: 按类别X统计数值特征B的总和
    bar_df_agg = demo_df.groupby('类别特征X')['数值特征B'].sum().reset_index()
    create_px_bar(bar_df_agg, x_col='类别特征X', y_col='数值特征B', color_col='类别特征X',
                  title='条形图: 各X类下特征B的总和', xlabel='X类别', ylabel='特征B总和',
                  text_auto='.2s', save_path="demo_px_bar_sum")

    create_px_histogram(demo_df, x_col='数值特征B', color_col='类别特征Y',
                        title='直方图: 特征B的分布 (按Y类)', xlabel='特征B的值',
                        marginal='box', save_path="demo_px_histogram")

    create_px_box(demo_df, x_col='类别特征X', y_col='数值特征A', color_col='类别特征Y',
                  title='箱线图: 特征A按X类和Y类分布', xlabel='X类别', ylabel='特征A的值',
                  notched=True, points='all', save_path="demo_px_boxplot")

    # 3D散点图: 使用 Gapminder 数据集 (Plotly自带)
    gapminder_df = px.data.gapminder().query("year == 2007") # 获取2007年数据
    create_px_scatter_3d(gapminder_df, x_col='gdpPercap', y_col='lifeExp', z_col='pop', 
                         color_col='continent', size_col='pop', # 用人口数量调整点大小
                         title='3D散点图: 2007年各国GDP、预期寿命与人口 (按大洲)',
                         xlabel='人均GDP', ylabel='预期寿命', zlabel='人口数量',
                         log_x=True, size_max=60, # X轴对数化，设置最大点尺寸
                         save_path="demo_px_scatter3d_gapminder")

    # Plotly Graph Objects 示例
    print("\n--- Plotly Graph Objects 示例 ---")
    traces_example = [
        {
            'type': go.Scatter,
            'x': demo_df['时间戳'],
            'y': demo_df['数值特征A'],
            'mode': 'lines+markers',
            'name': '特征A随时间变化',
            'line': {'color': 'darkorange', 'width': 2},
            'marker': {'size': 6, 'symbol': 'diamond'}
        },
        {
            'type': go.Bar,
            'x': demo_df['类别特征X'].value_counts().index,
            'y': demo_df['类别特征X'].value_counts().values,
            'name': 'X类别计数',
            'marker_color': 'lightseagreen',
            'yaxis': 'y2' # 指定使用第二个Y轴
        }
    ]
    layout_example = {
        'xaxis_title': '时间 / X类别',
        'yaxis_title': '特征A的值',
        'yaxis2': {
            'title': '计数',
            'overlaying': 'y',
            'side': 'right',
            'showgrid': False
        },
        'legend_title_text': '图例',
        'template': 'plotly_dark' # 尝试不同模板: 'plotly', 'ggplot2', 'seaborn', 'plotly_white'
    }
    create_custom_go_plot(traces_example, layout_example, 
                          title='自定义GO图表: 特征A时间序列与X类别计数 (双轴)', 
                          save_path="demo_go_custom_dual_axis")

    print(f"--- Plotly 交互式可视化接口化演示完成 ---")

if __name__ == '__main__':
    run_plotly_demos() 