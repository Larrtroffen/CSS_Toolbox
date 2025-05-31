import altair as alt
import pandas as pd
import numpy as np
import os

# ==============================================================================
# 接口化绘图函数 (API-like Plotting Functions)
# ==============================================================================

OUTPUT_DIR = "altair_charts_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def _determine_altair_type(series):
    """根据Pandas Series的dtype推断Altair的类型。"""
    if pd.api.types.is_numeric_dtype(series):
        return 'quantitative'
    elif pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_period_dtype(series):
        return 'temporal'
    elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return 'nominal' # 或 'ordinal' 如果有序
    return 'nominal' # 默认

def create_altair_chart(df, mark_type='circle', 
                        x_col=None, y_col=None, 
                        color_col=None, shape_col=None, size_col=None, row_col=None, column_col=None,
                        x_type=None, y_type=None, color_type=None, shape_type=None, size_type=None, row_type=None, column_type=None,
                        x_bin=False, y_bin=False, x_aggregate=None, y_aggregate=None,
                        x_title=None, y_title=None, color_title=None, shape_title=None, size_title=None, row_title=None, column_title=None,
                        tooltip_cols=None, title='Altair 图表', 
                        width=400, height=300, interactive=True,
                        save_filename_html=None, properties=None, **kwargs_encode):
    """
    使用 Altair 创建可定制的交互式图表 (通用接口)。

    参数:
    df (pd.DataFrame): 输入的数据框。
    mark_type (str or alt.MarkDef): 图表标记类型 ('circle', 'bar', 'line', 'area', 'point', alt.MarkDef(...)).
    x_col, y_col, color_col, etc. (str, optional): 用于各通道编码的列名。
    x_type, y_type, color_type, etc. (str, optional): 各通道的Altair类型 ('quantitative', 'nominal', 'ordinal', 'temporal'). 
                                                  如果为None，则尝试从数据类型自动推断。
    x_bin, y_bin (bool or alt.Bin, optional): 是否对X/Y轴分箱 (用于直方图等)。
    x_aggregate, y_aggregate (str, optional): 对X/Y轴的聚合操作 ('mean', 'sum', 'count', etc.)。
    x_title, y_title, color_title, etc. (str, optional): 各通道的自定义标题。
    tooltip_cols (list of str, optional): 在悬停提示中显示的列名列表。如果None，则显示所有列。
    title (str, optional): 图表主标题。
    width (int, optional): 图表宽度。
    height (int, optional): 图表高度。
    interactive (bool, optional): 是否启用基本交互 (平移/缩放)。
    save_filename_html (str, optional): HTML文件名 (例如 'my_chart.html')。如果None，则尝试show()。
    properties (dict, optional): 传递给 chart.properties() 的额外属性。
    **kwargs_encode: 传递给 alt.Chart().encode() 的其他关键字参数，例如 order, opacity。

    返回:
    alt.Chart: 创建的Altair图表对象。
    """
    chart_props = {'width': width, 'height': height, 'title': title}
    if properties:
        chart_props.update(properties)
    
    chart = alt.Chart(df).properties(**chart_props)

    if isinstance(mark_type, str):
        chart = chart.mark_point() if mark_type == 'point' else \
                chart.mark_circle() if mark_type == 'circle' else \
                chart.mark_line() if mark_type == 'line' else \
                chart.mark_bar() if mark_type == 'bar' else \
                chart.mark_area() if mark_type == 'area' else \
                chart.mark_rect() if mark_type == 'rect' else \
                chart.mark_text() if mark_type == 'text' else \
                chart.mark_tick() if mark_type == 'tick' else chart.mark_point() # 默认
    elif isinstance(mark_type, alt.MarkDef):
        chart = chart.mark(mark_type)
    else:
        print(f"警告: 未知的 mark_type '{mark_type}'. 使用默认点标记.")
        chart = chart.mark_point()

    encodings = {}
    def add_encoding(channel_name, col_name, specified_type, title_override, bin_param=False, aggregate_param=None):
        if col_name:
            dtype = specified_type if specified_type else _determine_altair_type(df[col_name])
            field_def_args = {'type': dtype, 'title': title_override if title_override else col_name.replace('_',' ').title()}
            
            if bin_param:
                field_def_args['bin'] = alt.Bin(maxbins=20) if isinstance(bin_param, bool) and bin_param else bin_param 
            if aggregate_param:
                field_def_args['aggregate'] = aggregate_param
            
            # 使用 alt.X, alt.Y etc. 构造函数
            if channel_name == 'x': encodings[channel_name] = alt.X(col_name, **field_def_args)
            elif channel_name == 'y': encodings[channel_name] = alt.Y(col_name, **field_def_args)
            elif channel_name == 'color': encodings[channel_name] = alt.Color(col_name, **field_def_args)
            elif channel_name == 'shape': encodings[channel_name] = alt.Shape(col_name, **field_def_args)
            elif channel_name == 'size': encodings[channel_name] = alt.Size(col_name, **field_def_args)
            elif channel_name == 'row': encodings[channel_name] = alt.Row(col_name, **field_def_args)
            elif channel_name == 'column': encodings[channel_name] = alt.Column(col_name, **field_def_args)
            # Add other channels as needed (e.g., opacity, detail)

        elif aggregate_param and channel_name in ['x', 'y']: #例如 Y='count()'
            title_val = title_override if title_override else aggregate_param.capitalize().replace("()","")
            if channel_name == 'y':
                 encodings[channel_name] = alt.Y(aggregate=aggregate_param, type='quantitative', title=title_val)
            elif channel_name == 'x':
                 encodings[channel_name] = alt.X(aggregate=aggregate_param, type='quantitative', title=title_val)


    add_encoding('x', x_col, x_type, x_title, x_bin, x_aggregate)
    add_encoding('y', y_col, y_type, y_title, y_bin, y_aggregate)
    add_encoding('color', color_col, color_type, color_title)
    add_encoding('shape', shape_col, shape_type, shape_title)
    add_encoding('size', size_col, size_type, size_title)
    add_encoding('row', row_col, row_type, row_title)
    add_encoding('column', column_col, column_type, column_title)

    # 合并 kwargs_encode
    encodings.update(kwargs_encode)

    if tooltip_cols is None:
        tooltip_encodings = [alt.Tooltip(col, type=_determine_altair_type(df[col])) for col in df.columns]
    elif isinstance(tooltip_cols, list) and len(tooltip_cols) > 0:
        tooltip_encodings = []
        for col_spec in tooltip_cols:
            if isinstance(col_spec, str): # 简单的列名
                 tooltip_encodings.append(alt.Tooltip(col_spec, type=_determine_altair_type(df[col_spec]) if col_spec in df.columns else 'nominal'))
            elif isinstance(col_spec, alt.Tooltip): # 已经是Tooltip对象
                 tooltip_encodings.append(col_spec)
            # 可以添加对字典格式 tooltip 的支持
        
        # 对于 count() 或其他聚合的特殊处理
        if y_aggregate and not y_col: # e.g. y='count()'
            tt_title_y = y_title if y_title else y_aggregate.capitalize().replace("()","")
            tooltip_encodings.append(alt.Tooltip(y_aggregate, type='quantitative', title=tt_title_y))
        if x_aggregate and not x_col:
            tt_title_x = x_title if x_title else x_aggregate.capitalize().replace("()","")
            tooltip_encodings.append(alt.Tooltip(x_aggregate, type='quantitative', title=tt_title_x))
    else:
        tooltip_encodings = alt.Undefined
    
    encodings['tooltip'] = tooltip_encodings
    chart = chart.encode(**encodings)

    if interactive and not (row_col or column_col): # Faceted charts might need specific interaction config
        chart = chart.interactive()
    
    if save_filename_html:
        full_save_path = os.path.join(OUTPUT_DIR, save_filename_html if save_filename_html.lower().endswith('.html') else f"{save_filename_html}.html")
        try:
            chart.save(full_save_path)
            print(f"Altair 图表已保存至: {full_save_path}")
        except Exception as e:
            print(f"保存Altair图表失败: {e}. 尝试确保渲染器已启用或环境支持保存。")
    else:
        try:
            chart.show()
        except Exception as e:
            print(f"显示Altair图表失败: {e}. 在某些环境中，可能需要配置渲染器 (例如，`alt.renderers.enable('notebook')` 或 `alt.renderers.enable('default')`)。")
    return chart

def create_concatenated_altair_charts(charts_list, direction='h', title="组合图表", save_filename_html=None, **kwargs):
    """
    将多个Altair图表连接(concat)或分层(layer)显示。

    参数:
    charts_list (list of alt.Chart): 要组合的Altair图表对象列表。
    direction (str): 组合方向: 'h' (水平), 'v' (垂直), 'layer' (分层)。
    title (str, optional): 组合图表的整体标题 (如果适用，通常在单个图表上设置标题)。
    save_filename_html (str, optional): HTML文件名。
    **kwargs: 传递给 alt.hconcat, alt.vconcat, alt.layer 的其他参数。

    返回:
    alt.Chart: 组合后的Altair图表对象。
    """
    if not charts_list or not all(isinstance(ch, alt.Chart) for ch in charts_list):
        raise ValueError("charts_list 必须是非空列表，且所有元素都是Altair图表对象。")

    if direction == 'h':
        combined_chart = alt.hconcat(*charts_list, **kwargs)
    elif direction == 'v':
        combined_chart = alt.vconcat(*charts_list, **kwargs)
    elif direction == 'layer':
        combined_chart = alt.layer(*charts_list, **kwargs)
    else:
        raise ValueError("direction 必须是 'h', 'v', 或 'layer' 之一。")
    
    # 整体标题可以通过外部HTML结构添加，或者通过更复杂的组合图表规范
    # 这里不直接对组合图表设置properties.title，因为它可能不会按预期工作

    if save_filename_html:
        full_save_path = os.path.join(OUTPUT_DIR, save_filename_html if save_filename_html.lower().endswith('.html') else f"{save_filename_html}.html")
        try:
            combined_chart.save(full_save_path)
            print(f"Altair 组合图表已保存至: {full_save_path}")
        except Exception as e:
            print(f"保存Altair组合图表失败: {e}")
    else:
        try:
            combined_chart.show()
        except Exception as e:
            print(f"显示Altair组合图表失败: {e}")
    return combined_chart

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================
def run_altair_demos():
    """运行所有Altair演示函数（使用接口化函数和示例数据）。"""
    print(f"--- Altair 声明式可视化接口化演示 (图表将保存到 '{OUTPUT_DIR}' 目录) ---")

    # 准备演示数据DataFrame
    np.random.seed(456)
    demo_df = pd.DataFrame({
        '数量指标A': np.random.rand(80) * 100,
        '数量指标B': np.random.randn(80) * 20 + 50,
        '类别因素X': np.random.choice(['甲', '乙', '丙', '丁', '戊'], 80),
        '类别因素Y': np.random.choice(['优', '良', '中', '差'], 80, p=[0.4,0.3,0.2,0.1]),
        '时间序列点': pd.to_datetime(pd.date_range('2023-03-01', periods=80, freq='2D')),
        '点大小参数': np.random.poisson(7, 80) * 5 + 10 # 用于size通道
    })

    # 1. 散点图
    create_altair_chart(demo_df, mark_type='circle',
                        x_col='数量指标A', y_col='数量指标B',
                        color_col='类别因素X', shape_col='类别因素Y', size_col='点大小参数',
                        title='演示：综合散点图', 
                        tooltip_cols=['数量指标A', '数量指标B', '类别因素X', '类别因素Y', '点大小参数'],
                        save_filename_html="demo_altair_scatter.html")

    # 2. 折线图
    # 对于折线图，通常数据需要按X轴排序，特别是时间序列
    line_df_sorted = demo_df.sort_values(by='时间序列点')
    create_altair_chart(line_df_sorted, mark_type='line',
                        x_col='时间序列点', y_col='数量指标A',
                        color_col='类别因素X',
                        title='演示：时间序列折线图 (按X类分色)', 
                        properties = {'mark': alt.MarkDef(point=True)}, # 在线上加点
                        kwargs_encode={'strokeDash': alt.StrokeDash('类别因素Y:N', title='Y类因素')},
                        save_filename_html="demo_altair_line.html")

    # 3. 条形图 (聚合)
    # Altair条形图通常直接使用聚合
    create_altair_chart(demo_df, mark_type='bar',
                        x_col='类别因素X', y_aggregate='mean', y_col='数量指标A', # Y轴是A的均值
                        color_col='类别因素X',
                        title='演示：条形图 (X类下A指标均值)', 
                        y_title='A指标均值',
                        save_filename_html="demo_altair_bar_aggregated.html")

    # 4. 直方图
    create_altair_chart(demo_df, mark_type='bar',
                        x_col='数量指标B', x_bin=alt.Bin(maxbins=15),
                        y_aggregate='count()', # Y轴是计数
                        color_col='类别因素Y',
                        title='演示：直方图 (B指标分布，按Y类分色)',
                        x_title='数量指标B (分箱)', y_title='频数',
                        save_filename_html="demo_altair_histogram.html")

    # 5. 分面直方图 (Faceted Histogram)
    create_altair_chart(demo_df, mark_type='bar',
                        x_col='数量指标A', x_bin=True,
                        y_aggregate='count()',
                        column_col='类别因素X', # 按X类分列
                        color_col='类别因素Y', 
                        title='演示：分面直方图 (A分布，按Y色，按X列)',
                        width=150, height=150, # 分面图通常需要调整单个图的大小
                        save_filename_html="demo_altair_histogram_faceted.html")

    # 6. 组合图表
    scatter_sub = alt.Chart(demo_df).mark_point().encode(
        x='数量指标A:Q',
        y='数量指标B:Q',
        color='类别因素X:N'
    ).properties(title='子图：散点A vs B', width=250, height=200).interactive()

    bar_sub = alt.Chart(demo_df).mark_bar().encode(
        x='类别因素Y:N',
        y='average(点大小参数):Q', # 使用average聚合
        color='类别因素Y:N'
    ).properties(title='子图：Y类下平均点大小', width=250, height=200)

    create_concatenated_altair_charts([scatter_sub, bar_sub], direction='h',
                                      save_filename_html="demo_altair_concatenated.html")
    
    # 示例：分层图 (例如，散点图上叠加回归线)
    base_scatter = alt.Chart(demo_df).mark_circle(opacity=0.6).encode(
        x='数量指标A:Q',
        y='数量指标B:Q',
        color='类别因素Y:N'
    ).properties(width=500, height=300, title='演示：分层图 (散点+回归线)')
    
    # 使用transform_regression添加回归线
    regression_line = base_scatter.transform_regression(
        '数量指标A', '数量指标B', groupby=['类别因素Y'] # 为每个Y类别画一条回归线
    ).mark_line(size=3) # 线条可以设置不同颜色或样式

    create_concatenated_altair_charts([base_scatter, regression_line], direction='layer',
                                      save_filename_html="demo_altair_layered_regression.html")


    print(f"--- Altair 声明式可视化接口化演示完成 ---")

if __name__ == '__main__':
    run_altair_demos()