from bokeh.plotting import figure, show, save, output_file
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper, NumeralTickFormatter, CDSView, BooleanFilter
from bokeh.palettes import Category10, Spectral6, Viridis256
from bokeh.layouts import gridplot
from bokeh.transform import factor_cmap, jitter
import pandas as pd
import numpy as np
import os

# ==============================================================================
# 接口化绘图函数 (API-like Plotting Functions)
# ==============================================================================

OUTPUT_DIR = "bokeh_charts_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def _prepare_bokeh_source_and_output(data_input, save_filename_html, title):
    """
    辅助函数：准备ColumnDataSource并设置输出文件。
    返回 (ColumnDataSource, is_saving_flag)。
    """
    if isinstance(data_input, pd.DataFrame):
        source = ColumnDataSource(data_input)
    elif isinstance(data_input, ColumnDataSource):
        source = data_input
    else:
        raise ValueError("输入数据必须是Pandas DataFrame或Bokeh ColumnDataSource")

    is_saving = False
    if save_filename_html:
        #确保文件名以.html结尾
        if not save_filename_html.lower().endswith('.html'):
            save_filename_html += '.html'
        output_file(filename=os.path.join(OUTPUT_DIR, save_filename_html), title=title)
        is_saving = True
    return source, is_saving

def create_bokeh_scatter(data_input, x_col, y_col, 
                         cat_col_for_color=None, cat_col_for_marker=None, 
                         size_col=None, fixed_size=10,
                         title='Bokeh 交互式散点图', 
                         xlabel=None, ylabel=None, 
                         tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                         hover_tooltips=None, legend_location="top_right",
                         palette=Category10[10], marker_types=['circle', 'square', 'triangle', 'cross', 'x'],
                         save_filename_html=None, **kwargs):
    """
    使用 Bokeh 创建交互式散点图。

    参数:
    data_input (pd.DataFrame or ColumnDataSource): 输入数据。
    x_col (str): X轴列名。
    y_col (str): Y轴列名。
    cat_col_for_color (str, optional): 用于颜色编码的类别列名。
    cat_col_for_marker (str, optional): 用于标记形状编码的类别列名。
    size_col (str, optional): 用于大小编码的数值列名。如果提供，fixed_size将被忽略。
    fixed_size (int, optional): 当size_col未提供时，点的固定大小。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签 (默认为x_col)。
    ylabel (str, optional): Y轴标签 (默认为y_col)。
    tools (str, optional): Bokeh工具栏字符串。
    hover_tooltips (list of tuples, optional): HoverTool的提示信息。例: [('名称', '@col_name'), ...]
                                              如果为None，将尝试生成默认提示。
    legend_location (str, optional): 图例位置。
    palette (list of str, optional): 用于颜色编码的调色板。
    marker_types (list of str, optional): 用于标记形状编码的标记类型列表。
    save_filename_html (str, optional): HTML文件的保存名 (例如 'my_scatter.html')。如果None，则尝试show()。
    **kwargs: 传递给 p.scatter 的其他参数。
    """
    source, is_saving = _prepare_bokeh_source_and_output(data_input, save_filename_html, title)

    p = figure(title=title, tools=tools, 
               x_axis_label=xlabel if xlabel else x_col, 
               y_axis_label=ylabel if ylabel else y_col)

    scatter_params = {'x': x_col, 'y': y_col, 'source': source, 'alpha': 0.7}
    legend_field = None

    if cat_col_for_color:
        unique_colors = sorted(source.data[cat_col_for_color].unique().tolist())
        color_mapper = factor_cmap(cat_col_for_color, palette=palette, factors=unique_colors)
        scatter_params['color'] = color_mapper
        legend_field = cat_col_for_color
        scatter_params['legend_label'] = f'{cat_col_for_color} 分类' # 更通用的图例标签
    else:
        scatter_params['color'] = palette[0] if palette else 'blue'
    
    if cat_col_for_marker:
        unique_markers = sorted(source.data[cat_col_for_marker].unique().tolist())
        # Bokeh的p.scatter不支持直接通过列转换marker，需要分别绘制或使用更复杂的GlyphRenderer
        # 这里简化处理：如果同时有颜色和标记分类，颜色优先，标记忽略。或提示用户。
        if cat_col_for_color:
            print(f"警告: 同时指定了 cat_col_for_color 和 cat_col_for_marker。 Bokeh的p.scatter对marker的因子映射支持有限，将优先使用颜色分类 ('{cat_col_for_color}')。标记形状将统一。")
            scatter_params['marker'] = marker_types[0]
        else: # 仅按标记分类
            # 此部分需要为每个marker类型单独调用p.scatter，这里简化，只取第一个marker类型
            # 或创建一个更复杂的函数来处理多标记的因子映射
            print(f"提示: cat_col_for_marker 功能在当前实现中简化为单一标记类型。如需多种标记，请考虑分别绘制或扩展此函数。")
            scatter_params['marker'] = marker_types[0]
            # legend_field = cat_col_for_marker # 如果要为marker创建图例
    else:
        scatter_params['marker'] = marker_types[0]

    if size_col:
        scatter_params['size'] = size_col
    else:
        scatter_params['size'] = fixed_size

    # 合并kwargs到scatter_params
    scatter_params.update(kwargs) 

    # 移除可能冲突的legend_label，如果legend_field已设置
    if legend_field and 'legend_label' in scatter_params: 
        del scatter_params['legend_label']
        scatter_params['legend_field'] = legend_field

    p.scatter(**scatter_params)

    if hover_tooltips is None:
        hover_tooltips = [('索引', '$index'), (x_col, f'@{x_col}{{0.00a}}'), (y_col, f'@{y_col}{{0.00a}}')]
        if cat_col_for_color: hover_tooltips.append((cat_col_for_color, f'@{cat_col_for_color}'))
        if cat_col_for_marker: hover_tooltips.append((cat_col_for_marker, f'@{cat_col_for_marker}'))
        if size_col: hover_tooltips.append((size_col, f'@{size_col}{{0.00}}'))
    
    p.select(dict(type=HoverTool)).tooltips = hover_tooltips
    if legend_field:
        p.legend.location = legend_location
        p.legend.click_policy = "hide"
        p.legend.title = legend_field.replace('_',' ').title()

    if is_saving:
        save(p)
        print(f"Bokeh 散点图已保存至: {os.path.join(OUTPUT_DIR, save_filename_html)}")
    else:
        show(p)
    return p

def create_bokeh_line(data_input, x_col, y_cols_list, 
                      line_colors=Category10[10], legend_labels=None,
                      title='Bokeh 交互式折线图', 
                      xlabel=None, ylabel='值', 
                      tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                      hover_tooltips=None, legend_location="top_left",
                      save_filename_html=None, **kwargs):
    """
    使用 Bokeh 创建交互式折线图，可包含多条折线。

    参数:
    data_input (pd.DataFrame or ColumnDataSource): 输入数据。
    x_col (str): X轴列名。
    y_cols_list (list of str): Y轴列名列表，每条线一个列名。
    line_colors (list of str, optional): 每条线的颜色列表。
    legend_labels (list of str, optional): 每条线的图例标签列表。默认为y_cols_list。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。
    ylabel (str, optional): Y轴标签。
    tools (str, optional): Bokeh工具栏字符串。
    hover_tooltips (list of tuples, optional): HoverTool提示。如果None则自动生成。
    legend_location (str, optional): 图例位置。
    save_filename_html (str, optional): HTML保存文件名。
    **kwargs: 传递给 p.line 的其他参数。
    """
    source, is_saving = _prepare_bokeh_source_and_output(data_input, save_filename_html, title)
    if legend_labels is None:
        legend_labels = y_cols_list

    p = figure(title=title, tools=tools, 
               x_axis_label=xlabel if xlabel else x_col, 
               y_axis_label=ylabel)

    for i, y_col_name in enumerate(y_cols_list):
        p.line(x=x_col, y=y_col_name, source=source, 
               legend_label=str(legend_labels[i]), 
               color=line_colors[i % len(line_colors)], 
               line_width=2, **kwargs)

    if hover_tooltips is None:
        hover_tooltips = [(x_col, f'@{x_col}{{0.00a}}')] + \
                         [(str(leg), f'@{y_col}{{0.00a}}') for leg, y_col in zip(legend_labels, y_cols_list)]
    p.select(dict(type=HoverTool)).tooltips = hover_tooltips
    
    p.legend.location = legend_location
    p.legend.click_policy="hide"

    if is_saving:
        save(p)
        print(f"Bokeh 折线图已保存至: {os.path.join(OUTPUT_DIR, save_filename_html)}")
    else:
        show(p)
    return p

def create_bokeh_bar(data_input, categories_col, values_cols_list, 
                     legend_labels=None, stack=False, orientation='v', # 'v' or 'h'
                     title='Bokeh 交互式条形图', 
                     xlabel=None, ylabel='值',
                     palette=Spectral6, tools="hover,save,pan,reset",
                     hover_tooltips=None, legend_location="top_right",
                     save_filename_html=None, **kwargs):
    """
    使用 Bokeh 创建交互式条形图 (分组或堆叠)。

    参数:
    data_input (pd.DataFrame or ColumnDataSource): 输入数据。
    categories_col (str): 类别轴的列名。
    values_cols_list (list of str): 值轴的列名列表 (每个bar series一列)。
    legend_labels (list of str, optional): 图例标签。默认为values_cols_list。
    stack (bool, optional): 是否堆叠条形图。False则为分组。
    orientation (str, optional): 'v' (垂直) 或 'h' (水平)。
    title (str, optional): 图表标题。
    xlabel (str, optional): X轴标签。
    ylabel (str, optional): Y轴标签。
    palette (list of str, optional): 调色板。
    tools (str, optional): Bokeh工具。
    hover_tooltips (list of tuples, optional): 自定义悬停提示。
    legend_location (str, optional): 图例位置。
    save_filename_html (str, optional): HTML文件名。
    **kwargs: 传递给 p.vbar/hbar 或 p.vbar_stack/hbar_stack 的参数。
    """
    source, is_saving = _prepare_bokeh_source_and_output(data_input, save_filename_html, title)
    if legend_labels is None:
        legend_labels = values_cols_list
    
    cat_data = source.data[categories_col].tolist() # Bokeh x_range需要列表

    if orientation == 'v':
        p = figure(x_range=cat_data, title=title, 
                   y_axis_label=ylabel, x_axis_label=xlabel if xlabel else categories_col,
                   height=350, tools=tools)
        bar_func = p.vbar
        stack_func = p.vbar_stack
        dodge_dim = 'x'
        value_dim = 'top'
        coord_name_for_dodge = categories_col
    elif orientation == 'h':
        p = figure(y_range=cat_data, title=title, 
                   x_axis_label=ylabel, y_axis_label=xlabel if xlabel else categories_col,
                   width=500, tools=tools)
        bar_func = p.hbar
        stack_func = p.hbar_stack
        dodge_dim = 'y'
        value_dim = 'right'
        coord_name_for_dodge = categories_col
    else:
        raise ValueError("orientation 必须是 'v' 或 'h'")

    num_series = len(values_cols_list)
    bar_colors = palette[:num_series] if num_series <= len(palette) else (palette * (num_series // len(palette) + 1))[:num_series]

    if stack:
        # Bokeh的堆叠函数需要stackers参数是列名，而不是legend_labels
        stack_func(stackers=values_cols_list, **{dodge_dim: categories_col, 'width': 0.9 if orientation == 'v' else 0.9, 'height': 0.9 if orientation == 'h' else None},
                   color=bar_colors, source=source, legend_label=legend_labels, **kwargs)
    else:
        dodge_total_width = 0.8 
        bar_ind_width = dodge_total_width / num_series * 0.9
        for i, val_col_name in enumerate(values_cols_list):
            dodge_val = (i - (num_series - 1) / 2) * (dodge_total_width / num_series)
            bar_params = {
                dodge_dim: jitter(coord_name_for_dodge, width=dodge_val, range=p.x_range if orientation == 'v' else p.y_range),
                value_dim: val_col_name,
                'source': source,
                'legend_label': str(legend_labels[i]),
                'color': bar_colors[i]
            }
            if orientation == 'v': bar_params['width'] = bar_ind_width
            else: bar_params['height'] = bar_ind_width
            bar_params.update(kwargs)
            bar_func(**bar_params)
    
    if hover_tooltips is None:
        hover_tooltips = [(categories_col, f'@{categories_col}')] + \
                         [(str(leg), f'@{val_col}{{0.00a}}') for leg, val_col in zip(legend_labels, values_cols_list)]
    p.select(dict(type=HoverTool)).tooltips = hover_tooltips

    if orientation == 'v': p.xgrid.grid_line_color = None
    else: p.ygrid.grid_line_color = None
    # p.y_range.start = 0 # 可能不适用于所有条形图 (例如有负值)
    p.legend.location = legend_location
    p.legend.orientation = "horizontal" if len(legend_labels) > 2 else "vertical"
    p.legend.click_policy = "hide"

    if is_saving:
        save(p)
        print(f"Bokeh 条形图已保存至: {os.path.join(OUTPUT_DIR, save_filename_html)}")
    else:
        show(p)
    return p

def create_bokeh_grid(plots_list_of_lists, save_filename_html=None, title="Bokeh GridPlot", sizing_mode='scale_width', **kwargs):
    """
    将多个Bokeh图表排列在一个网格中。

    参数:
    plots_list_of_lists (list of lists of Plot): Bokeh图表对象的嵌套列表。
                                               例如: [[plot1, plot2], [plot3, None]]
    save_filename_html (str, optional): HTML文件名。
    title (str, optional): 网格图的整体HTML标题。
    sizing_mode (str, optional): 网格图的尺寸调整模式。
    **kwargs: 传递给 gridplot 的其他参数。
    """
    is_saving = False
    if save_filename_html:
        if not save_filename_html.lower().endswith('.html'):
            save_filename_html += '.html'
        output_file(filename=os.path.join(OUTPUT_DIR, save_filename_html), title=title)
        is_saving = True

    grid = gridplot(plots_list_of_lists, sizing_mode=sizing_mode, **kwargs)
    if is_saving:
        save(grid)
        print(f"Bokeh 网格图已保存至: {os.path.join(OUTPUT_DIR, save_filename_html)}")
    else:
        show(grid)
    return grid


# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================
def run_bokeh_demos():
    """运行所有Bokeh演示函数（使用接口化函数和示例数据）。"""
    print(f"--- Bokeh 交互式可视化接口化演示 (图表将保存到 '{OUTPUT_DIR}' 目录) ---")

    # 准备演示数据DataFrame
    np.random.seed(888)
    N_demo = 150
    demo_df = pd.DataFrame({
        '横坐标X': np.random.rand(N_demo) * 100,
        '纵坐标Y_系列1': np.random.randn(N_demo).cumsum() * 5 + 50,
        '纵坐标Y_系列2': np.random.randn(N_demo).cumsum() * 3 - 20,
        '点半径': np.random.rand(N_demo) * 8 + 2, # 2 to 10
        '颜色分类': np.random.choice(['苹果', '香蕉', '樱桃', '橙子'], N_demo),
        '形状分类': np.random.choice(['实验组', '对照组'], N_demo),
        '额外数据': np.random.normal(1000, 200, N_demo)
    })

    # 1. 散点图演示
    scatter_tips = [
        ("种类", "@颜色分类"),
        ("(X,Y)", "(@横坐标X{0.0}, @纵坐标Y_系列1{0.0})"),
        ("半径", "@点半径{0.1}"),
        ("额外", "@额外数据{0,0.00}")
    ]
    create_bokeh_scatter(demo_df, x_col='横坐标X', y_col='纵坐标Y_系列1', 
                         cat_col_for_color='颜色分类', 
                         size_col='点半径', 
                         title='Bokeh演示：散点图 (颜色按水果, 大小按半径)',
                         hover_tooltips=scatter_tips,
                         save_filename_html="demo_bokeh_scatter.html")

    # 2. 折线图演示
    line_tips = [
        ("X坐标", "@横坐标X{0.00}"),
        ("Y系列1", "@纵坐标Y_系列1{0.00}"),
        ("Y系列2", "@纵坐标Y_系列2{0.00}")
    ]
    create_bokeh_line(demo_df.sort_values('横坐标X'), # 折线图数据最好排序
                      x_col='横坐标X', y_cols_list=['纵坐标Y_系列1', '纵坐标Y_系列2'],
                      legend_labels=['Y数据1', 'Y数据2'], 
                      line_colors=[Category10[3][0], Category10[3][1]],
                      title='Bokeh演示：双折线图', hover_tooltips=line_tips,
                      save_filename_html="demo_bokeh_lines.html")

    # 3. 条形图演示 (分组和堆叠)
    # 准备条形图数据 (通常是聚合后的)
    bar_data_df = pd.DataFrame({
        '季度': ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4'],
        '产品线': ['产品A']*4 + ['产品B']*4,
        '销售额': [120, 150, 130, 180, 80, 90, 110, 100],
        '利润': [30, 40, 35, 50, 20, 25, 30, 28]
    })
    # 将数据透视以便于分组/堆叠条形图的创建
    pivoted_bar_df = bar_data_df.pivot_table(index='季度', columns='产品线', values=['销售额', '利润']).reset_index()
    # Bokeh的vbar_stack/hbar_stack和dodge需要ColumnDataSource中的列名直接对应stackers或values
    # 我们需要将多级列名扁平化
    pivoted_bar_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in pivoted_bar_df.columns.values]
    
    bar_tooltips = [("季度", "@季度")] + [(col, f"@{col}") for col in pivoted_bar_df.columns if col != '季度']
    
    create_bokeh_bar(pivoted_bar_df, categories_col='季度', 
                     values_cols_list=['销售额_产品A', '销售额_产品B'], 
                     legend_labels=['产品A销售额', '产品B销售额'], 
                     stack=False, title='Bokeh演示：分组条形图 (季度销售额)', 
                     hover_tooltips=bar_tooltips,
                     save_filename_html="demo_bokeh_bar_grouped.html")
    
    create_bokeh_bar(pivoted_bar_df, categories_col='季度', 
                     values_cols_list=['利润_产品A', '利润_产品B'], 
                     legend_labels=['产品A利润', '产品B利润'], 
                     stack=True, title='Bokeh演示：堆叠条形图 (季度利润)', 
                     palette=Viridis256[:2], # 使用不同的调色板
                     hover_tooltips=bar_tooltips, # 可以定制更具体的tooltips
                     save_filename_html="demo_bokeh_bar_stacked.html")

    # 4. 网格图演示 (GridPlot)
    # 创建几个简单的图用于网格
    p1 = figure(width=250, height=250, title="子图1: Y1 vs X")
    p1.circle(x='横坐标X', y='纵坐标Y_系列1', source=ColumnDataSource(demo_df), size=5, color="navy", alpha=0.5)
    p2 = figure(width=250, height=250, title="子图2: Y2 vs X")
    p2.line(x='横坐标X', y='纵坐标Y_系列2', source=ColumnDataSource(demo_df.sort_values('横坐标X')), line_width=2, color="firebrick")
    # 简单的条形图数据
    fruits = ['苹果', '香蕉', '橙子', '葡萄']
    counts = [5, 3, 4, 2]
    p3 = figure(x_range=fruits, width=250, height=250, title="子图3: 水果计数")
    p3.vbar(x=fruits, top=counts, width=0.9, color=Spectral6[0])
    
    grid_plots = [[p1, p2], [p3, None]] # None可以用来留空
    create_bokeh_grid(grid_plots, save_filename_html="demo_bokeh_grid.html", title="Bokeh演示：网格图布局")

    print(f"--- Bokeh 交互式可视化接口化演示完成 ---")

if __name__ == '__main__':
    run_bokeh_demos() 