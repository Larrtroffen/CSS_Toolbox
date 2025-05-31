import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ==============================================================================
# 接口化绘图函数 (API-like Plotting Functions)
# ==============================================================================

# 确保输出目录存在
OUTPUT_DIR = "matplotlib_charts_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_histogram(data, bins=10, title='直方图', xlabel='值', ylabel='频率', color='skyblue', edgecolor='black', save_path=None):
    """
    生成并（可选）保存直方图。

    参数:
    data (array-like): 用于绘制直方图的数据。
    bins (int or sequence, optional): 直方图的箱数或箱的边缘。默认为10。
    title (str, optional): 图表标题。默认为'直方图'。
    xlabel (str, optional): X轴标签。默认为'值'。
    ylabel (str, optional): Y轴标签。默认为'频率'。
    color (str, optional): 直方图颜色。默认为'skyblue'。
    edgecolor (str, optional): 直方图边框颜色。默认为'black'。
    save_path (str, optional): 保存图表的路径。如果为None，则显示图表。默认为None。

    返回:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"直方图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_bar_chart(categories, values, title='条形图', xlabel='类别', ylabel='计数', horizontal=False, color='lightcoral', edgecolor='black', save_path=None):
    """
    生成并（可选）保存条形图（垂直或水平）。

    参数:
    categories (list or array-like): 条形的类别标签。
    values (list or array-like): 每个类别对应的值。
    title (str, optional): 图表标题。默认为'条形图'。
    xlabel (str, optional): X轴标签。默认为'类别'。
    ylabel (str, optional): Y轴标签。默认为'计数'。
    horizontal (bool, optional): 是否绘制水平条形图。默认为False (垂直)。
    color (str or list, optional): 条形颜色。默认为'lightcoral'。
    edgecolor (str or list, optional): 条形边框颜色。默认为'black'。
    save_path (str, optional): 保存图表的路径。如果为None，则显示图表。默认为None。

    返回:
    None
    """
    plt.figure(figsize=(8, 6))
    if horizontal:
        plt.barh(categories, values, color=color, edgecolor=edgecolor)
        plt.xlabel(ylabel) # 水平条形图时，xlabel 和 ylabel 意义互换
        plt.ylabel(xlabel)
    else:
        plt.bar(categories, values, color=color, edgecolor=edgecolor)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.title(title)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"条形图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_line_chart(x_data, y_data_list, labels, title='折线图', xlabel='X轴', ylabel='Y轴', colors=None, linestyles=None, save_path=None):
    """
    生成并（可选）保存折线图，可包含多条折线。

    参数:
    x_data (array-like): X轴数据。
    y_data_list (list of array-like): Y轴数据列表，每个元素是一条折线的数据。
    labels (list of str): 每条折线的标签。
    title (str, optional): 图表标题。默认为'折线图'。
    xlabel (str, optional): X轴标签。默认为'X轴'。
    ylabel (str, optional): Y轴标签。默认为'Y轴'。
    colors (list of str, optional): 每条折线的颜色。如果为None，使用默认颜色。
    linestyles (list of str, optional): 每条折线的样式。如果为None，使用默认样式。
    save_path (str, optional): 保存图表的路径。如果为None，则显示图表。默认为None。

    返回:
    None
    """
    plt.figure(figsize=(10, 6))
    num_lines = len(y_data_list)
    if colors is None:
        colors = [None] * num_lines # Matplotlib会自动选择颜色
    if linestyles is None:
        linestyles = ['-'] * num_lines # 默认实线

    for i in range(num_lines):
        plt.plot(x_data, y_data_list[i], label=labels[i], color=colors[i], linestyle=linestyles[i])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"折线图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_scatter_plot(x_data, y_data, title='散点图', xlabel='X值', ylabel='Y值', color='blue', marker='o', alpha=0.7, s=None, c=None, cmap=None, save_path=None):
    """
    生成并（可选）保存散点图。

    参数:
    x_data (array-like): X轴数据。
    y_data (array-like): Y轴数据。
    title (str, optional): 图表标题。默认为'散点图'。
    xlabel (str, optional): X轴标签。默认为'X值'。
    ylabel (str, optional): Y轴标签。默认为'Y值'。
    color (str, optional): 点的颜色 (如果c未指定)。默认为'blue'。
    marker (str, optional): 点的标记样式。默认为'o' (圆圈)。
    alpha (float, optional): 点的透明度。默认为0.7。
    s (scalar or array-like, optional): 点的大小。
    c (array-like or list of colors or color, optional): 点的颜色序列。
    cmap (str or Colormap, optional): 当c是浮点数数组时使用的颜色映射。
    save_path (str, optional): 保存图表的路径。如果为None，则显示图表。默认为None。

    返回:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, color=color if c is None else None, marker=marker, alpha=alpha, s=s, c=c, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"散点图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_boxplot(data_list, labels, title='箱线图', ylabel='值', showfliers=True, notch=False, vert=True, patch_artist=False, save_path=None):
    """
    生成并（可选）保存箱线图。

    参数:
    data_list (list of array-like): 用于绘制箱线图的数据列表，每个元素是一组数据。
    labels (list of str): 每个箱线的标签。
    title (str, optional): 图表标题。默认为'箱线图'。
    ylabel (str, optional): Y轴标签 (对于垂直箱线图)。默认为'值'。
    showfliers (bool, optional): 是否显示异常值。默认为True。
    notch (bool, optional): 是否绘制带缺口的箱线图。默认为False。
    vert (bool, optional): 是否绘制垂直箱线图。默认为True。
    patch_artist (bool, optional): 如果为True，则使用补丁对象填充箱体，否则使用线条绘制。默认为False。
    save_path (str, optional): 保存图表的路径。如果为None，则显示图表。默认为None。

    返回:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_list, labels=labels, showfliers=showfliers, notch=notch, vert=vert, patch_artist=patch_artist)
    plt.title(title)
    if vert:
        plt.ylabel(ylabel)
    else:
        plt.xlabel(ylabel) # 水平箱线图时，Y轴标签作为X轴标签
    plt.grid(True, axis='y' if vert else 'x', linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"箱线图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_pie_chart(sizes, labels, title='饼图', colors=None, autopct='%1.1f%%', startangle=90, explode=None, save_path=None):
    """
    生成并（可选）保存饼图。

    参数:
    sizes (array-like): 饼图各部分的大小。
    labels (list of str): 饼图各部分的标签。
    title (str, optional): 图表标题。默认为'饼图'。
    colors (list of str, optional): 各部分的颜色。如果为None，使用默认颜色。
    autopct (str or function, optional): 格式化每块百分比的字符串或函数。默认为'%1.1f%%'。
    startangle (float, optional): 起始角度。默认为90度 (从顶部开始)。
    explode (tuple of floats, optional): 如果提供，则用于突出显示某些部分。长度应与sizes相同。
    save_path (str, optional): 保存图表的路径。如果为None，则显示图表。默认为None。

    返回:
    None
    """
    plt.figure(figsize=(8, 8)) # 饼图通常为正方形
    plt.pie(sizes, labels=labels, colors=colors, autopct=autopct, startangle=startangle, explode=explode, shadow=True)
    plt.title(title)
    plt.axis('equal')  # 确保饼图是圆形的
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path))
        print(f"饼图已保存至: {os.path.join(OUTPUT_DIR, save_path)}")
    else:
        plt.show()
    plt.close()

def create_subplots_example(save_path_prefix=None):
    """
    演示如何创建包含多个子图的图形，并（可选）保存。
    这是一个更复杂的函数，用于展示子图的创建，具体子图内容需要用户根据需求定制。
    此函数主要用于演示，实际应用中用户可能需要传入更多参数来定制每个子图。

    参数:
    save_path_prefix (str, optional): 保存子图文件的前缀。例如 'my_analysis'，会保存为 'my_analysis_subplots.png'。
                                  如果为None，则显示图表。默认为None。

    返回:
    None
    """
    # 准备演示数据
    np.random.seed(42)
    data_hist = np.random.randn(100)
    data_line_x = np.linspace(0, 10, 100)
    data_line_y1 = np.sin(data_line_x)
    data_line_y2 = np.cos(data_line_x)
    data_scatter_x = np.random.rand(50)
    data_scatter_y = 2 * data_scatter_x + np.random.rand(50) * 0.5
    bar_categories = ['A', 'B', 'C', 'D']
    bar_values = [15, 30, 22, 10]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 创建2x2的子图网格
    fig.suptitle('Matplotlib 子图演示', fontsize=16)

    # 子图 1: 直方图
    axes[0, 0].hist(data_hist, bins=10, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('子图1: 直方图')
    axes[0, 0].set_xlabel('值')
    axes[0, 0].set_ylabel('频率')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # 子图 2: 折线图
    axes[0, 1].plot(data_line_x, data_line_y1, label='Sin(x)', color='green')
    axes[0, 1].plot(data_line_x, data_line_y2, label='Cos(x)', color='red', linestyle='--')
    axes[0, 1].set_title('子图2: 折线图')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    # 子图 3: 散点图
    axes[1, 0].scatter(data_scatter_x, data_scatter_y, color='purple', alpha=0.7, marker='^')
    axes[1, 0].set_title('子图3: 散点图')
    axes[1, 0].set_xlabel('X 散点')
    axes[1, 0].set_ylabel('Y 散点')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    # 子图 4: 条形图
    axes[1, 1].bar(bar_categories, bar_values, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    axes[1, 1].set_title('子图4: 条形图')
    axes[1, 1].set_xlabel('类别')
    axes[1, 1].set_ylabel('数量')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以防止标题重叠

    if save_path_prefix:
        path = os.path.join(OUTPUT_DIR, f"{save_path_prefix}_subplots.png")
        plt.savefig(path)
        print(f"子图图形已保存至: {path}")
    else:
        plt.show()
    plt.close()

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================
def run_matplotlib_demos():
    """运行所有Matplotlib演示函数（使用接口化函数和示例数据）。"""
    print(f"--- Matplotlib 可视化接口化演示 (图表将保存到 '{OUTPUT_DIR}' 目录) ---")
    
    # 准备一些通用的演示数据
    np.random.seed(123)
    hist_data = np.random.normal(loc=0, scale=1, size=1000)
    bar_cats = ['类别A', '类别B', '类别C', '类别D']
    bar_vals = [25, 40, 30, 50]
    line_x = np.arange(0, 10, 0.1)
    line_y1 = np.sin(line_x) + np.random.normal(0, 0.2, len(line_x))
    line_y2 = np.cos(line_x) + np.random.normal(0, 0.2, len(line_x))
    scatter_x_demo = np.random.rand(50) * 10
    scatter_y_demo = 0.5 * scatter_x_demo + np.random.randn(50) * 2
    boxplot_data = [np.random.normal(0, std, 100) for std in [0.5, 1.0, 1.5]]
    boxplot_labels = ['组1', '组2', '组3']
    pie_sizes_demo = [15, 30, 45, 10]
    pie_labels_demo = ['苹果', '香蕉', '樱桃', '枣子']
    pie_colors_demo = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    pie_explode_demo = (0, 0.1, 0, 0) # 突出显示第二块 (香蕉)

    # 1. 直方图
    create_histogram(hist_data, bins=30, title='示例直方图: 正态分布数据', 
                     xlabel='数据值', ylabel='频数', save_path="demo_histogram.png")

    # 2. 条形图
    create_bar_chart(bar_cats, bar_vals, title='示例垂直条形图', 
                     xlabel='产品类别', ylabel='销量', save_path="demo_bar_chart_vertical.png")
    create_bar_chart(bar_cats, bar_vals, title='示例水平条形图', 
                     xlabel='产品类别', ylabel='销量', horizontal=True, color='skyblue', 
                     save_path="demo_bar_chart_horizontal.png")

    # 3. 折线图
    create_line_chart(line_x, [line_y1, line_y2], labels=['含噪声的正弦波', '含噪声的余弦波'],
                      title='示例折线图: 模拟传感器数据', xlabel='时间 (秒)', ylabel='信号值',
                      colors=['deeppink', 'darkviolet'], linestyles=['-', '--'],
                      save_path="demo_line_chart.png")

    # 4. 散点图
    create_scatter_plot(scatter_x_demo, scatter_y_demo, title='示例散点图: X vs Y',
                        xlabel='自变量X', ylabel='因变量Y', color='green', marker='x',
                        save_path="demo_scatter_plot.png")
    # 带颜色和大小映射的散点图
    scatter_colors_mapped = scatter_x_demo # 用X值映射颜色
    scatter_sizes_mapped = np.abs(scatter_y_demo) * 10 # 用Y的绝对值映射大小
    create_scatter_plot(scatter_x_demo, scatter_y_demo, title='示例散点图: 带颜色和大小映射',
                        xlabel='自变量X', ylabel='因变量Y', s=scatter_sizes_mapped, c=scatter_colors_mapped,
                        cmap='viridis', alpha=0.6, save_path="demo_scatter_plot_mapped.png")


    # 5. 箱线图
    create_boxplot(boxplot_data, boxplot_labels, title='示例箱线图: 三组数据比较',
                   ylabel='测量值', patch_artist=True, save_path="demo_boxplot.png")
    create_boxplot(boxplot_data, boxplot_labels, title='示例水平箱线图 (带缺口)',
                   ylabel='测量值', vert=False, notch=True, save_path="demo_boxplot_horizontal.png")

    # 6. 饼图
    create_pie_chart(pie_sizes_demo, pie_labels_demo, title='示例饼图: 水果市场份额',
                     colors=pie_colors_demo, explode=pie_explode_demo,
                     save_path="demo_pie_chart.png")
    
    # 7. 子图演示
    create_subplots_example(save_path_prefix="demo")

    print(f"--- Matplotlib 可视化接口化演示完成 ---")

if __name__ == '__main__':
    run_matplotlib_demos()