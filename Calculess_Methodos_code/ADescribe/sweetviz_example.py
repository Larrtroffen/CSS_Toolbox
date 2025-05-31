import pandas as pd
import sweetviz as sv
from sklearn.datasets import load_iris
import numpy as np
import os

# ==============================================================================
# 常量和输出目录设置 (Constants and Output Directory Setup)
# ==============================================================================
OUTPUT_DIR = "autoeda_reports" # 统一存放自动EDA报告

# ==============================================================================
# 示例数据框创建 (Sample DataFrame Creation)
# ==============================================================================

def create_sample_dataframes_for_sweetviz(base_n_samples=150, random_state=None):
    """
    为Sweetviz演示创建样本DataFrames，包括一个用于比较的修改版DataFrame。

    参数:
    base_n_samples (int): 基础DataFrame的样本数量。
    random_state (int, optional): 随机种子，用于可复现性。

    返回:
    tuple: (df_base, df_modified)
        df_base (pd.DataFrame): 基础的样本DataFrame。
        df_modified (pd.DataFrame): 修改后的样本DataFrame，用于比较。
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 创建基础DataFrame (df1)
    if base_n_samples <= 150:
        iris = load_iris()
        df1 = pd.DataFrame(data=iris.data[:base_n_samples], columns=iris.feature_names)
        df1['species'] = pd.Series(iris.target[:base_n_samples]).map({i: name for i, name in enumerate(iris.target_names)})
    else:
        data1 = np.random.rand(base_n_samples, 4) * np.array([8, 4, 7, 3]) # Different scales
        df1 = pd.DataFrame(data1, columns=[f'feature_num_{i+1}' for i in range(4)])
        df1['species'] = np.random.choice(['Type_X', 'Type_Y', 'Type_Z'], base_n_samples)
    
    df1['random_value'] = np.random.randn(base_n_samples) * 20 + 50
    df1['boolean_flag'] = np.random.choice([True, False, None], base_n_samples, p=[0.45, 0.45, 0.1])
    df1['text_field'] = [f'Sample text entry {i} with some variation' for i in range(base_n_samples)]
    # 引入一些缺失值
    nan_indices_num = np.random.choice(df1.index, size=base_n_samples//10, replace=False)
    if df1.columns[0].startswith('sepal'): # Iris based
         df1.loc[nan_indices_num, 'sepal width (cm)'] = np.nan
    else:
        df1.loc[nan_indices_num, 'feature_num_2'] = np.nan
    df1.loc[np.random.choice(df1.index, size=base_n_samples//15, replace=False), 'boolean_flag'] = np.nan

    print(f"已创建基础DataFrame (df1)，包含 {df1.shape[0]} 行和 {df1.shape[1]} 列。")

    # 创建用于比较的修改版DataFrame (df2)
    df2 = df1.copy()
    # 修改数值列
    if df2.columns[0].startswith('sepal'): # Iris based
        df2['sepal length (cm)'] = df2['sepal length (cm)'] * np.random.uniform(0.8, 1.2, size=len(df2))
        df2['petal width (cm)'] = df2['petal width (cm)'] + np.random.normal(0, 0.1, size=len(df2))
    else:
        df2['feature_num_1'] = df2['feature_num_1'] * np.random.uniform(0.7, 1.3, size=len(df2))
        df2['feature_num_3'] = df2['feature_num_3'] + np.random.normal(0, 5, size=len(df2))
    
    # 修改分类/目标列
    if 'species' in df2.columns:
        modification_map = {df1['species'].unique()[0]: str(df1['species'].unique()[0]) + '_MODIFIED'}
        df2['species'] = df2['species'].apply(lambda x: modification_map.get(x, x))
        # 随机改变一些值
        change_indices = np.random.choice(df2.index, size=len(df2)//8, replace=False)
        if len(df1['species'].unique()) > 1:
            df2.loc[change_indices, 'species'] = df1['species'].unique()[1]
        else: # Handle case with only one class
            df2.loc[change_indices, 'species'] = str(df1['species'].unique()[0]) + '_CHANGED'
            
    # 引入/移除一些缺失值
    df2.loc[np.random.choice(df2.index, size=len(df2)//12, replace=False), 'random_value'] = np.nan
    if df2.columns[1].startswith('sepal'):
        df2.loc[df2['sepal width (cm)'].isnull().sample(frac=0.3).index, 'sepal width (cm)'] = np.random.rand() * 4 # Fill some NaNs
    elif 'feature_num_2' in df2.columns:
         df2.loc[df2['feature_num_2'].isnull().sample(frac=0.3).index, 'feature_num_2'] = np.random.rand() * 4

    # 对数据进行抽样，使其与df1不完全相同
    df2 = df2.sample(frac=np.random.uniform(0.75, 0.95), random_state=random_state if random_state else None).reset_index(drop=True)

    print(f"已创建修改版DataFrame (df2) 用于比较，包含 {df2.shape[0]} 行和 {df2.shape[1]} 列。")
    return df1, df2

# ==============================================================================
# Sweetviz 报告生成 (Sweetviz Report Generation)
# ==============================================================================

def generate_sweetviz_report_single(df, report_title="Sweetviz 单数据集分析报告", 
                                    output_html_filename="sweetviz_single_report.html",
                                    target_feature=None, **report_kwargs):
    """
    为单个Pandas DataFrame生成Sweetviz EDA报告。

    参数:
    df (pd.DataFrame): 需要分析的输入DataFrame。
    report_title (str, optional): 报告的显示标题 (在HTML内部，非文件名)。
    output_html_filename (str, optional): 输出HTML报告的文件名 (例如 'my_single_report.html')。
                                       报告将保存在模块级的 OUTPUT_DIR (autoeda_reports) 目录下。
    target_feature (str, optional): 目标特征的列名。如果提供，Sweetviz将围绕它进行分析。
    **report_kwargs: 传递给 sweetviz.analyze() 或 report.show_html() 的其他参数。
                     例如: feature_config, layout='widescreen', scale=0.9

    返回:
    str: 生成的HTML报告的完整路径。如果生成失败则返回None。
    """
    if not isinstance(df, pd.DataFrame):
        print("错误: 输入数据必须是Pandas DataFrame。")
        return None
    if df.empty:
        print("错误: 输入的DataFrame为空，无法生成报告。")
        return None

    print(f"\n--- 正在为 '{report_title}' 生成Sweetviz单数据集报告 --- ")
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"已创建输出目录: {OUTPUT_DIR}")
        except OSError as e:
            print(f"创建输出目录 '{OUTPUT_DIR}' 失败: {e}")
    final_output_path = os.path.join(OUTPUT_DIR, output_html_filename)

    # 分离 show_html 参数
    show_html_params = {k: report_kwargs.pop(k) for k in ['layout', 'scale', 'open_browser'] if k in report_kwargs}
    if 'open_browser' not in show_html_params: # 默认不自动打开浏览器
        show_html_params['open_browser'] = False
    if 'layout' not in show_html_params:
        show_html_params['layout'] = 'widescreen'

    try:
        report_object = sv.analyze(source=df, target_feat=target_feature, **report_kwargs)
        report_object.show_html(filepath=final_output_path, **show_html_params)
        # Sweetviz 的 show_html 会自动设置一个标题，但我们可以用 report_title 进一步说明
        print(f"Sweetviz报告 ('{report_title}') 已成功生成: {final_output_path}")
        print("请在浏览器中打开此HTML文件查看报告。")
        return final_output_path
    except Exception as e:
        print(f"生成Sweetviz单数据集报告 ('{report_title}') 时发生错误: {e}")
        print("请确保Sweetviz已正确安装 ('pip install sweetviz')。")
        return None

def generate_sweetviz_report_comparison(df_source, df_compare, 
                                        source_name="数据集1", compare_name="数据集2",
                                        report_title="Sweetviz 双数据集对比报告", 
                                        output_html_filename="sweetviz_comparison_report.html",
                                        target_feature=None, **report_kwargs):
    """
    为两个Pandas DataFrame生成Sweetviz对比EDA报告。

    参数:
    df_source (pd.DataFrame): 源/基础DataFrame。
    df_compare (pd.DataFrame): 用于对比的DataFrame。
    source_name (str, optional): 源DataFrame在报告中的名称。
    compare_name (str, optional): 对比DataFrame在报告中的名称。
    report_title (str, optional): 报告的显示标题。
    output_html_filename (str, optional): 输出HTML报告的文件名。
    target_feature (str, optional): 目标特征的列名。
    **report_kwargs: 传递给 sweetviz.compare() 或 report.show_html() 的其他参数。

    返回:
    str: 生成的HTML报告的完整路径。如果生成失败则返回None。
    """
    if not (isinstance(df_source, pd.DataFrame) and isinstance(df_compare, pd.DataFrame)):
        print("错误: 输入数据必须都是Pandas DataFrame。")
        return None
    if df_source.empty or df_compare.empty:
        print("错误: 输入的DataFrame中至少有一个为空，无法生成对比报告。")
        return None

    print(f"\n--- 正在为 '{report_title}' 生成Sweetviz双数据集对比报告 --- ")
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
        except OSError as e:
             print(f"创建输出目录 '{OUTPUT_DIR}' 失败: {e}")
    final_output_path = os.path.join(OUTPUT_DIR, output_html_filename)

    show_html_params = {k: report_kwargs.pop(k) for k in ['layout', 'scale', 'open_browser'] if k in report_kwargs}
    if 'open_browser' not in show_html_params:
        show_html_params['open_browser'] = False
    if 'layout' not in show_html_params:
        show_html_params['layout'] = 'widescreen'

    try:
        compare_report_object = sv.compare(source=[df_source, source_name], 
                                           compare=[df_compare, compare_name], 
                                           target_feat=target_feature, 
                                           **report_kwargs)
        compare_report_object.show_html(filepath=final_output_path, **show_html_params)
        print(f"Sweetviz对比报告 ('{report_title}') 已成功生成: {final_output_path}")
        print("请在浏览器中打开此HTML文件查看报告。")
        return final_output_path
    except Exception as e:
        print(f"生成Sweetviz对比报告 ('{report_title}') 时发生错误: {e}")
        print("请确保Sweetviz已正确安装 ('pip install sweetviz')。")
        return None

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================

def run_sweetviz_demos():
    """运行Sweetviz的各种演示。"""
    print("--- Sweetviz 自动EDA接口化演示 ---")

    # 1. 创建样本数据
    df_main, df_comp = create_sample_dataframes_for_sweetviz(base_n_samples=180, random_state=101)
    
    target_col = None
    if 'species' in df_main.columns: # 检查是否存在目标列
        target_col = 'species'
    elif any(col.startswith('target') for col in df_main.columns):
        target_col = [col for col in df_main.columns if col.startswith('target')][0]

    # 2. 单个DataFrame分析报告
    generate_sweetviz_report_single(df_main, 
                                    report_title="Sweetviz主数据集分析报告", 
                                    output_html_filename="demo_sweetviz_main_report.html",
                                    target_feature=target_col,
                                    # feature_config=sv.FeatureConfig(skip=['text_field'], force_num=['random_value'])
                                    )

    # 3. 两个DataFrame对比报告
    generate_sweetviz_report_comparison(df_main, df_comp, 
                                        source_name="主数据集 (df_main)", 
                                        compare_name="对比数据集 (df_comp)",
                                        report_title="Sweetviz 主数据集 vs 对比数据集", 
                                        output_html_filename="demo_sweetviz_main_vs_comp_report.html",
                                        target_feature=target_col)
    
    # 4. 无目标特征的分析
    # generate_sweetviz_report_single(df_main.drop(columns=[target_col] if target_col else []), 
    #                                 report_title="Sweetviz主数据集分析报告 (无目标特征)", 
    #                                 output_html_filename="demo_sweetviz_no_target_report.html")

    print(f"\n--- Sweetviz演示完成。请检查 '{OUTPUT_DIR}' 目录中的HTML报告。 ---")

if __name__ == '__main__':
    run_sweetviz_demos() 