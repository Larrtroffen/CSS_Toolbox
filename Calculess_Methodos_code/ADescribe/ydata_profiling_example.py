import pandas as pd
from sklearn.datasets import load_iris
# YData Profiling was formerly known as pandas-profiling
from ydata_profiling import ProfileReport
import numpy as np
import os

# ==============================================================================
# 常量和输出目录设置 (Constants and Output Directory Setup)
# ==============================================================================
OUTPUT_DIR = "autoeda_reports" # 统一存放自动EDA报告
# 此处不立即创建目录，由调用函数根据具体文件名创建，或在主演示函数中统一创建

# ==============================================================================
# 示例数据框创建 (Sample DataFrame Creation)
# ==============================================================================

def create_sample_dataframe_for_profiling(n_samples=150, include_diverse_types=True, random_state=None):
    """
    创建一个多样化的 Pandas DataFrame 用于ydata-profiling演示。

    参数:
    n_samples (int): 生成的样本数量。如果使用iris数据集且n_samples <= 150, 则iris数据会被使用。
    include_diverse_types (bool): 是否在基础数据上添加更多样的数据类型 (文本、日期、布尔、高基数等)。
    random_state (int, optional): 随机种子，用于可复现性。

    返回:
    pd.DataFrame: 生成的样本DataFrame。
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 基础数据: Iris数据集 (如果样本量合适)
    if n_samples <= 150:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data[:n_samples], columns=iris.feature_names)
        df['target_class'] = pd.Series(iris.target[:n_samples]).map({i: name for i, name in enumerate(iris.target_names)})
    else: # 如果需要更多样本，则生成合成数据
        data = np.random.rand(n_samples, 4) * 10
        df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(4)])
        df['target_class'] = np.random.choice(['ClassA', 'ClassB', 'ClassC'], n_samples)

    if include_diverse_types:
        # 添加更多样的数据类型
        df['random_numeric'] = np.random.rand(n_samples) * 100
        df['random_categorical'] = pd.Series(np.random.choice(['TypeA', 'TypeB', 'TypeC', 'TypeD', np.nan], n_samples)).astype('category')
        df['random_boolean'] = np.random.choice([True, False, np.nan], n_samples)
        df['text_data'] = [f'这是示例文本 {i}，包含一些关键词。' for i in range(n_samples)]
        df.loc[np.random.choice(df.index, size=n_samples//10, replace=False), 'random_numeric'] = np.nan # 引入一些缺失值
        df['highly_cardinal_numeric'] = np.arange(n_samples) + np.random.randn(n_samples) * 0.1 # 高基数数值
        df['constant_col'] = "固定值"
        df['date_col'] = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_samples, freq='D'))
        # 引入一些重复行
        if n_samples > 20:
            df = pd.concat([df, df.sample(n=n_samples//20, random_state=random_state)], ignore_index=True)

    print(f"已创建用于ydata-profiling的样本DataFrame，包含 {df.shape[0]} 行和 {df.shape[1]} 列。")
    # print(df.head())
    # print(df.info())
    return df

# ==============================================================================
# YData Profiling 报告生成 (YData Profiling Report Generation)
# ==============================================================================

def generate_ydata_profile_report(df, report_title="数据分析报告 (YData Profiling)", 
                                  output_html_filename="ydata_profile_report.html", 
                                  config_minimal=False, **profile_kwargs):
    """
    为给定的Pandas DataFrame生成YData Profiling (原pandas-profiling) EDA报告。

    参数:
    df (pd.DataFrame): 需要分析的输入DataFrame。
    report_title (str, optional): 报告的标题。
    output_html_filename (str, optional): 输出HTML报告的文件名 (例如 'my_report.html')。
                                       报告将保存在模块级的 OUTPUT_DIR (autoeda_reports) 目录下。
    config_minimal (bool, optional): 是否使用最小化配置生成报告 (更快，但信息较少)。默认为False。
    **profile_kwargs: 传递给 ydata_profiling.ProfileReport() 的其他参数。
                       例如: explorative=True, dark_mode=True, html={'style':{'full_width':True}}

    返回:
    str: 生成的HTML报告的完整路径。如果生成失败则返回None。
    """
    if not isinstance(df, pd.DataFrame):
        print("错误: 输入数据必须是Pandas DataFrame。")
        return None
    if df.empty:
        print("错误: 输入的DataFrame为空，无法生成报告。")
        return None

    print(f"\n--- 正在为 '{report_title}' 生成YData Profiling报告 --- ")
    print(f"数据量: {df.shape[0]} 行, {df.shape[1]} 列。这可能需要一些时间...")

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"已创建输出目录: {OUTPUT_DIR}")
        except OSError as e:
            print(f"创建输出目录 '{OUTPUT_DIR}' 失败: {e}。报告将尝试保存在当前目录。")
            final_output_path = output_html_filename
    else:
        pass # 目录已存在
    
    final_output_path = os.path.join(OUTPUT_DIR, output_html_filename)

    default_kwargs = {
        'explorative': True,
        'dark_mode': False,
        'html': {'style': {'theme': 'flatly'}} # 使用bootswatch主题示例
    }
    # 合并用户传入的kwargs，用户传入的优先
    merged_kwargs = {**default_kwargs, **profile_kwargs}

    try:
        profile = ProfileReport(df, title=report_title, config_minimal=config_minimal, **merged_kwargs)
        profile.to_file(final_output_path)
        print(f"YData Profiling报告已成功生成: {final_output_path}")
        print("请在浏览器中打开此HTML文件查看报告。")
        return final_output_path
    except Exception as e:
        print(f"生成YData Profiling报告时发生错误: {e}")
        print("请确保ydata-profiling已正确安装 ('pip install ydata-profiling[notebook]') 及其所有依赖项均满足要求。")
        print("对于非常大的或复杂的数据集，如果遇到内存问题，请考虑减少特征数量、样本量，或设置 config_minimal=True。")
        return None

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================

def run_ydata_profiling_demo():
    """运行ydata-profiling演示。"""
    print("--- YData Profiling (原pandas-profiling) 自动EDA接口化演示 ---")
    
    # 1. 创建样本数据
    sample_df = create_sample_dataframe_for_profiling(n_samples=200, random_state=42)
    
    # 2. 生成报告 (默认配置)
    generate_ydata_profile_report(sample_df, 
                                  report_title="详细EDA报告 (YData Profiling - 默认配置)", 
                                  output_html_filename="ydata_default_report.html")

    # 3. 生成报告 (最小化配置)
    # sample_df_small = create_sample_dataframe_for_profiling(n_samples=50, include_diverse_types=False, random_state=123)
    # generate_ydata_profile_report(sample_df_small, 
    #                               report_title="小型数据集EDA报告 (YData Profiling - 最小配置)", 
    #                               output_html_filename="ydata_minimal_report.html",
    #                               config_minimal=True,
    #                               dark_mode=True) # 传递额外kwargs示例

    print(f"\n--- YData Profiling演示完成。请检查 '{OUTPUT_DIR}' 目录中的HTML报告。 ---")

if __name__ == '__main__':
    run_ydata_profiling_demo() 