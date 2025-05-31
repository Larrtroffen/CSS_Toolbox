import pandas as pd
import numpy as np

# ==============================================================================
# 接口化函数 (API-like Functions)
# ==============================================================================

def create_dataframe(data_dict):
    """
    根据字典创建Pandas DataFrame。

    参数:
    data_dict (dict): 用于创建DataFrame的字典，例如 {'col1': [1,2], 'col2': ['A','B']}。

    返回:
    pd.DataFrame: 创建的DataFrame。
    """
    return pd.DataFrame(data_dict)

def read_csv_data(file_path, **kwargs):
    """
    从CSV文件读取数据到Pandas DataFrame。

    参数:
    file_path (str): CSV文件的路径。
    **kwargs: 传递给 pd.read_csv的其他参数 (例如 sep=',', header=0)。

    返回:
    pd.DataFrame: 从CSV读取的DataFrame。
    """
    return pd.read_csv(file_path, **kwargs)

def write_csv_data(dataframe, file_path, index=False, **kwargs):
    """
    将Pandas DataFrame写入CSV文件。

    参数:
    dataframe (pd.DataFrame): 要写入的DataFrame。
    file_path (str): CSV文件的保存路径。
    index (bool, optional): 是否将DataFrame索引写入文件。默认为False。
    **kwargs: 传递给 df.to_csv的其他参数。
    """
    dataframe.to_csv(file_path, index=index, **kwargs)
    print(f"DataFrame已保存至: {file_path}")

def get_descriptive_statistics(dataframe, include_types='all', specific_columns=None, **kwargs):
    """
    计算DataFrame的描述性统计信息。

    参数:
    dataframe (pd.DataFrame): 输入的数据框。
    include_types (str or list-like, optional): 要包含的数据类型。默认为 'all'。
    specific_columns (list-like, optional): 如果提供，则只对指定列进行描述。默认为 None (所有列)。
    **kwargs: 传递给 df.describe的其他参数 (例如 percentiles=[.25, .5, .75])。

    返回:
    pd.DataFrame: 描述性统计结果。
    """
    if specific_columns:
        if not isinstance(specific_columns, list):
            specific_columns = [specific_columns]
        return dataframe[specific_columns].describe(include=include_types, **kwargs)
    return dataframe.describe(include=include_types, **kwargs)

def get_dataframe_info(dataframe, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, null_counts=None):
    """
    打印DataFrame的简明摘要，包括索引dtype、列dtypes、非空值和内存使用情况。

    参数:
    dataframe (pd.DataFrame): 输入的数据框。
    verbose (bool, optional): 是否打印完整的摘要。默认为None (根据DataFrame大小自动确定)。
    buf (writable buffer, defaults to sys.stdout): 输出信息的缓冲区。
    max_cols (int, optional): 当verbose=False时，切换到简短格式的阈值。默认为None。
    memory_usage (bool, str or None, optional): 指定是否应包括DataFrame的内存使用情况。默认为None。
    show_counts (bool, optional): 是否显示非空计数。默认为None。
    null_counts (bool, optional): 是否显示空值计数 (替代 show_counts)。默认为None。
    """
    print("\nDataFrame Info (.info()):")
    dataframe.info(verbose=verbose, buf=buf, max_cols=max_cols, memory_usage=memory_usage, show_counts=show_counts, null_counts=null_counts)

def get_value_counts(series, dropna=True, sort=True, ascending=False, normalize=False, bins=None):
    """
    计算Series中各个唯一值的频次。

    参数:
    series (pd.Series): 输入的序列。
    dropna (bool, optional): 是否排除NaN值。默认为 True。
    sort (bool, optional): 是否按频率排序。默认为 True。
    ascending (bool, optional): 如果sort=True，则按升序排序。默认为 False。
    normalize (bool, optional): 是否返回相对频率而不是绝对频率。默认为 False。
    bins (int, optional): 如果提供，则在计数之前对数据进行分箱。默认为None。

    返回:
    pd.Series: 各个值的频次统计。
    """
    return series.value_counts(dropna=dropna, sort=sort, ascending=ascending, normalize=normalize, bins=bins)

def group_and_aggregate(dataframe, group_by_columns, aggregations):
    """
    对DataFrame进行分组和聚合操作。

    参数:
    dataframe (pd.DataFrame): 输入的数据框。
    group_by_columns (str or list-like): 用于分组的列名。
    aggregations (dict): 定义聚合操作的字典。
                         键是结果列名，值可以是单个聚合函数字符串（如 'sum', 'mean'），
                         或者是一个元组 (column_to_aggregate_on, aggregation_function)。
                         示例: aggregations = {
                                   'total_col1': ('col1', 'sum'), 
                                   'mean_col3': ('col3', 'mean')
                               }

    返回:
    pd.DataFrame: 分组聚合后的结果。
    """
    if not aggregations:
        print("警告: 未指定聚合操作。请提供 aggregations 字典。")
        return dataframe.groupby(group_by_columns).size().to_frame(name='_count') # 默认返回计数
    return dataframe.groupby(group_by_columns).agg(**aggregations)

def create_pivot_table(dataframe, values, index, columns=None, aggfunc='mean', fill_value=None, margins=False, margins_name='All'):
    """
    创建透视表。

    参数:
    dataframe (pd.DataFrame): 输入的数据框。
    values (str or list-like, optional): 要聚合的列。
    index (str or list-like): 透视表的行索引。
    columns (str or list-like, optional): 透视表的列。默认为None。
    aggfunc (function, list of functions, dict, default 'mean'): 聚合函数或函数列表。
    fill_value (scalar, optional): 替换缺失值的值。默认为None。
    margins (bool, optional): 是否添加行/列边距（小计）。默认为False。
    margins_name (str, optional): 当margins=True时的边距行/列的名称。默认为'All'。

    返回:
    pd.DataFrame: 透视表。
    """
    return pd.pivot_table(dataframe, values=values, index=index, columns=columns, 
                          aggfunc=aggfunc, fill_value=fill_value, margins=margins, margins_name=margins_name)

def create_crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False):
    """
    计算两个（或多个）因子的简单交叉表。

    参数:
    index (array-like, Series, or list of arrays/Series): 用于行分组的值。
    columns (array-like, Series, or list of arrays/Series): 用于列分组的值。
    values (array-like, optional): 根据因子聚合的值数组。默认为None。
    rownames (sequence, optional): 如果传递，则必须匹配传递的行数组的数量。默认为None。
    colnames (sequence, optional): 如果传递，则必须匹配传递的列数组的数量。默认为None。
    aggfunc (function, optional): 如果指定，则还需要指定 `values`。默认为None。
    margins (bool, optional): 是否添加行/列边距（小计）。默认为False。
    margins_name (str, optional): 当margins=True时的边距行/列的名称。默认为'All'。
    dropna (bool, optional): 是否包含全部为NaN的列。默认为True。
    normalize (bool, str or {‘all’, ‘index’, ‘columns’}, optional): 通过将所有值除以值的总和来归一化。
                                                                默认为False。

    返回:
    pd.DataFrame: 交叉表。
    """
    return pd.crosstab(index, columns, values=values, rownames=rownames, colnames=colnames, 
                       aggfunc=aggfunc, margins=margins, margins_name=margins_name, dropna=dropna, normalize=normalize)

# ... (其他pandas操作，如数据清洗、合并、转换等，都可以类似地接口化)

# ==============================================================================
# 示例数据和演示函数 (保持用于独立运行和测试)
# ==============================================================================
def create_example_dataframe_for_demo():
    """仅为演示目的创建样本DataFrame。"""
    data = {
        'StudentID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Major': ['CS', 'Math', 'CS', 'Physics', 'Math', 'CS', 'Physics', 'Math', 'CS', 'Math'],
        'Score': [85, 92, 78, 88, 95, 82, 75, 90, 89, 91],
        'Attendance': [0.9, 0.95, 0.8, 0.92, 0.98, 0.85, 0.78, 0.96, 0.93, 0.97],
        'StudyHours': [10, 12, 8, 15, 18, 9, 7, 14, 16, 13],
        'EnrollDate': pd.to_datetime(['2022-09-01', '2022-09-01', '2023-01-15', '2022-09-01', '2023-01-15', 
                                     '2022-09-01', '2023-01-15', '2022-09-01', '2023-01-15', '2022-09-01'])
    }
    return pd.DataFrame(data)

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
    # 为了演示更复杂的分组，使用另一个数据集
    demo_df = create_example_dataframe_for_demo()
    print("用于分组聚合的演示数据:\n", demo_df.head())
    
    agg_rules_major_gender = {
        '平均成绩': ('Score', 'mean'),
        '最高出勤率': ('Attendance', 'max'),
        '总学习小时数': ('StudyHours', 'sum'),
        '学生人数': ('StudentID', 'count')
    }
    grouped_data = group_and_aggregate(demo_df, group_by_columns=['Major', 'Gender'], aggregations=agg_rules_major_gender)
    print("\n按'专业'和'性别'分组聚合的结果:\n", grouped_data)

    agg_rules_major = {
        '中位成绩': ('Score', 'median')
    }
    grouped_major_median_score = group_and_aggregate(demo_df, group_by_columns='Major', aggregations=agg_rules_major)
    print("\n按'专业'分组计算成绩中位数:\n", grouped_major_median_score)

    # 5. 透视表示例
    print("\n--- 5. 透视表 --- ")
    pivot_score_by_major_gender = create_pivot_table(demo_df, values='Score', index='Major', columns='Gender', aggfunc='mean', margins=True)
    print("\n按'专业'和'性别'统计的平均成绩透视表:\n", pivot_score_by_major_gender)

    pivot_studyhours_by_date_major = create_pivot_table(demo_df, 
                                                       values='StudyHours', 
                                                       index=demo_df['EnrollDate'].dt.month_name(), # 按月份名称
                                                       columns='Major', 
                                                       aggfunc='sum',
                                                       fill_value=0)
    print("\n按'入学月份'和'专业'统计的总学习小时透视表:\n", pivot_studyhours_by_date_major)

    # 6. 交叉表示例
    print("\n--- 6. 交叉表 --- ")
    # 简单交叉表：专业 vs 性别
    major_gender_crosstab = create_crosstab(demo_df['Major'], demo_df['Gender'], margins=True)
    print("\n专业 vs 性别 交叉表:\n", major_gender_crosstab)

    # 带值的交叉表：专业 vs 性别，值为平均成绩
    major_gender_score_crosstab = create_crosstab(demo_df['Major'], demo_df['Gender'], 
                                                  values=demo_df['Score'], aggfunc='mean')
    print("\n专业 vs 性别 (平均成绩) 交叉表:\n", major_gender_score_crosstab)
    
    # 归一化的交叉表 (按行)
    major_gender_crosstab_norm = create_crosstab(demo_df['Major'], demo_df['Gender'], normalize='index')
    print("\n专业 vs 性别 交叉表 (按行归一化，显示各专业内性别比例):\n", major_gender_crosstab_norm)

    print("\n--- Pandas 核心操作接口化演示完成 ---")

if __name__ == '__main__':
    run_all_pandas_demos()