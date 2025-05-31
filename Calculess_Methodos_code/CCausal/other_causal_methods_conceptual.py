import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os # For creating output directory

# --- API Functions ---

def generate_did_data_api(n_per_group: int = 100, seed: int = 101) -> pd.DataFrame:
    """
    生成用于演示Difference-in-Differences (DiD)的合成面板数据。
    包含两组 (处理组、控制组) 和两个时期 (处理前、处理后)。
    处理组仅在处理后时期接受处理。

    参数:
    - n_per_group (int): 每组（处理组/控制组在每个时期）的样本数量。
    - seed (int): 随机种子。

    返回:
    - pd.DataFrame: 生成的DiD数据集，包含 'id', 'time', 'group', 'treated_post', 'outcome' 列。
    """
    np.random.seed(seed)
    # 时间时期: 0 (处理前), 1 (处理后)
    time = np.array([0]*n_per_group + [1]*n_per_group + [0]*n_per_group + [1]*n_per_group)
    
    # 分组: 0 (控制组), 1 (处理组)
    group = np.array([0]*2*n_per_group + [1]*2*n_per_group)
    
    # 创建交互项: treated_post = 1 如果 group=1 且 time=1, 否则为 0
    treated_post_interaction = ((group == 1) & (time == 1)).astype(int)
    
    # 结果 Y
    # Y = intercept + time_effect + group_effect + did_effect*treated_post + noise
    intercept = 10
    time_effect_val = 2  # 一般时间趋势
    group_specific_effect = 3 # 组间基线差异
    true_did_effect = 5   # 真实的处理效应
    
    Y = intercept + time_effect_val * time + group_specific_effect * group + true_did_effect * treated_post_interaction + np.random.normal(0, 1.5, 4*n_per_group)
    
    df = pd.DataFrame({
        'id': np.tile(np.arange(2*n_per_group), 2), # 个体ID
        'time': time,          # 0 为处理前, 1 为处理后
        'group': group,        # 0 为控制组, 1 为处理组
        'treated_post': treated_post_interaction, # DiD的交互项
        'outcome': Y
    })
    # print(f"API: 已生成DiD数据集，形状: {df.shape}")
    return df

def run_did_example_api(df_did: pd.DataFrame, output_dir: str = "other_causal_outputs_demo") -> tuple[sm.regression.linear_model.RegressionResults | None, str | None]:
    """
    使用statsmodels对提供的DiD数据运行一个简单的OLS回归模型，并绘制结果图。

    参数:
    - df_did (pd.DataFrame): 包含 'outcome', 'time', 'group', 'treated_post' 列的DiD数据。
    - output_dir (str): 保存图表的目录。

    返回:
    - tuple[sm.regression.linear_model.RegressionResults | None, str | None]: 
        statsmodels拟合的模型结果对象 (如果成功)，以及图表保存路径 (如果成功)。
    """
    print("\n--- 运行DiD示例 (API) ---")
    if not all(col in df_did.columns for col in ['outcome', 'time', 'group', 'treated_post']):
        print("  错误: DiD数据不完整，缺少必要的列。")
        return None, None

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"  创建输出目录: {output_dir}")
        except OSError as e:
            print(f"  创建目录 {output_dir} 失败: {e}")
            return None, None #无法保存图片则退出

    plot_path = None
    model_results = None
    try:
        did_model = smf.ols('outcome ~ time + group + treated_post', data=df_did).fit()
        print("  DiD模型摘要 (statsmodels OLS):")
        print(did_model.summary())
        model_results = did_model
        did_estimate = did_model.params.get('treated_post', np.nan)
        print(f"  估计的DiD效应 (treated_post的系数): {did_estimate:.4f} (此演示中真实值为5.0)")
        
        # 可视化均值图
        means = df_did.groupby(['time', 'group'])['outcome'].mean().unstack()
        fig, ax = plt.subplots(figsize=(8,5))
        means.plot(ax=ax, marker='o')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['处理前', '处理后'])
        ax.set_title('各组别和时间段的平均结果 (DiD)')
        ax.set_ylabel('平均结果')
        ax.legend(title='组别', labels=['控制组 (0)', '处理组 (1)'])
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plot_path = os.path.join(output_dir, "did_demo_plot.png")
        plt.savefig(plot_path)
        print(f"  DiD图表已保存至: {plot_path}")
        plt.close(fig)

    except Exception as e:
        print(f"  运行DiD示例时出错: {e}")
        return None, None # 如果模型或绘图失败
        
    return model_results, plot_path


def explain_did_conceptual_api():
    """打印Difference-in-Differences (DiD) 的概念性解释。"""
    print("\n--- 概念解释: Difference-in-Differences (DiD) (API) ---")
    print("DiD通过比较处理组和控制组在一段时间内结果的变化来估计处理的因果效应。")
    print("关键假设: 平行趋势 (Parallel Trends) - 即在没有处理的情况下，")
    print("处理组的结果均值变化趋势应与控制组相同。")
    
    print("\n基本DiD模型方程: Y_it = β0 + β1*Time_t + β2*TreatGroup_i + β3*(Time_t * TreatGroup_i) + ε_it")
    print("  - Y_it: 个体i在时间t的结果。")
    print("  - Time_t: 处理后时期的虚拟变量 (处理后为1, 处理前为0)。")
    print("  - TreatGroup_i: 处理组的虚拟变量 (处理组为1, 控制组为0)。")
    print("  - (Time_t * TreatGroup_i): 时间和处理组的交互项。其系数 (β3) 即为DiD估计量。")

    print("\nDiD的注意事项:")
    print("- 稳健标准误: 例如，按组/个体进行聚类调整。statsmodels可以处理。")
    print("- 交错采纳 (Staggered Adoption): 不同单元在不同时间开始处理 -> 需要更复杂的模型 (如Callaway & Sant'Anna)。相关库: R语言的`DiD`包, Python的`didpy`。")
    print("- 事件研究图 (Event Study Plots): 用于检查处理前的平行趋势和动态效应。")
    print("- 加入协变量 (Covariates): 如果平行趋势假设仅在控制了某些协变量后才成立，则需要在模型中加入这些协变量。")


def explain_rdd_conceptual_api():
    """打印Regression Discontinuity Design (RDD) 的概念性解释。"""
    print("\n--- 概念解释: Regression Discontinuity Design (RDD) (API) ---")
    print("RDD适用于处理分配取决于一个可观测的连续变量 (通常称为“分配变量”或“驱动变量”) 是否超过一个特定断点的情况。")
    print("核心思想: 恰好在断点以上和以下的个体在其他方面非常相似，从而在断点附近形成了一个局部随机实验。")
    print("关键假设: 潜在结果的条件期望函数在断点处是连续的。")
    
    print("\nRDD类型:")
    print("- 精确RDD (Sharp RDD): 处理状态在断点处确定性地改变。")
    print("- 模糊RDD (Fuzzy RDD): 处理的概率在断点处发生不连续变化 (但不是从0变为1，或从1变为0)。")

    print("\n精确RDD概念模型 (局部线性回归):")
    print("  Y_i = β0 + β1*(X_i - c) + β2*T_i + β3*T_i*(X_i - c) + ε_i")
    print("    - Y_i: 个体i的结果。")
    print("    - X_i: 个体i的分配变量。")
    print("    - c: 断点值。")
    print("    - T_i: 处理虚拟变量 (对于精确RDD，如果 X_i >= c 则为1, 否则为0)。")
    print("    - (X_i - c): 以断点为中心化的分配变量。")
    print("    - β2 是RDD的估计量 (在断点处的处理效应)。")
    print("    - 估计通常在断点附近选定的带宽内的数据上进行。")

    print("\nRDD实现注意事项:")
    print("- Python库: `rdrobust` (R包的Python封装), `rddpy` (较早的库), `statsmodels` (可手动实现)。")
    print("- 带宽选择至关重要 (例如，使用`rdrobust.rdbwselect`进行Imbens-Kalyanaraman最优带宽选择)。")
    print("- 绘制断点图对于视觉检查非常重要。")
    print("- 安慰剂检验 (Placebo tests): 例如，在分配变量的其他点检查是否存在“伪断点效应”。")
    
    rdd_code_example = (
        "# RDD的statsmodels简化概念代码 (仅为说明，非完整实现):\n"
        "# 假设df_rdd包含列: 'outcome', 'running_var', 'cutoff_value'已定义\n"
        "# df_rdd[\'centered_running_var\'] = df_rdd[\'running_var\'] - cutoff_value\n"
        "# df_rdd[\'treatment_dummy\'] = (df_rdd[\'running_var\'] >= cutoff_value).astype(int) # 精确RDD\n"
        "# df_rdd[\'treat_x_centered\'] = df_rdd[\'treatment_dummy\'] * df_rdd[\'centered_running_var\']\n"
        "# # 选择带宽 h (例如，通过rdbwselect得到)\n"
        "# df_local = df_rdd[np.abs(df_rdd[\'centered_running_var\']) <= h]\n"
        "# if not df_local.empty:\n"
        "#     rdd_formula = 'outcome ~ centered_running_var + treatment_dummy + treat_x_centered'\n"
        "#     rdd_model = smf.ols(rdd_formula, data=df_local).fit()\n"
        "#     print(rdd_model.summary())\n"
        "#     rdd_effect = rdd_model.params.get(\'treatment_dummy\')\n"
        "#     print(f\"RDD效应估计: {rdd_effect:.4f}\")\n"
        "# else: print(\"带宽内无数据。\")"
    )
    print("\nRDD Python概念代码片段 (说明性):")
    print(rdd_code_example)


def explain_causal_discovery_conceptual_api():
    """打印因果发现 (Causal Discovery) 的概念性解释。"""
    print("\n--- 概念解释: 因果发现 (Causal Discovery) (API) ---")
    print("因果发现算法旨在从观测数据中推断因果结构 (即因果图)。")
    print("这是一个复杂的问题，通常依赖于强假设 (例如，因果马尔可夫条件、忠实性假设、因果充分性等)。")
    
    print("\n常见算法类型:")
    print("- 基于约束的算法 (例如 PC, FCI):")
    print("    - 从一个全连接的图开始。")
    print("    - 进行条件独立性检验以移除边。")
    print("    - 基于d-分离规则确定部分边的方向。")
    print("    - PC (Peter-Clark) 算法假设没有潜在混杂因子。")
    print("    - FCI (Fast Causal Inference) 算法可以处理潜在混杂因子 (输出可能包含如 A <-> B 或 A o-o B 的边)。")
    print("- 基于分数的算法 (例如 GES - Greedy Equivalence Search):")
    print("    - 定义一个评分函数 (如 BIC, BDeu)，衡量一个图与数据的拟合程度。")
    print("    - 在所有可能的图空间中搜索，以找到使分数最大化的图。")
    print("- 函数型因果模型 (例如 LiNGAM - Linear Non-Gaussian Acyclic Model):")
    print("    - 假设因果关系具有特定的函数形式 (如线性)，并且噪声是非高斯的。")
    print("    - 通常能够识别出完整的因果方向。")

    print("\nPython中的因果发现库:")
    print("- `pgmpy`: 实现了多种学习贝叶斯网络的算法 (在一定假设下可解释为因果图)，例如PC算法、基于分数的算法。")
    print("- `cdt` (Causal Discovery Toolbox): 封装了许多R语言包，并提供了自己的实现 (如 PC, GES, LiNGAM, NOTEARS)。")
    print("- `gcastle`: 一个较新的库，包含多种因果发现算法的实现。")

    causal_discovery_pgmpy_conceptual = (
        "# pgmpy中PC算法的概念代码片段 (说明性):\n"
        "# from pgmpy.estimators import PC\n"
        "# # 假设 data_df 是包含变量的Pandas DataFrame\n"
        "# estimator_pc = PC(data=data_df)\n"
        "# try:\n"
        "#     # variant参数可选 'stable', 'parallel'等。ci_test指定条件独立检验方法，significance_level为显著性水平。
"
        "#     # estimated_model_pc = estimator_pc.estimate(variant='stable', ci_test='chi_square', significance_level=0.05)\n"
        "#     # # 对于较新版本的pgmpy，输出可能是DAG或PAG，取决于假设和算法变体
"
        "#     # # 示例：构建骨架 (无向图)
"
        "#     # skeleton, _ = estimator_pc.build_skeleton(ci_test='chi_square', significance_level=0.05)\n"
        "#     # print(f\"PC算法估计的因果骨架边: {skeleton.edges()}\')\n"
        "#     # # 完整模型估计 (可能返回CPDAG或PAG)
"
        "#     model = estimator_pc.estimate(variant=\"stable\", significance_level=0.05)\n"
        "#     print(f\"PC算法估计的模型边: {model.edges()}\')\n"
        "# except Exception as e_cd:\n"
        "#     print(f\"pgmpy PC算法示例出错: {e_cd}\")"
    )
    print("\npgmpy PC算法概念代码片段 (说明性):")
    print(causal_discovery_pgmpy_conceptual)

def explain_mediation_analysis_conceptual_api():
    """打印中介分析 (Mediation Analysis) 的概念性解释。"""
    print("\n--- 概念解释: 中介分析 (Mediation Analysis) (API) ---")
    print("中介分析用于研究一个自变量 (X) 如何通过一个或多个中介变量 (M) 影响因变量 (Y)。")
    print("它旨在分解X对Y的总效应，为直接效应 (X -> Y) 和间接效应 (X -> M -> Y)。")
    
    print("\n经典方法 (Baron & Kenny, 1986) 的步骤:")
    print("  1. 检验X对Y的总效应 (路径c): Y = i1 + c*X + e1")
    print("  2. 检验X对M的效应 (路径a): M = i2 + a*X + e2")
    print("  3. 检验M对Y的效应，同时控制X (路径b): Y = i3 + c'*X + b*M + e3")
    print("     - 如果a和b都显著，则存在中介效应。")
    print("     - 间接效应估计为 a*b。")
    print("     - c' 是控制了M后X对Y的直接效应。")
    print("     - 如果c'不显著，则为完全中介；如果c'显著但小于c，则为部分中介。")

    print("\n现代方法与考虑:")
    print("- 统计检验间接效应 (a*b): Sobel检验曾被使用，但现在更推荐使用Bootstrap方法获取置信区间，因为它对a*b的分布不作正态假设。")
    print("- 多重中介: 当有多个中介变量时，模型会更复杂。")
    print("- 纵向中介: 在纵向数据中考察中介效应。")
    print("- 因果中介分析: 试图在潜在结果框架下更严格地定义和估计直接与间接效应，处理混杂等问题。需要更强的假设。")
    
    print("\nPython中的实现:")
    print("- `statsmodels`: 可以手动拟合上述回归模型并计算效应。")
    print("- `pingouin`: 提供了 `mediation_analysis` 函数，可以执行基于回归的中介分析并提供Bootstrap结果。")
    print("- `pyprocessmacro`: Andrew Hayes的PROCESS宏的Python实现 (仍在开发中，或通过Rpy2调用R的PROCESS)。")

    mediation_pingouin_conceptual = (
        "# pingouin中介分析概念代码片段 (说明性):\n"
        "# import pingouin as pg\n"
        "# # 假设 data_df 包含 'X', 'M', 'Y' 列\n"
        "# try:\n"
        "#     mediation_results = pg.mediation_analysis(data=data_df, x='X', m='M', y='Y', alpha=0.05, n_boot=1000) # n_boot建议至少1000-5000
"
        "#     print(\"\nPingouin中介分析结果 (概念性):\")\n"
        "#     print(mediation_results)\n"
        "# except Exception as e_med:\n"
        "#     print(f\"Pingouin中介分析示例出错: {e_med}\")"
    )
    print("\nPingouin中介分析概念代码片段 (说明性):")
    print(mediation_pingouin_conceptual)

if __name__ == '__main__':
    print("========== (C 部分) 其他因果推断方法概念演示 ==========")
    output_dir_conceptual = "other_causal_outputs_main_demo"
    if not os.path.exists(output_dir_conceptual):
        os.makedirs(output_dir_conceptual)
        print(f"创建主输出目录: {output_dir_conceptual}")

    # --- 1. DiD 演示 ---
    explain_did_conceptual_api()
    print("\n  --- 运行DiD实际示例 ---")
    df_did_data_demo = generate_did_data_api(n_per_group=150, seed=202)
    print(f"  生成DiD数据，形状: {df_did_data_demo.shape}")
    did_model_res, did_plot_file = run_did_example_api(df_did_data_demo, output_dir=output_dir_conceptual)
    if did_model_res:
        print(f"  DiD模型已拟合。图表保存于: {did_plot_file if did_plot_file else '未生成'}")
    else:
        print("  DiD示例运行失败或未生成模型/图表。")
    print("  --- DiD实际示例结束 ---")

    # --- 2. RDD 概念解释 ---
    explain_rdd_conceptual_api()

    # --- 3. 因果发现 概念解释 ---
    explain_causal_discovery_conceptual_api()

    # --- 4. 中介分析 概念解释 ---
    explain_mediation_analysis_conceptual_api()

    print("\n\n========== (C 部分) 其他因果推断方法概念演示结束 ==========")
    print(f"注意: DiD图表示例 (如果成功) 已保存在 '{output_dir_conceptual}' 目录中。其他部分主要是概念解释。")