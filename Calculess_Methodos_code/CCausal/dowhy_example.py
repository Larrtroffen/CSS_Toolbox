import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
import os
# import dowhy.datasets # DoWhy自带数据集生成器，但我们这里自定义一个更灵活的

# 可选: 用于可视化因果图 (如果环境中有graphviz)
# import graphviz 

# --- API 函数 ---

def generate_dowhy_data_api(
    n_samples: int = 1000, 
    seed: int = 42,
    treatment_type: str = 'continuous', # 'continuous' 或 'binary'
    outcome_type: str = 'continuous'   # 'continuous' 或 'binary'
) -> tuple[pd.DataFrame, str, str, str, str | None]:
    """
    生成用于DoWhy因果推断演示的合成数据集。

    包含基本因果结构:
    - W: 混杂因子 (Confounder), W -> T 且 W -> Y
    - Z: 工具变量 (Instrument), Z -> T (且不直接影响Y，除非通过T)
    - T: 处理变量 (Treatment)
    - Y: 结果变量 (Outcome)

    参数:
    - n_samples (int): 生成的样本数量。
    - seed (int): 随机种子，用于可复现性。
    - treatment_type (str): 'continuous' 或 'binary'。
    - outcome_type (str): 'continuous' 或 'binary'。

    返回:
    - pd.DataFrame: 包含 W, Z, T, Y 的数据集。
    - str: 混杂因子列名 ('W')。
    - str: 工具变量列名 ('Z')。
    - str: 处理变量列名 (根据 treatment_type 决定，例如 'T_continuous' 或 'T_binary')。
    - str: 结果变量列名 (根据 outcome_type 决定，例如 'Y_continuous' 或 'Y_binary')。
    """
    np.random.seed(seed)
    
    W_col = 'W' # 混杂因子
    Z_col = 'Z' # 工具变量
    
    W = np.random.normal(0, 1, n_samples)
    Z = np.random.normal(0, 1, n_samples)
    
    # 处理变量 T (受W和Z影响)
    # 真实的处理效应系数 (T对Y) 将是2.0
    # W对T的效应是0.5, Z对T的效应是0.8
    T_latent = 0.5 * W + 0.8 * Z + np.random.normal(0, 0.5, n_samples)
    
    if treatment_type == 'binary':
        T_data = (T_latent > np.median(T_latent)).astype(int)
        T_col_name = 'T_binary'
    else: # 默认为连续
        T_data = T_latent
        T_col_name = 'T_continuous'
        
    # 结果变量 Y (受T和W影响)
    # T对Y的真实因果效应是 2.0
    # W对Y的真实因果效应是 1.5
    Y_latent = 2.0 * T_data + 1.5 * W + np.random.normal(0, 1, n_samples)
    
    if outcome_type == 'binary':
        Y_data = (Y_latent > np.median(Y_latent)).astype(int)
        Y_col_name = 'Y_binary'
    else: # 默认为连续
        Y_data = Y_latent
        Y_col_name = 'Y_continuous'
        
    df = pd.DataFrame({
        W_col: W,
        Z_col: Z,
        T_col_name: T_data,
        Y_col_name: Y_data
    })
    # print(f"API: 已生成DoWhy数据集，形状: {df.shape}, 处理: {T_col_name}, 结果: {Y_col_name}")
    return df, W_col, Z_col, T_col_name, Y_col_name


def define_causal_model_api(
    df: pd.DataFrame, 
    treatment_name: str, 
    outcome_name: str, 
    common_causes_names: list[str] | None = None, 
    instrument_names: list[str] | None = None, 
    graph_dot_string: str | None = None,
    proceed_when_unidentifiable: bool = True # DoWhy 0.9+ 新参数
) -> CausalModel | None:
    """
    使用DoWhy定义因果模型。

    参数:
    - df (pd.DataFrame): 输入数据。
    - treatment_name (str): 处理变量名。
    - outcome_name (str): 结果变量名。
    - common_causes_names (list[str] | None): 共同原因 (混杂因子) 列表。
    - instrument_names (list[str] | None): 工具变量列表。
    - graph_dot_string (str | None): 可选的DOT格式字符串，用于直接定义因果图。
                                      如果提供，common_causes和instruments参数将被忽略。
    - proceed_when_unidentifiable (bool): 即使模型不可识别也继续的标志(DoWhy v0.9+)。

    返回:
    - CausalModel | None: 定义好的DoWhy CausalModel对象，如果失败则为None。
    """
    print(f"\n--- 定义因果模型 (API): {treatment_name} -> {outcome_name} ---")
    try:
        if graph_dot_string:
            model = CausalModel(
                data=df,
                treatment=treatment_name,
                outcome=outcome_name,
                graph=graph_dot_string,
                proceed_when_unidentifiable=proceed_when_unidentifiable
            )
            print("  使用DOT字符串图定义因果模型成功。")
        else:
            model = CausalModel(
                data=df,
                treatment=treatment_name,
                outcome=outcome_name,
                common_causes=common_causes_names,
                instrument_names=instrument_names,
                proceed_when_unidentifiable=proceed_when_unidentifiable
            )
            print("  使用变量角色 (common_causes, instruments) 定义因果模型成功。")
        
        # 可选: 显示/保存因果图 (需要graphviz库)
        # try:
        #     model.view_model(layout="dot", file_name="dowhy_causal_graph_api") 
        #     print("  因果图已尝试显示或保存为 'dowhy_causal_graph_api.png' (需安装graphviz)。")
        # except Exception as e_graph:
        #     print(f"  无法可视化因果图 (可能未安装graphviz或配置问题): {e_graph}")
        # print("  因果图 (DOT格式):\n", model._graph.to_dot())
        return model
    except Exception as e:
        print(f"  定义因果模型失败: {e}")
        return None


def identify_causal_effect_api(model: CausalModel, proceed_when_unidentifiable: bool = True) -> dowhy. παιδιά.Estimand | None:
    """
    使用定义的因果模型识别因果效应的估计量 (estimand)。

    参数:
    - model (CausalModel): DoWhy因果模型对象。
    - proceed_when_unidentifiable (bool): 如果效应不可识别，是否继续。

    返回:
    - dowhy. παιδιά.Estimand | None: 识别出的估计量对象，如果失败或不可识别 (且proceed_when_unidentifiable=False) 则为None。
    """
    print("\n--- 识别因果效应 (API) ---")
    if not model:
        print("  错误: 无效的因果模型传入。")
        return None
    try:
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=proceed_when_unidentifiable)
        print("  已识别的估计量:")
        print(f"  {identified_estimand}")
        return identified_estimand
    except Exception as e:
        print(f"  识别因果效应失败: {e}")
        return None


def estimate_causal_effect_api(
    model: CausalModel, 
    identified_estimand: dowhy. παιδιά.Estimand, 
    method_name: str = "backdoor.linear_regression", 
    test_significance: bool = True, 
    method_params: dict | None = None
) -> dowhy.causal_estimator.Estimate | None:
    """
    使用指定方法估计已识别的因果效应。

    参数:
    - model (CausalModel): DoWhy因果模型。
    - identified_estimand (dowhy. παιδιά.Estimand): 已识别的估计量。
    - method_name (str): 估计方法名 (例如 "backdoor.linear_regression", "iv.instrumental_variable")。
    - test_significance (bool): 是否进行统计显著性检验。
    - method_params (dict | None): 特定估计方法的参数。

    返回:
    - dowhy.causal_estimator.Estimate | None: 包含估计结果的对象，如果失败则为None。
    """
    print(f"\n--- 使用方法 '{method_name}' 估计因果效应 (API) ---")
    if not model or not identified_estimand:
        print("  错误: 无效的模型或估计量传入。")
        return None
    
    # 为某些方法设置默认参数 (如果用户未提供)
    params = method_params if method_params is not None else {}
    # 例如，对于倾向性得分匹配，可以指定匹配方法
    # if method_name == "backdoor.propensity_score_matching" and 'matching_method' not in params:
    #     params['matching_method'] = 'nearest_neighbor'

    try:
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method_name,
            test_significance=test_significance,
            method_params=params 
            # control_value 和 treatment_value 对二元处理有用，但通常DoWhy能自动处理
        )
        print("  因果效应估计结果:")
        print(f"  {estimate}")
        return estimate
    except Exception as e:
        print(f"  使用 '{method_name}' 进行因果效应估计时出错: {e}")
        print("  可能原因: 数据类型不匹配 (例如，连续处理变量用于需要二元处理的方法)，数据问题 (如共线性)，或方法本身的特定要求。")
        return None


def refute_causal_estimate_api(
    model: CausalModel, 
    identified_estimand: dowhy. παιδιά.Estimand, 
    estimate: dowhy.causal_estimator.Estimate, 
    method_name: str = "random_common_cause", 
    **kwargs # 其他反驳器特定参数
) -> dowhy.causal_refuter.RefuteResult | None:
    """
    对已估计的因果效应进行反驳检验。

    参数:
    - model (CausalModel): DoWhy因果模型。
    - identified_estimand (dowhy. παιδιά.Estimand): 已识别的估计量。
    - estimate (dowhy.causal_estimator.Estimate): 已估计的因果效应。
    - method_name (str): 反驳方法名 (例如 "random_common_cause", "placebo_treatment_refuter")。
    - **kwargs: 传递给特定反驳器的额外参数。

    返回:
    - dowhy.causal_refuter.RefuteResult | None: 反驳检验结果对象，如果失败则为None。
    """
    print(f"\n--- 使用方法 '{method_name}' 进行反驳检验 (API) ---")
    if not model or not identified_estimand or not estimate:
        print("  错误: 无效的模型、估计量或估计结果传入。")
        return None
    try:
        # 某些反驳器可能需要特定的参数，通过kwargs传入
        # 例如: placebo_treatment_refuter 可能需要 placebo_type
        #       dummy_outcome_refuter 可能需要 effect_strength_on_outcome
        refutation = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name=method_name,
            **kwargs 
        )
        print("  反驳检验结果:")
        print(f"  {refutation}")
        return refutation
    except Exception as e:
        print(f"  使用 '{method_name}' 进行反驳检验时出错: {e}")
        return None


if __name__ == '__main__':
    print("========== (C 部分) DoWhy 因果推断演示 ==========")
    output_directory_dowhy = "dowhy_outputs_main_demo"
    if not os.path.exists(output_directory_dowhy):
        os.makedirs(output_directory_dowhy)
        print(f"创建主输出目录: {output_directory_dowhy}")

    # --- 示例1: 后门调整 (连续结果, 连续处理) ---
    print("\n\n*** DoWhy 示例 1: 后门调整 (连续结果, 连续处理) ***")
    df_ex1, w_ex1, z_ex1, t_ex1, y_ex1 = generate_dowhy_data_api(seed=11, treatment_type='continuous', outcome_type='continuous')
    
    # 定义因果图: W是T和Y的共同原因; Z是T的工具变量 (此处后门调整不使用Z)
    # 后门路径: T <- W -> Y. 需要调整W.
    graph_ex1_dot = f"digraph {{ {w_ex1}; {z_ex1}; {t_ex1}; {y_ex1}; {w_ex1} -> {t_ex1}; {w_ex1} -> {y_ex1}; {z_ex1} -> {t_ex1}; {t_ex1} -> {y_ex1}; }}"
    # 或者，不使用graph_dot_string，而是指定 common_causes_names=['W']
    
    model_ex1 = define_causal_model_api(df_ex1, treatment_name=t_ex1, outcome_name=y_ex1, graph_dot_string=graph_ex1_dot)
    # model_ex1 = define_causal_model_api(df_ex1, treatment_name=t_ex1, outcome_name=y_ex1, common_causes_names=[w_ex1])

    if model_ex1:
        estimand_ex1 = identify_causal_effect_api(model_ex1)
        if estimand_ex1:
            estimate_ex1 = estimate_causal_effect_api(model_ex1, estimand_ex1, method_name="backdoor.linear_regression")
            if estimate_ex1:
                print(f"  估计的ATE ({t_ex1} -> {y_ex1}): {estimate_ex1.value:.4f} (真实值约为2.0)")
                refute_causal_estimate_api(model_ex1, estimand_ex1, estimate_ex1, method_name="random_common_cause")
                refute_causal_estimate_api(model_ex1, estimand_ex1, estimate_ex1, method_name="placebo_treatment_refuter", placebo_type="permute") # permute是较新的placebo类型

    # --- 示例2: 工具变量法 (连续结果, 连续处理) ---
    print("\n\n*** DoWhy 示例 2: 工具变量法 (连续结果, 连续处理) ***")
    # 使用与示例1相同的数据，但这次利用Z作为工具变量
    # 假设W是未观测到的混杂 (或者我们选择不调整它，而使用IV)
    # 因果图: Z -> T -> Y; W -> T; W -> Y (W是混杂)
    # 如果我们只提供Z作为工具变量，DoWhy会尝试找到合适的路径
    graph_ex2_dot = f"digraph {{ {w_ex1}; {z_ex1}; {t_ex1}; {y_ex1}; {w_ex1} -> {t_ex1}; {w_ex1} -> {y_ex1}; {z_ex1} -> {t_ex1}; {t_ex1} -> {y_ex1}; }}"

    model_ex2 = define_causal_model_api(df_ex1, treatment_name=t_ex1, outcome_name=y_ex1, 
                                      instrument_names=[z_ex1], # 关键: 指定Z为工具变量
                                      graph_dot_string=graph_ex2_dot # 提供完整图，让DoWhy从中识别IV路径
                                     )
    # 如果不提供图，仅提供IV，DoWhy需要推断W的角色，可能更复杂
    # model_ex2 = define_causal_model_api(df_ex1, treatment_name=t_ex1, outcome_name=y_ex1, instrument_names=[z_ex1])

    if model_ex2:
        estimand_ex2 = identify_causal_effect_api(model_ex2)
        if estimand_ex2:
            # 常见IV估计方法: "iv.instrumental_variable" (通常是2SLS)
            estimate_ex2 = estimate_causal_effect_api(model_ex2, estimand_ex2, method_name="iv.instrumental_variable")
            if estimate_ex2:
                print(f"  估计的ATE ({t_ex1} -> {y_ex1} using IV {z_ex1}): {estimate_ex2.value:.4f} (真实值约为2.0)")
                # IV的常见反驳方法：dummy_outcome_refuter, data_subset_refuter
                refute_causal_estimate_api(model_ex2, estimand_ex2, estimate_ex2, method_name="dummy_outcome_refuter", effect_strength_on_outcome=0.01) 
                refute_causal_estimate_api(model_ex2, estimand_ex2, estimate_ex2, method_name="data_subset_refuter", subset_fraction=0.8)

    # --- 示例3: 后门调整 (二元结果, 二元处理, 使用倾向性得分) ---
    print("\n\n*** DoWhy 示例 3: 后门调整 (二元结果, 二元处理) 与倾向性得分 ***")
    df_ex3, w_ex3, z_ex3, t_ex3, y_ex3 = generate_dowhy_data_api(seed=33, treatment_type='binary', outcome_type='binary')
    graph_ex3_dot = f"digraph {{ {w_ex3}; {t_ex3}; {y_ex3}; {w_ex3} -> {t_ex3}; {w_ex3} -> {y_ex3}; {t_ex3} -> {y_ex3}; }}"
    
    model_ex3 = define_causal_model_api(df_ex3, treatment_name=t_ex3, outcome_name=y_ex3, graph_dot_string=graph_ex3_dot)
    # model_ex3 = define_causal_model_api(df_ex3, treatment_name=t_ex3, outcome_name=y_ex3, common_causes_names=[w_ex3])

    if model_ex3:
        estimand_ex3 = identify_causal_effect_api(model_ex3)
        if estimand_ex3:
            # 对于二元处理，倾向性得分方法很常用
            estimate_ex3_psm = estimate_causal_effect_api(model_ex3, estimand_ex3, 
                                                            method_name="backdoor.propensity_score_matching")
            if estimate_ex3_psm:
                print(f"  估计的ATE (PSM, {t_ex3} -> {y_ex3}): {estimate_ex3_psm.value:.4f}")
                refute_causal_estimate_api(model_ex3, estimand_ex3, estimate_ex3_psm, "data_subset_refuter", subset_fraction=0.8)

            estimate_ex3_psw = estimate_causal_effect_api(model_ex3, estimand_ex3, 
                                                            method_name="backdoor.propensity_score_weighting")
            if estimate_ex3_psw:
                 print(f"  估计的ATE (PSW, {t_ex3} -> {y_ex3}): {estimate_ex3_psw.value:.4f}")
                 # 可以用不同的反驳器

    print("\n\n========== (C 部分) DoWhy 因果推断演示结束 ==========")
    print(f"注意: DoWhy因果图 (如果生成) 可能保存在当前工作目录或 '{output_directory_dowhy}' (如果view_model被配置为保存)。") 