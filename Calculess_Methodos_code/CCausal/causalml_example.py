import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CausalML imports
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier # BaseRClassifier not used in original, can be added if needed
from causalml.metrics import plot_gain, plot_uplift_curve, plot_qini_curve, auuc_score, qini_score

# Scikit-learn for base models and utilities
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# --- API Functions ---

def generate_uplift_data_api(
    n_samples: int = 5000, 
    n_features: int = 3, # X1, X2, X3 as in original
    random_seed: int = 4242,
    treatment_group_ratio: float = 0.5
) -> tuple[pd.DataFrame, list[str]]:
    """
    生成用于uplift建模的合成数据集。

    数据包含特征 (X), 干预 (T - treatment), 和结果 (Y - outcome)。
    模拟了一个场景，其中干预对不同子群体产生不同的效应 (uplift)。

    参数:
    - n_samples (int): 生成的样本数量。
    - n_features (int): 特征数量 (固定为3以匹配原始逻辑: X1, X2, X3)。
    - random_seed (int): 随机种子，用于可复现性。
    - treatment_group_ratio (float): 分配到干预组的样本比例。

    返回:
    - pd.DataFrame: 包含特征、干预、结果以及真实uplift的数据集。
    - list[str]: 特征列的名称列表。
    """
    if n_features != 3:
        print(f"警告: 此API设计为生成3个特征(X1,X2,X3)。请求的 n_features={n_features} 将被忽略。")

    np.random.seed(random_seed)
    
    # 生成特征
    X1 = np.random.normal(0, 1, n_samples)  # 数值型特征
    X2 = np.random.choice([0, 1], n_samples, p=[0.6, 0.4]) # 二元类别型特征
    X3 = np.random.uniform(0, 5, n_samples) # 数值型特征
    feature_cols = ['X1', 'X2', 'X3']

    # 干预分配 (为简单起见，采用随机分配)
    # 现实中，干预可能不是随机的，这需要更复杂的CATE估计方法。
    T_binary = np.random.choice([0, 1], n_samples, p=[1-treatment_group_ratio, treatment_group_ratio])

    # 定义真实的uplift (CATE - Conditional Average Treatment Effect)
    # 这是我们希望模型估计的目标
    # 组1 (高uplift): X1 > 0 且 X2 == 1
    # 组2 (中等uplift): X1 <= 0 且 X2 == 1
    # 组3 (低/负uplift): X2 == 0
    true_uplift = np.zeros(n_samples)
    group1_mask = (X1 > 0) & (X2 == 1)
    group2_mask = (X1 <= 0) & (X2 == 1)
    group3_mask = (X2 == 0)

    true_uplift[group1_mask] = 0.20  # 高正向uplift
    true_uplift[group2_mask] = 0.08  # 中等正向uplift
    true_uplift[group3_mask] = -0.03 # 低/负向uplift

    # 基础转化率 (未接受干预时的结果概率)
    base_conversion_prob = 0.10 + 0.02 * X1 - 0.01 * X3 + 0.05 * X2
    base_conversion_prob = np.clip(base_conversion_prob, 0.01, 0.99) # 确保概率有效

    # 最终结果概率 (Y_prob)
    # Y_prob = 基础转化率 (如果 T=0) 或 基础转化率 + uplift (如果 T=1)
    Y_prob = base_conversion_prob + T_binary * true_uplift
    Y_prob = np.clip(Y_prob, 0.01, 0.99) # 再次确保概率有效
    
    # 生成二元结果 (例如，是否转化)
    Y_binary = (np.random.uniform(size=n_samples) < Y_prob).astype(int)

    df = pd.DataFrame({
        feature_cols[0]: X1,
        feature_cols[1]: X2,
        feature_cols[2]: X3,
        'treatment': T_binary, # CausalML通常要求干预列名为 'treatment'
        'outcome': Y_binary,   # 结果列名，或根据评估函数要求调整
        'true_uplift': true_uplift # 用于评估，不用于模型训练
    })
    # print(f"API: 已生成uplift数据集，形状: {df.shape}")
    return df, feature_cols


def train_evaluate_uplift_tree_api(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    feature_cols: list[str],
    treatment_col: str = 'treatment', 
    outcome_col: str = 'outcome',
    control_group_identifier = 0,
    max_depth: int = 4, 
    min_samples_leaf: int = 100, 
    min_samples_treatment: int = 50,
    random_state: int = 123 
) -> tuple[UpliftTreeClassifier | None, pd.DataFrame]:
    """
    训练UpliftTreeClassifier并对测试集进行预测。

    参数:
    - df_train (pd.DataFrame): 训练数据集。
    - df_test (pd.DataFrame): 测试数据集。
    - feature_cols (list[str]): 特征列名列表。
    - treatment_col (str): 干预列名。
    - outcome_col (str): 结果列名。
    - control_group_identifier: 干预列中代表控制组的值。
    - max_depth, min_samples_leaf, min_samples_treatment, random_state: UpliftTreeClassifier的参数。

    返回:
    - tuple[UpliftTreeClassifier | None, pd.DataFrame]: 训练好的模型（如果成功），带有uplift预测的测试集副本。
    """
    print("\n--- 训练和评估 Uplift Tree (API) ---")
    X_train, y_train, T_train = df_train[feature_cols], df_train[outcome_col], df_train[treatment_col]
    X_test = df_test[feature_cols]
    df_test_pred = df_test.copy()

    model = UpliftTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
        min_samples_treatment=min_samples_treatment, control_name=control_group_identifier, 
        random_state=random_state
    )
    try:
        model.fit(X_train.values, treatment=T_train.values, y=y_train.values) # CausalML树模型通常需要numpy数组
        print("  UpliftTreeClassifier 拟合完成。")
        uplift_preds = model.predict(X_test.values)
        if isinstance(uplift_preds, list): uplift_preds = uplift_preds[0] # 某些版本返回列表
        df_test_pred['uplift_tree'] = uplift_preds.flatten() if uplift_preds.ndim > 1 else uplift_preds
        print(f"  Uplift Tree 平均预测uplift: {df_test_pred['uplift_tree'].mean():.4f}")
        return model, df_test_pred
    except Exception as e:
        print(f"  训练/预测 UpliftTreeClassifier 时出错: {e}")
        return None, df_test_pred


def train_evaluate_uplift_rf_api(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    feature_cols: list[str],
    treatment_col: str = 'treatment', 
    outcome_col: str = 'outcome',
    control_group_identifier = 0,
    n_estimators: int = 50, 
    max_depth: int = 5, 
    min_samples_leaf: int = 150, 
    min_samples_treatment: int = 70,
    random_state: int = 123, 
    n_jobs: int = -1
) -> tuple[UpliftRandomForestClassifier | None, pd.DataFrame]:
    """
    训练UpliftRandomForestClassifier并对测试集进行预测。
    参数与Uplift Tree类似, 增加了n_estimators, n_jobs。

    返回:
    - tuple[UpliftRandomForestClassifier | None, pd.DataFrame]: 训练好的模型，带有uplift预测的测试集副本。
    """
    print("\n--- 训练和评估 Uplift Random Forest (API) ---")
    X_train, y_train, T_train = df_train[feature_cols], df_train[outcome_col], df_train[treatment_col]
    X_test = df_test[feature_cols]
    df_test_pred = df_test.copy()

    model = UpliftRandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf, min_samples_treatment=min_samples_treatment,
        control_name=control_group_identifier, random_state=random_state, n_jobs=n_jobs
    )
    try:
        model.fit(X_train.values, treatment=T_train.values, y=y_train.values)
        print("  UpliftRandomForestClassifier 拟合完成。")
        uplift_preds = model.predict(X_test.values)
        if isinstance(uplift_preds, list): uplift_preds = uplift_preds[0]
        df_test_pred['uplift_rf'] = uplift_preds.flatten() if uplift_preds.ndim > 1 else uplift_preds
        print(f"  Uplift RF 平均预测uplift: {df_test_pred['uplift_rf'].mean():.4f}")
        return model, df_test_pred
    except Exception as e:
        print(f"  训练/预测 UpliftRandomForestClassifier 时出错: {e}")
        return None, df_test_pred


def train_evaluate_slearner_api(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    feature_cols: list[str],
    treatment_col: str = 'treatment', 
    outcome_col: str = 'outcome',
    base_learner_class = RandomForestClassifier, # Sklearn兼容的分类器
    base_learner_params: dict | None = None
) -> tuple[BaseSClassifier | None, pd.DataFrame]:
    """
    训练CausalML的S-Learner (BaseSClassifier) 并预测uplift。

    参数:
    - base_learner_class: 用于S-Learner的Scikit-learn基础学习器类。
    - base_learner_params: 基础学习器的参数字典。

    返回:
    - tuple[BaseSClassifier | None, pd.DataFrame]: 训练好的S-Learner模型，带有uplift预测的测试集副本。
    """
    print("\n--- 训练和评估 S-Learner (CausalML API) ---")
    X_train, y_train, T_train = df_train[feature_cols], df_train[outcome_col], df_train[treatment_col]
    X_test = df_test[feature_cols]
    df_test_pred = df_test.copy()

    learner_params = base_learner_params if base_learner_params is not None else {'random_state': 123}
    if 'n_estimators' not in learner_params and base_learner_class == RandomForestClassifier: learner_params['n_estimators'] = 50 # Default for demo
    if 'max_depth' not in learner_params and base_learner_class == RandomForestClassifier: learner_params['max_depth'] = 5 # Default for demo

    model = BaseSClassifier(learner=base_learner_class(**learner_params))
    try:
        model.fit(X=X_train, treatment=T_train, y=y_train)
        print("  S-Learner 拟合完成。")
        uplift_preds = model.predict(X_test)
        df_test_pred['uplift_slearner'] = uplift_preds
        print(f"  S-Learner 平均预测uplift: {df_test_pred['uplift_slearner'].mean():.4f}")
        return model, df_test_pred
    except Exception as e:
        print(f"  训练/预测 S-Learner 时出错: {e}")
        return None, df_test_pred


def train_evaluate_tlearner_api(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    feature_cols: list[str],
    treatment_col: str = 'treatment', 
    outcome_col: str = 'outcome',
    base_learner_class = RandomForestClassifier,
    base_learner_params: dict | None = None
) -> tuple[BaseTClassifier | None, pd.DataFrame]:
    """
    训练CausalML的T-Learner (BaseTClassifier) 并预测uplift。
    参数与S-Learner类似。
    返回: 
    - tuple[BaseTClassifier | None, pd.DataFrame]: 训练好的T-Learner模型，带有uplift预测的测试集副本。
    """
    print("\n--- 训练和评估 T-Learner (CausalML API) ---")
    X_train, y_train, T_train = df_train[feature_cols], df_train[outcome_col], df_train[treatment_col]
    X_test = df_test[feature_cols]
    df_test_pred = df_test.copy()

    learner_params = base_learner_params if base_learner_params is not None else {'random_state': 123}
    if 'n_estimators' not in learner_params and base_learner_class == RandomForestClassifier: learner_params['n_estimators'] = 50
    if 'max_depth' not in learner_params and base_learner_class == RandomForestClassifier: learner_params['max_depth'] = 5

    model = BaseTClassifier(learner=base_learner_class(**learner_params))
    try:
        model.fit(X=X_train, treatment=T_train, y=y_train)
        print("  T-Learner 拟合完成。")
        uplift_preds = model.predict(X_test)
        df_test_pred['uplift_tlearner'] = uplift_preds
        print(f"  T-Learner 平均预测uplift: {df_test_pred['uplift_tlearner'].mean():.4f}")
        return model, df_test_pred
    except Exception as e:
        print(f"  训练/预测 T-Learner 时出错: {e}")
        return None, df_test_pred

def train_evaluate_xlearner_cml_api(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    feature_cols: list[str],
    treatment_col: str = 'treatment', 
    outcome_col: str = 'outcome',
    outcome_learner_class = RandomForestRegressor,
    outcome_learner_params: dict | None = None,
    effect_learner_class = RandomForestRegressor,
    effect_learner_params: dict | None = None,
    propensity_learner_class = LogisticRegression,
    propensity_learner_params: dict | None = None
) -> tuple[BaseXClassifier | None, pd.DataFrame]:
    """
    训练CausalML的X-Learner (BaseXClassifier) 并预测uplift。

    参数:
    - outcome_learner_class/params: 用于结果建模的学习器。
    - effect_learner_class/params: 用于效应建模的学习器。
    - propensity_learner_class/params: 用于倾向性得分建模的学习器。

    返回:
    - tuple[BaseXClassifier | None, pd.DataFrame]: 训练好的X-Learner模型，带有uplift预测的测试集副本。
    """
    print("\n--- 训练和评估 X-Learner (CausalML API) ---")
    X_train, y_train, T_train = df_train[feature_cols], df_train[outcome_col], df_train[treatment_col]
    X_test = df_test[feature_cols]
    df_test_pred = df_test.copy()

    # 设置默认参数 (如果未提供)
    o_params = outcome_learner_params if outcome_learner_params is not None else {'n_estimators':30, 'max_depth':4, 'random_state':123}
    e_params = effect_learner_params if effect_learner_params is not None else {'n_estimators':30, 'max_depth':4, 'random_state':123}
    p_params = propensity_learner_params if propensity_learner_params is not None else {'random_state':123, 'solver':'liblinear'}

    model = BaseXClassifier(
        outcome_learner=outcome_learner_class(**o_params),
        effect_learner=effect_learner_class(**e_params),
        propensity_learner=propensity_learner_class(**p_params)
    )
    try:
        model.fit(X=X_train, treatment=T_train, y=y_train)
        print("  X-Learner (CausalML) 拟合完成。")
        uplift_preds = model.predict(X_test)
        df_test_pred['uplift_xlearner_cml'] = uplift_preds
        print(f"  X-Learner (CausalML) 平均预测uplift: {df_test_pred['uplift_xlearner_cml'].mean():.4f}")
        return model, df_test_pred
    except Exception as e:
        print(f"  训练/预测 X-Learner (CausalML) 时出错: {e}")
        return None, df_test_pred

def evaluate_uplift_predictions_api(
    df_eval_with_preds: pd.DataFrame, 
    treatment_col: str = 'treatment', 
    outcome_col: str = 'outcome',
    true_uplift_col: str | None = 'true_uplift', # 可选，用于计算MSE
    output_dir: str = "causalml_outputs",
    plot_metrics_for_col: str | None = None # 指定为哪个预测列绘图，例如 'uplift_rf'
) -> dict:
    """
    评估DataFrame中的uplift预测，计算AUUC, Qini分数，并可选地绘制评估曲线。

    参数:
    - df_eval_with_preds (pd.DataFrame): 包含特征、干预、结果以及一个或多个uplift预测列的DataFrame。
    - treatment_col (str): 干预列名。
    - outcome_col (str): 结果列名。
    - true_uplift_col (str | None): 真实uplift列名 (如果存在，用于计算MSE)。
    - output_dir (str): 保存图表的目录。
    - plot_metrics_for_col (str | None): 如果提供，则为该预测列生成Gain/Uplift/Qini曲线图。

    返回:
    - dict: 包含每个预测列的AUUC, Qini分数和MSE (如果计算) 的字典。
    """
    print("\n--- 评估 Uplift 模型预测 (API) ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    predicted_uplift_cols = [col for col in df_eval_with_preds.columns if col.startswith('uplift_')]
    if not predicted_uplift_cols:
        print("  在DataFrame中未找到uplift预测列进行评估。")
        return {}

    evaluation_results = {}
    for pred_col_name in predicted_uplift_cols:
        print(f"\n  评估预测列: {pred_col_name}")
        current_results = {}
        try:
            auuc = auuc_score(df_eval_with_preds, prediction_col=pred_col_name, 
                              outcome_col=outcome_col, treatment_col=treatment_col)
            qini = qini_score(df_eval_with_preds, prediction_col=pred_col_name, 
                              outcome_col=outcome_col, treatment_col=treatment_col)
            current_results['AUUC'] = auuc
            current_results['Qini'] = qini
            print(f"    AUUC 分数: {auuc:.4f}")
            print(f"    Qini 分数: {qini:.4f}")

            if true_uplift_col and true_uplift_col in df_eval_with_preds.columns:
                mse = np.mean((df_eval_with_preds[true_uplift_col] - df_eval_with_preds[pred_col_name])**2)
                current_results['MSE_vs_True'] = mse
                print(f"    相对于真实uplift的MSE: {mse:.4f}")
            evaluation_results[pred_col_name] = current_results
        except Exception as e_eval:
            print(f"    评估 {pred_col_name} 时出错: {e_eval}")
            evaluation_results[pred_col_name] = {'error': str(e_eval)}

    # 绘图逻辑
    if plot_metrics_for_col and plot_metrics_for_col in predicted_uplift_cols:
        print(f"\n  为 {plot_metrics_for_col} 生成评估图表...")
        try:
            fig, axes = plt.subplots(1, 3, figsize=(21, 6))
            plot_gain(df_eval_with_preds, outcome_col=outcome_col, treatment_col=treatment_col, 
                      prediction_col=plot_metrics_for_col, ax=axes[0], title=f'Gain Curve ({plot_metrics_for_col})')
            plot_uplift_curve(df_eval_with_preds, outcome_col=outcome_col, treatment_col=treatment_col, 
                              prediction_col=plot_metrics_for_col, ax=axes[1], title=f'Uplift Curve ({plot_metrics_for_col})')
            plot_qini_curve(df_eval_with_preds, outcome_col=outcome_col, treatment_col=treatment_col, 
                              prediction_col=plot_metrics_for_col, ax=axes[2], title=f'Qini Curve ({plot_metrics_for_col})')
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f"causalml_eval_plots_{plot_metrics_for_col}.png")
            plt.savefig(plot_filename)
            print(f"    评估图表已保存至: {plot_filename}")
            plt.close()
        except Exception as e_plot:
            print(f"    为 {plot_metrics_for_col} 生成图表时出错: {e_plot}")
    elif plot_metrics_for_col:
        print(f"  警告: 请求为列 '{plot_metrics_for_col}' 绘图，但该列未在预测中找到或评估失败。")
    
    return evaluation_results


if __name__ == '__main__':
    print("========== (C 部分) CausalML Uplift建模演示 ==========")
    # --- (CML.0) 全局设置与数据生成 ---
    main_seed_cml = 123
    num_samples_cml = 8000 # Uplift模型通常需要较多数据
    output_directory_cml = "causalml_outputs"

    if not os.path.exists(output_directory_cml):
        os.makedirs(output_directory_cml)
        print(f"创建主输出目录: {output_directory_cml}")

    df_uplift_cml, feature_cols_cml = generate_uplift_data_api(
        n_samples=num_samples_cml, random_seed=main_seed_cml
    )
    print(f"CausalML演示数据已生成: {df_uplift_cml.shape}")

    # --- (CML.1) 数据分割 ---
    # 如果类别非常不平衡，可以考虑按干预和结果进行分层抽样
    # stratify_cols = ['treatment', 'outcome'] if len(df_uplift_cml[['treatment','outcome']].drop_duplicates()) > 1 else None
    df_train_cml, df_test_cml = train_test_split(
        df_uplift_cml, test_size=0.4, random_state=main_seed_cml + 1 # 较大的测试集以获得更稳定的评估
        # stratify=df_uplift_cml[stratify_cols] if stratify_cols else None
    )
    print(f"训练集大小: {df_train_cml.shape}, 测试集大小: {df_test_cml.shape}")

    # --- (CML.2) 训练模型并收集预测 --- 
    # 创建一个测试集副本，用于逐步添加不同模型的预测列
    df_test_preds_collected = df_test_cml.copy()

    # 2.1 Uplift Tree
    _, df_test_preds_collected = train_evaluate_uplift_tree_api(
        df_train_cml, df_test_preds_collected, feature_cols_cml, random_state=main_seed_cml
    )
    # 2.2 Uplift Random Forest
    _, df_test_preds_collected = train_evaluate_uplift_rf_api(
        df_train_cml, df_test_preds_collected, feature_cols_cml, random_state=main_seed_cml, n_estimators=30 # 减少n_estimators以加速演示
    )
    # 2.3 S-Learner
    _, df_test_preds_collected = train_evaluate_slearner_api(
        df_train_cml, df_test_preds_collected, feature_cols_cml, 
        base_learner_params={'n_estimators':30, 'max_depth':4, 'random_state':main_seed_cml} # 调整参数
    )
    # 2.4 T-Learner
    _, df_test_preds_collected = train_evaluate_tlearner_api(
        df_train_cml, df_test_preds_collected, feature_cols_cml,
        base_learner_params={'n_estimators':30, 'max_depth':4, 'random_state':main_seed_cml}
    )
    # 2.5 X-Learner (CausalML)
    _, df_test_preds_collected = train_evaluate_xlearner_cml_api(
        df_train_cml, df_test_preds_collected, feature_cols_cml,
        outcome_learner_params={'n_estimators':20, 'max_depth':3, 'random_state':main_seed_cml},
        effect_learner_params={'n_estimators':20, 'max_depth':3, 'random_state':main_seed_cml},
        propensity_learner_params={'random_state':main_seed_cml, 'solver':'liblinear'}
    )

    # --- (CML.3) 评估所有模型的预测 --- 
    # 选择一个表现较好或有代表性的模型进行绘图，例如 Uplift RF
    # 如果uplift_rf列存在，则为其绘图
    col_for_plotting = 'uplift_rf' if 'uplift_rf' in df_test_preds_collected.columns else None
    # 如果uplift_rf不存在，尝试选择其他存在的预测列
    if not col_for_plotting:
        available_pred_cols = [c for c in df_test_preds_collected.columns if c.startswith('uplift_')]
        if available_pred_cols: col_for_plotting = available_pred_cols[0]
            
    all_eval_metrics = evaluate_uplift_predictions_api(
        df_test_preds_collected, 
        output_dir=output_directory_cml,
        plot_metrics_for_col=col_for_plotting 
    )

    print("\n所有模型的评估指标汇总:")
    for model_pred_col, metrics in all_eval_metrics.items():
        print(f"  模型 ({model_pred_col}): {metrics}")
    
    print("\n\n========== (C 部分) CausalML Uplift建模演示结束 ==========")
    print(f"注意: Uplift模型评估图表已保存在 '{output_directory_cml}' 目录中 (如果成功生成)。")
