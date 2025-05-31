import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# EconML imports
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML
from econml.metalearners import XLearner, TLearner, SLearner # T, S, X learners
from econml.dr import DRLearner # Doubly Robust Learner
# from econml.iv.dml import DMLIV # For IV with DML - can be added if specific examples are needed
# from econml.cate_interpreter import CateInterpreter # For interpreting CATE models

# Scikit-learn for models and utilities
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler # StandardScaler can be useful
from sklearn.pipeline import Pipeline

# --- API Functions ---

def generate_heterogeneous_data_api(
    n_samples: int = 2000, 
    n_features: int = 5, 
    n_instruments: int = 1, 
    seed: int = 123,
    effect_type: str = 'linear' # 'linear' or 'nonlinear' for CATE
) -> tuple[pd.DataFrame, list[str], list[str], list[str] | None, str, str]:
    """
    生成用于异质因果效应 (CATE) 建模的合成数据集。

    数据包含:
    - X: 特征，其中一些可能影响处理效应的大小 (效应修正因子)。
    - W: 混杂因子，影响处理分配和结果。
    - Z: 工具变量，影响处理分配，但不直接影响结果 (除非通过处理)。
    - T: 处理变量 (二元)。
    - Y: 结果变量 (连续)。
    - true_CATE: 真实的条件平均处理效应 (用于评估)。

    参数:
    - n_samples (int): 样本数量。
    - n_features (int): 特征 (X) 的数量。
    - n_instruments (int): 工具变量 (Z) 的数量。
    - seed (int): 随机种子。
    - effect_type (str): 'linear' 表示线性CATE, 'nonlinear' 表示非线性CATE。

    返回:
    - pd.DataFrame: 生成的数据集。
    - list[str]: 效应修正因子列名 (X_cols)。
    - list[str]: 混杂因子列名 (W_cols)。
    - list[str] | None: 工具变量列名 (Z_cols)，如果 n_instruments > 0。
    - str: 处理列名 (T_col)。
    - str: 结果列名 (Y_col)。
    """
    np.random.seed(seed)
    
    # 特征 X (效应修正因子, 部分也可能作为混杂因子)
    X_data = pd.DataFrame(np.random.multivariate_normal(np.zeros(n_features), np.eye(n_features), size=n_samples),
                          columns=[f'X{i}' for i in range(n_features)])
    X_cols = list(X_data.columns)

    # 混杂因子 W (这里简单地让X的前两个特征作为W的一部分)
    # 现实中W的定义会更复杂
    confounder_features_count = min(n_features, 2) 
    W_data = X_data[[f'X{i}' for i in range(confounder_features_count)]].copy().add_suffix('_conf')
    W_cols = list(W_data.columns)

    # 工具变量 Z
    if n_instruments > 0:
        Z_data = pd.DataFrame(np.random.multivariate_normal(np.zeros(n_instruments), np.eye(n_instruments), size=n_samples),
                              columns=[f'Z{i}' for i in range(n_instruments)])
        Z_cols = list(Z_data.columns)
    else:
        Z_data = None
        Z_cols = None

    # 真实的 CATE
    if effect_type == 'linear':
        true_cate = 1.0 + 0.8 * X_data['X0'] - 0.4 * X_data['X1']
    elif effect_type == 'nonlinear':
        true_cate = 1.0 + np.cos(X_data['X0'] * np.pi) + 0.5 * np.sin(X_data['X1'] * np.pi)
    else: # 默认为线性
        true_cate = 1.0 + 0.5 * X_data['X0']

    # 处理分配 T (二元)
    # T 的分配受 X (部分作为W) 和 Z 的影响
    propensity_logit = 0.5 * X_data['X0'] - 0.3 * X_data['X1'] # 基础倾向性
    if Z_data is not None and 'Z0' in Z_data.columns:
        propensity_logit += 0.6 * Z_data['Z0'] # 工具变量的影响
    
    propensity = 1 / (1 + np.exp(-np.clip(propensity_logit, -10, 10))) # Sigmoid, clip to avoid overflow
    T_binary = (np.random.uniform(size=n_samples) < propensity).astype(int)
    T_col = 'Treatment'

    # 结果 Y (连续)
    # Y = T * CATE(X) + f(X) + g(W) + noise
    Y_continuous = (T_binary * true_cate + 
                    2.0 * X_data['X0'] + np.sin(np.pi * X_data['X1']) + # X对Y的直接影响
                    0.2 * X_data[[f'X{i}' for i in range(confounder_features_count)]].sum(axis=1) + # W对Y的直接影响
                    np.random.normal(0, 0.5, n_samples)) # 噪声
    Y_col = 'Outcome'

    # 合并数据
    df_list = [X_data, W_data]
    if Z_data is not None:
        df_list.append(Z_data)
    df_list.extend([pd.Series(T_binary, name=T_col), 
                    pd.Series(Y_continuous, name=Y_col),
                    pd.Series(true_cate, name='true_CATE')])
    
    df = pd.concat(df_list, axis=1)
    df = df.loc[:,~df.columns.duplicated()] # 清理重复列名 (例如X和W有重叠时)

    # print(f"API: 已生成异质效应数据集，形状: {df.shape}")
    # print(f"API: X_cols={X_cols}, W_cols={W_cols}, Z_cols={Z_cols}, T_col={T_col}, Y_col={Y_col}")
    return df, X_cols, W_cols, Z_cols, T_col, Y_col


def train_evaluate_lineardml_api(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame,
    Y_col: str, T_col: str, X_cols: list[str], W_cols: list[str] | None,
    model_y_class=GradientBoostingRegressor, model_y_params: dict | None = None,
    model_t_class=GradientBoostingClassifier, model_t_params: dict | None = None,
    featurizer=PolynomialFeatures(degree=1, include_bias=True), # CATE的特征转换器
    random_state: int = 123,
    mc_iters: int = 3 # 交叉拟合的折数
) -> tuple[LinearDML | None, pd.DataFrame]:
    """
    训练EconML的LinearDML模型并评估。LinearDML假设CATE对X是线性的。

    参数:
    - df_train, df_test: 训练集和测试集。
    - Y_col, T_col, X_cols, W_cols: 结果、处理、效应修正因子、混杂因子列名。
    - model_y_class, model_t_class: 结果模型和处理模型的Scikit-learn类。
    - model_y_params, model_t_params: 模型参数。
    - featurizer: 用于最终CATE模型的特征转换器 (例如多项式特征)。
    - random_state, mc_iters: LinearDML参数。

    返回:
    - tuple[LinearDML | None, pd.DataFrame]: 训练好的模型和带有CATE预测的测试集副本。
    """
    print("\n--- 训练和评估 LinearDML (EconML API) ---")
    df_test_pred = df_test.copy()

    m_y_params = model_y_params if model_y_params else {'n_estimators': 50, 'max_depth': 3, 'random_state': random_state}
    m_t_params = model_t_params if model_t_params else {'n_estimators': 50, 'max_depth': 3, 'random_state': random_state}

    estimator = LinearDML(
        model_y=model_y_class(**m_y_params),
        model_t=model_t_class(**m_t_params),
        featurizer=featurizer,
        random_state=random_state,
        mc_iters=mc_iters
    )
    try:
        # W_cols可能为None，如果为None，DML内部会用X_cols作为控制变量
        W_train_data = df_train[W_cols] if W_cols and all(col in df_train.columns for col in W_cols) else None
        estimator.fit(df_train[Y_col], df_train[T_col], X=df_train[X_cols], W=W_train_data)
        print("  LinearDML 拟合完成。")

        X_test_data = df_test[X_cols]
        cate_preds = estimator.effect(X_test_data)
        df_test_pred['cate_lineardml'] = cate_preds
        print(f"  LinearDML 平均预测CATE: {cate_preds.mean():.4f}")
        
        if 'true_CATE' in df_test_pred.columns:
            mse = np.mean((df_test_pred['true_CATE'] - df_test_pred['cate_lineardml'])**2)
            print(f"  LinearDML MSE vs True CATE: {mse:.4f}")
        
        # 系数 (如果最终模型是线性的)
        if hasattr(estimator, 'coef__final_'):
            final_feature_names = estimator.cate_feature_names(X_cols)
            coef_series = pd.Series(estimator.coef__final_.flatten(), index=final_feature_names)
            print("  LinearDML CATE模型系数:\n", coef_series)
        return estimator, df_test_pred
    except Exception as e:
        print(f"  训练/预测 LinearDML 时出错: {e}")
        return None, df_test_pred


def train_evaluate_causalforestdml_api(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame,
    Y_col: str, T_col: str, X_cols: list[str], W_cols: list[str] | None,
    model_y_class=GradientBoostingRegressor, model_y_params: dict | None = None,
    model_t_class=GradientBoostingClassifier, model_t_params: dict | None = None,
    discrete_treatment: bool = True, # T是否为离散值
    cf_params: dict | None = None, # CausalForestDML特定参数
    random_state: int = 123,
    mc_iters: int = 3
) -> tuple[CausalForestDML | None, pd.DataFrame]:
    """
    训练EconML的CausalForestDML模型并评估。它可以捕获非线性的CATE。

    参数:
    - cf_params: CausalForestDML的参数 (如 n_estimators, min_samples_leaf)。

    返回:
    - tuple[CausalForestDML | None, pd.DataFrame]: 训练好的模型和带有CATE预测的测试集副本。
    """
    print("\n--- 训练和评估 CausalForestDML (EconML API) ---")
    df_test_pred = df_test.copy()

    m_y_params = model_y_params if model_y_params else {'n_estimators': 50, 'max_depth': 4, 'random_state': random_state}
    m_t_params = model_t_params if model_t_params else {'n_estimators': 50, 'max_depth': 4, 'random_state': random_state}
    
    default_cf_params = {'n_estimators': 100, 'min_samples_leaf': 10, 'max_depth': 5, 'random_state': random_state}
    current_cf_params = {**default_cf_params, **(cf_params if cf_params else {})}

    estimator = CausalForestDML(
        model_y=model_y_class(**m_y_params),
        model_t=model_t_class(**m_t_params),
        discrete_treatment=discrete_treatment,
        **current_cf_params, # 解包Causal Forest的参数
        mc_iters=mc_iters
    )
    try:
        W_train_data = df_train[W_cols] if W_cols and all(col in df_train.columns for col in W_cols) else None
        estimator.fit(df_train[Y_col], df_train[T_col], X=df_train[X_cols], W=W_train_data)
        print("  CausalForestDML 拟合完成。")

        X_test_data = df_test[X_cols]
        cate_preds = estimator.effect(X_test_data)
        df_test_pred['cate_cfdml'] = cate_preds
        print(f"  CausalForestDML 平均预测CATE: {cate_preds.mean():.4f}")

        if 'true_CATE' in df_test_pred.columns:
            mse = np.mean((df_test_pred['true_CATE'] - df_test_pred['cate_cfdml'])**2)
            print(f"  CausalForestDML MSE vs True CATE: {mse:.4f}")
        
        # 特征重要性
        if hasattr(estimator, 'feature_importances_'):
             importance_series = pd.Series(estimator.feature_importances_, index=X_cols)
             print("  CausalForestDML CATE模型特征重要性:\n", importance_series.sort_values(ascending=False))
        return estimator, df_test_pred
    except Exception as e:
        print(f"  训练/预测 CausalForestDML 时出错: {e}")
        return None, df_test_pred


def train_evaluate_xlearner_econml_api(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame,
    Y_col: str, T_col: str, X_cols: list[str], # X_cols包含了所有用于CATE估计的特征
    model_regression_class=GradientBoostingRegressor, model_regression_params: dict | None = None,
    model_propensity_class=GradientBoostingClassifier, model_propensity_params: dict | None = None
) -> tuple[XLearner | None, pd.DataFrame]:
    """
    训练EconML的X-Learner模型并评估。

    参数:
    - model_regression_class/params: 用于结果回归阶段的模型。
    - model_propensity_class/params: 用于倾向性得分估计的模型。

    返回:
    - tuple[XLearner | None, pd.DataFrame]: 训练好的模型和带有CATE预测的测试集副本。
    """
    print("\n--- 训练和评估 X-Learner (EconML API) ---")
    df_test_pred = df_test.copy()

    m_reg_params = model_regression_params if model_regression_params else {'n_estimators': 30, 'max_depth': 3, 'random_state': 123}
    m_prop_params = model_propensity_params if model_propensity_params else {'n_estimators': 30, 'max_depth': 3, 'random_state': 123}

    estimator = XLearner(
        models=model_regression_class(**m_reg_params),
        propensity_model=model_propensity_class(**m_prop_params)
    )
    try:
        # X-Learner的fit方法中，X参数用于CATE估计，内部处理混杂
        estimator.fit(df_train[Y_col], df_train[T_col], X=df_train[X_cols])
        print("  X-Learner (EconML) 拟合完成。")

        X_test_data = df_test[X_cols]
        cate_preds = estimator.effect(X_test_data)
        df_test_pred['cate_xlearner_econml'] = cate_preds
        print(f"  X-Learner (EconML) 平均预测CATE: {cate_preds.mean():.4f}")

        if 'true_CATE' in df_test_pred.columns:
            mse = np.mean((df_test_pred['true_CATE'] - df_test_pred['cate_xlearner_econml'])**2)
            print(f"  X-Learner (EconML) MSE vs True CATE: {mse:.4f}")
        return estimator, df_test_pred
    except Exception as e:
        print(f"  训练/预测 X-Learner (EconML) 时出错: {e}")
        return None, df_test_pred


def plot_cate_comparison_api(
    df_with_cates: pd.DataFrame, 
    true_cate_col: str = 'true_CATE',
    predicted_cate_cols: list[str] | None = None, # 例如 ['cate_lineardml', 'cate_cfdml']
    output_dir: str = "econml_outputs",
    prefix: str = "econml_cate_comparison"
):
    """
    绘制真实CATE与一个或多个模型预测的CATE的比较图。

    参数:
    - df_with_cates (pd.DataFrame): 包含真实CATE和预测CATE列的DataFrame。
    - true_cate_col (str): 真实CATE列名。
    - predicted_cate_cols (list[str] | None): 要绘制的预测CATE列名列表。如果为None, 则自动选择所有以'cate_'开头的列。
    - output_dir (str): 保存图表的目录。
    - prefix (str): 图表文件名前缀。
    """
    print("\n--- 绘制CATE比较图 (API) ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建输出目录: {output_dir}")

    if true_cate_col not in df_with_cates.columns:
        print(f"  警告: 真实CATE列 '{true_cate_col}' 不在DataFrame中，无法绘图。")
        return

    if predicted_cate_cols is None:
        predicted_cate_cols = [col for col in df_with_cates.columns if col.startswith('cate_') and col != true_cate_col]
    
    if not predicted_cate_cols:
        print("  没有找到预测的CATE列进行绘图。")
        return

    num_plots = len(predicted_cate_cols)
    if num_plots == 0: return

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), squeeze=False)
    axes = axes.flatten() #确保axes是一维的

    for i, pred_col in enumerate(predicted_cate_cols):
        if pred_col not in df_with_cates.columns:
            print(f"  警告: 预测CATE列 '{pred_col}' 不在DataFrame中，跳过。")
            continue
        
        ax = axes[i]
        sns.scatterplot(x=df_with_cates[true_cate_col], y=df_with_cates[pred_col], ax=ax, alpha=0.5, label=f"Pred: {pred_col.split('_')[-1]}")
        min_val = min(df_with_cates[true_cate_col].min(), df_with_cates[pred_col].min())
        max_val = max(df_with_cates[true_cate_col].max(), df_with_cates[pred_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Match")
        ax.set_xlabel(f"真实 CATE ({true_cate_col})")
        ax.set_ylabel(f"预测 CATE ({pred_col})")
        ax.set_title(f"真实 vs. 预测 CATE: {pred_col}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"{prefix}_scatter.png")
    plt.savefig(plot_filename)
    print(f"  CATE比较散点图已保存至: {plot_filename}")
    plt.close()

    # 可选: 绘制CATE分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df_with_cates[true_cate_col], label='True CATE', ax=ax, fill=True, alpha=0.3)
    for pred_col in predicted_cate_cols:
        if pred_col in df_with_cates.columns:
            sns.kdeplot(df_with_cates[pred_col], label=f"Pred: {pred_col}", ax=ax, fill=True, alpha=0.3)
    ax.set_title("CATE 分布对比")
    ax.set_xlabel("CATE 值")
    ax.legend()
    ax.grid(True)
    dist_plot_filename = os.path.join(output_dir, f"{prefix}_distributions.png")
    plt.savefig(dist_plot_filename)
    print(f"  CATE分布图已保存至: {dist_plot_filename}")
    plt.close()


if __name__ == '__main__':
    print("========== (C 部分) EconML CATE 建模演示 ==========")
    
    # --- (EconML.0) 全局设置与数据生成 ---
    main_seed_econml = 42
    num_samples_econml = 2500 # 适当数量的样本
    num_features_econml = 5
    num_instruments_econml = 1 # 是否使用工具变量
    output_directory_econml = "econml_outputs_main_demo" # 主演示的输出目录

    if not os.path.exists(output_directory_econml):
        os.makedirs(output_directory_econml)
        print(f"创建主输出目录: {output_directory_econml}")

    print("\n--- (EconML.0) 生成异质效应数据 (用于演示) ---")
    # 生成包含非线性效应的数据，以测试CausalForestDML等模型的能力
    df_econml_data, x_cols, w_cols, z_cols, t_col, y_col = generate_heterogeneous_data_api(
        n_samples=num_samples_econml, 
        n_features=num_features_econml,
        n_instruments=num_instruments_econml,
        seed=main_seed_econml,
        effect_type='nonlinear' # 'linear' 或 'nonlinear'
    )
    print(f"EconML演示数据已生成: {df_econml_data.shape}")
    print(f"  X特征: {x_cols}, W特征: {w_cols}, Z特征: {z_cols}, 处理: {t_col}, 结果: {y_col}")

    # --- (EconML.1) 数据分割 ---
    print("\n--- (EconML.1) 数据分割 ---")
    df_train_econml, df_test_econml = train_test_split(df_econml_data, test_size=0.35, random_state=main_seed_econml + 1)
    print(f"训练集大小: {df_train_econml.shape}, 测试集大小: {df_test_econml.shape}")

    # --- (EconML.2) 模型训练与评估 ---
    df_test_preds_collected_econml = df_test_econml.copy() # 用于收集所有模型的预测

    # 2.1 LinearDML
    # LinearDML可能不适合非线性CATE数据，但作为对比
    _, df_test_preds_collected_econml = train_evaluate_lineardml_api(
        df_train_econml, df_test_preds_collected_econml,
        Y_col=y_col, T_col=t_col, X_cols=x_cols, W_cols=w_cols, # z_cols 不直接用于LinearDML的X,W参数
        model_y_params={'n_estimators': 30, 'max_depth': 3, 'random_state': main_seed_econml},
        model_t_params={'n_estimators': 30, 'max_depth': 3, 'random_state': main_seed_econml},
        random_state=main_seed_econml
    )

    # 2.2 CausalForestDML (更适合非线性CATE)
    _, df_test_preds_collected_econml = train_evaluate_causalforestdml_api(
        df_train_econml, df_test_preds_collected_econml,
        Y_col=y_col, T_col=t_col, X_cols=x_cols, W_cols=w_cols,
        model_y_params={'n_estimators': 40, 'max_depth': 4, 'random_state': main_seed_econml},
        model_t_params={'n_estimators': 40, 'max_depth': 4, 'random_state': main_seed_econml},
        cf_params={'n_estimators': 80, 'min_samples_leaf': 15, 'max_depth': 6, 'random_state': main_seed_econml},
        random_state=main_seed_econml
    )

    # 2.3 X-Learner (EconML)
    # X-Learner的X参数应包含所有用于CATE估计的特征 (这里是x_cols)
    # 混杂由其内部模型处理
    _, df_test_preds_collected_econml = train_evaluate_xlearner_econml_api(
        df_train_econml, df_test_preds_collected_econml,
        Y_col=y_col, T_col=t_col, X_cols=x_cols, 
        model_regression_params={'n_estimators': 30, 'max_depth': 3, 'random_state': main_seed_econml},
        model_propensity_params={'n_estimators': 30, 'max_depth': 3, 'random_state': main_seed_econml}
    )
    
    # --- (EconML.3) 绘制CATE比较图 ---
    print("\n--- (EconML.3) 绘制CATE图表 ---")
    # 从收集的预测中确定哪些CATE列可用于绘图
    econml_pred_cols = [col for col in df_test_preds_collected_econml.columns if col.startswith('cate_') and col != 'true_CATE']
    
    if 'true_CATE' in df_test_preds_collected_econml.columns and econml_pred_cols:
        plot_cate_comparison_api(
            df_test_preds_collected_econml,
            true_cate_col='true_CATE',
            predicted_cate_cols=econml_pred_cols,
            output_dir=output_directory_econml,
            prefix="main_demo_econml_cate_comparison"
        )
    else:
        print("  未能生成CATE比较图，因为缺少true_CATE或预测的CATE列。")

    print("========== (C 部分) EconML CATE 建模演示结束 ==========")
    print(f"注意: EconML模型评估图表已保存在 '{output_directory_econml}' 目录中 (如果成功生成)。")
    print("可以尝试修改 generate_heterogeneous_data_api 中的 effect_type 为 'linear' 并重新运行，观察LinearDML的表现。") 