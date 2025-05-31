import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib # For saving/loading models if needed by LIME/SHAP conceptually

# Scikit-learn utilities
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models for interpretability
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# XGBoost can also be used, but for simplicity in this refactoring, we'll stick to sklearn native for now
# from xgboost import XGBClassifier, XGBRegressor

# Conceptual LIME and SHAP imports (actual usage would require installation)
# import lime
# import lime.lime_tabular
# import shap

# --- API Functions ---

def create_interpretability_sample_data_api(
    n_samples: int = 300, 
    n_features: int = 10, 
    n_informative_num: int = 5, 
    n_cat_features: int = 2,
    task: str = 'classification', 
    n_classes: int = 2, 
    random_state: int = 789
) -> tuple[pd.DataFrame, pd.Series]:
    """
    生成用于模型可解释性演示的样本数据 (数值型和类别型特征)。

    参数:
    - n_samples (int): 样本数量。
    - n_features (int): 特征总数 (数值型 + 类别型)。
    - n_informative_num (int): 数值特征中的信息性特征数量。
    - n_cat_features (int): 类别特征的数量。
    - task (str): 'classification' 或 'regression'。
    - n_classes (int): 如果是分类任务, 分类的类别数量。
    - random_state (int): 随机种子。

    返回:
    - tuple[pd.DataFrame, pd.Series]: 特征DataFrame (X) 和目标Series (y)。
    """
    if n_features <= n_cat_features:
        raise ValueError("总特征数必须大于类别特征数。")
    
    n_num_features = n_features - n_cat_features

    if task == 'classification':
        X_num, y_array = make_classification(
            n_samples=n_samples, n_features=n_num_features, 
            n_informative=n_informative_num, n_classes=n_classes, 
            random_state=random_state, n_redundant=max(0, n_num_features - n_informative_num -1), n_repeated=0
        )
    elif task == 'regression':
        X_num, y_array = make_regression(
            n_samples=n_samples, n_features=n_num_features,
            n_informative=n_informative_num, random_state=random_state
        )
    else:
        raise ValueError("任务类型必须是 'classification' 或 'regression'")

    df_num = pd.DataFrame(X_num, columns=[f'num_feat_{i+1}' for i in range(n_num_features)])
    df_cat = pd.DataFrame()

    for i in range(n_cat_features):
        col_name = f'cat_feat_{i+1}'
        # 生成更多样化的类别
        if n_samples < 5: num_unique_cats = 2
        elif n_samples < 20: num_unique_cats = np.random.randint(2,4)
        else: num_unique_cats = np.random.randint(2,5)
        
        choices = [f'{col_name}_v{j}' for j in range(num_unique_cats)]
        # 确保随机选择不因样本过少而出错
        cat_data = np.random.choice(choices, n_samples if n_samples > 0 else 1)
        df_cat[col_name] = cat_data
    
    X_df = pd.concat([df_num, df_cat], axis=1)
    y_series = pd.Series(y_array, name='target')
    
    # print(f"API: 已创建可解释性 {task} 数据集: X 形状 {X_df.shape}, y 形状 {y_series.shape}")
    return X_df, y_series


def get_interpretability_preprocessor_api(
    X_df: pd.DataFrame,
    num_impute_strategy: str = 'median',
    cat_impute_strategy: str = 'most_frequent',
    scaler_type: str = 'standard' # 'standard' or 'minmax' or None
) -> ColumnTransformer:
    """
    为可解释性模型创建一个预处理器 (数值型特征缩放，类别型特征独热编码)。

    参数:
    - X_df (pd.DataFrame): 输入的特征DataFrame。
    - num_impute_strategy (str): 数值特征的插补策略。
    - cat_impute_strategy (str): 类别特征的插补策略。
    - scaler_type (str): 数值特征的缩放类型 ('standard', 'minmax', None)。

    返回:
    - ColumnTransformer: 配置好的预处理器。
    """
    numeric_features = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_df.select_dtypes(include='object').columns.tolist()
    
    numeric_steps = [('imputer', SimpleImputer(strategy=num_impute_strategy))]
    if scaler_type == 'standard':
        numeric_steps.append(('scaler', StandardScaler()))
    elif scaler_type == 'minmax':
        numeric_steps.append(('scaler', MinMaxScaler()))
        
    numeric_pipeline = Pipeline(numeric_steps)
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=cat_impute_strategy)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easier feature name handling
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ], 
        remainder='passthrough' # 通常不推荐在最终模型中使用passthrough，除非明确知道其含义
    )
    return preprocessor

def explain_tree_feature_importance_api(
    pipeline_model: Pipeline, 
    X_df_original_cols: list[str], # 用于在获取特征名失败时的回退
    plot_top_n: int = 10,
    output_dir: str = "model_interpretability_outputs"
) -> pd.DataFrame | None:
    """
    从基于树的模型Pipeline中提取并可视化特征重要性。

    参数:
    - pipeline_model (Pipeline): 包含预处理器和树模型的Pipeline。
    - X_df_original_cols (list[str]): 原始DataFrame的列名列表，用于回退。
    - plot_top_n (int): 可视化最重要的N个特征。
    - output_dir (str): 保存图表的目录。

    返回:
    - pd.DataFrame | None: 包含特征重要性的DataFrame，如果模型不支持则返回None。
    """
    print("\n--- 解释树模型特征重要性 (API) ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    model_step = pipeline_model.named_steps.get('model')
    if not model_step or not hasattr(model_step, 'feature_importances_'):
        print(f"模型 {type(model_step).__name__ if model_step else '未知'} 不支持 'feature_importances_' 属性。")
        return None

    importances = model_step.feature_importances_
    
    preprocessor_step = pipeline_model.named_steps.get('preprocessor')
    feature_names_out = []
    try:
        if preprocessor_step:
            feature_names_out = preprocessor_step.get_feature_names_out()
    except Exception as e:
        print(f"从预处理器获取特征名失败: {e}。尝试使用原始列名或通用名。")

    if not list(feature_names_out) or len(feature_names_out) != len(importances):
        print(f"特征名数量 ({len(feature_names_out)}) 与重要性值数量 ({len(importances)}) 不匹配。")
        if len(X_df_original_cols) == len(importances): # 这种情况不太可能，因为预处理会改变列数
            feature_names_out = X_df_original_cols
            print("  警告: 使用原始列名，可能与处理后的特征不完全对应。")
        else:
            feature_names_out = [f"processed_feat_{i}" for i in range(len(importances))]
            print("  警告: 使用通用特征名。")
            
    importance_df = pd.DataFrame({'feature': feature_names_out, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False).head(plot_top_n)
    
    print(f"模型 ({type(model_step).__name__}) 的前 {plot_top_n} 个特征重要性:")
    print(importance_df)

    plt.figure(figsize=(12, 7))
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    plt.title(f'前 {plot_top_n} 特征重要性 ({type(model_step).__name__})')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"feature_importance_{type(model_step).__name__}.png")
    plt.savefig(plot_path)
    print(f"特征重要性图表已保存至: {plot_path}")
    plt.close()
    return importance_df

def explain_linear_model_coefficients_api(
    pipeline_model: Pipeline, 
    X_df_original_cols: list[str],
    plot_top_n: int = 10,
    output_dir: str = "model_interpretability_outputs"
) -> pd.DataFrame | None:
    """
    从线性模型Pipeline中提取并可视化系数。

    参数:
    - pipeline_model (Pipeline): 包含预处理器和线性模型的Pipeline。
    - X_df_original_cols (list[str]): 原始DataFrame的列名列表，用于回退。
    - plot_top_n (int): 可视化绝对值最大的N个系数。
    - output_dir (str): 保存图表的目录。

    返回:
    - pd.DataFrame | None: 包含模型系数的DataFrame，如果模型不支持则返回None。
    """
    print("\n--- 解释线性模型系数 (API) ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    model_step = pipeline_model.named_steps.get('model')
    if not model_step or not hasattr(model_step, 'coef_'):
        print(f"模型 {type(model_step).__name__ if model_step else '未知'} 不支持 'coef_' 属性。")
        return None

    coefficients = model_step.coef_.flatten() # 对多分类(如OvR LogisticRegression)的coef_可能是二维的
    
    preprocessor_step = pipeline_model.named_steps.get('preprocessor')
    feature_names_out = []
    try:
        if preprocessor_step:
            feature_names_out = preprocessor_step.get_feature_names_out()
    except Exception as e:
        print(f"从预处理器获取特征名失败: {e}。尝试使用原始列名或通用名。")

    if not list(feature_names_out) or len(feature_names_out) != len(coefficients):
        print(f"特征名数量 ({len(feature_names_out)}) 与系数值数量 ({len(coefficients)}) 不匹配。")
        # 逻辑回归二分类时，coef_是 (1, n_features)。多分类(k类)时是(k, n_features)或(k-1, n_features)
        # 这里简单处理，假设我们关心的是每个特征的一个系数值
        # 对于多分类LogisticRegression且coef_是2D的, 我们可能需要为每个类别分别显示或选择一个。
        # 为了简化，如果coefficients是一维的，而feature_names_out是多维的，这里可能存在问题。
        # 此处假设coefficients已经被正确展平为与预期特征数量相符。
        if model_step.coef_.ndim > 1 and len(coefficients) > len(feature_names_out) and len(feature_names_out) > 0 :
             # 例如，3分类产生3组系数，但get_feature_names_out只有一套特征名。
             # 需要决定如何展示。一种方式是为每个类分别展示，或取平均/最大等。
             # 此处简化：如果系数数量是特征名的倍数，则可能意味着多类别系数。
             num_classes_in_coef = model_step.coef_.shape[0] if model_step.coef_.ndim > 1 else 1
             print(f"  警告: 模型系数可能是多分类的 ({num_classes_in_coef} 组系数)，但只有一套特征名。将显示第一组系数。")
             coefficients = model_step.coef_[0] # 取第一组系数

        if len(X_df_original_cols) == len(coefficients) and len(feature_names_out) != len(coefficients) : # 不太可能
            feature_names_out = X_df_original_cols
            print("  警告: 使用原始列名，可能与处理后的特征不完全对应。")
        elif len(feature_names_out) != len(coefficients): # 如果仍然不匹配，使用通用名
            feature_names_out = [f"processed_feat_{i}" for i in range(len(coefficients))]
            print("  警告: 使用通用特征名。")

    coef_df = pd.DataFrame({'feature': feature_names_out, 'coefficient': coefficients})
    coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False).head(plot_top_n)
    
    print(f"模型 ({type(model_step).__name__}) 的前 {plot_top_n} 个系数 (按绝对值):")
    print(coef_df[['feature', 'coefficient']])

    plt.figure(figsize=(12, 7))
    sns.barplot(x='coefficient', y='feature', data=coef_df, palette='coolwarm_r') # 反转coolwarm调色板
    plt.title(f'前 {plot_top_n} 系数 ({type(model_step).__name__})')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"coefficients_{type(model_step).__name__}.png")
    plt.savefig(plot_path)
    print(f"模型系数图表已保存至: {plot_path}")
    plt.close()
    return coef_df

def explain_shap_conceptual_api():
    """提供SHAP (SHapley Additive exPlanations) 的概念性解释。"""
    print("\n--- 3. SHAP (SHapley Additive exPlanations) - 概念性解释 (API) ---")
    print("SHAP (SHapley Additive exPlanations) 是一种博弈论方法，用于解释任何机器学习模型的输出。")
    print("它将预测归因于每个特征，即每个特征对特定预测的贡献值。")
    print("\n核心概念:")
    print("  - 模型无关性: 可用于解释任何模型。")
    print("  - 局部解释: 解释单个预测。")
    print("  - 全局解释: 通过聚合局部解释来理解模型的整体行为。")
    print("  - 特征重要性: SHAP值可以提供比标准特征重要性更稳健的度量。")
    print("  - SHAP图: summary_plot, dependence_plot, force_plot等可视化工具。")
    print("\n使用步骤概览 (以TreeExplainer为例):")
    print("  1. 安装: `pip install shap`")
    print("  2. 训练模型: `model = RandomForestClassifier().fit(X_train_processed, y_train)`")
    print("  3. 创建Explainer: `explainer = shap.TreeExplainer(model)`")
    print("     (对于非树模型，可使用 `shap.KernelExplainer(model.predict_proba, X_train_processed_sample)` 或其他特定explainer)")
    print("  4. 计算SHAP值: `shap_values = explainer.shap_values(X_test_processed)`")
    print("     (对于分类，`shap_values` 通常是一个列表，每个类别对应一组SHAP值)")
    print("  5. 可视化:")
    print("     - `shap.summary_plot(shap_values, X_test_processed, plot_type='bar')` (全局特征重要性)")
    print("     - `shap.summary_plot(shap_values[class_index], X_test_processed)` (特定类别的特征影响)")
    print("     - `shap.dependence_plot('feature_name', shap_values[class_index], X_test_processed)` (特征交互)")
    print("     - `shap.force_plot(explainer.expected_value[class_index], shap_values[class_index][instance_index,:], X_test_processed.iloc[instance_index,:])` (单个预测解释)")
    print("\n注意:")
    print("  - 数据预处理: SHAP通常应用于预处理后的数据。特征名称的正确传递对于解释至关重要。")
    print("  - 计算成本: KernelExplainer可能计算量较大。")
    print("  - `X_test_processed` 最好是带有列名的Pandas DataFrame，以便绘图时显示正确的特征。")

def explain_lime_conceptual_api():
    """提供LIME (Local Interpretable Model-agnostic Explanations) 的概念性解释。"""
    print("\n--- 4. LIME (Local Interpretable Model-agnostic Explanations) - 概念性解释 (API) ---")
    print("LIME是一种模型无关的技术，通过在预测点附近学习一个可解释的局部代理模型来解释任何黑箱模型的单个预测。")
    print("\n核心概念:")
    print("  - 局部保真度: LIME试图在预测的邻域内准确地模拟黑箱模型。")
    print("  - 可解释性: 代理模型（通常是稀疏线性模型）易于理解。")
    print("  - 模型无关性: 可以应用于任何分类或回归模型。")
    print("\n使用步骤概览:")
    print("  1. 安装: `pip install lime`")
    print("  2. 训练模型: `pipeline_model.fit(X_train_df, y_train)`")
    print("  3. 创建Explainer:")
    print("     `import lime.lime_tabular`")
    print("     `explainer = lime.lime_tabular.LimeTabularExplainer(`")
    print("         `training_data=X_train_processed_np, # 预处理后的训练数据 (numpy array)`")
    print("         `feature_names=preprocessor.get_feature_names_out(), # 处理后的特征名`")
    print("         `class_names=['class0', 'class1'], # 类别名 (分类任务)`")
    print("         `categorical_features=[idx1, idx2...], # 类别特征在处理后数据中的索引`")
    print("         `mode='classification' # 或 'regression'`")
    print("     `)`")
    print("  4. 解释实例:")
    print("     `instance_to_explain = X_test_processed_np[idx]`")
    print("     `# LIME的 predict_fn 需要能处理与 training_data 格式相同的输入，并返回概率`")
    print("     `# 如果模型是Pipeline， predict_fn = lambda x: pipeline_model.predict_proba(preprocessor.inverse_transform(x)) 如果explainer在原始空间扰动`")
    print("     `# 或者，如果explainer在处理后空间扰动，则 predict_fn = lambda x: model_step.predict_proba(x)`")
    print("     `# 简化示例假设 explainer 在处理后的空间工作，且 model_step 是 pipeline 中的模型组件`")
    print("     `predict_fn = lambda x: pipeline_model.named_steps['model'].predict_proba(x) # 假设x是已处理的数据`")
    print("     `explanation = explainer.explain_instance(`")
    print("         `data_row=instance_to_explain,`")
    print("         `predict_fn=predict_fn,`")
    print("         `num_features=5 # 解释中包含的特征数量`")
    print("     `)`")
    print("  5. 可视化/输出解释:")
    print("     `explanation.show_in_notebook(show_table=True)`")
    print("     `# explanation.as_list() # 获取解释列表`")
    print("     `# explanation.save_to_file('lime_report.html')`")
    print("\n注意:")
    print("  - `predict_fn` 的正确设置至关重要，它必须接受与LIME扰动样本相同格式的数据。")
    print("  - 对于包含预处理步骤的Pipeline，`predict_fn` 可能需要先对扰动样本应用逆变换（如果扰动在原始空间），或者直接传递给Pipeline中的模型组件（如果扰动在变换空间）。")
    print("  - 特征选择和扰动方法可能会影响解释的稳定性和质量。")


if __name__ == '__main__':
    print("========== (B 部分) 模型可解释性演示 ==========")
    
    # --- (MI.0) 全局设置 ---
    main_random_seed_mi = 800
    output_directory_mi = "model_interpretability_outputs" # 主输出目录
    if not os.path.exists(output_directory_mi):
        os.makedirs(output_directory_mi)
        print(f"创建主输出目录: {output_directory_mi}")

    # --- (MI.A) 分类任务模型可解释性 ---
    print("\n\n===== MI.A: 分类任务可解释性演示 =====")
    # A.1 生成分类数据
    X_clf_mi_df, y_clf_mi = create_interpretability_sample_data_api(
        n_samples=350, n_features=12, n_informative_num=6, n_cat_features=3,
        task='classification', n_classes=2, random_state=main_random_seed_mi
    )
    print(f"分类数据: X {X_clf_mi_df.shape}, y {y_clf_mi.shape}, 类别 {np.unique(y_clf_mi)}")
    
    # 获取原始列名列表
    original_clf_columns = X_clf_mi_df.columns.tolist()

    # A.2 数据分割
    X_train_clf_df, X_test_clf_df, y_train_clf, y_test_clf = train_test_split(
        X_clf_mi_df, y_clf_mi, test_size=0.25, random_state=main_random_seed_mi, stratify=y_clf_mi
    )

    # A.3 创建预处理器
    preprocessor_clf_mi = get_interpretability_preprocessor_api(X_train_clf_df.copy()) # 使用副本以防修改原始数据

    # A.4.1 逻辑回归 (线性模型)
    print("\n--- MI.A.4.1: 逻辑回归分类器 ---")
    log_reg_mi_pipeline = Pipeline([
        ('preprocessor', preprocessor_clf_mi),
        ('model', LogisticRegression(solver='liblinear', random_state=main_random_seed_mi))
    ])
    log_reg_mi_pipeline.fit(X_train_clf_df, y_train_clf)
    print(f"逻辑回归分类器在测试集上的准确度: {log_reg_mi_pipeline.score(X_test_clf_df, y_test_clf):.4f}")
    explain_linear_model_coefficients_api(
        log_reg_mi_pipeline, 
        X_df_original_cols=original_clf_columns, 
        output_dir=output_directory_mi
    )

    # A.4.2 随机森林分类器 (树模型)
    print("\n--- MI.A.4.2: 随机森林分类器 ---")
    # 需要为每个pipeline重新实例化预处理器，或者深拷贝，以避免状态污染
    preprocessor_clf_rf_mi = get_interpretability_preprocessor_api(X_train_clf_df.copy()) 
    rf_clf_mi_pipeline = Pipeline([
        ('preprocessor', preprocessor_clf_rf_mi), 
        ('model', RandomForestClassifier(n_estimators=60, random_state=main_random_seed_mi, max_depth=6))
    ])
    rf_clf_mi_pipeline.fit(X_train_clf_df, y_train_clf)
    print(f"随机森林分类器在测试集上的准确度: {rf_clf_mi_pipeline.score(X_test_clf_df, y_test_clf):.4f}")
    explain_tree_feature_importance_api(
        rf_clf_mi_pipeline, 
        X_df_original_cols=original_clf_columns,
        output_dir=output_directory_mi
    )

    # --- (MI.B) 回归任务模型可解释性 ---
    print("\n\n===== MI.B: 回归任务可解释性演示 =====")
    # B.1 生成回归数据
    X_reg_mi_df, y_reg_mi = create_interpretability_sample_data_api(
        n_samples=320, n_features=10, n_informative_num=5, n_cat_features=2,
        task='regression', random_state=main_random_seed_mi + 1
    )
    print(f"回归数据: X {X_reg_mi_df.shape}, y {y_reg_mi.shape}")
    original_reg_columns = X_reg_mi_df.columns.tolist()

    # B.2 数据分割
    X_train_reg_df, X_test_reg_df, y_train_reg, y_test_reg = train_test_split(
        X_reg_mi_df, y_reg_mi, test_size=0.25, random_state=main_random_seed_mi + 1
    )

    # B.3 创建预处理器
    preprocessor_reg_mi = get_interpretability_preprocessor_api(X_train_reg_df.copy())

    # B.4.1 线性回归
    print("\n--- MI.B.4.1: 线性回归器 ---")
    lin_reg_mi_pipeline = Pipeline([
        ('preprocessor', preprocessor_reg_mi),
        ('model', LinearRegression())
    ])
    lin_reg_mi_pipeline.fit(X_train_reg_df, y_train_reg)
    print(f"线性回归器在测试集上的 R^2 分数: {lin_reg_mi_pipeline.score(X_test_reg_df, y_test_reg):.4f}")
    explain_linear_model_coefficients_api(
        lin_reg_mi_pipeline, 
        X_df_original_cols=original_reg_columns,
        output_dir=output_directory_mi
    )
    
    # B.4.2 随机森林回归器
    print("\n--- MI.B.4.2: 随机森林回归器 ---")
    preprocessor_reg_rf_mi = get_interpretability_preprocessor_api(X_train_reg_df.copy())
    rf_reg_mi_pipeline = Pipeline([
        ('preprocessor', preprocessor_reg_rf_mi),
        ('model', RandomForestRegressor(n_estimators=55, random_state=main_random_seed_mi + 1, max_depth=5))
    ])
    rf_reg_mi_pipeline.fit(X_train_reg_df, y_train_reg)
    print(f"随机森林回归器在测试集上的 R^2 分数: {rf_reg_mi_pipeline.score(X_test_reg_df, y_test_reg):.4f}")
    explain_tree_feature_importance_api(
        rf_reg_mi_pipeline, 
        X_df_original_cols=original_reg_columns,
        output_dir=output_directory_mi
    )

    # --- (MI.C) LIME 和 SHAP 概念解释 ---
    print("\n\n===== MI.C: LIME 和 SHAP 概念解释 =====")
    explain_shap_conceptual_api()
    explain_lime_conceptual_api()

    print("\n\n========== (B 部分) 模型可解释性演示结束 ==========") 