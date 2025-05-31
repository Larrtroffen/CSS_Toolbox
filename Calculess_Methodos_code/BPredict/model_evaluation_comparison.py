import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Scikit-learn utilities
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models for comparison (can be extended)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor # XGBoost as an example

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)

# 定义输出目录
OUTPUT_DIR = "model_evaluation_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建目录: {OUTPUT_DIR}")

def create_model_comparison_sample_data_api(
    n_samples: int = 500, 
    n_features: int = 15, 
    n_informative_num: int = 7, 
    n_cat_features: int = 3,
    task: str = 'classification', 
    n_classes: int = 2,
    nan_percentage_num: float = 0.1,
    nan_percentage_cat: float = 0.1,
    random_state: int = 123
) -> tuple[pd.DataFrame, pd.Series]:
    """
    创建用于模型比较的样本数据集，包含数值和类别特征及NaN值。

    参数:
    - n_samples (int): 样本数量。
    - n_features (int): 特征总数 (数值 + 类别)。
    - n_informative_num (int): 有信息量的数值特征数。
    - n_cat_features (int): 类别特征数。
    - task (str): 'classification' 或 'regression'。
    - n_classes (int): (仅分类) 类别数。
    - nan_percentage_num (float): 数值特征中NaN的比例。
    - nan_percentage_cat (float): 类别特征中NaN的比例。
    - random_state (int): 随机种子。

    返回:
    - pd.DataFrame: 特征DataFrame。
    - pd.Series: 目标Series。
    """
    np.random.seed(random_state)
    n_num_features = n_features - n_cat_features
    if n_num_features < 0: raise ValueError("总特征数必须大于等于类别特征数。")
    if n_informative_num > n_num_features: n_informative_num = n_num_features

    if task == 'classification':
        X_num_core, y = make_classification(
            n_samples=n_samples, n_features=n_informative_num, 
            n_informative=n_informative_num, n_redundant=0, n_repeated=0, 
            n_classes=n_classes, random_state=random_state, n_clusters_per_class=1
        )
        if n_num_features > n_informative_num:
            X_num_noise = np.random.randn(n_samples, n_num_features - n_informative_num) * 0.5
            X_num = np.hstack((X_num_core, X_num_noise))
        else:
            X_num = X_num_core
    else: # 回归
        X_num_core, y = make_regression(
            n_samples=n_samples, n_features=n_informative_num, 
            n_informative=n_informative_num, noise=0.2, random_state=random_state
        )
        if n_num_features > n_informative_num:
            X_num_noise = np.random.randn(n_samples, n_num_features - n_informative_num) * 0.1
            X_num = np.hstack((X_num_core, X_num_noise))
        else:
            X_num = X_num_core

    df = pd.DataFrame(X_num, columns=[f'num_feat_{i}' for i in range(n_num_features)])
    
    cat_options = ['P_Opt1', 'P_Opt2', 'Q_OptA', 'Q_OptB', 'R_ChoiceX', 'R_ChoiceY', 'S_ItemHigh', 'S_ItemLow']
    for i in range(n_cat_features):
        cat_name = f'cat_feat_comp_{chr(ord("A")+i)}'
        num_unique_this_cat = np.random.randint(2, 4)
        choices = np.random.choice(cat_options, num_unique_this_cat, replace=False).tolist()
        df[cat_name] = np.random.choice(choices, n_samples)

    # 引入NaN
    for col_idx in range(n_num_features):
        if nan_percentage_num > 0:
            nan_indices = np.random.choice(df.index, size=int(n_samples * nan_percentage_num), replace=False)
            df.iloc[nan_indices, col_idx] = np.nan
    for col_name in df.select_dtypes(include='object').columns:
        if nan_percentage_cat > 0:
            nan_indices = np.random.choice(df.index, size=int(n_samples * nan_percentage_cat), replace=False)
            df.loc[nan_indices, col_name] = np.nan 
            
    print(f"创建模型比较 {task} 数据集: {df.shape}, 目标唯一值: {len(np.unique(y))}")
    return df, pd.Series(y, name='target')

def create_model_comparison_preprocessor_api(X_df_train: pd.DataFrame) -> ColumnTransformer:
    """
    创建并拟合一个用于模型比较的Scikit-learn ColumnTransformer预处理器。
    包含对NaN的插补、数值特征标准化和类别特征独热编码。

    参数:
    - X_df_train (pd.DataFrame): 用于拟合预处理器的训练特征DataFrame。

    返回:
    - ColumnTransformer: 拟合好的预处理器。
    """
    numeric_features = X_df_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_df_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop' # 丢弃其他未指定列
    )
    print("创建模型比较预处理器...")
    preprocessor.fit(X_df_train) # 在训练数据上拟合
    print("模型比较预处理器拟合完成。")
    return preprocessor

def train_and_evaluate_single_model_api(
    model_pipeline: Pipeline, 
    X_train: pd.DataFrame, # Changed to pd.DataFrame for consistency with how preprocessor is defined
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model_name: str, 
    task: str = 'classification'
) -> dict:
    """
    训练单个模型（作为Pipeline的一部分）并评估其性能。

    参数:
    - model_pipeline (Pipeline): 包含预处理器和模型的完整Pipeline。
    - X_train, y_train: 训练数据和标签 (应为原始DataFrame/Series)。
    - X_test, y_test: 测试数据和标签 (应为原始DataFrame/Series)。
    - model_name (str): 模型名称。
    - task (str): 'classification' 或 'regression'。

    返回:
    - dict: 包含模型名称和各项评估指标的字典。
    """
    print(f"--- 训练与评估: {model_name} ({task.capitalize()}) ---")
    model_pipeline.fit(X_train, y_train) 
    y_pred = model_pipeline.predict(X_test)
    
    results = {'model_name': model_name, 'task': task}
    num_unique_y_test = len(np.unique(y_test))

    if task == 'classification':
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision_weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        results['recall_weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        results['f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            if hasattr(model_pipeline.named_steps['model'], 'predict_proba'):
                if num_unique_y_test == 2: # Binary classification
                    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
                    results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                    results['y_pred_proba_for_roc_plot'] = y_pred_proba 
                elif num_unique_y_test > 2: # Multiclass classification
                    y_pred_proba = model_pipeline.predict_proba(X_test)
                    results['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                else: # Single class in y_test, ROC AUC not applicable
                    results['roc_auc'] = np.nan
            else: # Model does not have predict_proba
                results['roc_auc'] = np.nan
        except Exception as e_auc:
            print(f"  计算 {model_name} 的ROC AUC失败: {e_auc}")
            results['roc_auc'] = np.nan
        
        print(f"  准确率: {results.get('accuracy', np.nan):.4f}, F1加权: {results.get('f1_weighted', np.nan):.4f}, ROC AUC: {results.get('roc_auc', results.get('roc_auc_ovr', np.nan)):.4f}")
    else: # 回归
        results['r2_score'] = r2_score(y_test, y_pred)
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(y_test, y_pred)
        print(f"  R2分数: {results.get('r2_score', np.nan):.4f}, RMSE: {results.get('rmse', np.nan):.4f}")
    return results

def compare_multiple_models_api(
    X_df: pd.DataFrame, 
    y_series: pd.Series, 
    model_configs: list[tuple[str, any]], 
    task: str = 'classification', 
    test_size: float = 0.25, 
    random_state: int = 42,
    plot_roc_for_binary_clf: bool = True
) -> tuple[pd.DataFrame, str | None]:
    """
    比较多个模型在给定数据集上的性能。

    参数:
    - X_df (pd.DataFrame): 原始特征DataFrame。
    - y_series (pd.Series): 原始目标Series。
    - model_configs (list): 模型配置列表，每个元素是 (模型名称, 模型实例) 的元组。
    - task (str): 'classification' 或 'regression'。
    - test_size (float): 测试集比例。
    - random_state (int): 随机种子。
    - plot_roc_for_binary_clf (bool): 如果是二分类任务，是否绘制组合ROC曲线图。

    返回:
    - pd.DataFrame: 包含各模型性能指标的DataFrame。
    - str | None: 如果绘图，则为ROC曲线图的保存路径，否则为None。
    """
    print(f"\n===== 开始多模型比较 ({task.capitalize()}) =====")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y_series, test_size=test_size, random_state=random_state,
        stratify=y_series if task == 'classification' and y_series.nunique() > 1 else None
    )

    all_results_list = []

    for model_name, model_instance in model_configs:
        # 定义预处理器 (每次循环都重新定义，以确保Pipeline的独立性)
        numeric_features = X_train_df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_pipeline_def = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ])
        categorical_pipeline_def = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        current_preprocessor_def = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline_def, numeric_features),
                ('cat', categorical_pipeline_def, categorical_features)
            ], 
            remainder='drop'
        )

        full_pipeline = Pipeline(steps=[
            ('preprocessor', current_preprocessor_def),
            ('model', model_instance)
        ])
        
        model_eval_results = train_and_evaluate_single_model_api(
            full_pipeline, X_train_df, y_train, X_test_df, y_test, model_name, task
        )
        all_results_list.append(model_eval_results)
    
    results_summary_df = pd.DataFrame(all_results_list)
    if not results_summary_df.empty:
        results_summary_df.set_index('model_name', inplace=True)
        
    print("\n--- 模型性能总结表 (Train/Test Split) ---")
    print(results_summary_df)
    summary_csv_path = os.path.join(OUTPUT_DIR, f"model_comparison_{task}_summary_traintest.csv")
    results_summary_df.to_csv(summary_csv_path)
    print(f"  Train/Test性能总结已保存到: {summary_csv_path}")

    roc_plot_path = None
    if task == 'classification' and y_series.nunique() == 2 and plot_roc_for_binary_clf:
        plt.figure(figsize=(11, 9))
        for res_dict in all_results_list:
            if res_dict.get('y_pred_proba_for_roc_plot') is not None:
                fpr, tpr, _ = roc_curve(y_test, res_dict['y_pred_proba_for_roc_plot'])
                auc_val = res_dict.get('roc_auc', np.nan)
                plt.plot(fpr, tpr, label=f"{res_dict['model_name']} (AUC = {auc_val:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测线')
        plt.xlabel('假正例率 (False Positive Rate)')
        plt.ylabel('真正例率 (True Positive Rate)')
        plt.title(f'模型比较：组合ROC曲线 ({task.capitalize()})')
        plt.legend(loc='lower right')
        plt.grid(True)
        roc_plot_path = os.path.join(OUTPUT_DIR, f"model_comparison_{task}_roc_curves.png")
        plt.savefig(roc_plot_path)
        plt.close()
        print(f"  组合ROC曲线图已保存到: {roc_plot_path}")
        
    return results_summary_df, roc_plot_path

def perform_cross_validation_comparison_api(
    X_df: pd.DataFrame, 
    y_series: pd.Series, 
    model_configs: list[tuple[str, any]],
    task: str = 'classification',
    cv_folds: int = 5,
    random_state: int = 42,
    scoring_metric: str | None = None 
) -> pd.DataFrame:
    """
    使用交叉验证比较多个模型的性能 (在完整的Pipeline上)。

    参数:
    - X_df (pd.DataFrame): 原始特征DataFrame。
    - y_series (pd.Series): 原始目标Series。
    - model_configs (list): 模型配置列表。
    - task (str): 'classification' 或 'regression'。
    - cv_folds (int): 交叉验证折数。
    - random_state (int): 随机种子 (用于KFold)。
    - scoring_metric (str | None): cross_val_score使用的评估指标。如果None，则使用默认。

    返回:
    - pd.DataFrame: 包含各模型CV性能指标的DataFrame。
    """
    print(f"\n===== 开始交叉验证模型比较 ({task.capitalize()}) =====")
    cv_results_list = []
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    numeric_features = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_pipeline_def = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipeline_def = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor_def = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline_def, numeric_features),
            ('cat', categorical_pipeline_def, categorical_features)
        ], remainder='drop'
    )

    current_scoring_metric = scoring_metric
    if current_scoring_metric is None:
        if task == 'classification':
            current_scoring_metric = 'roc_auc' if y_series.nunique() == 2 else 'accuracy'
        else:
            current_scoring_metric = 'r2'
    print(f"  CV将使用评估指标: {current_scoring_metric}")

    for model_name, model_instance in model_configs:
        print(f"  交叉验证模型: {model_name}...")
        full_pipeline_cv = Pipeline(steps=[
            ('preprocessor', preprocessor_def), 
            ('model', model_instance)
        ])
        try:
            cv_scores = cross_val_score(full_pipeline_cv, X_df, y_series, cv=kfold, scoring=current_scoring_metric, n_jobs=-1)
            cv_results_list.append({
                'model_name': model_name,
                'task': task,
                'cv_metric': current_scoring_metric,
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'all_cv_scores': cv_scores.tolist()
            })
            print(f"    {model_name} CV 平均 {current_scoring_metric}: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})\n")
        except Exception as e_cv:
            print(f"    {model_name} 的交叉验证失败: {e_cv}")
            cv_results_list.append({
                'model_name': model_name, 'task': task, 'cv_metric': current_scoring_metric,
                'mean_score': np.nan, 'std_score': np.nan, 'all_cv_scores': []
            })

    cv_summary_df = pd.DataFrame(cv_results_list)
    if not cv_summary_df.empty:
        cv_summary_df.set_index('model_name', inplace=True)
        
    print("\n--- 交叉验证性能总结表 ---")
    print(cv_summary_df)
    cv_summary_csv_path = os.path.join(OUTPUT_DIR, f"model_cv_comparison_{task}_summary.csv")
    cv_summary_df.to_csv(cv_summary_csv_path)
    print(f"  交叉验证总结已保存到: {cv_summary_csv_path}")
    return cv_summary_df

def discuss_model_selection_factors_api() -> None:
    """打印选择模型时需要考虑的关键因素。"""
    print("\n--- 讨论：模型选择的关键因素 --- ")
    factors = {
        "性能指标 (Performance Metrics)": "根据具体业务问题选择最重要的指标 (例如，准确率、精确率、召回率、F1分数、ROC AUC、R2分数、RMSE等)。不同任务和场景下，指标的侧重点不同 (例如，医疗诊断中可能更关注召回率)。",
        "计算成本 (Computational Cost)": "包括训练时间、预测时间、内存消耗。简单模型 (如线性/逻辑回归) 通常计算成本较低，复杂模型 (如集成学习、深度神经网络) 成本较高。",
        "可解释性 (Interpretability)": "模型是否易于理解其决策过程。线性模型可解释性强，树模型提供特征重要性，深度学习模型通常被视为“黑箱”。在金融、医疗等高风险领域，可解释性非常重要。",
        "数据量与数据复杂度 (Data Size & Complexity)": "小数据集可能更适合简单模型以避免过拟合。大规模、高维度、复杂关系的数据可能从复杂模型中受益。",
        "模型假设 (Assumptions of the Model)": "了解模型的基本假设 (例如，线性回归的线性关系、特征独立性等)。如果数据不满足模型假设，模型性能可能会受影响。",
        "鲁棒性与稳定性 (Robustness & Stability)": "模型对数据噪声、异常值或分布变化的敏感程度。交叉验证有助于评估模型的稳定性。",
        "部署与维护 (Deployment & Maintenance)": "模型的部署难易程度，依赖库，以及后续维护成本。",
        "过拟合与欠拟合 (Overfitting & Underfitting)": "通过比较训练集和验证集/测试集的性能来判断。使用正则化、交叉验证等方法来缓解。"
    }
    for factor, description in factors.items():
        print(f"\n{factor}:")
        print(f"  - {description}")
    print("\n结论：模型选择是一个权衡的过程，通常没有“一体适用”的最佳模型。需要结合具体问题、数据特性和资源限制进行综合考虑，并通过实验和严格评估来做出决策。")

if __name__ == '__main__':
    # --- 分类任务模型比较演示 ---
    print("===== 开始模型评估与比较演示 (分类) =====\n")
    X_clf_comp, y_clf_comp = create_model_comparison_sample_data_api(
        n_samples=600, n_features=18, n_informative_num=8, n_cat_features=4,
        task='classification', n_classes=2, # 确保二分类以演示ROC图
        random_state=456, nan_percentage_num=0.05, nan_percentage_cat=0.05
    )
    
    clf_models_to_test = [
        ('逻辑回归', LogisticRegression(solver='liblinear', random_state=456, max_iter=250)),
        ('随机森林分类器', RandomForestClassifier(n_estimators=60, random_state=456, max_depth=6, class_weight='balanced')),
        ('XGBoost分类器', XGBClassifier(n_estimators=60, random_state=456, use_label_encoder=False, eval_metric='logloss', max_depth=4))
    ]
    
    clf_results_df, clf_roc_path = compare_multiple_models_api(
        X_clf_comp, y_clf_comp, clf_models_to_test, task='classification', plot_roc_for_binary_clf=True, random_state=456
    )
    if clf_roc_path: print(f"ROC曲线图保存在: {clf_roc_path}")

    clf_cv_results_df = perform_cross_validation_comparison_api(
        X_clf_comp, y_clf_comp, clf_models_to_test, task='classification', cv_folds=3, 
        scoring_metric='f1_weighted', random_state=456
    )

    print("\n" + "="*70 + "\n")

    # --- 回归任务模型比较演示 ---
    print("===== 开始模型评估与比较演示 (回归) =====\n")
    X_reg_comp, y_reg_comp = create_model_comparison_sample_data_api(
        n_samples=550, n_features=16, n_informative_num=7, n_cat_features=3,
        task='regression', random_state=789, nan_percentage_num=0.05, nan_percentage_cat=0.05
    )
    
    reg_models_to_test = [
        ('线性回归', LinearRegression()),
        ('随机森林回归器', RandomForestRegressor(n_estimators=55, random_state=789, max_depth=5)),
        ('XGBoost回归器', XGBRegressor(n_estimators=55, random_state=789, max_depth=4))
    ]
    
    reg_results_df, _ = compare_multiple_models_api(
        X_reg_comp, y_reg_comp, reg_models_to_test, task='regression', random_state=789
    )

    reg_cv_results_df = perform_cross_validation_comparison_api(
        X_reg_comp, y_reg_comp, reg_models_to_test, task='regression', cv_folds=3, 
        scoring_metric='neg_root_mean_squared_error', random_state=789
    )

    discuss_model_selection_factors_api()

    print("\n\n===== 模型评估与比较演示全部完成。 =====")
    print(f"所有输出 (CSV表格, 图表) 保存在 '{OUTPUT_DIR}' 目录中。")
