import numpy as np
import pandas as pd
import os
import json

# Scikit-learn utilities
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Example Models for tuning
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# Metrics (can be used for scoring in HPO)
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error, get_scorer

# 定义输出目录 (可选，如果HPO过程需要保存中间结果或最终模型)
OUTPUT_DIR = "hyperparameter_tuning_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建目录: {OUTPUT_DIR}")

def create_hpo_sample_data_api(
    n_samples: int = 500, 
    n_features: int = 15, 
    n_informative_num: int = 7, 
    n_cat_features: int = 3,
    task: str = 'classification', 
    n_classes: int = 2,
    nan_percentage_num: float = 0.0, # Default to no NaNs for HPO examples to simplify
    nan_percentage_cat: float = 0.0, # Focus is on HPO, not complex preprocessing here
    random_state: int = 456
) -> tuple[pd.DataFrame, pd.Series]:
    """
    创建用于超参数优化的样本数据集。
    默认不引入NaN值以简化HPO演示，假设预处理在Pipeline中处理。

    参数与返回值同 model_evaluation_comparison.py 中的对应函数。
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
            X_num_noise = np.random.randn(n_samples, n_num_features - n_informative_num) * 0.3
            X_num = np.hstack((X_num_core, X_num_noise))
        else: X_num = X_num_core
    else: # 回归
        X_num_core, y = make_regression(
            n_samples=n_samples, n_features=n_informative_num, 
            n_informative=n_informative_num, noise=0.15, random_state=random_state
        )
        if n_num_features > n_informative_num:
            X_num_noise = np.random.randn(n_samples, n_num_features - n_informative_num) * 0.05
            X_num = np.hstack((X_num_core, X_num_noise))
        else: X_num = X_num_core

    df = pd.DataFrame(X_num, columns=[f'num_hpo_{i}' for i in range(n_num_features)])
    cat_options = ['CatX_A', 'CatX_B', 'CatY_P', 'CatY_Q', 'CatZ_1', 'CatZ_2']
    for i in range(n_cat_features):
        cat_name = f'cat_hpo_{chr(ord("M")+i)}'
        num_unique_this_cat = np.random.randint(2, 4)
        choices = np.random.choice(cat_options, num_unique_this_cat, replace=False).tolist()
        df[cat_name] = np.random.choice(choices, n_samples)
    
    if nan_percentage_num > 0 or nan_percentage_cat > 0:
        # Introduce NaNs for numeric features
        for col_idx in range(n_num_features):
            if nan_percentage_num > 0:
                nan_indices_num = np.random.choice(df.index, size=int(n_samples * nan_percentage_num), replace=False)
                df.iloc[nan_indices_num, col_idx] = np.nan
        # Introduce NaNs for categorical features
        for col_name_cat in df.select_dtypes(include='object').columns:
            if nan_percentage_cat > 0:
                nan_indices_cat = np.random.choice(df.index, size=int(n_samples * nan_percentage_cat), replace=False)
                df.loc[nan_indices_cat, col_name_cat] = np.nan
        print(f"  注意: 在HPO数据中引入了NaN值。")
            
    print(f"创建HPO {task} 数据集: {df.shape}, 目标唯一值: {len(np.unique(y))}, NaN总数: {df.isnull().sum().sum()}")
    return df, pd.Series(y, name='target_hpo')

def get_hpo_preprocessor_definition_api(X_example_df: pd.DataFrame) -> ColumnTransformer:
    """
    获取用于超参数优化的Scikit-learn ColumnTransformer预处理器 *定义* (未拟合)。
    HPO工具会在CV的每一折或每个试验中独立拟合它。

    参数:
    - X_example_df (pd.DataFrame): 一个样本DataFrame，用于推断数值和类别列名。

    返回:
    - ColumnTransformer: 未拟合的预处理器定义。
    """
    numeric_features = X_example_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_example_df.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)) # sparse=True often better for HPO
    ])
    
    preprocessor_definition = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop' 
    )
    print("获取HPO预处理器定义完成。")
    return preprocessor_definition

def perform_grid_search_cv_api(
    X_df: pd.DataFrame, 
    y_series: pd.Series, 
    model_instance: any, 
    param_grid: dict, 
    task: str = 'classification',
    cv_folds: int = 3, 
    scoring: str | None = None, 
    random_state_kfold: int = 42,
    n_jobs: int = -1,
    save_best_model_path: str | None = None # Optional path to save the best model
) -> tuple[any, dict, float]:
    """
    使用GridSearchCV执行超参数优化。

    参数:
    - X_df, y_series: 原始特征和目标。
    - model_instance: 要调优的模型实例 (e.g., RandomForestClassifier()).
    - param_grid: GridSearchCV的参数网格 (e.g., {'model__n_estimators': [50, 100]}).
                  注意Pipeline中的命名约定，如 'model__param_name'。
    - task (str): 'classification' 或 'regression'。
    - cv_folds (int): 交叉验证折数。
    - scoring (str | None): 评估指标。如果None，分类用accuracy/roc_auc，回归用r2。
    - random_state_kfold (int): KFold的随机种子。
    - n_jobs (int): GridSearchCV的并行任务数。
    - save_best_model_path (str | None): 如果提供，最佳模型将被保存到此路径 (使用joblib)。

    返回:
    - any: 最佳估计器 (拟合好的Pipeline)。
    - dict: 找到的最佳参数。
    - float: 最佳交叉验证得分。
    """
    print(f"\n--- 开始 GridSearchCV ({model_instance.__class__.__name__}, {task}) ---")
    # HPO 通常在整个可用数据上进行（或在训练集上，然后用独立的测试集评估最终模型）
    # 这里，我们将在传入的X_df, y_series上直接进行CV，不创建额外的train/test split内部。
    # 调用者负责数据的分割策略。

    preprocessor_def = get_hpo_preprocessor_definition_api(X_df) # Get definition based on X_df
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_def),
        ('model', model_instance) 
    ])

    if scoring is None:
        if task == 'classification':
            scoring = 'roc_auc' if y_series.nunique() == 2 else 'accuracy'
        else:
            scoring = 'r2'
    print(f"  GridSearchCV 使用评估指标: {scoring}")

    grid_search = GridSearchCV(
        full_pipeline, param_grid, 
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state_kfold),
        scoring=scoring, verbose=1, n_jobs=n_jobs, refit=True # refit=True ensures best_estimator_ is refitted on whole data
    )
    
    print(f"  正在对参数网格进行GridSearchCV调优... (这可能需要一些时间)")
    grid_search.fit(X_df, y_series) # Fit on the provided X_df, y_series

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"  GridSearchCV 完成。最佳参数: {best_params}")
    print(f"  最佳CV ({scoring}) 得分: {best_cv_score:.4f}")

    if save_best_model_path:
        try:
            import joblib
            joblib.dump(best_estimator, save_best_model_path)
            print(f"  最佳模型已保存到: {save_best_model_path}")
        except Exception as e_save:
            print(f"  保存最佳模型失败: {e_save}")
            
    return best_estimator, best_params, best_cv_score

def perform_randomized_search_cv_api(
    X_df: pd.DataFrame, 
    y_series: pd.Series, 
    model_instance: any, 
    param_distributions: dict, 
    n_iter: int = 10,
    task: str = 'classification',
    cv_folds: int = 3, 
    scoring: str | None = None, 
    random_state_search: int = 42, # For RandomizedSearchCV's sampling
    random_state_kfold: int = 42,  # For KFold
    n_jobs: int = -1,
    save_best_model_path: str | None = None # Optional path to save the best model
) -> tuple[any, dict, float]:
    """
    使用RandomizedSearchCV执行超参数优化。

    参数:
    - param_distributions: RandomizedSearchCV的参数分布字典。
    - n_iter (int): 参数设置的抽样次数。
    - save_best_model_path (str | None): 如果提供，最佳模型将被保存到此路径 (使用joblib)。
    - 其他参数同 perform_grid_search_cv_api。


    返回:
    - 同 perform_grid_search_cv_api。
    """
    print(f"\n--- 开始 RandomizedSearchCV ({model_instance.__class__.__name__}, {task}) ---")
    
    preprocessor_def = get_hpo_preprocessor_definition_api(X_df)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_def),
        ('model', model_instance)
    ])

    if scoring is None:
        if task == 'classification':
            scoring = 'roc_auc' if y_series.nunique() == 2 else 'accuracy'
        else:
            scoring = 'r2'
    print(f"  RandomizedSearchCV 使用评估指标: {scoring}")

    random_search = RandomizedSearchCV(
        full_pipeline, param_distributions, n_iter=n_iter,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state_kfold),
        scoring=scoring, verbose=1, random_state=random_state_search, n_jobs=n_jobs, refit=True
    )

    print(f"  正在进行RandomizedSearchCV调优 (迭代次数: {n_iter})... (这可能需要一些时间)")
    random_search.fit(X_df, y_series)

    best_estimator = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_

    print(f"  RandomizedSearchCV 完成。最佳参数: {best_params}")
    print(f"  最佳CV ({scoring}) 得分: {best_cv_score:.4f}")

    if save_best_model_path:
        try:
            import joblib
            joblib.dump(best_estimator, save_best_model_path)
            print(f"  最佳模型已保存到: {save_best_model_path}")
        except Exception as e_save:
            print(f"  保存最佳模型失败: {e_save}")

    return best_estimator, best_params, best_cv_score

def explain_bayesian_optimization_hpo_conceptual_api() -> None:
    """打印贝叶斯优化进行超参数优化的概念性解释，并提及Hyperopt和Optuna。"""
    print("\n--- 概念: 贝叶斯优化用于超参数优化 (例如 Hyperopt, Optuna) --- ")
    # ... (内容同之前版本，为简洁此处省略) ...
    print("贝叶斯优化是一种序列模型 기반 최적화 (SMBO) 策略，用于在评估成本高昂的黑箱函数中寻找最优解。")
    print("在超参数优化 (HPO) 的背景下:")
    print("1.  目标函数 (Objective Function): 通常是交叉验证的平均性能指标 (例如，AUC、准确率、MSE)。")
    print("2.  黑箱 (Black Box): 我们不知道目标函数的确切数学形式，只能通过实际训练和评估模型来获取其值。")
    print("3.  评估成本高昂 (Expensive Evaluation): 每次评估一组超参数 (即训练和交叉验证模型) 都可能非常耗时。")
    print("\n工作原理:")
    print("   a. 代理模型 (Surrogate Model): 贝叶斯优化维护一个目标函数的概率模型 (通常是高斯过程 - Gaussian Process)。")
    print("      这个模型根据已评估的超参数点来近似真实的目标函数表面，并提供不确定性估计。")
    print("   b. 采集函数 (Acquisition Function): 用于决定下一个要评估的超参数点。它平衡了以下两者：")
    print("      - **探索 (Exploration)**: 在不确定性高的区域进行采样，以发现可能的新最优区域。")
    print("      - **利用 (Exploitation)**: 在代理模型预测性能较好的区域附近进行采样，以期改进当前最佳解。")
    print("      常见的采集函数包括：期望提升 (Expected Improvement, EI)、概率提升 (Probability of Improvement, PI)、上限置信区间 (Upper Confidence Bound, UCB)。")
    print("   c. 迭代过程: 重复 a 和 b，直到达到预设的评估次数或时间限制。")
    print("\n优点:")
    print("   - 样本效率高: 相较于网格搜索或随机搜索，通常能用更少的评估次数找到更好的超参数组合，尤其适用于高维参数空间。")
    print("   - 适应性: 能够根据历史评估结果智能地指导搜索方向。")
    print("\n常用库:")
    print("   - Hyperopt: 一个流行的Python库，实现了多种优化算法，包括基于树的Parzen估计器 (TPE)，这是一种有效的贝叶斯优化变体。")
    print("     概念代码结构:")
    print("       - 定义 `objective(params)` 函数，返回损失值 (和状态)。")
    print("       - 定义 `space` (使用 `hp.choice`, `hp.uniform`, `hp.loguniform` 等)。")
    print("       - 使用 `fmin(objective, space, algo=tpe.suggest, max_evals=N)` 进行优化。")
    print("   - Optuna: 另一个非常流行的HPO框架，具有易用API、剪枝 (pruning) 功能以提前终止不佳试验、分布式优化等特性。")
    print("     概念代码结构:")
    print("       - 定义 `objective(trial)` 函数，使用 `trial.suggest_float`, `trial.suggest_int`, `trial.suggest_categorical` 等定义参数，返回性能得分。")
    print("       - `study = optuna.create_study(direction=\'maximize\')`")
    print("       - `study.optimize(objective, n_trials=N)`")
    print("选择贝叶斯优化通常是因为它在计算资源有限的情况下，寻找复杂模型良好超参数的潜力。")


def explain_automated_hpo_tools_conceptual_api() -> None:
    """打印自动化超参数优化 (AutoHPO) 工具和服务的概念性解释。"""
    print("\n--- 概念: 自动化超参数优化 (AutoHPO) 工具与服务 --- ")
    # ... (内容同之前版本，为简洁此处省略) ...
    print("除了像Hyperopt和Optuna这样的库之外，还有更广泛的自动化机器学习 (AutoML) 工具和云服务，它们将HPO作为核心功能之一:")
    print("1.  **云平台HPO服务:**")
    print("    - Google Cloud Vertex AI Vizier: 一个托管的黑箱优化服务，可用于超参数调优。用户定义搜索空间和目标，Vizier管理试验和搜索策略。")
    print("    - Amazon SageMaker Automatic Model Tuning: 与SageMaker训练作业集成，自动寻找最佳超参数。支持随机、贝叶斯和Hyperband等搜索策略。")
    print("    - Azure Machine Learning Hyperparameter Tuning: 提供多种调优算法 (网格、随机、贝叶斯)，并与Azure ML工作空间和计算资源紧密集成。")
    print("    优点: 可扩展性强，通常与云端计算和存储无缝集成，简化了大规模HPO任务的管理。")
    print("\n2.  **开源AutoML库 (通常包含HPO):**")
    print("    - Auto-Sklearn: 基于Scikit-learn，自动执行算法选择和超参数调优 (使用贝叶斯优化等)。")
    print("    - TPOT (Tree-based Pipeline Optimization Tool): 使用遗传编程来优化机器学习管道，包括特征选择、预处理、模型选择和超参数。")
    print("    - FLAML (A Fast and Lightweight AutoML Library): 由微软开发，专注于快速找到高质量模型，支持多种自定义选项。")
    print("    优点: 开源，灵活性高，可以本地运行，通常提供端到端的AutoML解决方案。")
    print("\n这些工具和服务的共同目标是:")
    print("   - **减少手动调参的负担:** 自动化繁琐的搜索过程。")
    print("   - **提高模型性能:** 系统地探索参数空间以找到更优的配置。")
    print("   - **加速开发周期:** 更快地获得高性能模型。")
    print("选择合适的工具取决于具体需求，如预算、现有基础设施、对控制级别的要求以及项目的规模。")


if __name__ == '__main__':
    print("===== 超参数优化 (HPO) API化功能演示 =====")
    hpo_seed = 777
    main_output_dir = OUTPUT_DIR # 使用模块顶部的OUTPUT_DIR

    # --- 1. 分类任务 HPO 演示 ---
    print("\n\n*** 1. HPO 分类任务 (RandomForestClassifier) ***")
    X_clf_hpo, y_clf_hpo = create_hpo_sample_data_api(
        n_samples=300, n_features=10, n_informative_num=5, n_cat_features=2, # 减小数据量加速
        task='classification', n_classes=2, random_state=hpo_seed,
        nan_percentage_num=0.05, nan_percentage_cat=0.05 # 添加一些NaN测试预处理器
    )
    
    # GridSearchCV 演示 (分类)
    rf_clf = RandomForestClassifier(random_state=hpo_seed)
    rf_clf_param_grid = {
        'model__n_estimators': [15, 30], # 进一步减少以极速演示
        'model__max_depth': [3, 4],
        'model__min_samples_split': [2, 4]
    }
    gs_clf_model_path = os.path.join(main_output_dir, "best_rf_classifier_gs.joblib")
    best_rf_clf_gs, params_rf_clf_gs, score_rf_clf_gs = perform_grid_search_cv_api(
        X_clf_hpo.copy(), y_clf_hpo.copy(), rf_clf, rf_clf_param_grid, task='classification',
        cv_folds=2, scoring='roc_auc', random_state_kfold=hpo_seed,
        save_best_model_path=gs_clf_model_path
    )
    print(f"  GridSearchCV (RF分类) 最佳估计器 Pipeline保存在: {gs_clf_model_path if os.path.exists(gs_clf_model_path) else '保存失败'}")

    # RandomizedSearchCV 演示 (XGBoost分类)
    print("\n\n*** 1.2 HPO 分类任务 (XGBClassifier) 使用 RandomizedSearch ***")
    xgb_clf = XGBClassifier(random_state=hpo_seed, use_label_encoder=False if pd.__version__.startswith("1") else None, eval_metric='logloss')
    xgb_clf_param_dist = {
        'model__n_estimators': [20, 40, 60],
        'model__learning_rate': [0.05, 0.1, 0.15],
        'model__max_depth': [2, 3],
    }
    rs_clf_model_path = os.path.join(main_output_dir, "best_xgb_classifier_rs.joblib")
    best_xgb_clf_rs, params_xgb_clf_rs, score_xgb_clf_rs = perform_randomized_search_cv_api(
        X_clf_hpo.copy(), y_clf_hpo.copy(), xgb_clf, xgb_clf_param_dist, n_iter=3, # 极速迭代
        task='classification', cv_folds=2, scoring='roc_auc', 
        random_state_search=hpo_seed, random_state_kfold=hpo_seed,
        save_best_model_path=rs_clf_model_path
    )
    print(f"  RandomizedSearchCV (XGB分类) 最佳估计器 Pipeline保存在: {rs_clf_model_path if os.path.exists(rs_clf_model_path) else '保存失败'}")


    # --- 2. 回归任务 HPO 演示 ---
    print("\n\n*** 2. HPO 回归任务 (RandomForestRegressor) ***")
    X_reg_hpo, y_reg_hpo = create_hpo_sample_data_api(
        n_samples=250, n_features=8, n_informative_num=4, n_cat_features=1, # 减小数据量
        task='regression', random_state=hpo_seed + 1,
        nan_percentage_num=0.03, nan_percentage_cat=0.03
    )

    rf_reg = RandomForestRegressor(random_state=hpo_seed + 1)
    rf_reg_param_grid = {
        'model__n_estimators': [20, 35],
        'model__max_depth': [3, 5],
        'model__min_samples_leaf': [2, 5]
    }
    gs_reg_model_path = os.path.join(main_output_dir, "best_rf_regressor_gs.joblib")
    best_rf_reg_gs, params_rf_reg_gs, score_rf_reg_gs = perform_grid_search_cv_api(
        X_reg_hpo.copy(), y_reg_hpo.copy(), rf_reg, rf_reg_param_grid, task='regression',
        cv_folds=2, scoring='r2', random_state_kfold=hpo_seed + 1,
        save_best_model_path=gs_reg_model_path
    )
    print(f"  GridSearchCV (RF回归) 最佳估计器 Pipeline保存在: {gs_reg_model_path if os.path.exists(gs_reg_model_path) else '保存失败'}")

    # --- 3. 概念解释 ---
    explain_bayesian_optimization_hpo_conceptual_api()
    explain_automated_hpo_tools_conceptual_api()

    print(f"\n\n===== 超参数优化 (HPO) API化功能演示完成 (输出保存在 '{main_output_dir}' 目录中) =====")
