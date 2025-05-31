import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import BPredict module functions
from BPredict.sklearn_examples import (
    create_sklearn_sample_data,
    preprocess_data_sklearn,
    perform_feature_engineering_sklearn,
    train_evaluate_classification_sklearn,
    train_evaluate_regression_sklearn,
    # demonstrate_pipeline_classification_sklearn, # Complex for direct call
    # demonstrate_pipeline_regression_sklearn # Complex for direct call
)
from BPredict.xgboost_examples import (
    create_xgboost_sample_data,
    train_evaluate_xgboost_native,
    train_evaluate_xgboost_sklearn
)
from BPredict.lightgbm_examples import (
    create_lightgbm_sample_data,
    train_evaluate_lightgbm_native,
    train_evaluate_lightgbm_sklearn
)
from BPredict.catboost_examples import (
    create_catboost_sample_data_api,
    train_evaluate_catboost_native_api,
    train_evaluate_catboost_sklearn_api
)
from BPredict.tensorflow_keras_examples import (
    create_tf_keras_sample_data_api,
    create_keras_preprocessor_api,
    build_keras_sequential_model_api,
    train_evaluate_keras_model_api,
    explain_keras_functional_api_conceptual
)
from BPredict.pytorch_examples import (
    create_pytorch_sample_data_api,
    TabularDataset, # Class
    PyTorchTabularModel, # Class
    train_pytorch_model_api,
    evaluate_pytorch_model_api,
    explain_pytorch_ecosystem_conceptual_api
)
from BPredict.model_evaluation_comparison import (
    create_model_comparison_sample_data_api,
    # create_model_comparison_preprocessor_api, # Preprocessor is defined within compare_multiple_models_api and perform_cross_validation_comparison_api
    compare_multiple_models_api,
    perform_cross_validation_comparison_api,
    discuss_model_selection_factors_api
)
from BPredict.hyperparameter_tuning_examples import (
    create_hpo_sample_data_api,
    # get_hpo_preprocessor_definition_api, # Usually called internally by HPO functions
    perform_grid_search_cv_api,
    perform_randomized_search_cv_api,
    explain_bayesian_optimization_hpo_conceptual_api,
    explain_automated_hpo_tools_conceptual_api
)
from BPredict.model_interpretability_examples import (
    create_interpretability_sample_data_api,
    get_interpretability_preprocessor_api,
    explain_tree_feature_importance_api,
    explain_linear_model_coefficients_api,
    explain_shap_conceptual_api,
    explain_lime_conceptual_api
)
from BPredict.model_deployment_conceptual import (
    create_sample_sklearn_pipeline_for_deployment_api,
    explain_sklearn_model_persistence_api,
    explain_keras_model_persistence_conceptual_api,
    explain_pytorch_model_persistence_conceptual_api,
    explain_flask_api_conceptual_api,
    explain_docker_containerization_conceptual_api,
    explain_cloud_deployment_options_conceptual_api,
    explain_production_considerations_conceptual_api
)
# Placeholder for future BPredict modules, e.g.:
# from BPredict.model_deployment_conceptual import ...

def demo_sklearn_core_functionalities():
    """演示 BPredict.sklearn_examples 中的核心功能。"""
    print("\n\n===== Scikit-learn 核心功能接口化演示 =====")
    main_random_seed_skl = 420 # 主演示的随机种子

    # --- (SKL.A) 数据创建与预处理 ---
    print("\n--- (SKL.A) 数据创建与预处理 ---")
    # A.1 生成分类数据
    X_clf_skl, y_clf_skl, feature_names_clf_skl = create_sklearn_sample_data(
        n_samples=300, n_features=15, n_informative=8, n_redundant=3, n_repeated=1,
        n_classes=3, task_type='classification', random_state=main_random_seed_skl
    )
    print(f"SKL 分类数据 X_clf_skl 形状: {X_clf_skl.shape}, y_clf_skl 标签类别: {np.unique(y_clf_skl)}")
    df_X_clf_skl = pd.DataFrame(X_clf_skl, columns=feature_names_clf_skl)
    df_y_clf_skl = pd.Series(y_clf_skl, name='target_clf')

    # A.2 数据预处理 (以分类数据为例)
    print("\n  A.2. 数据预处理 (分类数据)... ")
    # 引入一些 NaN 值以测试填充
    nan_indices_clf = np.random.choice(df_X_clf_skl.index, size=int(0.1 * len(df_X_clf_skl)), replace=False)
    df_X_clf_skl.iloc[nan_indices_clf, 0] = np.nan # 在第一列引入NaN
    if df_X_clf_skl.shape[1] > 1:
        df_X_clf_skl.iloc[nan_indices_clf, 1] = np.nan # 在第二列引入NaN
        
    X_clf_processed_skl, preprocessor_obj_clf_skl = preprocess_data_sklearn(
        df_X_clf_skl.copy(), 
        numerical_cols=df_X_clf_skl.columns.tolist(), # 假设所有列都是数值型，没有显式类别列
        categorical_cols=[],
        imputation_strategy_num='median', 
        scaler_type='standard'
    )
    print(f"SKL 分类数据预处理后形状: {X_clf_processed_skl.shape}")
    print(f"  预处理转换器对象类型: {type(preprocessor_obj_clf_skl)}")
    if X_clf_processed_skl.shape[0] > 0:
        print(f"  预处理后数据均值 (前2列): {X_clf_processed_skl[:, :2].mean(axis=0)}")
        print(f"  预处理后数据标准差 (前2列): {X_clf_processed_skl[:, :2].std(axis=0)}")

    # A.3 特征工程 (以分类数据为例，使用PCA)
    print("\n  A.3. 特征工程 (PCA)... ")
    X_clf_featured_skl, feature_transformer_obj_skl = perform_feature_engineering_sklearn(
        pd.DataFrame(X_clf_processed_skl, columns=feature_names_clf_skl), # 转换为DataFrame
        method='pca', 
        n_components_pca=5, # 降至5个主成分
        random_state_pca=main_random_seed_skl
    )
    print(f"SKL PCA降维后数据形状: {X_clf_featured_skl.shape}")
    if hasattr(feature_transformer_obj_skl, 'explained_variance_ratio_'):
        print(f"  PCA解释的方差比例: {feature_transformer_obj_skl.explained_variance_ratio_}")

    # --- (SKL.B) 分类模型训练与评估 ---
    print("\n\n--- (SKL.B) 分类模型训练与评估 ---")
    # 使用PCA降维后的数据进行训练
    clf_model_skl, clf_metrics_skl = train_evaluate_classification_sklearn(
        pd.DataFrame(X_clf_featured_skl), # 输入应为DataFrame
        df_y_clf_skl,
        model_type='LogisticRegression', 
        model_params={'C': 0.5, 'solver': 'liblinear', 'random_state': main_random_seed_skl},
        use_pipeline_with_preprocessing=False, # 因为我们已手动预处理和特征工程
        # preprocessor_for_pipeline=preprocessor_obj_clf_skl, # 如果 use_pipeline_with_preprocessing=True
        test_size=0.25, random_state=main_random_seed_skl
    )
    print(f"SKL 分类模型 ({type(clf_model_skl).__name__}) 训练完成。")
    print(f"  测试集评估指标: {clf_metrics_skl}")

    # --- (SKL.C) 回归模型训练与评估 ---
    print("\n\n--- (SKL.C) 回归模型训练与评估 ---")
    # C.1 生成回归数据
    X_reg_skl, y_reg_skl, feature_names_reg_skl = create_sklearn_sample_data(
        n_samples=250, n_features=10, n_informative=6, 
        task_type='regression', noise=0.2, random_state=main_random_seed_skl + 1
    )
    print(f"SKL 回归数据 X_reg_skl 形状: {X_reg_skl.shape}")
    df_X_reg_skl = pd.DataFrame(X_reg_skl, columns=feature_names_reg_skl)
    df_y_reg_skl = pd.Series(y_reg_skl, name='target_reg')

    # C.2 回归模型训练 (这里不单独做预处理和特征工程，演示带预处理的Pipeline能力)
    print("\n  C.2. 回归模型训练 (使用带预处理的Pipeline)... ")
    # 创建一个新的预处理器对象给回归任务的Pipeline
    _, reg_preprocessor_for_pipeline = preprocess_data_sklearn(
        df_X_reg_skl.copy(), # 使用原始回归数据
        numerical_cols=df_X_reg_skl.columns.tolist(), 
        categorical_cols=[],
        imputation_strategy_num='mean', 
        scaler_type='standard'
    )
    
    reg_model_skl, reg_metrics_skl = train_evaluate_regression_sklearn(
        df_X_reg_skl, # 使用原始回归特征 (DataFrame)
        df_y_reg_skl,
        model_type='Ridge', 
        model_params={'alpha': 1.0, 'random_state': main_random_seed_skl + 1},
        use_pipeline_with_preprocessing=True, # 重点：使用Pipeline
        preprocessor_for_pipeline=reg_preprocessor_for_pipeline, # 传入预处理器
        test_size=0.2, random_state=main_random_seed_skl + 1
    )
    print(f"SKL 回归模型 ({type(reg_model_skl).__name__ if not isinstance(reg_model_skl, pd.DataFrame) else 'Pipeline'}) 训练完成。") # 模型可能是Pipeline对象
    print(f"  测试集评估指标: {reg_metrics_skl}")

    print("\n\n===== Scikit-learn 核心功能接口化演示结束 ======")


def demo_xgboost_functionalities():
    """演示 BPredict.xgboost_examples 中的核心功能。"""
    print("\n\n===== XGBoost 功能接口化演示 =====")
    main_random_seed_xgb = 421 # XGBoost演示的随机种子

    # --- (XGB.A) 分类任务演示 ---
    print("\n\n--- (XGB.A) XGBoost 分类任务演示 ---")
    # A.1 生成分类数据 (包含一些NaN值和类别特征)
    X_clf_xgb, y_clf_xgb = create_xgboost_sample_data(
        n_samples=320, n_features=18, n_informative_num=7, n_cat_features=3,
        n_classes=3, task='classification', nan_percentage_num=0.1, nan_percentage_cat=0.05,
        random_state=main_random_seed_xgb
    )
    print(f"XGB 分类数据 X_clf_xgb 形状: {X_clf_xgb.shape}, y_clf_xgb 标签类别: {np.unique(y_clf_xgb)}")
    # 检查数据类型，确保有类别特征，XGBoost原生API对类别特征有特定处理方式
    print(f"  X_clf_xgb dtypes:\n{X_clf_xgb.dtypes.value_counts()}")

    # A.2 使用原生API进行分类训练与评估 (带CV和早停)
    print("\n--- XGB.A.2. 原生API 分类 (含CV和早停) ---")
    native_clf_params_xgb = {
        'eta': 0.05, 'max_depth': 5, 'objective': 'multi:softprob', 
        'num_class': 3, 'eval_metric': ['mlogloss', 'merror'], 'seed': main_random_seed_xgb,
        'tree_method': 'hist' # 使用hist以支持类别特征和NaN
    }
    # 对于原生API的类别特征，需要将DataFrame的category类型转换为整数，并设置 enable_categorical=True
    # X_clf_xgb_native_prepared = X_clf_xgb.copy()
    # for col in X_clf_xgb_native_prepared.select_dtypes(include='category').columns:
    #     X_clf_xgb_native_prepared[col] = X_clf_xgb_native_prepared[col].cat.codes

    native_clf_model_xgb, native_clf_metrics_xgb = train_evaluate_xgboost_native(
        X_clf_xgb.copy(), y_clf_xgb.copy(), task='classification', xgb_params=native_clf_params_xgb,
        num_boost_round=150, early_stopping_rounds=15, perform_cv=True, cv_folds=2,
        enable_categorical_for_native=True, # 重要：让原生API识别类别特征
        random_state=main_random_seed_xgb, plot_importance=True
    )
    if native_clf_model_xgb: # perform_cv=True时返回CV结果字典，不直接返回模型
        print(f"  XGB 原生API分类CV完成。提取的CV指标: {native_clf_metrics_xgb}")
        # 如果需要训练一个最终模型（非CV），可以再调用一次 perform_cv=False

    # A.3 使用Scikit-Learn包装器进行分类 (使用GridSearchCV)
    print("\n--- XGB.A.3. Scikit-Learn包装器分类 (含GridSearchCV) ---")
    skl_clf_param_grid_xgb = {
        'n_estimators': [80, 120], 'learning_rate': [0.03, 0.07],
        'max_depth': [4, 6], 'colsample_bytree': [0.7, 0.9]
    }
    # SKL接口对类别特征的处理方式不同，通常需要预先编码
    # train_evaluate_xgboost_sklearn 内部会尝试处理
    skl_clf_gs_obj_xgb, skl_clf_gs_metrics_xgb = train_evaluate_xgboost_sklearn(
        X_clf_xgb.copy(), y_clf_xgb.copy(), task='classification',
        use_grid_search=True, param_grid=skl_clf_param_grid_xgb, 
        cv_folds_gridsearch=2, enable_categorical_for_sklearn=True,
        random_state=main_random_seed_xgb, plot_importance=True # GridSearchCV后绘制最佳模型的重要性
    )
    if skl_clf_gs_obj_xgb and hasattr(skl_clf_gs_obj_xgb, 'best_estimator_'):
        print(f"  XGB SKL包装器分类模型 (GridSearchCV) 训练完成。最佳估计器类型: {type(skl_clf_gs_obj_xgb.best_estimator_)}")
        print(f"  测试集评估指标 (使用最佳模型): {skl_clf_gs_metrics_xgb}")

    # --- (XGB.B) 回归任务演示 ---
    print("\n\n--- (XGB.B) XGBoost 回归任务演示 ---")
    # B.1 生成回归数据
    X_reg_xgb, y_reg_xgb = create_xgboost_sample_data(
        n_samples=280, n_features=16, n_informative_num=6, n_cat_features=2,
        task='regression', nan_percentage_num=0.08, random_state=main_random_seed_xgb + 1
    )
    print(f"XGB 回归数据 X_reg_xgb 形状: {X_reg_xgb.shape}")
    print(f"  X_reg_xgb dtypes:\n{X_reg_xgb.dtypes.value_counts()}")

    # B.2 使用原生API进行回归训练与评估
    print("\n--- XGB.B.2. 原生API 回归 (直接训练) ---")
    native_reg_params_xgb = {
        'eta': 0.04, 'max_depth': 4, 'objective': 'reg:squarederror',
        'eval_metric': ['rmse', 'mae'], 'seed': main_random_seed_xgb + 1,
        'tree_method': 'hist'
    }
    native_reg_model_xgb, native_reg_metrics_xgb = train_evaluate_xgboost_native(
        X_reg_xgb.copy(), y_reg_xgb.copy(), task='regression', xgb_params=native_reg_params_xgb,
        num_boost_round=130, early_stopping_rounds=10, perform_cv=False, # 直接训练
        enable_categorical_for_native=True,
        random_state=main_random_seed_xgb + 1, plot_importance=True, plot_importance_type='gain'
    )
    if native_reg_model_xgb:
        print(f"  XGB 原生API回归模型训练完成。类型: {type(native_reg_model_xgb)}")
        print(f"  测试集评估指标: {native_reg_metrics_xgb}")

    # B.3 使用Scikit-Learn包装器进行回归 (不使用GridSearchCV)
    print("\n--- XGB.B.3. Scikit-Learn包装器回归 (无GridSearch) ---")
    skl_reg_model_params_xgb = {
        'n_estimators': 100, 'learning_rate': 0.06,
        'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8
    }
    skl_reg_direct_model_xgb, skl_reg_direct_metrics_xgb = train_evaluate_xgboost_sklearn(
        X_reg_xgb.copy(), y_reg_xgb.copy(), task='regression',
        use_grid_search=False, xgb_model_params=skl_reg_model_params_xgb,
        enable_categorical_for_sklearn=True, # 重要
        random_state=main_random_seed_xgb + 1, plot_importance=True
    )
    if skl_reg_direct_model_xgb:
        print(f"  XGB SKL包装器回归模型 (无GridSearch) 训练完成。类型: {type(skl_reg_direct_model_xgb)}")
        print(f"  测试集评估指标: {skl_reg_direct_metrics_xgb}")

    print("\n\n===== XGBoost 功能接口化演示结束 ======")


def demo_lightgbm_functionalities():
    """演示 BPredict.lightgbm_examples 中的核心功能。"""
    print("\n\n===== LightGBM 功能接口化演示 =====")
    main_random_seed_lgbm = 422 # LightGBM演示的随机种子

    # --- (LGBM.A) 分类任务演示 ---
    print("\n\n--- (LGBM.A) LightGBM 分类任务演示 ---")
    # A.1 生成分类数据
    X_clf_lgbm, y_clf_lgbm = create_lightgbm_sample_data(
        n_samples=280, n_features=14, n_informative_num=6, n_cat_features=4, # 增加类别特征
        n_classes=3, task='classification', nan_percentage_num=0.07, nan_percentage_cat=0.06,
        random_state=main_random_seed_lgbm
    )
    print(f"LGBM 分类数据 X_clf_lgbm 形状: {X_clf_lgbm.shape}, y_clf_lgbm 标签类别: {np.unique(y_clf_lgbm)}")
    print(f"  X_clf_lgbm dtypes:\n{X_clf_lgbm.dtypes.value_counts()}")

    # A.2 使用原生API进行分类训练与评估 (不进行CV，直接训练)
    print("\n--- LGBM.A.2. 原生API 分类 (直接训练) ---")
    native_clf_params_lgbm = {
        'learning_rate': 0.04, 'num_leaves': 28,
        'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'bagging_freq': 3,
        'lambda_l1': 0.05, 'lambda_l2': 0.05
    }
    native_clf_model_lgbm, native_clf_metrics_lgbm = train_evaluate_lightgbm_native(
        X_clf_lgbm.copy(), y_clf_lgbm.copy(), task='classification', lgb_params=native_clf_params_lgbm,
        num_boost_round=110, early_stopping_rounds=12, perform_cv=False,
        categorical_features='auto', # 关键：让LGBM自动处理Pandas的category类型
        random_state=main_random_seed_lgbm, plot_importance=True, plot_importance_type='gain'
    )
    if native_clf_model_lgbm:
        print(f"  LGBM 原生API分类模型训练完成。")
        print(f"  测试集评估指标: {native_clf_metrics_lgbm}")

    # A.3 使用原生API进行分类 (仅执行CV示例)
    print("\n--- LGBM.A.3. 原生API 分类 (仅CV) ---")
    native_cv_output_lgbm, native_cv_metrics_extracted_lgbm = train_evaluate_lightgbm_native(
        X_clf_lgbm.copy(), y_clf_lgbm.copy(), task='classification', perform_cv=True,
        cv_folds=2, num_boost_round=80, early_stopping_rounds=10,
        categorical_features='auto',
        cv_metrics=['auc_mu', 'multi_logloss'], # 针对多分类
        random_state=main_random_seed_lgbm
    )
    if native_cv_output_lgbm is not None:
        print(f"  LGBM 原生API CV 完成。提取的CV指标: {native_cv_metrics_extracted_lgbm}")


    # A.4 使用Scikit-Learn包装器进行分类 (使用GridSearchCV)
    print("\n--- LGBM.A.4. Scikit-Learn包装器分类 (含GridSearchCV) ---")
    skl_clf_param_grid_lgbm = {
        'n_estimators': [50, 80], 'learning_rate': [0.05, 0.1],
        'num_leaves': [20, 30], 'colsample_bytree': [0.7, 0.8],
    }
    # 对于SKL接口，通常推荐先对类别特征进行整数编码，但其内部也能尝试处理
    # 此处依赖 train_evaluate_lightgbm_sklearn 内部的 encode_categoricals=True (默认)
    skl_clf_gs_obj_lgbm, skl_clf_gs_metrics_lgbm = train_evaluate_lightgbm_sklearn(
        X_clf_lgbm.copy(), y_clf_lgbm.copy(), task='classification',
        use_grid_search=True, param_grid=skl_clf_param_grid_lgbm,
        cv_folds_gridsearch=2, random_state=main_random_seed_lgbm,
        plot_importance=True # GridSearchCV后绘制最佳模型的重要性
    )
    if skl_clf_gs_obj_lgbm and hasattr(skl_clf_gs_obj_lgbm, 'best_estimator_'):
        print(f"  LGBM SKL包装器分类模型 (GridSearchCV) 训练完成。")
        print(f"  测试集评估指标 (使用最佳模型): {skl_clf_gs_metrics_lgbm}")


    # --- (LGBM.B) 回归任务演示 ---
    print("\n\n--- (LGBM.B) LightGBM 回归任务演示 ---")
    # B.1 生成回归数据
    X_reg_lgbm, y_reg_lgbm = create_lightgbm_sample_data(
        n_samples=260, n_features=13, n_informative_num=7, n_cat_features=3,
        task='regression', nan_percentage_num=0.04, nan_percentage_cat=0.04,
        random_state=main_random_seed_lgbm + 1
    )
    print(f"LGBM 回归数据 X_reg_lgbm 形状: {X_reg_lgbm.shape}")
    print(f"  X_reg_lgbm dtypes:\n{X_reg_lgbm.dtypes.value_counts()}")


    # B.2 使用原生API进行回归训练与评估
    print("\n--- LGBM.B.2. 原生API 回归 (直接训练) ---")
    native_reg_params_lgbm = {
        'learning_rate': 0.05, 'num_leaves': 30, 'max_depth': 7,
        'lambda_l1': 0.1, 'lambda_l2': 0.1, 'min_child_samples': 18
    }
    native_reg_model_lgbm, native_reg_metrics_lgbm = train_evaluate_lightgbm_native(
        X_reg_lgbm.copy(), y_reg_lgbm.copy(), task='regression', lgb_params=native_reg_params_lgbm,
        num_boost_round=100, early_stopping_rounds=10, perform_cv=False,
        categorical_features='auto',
        random_state=main_random_seed_lgbm + 1, plot_importance=True
    )
    if native_reg_model_lgbm:
        print(f"  LGBM 原生API回归模型训练完成。")
        print(f"  测试集评估指标: {native_reg_metrics_lgbm}")

    # B.3 使用Scikit-Learn包装器进行回归 (不使用GridSearchCV，直接训练)
    print("\n--- LGBM.B.3. Scikit-Learn包装器回归 (无GridSearch) ---")
    skl_reg_model_params_lgbm = {
        'n_estimators': 70, 'learning_rate': 0.07,
        'num_leaves': 28, 'subsample': 0.75
    }
    skl_reg_direct_model_lgbm, skl_reg_direct_metrics_lgbm = train_evaluate_lightgbm_sklearn(
        X_reg_lgbm.copy(), y_reg_lgbm.copy(), task='regression',
        use_grid_search=False, lgbm_model_params=skl_reg_model_params_lgbm,
        random_state=main_random_seed_lgbm + 1, plot_importance=True, plot_importance_type='split'
    )
    if skl_reg_direct_model_lgbm:
        print(f"  LGBM SKL包装器回归模型 (无GridSearch) 训练完成。")
        print(f"  测试集评估指标: {skl_reg_direct_metrics_lgbm}")

    print("\n\n===== LightGBM 功能接口化演示结束 ======")


def demo_catboost_functionalities():
    """演示 BPredict.catboost_examples 中的核心功能。"""
    print("\n\n===== CatBoost 功能接口化演示 =====")
    main_random_seed_cb = 423 # CatBoost演示的随机种子

    # --- (CB.A) 分类任务演示 ---
    print("\n\n--- (CB.A) CatBoost 分类任务演示 ---")
    # A.1 生成分类数据
    X_clf_cb, y_clf_cb = create_catboost_sample_data_api(
        n_samples=260, n_features=15, n_informative_num=7, n_cat_features=4,
        n_classes=3, task='classification', nan_percentage_num=0.06, nan_percentage_cat=0.05,
        random_state=main_random_seed_cb
    )
    print(f"CatBoost 分类数据 X_clf_cb 形状: {X_clf_cb.shape}, y_clf_cb 标签类别: {np.unique(y_clf_cb)}")
    print(f"  X_clf_cb dtypes:\n{X_clf_cb.dtypes.value_counts()}")

    # A.2 使用原生API进行分类训练与评估 (直接训练)
    print("\n--- CB.A.2. 原生API 分类 (直接训练) ---")
    native_clf_params_cb = {
        'learning_rate': 0.06, 'depth': 5, 'l2_leaf_reg': 3,
        # 'loss_function' and 'eval_metric' will be auto-set by the API
    }
    native_clf_model_cb, native_clf_metrics_cb = train_evaluate_catboost_native_api(
        X_clf_cb.copy(), y_clf_cb.copy(), task='classification', catboost_params=native_clf_params_cb,
        num_boost_round=90, early_stopping_rounds=9, perform_cv=False,
        random_state=main_random_seed_cb, plot_importance=True, plot_importance_type='FeatureImportance'
    )
    if native_clf_model_cb:
        print(f"  CatBoost 原生API分类模型 ({type(native_clf_model_cb).__name__}) 训练完成。")
        print(f"  测试集评估指标: {native_clf_metrics_cb}")

    # A.3 使用原生API进行分类 (仅执行CV示例)
    print("\n--- CB.A.3. 原生API 分类 (仅CV) ---")
    native_cv_clf_params_cb = {'learning_rate': 0.08, 'depth': 4}
    cv_clf_results_df_cb, cv_clf_metrics_cb = train_evaluate_catboost_native_api(
        X_clf_cb.copy(), y_clf_cb.copy(), task='classification', catboost_params=native_cv_clf_params_cb,
        num_boost_round=60, early_stopping_rounds=6, perform_cv=True, cv_folds=2,
        cv_metrics=['MultiClass', 'AUC'], # Specify for multi-class
        random_state=main_random_seed_cb
    )
    if cv_clf_results_df_cb is not None:
        print(f"  CatBoost 原生API CV 完成。提取的CV指标: {cv_clf_metrics_cb}")

    # A.4 使用Scikit-Learn包装器进行分类 (使用GridSearchCV)
    print("\n--- CB.A.4. Scikit-Learn包装器分类 (含GridSearchCV) ---")
    skl_clf_param_grid_cb = {
        'n_estimators': [50, 70], 'learning_rate': [0.07, 0.1],
        'depth': [3, 5]
    }
    # Pass early_stopping_rounds via fit_params for GridSearchCV
    skl_clf_fit_params_cb = {'early_stopping_rounds': 8} 
    skl_clf_gs_obj_cb, skl_clf_gs_metrics_cb = train_evaluate_catboost_sklearn_api(
        X_clf_cb.copy(), y_clf_cb.copy(), task='classification',
        use_grid_search=True, param_grid=skl_clf_param_grid_cb,
        cv_folds_gridsearch=2, gridsearch_scoring='accuracy', # or 'roc_auc_ovr'
        random_state=main_random_seed_cb, plot_importance_best_model=True,
        fit_params_gridsearch=skl_clf_fit_params_cb
    )
    if skl_clf_gs_obj_cb and hasattr(skl_clf_gs_obj_cb, 'best_estimator_'):
        print(f"  CatBoost SKL包装器分类模型 (GridSearchCV) 训练完成。")
        print(f"  测试集评估指标 (使用最佳模型): {skl_clf_gs_metrics_cb}")


    # --- (CB.B) 回归任务演示 ---
    print("\n\n--- (CB.B) CatBoost 回归任务演示 ---")
    # B.1 生成回归数据
    X_reg_cb, y_reg_cb = create_catboost_sample_data_api(
        n_samples=240, n_features=13, n_informative_num=6, n_cat_features=3,
        task='regression', nan_percentage_num=0.05, nan_percentage_cat=0.06,
        random_state=main_random_seed_cb + 1
    )
    print(f"CatBoost 回归数据 X_reg_cb 形状: {X_reg_cb.shape}")
    print(f"  X_reg_cb dtypes:\n{X_reg_cb.dtypes.value_counts()}")

    # B.2 使用原生API进行回归训练与评估
    print("\n--- CB.B.2. 原生API 回归 (直接训练) ---")
    native_reg_params_cb = {
        'learning_rate': 0.07, 'depth': 4, 'loss_function': 'RMSE', 'eval_metric':'R2'
    }
    native_reg_model_cb, native_reg_metrics_cb = train_evaluate_catboost_native_api(
        X_reg_cb.copy(), y_reg_cb.copy(), task='regression', catboost_params=native_reg_params_cb,
        num_boost_round=80, early_stopping_rounds=8, perform_cv=False,
        random_state=main_random_seed_cb + 1, plot_importance=True, plot_importance_type='ShapValues'
    )
    if native_reg_model_cb:
        print(f"  CatBoost 原生API回归模型训练完成。")
        print(f"  测试集评估指标: {native_reg_metrics_cb}")

    # B.3 使用Scikit-Learn包装器进行回归 (不使用GridSearchCV，直接训练)
    print("\n--- CB.B.3. Scikit-Learn包装器回归 (无GridSearch) ---")
    skl_reg_direct_params_cb = {
        'n_estimators': 70, 'learning_rate': 0.08, 'depth': 5,
        'early_stopping_rounds': 7 # SKL wrapper can take early_stopping_rounds in constructor
    }
    # For direct training, eval_set for early stopping is handled if fit_params are passed correctly.
    # The API train_evaluate_catboost_sklearn_api manages this.
    skl_reg_direct_model_cb, skl_reg_direct_metrics_cb = train_evaluate_catboost_sklearn_api(
        X_reg_cb.copy(), y_reg_cb.copy(), task='regression',
        catboost_model_params=skl_reg_direct_params_cb,
        use_grid_search=False,
        random_state=main_random_seed_cb + 1, plot_importance_best_model=True
    )
    if skl_reg_direct_model_cb:
        print(f"  CatBoost SKL包装器回归模型 (无GridSearch) 训练完成。")
        print(f"  测试集评估指标: {skl_reg_direct_metrics_cb}")

    print("\n\n===== CatBoost 功能接口化演示结束 ======")


def demo_tensorflow_keras_functionalities():
    """演示 BPredict.tensorflow_keras_examples 中的核心功能。"""
    print("\n\n===== TensorFlow/Keras 功能接口化演示 =====")
    main_tf_seed = 424 # TensorFlow/Keras 演示的随机种子
    # TensorFlow/Keras API内部会处理种子设置，这里主要用于数据生成

    # --- (TF.A) 分类任务演示 ---
    print("\n\n--- (TF.A) Keras 分类任务演示 ---")
    # A.1 生成分类数据
    X_df_clf_tf, y_s_clf_tf = create_tf_keras_sample_data_api(
        n_samples=550, n_features=13, n_informative_num=7, n_cat_features=3,
        n_classes=3, task='classification', nan_percentage_num=0.06, nan_percentage_cat=0.06,
        random_state=main_tf_seed
    )
    print(f"Keras 分类数据 X_df_clf_tf 形状: {X_df_clf_tf.shape}, y_s_clf_tf 标签类别: {np.unique(y_s_clf_tf)}")

    # A.2 准备目标变量 (如果多分类则进行One-Hot编码)
    num_unique_classes_clf_tf = y_s_clf_tf.nunique()
    if num_unique_classes_clf_tf > 2:
        try:
            # 需要导入 keras for to_categorical
            from tensorflow import keras as tf_keras_internal
            y_clf_tf_prepared = tf_keras_internal.utils.to_categorical(y_s_clf_tf, num_classes=num_unique_classes_clf_tf)
            print(f"  目标变量已进行One-Hot编码 (多分类)，形状: {y_clf_tf_prepared.shape}")
        except Exception as e_cat:
            print(f"Keras to_categorical 错误: {e_cat}. 将跳过此部分分类演示。")
            y_clf_tf_prepared = None # 标记错误
    else:
        y_clf_tf_prepared = y_s_clf_tf.copy()
        print(f"  目标变量为二分类，无需One-Hot编码。形状: {y_clf_tf_prepared.shape}")

    if y_clf_tf_prepared is not None:
        # A.3 数据分割与预处理
        X_train_clf_df_tf, X_test_clf_df_tf, y_train_clf_tf, y_test_clf_tf = train_test_split(
            X_df_clf_tf, y_clf_tf_prepared, test_size=0.2, random_state=main_tf_seed,
            stratify=y_s_clf_tf if num_unique_classes_clf_tf > 1 else None
        )
        preprocessor_clf_tf = create_keras_preprocessor_api(X_train_clf_df_tf)
        X_train_clf_processed_tf = preprocessor_clf_tf.transform(X_train_clf_df_tf)
        X_test_clf_processed_tf = preprocessor_clf_tf.transform(X_test_clf_df_tf)
        print(f"  Keras分类数据预处理后: 训练集 {X_train_clf_processed_tf.shape}, 测试集 {X_test_clf_processed_tf.shape}")

        # A.4 构建、训练和评估分类模型
        clf_model_layers_tf = [
            {'units': 100, 'activation': 'relu', 'dropout': 0.28, 'batch_norm': True},
            {'units': 50, 'activation': 'relu', 'dropout': 0.18, 'batch_norm': True}
        ]
        clf_optimizer_tf = {'name': 'adam', 'learning_rate': 0.0012}
        keras_clf_model_tf = build_keras_sequential_model_api(
            input_dim=X_train_clf_processed_tf.shape[1],
            task='classification',
            num_classes=num_unique_classes_clf_tf,
            layers_config=clf_model_layers_tf,
            optimizer_config=clf_optimizer_tf
        )
        clf_history_tf, clf_metrics_tf = train_evaluate_keras_model_api(
            model=keras_clf_model_tf, 
            X_train_processed=X_train_clf_processed_tf, y_train_prepared=y_train_clf_tf,
            X_test_processed=X_test_clf_processed_tf, y_test_prepared=y_test_clf_tf,
            task='classification', epochs=35, batch_size=64, early_stopping_patience=6,
            num_classes_for_eval=num_unique_classes_clf_tf
        )
        print(f"  Keras分类模型训练历史: {clf_history_tf.history.keys()}")
        print(f"  Keras分类模型评估指标: {clf_metrics_tf}")
    else:
        print("  由于目标变量准备失败，跳过Keras分类任务演示。")

    # --- (TF.B) 回归任务演示 ---
    print("\n\n--- (TF.B) Keras 回归任务演示 ---")
    # B.1 生成回归数据
    X_df_reg_tf, y_s_reg_tf = create_tf_keras_sample_data_api(
        n_samples=500, n_features=11, n_informative_num=6, n_cat_features=2,
        task='regression', nan_percentage_num=0.05, nan_percentage_cat=0.04,
        random_state=main_tf_seed + 1
    )
    print(f"Keras 回归数据 X_df_reg_tf 形状: {X_df_reg_tf.shape}")

    # B.2 数据分割与预处理 (回归目标变量通常不需要特殊准备)
    X_train_reg_df_tf, X_test_reg_df_tf, y_train_reg_tf, y_test_reg_tf = train_test_split(
        X_df_reg_tf, y_s_reg_tf, test_size=0.2, random_state=main_tf_seed + 1
    )
    preprocessor_reg_tf = create_keras_preprocessor_api(X_train_reg_df_tf)
    X_train_reg_processed_tf = preprocessor_reg_tf.transform(X_train_reg_df_tf)
    X_test_reg_processed_tf = preprocessor_reg_tf.transform(X_test_reg_df_tf)
    print(f"  Keras回归数据预处理后: 训练集 {X_train_reg_processed_tf.shape}, 测试集 {X_test_reg_processed_tf.shape}")

    # B.3 构建、训练和评估回归模型
    reg_model_layers_tf = [
        {'units': 88, 'activation': 'relu', 'dropout': 0.22, 'batch_norm': False},
        {'units': 44, 'activation': 'relu', 'dropout': 0.12, 'batch_norm': False}
    ]
    reg_optimizer_tf = {'name': 'rmsprop', 'learning_rate': 0.001}
    keras_reg_model_tf = build_keras_sequential_model_api(
        input_dim=X_train_reg_processed_tf.shape[1],
        task='regression',
        layers_config=reg_model_layers_tf,
        optimizer_config=reg_optimizer_tf
    )
    reg_history_tf, reg_metrics_tf = train_evaluate_keras_model_api(
        model=keras_reg_model_tf,
        X_train_processed=X_train_reg_processed_tf, y_train_prepared=y_train_reg_tf.values,
        X_test_processed=X_test_reg_processed_tf, y_test_prepared=y_test_reg_tf.values,
        task='regression', epochs=40, batch_size=32, early_stopping_patience=7
    )
    print(f"  Keras回归模型训练历史: {reg_history_tf.history.keys()}")
    print(f"  Keras回归模型评估指标: {reg_metrics_tf}")

    # --- (TF.C) Functional API 概念解释 ---
    explain_keras_functional_api_conceptual()

    print("\n\n===== TensorFlow/Keras 功能接口化演示结束 ======")


def demo_pytorch_functionalities():
    """演示 BPredict.pytorch_examples 中的核心功能。"""
    print("\n\n===== PyTorch 功能接口化演示 =====")
    main_pytorch_seed = 425
    # PyTorch API内部会处理种子设置

    # --- (PT.A) 分类任务演示 ---
    print("\n\n--- (PT.A) PyTorch 分类任务演示 ---")
    # A.1 生成分类数据
    X_df_clf_pt, y_s_clf_pt = create_pytorch_sample_data_api(
        n_samples=480, n_features=14, n_informative_num=7, n_cat_features=3,
        n_classes=3, task='classification', nan_percentage_num=0.05, nan_percentage_cat=0.05,
        random_state=main_pytorch_seed
    )
    print(f"PyTorch 分类数据 X_df_clf_pt 形状: {X_df_clf_pt.shape}, y_s_clf_pt 标签类别: {np.unique(y_s_clf_pt)}")

    # A.2 数据预处理: 插补, 类别编码, 标准化 (与模块内__main__类似)
    numeric_cols_clf_pt = X_df_clf_pt.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_clf_pt = X_df_clf_pt.select_dtypes(include='object').columns.tolist()
    
    X_df_clf_pt_imputed = X_df_clf_pt.copy()
    for col in numeric_cols_clf_pt:
        X_df_clf_pt_imputed[col].fillna(X_df_clf_pt_imputed[col].median(), inplace=True)
    for col in categorical_cols_clf_pt:
        X_df_clf_pt_imputed[col].fillna(X_df_clf_pt_imputed[col].mode()[0] if not X_df_clf_pt_imputed[col].mode().empty else 'Unknown', inplace=True)

    # 为了简单，这里也用LabelEncoder，但PyTorch中更常见的是用Embedding层处理类别特征
    # 这意味着模型输入将是编码后的整数，模型内部创建Embedding
    label_encoders_clf_pt = {}
    X_df_clf_pt_cat_encoded = X_df_clf_pt_imputed.copy()
    for col in categorical_cols_clf_pt:
        le = LabelEncoder()
        X_df_clf_pt_cat_encoded[col] = le.fit_transform(X_df_clf_pt_cat_encoded[col])
        label_encoders_clf_pt[col] = le
    
    # 标准化所有特征 (现在它们都是数值型)
    scaler_clf_pt = StandardScaler()
    X_clf_pt_processed = scaler_clf_pt.fit_transform(X_df_clf_pt_cat_encoded)
    y_clf_pt_prepared = y_s_clf_pt.values # Numpy array of labels
    num_classes_clf_pt = len(np.unique(y_clf_pt_prepared))

    X_train_clf_pt, X_test_clf_pt, y_train_clf_pt, y_test_clf_pt = train_test_split(
        X_clf_pt_processed, y_clf_pt_prepared, test_size=0.2, random_state=main_pytorch_seed,
        stratify=y_clf_pt_prepared if num_classes_clf_pt > 1 else None
    )

    train_ds_clf = TabularDataset(X_train_clf_pt, y_train_clf_pt, task='classification')
    val_ds_clf = TabularDataset(X_test_clf_pt, y_test_clf_pt, task='classification') # 使用测试集作为验证集
    test_ds_clf = TabularDataset(X_test_clf_pt, y_test_clf_pt, task='classification')

    train_loader_clf_pt = DataLoader(train_ds_clf, batch_size=64, shuffle=True)
    val_loader_clf_pt = DataLoader(val_ds_clf, batch_size=64, shuffle=False)
    test_loader_clf_pt = DataLoader(test_ds_clf, batch_size=64, shuffle=False)
    print(f"  PyTorch 分类数据加载器创建完成。训练集大小: {len(train_loader_clf_pt.dataset)}")

    # A.3 构建、训练和评估分类模型
    input_dim_clf_pt = X_train_clf_pt.shape[1]
    # 对于CrossEntropyLoss, output_dim是类别数。对于BCEWithLogitsLoss (二分类), output_dim是1。
    output_dim_clf_pt = num_classes_clf_pt if num_classes_clf_pt > 2 else (1 if num_classes_clf_pt == 2 else num_classes_clf_pt)
    
    clf_layers_config_pt = [{'units': 100, 'dropout': 0.3}, {'units': 50, 'dropout': 0.2}]
    model_clf_pt = PyTorchTabularModel(input_dim_clf_pt, output_dim_clf_pt, layers_config=clf_layers_config_pt)
    
    if num_classes_clf_pt == 2:
        criterion_clf_pt = nn.BCEWithLogitsLoss() # 模型输出一个logit
    else:
        criterion_clf_pt = nn.CrossEntropyLoss() # 模型输出 num_classes个logits
    optimizer_clf_pt = optim.AdamW(model_clf_pt.parameters(), lr=0.002)

    trained_model_clf_pt, history_clf_pt = train_pytorch_model_api(
        model_clf_pt, train_loader_clf_pt, val_loader_clf_pt, criterion_clf_pt, optimizer_clf_pt,
        epochs=30, task='classification', num_classes=num_classes_clf_pt
    )
    eval_metrics_clf_pt = evaluate_pytorch_model_api(
        trained_model_clf_pt, test_loader_clf_pt, criterion_clf_pt, task='classification', num_classes=num_classes_clf_pt
    )
    # 保存模型和指标
    torch.save(trained_model_clf_pt.state_dict(), os.path.join("pytorch_outputs", 'pytorch_clf_model.pth'))
    with open(os.path.join("pytorch_outputs", 'pytorch_clf_metrics.json'), 'w') as f:
        json.dump(eval_metrics_clf_pt, f, indent=4)
    print(f"  PyTorch分类模型及指标已保存。训练历史条目数: {len(history_clf_pt)}")


    # --- (PT.B) 回归任务演示 ---
    print("\n\n--- (PT.B) PyTorch 回归任务演示 ---")
    # B.1 生成回归数据
    X_df_reg_pt, y_s_reg_pt = create_pytorch_sample_data_api(
        n_samples=420, n_features=11, n_informative_num=6, n_cat_features=2,
        task='regression', nan_percentage_num=0.04, nan_percentage_cat=0.04,
        random_state=main_pytorch_seed + 1
    )

    # B.2 数据预处理
    numeric_cols_reg_pt = X_df_reg_pt.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_reg_pt = X_df_reg_pt.select_dtypes(include='object').columns.tolist()

    X_df_reg_pt_imputed = X_df_reg_pt.copy()
    for col in numeric_cols_reg_pt: X_df_reg_pt_imputed[col].fillna(X_df_reg_pt_imputed[col].median(), inplace=True)
    for col in categorical_cols_reg_pt: X_df_reg_pt_imputed[col].fillna(X_df_reg_pt_imputed[col].mode()[0] if not X_df_reg_pt_imputed[col].mode().empty else 'Unknown', inplace=True)
    
    label_encoders_reg_pt = {}
    X_df_reg_pt_cat_encoded = X_df_reg_pt_imputed.copy()
    for col in categorical_cols_reg_pt: 
        le_reg = LabelEncoder()
        X_df_reg_pt_cat_encoded[col] = le_reg.fit_transform(X_df_reg_pt_cat_encoded[col])
        label_encoders_reg_pt[col] = le_reg

    scaler_reg_pt = StandardScaler()
    X_reg_pt_processed = scaler_reg_pt.fit_transform(X_df_reg_pt_cat_encoded)
    y_reg_pt_prepared = y_s_reg_pt.values

    X_train_reg_pt, X_test_reg_pt, y_train_reg_pt, y_test_reg_pt = train_test_split(
        X_reg_pt_processed, y_reg_pt_prepared, test_size=0.2, random_state=main_pytorch_seed + 1
    )

    train_ds_reg = TabularDataset(X_train_reg_pt, y_train_reg_pt, task='regression')
    val_ds_reg = TabularDataset(X_test_reg_pt, y_test_reg_pt, task='regression')
    test_ds_reg = TabularDataset(X_test_reg_pt, y_test_reg_pt, task='regression')

    train_loader_reg_pt = DataLoader(train_ds_reg, batch_size=32, shuffle=True)
    val_loader_reg_pt = DataLoader(val_ds_reg, batch_size=32, shuffle=False)
    test_loader_reg_pt = DataLoader(test_ds_reg, batch_size=32, shuffle=False)
    print(f"  PyTorch 回归数据加载器创建完成。训练集大小: {len(train_loader_reg_pt.dataset)}")

    # B.3 构建、训练和评估回归模型
    input_dim_reg_pt = X_train_reg_pt.shape[1]
    output_dim_reg_pt = 1
    reg_layers_config_pt = [{'units': 90, 'dropout': 0.2}, {'units': 45, 'dropout': 0.1}]
    model_reg_pt = PyTorchTabularModel(input_dim_reg_pt, output_dim_reg_pt, layers_config=reg_layers_config_pt)
    criterion_reg_pt = nn.MSELoss()
    optimizer_reg_pt = optim.Adam(model_reg_pt.parameters(), lr=0.001)

    trained_model_reg_pt, history_reg_pt = train_pytorch_model_api(
        model_reg_pt, train_loader_reg_pt, val_loader_reg_pt, criterion_reg_pt, optimizer_reg_pt,
        epochs=35, task='regression'
    )
    eval_metrics_reg_pt = evaluate_pytorch_model_api(
        trained_model_reg_pt, test_loader_reg_pt, criterion_reg_pt, task='regression'
    )
    torch.save(trained_model_reg_pt.state_dict(), os.path.join("pytorch_outputs", 'pytorch_reg_model.pth'))
    with open(os.path.join("pytorch_outputs", 'pytorch_reg_metrics.json'), 'w') as f:
        json.dump(eval_metrics_reg_pt, f, indent=4)
    print(f"  PyTorch回归模型及指标已保存。训练历史条目数: {len(history_reg_pt)}")

    # --- (PT.C) PyTorch生态概念解释 ---
    explain_pytorch_ecosystem_conceptual_api()

    print("\n\n===== PyTorch 功能接口化演示结束 ======")


def demo_model_evaluation_comparison_functionalities():
    """演示 BPredict.model_evaluation_comparison 中的核心功能。"""
    print("\n\n===== 模型评估与比较功能接口化演示 =====")
    main_eval_seed = 426

    # 定义用于比较的模型 (Scikit-learn API兼容)
    # 这些模型将在 compare_multiple_models_api 和 perform_cross_validation_comparison_api 中使用
    # 注意: XGBoost的参数如 use_label_encoder 和 eval_metric 在SKL接口中通常不是直接在构造函数中设置，
    # 而是在 .fit() 中，或者某些版本会自动处理。API内部实现会处理这些。

    classification_model_configs = [
        ('逻辑回归 (评估)', LogisticRegression(solver='liblinear', random_state=main_eval_seed, max_iter=200)),
        ('随机森林分类器 (评估)', RandomForestClassifier(n_estimators=50, random_state=main_eval_seed, max_depth=5)),
        ('XGBoost分类器 (评估)', XGBClassifier(n_estimators=50, random_state=main_eval_seed, objective='binary:logistic' if 2 == 2 else 'multi:softprob')) # Assuming binary for demo
    ]

    regression_model_configs = [
        ('线性回归 (评估)', LinearRegression()),
        ('随机森林回归器 (评估)', RandomForestRegressor(n_estimators=50, random_state=main_eval_seed, max_depth=5)),
        ('XGBoost回归器 (评估)', XGBRegressor(n_estimators=50, random_state=main_eval_seed, objective='reg:squarederror'))
    ]

    # --- (MEC.A) 分类模型比较 --- 
    print("\n\n--- (MEC.A) 分类模型比较 --- ")
    # A.1 生成分类数据
    X_clf_mc, y_clf_mc = create_model_comparison_sample_data_api(
        n_samples=550, n_features=16, n_informative_num=8, n_cat_features=3,
        task='classification', n_classes=2, # 确保二分类以测试ROC图
        random_state=main_eval_seed
    )
    print(f"模型比较分类数据 X_clf_mc: {X_clf_mc.shape}, y_clf_mc: {y_clf_mc.shape}")

    # A.2 使用Train/Test Split比较分类模型
    clf_results_df_tts, clf_roc_path_tts = compare_multiple_models_api(
        X_clf_mc, y_clf_mc, classification_model_configs, 
        task='classification', random_state=main_eval_seed, plot_roc_for_binary_clf=True
    )
    print("  Train/Test Split 分类模型比较完成。结果保存在DataFrame和可能的ROC图中。")
    if clf_roc_path_tts:
        print(f"    ROC曲线图 (TTS): {clf_roc_path_tts}")

    # A.3 使用交叉验证比较分类模型
    clf_results_df_cv = perform_cross_validation_comparison_api(
        X_clf_mc, y_clf_mc, classification_model_configs, 
        task='classification', cv_folds=3, random_state=main_eval_seed, scoring_metric='roc_auc'
    )
    print("  交叉验证分类模型比较完成。结果保存在DataFrame中。")

    # --- (MEC.B) 回归模型比较 --- 
    print("\n\n--- (MEC.B) 回归模型比较 --- ")
    # B.1 生成回归数据
    X_reg_mc, y_reg_mc = create_model_comparison_sample_data_api(
        n_samples=500, n_features=14, n_informative_num=6, n_cat_features=3,
        task='regression', random_state=main_eval_seed + 1
    )
    print(f"模型比较回归数据 X_reg_mc: {X_reg_mc.shape}, y_reg_mc: {y_reg_mc.shape}")

    # B.2 使用Train/Test Split比较回归模型
    reg_results_df_tts, _ = compare_multiple_models_api(
        X_reg_mc, y_reg_mc, regression_model_configs, 
        task='regression', random_state=main_eval_seed + 1
    )
    print("  Train/Test Split 回归模型比较完成。结果保存在DataFrame中。")

    # B.3 使用交叉验证比较回归模型
    reg_results_df_cv = perform_cross_validation_comparison_api(
        X_reg_mc, y_reg_mc, regression_model_configs, 
        task='regression', cv_folds=3, random_state=main_eval_seed + 1, scoring_metric='r2'
    )
    print("  交叉验证回归模型比较完成。结果保存在DataFrame中。")

    # --- (MEC.C) 讨论模型选择因素 ---
    discuss_model_selection_factors_api()

    print("\n\n===== 模型评估与比较功能接口化演示结束 ======")


def demo_hyperparameter_tuning_functionalities():
    """演示 BPredict.hyperparameter_tuning_examples 中的核心功能。"""
    print("\n\n===== 超参数优化 (HPO) 功能接口化演示 =====")
    hpo_demo_seed = 427
    # Define models from sklearn.ensemble for HPO demos
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from xgboost import XGBClassifier # XGBRegressor can be added if needed

    # --- (HPO.A) 分类任务HPO --- 
    print("\n\n--- (HPO.A) 分类任务超参数优化 --- ")
    # A.1 生成分类数据
    X_clf_hpo, y_clf_hpo = create_hpo_sample_data_api(
        n_samples=350, n_features=10, n_informative_num=6, n_cat_features=2, # Smaller for faster demo
        task='classification', n_classes=2, random_state=hpo_demo_seed,
        nan_percentage_num=0.02, nan_percentage_cat=0.02 # Minimal NaNs
    )
    print(f"HPO 分类数据 X_clf_hpo: {X_clf_hpo.shape}, y_clf_hpo: {y_clf_hpo.shape}")

    # A.2 使用GridSearchCV对RandomForestClassifier进行调优
    print("\n  A.2 GridSearchCV for RandomForestClassifier...")
    rf_classifier = RandomForestClassifier(random_state=hpo_demo_seed)
    rf_clf_param_grid = {
        'model__n_estimators': [20, 35], # Reduced for demo speed
        'model__max_depth': [3, 4],
        'model__min_samples_split': [2, 4]
    }
    gs_rf_clf_path = os.path.join("hyperparameter_tuning_outputs", "gs_rf_clf_best_model.joblib")
    best_rf_clf_gs, params_rf_clf_gs, score_rf_clf_gs = perform_grid_search_cv_api(
        X_clf_hpo.copy(), y_clf_hpo.copy(), rf_classifier, rf_clf_param_grid, 
        task='classification', cv_folds=2, scoring='roc_auc', 
        random_state_kfold=hpo_demo_seed, save_best_model_path=gs_rf_clf_path
    )
    print(f"    GridSearchCV (RandomForest 分类) 最佳参数: {params_rf_clf_gs}, 最佳CV得分: {score_rf_clf_gs:.4f}")

    # A.3 使用RandomizedSearchCV对XGBClassifier进行调优
    print("\n  A.3 RandomizedSearchCV for XGBClassifier...")
    # Ensure correct handling of use_label_encoder based on pandas version for XGBoost
    use_label_enc = False if pd.__version__.startswith("1") else None
    xgb_classifier = XGBClassifier(random_state=hpo_demo_seed, eval_metric='logloss', use_label_encoder=use_label_enc)
    xgb_clf_param_dist = {
        'model__n_estimators': [25, 40, 55],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [2, 3]
    }
    rs_xgb_clf_path = os.path.join("hyperparameter_tuning_outputs", "rs_xgb_clf_best_model.joblib")
    best_xgb_clf_rs, params_xgb_clf_rs, score_xgb_clf_rs = perform_randomized_search_cv_api(
        X_clf_hpo.copy(), y_clf_hpo.copy(), xgb_classifier, xgb_clf_param_dist, 
        n_iter=3, task='classification', cv_folds=2, scoring='roc_auc',
        random_state_search=hpo_demo_seed, random_state_kfold=hpo_demo_seed,
        save_best_model_path=rs_xgb_clf_path
    )
    print(f"    RandomizedSearchCV (XGBoost 分类) 最佳参数: {params_xgb_clf_rs}, 最佳CV得分: {score_xgb_clf_rs:.4f}")

    # --- (HPO.B) 回归任务HPO --- 
    print("\n\n--- (HPO.B) 回归任务超参数优化 --- ")
    # B.1 生成回归数据
    X_reg_hpo, y_reg_hpo = create_hpo_sample_data_api(
        n_samples=300, n_features=8, n_informative_num=5, n_cat_features=1,
        task='regression', random_state=hpo_demo_seed + 1,
        nan_percentage_num=0.01, nan_percentage_cat=0.01
    )
    print(f"HPO 回归数据 X_reg_hpo: {X_reg_hpo.shape}, y_reg_hpo: {y_reg_hpo.shape}")

    # B.2 使用GridSearchCV对RandomForestRegressor进行调优
    print("\n  B.2 GridSearchCV for RandomForestRegressor...")
    rf_regressor = RandomForestRegressor(random_state=hpo_demo_seed + 1)
    rf_reg_param_grid = {
        'model__n_estimators': [20, 30],
        'model__max_depth': [3, 4],
        'model__min_samples_leaf': [2, 4]
    }
    gs_rf_reg_path = os.path.join("hyperparameter_tuning_outputs", "gs_rf_reg_best_model.joblib")
    best_rf_reg_gs, params_rf_reg_gs, score_rf_reg_gs = perform_grid_search_cv_api(
        X_reg_hpo.copy(), y_reg_hpo.copy(), rf_regressor, rf_reg_param_grid, 
        task='regression', cv_folds=2, scoring='r2', 
        random_state_kfold=hpo_demo_seed + 1, save_best_model_path=gs_rf_reg_path
    )
    print(f"    GridSearchCV (RandomForest 回归) 最佳参数: {params_rf_reg_gs}, 最佳CV得分: {score_rf_reg_gs:.4f}")

    # --- (HPO.C) 概念解释 ---
    explain_bayesian_optimization_hpo_conceptual_api()
    explain_automated_hpo_tools_conceptual_api()

    print("\n\n===== 超参数优化 (HPO) 功能接口化演示结束 ======")


def demo_model_interpretability_functionalities():
    """演示 BPredict.model_interpretability_examples 中的核心功能。"""
    print("\n\n===== 模型可解释性 (MI) 功能接口化演示 =====")
    main_mi_seed = 800
    output_dir_mi = "model_interpretability_outputs" # 确保与模块内一致

    # 模型导入 (与模块内__main__保持一致)
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.pipeline import Pipeline # Pipeline is already imported globally but good for clarity

    # --- (MI.A) 分类任务可解释性 --- 
    print("\n\n--- (MI.A) 分类任务可解释性 --- ")
    # A.1 生成数据
    X_clf_mi, y_clf_mi = create_interpretability_sample_data_api(
        n_samples=320, n_features=11, n_informative_num=6, n_cat_features=3,
        task='classification', n_classes=2, random_state=main_mi_seed
    )
    original_clf_cols_mi = X_clf_mi.columns.tolist()
    X_train_clf_mi, X_test_clf_mi, y_train_clf_mi, y_test_clf_mi = train_test_split(
        X_clf_mi, y_clf_mi, test_size=0.22, random_state=main_mi_seed, stratify=y_clf_mi
    )

    # A.2 逻辑回归演示
    print("\n  A.2 逻辑回归系数解释...")
    preprocessor_logreg_clf_mi = get_interpretability_preprocessor_api(X_train_clf_mi.copy())
    pipeline_logreg_clf_mi = Pipeline([
        ('preprocessor', preprocessor_logreg_clf_mi),
        ('model', LogisticRegression(solver='liblinear', random_state=main_mi_seed))
    ])
    pipeline_logreg_clf_mi.fit(X_train_clf_mi, y_train_clf_mi)
    print(f"    逻辑回归 (分类) 测试集准确度: {pipeline_logreg_clf_mi.score(X_test_clf_mi, y_test_clf_mi):.4f}")
    explain_linear_model_coefficients_api(
        pipeline_logreg_clf_mi, X_df_original_cols=original_clf_cols_mi, output_dir=output_dir_mi
    )

    # A.3 随机森林演示
    print("\n  A.3 随机森林特征重要性解释...")
    preprocessor_rf_clf_mi = get_interpretability_preprocessor_api(X_train_clf_mi.copy())
    pipeline_rf_clf_mi = Pipeline([
        ('preprocessor', preprocessor_rf_clf_mi),
        ('model', RandomForestClassifier(n_estimators=55, random_state=main_mi_seed, max_depth=5))
    ])
    pipeline_rf_clf_mi.fit(X_train_clf_mi, y_train_clf_mi)
    print(f"    随机森林 (分类) 测试集准确度: {pipeline_rf_clf_mi.score(X_test_clf_mi, y_test_clf_mi):.4f}")
    explain_tree_feature_importance_api(
        pipeline_rf_clf_mi, X_df_original_cols=original_clf_cols_mi, output_dir=output_dir_mi
    )

    # --- (MI.B) 回归任务可解释性 --- 
    print("\n\n--- (MI.B) 回归任务可解释性 --- ")
    # B.1 生成数据
    X_reg_mi, y_reg_mi = create_interpretability_sample_data_api(
        n_samples=280, n_features=10, n_informative_num=5, n_cat_features=2,
        task='regression', random_state=main_mi_seed + 1
    )
    original_reg_cols_mi = X_reg_mi.columns.tolist()
    X_train_reg_mi, X_test_reg_mi, y_train_reg_mi, y_test_reg_mi = train_test_split(
        X_reg_mi, y_reg_mi, test_size=0.23, random_state=main_mi_seed + 1
    )

    # B.2 线性回归演示
    print("\n  B.2 线性回归系数解释...")
    preprocessor_linreg_mi = get_interpretability_preprocessor_api(X_train_reg_mi.copy())
    pipeline_linreg_mi = Pipeline([
        ('preprocessor', preprocessor_linreg_mi),
        ('model', LinearRegression())
    ])
    pipeline_linreg_mi.fit(X_train_reg_mi, y_train_reg_mi)
    print(f"    线性回归测试集 R2 分数: {pipeline_linreg_mi.score(X_test_reg_mi, y_test_reg_mi):.4f}")
    explain_linear_model_coefficients_api(
        pipeline_linreg_mi, X_df_original_cols=original_reg_cols_mi, output_dir=output_dir_mi
    )

    # B.3 随机森林回归演示
    print("\n  B.3 随机森林回归特征重要性解释...")
    preprocessor_rf_reg_mi = get_interpretability_preprocessor_api(X_train_reg_mi.copy())
    pipeline_rf_reg_mi = Pipeline([
        ('preprocessor', preprocessor_rf_reg_mi),
        ('model', RandomForestRegressor(n_estimators=50, random_state=main_mi_seed + 1, max_depth=4))
    ])
    pipeline_rf_reg_mi.fit(X_train_reg_mi, y_train_reg_mi)
    print(f"    随机森林 (回归) 测试集 R2 分数: {pipeline_rf_reg_mi.score(X_test_reg_mi, y_test_reg_mi):.4f}")
    explain_tree_feature_importance_api(
        pipeline_rf_reg_mi, X_df_original_cols=original_reg_cols_mi, output_dir=output_dir_mi
    )

    # --- (MI.C) LIME 和 SHAP 概念 --- 
    print("\n\n--- (MI.C) LIME 和 SHAP 概念回顾 --- ")
    explain_shap_conceptual_api()
    explain_lime_conceptual_api()

    print("\n\n===== 模型可解释性 (MI) 功能接口化演示结束 ======")


def demo_model_deployment_conceptual_functionalities():
    """演示 BPredict.model_deployment_conceptual 中的核心概念。"""
    print("\n\n===== 模型部署概念 (MDC) 功能接口化演示 =====")
    main_mdc_seed = 420 # 与模块内一致
    output_dir_mdc = "model_deployment_outputs" # 与模块内一致

    # --- (MDC.A) Scikit-learn 模型持久化演示 ---
    print("\n--- (MDC.A) Scikit-learn 模型持久化演示 ---")
    # A.1 创建并训练一个示例Pipeline
    sklearn_pipeline_mdc, X_train_mdc, _ = create_sample_sklearn_pipeline_for_deployment_api(
        n_samples=110, n_features=6, random_state=main_mdc_seed
    )
    print(f"  示例Scikit-learn Pipeline已创建并训练 (使用 {X_train_mdc.shape[0]} 样本)。")

    # A.2 演示模型的保存和加载
    success_persistence = explain_sklearn_model_persistence_api(
        model_pipeline=sklearn_pipeline_mdc,
        model_name="demo_sklearn_pipeline_for_deployment.joblib",
        output_dir=output_dir_mdc,
        sample_data_for_test=X_train_mdc # 使用部分训练数据测试
    )
    if success_persistence:
        print("  Scikit-learn模型持久化演示成功。")
    else:
        print("  Scikit-learn模型持久化演示失败。")

    # --- (MDC.B) 其他部署概念回顾 --- 
    print("\n--- (MDC.B) 其他重要部署概念回顾 (无代码执行) ---")
    explain_keras_model_persistence_conceptual_api()
    explain_pytorch_model_persistence_conceptual_api()
    explain_flask_api_conceptual_api()
    explain_docker_containerization_conceptual_api()
    explain_cloud_deployment_options_conceptual_api()
    explain_production_considerations_conceptual_api()

    print("\n\n===== 模型部署概念 (MDC) 功能接口化演示结束 ======")


def run_all_bpredict_demos():
    """运行 BPredict 部分的所有演示函数。"""
    print("========== 开始 B: 预测性建模 演示 ==========")

    # 确保输出目录存在
    output_dirs_b = [
        "sklearn_outputs", "xgboost_outputs", "lightgbm_outputs",
        "catboost_outputs", "tensorflow_keras_outputs", "pytorch_outputs",
        "model_evaluation_outputs", "hyperparameter_tuning_outputs",
        "model_interpretability_outputs", "model_deployment_outputs" # Added new output dir
    ]
    for out_dir in output_dirs_b:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f"创建输出目录: {out_dir}")

    demo_sklearn_core_functionalities()
    demo_xgboost_functionalities()
    demo_lightgbm_functionalities()
    demo_catboost_functionalities()
    demo_tensorflow_keras_functionalities()
    demo_pytorch_functionalities()
    demo_model_evaluation_comparison_functionalities()
    demo_hyperparameter_tuning_functionalities()
    demo_model_interpretability_functionalities()
    demo_model_deployment_conceptual_functionalities()

    print("========== B: 预测性建模 演示结束 ==========\n\n")

if __name__ == '__main__':
    run_all_bpredict_demos() 