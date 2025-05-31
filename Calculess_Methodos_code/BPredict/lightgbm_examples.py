import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, r2_score, log_loss
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 常量和输出目录设置 (Constants and Output Directory Setup)
# ==============================================================================
OUTPUT_DIR = "lightgbm_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建输出目录: {OUTPUT_DIR}")
# else:
    # print(f"输出目录 {OUTPUT_DIR} 已存在。") # 可选：如果目录已存在，不重复打印

# ==============================================================================
# 数据生成函数 (Data Generation Function)
# ==============================================================================

def create_lightgbm_sample_data(n_samples=250, n_features=12, n_informative_num=6, 
                               n_cat_features=2, n_classes=2, task='classification', 
                               nan_percentage_num=0.05, nan_percentage_cat=0.03, random_state=None):
    """
    创建包含数值和类别特征的样本数据集，用于LightGBM演示，可包含缺失值。

    参数:
    n_samples (int): 样本数量。
    n_features (int): 总特征数量 (数值 + 类别)。
    n_informative_num (int): 数值特征中的信息特征数量。
    n_cat_features (int): 类别特征的数量。
    n_classes (int): 分类任务的类别数量。如果task='regression'，则忽略。
    task (str): 'classification' 或 'regression'。
    nan_percentage_num (float): 在数值特征中引入的NaN值百分比 (0到1之间)。
    nan_percentage_cat (float): 在类别特征中引入的NaN值百分比 (0到1之间)。
    random_state (int, optional): 随机种子。

    返回:
    pd.DataFrame, pd.Series: 特征DataFrame (X) 和目标Series (y)。
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_num_features = n_features - n_cat_features
    if n_num_features < 0: # 如果类别特征数大于总特征数，则不合理
        raise ValueError("总特征数 (n_features) 必须大于或等于类别特征数 (n_cat_features)。")
    if n_num_features == 0 and n_cat_features == 0:
        raise ValueError("必须至少指定一个数值或类别特征。")
        
    if n_informative_num > n_num_features and n_num_features > 0 :
        print(f"警告: 信息数值特征数 ({n_informative_num}) 大于可用数值特征数 ({n_num_features})。已调整为 {n_num_features}。")
        n_informative_num = n_num_features
    elif n_num_features == 0 and n_informative_num > 0:
        print(f"警告: 请求了信息数值特征，但数值特征数量为0。将不生成信息数值特征。")
        n_informative_num = 0


    # 1. 生成数值特征和目标变量
    X_num_data = np.array([[] for _ in range(n_samples)]).reshape(n_samples, 0) # 初始化以处理n_num_features=0的情况
    y_temp = None

    if n_num_features > 0 :
        if task == 'classification':
            from sklearn.datasets import make_classification # 延迟导入
            X_num_data, y_temp = make_classification(
                n_samples=n_samples, n_features=n_num_features,
                n_informative=n_informative_num, n_classes=n_classes,
                n_redundant=max(0, n_num_features - n_informative_num - (1 if n_informative_num < n_num_features else 0) ), # 确保至少有一个非冗余非信息特征（如果空间允许）
                n_repeated=0, random_state=random_state, shuffle=False
            )
        else:  # regression
            from sklearn.datasets import make_regression # 延迟导入
            X_num_data, y_temp = make_regression(
                n_samples=n_samples, n_features=n_num_features,
                n_informative=n_informative_num, noise=0.1, random_state=random_state,
                shuffle=False
            )
    elif task == 'classification': # 如果只有类别特征
        y_temp = np.random.randint(0, n_classes, n_samples)
    else: # 回归且只有类别特征 (这种情况比较少见，目标通常是连续的)
        y_temp = np.random.rand(n_samples) * 100


    y = pd.Series(y_temp, name='target')
    df_num = pd.DataFrame(X_num_data, columns=[f'num_feat_{i}' for i in range(n_num_features)])

    # 2. 生成类别特征
    df_cat_list = []
    if n_cat_features > 0:
        for i in range(n_cat_features):
            num_unique_cats = np.random.randint(3, 7) 
            cat_choices = [f'Cat{i+1}_Val{j+1}' for j in range(num_unique_cats)]
            cat_series = pd.Series(np.random.choice(cat_choices, n_samples), name=f'cat_feat_{i+1}').astype('category')
            df_cat_list.append(cat_series)
        df_cat = pd.concat(df_cat_list, axis=1)
    else:
        df_cat = pd.DataFrame(index=df_num.index if not df_num.empty else pd.RangeIndex(n_samples))

    # 3. 合并特征
    X_df = pd.concat([df_num, df_cat], axis=1)

    # 4. 引入NaN值
    if nan_percentage_num > 0 and n_num_features > 0:
        for col in df_num.columns:
            if X_df[col].empty: continue
            nan_indices = np.random.choice(X_df.index, size=int(X_df.shape[0] * nan_percentage_num), replace=False)
            X_df.loc[nan_indices, col] = np.nan
    
    if nan_percentage_cat > 0 and n_cat_features > 0:
        for col in df_cat.columns:
            if X_df[col].empty: continue
            nan_indices = np.random.choice(X_df.index, size=int(X_df.shape[0] * nan_percentage_cat), replace=False)
            X_df.loc[nan_indices, col] = np.nan

    print(f"已创建LightGBM样本数据集 ({task}): {X_df.shape[0]} 行, {X_df.shape[1]} 列。")
    nan_stats_num = X_df.select_dtypes(include=np.number).isnull().sum().sum()
    nan_stats_cat = X_df.select_dtypes(include='category').isnull().sum().sum()
    print(f"  数值特征NaN数量: {nan_stats_num}")
    print(f"  类别特征NaN数量: {nan_stats_cat}")
    
    # 打乱顺序 (确保X和y同步打乱)
    permuted_indices = np.random.permutation(X_df.index) # Use X_df.index for permutation
    X_df = X_df.iloc[permuted_indices].reset_index(drop=True)
    y = y.iloc[permuted_indices].reset_index(drop=True)
    
    return X_df, y

# ==============================================================================
# LightGBM 原生API 训练与评估 (LightGBM Native API Training & Evaluation)
# ==============================================================================

def train_evaluate_lightgbm_native(
    X_df, y, task='classification',
    categorical_features='auto', # 自动识别Pandas的category类型，或传入列名列表
    lgb_params=None, # LightGBM参数字典
    num_boost_round=100, # 最大提升轮数
    early_stopping_rounds=10, # 早停轮数，None则不使用
    perform_cv=False, # 是否执行lgb.cv
    cv_folds=3, # CV折数
    cv_metrics=None, # CV评估指标列表，None则根据任务自动选择
    test_size=0.25, # 测试集比例
    random_state=None, # 随机种子
    plot_importance=True, # 是否绘制并保存特征重要性图
    plot_importance_type='split' # 'split' or 'gain'
):
    """
    使用LightGBM原生API训练模型并进行评估。

    参数:
    X_df (pd.DataFrame): 特征数据。
    y (pd.Series): 目标变量。
    task (str): 'classification' (分类) 或 'regression' (回归)。
    categorical_features (list of str or 'auto'): 类别特征的列名列表。
                                                 'auto'将自动识别Pandas的'category'类型特征。
    lgb_params (dict, optional): LightGBM的参数字典。如果None，则使用基于任务的默认参数。
    num_boost_round (int): 最大提升轮数。
    early_stopping_rounds (int, optional): 早停轮数。如果None，则不使用早停。
    perform_cv (bool): 是否执行交叉验证 (lgb.cv)。
    cv_folds (int): 交叉验证的折数。
    cv_metrics (list of str, optional): 交叉验证中使用的评估指标。如果None，根据任务类型自动选择。
    test_size (float): 测试集在数据分割时所占的比例。
    random_state (int, optional): 随机种子，用于可复现性。
    plot_importance (bool): 是否绘制并保存特征重要性图。
    plot_importance_type (str): 特征重要性类型 ('split' 或 'gain')。

    返回:
    tuple: (trained_model, evaluation_metrics_dict)
        trained_model (lgb.Booster or dict): 训练好的LightGBM Booster模型对象。
                                             如果 perform_cv=True, 则返回lgb.cv()的结果字典。
        evaluation_metrics_dict (dict): 在测试集上计算的评估指标字典。
                                        如果 perform_cv=True, 则为从CV结果中提取的指标字典。
    """
    if random_state is not None:
        np.random.seed(random_state)

    y_processed = y.copy()
    le = None 
    
    final_lgb_params = {}
    if task == 'classification':
        if not pd.api.types.is_integer_dtype(y_processed) or y_processed.min() != 0: # Check if already 0-indexed integers
            le = LabelEncoder()
            y_processed = pd.Series(le.fit_transform(y_processed), name=y.name)
        
        n_unique_classes = len(y_processed.unique())
        default_objective = 'binary' if n_unique_classes <= 2 else 'multiclass' 
        default_metric = [('binary_logloss' if n_unique_classes <=2 else 'multi_logloss'), ('auc' if n_unique_classes <=2 else 'auc_mu')]


        default_params_clf = {
            'objective': default_objective, 'metric': default_metric,
            'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'seed': random_state, 'n_jobs': -1, 'verbose': -1
        }
        if n_unique_classes > 2: 
            default_params_clf['num_class'] = n_unique_classes
        
        final_lgb_params = {**default_params_clf, **(lgb_params if lgb_params else {})}
        
        if cv_metrics is None:
            cv_metrics = ['auc'] if final_lgb_params.get('objective') == 'binary' else ['multi_logloss', 'auc_mu'] 
    
    elif task == 'regression':
        default_params_reg = {
            'objective': 'regression_l1', 'metric': ['rmse', 'l1'],
            'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.9, 'seed': random_state, 'n_jobs': -1, 'verbose': -1
        }
        final_lgb_params = {**default_params_reg, **(lgb_params if lgb_params else {})}\n        if cv_metrics is None:
            cv_metrics = ['rmse', 'l1']
    else:
        raise ValueError(f"不支持的任务类型: {task}。请选择 'classification' 或 'regression'。")

    X_train, X_test, y_train, y_test_actual = train_test_split(
        X_df, y_processed, test_size=test_size, random_state=random_state, 
        stratify=y_processed if task == 'classification' and len(y_processed.unique()) > 1 else None
    )

    cat_feat_for_lgb = []
    if categorical_features == 'auto':
        cat_feat_for_lgb = X_train.select_dtypes(include='category').columns.tolist()
    elif isinstance(categorical_features, list):
        cat_feat_for_lgb = [col for col in categorical_features if col in X_train.columns]

    lgb_train_data = lgb.Dataset(X_train, y_train, categorical_feature=cat_feat_for_lgb or 'auto', free_raw_data=False)
    lgb_eval_data = lgb.Dataset(X_test, y_test_actual, reference=lgb_train_data, categorical_feature=cat_feat_for_lgb or 'auto', free_raw_data=False)
    
    callbacks = [lgb.log_evaluation(period=0)]
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.insert(0, lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=-1))

    evaluation_results = {}

    if perform_cv:
        print(f"执行LightGBM原生API交叉验证 ({task})...")
        cv_params_for_run = final_lgb_params.copy()
        current_metric_cv = cv_params_for_run.get('metric', [])
        if isinstance(current_metric_cv, str):
            cv_params_for_run['metric'] = [current_metric_cv]
        elif not current_metric_cv: # Ensure metric is set for CV
             cv_params_for_run['metric'] = cv_metrics
        
        if cv_params_for_run.get('objective') == 'binary' and 'num_class' in cv_params_for_run:
            del cv_params_for_run['num_class']

        try:
            cv_output = lgb.cv(
                cv_params_for_run, lgb_train_data, num_boost_round=num_boost_round, nfold=cv_folds,
                stratified=(task == 'classification' and len(y_train.unique()) > 1),
                callbacks=callbacks, seed=random_state, return_cvbooster=False 
            )
            print("LightGBM CV 完成。")
            for metric_key in cv_output.keys(): 
                if metric_key.endswith('-mean'):
                   clean_metric_name = metric_key.replace('valid ', '').replace('-mean', '')
                   # Check if this metric (or part of it, e.g. 'auc' in 'auc_mu') is in our target cv_metrics
                   if any(target_m in clean_metric_name for target_m in cv_metrics):
                       evaluation_results[f"cv_{clean_metric_name}_mean"] = cv_output[metric_key][-1]
            
            print("  CV评估指标 (最后一轮平均值):")
            for k, v_val in evaluation_results.items(): print(f"    {k}: {v_val:.4f}")
            return cv_output, evaluation_results
        except Exception as e:
            print(f"LightGBM CV 失败: {e}")
            return None, {"error": f"CV失败: {str(e)}"}

    print(f"训练LightGBM原生API模型 ({task})...\")
    booster_model = lgb.train(
        final_lgb_params, lgb_train_data, num_boost_round=num_boost_round,
        valid_sets=[lgb_train_data, lgb_eval_data], valid_names=['train', 'eval'],
        callbacks=callbacks
    )
    print(f"模型训练完成。最佳迭代次数: {booster_model.best_iteration}")

    y_pred_raw = booster_model.predict(X_test, num_iteration=booster_model.best_iteration)

    if task == 'classification':
        n_train_classes_check = len(y_train.unique())
        if n_train_classes_check <= 2: 
            y_pred_labels = (y_pred_raw > 0.5).astype(int)
            evaluation_results['accuracy'] = accuracy_score(y_test_actual, y_pred_labels)
            if n_train_classes_check == 2: 
                try: evaluation_results['logloss'] = log_loss(y_test_actual, y_pred_raw)
                except ValueError: evaluation_results['logloss'] = np.nan
                try: evaluation_results['roc_auc'] = roc_auc_score(y_test_actual, y_pred_raw)
                except ValueError: evaluation_results['roc_auc'] = np.nan
        else: 
            y_pred_labels = y_pred_raw.argmax(axis=1)
            evaluation_results['accuracy'] = accuracy_score(y_test_actual, y_pred_labels)
            try: evaluation_results['logloss'] = log_loss(y_test_actual, y_pred_raw)
            except ValueError: evaluation_results['logloss'] = np.nan
            try: evaluation_results['roc_auc_ovr'] = roc_auc_score(y_test_actual, y_pred_raw, multi_class='ovr')
            except ValueError: evaluation_results['roc_auc_ovr'] = np.nan
        print(f"  原生API测试集评估 ({task}): {evaluation_results}")
    
    elif task == 'regression':
        evaluation_results['rmse'] = np.sqrt(mean_squared_error(y_test_actual, y_pred_raw))
        evaluation_results['mae'] = np.mean(np.abs(y_test_actual - y_pred_raw))
        evaluation_results['r2_score'] = r2_score(y_test_actual, y_pred_raw)
        print(f"  原生API测试集评估 ({task}): {evaluation_results}")

    if plot_importance:
        if booster_model and hasattr(booster_model, 'feature_importance'):
            ax = lgb.plot_importance(booster_model, max_num_features=min(15, X_df.shape[1]), 
                                     importance_type=plot_importance_type, figsize=(10,7))
            plt.title(f'LightGBM 特征重要性 ({task.capitalize()}, 原生API, 类型: {plot_importance_type})')
            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"lgbm_native_importance_{task}_{plot_importance_type}.png")
            try:
                plt.savefig(save_path)
                print(f"特征重要性图已保存至: {save_path}")
            except Exception as e: print(f"保存特征重要性图失败: {e}")
            plt.close()
        else: print("未能生成特征重要性图：模型对象无效或无feature_importance方法。")
            
    return booster_model, evaluation_results

# ==============================================================================
# LightGBM Scikit-Learn 包装器与GridSearchCV (LGBM Scikit-Learn Wrapper & GridSearchCV)
# ==============================================================================

def train_evaluate_lightgbm_sklearn(
    X_df, y, task='classification',
    impute_nans=True, 
    encode_categoricals=True, 
    lgbm_model_params=None, 
    use_grid_search=False, 
    param_grid=None, 
    cv_folds_gridsearch=3, 
    scoring_metric_gridsearch=None, 
    test_size=0.25, 
    random_state=None, 
    plot_importance=True,
    plot_importance_type='split'
):
    """
    使用LightGBM的Scikit-Learn包装器进行训练、评估，并可选地进行GridSearchCV。

    参数:
    X_df (pd.DataFrame): 特征数据。
    y (pd.Series): 目标变量。
    task (str): 'classification' (分类) 或 'regression' (回归)。
    impute_nans (bool): 是否对NaN值进行简单填充 (均值/中位数/众数)。
    encode_categoricals (bool): 是否将Pandas类别型特征转换为整数编码。
    lgbm_model_params (dict, optional): 直接传递给LGBMClassifier或LGBMRegressor的参数。
    use_grid_search (bool): 是否使用GridSearchCV进行超参数搜索。
    param_grid (dict, optional): 如果use_grid_search=True，此为GridSearchCV的参数网格。
    cv_folds_gridsearch (int): GridSearchCV的交叉验证折数。
    scoring_metric_gridsearch (str, optional): GridSearchCV的评分指标。如果None，根据任务自动选择。
    test_size (float): 测试集在数据分割时所占的比例。
    random_state (int, optional): 随机种子，用于可复现性。
    plot_importance (bool): 如果不使用GridSearchCV，是否绘制并保存特征重要性图。
    plot_importance_type (str): 特征重要性类型 ('split' 或 'gain')。

    返回:
    tuple: (trained_model_object, evaluation_metrics_dict)
        trained_model_object (lgb.LGBMModel or sklearn.model_selection.GridSearchCV):
                                训练好的LGBM模型对象，或拟合后的GridSearchCV对象。
        evaluation_metrics_dict (dict): 在测试集上计算的评估指标字典 (使用最佳模型)。
    """
    if random_state is not None:
        np.random.seed(random_state)

    X_processed = X_df.copy()
    y_processed = y.copy()
    original_cat_cols = X_processed.select_dtypes(include='category').columns.tolist()
    # For LGBM scikit-learn, categorical_feature should be list of int indices or 'auto'
    # Default to 'auto' if not encoding or no categoricals found after encoding
    fit_params_cat_feature = {'categorical_feature': 'auto'}


    if encode_categoricals and original_cat_cols:
        print("  对类别特征进行整数编码...")
        for col in original_cat_cols:
            X_processed[col] = X_processed[col].cat.codes 
            if (X_processed[col] == -1).any(): # Handle NaNs encoded as -1
                max_code = X_processed[col][X_processed[col] != -1].max()
                fill_value_for_nan_code = max_code + 1 if pd.notna(max_code) else 0
                X_processed[col] = X_processed[col].replace(-1, fill_value_for_nan_code)
        X_processed[original_cat_cols] = X_processed[original_cat_cols].astype(int)
        # Update fit_params with indices of encoded categorical features
        fit_params_cat_feature['categorical_feature'] = [X_processed.columns.get_loc(col) for col in original_cat_cols]


    if impute_nans:
        print("  对特征中的NaN值进行填充...")
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                else: 
                    mode_fill = X_processed[col].mode()
                    X_processed[col] = X_processed[col].fillna(mode_fill[0] if not mode_fill.empty else 0)
    
    le_skl = None
    if task == 'classification':
        if not pd.api.types.is_integer_dtype(y_processed) or y_processed.min() != 0:
            le_skl = LabelEncoder()
            y_processed = pd.Series(le_skl.fit_transform(y_processed), name=y.name)

    X_train, X_test, y_train, y_test_actual = train_test_split(
        X_processed, y_processed, test_size=test_size, random_state=random_state,
        stratify=y_processed if task == 'classification' and len(y_processed.unique()) > 1 else None
    )

    base_model_params = {
        'random_state': random_state, 'n_estimators': 100, 'verbose': -1,
        **(lgbm_model_params if lgbm_model_params else {})
    }

    model_instance = None
    if task == 'classification':
        n_train_unique_classes = len(y_train.unique())
        obj = 'binary' if n_train_unique_classes <= 2 else 'multiclass'
        base_model_params.update({'objective': obj, 'metric': 'logloss'}) # metric for potential early stopping in .fit()
        model_instance = lgb.LGBMClassifier(**base_model_params)
        
        if scoring_metric_gridsearch is None: 
            scoring_metric_gridsearch = 'roc_auc' if obj == 'binary' else 'accuracy'
        if param_grid is None and use_grid_search: 
            param_grid = {'num_leaves': [15, 31], 'learning_rate': [0.01, 0.05, 0.1]}
            print(f"警告: use_grid_search=True 但未提供param_grid。将使用默认网格: {param_grid}")
            
    elif task == 'regression':
        base_model_params.update({'objective': 'regression_l1', 'metric': 'rmse'})
        model_instance = lgb.LGBMRegressor(**base_model_params)
        if scoring_metric_gridsearch is None: scoring_metric_gridsearch = 'r2'
        if param_grid is None and use_grid_search:
            param_grid = {'num_leaves': [15, 31], 'learning_rate': [0.01, 0.05, 0.1]}
            print(f"警告: use_grid_search=True 但未提供param_grid。将使用默认网格: {param_grid}")
    else: raise ValueError(f"不支持的任务类型: {task}")

    final_model_object = None 
    best_estimator = None   

    if use_grid_search and param_grid:
        print(f"为LGBM Scikit-Learn包装器执行GridSearchCV ({task})...")
        grid_search_cv = GridSearchCV(model_instance, param_grid, cv=cv_folds_gridsearch, 
                                   scoring=scoring_metric_gridsearch, verbose=0, n_jobs=-1)
        try:
            grid_search_cv.fit(X_train, y_train, **fit_params_cat_feature)
            print(f"  GridSearchCV 完成。最佳参数: {grid_search_cv.best_params_}")
            print(f"  最佳CV {scoring_metric_gridsearch} 得分: {grid_search_cv.best_score_:.4f}")
            best_estimator = grid_search_cv.best_estimator_
            final_model_object = grid_search_cv
        except Exception as e:
            print(f"GridSearchCV 失败: {e}。将尝试使用基础参数训练单个模型。")
            use_grid_search = False 
            try: 
                best_estimator = model_instance.fit(X_train, y_train, **fit_params_cat_feature)
                final_model_object = best_estimator
            except Exception as e_fit:
                print(f"基础模型训练也失败: {e_fit}")
                best_estimator = None
                final_model_object = None

    else:
        if use_grid_search and not param_grid:
            print("警告: use_grid_search=True 但 param_grid 为空。将使用基础参数训练单个模型。")
        print(f"训练LGBM Scikit-Learn包装器 ({task}) (无GridSearch)...\")
        try:
            best_estimator = model_instance.fit(X_train, y_train, **fit_params_cat_feature)
            final_model_object = best_estimator
            print("模型训练完成。")
        except Exception as e:
            print(f"模型训练失败: {e}")
            best_estimator = None
            final_model_object = None


    evaluation_metrics_skl = {}
    if best_estimator is None:
        print("模型训练失败，无法进行评估。")
        return final_model_object, {"error": "模型训练失败"}
        
    y_pred_labels = best_estimator.predict(X_test)

    if task == 'classification':
        evaluation_metrics_skl['accuracy'] = accuracy_score(y_test_actual, y_pred_labels)
        if hasattr(best_estimator, "predict_proba"):
            y_pred_proba = best_estimator.predict_proba(X_test)
            try: evaluation_metrics_skl['logloss'] = log_loss(y_test_actual, y_pred_proba)
            except ValueError: evaluation_metrics_skl['logloss'] = np.nan
            
            # Check shape of y_pred_proba for roc_auc
            if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] <= 2: # Binary or effectively binary
                # For binary, roc_auc_score needs probabilities of the positive class
                # If y_pred_proba is (n_samples, 1) or (n_samples,), use it directly if it's for positive class
                # If y_pred_proba is (n_samples, 2), use proba of class 1
                proba_for_auc = y_pred_proba
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
                    proba_for_auc = y_pred_proba[:,1]
                try: evaluation_metrics_skl['roc_auc'] = roc_auc_score(y_test_actual, proba_for_auc)
                except ValueError: evaluation_metrics_skl['roc_auc'] = np.nan
            else: # Multiclass
                try: evaluation_metrics_skl['roc_auc_ovr'] = roc_auc_score(y_test_actual, y_pred_proba, multi_class='ovr')
                except ValueError: evaluation_metrics_skl['roc_auc_ovr'] = np.nan
        print(f"  SKL包装器模型测试集评估 ({task}): {evaluation_metrics_skl}")
    
    elif task == 'regression':
        evaluation_metrics_skl['rmse'] = np.sqrt(mean_squared_error(y_test_actual, y_pred_labels))
        evaluation_metrics_skl['mae'] = np.mean(np.abs(y_test_actual - y_pred_labels))
        evaluation_metrics_skl['r2_score'] = r2_score(y_test_actual, y_pred_labels)
        print(f"  SKL包装器模型测试集评估 ({task}): {evaluation_metrics_skl}")

    if plot_importance and not (use_grid_search and param_grid): # Only plot if not using a successful GridSearch
        if hasattr(best_estimator, 'feature_importances_'):
            try:
                ax = lgb.plot_importance(best_estimator, max_num_features=min(15, X_train.shape[1]), 
                                        importance_type=plot_importance_type, figsize=(10,7))
                plt.title(f'LightGBM 特征重要性 ({task.capitalize()}, SKL API, 类型: {plot_importance_type})')
                plt.tight_layout()
                save_path = os.path.join(OUTPUT_DIR, f"lgbm_sklearn_importance_{task}_{plot_importance_type}.png")
                plt.savefig(save_path)
                print(f"特征重要性图已保存至: {save_path}")
                plt.close()
            except Exception as e:
                 print(f"绘制或保存SKL模型特征重要性图失败: {e}")
        else:
            print("此SKL模型实例不直接提供 feature_importances_。无法绘制重要性图。")
            
    return final_model_object, evaluation_metrics_skl

# ==============================================================================
# 演示函数 (用于独立运行和测试) - __main__
# ==============================================================================
if __name__ == '__main__':
    print("===== LightGBM 功能接口化演示 =====")
    main_random_seed = 420 # 主演示的随机种子

    # --- 分类任务演示 ---
    print("\\n\\n--- (A) LightGBM 分类任务演示 ---")
    # A.1 生成分类数据
    X_clf, y_clf = create_lightgbm_sample_data(
        n_samples=300, n_features=15, n_informative_num=7, n_cat_features=3,
        n_classes=3, task='classification', nan_percentage_num=0.08, nan_percentage_cat=0.05,
        random_state=main_random_seed
    )
    print(f"分类数据 X_clf 形状: {X_clf.shape}, y_clf 标签类别: {np.unique(y_clf)}")

    # A.2 使用原生API进行分类训练与评估 (不进行CV，直接训练)
    print("\\n--- A.2. 原生API 分类 (直接训练) ---")
    native_clf_params_custom = {
        'learning_rate': 0.03, 'num_leaves': 25, 
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'reg_alpha': 0.1, 'reg_lambda': 0.1
    }
    native_clf_model, native_clf_metrics = train_evaluate_lightgbm_native(
        X_clf.copy(), y_clf.copy(), task='classification', lgb_params=native_clf_params_custom,
        num_boost_round=120, early_stopping_rounds=15, perform_cv=False,
        random_state=main_random_seed, plot_importance=True, plot_importance_type='gain'
    )
    if native_clf_model:
        print(f"  原生API分类模型训练完成。类型: {type(native_clf_model)}")
        print(f"  测试集评估指标: {native_clf_metrics}")

    # A.3 使用原生API进行分类 (仅执行CV示例)
    print("\\n--- A.3. 原生API 分类 (仅CV) ---")
    native_cv_output, native_cv_metrics_extracted = train_evaluate_lightgbm_native(
        X_clf.copy(), y_clf.copy(), task='classification', perform_cv=True, 
        cv_folds=2, early_stopping_rounds=8, 
        # For 3 classes, auc_mu is appropriate for multiclass AUC
        cv_metrics=['auc_mu', 'multi_logloss'],
        random_state=main_random_seed
    )
    if native_cv_output is not None: # Check if CV was successful
        print(f"  原生API CV 完成。提取的CV指标: {native_cv_metrics_extracted}")

    # A.4 使用Scikit-Learn包装器进行分类 (使用GridSearchCV)
    print("\\n--- A.4. Scikit-Learn包装器分类 (含GridSearchCV) ---")
    skl_clf_param_grid = {
        'n_estimators': [60, 90], 'learning_rate': [0.04, 0.08],
        'num_leaves': [20, 30], 'colsample_bytree': [0.7, 0.9],
    }
    skl_clf_gs_obj, skl_clf_gs_metrics = train_evaluate_lightgbm_sklearn(
        X_clf.copy(), y_clf.copy(), task='classification',
        use_grid_search=True, param_grid=skl_clf_param_grid, 
        cv_folds_gridsearch=2, random_state=main_random_seed,
    )
    if skl_clf_gs_obj and hasattr(skl_clf_gs_obj, 'best_estimator_'):
        print(f"  SKL包装器分类模型 (GridSearchCV) 训练完成。最佳估计器类型: {type(skl_clf_gs_obj.best_estimator_)}")
        print(f"  测试集评估指标 (使用最佳模型): {skl_clf_gs_metrics}")


    # --- 回归任务演示 ---
    print("\\n\\n--- (B) LightGBM 回归任务演示 ---")
    # B.1 生成回归数据
    X_reg, y_reg = create_lightgbm_sample_data(
        n_samples=280, n_features=12, n_informative_num=5, n_cat_features=2,
        task='regression', nan_percentage_num=0.06, nan_percentage_cat=0.02,
        random_state=main_random_seed + 1 
    )
    print(f"回归数据 X_reg 形状: {X_reg.shape}")

    # B.2 使用原生API进行回归训练与评估
    print("\\n--- B.2. 原生API 回归 (直接训练) ---")
    native_reg_params_custom = {
        'learning_rate': 0.04, 'num_leaves': 28, 'max_depth': 6,
        'lambda_l1': 0.05, 'lambda_l2': 0.05, 'min_child_samples': 15
    }
    native_reg_model, native_reg_metrics = train_evaluate_lightgbm_native(
        X_reg.copy(), y_reg.copy(), task='regression', lgb_params=native_reg_params_custom,
        num_boost_round=110, early_stopping_rounds=12, perform_cv=False,
        random_state=main_random_seed + 1, plot_importance=True
    )
    if native_reg_model:
        print(f"  原生API回归模型训练完成。类型: {type(native_reg_model)}")
        print(f"  测试集评估指标: {native_reg_metrics}")

    # B.3 使用Scikit-Learn包装器进行回归 (不使用GridSearchCV，直接训练)
    print("\\n--- B.3. Scikit-Learn包装器回归 (无GridSearch) ---")
    skl_reg_model_params = {
        'n_estimators': 80, 'learning_rate': 0.06,
        'num_leaves': 25, 'subsample': 0.8 
    }
    skl_reg_direct_model, skl_reg_direct_metrics = train_evaluate_lightgbm_sklearn(
        X_reg.copy(), y_reg.copy(), task='regression',
        use_grid_search=False, lgbm_model_params=skl_reg_model_params, 
        random_state=main_random_seed + 1, plot_importance=True, plot_importance_type='gain'
    )
    if skl_reg_direct_model:
        print(f"  SKL包装器回归模型 (无GridSearch) 训练完成。类型: {type(skl_reg_direct_model)}")
        print(f"  测试集评估指标: {skl_reg_direct_metrics}")

    print("\\n\\n===== LightGBM演示全部完成 ======")
    print(f"输出文件 (如图表) 已保存到 '{OUTPUT_DIR}/' 目录。")