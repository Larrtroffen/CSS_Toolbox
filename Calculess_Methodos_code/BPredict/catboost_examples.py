\
import catboost
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, metrics, cv
# from catboost.utils import get_roc_curve # For plotting ROC - 通常在更复杂的绘图函数中使用
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, r2_score, log_loss, make_scorer
# from sklearn.datasets import make_classification, make_regression # 改为内部实现更灵活的数据创建
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import json

# 定义输出目录
OUTPUT_DIR = "catboost_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建目录: {OUTPUT_DIR}")

def create_catboost_sample_data_api(
    n_samples: int = 200, 
    n_features: int = 10, 
    n_informative_num: int = 5,
    n_cat_features: int = 2,
    n_classes: int = 2, 
    task: str = 'classification', 
    nan_percentage_num: float = 0.05,
    nan_percentage_cat: float = 0.05,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    创建包含数值和对象类型类别特征（含NaN）的CatBoost演示数据集。

    参数:
    - n_samples (int): 样本数量。
    - n_features (int): 总特征数量 (数值特征 + 类别特征)。
    - n_informative_num (int): 有信息量的数值特征数量。
    - n_cat_features (int): 类别特征的数量。
    - n_classes (int): (仅分类任务) 类别数量。
    - task (str): 'classification' 或 'regression'。
    - nan_percentage_num (float): 数值特征中NaN的比例。
    - nan_percentage_cat (float): 类别特征中NaN的比例。
    - random_state (int): 随机种子。

    返回:
    - pd.DataFrame: 特征DataFrame。
    - pd.Series: 目标Series。
    """
    np.random.seed(random_state)
    n_num_features = n_features - n_cat_features
    if n_num_features < 0:
        raise ValueError("总特征数必须大于等于类别特征数。")
    if n_informative_num > n_num_features:
        n_informative_num = n_num_features
        print(f"警告: 有信息数值特征数调整为等于总数值特征数: {n_num_features}")

    # 生成数值特征的基础部分
    if task == 'classification':
        # 使用更受控的方式生成分类数据，避免sklearn.datasets的固定模式
        means = np.random.rand(n_classes, n_informative_num) * 5
        covs = [np.eye(n_informative_num) * (np.random.rand() + 0.5) for _ in range(n_classes)]
        X_list = []
        y_list = []
        samples_per_class = n_samples // n_classes
        for i in range(n_classes):
            X_class = np.random.multivariate_normal(means[i], covs[i], samples_per_class)
            X_list.append(X_class)
            y_list.extend([i] * samples_per_class)
        
        X_num_informative = np.vstack(X_list)
        y = np.array(y_list)

        # 补齐样本量 (如果n_samples不能被n_classes整除)
        if len(y) < n_samples:
            remaining_samples = n_samples - len(y)
            X_rem = np.random.multivariate_normal(means[0], covs[0], remaining_samples)
            X_num_informative = np.vstack([X_num_informative, X_rem])
            y = np.concatenate([y, [0]*remaining_samples])
            
        # 添加冗余/噪声数值特征
        if n_num_features > n_informative_num:
            X_num_other = np.random.randn(n_samples, n_num_features - n_informative_num)
            X_num = np.hstack((X_num_informative, X_num_other))
        else:
            X_num = X_num_informative

    else: # 回归任务
        # y = sum of some features + noise
        coeffs = np.random.rand(n_informative_num) * 10 - 5
        X_num = np.random.rand(n_samples, n_num_features) * 10
        y = np.dot(X_num[:, :n_informative_num], coeffs) + np.random.normal(0, 2, n_samples)

    df = pd.DataFrame(X_num, columns=[f'num_feat_{i}' for i in range(n_num_features)])
    
    # 引入类别特征 (对象类型)
    cat_base_names = ['颜色', '形状', '材质', '品牌', '地区']
    for i in range(n_cat_features):
        cat_name = f'cat_feat_{chr(65+i)}' # cat_feat_A, cat_feat_B, ...
        num_unique_cats = np.random.randint(3, 7) # 每列有3到6个唯一类别
        choices = [f'{cat_base_names[i%len(cat_base_names)]}_{j}' for j in range(num_unique_cats)]
        df[cat_name] = np.random.choice(choices, n_samples).astype(str)

    # 引入NaN值
    if n_samples > 0:
        # 数值特征NaN
        for col_idx in range(n_num_features):
            nan_indices_num = np.random.choice(df.index, size=int(n_samples * nan_percentage_num), replace=False)
            df.iloc[nan_indices_num, col_idx] = np.nan
        
        # 类别特征NaN (使用None或np.nan)
        for col_name in df.select_dtypes(include='object').columns:
            nan_indices_cat = np.random.choice(df.index, size=int(n_samples * nan_percentage_cat), replace=False)
            df.loc[nan_indices_cat, col_name] = np.nan if np.random.rand() > 0.5 else None
            
    print(f"创建 CatBoost {task} 数据集: {df.shape[0]} 样本, {df.shape[1]} 特征。")
    print(f"数值特征 NaN 数量: {df.select_dtypes(include=np.number).isnull().sum().sum()}")
    print(f"类别特征 NaN 数量: {df.select_dtypes(include='object').isnull().sum().sum()}")
    return df, pd.Series(y, name='target')

def train_evaluate_catboost_native_api(
    X_df: pd.DataFrame, 
    y_series: pd.Series, 
    task: str, 
    catboost_params: dict,
    test_size: float = 0.25,
    num_boost_round: int = 100,
    early_stopping_rounds: int | None = 10,
    perform_cv: bool = False,
    cv_folds: int = 3,
    cv_metrics: list[str] | None = None, # 例如 ['AUC', 'Logloss'] 或 ['RMSE']
    random_state: int = 42,
    plot_importance: bool = False,
    plot_importance_type: str = 'FeatureImportance' # 'FeatureImportance', 'Interaction', 'ShapValues'
) -> tuple[catboost.CatBoost | dict | None, dict | None]:
    """
    使用CatBoost原生API进行训练和评估 (分类或回归)。

    参数:
    - X_df (pd.DataFrame): 特征DataFrame。
    - y_series (pd.Series): 目标Series。
    - task (str): 'classification' 或 'regression'。
    - catboost_params (dict): CatBoost模型参数。
    - test_size (float): 测试集比例 (如果 perform_cv=False)。
    - num_boost_round (int): 提升轮数。
    - early_stopping_rounds (int | None): 早停轮数。
    - perform_cv (bool): 是否执行交叉验证。
    - cv_folds (int): CV折数。
    - cv_metrics (list[str] | None): CV评估指标列表。
    - random_state (int): 随机种子。
    - plot_importance (bool): 是否绘制特征重要性图。
    - plot_importance_type (str): 特征重要性类型。

    返回:
    - catboost.CatBoost | dict | None: 训练好的模型 (非CV) 或CV结果字典 (CV) 或 None (失败)。
    - dict | None: 评估指标字典或 None (失败)。
    """
    print(f"--- CatBoost 原生 API ({task}) ---")
    
    if task == 'classification' and len(np.unique(y_series)) > 2 and 'loss_function' not in catboost_params:
        catboost_params.setdefault('loss_function', 'MultiClass')
    elif task == 'classification' and 'loss_function' not in catboost_params:
        catboost_params.setdefault('loss_function', 'Logloss')
    elif task == 'regression' and 'loss_function' not in catboost_params:
        catboost_params.setdefault('loss_function', 'RMSE')

    if 'eval_metric' not in catboost_params:
        if task == 'classification':
            catboost_params['eval_metric'] = 'AUC' if len(np.unique(y_series)) == 2 else 'Accuracy'
        else:
            catboost_params['eval_metric'] = 'R2'
    
    catboost_params['random_seed'] = random_state
    catboost_params.setdefault('verbose', 0) # 默认不输出训练过程

    # 识别类别特征 (通过列名)
    categorical_features_names = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    # CatBoost Pool可以直接使用列名列表
    print(f"识别到的类别特征 (名称): {categorical_features_names}")

    if perform_cv:
        print(f"执行CatBoost CV ({cv_folds}折)...")
        cv_pool = Pool(data=X_df, label=y_series, cat_features=categorical_features_names)
        
        cv_params_dict = catboost_params.copy()
        cv_params_dict['iterations'] = num_boost_round # CV时 iterations 是 num_boost_round
        if early_stopping_rounds:
            cv_params_dict['early_stopping_rounds'] = early_stopping_rounds

        try:
            cv_results_df = cv(
                pool=cv_pool,
                params=cv_params_dict,
                fold_count=cv_folds,
                metrics=cv_metrics, # 如果为None, CatBoost会用eval_metric
                stratified=(task == 'classification' and len(np.unique(y_series)) > 1),
                plot=False # 不在函数内绘图，可外部处理
            )
            print("CatBoost CV 完成。")
            
            # 从cv_results_df提取最后一轮的指标
            final_metrics = {}
            for col in cv_results_df.columns:
                if col.startswith('test-') and col.endswith('-mean'): # 通常是这个格式
                    final_metrics[col.replace('test-', '').replace('-mean', '')] = cv_results_df[col].iloc[-1]
                elif col.startswith('train-') and col.endswith('-mean'):
                     final_metrics[f"train_{col.replace('train-', '').replace('-mean', '')}"] = cv_results_df[col].iloc[-1]

            if not final_metrics and not cv_results_df.empty: # 后备：如果上述提取失败，尝试通用提取
                 for metric in (cv_metrics or [catboost_params['eval_metric']]):
                    test_metric_col = f'test-{metric}-mean'
                    if test_metric_col in cv_results_df.columns:
                        final_metrics[metric] = cv_results_df[test_metric_col].iloc[-1]

            print(f"  CV最终平均指标: {final_metrics}")
            
            # 将CV结果保存为JSON和CSV
            cv_results_df.to_csv(os.path.join(OUTPUT_DIR, f"catboost_native_{task}_cv_results.csv"), index=False)
            with open(os.path.join(OUTPUT_DIR, f"catboost_native_{task}_cv_metrics.json"), 'w') as f:
                json.dump(final_metrics, f, indent=4)
            print(f"  CV结果已保存到 {OUTPUT_DIR}")
            return cv_results_df, final_metrics # 返回DataFrame和提取的指标

        except Exception as e:
            print(f"CatBoost CV失败: {e}")
            return None, None
    else: # 直接训练和评估
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=test_size, random_state=random_state, 
            stratify=y_series if task == 'classification' and len(np.unique(y_series)) > 1 else None
        )
        
        train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features_names)
        eval_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features_names)
        print("CatBoost Pools (训练集/测试集) 已创建。")

        model_params = catboost_params.copy()
        model_params['iterations'] = num_boost_round

        if task == 'classification':
            model = CatBoostClassifier(**model_params)
        else:
            model = CatBoostRegressor(**model_params)
        
        print(f"开始训练 CatBoost {task} 模型...")
        model.fit(
            train_pool, 
            eval_set=eval_pool, 
            early_stopping_rounds=early_stopping_rounds,
            verbose=0 # 已在params中设置，确保
        )
        print(f"训练完成。最佳迭代次数: {model.get_best_iteration()}")

        y_pred = model.predict(eval_pool)
        metrics_dict = {}

        if task == 'classification':
            y_pred_proba = model.predict_proba(eval_pool)
            metrics_dict['accuracy'] = accuracy_score(y_test, y_pred.flatten())
            if len(np.unique(y_series)) == 2: # 二分类
                metrics_dict['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                metrics_dict['logloss'] = log_loss(y_test, y_pred_proba)
            else: # 多分类
                metrics_dict['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                metrics_dict['logloss'] = log_loss(y_test, y_pred_proba) # sklearn的log_loss支持多分类
        else: # 回归
            metrics_dict['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics_dict['r2_score'] = r2_score(y_test, y_pred)
            metrics_dict['mse'] = mean_squared_error(y_test, y_pred)
        
        print(f"  测试集评估指标: {metrics_dict}")
        
        # 保存模型和指标
        model_path = os.path.join(OUTPUT_DIR, f"catboost_native_{task}_model.cbm")
        model.save_model(model_path)
        print(f"  模型已保存到: {model_path}")
        with open(os.path.join(OUTPUT_DIR, f"catboost_native_{task}_metrics.json"), 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        if plot_importance:
            try:
                if plot_importance_type == 'ShapValues':
                    # 计算SHAP值需要eval_pool
                    shap_values = model.get_feature_importance(data=eval_pool, type='ShapValues')
                    # 对于多分类，shap_values是 (n_objects, n_features + 1, n_classes) 或 (n_objects, n_features + 1)
                    # 对于二分类或回归，通常是 (n_objects, n_features + 1)
                    # CatBoost 的SHAP可能与 shap 库的输出格式略有不同，通常最后一行是期望值
                    if isinstance(shap_values, np.ndarray) and shap_values.ndim >=2:
                        expected_value = shap_values[0,-1] # 根据CatBoost文档，SHAP值的最后一列是期望值
                        shap_values_actual = shap_values[:,:-1] # 实际特征的SHAP值
                        
                        # 简单地绘制平均绝对SHAP值
                        mean_abs_shap = np.abs(shap_values_actual).mean(axis=0)
                        if mean_abs_shap.ndim > 1: # 多分类情况
                            mean_abs_shap = mean_abs_shap.mean(axis=1) # 对类别取平均或选择一个类别

                        shap_df = pd.DataFrame({'feature': X_df.columns, 'mean_abs_shap': mean_abs_shap})
                        shap_df = shap_df.sort_values(by='mean_abs_shap', ascending=False).head(15)
                        
                        plt.figure(figsize=(10, max(6, len(shap_df) * 0.4)))
                        plt.barh(shap_df['feature'], shap_df['mean_abs_shap'])
                        plt.xlabel("平均绝对SHAP值")
                        plt.title(f"CatBoost 特征重要性 (平均SHAP值) - {task}")
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        plot_path = os.path.join(OUTPUT_DIR, f"catboost_native_{task}_shap_importance.png")
                        plt.savefig(plot_path)
                        plt.close()
                        print(f"  SHAP重要性图已保存: {plot_path}")
                    else:
                        print("  SHAP值格式不符合预期，无法绘制。")

                else: # 'FeatureImportance' 或 'Interaction'
                    feat_imp = model.get_feature_importance(prettified=True, type=plot_importance_type)
                    # get_feature_importance(prettified=True) 返回DataFrame
                    plt.figure(figsize=(10, max(6, len(feat_imp.head(15)) * 0.4)))
                    plt.barh(feat_imp['Feature Id'].head(15), feat_imp['Importances'].head(15))
                    plt.xlabel("重要性")
                    plt.title(f"CatBoost 特征重要性 ({plot_importance_type}) - {task}")
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plot_path = os.path.join(OUTPUT_DIR, f"catboost_native_{task}_{plot_importance_type.lower()}_importance.png")
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"  特征重要性图 ({plot_importance_type}) 已保存: {plot_path}")

            except Exception as e:
                print(f"  绘制特征重要性图失败: {e}")
        
        return model, metrics_dict

def train_evaluate_catboost_sklearn_api(
    X_df: pd.DataFrame, 
    y_series: pd.Series, 
    task: str, 
    catboost_model_params: dict | None = None, # CatBoost模型的基础参数 (非GridSearch时)
    test_size: float = 0.25,
    use_grid_search: bool = False,
    param_grid: dict | None = None, # GridSearchCV的参数网格
    cv_folds_gridsearch: int = 3,
    gridsearch_scoring: str | None = None, # 例如 'roc_auc', 'accuracy', 'r2', 'neg_mean_squared_error'
    random_state: int = 42,
    plot_importance_best_model: bool = False,
    fit_params_gridsearch: dict | None = None # 传递给GridSearch.fit的额外参数 (如eval_set for early stopping)
) -> tuple[catboost.CatBoostClassifier | catboost.CatBoostRegressor | GridSearchCV | None, dict | None]:
    """
    使用CatBoost Scikit-Learn包装器进行训练和评估。

    参数:
    - X_df (pd.DataFrame): 特征DataFrame。
    - y_series (pd.Series): 目标Series。
    - task (str): 'classification' 或 'regression'。
    - catboost_model_params (dict | None): 直接训练时CatBoost模型参数。
    - test_size (float): 测试集比例 (如果 use_grid_search=False)。
    - use_grid_search (bool): 是否使用GridSearchCV。
    - param_grid (dict | None): GridSearchCV的参数网格。
    - cv_folds_gridsearch (int): GridSearchCV的CV折数。
    - gridsearch_scoring (str | None): GridSearchCV的评估指标。
    - random_state (int): 随机种子。
    - plot_importance_best_model (bool): 是否为GridSearch找到的最佳模型绘制特征重要性。
    - fit_params_gridsearch (dict | None): 传递给GridSearchCV.fit()的额外参数, 例如 {'early_stopping_rounds': 10}

    返回:
    - catboost.CatBoostClassifier/Regressor | GridSearchCV | None: 训练好的模型/GridSearchCV对象或None(失败)。
    - dict | None: 评估指标字典或None(失败)。
    """
    print(f"--- CatBoost Scikit-Learn包装器 ({task}) ---")
    
    # 确定类别特征 (名称) - SKL包装器也接受名称列表
    cat_feature_names = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"识别到的类别特征 (用于SKL包装器): {cat_feature_names}")

    # 分割数据 (如果非GridSearch，GridSearch内部会处理)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=test_size, random_state=random_state,
        stratify=y_series if task == 'classification' and len(np.unique(y_series)) > 1 else None
    )

    base_model_params = {
        'random_seed': random_state, 
        'verbose': 0, 
        'cat_features': cat_feature_names # 关键: 传递类别特征名称
    }
    if catboost_model_params: # 用户传入的参数会覆盖默认或添加到base_model_params
        base_model_params.update(catboost_model_params)

    # 如果fit_params_gridsearch中包含eval_set, 需要确保它是基于当前X_test, y_test的
    # 通常SKL接口的early_stopping_rounds会结合eval_set
    if fit_params_gridsearch and 'early_stopping_rounds' in base_model_params: # 或者在catboost_model_params里
        # eval_set for SKL interface should be a tuple (X_val, y_val)
        # CatBoost SKL wrapper's fit method uses 'eval_set'
        # GridSearchCV's fit_params are passed to the estimator's fit method.
        final_fit_params = fit_params_gridsearch.copy()
        if 'eval_set' not in final_fit_params:
             final_fit_params['eval_set'] = (X_test, y_test)
        if 'early_stopping_rounds' not in final_fit_params and 'early_stopping_rounds' in base_model_params:
            final_fit_params['early_stopping_rounds'] = base_model_params.pop('early_stopping_rounds')

    elif 'early_stopping_rounds' in base_model_params: # 非GridSearch,但有早停
        final_fit_params = {'eval_set': (X_test, y_test), 'early_stopping_rounds': base_model_params.pop('early_stopping_rounds')}
    else:
        final_fit_params = {}


    if use_grid_search:
        if not param_grid:
            print("错误: 使用GridSearchCV但未提供param_grid。")
            return None, None
        
        # 确保基础模型参数不与GridSearch的参数网格冲突
        # GridSearch会从param_grid中取值覆盖estimator的初始值
        gs_estimator_params = base_model_params.copy()
        for key in param_grid.keys(): # 从基础参数中移除在param_grid中指定的参数
            gs_estimator_params.pop(key, None)
        
        if task == 'classification':
            estimator = CatBoostClassifier(**gs_estimator_params)
            default_scoring = 'roc_auc' if len(np.unique(y_series)) == 2 else 'accuracy'
        else:
            estimator = CatBoostRegressor(**gs_estimator_params)
            default_scoring = 'r2'
        
        scoring = gridsearch_scoring or default_scoring
        
        print(f"执行GridSearchCV (CV折数={cv_folds_gridsearch}, 评估指标={scoring})...")
        grid_search = GridSearchCV(
            estimator=estimator, 
            param_grid=param_grid, 
            cv=cv_folds_gridsearch, 
            scoring=scoring, 
            verbose=0, 
            n_jobs=-1
        )
        try:
            grid_search.fit(X_train, y_train, **final_fit_params) # 传递 fit_params
            
            print(f"GridSearchCV 完成。最佳参数: {grid_search.best_params_}")
            print(f"  最佳CV {scoring} 分数: {grid_search.best_score_:.4f}")

            best_model = grid_search.best_estimator_
            y_pred_gs = best_model.predict(X_test)
            metrics_gs = {'best_cv_score': grid_search.best_score_, 'best_params': grid_search.best_params_}

            if task == 'classification':
                y_pred_proba_gs = best_model.predict_proba(X_test)
                metrics_gs['test_accuracy'] = accuracy_score(y_test, y_pred_gs.flatten())
                if len(np.unique(y_series)) == 2:
                    metrics_gs['test_roc_auc'] = roc_auc_score(y_test, y_pred_proba_gs[:, 1])
                    metrics_gs['test_logloss'] = log_loss(y_test, y_pred_proba_gs)
                else:
                    metrics_gs['test_roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba_gs, multi_class='ovr')
                    metrics_gs['test_logloss'] = log_loss(y_test, y_pred_proba_gs)
            else: # 回归
                metrics_gs['test_r2_score'] = r2_score(y_test, y_pred_gs)
                metrics_gs['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_gs))
                metrics_gs['test_mse'] = mean_squared_error(y_test, y_pred_gs)
            
            print(f"  最佳模型在测试集上的指标: { {k:v for k,v in metrics_gs.items() if k not in ['best_cv_score', 'best_params']} }")

            # 保存GridSearchCV结果和最佳模型指标
            gs_results_path = os.path.join(OUTPUT_DIR, f"catboost_sklearn_gridsearch_{task}_results.json")
            with open(gs_results_path, 'w') as f:
                json.dump(metrics_gs, f, indent=4, cls=NpEncoder) # 使用自定义Encoder处理Numpy类型
            # 可以考虑保存整个grid_search对象 (例如用joblib)，但这里只保存指标
            print(f"  GridSearchCV结果已保存到 {gs_results_path}")
            
            if plot_importance_best_model:
                try:
                    feat_imp_gs = best_model.get_feature_importance(prettified=True)
                    plt.figure(figsize=(10, max(6, len(feat_imp_gs.head(15)) * 0.4)))
                    plt.barh(feat_imp_gs['Feature Id'].head(15), feat_imp_gs['Importances'].head(15))
                    plt.xlabel("重要性")
                    plt.title(f"CatBoost SKL (GridSearch最佳模型) 特征重要性 - {task}")
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plot_path = os.path.join(OUTPUT_DIR, f"catboost_sklearn_gridsearch_{task}_importance.png")
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"  最佳模型特征重要性图已保存: {plot_path}")
                except Exception as e_plot:
                    print(f"  为GridSearch最佳模型绘制特征重要性图失败: {e_plot}")

            return grid_search, metrics_gs

        except Exception as e:
            print(f"GridSearchCV与CatBoost包装器执行失败: {e}")
            return None, None
            
    else: #直接训练 (不使用GridSearch)
        if task == 'classification':
            model = CatBoostClassifier(**base_model_params)
        else:
            model = CatBoostRegressor(**base_model_params)
        
        print(f"开始训练 CatBoost SKL {task} 模型 (无GridSearch)...")
        try:
            model.fit(X_train, y_train, **final_fit_params) # 使用 fit_params (可能包含eval_set)
            print("训练完成。")
            
            y_pred = model.predict(X_test)
            metrics_direct = {}
            if task == 'classification':
                y_pred_proba = model.predict_proba(X_test)
                metrics_direct['accuracy'] = accuracy_score(y_test, y_pred.flatten())
                if len(np.unique(y_series)) == 2:
                    metrics_direct['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                    metrics_direct['logloss'] = log_loss(y_test, y_pred_proba)
                else:
                    metrics_direct['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    metrics_direct['logloss'] = log_loss(y_test, y_pred_proba)
            else: # 回归
                metrics_direct['r2_score'] = r2_score(y_test, y_pred)
                metrics_direct['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics_direct['mse'] = mean_squared_error(y_test, y_pred)

            print(f"  测试集评估指标: {metrics_direct}")

            # 保存模型和指标
            model_path_skl = os.path.join(OUTPUT_DIR, f"catboost_sklearn_direct_{task}_model.cbm")
            model.save_model(model_path_skl)
            print(f"  模型已保存到: {model_path_skl}")
            with open(os.path.join(OUTPUT_DIR, f"catboost_sklearn_direct_{task}_metrics.json"), 'w') as f:
                json.dump(metrics_direct, f, indent=4)

            if plot_importance_best_model: # 复用此参数名，表示直接训练的模型
                try:
                    feat_imp_direct = model.get_feature_importance(prettified=True)
                    plt.figure(figsize=(10, max(6, len(feat_imp_direct.head(15)) * 0.4)))
                    plt.barh(feat_imp_direct['Feature Id'].head(15), feat_imp_direct['Importances'].head(15))
                    plt.xlabel("重要性")
                    plt.title(f"CatBoost SKL (直接训练) 特征重要性 - {task}")
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plot_path = os.path.join(OUTPUT_DIR, f"catboost_sklearn_direct_{task}_importance.png")
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"  特征重要性图已保存: {plot_path}")
                except Exception as e_plot_direct:
                    print(f"  绘制特征重要性图失败: {e_plot_direct}")
            
            return model, metrics_direct

        except Exception as e_direct:
            print(f"CatBoost SKL直接训练失败: {e_direct}")
            return None, None

# 自定义JSON Encoder来处理Numpy类型，避免GridSearchCV结果保存时的错误
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    print("===== CatBoost API化功能演示 =====")
    main_random_seed = 777

    # --- 1. 分类任务演示 ---
    print("\n\n*** 1. CatBoost 分类任务演示 ***")
    X_clf, y_clf = create_catboost_sample_data_api(
        n_samples=250, n_features=12, n_informative_num=6, n_cat_features=3,
        n_classes=3, task='classification', nan_percentage_num=0.08, nan_percentage_cat=0.06,
        random_state=main_random_seed
    )
    print(f"  分类数据 X_clf 形状: {X_clf.shape}, y_clf 标签类别: {np.unique(y_clf)}")
    print(f"  X_clf 列类型:\n{X_clf.dtypes.value_counts()}")


    # 1.1 原生API分类 (直接训练)
    print("\n--- 1.1 原生API 分类 (直接训练与评估) ---")
    native_clf_params = {
        'learning_rate': 0.05, 'depth': 4, 'l2_leaf_reg': 2,
        # loss_function 和 eval_metric 会被API自动设置
    }
    native_clf_model, native_clf_metrics = train_evaluate_catboost_native_api(
        X_clf.copy(), y_clf.copy(), task='classification', catboost_params=native_clf_params,
        num_boost_round=80, early_stopping_rounds=8, perform_cv=False,
        random_state=main_random_seed, plot_importance=True, plot_importance_type='FeatureImportance'
    )
    if native_clf_model:
        print(f"  原生API分类模型 ({type(native_clf_model).__name__}) 训练完成。")
        print(f"  测试集指标: {native_clf_metrics}")

    # 1.2 原生API分类 (CV)
    print("\n--- 1.2 原生API 分类 (交叉验证) ---")
    native_clf_cv_params = {'learning_rate': 0.07, 'depth': 3} # 简化CV参数
    cv_clf_results_df, cv_clf_metrics = train_evaluate_catboost_native_api(
        X_clf.copy(), y_clf.copy(), task='classification', catboost_params=native_clf_cv_params,
        num_boost_round=50, early_stopping_rounds=5, perform_cv=True, cv_folds=2,
        cv_metrics=['MultiClass', 'AUC'], # 为多分类指定
        random_state=main_random_seed
    )
    if cv_clf_results_df is not None:
        print(f"  原生API分类CV完成。提取的CV指标: {cv_clf_metrics}")
        # print(f"  CV结果DataFrame (前5行):\n{cv_clf_results_df.head()}")

    # 1.3 SKLearn包装器分类 (GridSearchCV)
    print("\n--- 1.3 Scikit-Learn包装器 分类 (GridSearchCV) ---")
    skl_clf_param_grid = {
        'n_estimators': [40, 60], # iterations -> n_estimators
        'learning_rate': [0.08, 0.12],
        'depth': [2, 4]
    }
    # 对于SKL包装器内的早停，需要通过 fit_params 传递
    skl_fit_params = {'early_stopping_rounds': 7} 

    gs_clf_obj, gs_clf_metrics = train_evaluate_catboost_sklearn_api(
        X_clf.copy(), y_clf.copy(), task='classification',
        use_grid_search=True, param_grid=skl_clf_param_grid, cv_folds_gridsearch=2,
        gridsearch_scoring='accuracy', # 或者 'roc_auc_ovr' for multi-class
        random_state=main_random_seed, plot_importance_best_model=True,
        fit_params_gridsearch=skl_fit_params
    )
    if gs_clf_obj:
        print(f"  SKL包装器分类 (GridSearchCV) 完成。最佳模型类型: {type(gs_clf_obj.best_estimator_).__name__}")
        print(f"  GridSearchCV指标: {gs_clf_metrics}")


    # --- 2. 回归任务演示 ---
    print("\n\n*** 2. CatBoost 回归任务演示 ***")
    X_reg, y_reg = create_catboost_sample_data_api(
        n_samples=220, n_features=10, n_informative_num=5, n_cat_features=2,
        task='regression', nan_percentage_num=0.06, nan_percentage_cat=0.07,
        random_state=main_random_seed + 1
    )
    print(f"  回归数据 X_reg 形状: {X_reg.shape}")
    print(f"  X_reg 列类型:\n{X_reg.dtypes.value_counts()}")

    # 2.1 原生API回归 (直接训练)
    print("\n--- 2.1 原生API 回归 (直接训练与评估) ---")
    native_reg_params = {'learning_rate': 0.06, 'depth': 5, 'loss_function': 'RMSE', 'eval_metric':'R2'}
    native_reg_model, native_reg_metrics = train_evaluate_catboost_native_api(
        X_reg.copy(), y_reg.copy(), task='regression', catboost_params=native_reg_params,
        num_boost_round=70, early_stopping_rounds=7, perform_cv=False,
        random_state=main_random_seed + 1, plot_importance=True, plot_importance_type='ShapValues' # 演示SHAP
    )
    if native_reg_model:
        print(f"  原生API回归模型 ({type(native_reg_model).__name__}) 训练完成。")
        print(f"  测试集指标: {native_reg_metrics}")

    # 2.2 SKLearn包装器回归 (无GridSearch, 直接训练)
    print("\n--- 2.2 Scikit-Learn包装器 回归 (直接训练) ---")
    skl_reg_direct_params = {
        'n_estimators': 60, 'learning_rate': 0.09, 'depth': 4,
        'early_stopping_rounds': 6 # SKL包装器可以直接在构造函数中接受early_stopping_rounds
    }
    # fit_params for direct training for eval_set if early_stopping_rounds is in model params
    skl_direct_fit_params_reg = {} 
    
    skl_reg_model, skl_reg_metrics = train_evaluate_catboost_sklearn_api(
        X_reg.copy(), y_reg.copy(), task='regression',
        catboost_model_params=skl_reg_direct_params, # 将早停参数放在这里
        use_grid_search=False,
        random_state=main_random_seed + 1, plot_importance_best_model=True,
        fit_params_gridsearch=skl_direct_fit_params_reg # 传递空的或包含eval_set的字典
    )
    if skl_reg_model:
         print(f"  SKL包装器回归 (直接训练) 完成。模型类型: {type(skl_reg_model).__name__}")
         print(f"  测试集指标: {skl_reg_metrics}")


    print("\n\n===== CatBoost API化功能演示完成 =====")
    print(f"所有输出 (模型, 指标JSON, 图表) 保存在 '{OUTPUT_DIR}' 目录中。")
