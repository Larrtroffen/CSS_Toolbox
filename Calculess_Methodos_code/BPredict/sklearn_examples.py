import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 数据预处理 (Data Preprocessing)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# 2. 特征工程与选择 (Feature Engineering & Selection)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, SelectFromModel
from sklearn.decomposition import PCA, NMF

# 3. 模型选择与评估 (Model Selection & Evaluation)
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 4. 监督学习算法 (Supervised Learning Algorithms)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGDClassifier
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB # MultinomialNB, BernoulliNB for text usually
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, HistGradientBoostingClassifier

# 5. 工作流构建 (Workflow Building)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 示例数据集 (Sample Datasets - for internal demo, user provides data generally)
from sklearn.datasets import make_classification, make_regression

# ==============================================================================
# 常量和输出目录设置 (Constants and Output Directory Setup)
# ==============================================================================
OUTPUT_DIR = "sklearn_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建输出目录: {OUTPUT_DIR}")

# ==============================================================================
# 0. 示例数据生成 (Sample Data Generation)
# ==============================================================================
def create_sklearn_sample_data(n_samples=200, n_features=10, n_informative=5, n_classes=2, task='classification', random_state=None):
    """
    为Scikit-learn演示创建一个包含混合特征类型（包括NaN）的数据集。

    参数:
    n_samples (int): 样本数量。
    n_features (int): 数值特征的总数量。
    n_informative (int): 有信息量的特征数量。
    n_classes (int): (仅分类任务) 类别数量。
    task (str): 'classification' 或 'regression'。
    random_state (int, optional): 随机种子。

    返回:
    pd.DataFrame: 特征DataFrame (X)。
    pd.Series: 目标Series (y)。
    list: 数值特征列名列表。
    list: 分类特征列名列表。
    """
    if random_state is not None:
        np.random.seed(random_state)

    if task == 'classification':
        X_np, y_np = make_classification(n_samples=n_samples, n_features=n_features, 
                                       n_informative=n_informative, n_classes=n_classes, 
                                       random_state=random_state, n_redundant=max(0, n_features - n_informative - 2), n_repeated=0)
    else: # regression
        X_np, y_np = make_regression(n_samples=n_samples, n_features=n_features,
                                   n_informative=n_informative, random_state=random_state)

    num_col_names = [f'num_feat_{i}' for i in range(n_features)]
    df = pd.DataFrame(X_np, columns=num_col_names)
    y = pd.Series(y_np, name='target')
    
    # 引入分类特征
    cat_col_names = []
    cat_choices1 = ['CatA_Val1', 'CatA_Val2', 'CatA_Val3', 'CatA_Val4']
    cat_choices2 = ['CatB_OptX', 'CatB_OptY', 'CatB_OptZ']
    if n_features >= 2: # 至少需要两个数值特征才能转换为分类特征
        df['cat_feat_1'] = pd.Series(np.random.choice(cat_choices1, n_samples)).astype('object')
        df['cat_feat_2'] = pd.Series(np.random.choice(cat_choices2, n_samples)).astype('object')
        cat_col_names = ['cat_feat_1', 'cat_feat_2']
        # 将一些原始数值列移除或修改，避免完美共线性 (可选，但更真实)
        if 'num_feat_0' in df.columns: df = df.drop(columns=['num_feat_0'])
        if 'num_feat_1' in df.columns and 'num_feat_0' not in num_col_names : num_col_names.pop(0) # adjust if dropped
        if 'num_feat_1' in df.columns: df = df.drop(columns=['num_feat_1'])
        if 'num_feat_1' not in num_col_names and len(num_col_names)>0 : num_col_names.pop(0) # adjust if dropped
        
    # 更新实际的数值列名列表
    current_num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # 引入NaNs
    all_cols_for_nans = current_num_cols + cat_col_names
    for col in all_cols_for_nans:
        nan_fraction = np.random.uniform(0.05, 0.15) # 每列5%到15%的NaN
        nan_indices = np.random.choice(df.index, size=int(n_samples * nan_fraction), replace=False)
        df.loc[nan_indices, col] = np.nan
        
    print(f"已创建{task}数据集: {df.shape[0]}样本, {df.shape[1]}特征。包含NaNs:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df, y, current_num_cols, cat_col_names

# ==============================================================================
# 1. 数据预处理 (Data Preprocessing)
# ==============================================================================
def preprocess_data(X_df, num_features, cat_features, 
                    num_impute_strategy='median', cat_impute_strategy='most_frequent',
                    scaler_type='standard', apply_powertransformer=False, apply_kbins=False):
    """
    对DataFrame进行预处理，包括缺失值填充、数值缩放和分类编码。

    参数:
    X_df (pd.DataFrame): 输入特征DataFrame。
    num_features (list): 数值特征列名列表。
    cat_features (list): 分类特征列名列表。
    num_impute_strategy (str): 数值特征的缺失值填充策略。
    cat_impute_strategy (str): 分类特征的缺失值填充策略。
    scaler_type (str): 数值特征的缩放类型 ('standard', 'minmax', 'robust')。
    apply_powertransformer (bool): 是否对数值特征应用PowerTransformer (Yeo-Johnson)。
    apply_kbins (bool): 是否对第一个数值特征应用KBinsDiscretizer。

    返回:
    pd.DataFrame: 预处理后的DataFrame。
    ColumnTransformer: (可选) 如果需要，返回配置好的preprocessor对象，以便后续在Pipeline中使用。
    """
    print("\n--- 1. 数据预处理 --- ")
    X_processed = X_df.copy()

    # 定义转换器
    numeric_steps = []
    if num_features:
        numeric_steps.append(('imputer', SimpleImputer(strategy=num_impute_strategy)))
        if scaler_type == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif scaler_type == 'minmax':
            numeric_steps.append(('scaler', MinMaxScaler()))
        elif scaler_type == 'robust':
            numeric_steps.append(('scaler', RobustScaler()))
        if apply_powertransformer:
            numeric_steps.append(('power_transform', PowerTransformer(method='yeo-johnson')))
    
    categorical_steps = []
    if cat_features:
        categorical_steps.append(('imputer', SimpleImputer(strategy=cat_impute_strategy)))
        categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))

    # 创建ColumnTransformer
    transformers_list = []
    if num_features:
        transformers_list.append(('num', Pipeline(steps=numeric_steps), num_features))
    if cat_features:
        transformers_list.append(('cat', Pipeline(steps=categorical_steps), cat_features))

    if not transformers_list:
        print("警告: 未提供数值或分类特征进行预处理。")
        return X_processed, None 
        
    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough') # 保留其他列
    
    # 应用预处理
    try:
        X_transformed_np = preprocessor.fit_transform(X_processed)
    except ValueError as e:
        print(f"预处理过程中发生错误: {e}。请检查特征列表和数据类型。")
        print("数值特征:", num_features)
        print("分类特征:", cat_features)
        print("数据信息:", X_processed.info())
        return X_processed, None # or raise e

    # 获取OneHotEncoder生成的特征名
    feature_names_out = []
    if cat_features and 'onehot' in preprocessor.named_transformers_['cat'].named_steps:
        ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
        # 构建最终的特征名列表
        num_processed_names = num_features
        feature_names_out.extend(num_processed_names)
        feature_names_out.extend(ohe_feature_names)
    else:
        feature_names_out.extend(num_features)
    
    # 处理 remainder='passthrough' 可能引入的列
    if preprocessor.remainder == 'passthrough' and hasattr(preprocessor, "feature_names_in_"):
        input_cols = preprocessor.feature_names_in_
        processed_cols_flat = []
        for name, _, cols in preprocessor.transformers_:
            if name != 'remainder': 
                processed_cols_flat.extend(cols)
        remainder_cols = [col for col in input_cols if col not in processed_cols_flat]
        feature_names_out.extend(remainder_cols)
    
    X_transformed_df = pd.DataFrame(X_transformed_np, columns=feature_names_out, index=X_processed.index)
    print("数据预处理完成。")
    print("缺失值处理后:", X_transformed_df.isnull().sum().sum(), "NaNs")
    print("数值特征已缩放，分类特征已进行独热编码。")

    # 可选的KBinsDiscretizer (在转换后的数据上，对第一个数值特征操作)
    if apply_kbins and num_features and X_transformed_df.shape[1] > 0:
        first_num_col_name = num_features[0] # 假设它仍然是第一个，或可以通过索引找到
        if first_num_col_name in X_transformed_df.columns:
            try:
                discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
                # 注意: KBinsDiscretizer需要2D输入
                binned_feature = discretizer.fit_transform(X_transformed_df[[first_num_col_name]])
                X_transformed_df[f'{first_num_col_name}_binned'] = binned_feature.astype(int)
                print(f"KBinsDiscretizer已应用于特征 '{first_num_col_name}'。")
            except ValueError as e:
                print(f"对 '{first_num_col_name}' 应用KBinsDiscretizer失败: {e}")
        else:
            print(f"警告: KBins无法找到特征 '{first_num_col_name}' 在转换后的DataFrame中。")

    return X_transformed_df, preprocessor # 返回预处理器以备Pipeline使用

# ==============================================================================
# 2. 特征工程与选择 (Feature Engineering & Selection)
# ==============================================================================
def select_features_sklearn(X_processed, y, task='classification', k_best=5, n_rfe=3):
    """
    对预处理后的数据进行特征选择。

    参数:
    X_processed (pd.DataFrame): 预处理后的特征DataFrame (全数值)。
    y (pd.Series): 目标变量。
    task (str): 'classification' 或 'regression'。
    k_best (int): SelectKBest要选择的特征数量。
    n_rfe (int): RFE要选择的特征数量。

    返回:
    pd.DataFrame: 包含所选特征的DataFrame (基于SelectKBest的结果)。
    list: SelectKBest选择的特征名列表。
    """
    print("\n--- 2. 特征选择 --- ")
    if not isinstance(X_processed, pd.DataFrame) or X_processed.select_dtypes(include='object').shape[1] > 0:
        print("警告: 特征选择需要全数值的DataFrame。请确保数据已正确预处理。")
        # 尝试转换为全数值，如果失败则跳过
        try:
            X_numeric = X_processed.astype(float)
        except ValueError:
            print("无法将X_processed转换为全数值，跳过特征选择。")
            return X_processed, list(X_processed.columns)
    else:
        X_numeric = X_processed

    k_to_select = min(k_best, X_numeric.shape[1])
    score_func = f_classif if task == 'classification' else f_regression
    selector_kbest = SelectKBest(score_func=score_func, k=k_to_select)
    
    try:
        X_kbest_np = selector_kbest.fit_transform(X_numeric, y)
        selected_kbest_names = X_numeric.columns[selector_kbest.get_support()].tolist()
        print(f"SelectKBest选择了 {len(selected_kbest_names)} 个特征: {selected_kbest_names}")
        X_selected_df = pd.DataFrame(X_kbest_np, columns=selected_kbest_names, index=X_numeric.index)
    except Exception as e:
        print(f"SelectKBest失败: {e}。返回原始处理数据。")
        return X_numeric, list(X_numeric.columns)

    # RFE (Recursive Feature Elimination)
    if X_numeric.shape[1] >= 2:
        n_to_select_rfe = min(n_rfe, X_numeric.shape[1])
        if task == 'classification':
            estimator_rfe = LogisticRegression(solver='liblinear', random_state=42, max_iter=100)
        else:
            estimator_rfe = LinearRegression()
        
        selector_rfe = RFE(estimator=estimator_rfe, n_features_to_select=n_to_select_rfe)
        try:
            selector_rfe.fit(X_numeric, y)
            selected_rfe_names = X_numeric.columns[selector_rfe.support_].tolist()
            print(f"RFE选择了 {len(selected_rfe_names)} 个特征: {selected_rfe_names}")
        except Exception as e:
            print(f"RFE失败: {e}")
    else:
        print("特征数量不足，跳过RFE。")

    return X_selected_df, selected_kbest_names # 返回基于KBest选择的特征

def apply_pca_sklearn(X_scaled_df, n_components=0.95, random_state=None):
    """
    对已缩放的数据应用PCA进行降维。

    参数:
    X_scaled_df (pd.DataFrame): 已标准化/缩放的特征DataFrame。
    n_components (int, float, str or None): PCA保留的主成分数量或方差比例。
    random_state (int, optional): 随机种子。

    返回:
    pd.DataFrame: 降维后的DataFrame。
    PCA: 拟合的PCA模型对象。
    """
    print("\n--- 应用PCA降维 --- ")
    if not isinstance(X_scaled_df, pd.DataFrame):
        print("警告: PCA需要Pandas DataFrame输入。")
        return X_scaled_df, None
    
    pca = PCA(n_components=n_components, random_state=random_state)
    try:
        X_pca_np = pca.fit_transform(X_scaled_df)
        X_pca_df = pd.DataFrame(X_pca_np, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=X_scaled_df.index)
        print(f"PCA完成。原始维度: {X_scaled_df.shape[1]}, 降维后维度: {pca.n_components_}。")
        print(f"  保留的累积解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")
        return X_pca_df, pca
    except Exception as e:
        print(f"PCA应用失败: {e}。返回原始缩放数据。")
        return X_scaled_df, None

# ==============================================================================
# 3. 模型训练、评估与工作流 (Model Training, Evaluation & Workflow)
# ==============================================================================
def train_evaluate_model(X_df, y, task='classification', model_type='random_forest', 
                           preprocessor_obj=None, test_size=0.25, random_state=None, 
                           model_params=None, cv_folds=3):
    """
    构建一个包含预处理（如果提供了preprocessor_obj）和模型的Pipeline，
    然后训练、评估模型，并进行交叉验证。

    参数:
    X_df (pd.DataFrame): 特征DataFrame。
    y (pd.Series): 目标变量。
    task (str): 'classification' 或 'regression'。
    model_type (str): 要使用的模型类型 ('random_forest', 'logistic_regression', 'linear_regression', etc.)
    preprocessor_obj (ColumnTransformer, optional): （可选）来自preprocess_data的预处理器对象。
                                                  如果为None，则假设X_df已完全预处理。
    test_size (float): 测试集比例。
    random_state (int, optional): 随机种子。
    model_params (dict, optional): 模型参数字典。
    cv_folds (int): 交叉验证折数。

    返回:
    Pipeline: 训练好的Pipeline对象。
    dict: 评估指标字典。
    """
    print(f"\n--- 3-5. 模型训练与评估 ({task} - {model_type}) --- ")    
    if model_params is None: model_params = {}
    if random_state is not None: model_params['random_state'] = random_state

    # 选择模型
    if task == 'classification':
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(**model_params)
        elif model_type == 'logistic_regression':
            base_model = LogisticRegression(solver='liblinear', **model_params)
        elif model_type == 'svc':
            base_model = SVC(probability=True, **model_params) # probability=True for roc_auc
        else:
            raise ValueError(f"不支持的分类模型类型: {model_type}")
        scoring_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_ovr']
    elif task == 'regression':
        if model_type == 'random_forest':
            base_model = RandomForestRegressor(**model_params)
        elif model_type == 'linear_regression':
            base_model = LinearRegression(**model_params) # LinearRegression may not have random_state
            if 'random_state' in model_params and not hasattr(base_model, 'random_state'):
                 del model_params['random_state'] # Remove if not applicable
                 base_model = LinearRegression(**model_params) # Re-initialize
        elif model_type == 'svr':
            base_model = SVR(**model_params)
        else:
            raise ValueError(f"不支持的回归模型类型: {model_type}")
        scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    else:
        raise ValueError(f"无效的任务类型: {task}")

    # 构建Pipeline
    pipeline_steps = []
    if preprocessor_obj:
        pipeline_steps.append(('preprocessor', preprocessor_obj))
    pipeline_steps.append(('model', base_model))
    full_pipeline = Pipeline(steps=pipeline_steps)

    # 数据分割
    stratify_opt = y if task == 'classification' and len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=test_size, random_state=random_state, stratify=stratify_opt)
    print(f"数据已分割: 训练集 {X_train.shape}, 测试集 {X_test.shape}")

    # 训练Pipeline
    print(f"正在使用Pipeline训练 {model_type} 模型...")
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    evaluation_results = {}
    print("\n模型评估结果 (测试集):")
    if task == 'classification':
        evaluation_results['accuracy'] = accuracy_score(y_test, y_pred)
        evaluation_results['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        evaluation_results['precision_macro'] = precision_score(y_test, y_pred, average='macro')
        evaluation_results['recall_macro'] = recall_score(y_test, y_pred, average='macro')
        print(f"  准确率: {evaluation_results['accuracy']:.4f}")
        print(f"  F1分数 (Macro): {evaluation_results['f1_macro']:.4f}")
        if hasattr(full_pipeline, "predict_proba"):
            y_pred_proba = full_pipeline.predict_proba(X_test)
            # ROC AUC for binary or multiclass (One-vs-Rest)
            if y_pred_proba.shape[1] == 2: # Binary
                evaluation_results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                print(f"  ROC AUC: {evaluation_results['roc_auc']:.4f}")
            else: # Multiclass
                try:
                    evaluation_results['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    print(f"  ROC AUC (OvR): {evaluation_results['roc_auc_ovr']:.4f}")
                except ValueError as e:
                    print(f"  计算ROC AUC (OvR)失败: {e}") # e.g. if only one class in y_true after split for a fold
        
        print("  分类报告:\n", classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title(f'混淆矩阵 - {model_type}')
        plt.xlabel('预测标签'); plt.ylabel('真实标签')
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{task}_{model_type}.png")
        plt.savefig(cm_path); plt.close()
        print(f"  混淆矩阵已保存至: {cm_path}")

    else: # Regression
        evaluation_results['r2'] = r2_score(y_test, y_pred)
        evaluation_results['mse'] = mean_squared_error(y_test, y_pred)
        evaluation_results['mae'] = mean_absolute_error(y_test, y_pred)
        print(f"  R2分数: {evaluation_results['r2']:.4f}")
        print(f"  均方误差 (MSE): {evaluation_results['mse']:.4f}")
        print(f"  平均绝对误差 (MAE): {evaluation_results['mae']:.4f}")

    # 交叉验证
    print(f"\n交叉验证 ({cv_folds}-fold):")
    cv_results = cross_validate(full_pipeline, X_df, y, cv=cv_folds, scoring=scoring_metrics, n_jobs=-1, error_score='raise')
    for metric in scoring_metrics:
        # cross_validate returns test_score, e.g. test_accuracy
        actual_metric_name_in_cv = f'test_{metric}' if f'test_{metric}' in cv_results else metric 
        if actual_metric_name_in_cv in cv_results:
            mean_score = np.mean(cv_results[actual_metric_name_in_cv])
            std_score = np.std(cv_results[actual_metric_name_in_cv])
            evaluation_results[f'cv_{metric}_mean'] = mean_score
            evaluation_results[f'cv_{metric}_std'] = std_score
            print(f"  {metric} (CV均值): {mean_score:.4f} (+/- {std_score*2:.4f})")
        else:
            print(f"  指标 {metric} 未在交叉验证结果中找到。可用: {list(cv_results.keys())}")
            
    return full_pipeline, evaluation_results

# ==============================================================================
# 6. 超参数调优 (概念性) (Hyperparameter Tuning - Conceptual)
# ==============================================================================
def conceptual_hyperparameter_tuning(pipeline_obj, X_train, y_train, task='classification'):
    """
    提供一个GridSearchCV的超参数调优概念性示例。
    实际运行可能非常耗时，因此这里主要展示结构。
    """
    print("\n--- 6. 超参数调优 (概念性示例 GridSearchCV) ---")
    # 注意: 这里的参数网格非常小，仅为演示。
    # `model__` 前缀用于访问Pipeline中名为 'model' 的步骤的参数。
    if task == 'classification' and isinstance(pipeline_obj.named_steps['model'], RandomForestClassifier):
        param_grid = {
            'model__n_estimators': [50, 100], # Pipeline步骤名__参数名
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 4]
        }
        scoring_tune = 'f1_macro'
    elif task == 'regression' and isinstance(pipeline_obj.named_steps['model'], RandomForestRegressor):
        param_grid = {
            'model__n_estimators': [50, 100],
            'model__max_depth': [5, 10, None]
        }
        scoring_tune = 'r2'
    elif task == 'classification' and isinstance(pipeline_obj.named_steps['model'], LogisticRegression):
        param_grid = {
            'model__C': [0.1, 1.0, 10.0],
            'model__penalty': ['l1', 'l2']
        }
        scoring_tune = 'f1_macro'
    else:
        print("当前模型类型的调优参数网格未定义，跳过GridSearchCV演示。")
        return None

    print(f"参数网格示例: {param_grid}")
    print(f"将使用 '{scoring_tune}' 进行评分。这可能需要很长时间...")
    
    # 为了避免在演示中运行过久，这里不实际执行fit，或只用极少数据
    # grid_search = GridSearchCV(pipeline_obj, param_grid, cv=2, scoring=scoring_tune, verbose=1, n_jobs=-1)
    # print("如需实际运行GridSearchCV, 请取消注释相关代码并准备好等待时间。")
    # try:
    #     # 使用一小部分数据进行快速演示 (如果需要运行)
    #     # X_sample_tune, _, y_sample_tune, _ = train_test_split(X_train, y_train, train_size=0.2, random_state=42, stratify=y_train if task=='classification' else None)
    #     # grid_search.fit(X_sample_tune, y_sample_tune)
    #     # print(f"最佳参数: {grid_search.best_params_}")
    #     # print(f"最佳 {scoring_tune} 分数: {grid_search.best_score_:.4f}")
    #     # return grid_search.best_estimator_
    # except Exception as e:
    #     print(f"GridSearchCV 运行 (或示例部分) 失败: {e}")
    print("GridSearchCV 设置完毕 (概念性，未实际运行以节约时间)。")
    return None # 在此概念性演示中不返回调优后的模型

# ==============================================================================
# 主演示流程 (Main Demonstration Flow)
# ==============================================================================
def run_sklearn_demos():
    """运行所有Scikit-learn核心功能演示。"""
    print("========== Scikit-learn核心功能接口化演示 ==========")

    # --- 分类任务演示 ---
    print("\n\n======== 分类任务演示 ========")
    X_clf_raw, y_clf, num_cols_clf, cat_cols_clf = create_sklearn_sample_data(
        n_samples=250, n_features=8, n_informative=5, n_classes=3, task='classification', random_state=42
    )
    X_clf_processed, preprocessor_clf = preprocess_data(X_clf_raw, num_cols_clf, cat_cols_clf, apply_kbins=True)
    
    if X_clf_processed is not None and not X_clf_processed.empty:
        X_clf_selected, sel_clf_names = select_features_sklearn(X_clf_processed, y_clf, task='classification', k_best=10, n_rfe=6) # k_best > num actual features from OHE
        
        # 使用预处理器对象进行模型训练 (更推荐的方式，确保测试集得到相同转换)
        # 注意: select_features_sklearn 返回的是已选择特征的DataFrame，它本身不再需要ColumnTransformer处理其内部的特征
        # 如果要在Pipeline中使用特征选择，需要将select_features_sklearn中的逻辑包装成一个Transformer
        # 这里为了API演示的清晰性，我们分步进行，然后传入已选择/处理好的X_clf_selected
        # 如果preprocessor_clf 用于 full_pipeline, 那么X_train/X_test应该是 X_clf_raw
        
        # 演示1: Logistic Regression with preprocessor_clf on raw data
        print("\n>>> 模型1: Logistic Regression (使用ColumnTransformer Pipeline) <<<")
        lr_pipeline, lr_metrics = train_evaluate_model(X_clf_raw, y_clf, task='classification', 
                                                       model_type='logistic_regression', 
                                                       preprocessor_obj=preprocessor_clf, # 传入原始数据和预处理器
                                                       random_state=42, model_params={'max_iter': 200})
        if lr_pipeline:
            conceptual_hyperparameter_tuning(lr_pipeline, X_clf_raw, y_clf, task='classification')

        # 演示2: Random Forest on ALREADY preprocessed and feature-selected data
        # 这种情况下，train_evaluate_model内部的preprocessor_obj应为None
        print("\n\n>>> 模型2: Random Forest (在已手动处理和选择的数据上) <<<")
        rf_pipeline_manual, rf_metrics_manual = train_evaluate_model(X_clf_selected, y_clf, task='classification', 
                                                                  model_type='random_forest', 
                                                                  preprocessor_obj=None, # 数据已处理
                                                                  random_state=42, model_params={'n_estimators': 100})
        # 对于已经手动处理的数据，如果想做超参数调优，需要创建一个只包含模型的pipeline
        if rf_pipeline_manual:
             simple_rf_pipeline = Pipeline([('model', rf_pipeline_manual.named_steps['model'])]) # 从手动流程中提取模型
             conceptual_hyperparameter_tuning(simple_rf_pipeline, X_clf_selected, y_clf, task='classification')

        # PCA 演示 (在预处理后的数据上)
        X_clf_pca, pca_model_clf = apply_pca_sklearn(X_clf_processed, n_components=0.90, random_state=42)
        if X_clf_pca is not None:
            print(f"PCA降维后的数据形状 (分类): {X_clf_pca.shape}")
            # 可以在X_clf_pca上训练模型作为对比

    # --- 回归任务演示 ---
    print("\n\n======== 回归任务演示 ========")
    X_reg_raw, y_reg, num_cols_reg, cat_cols_reg = create_sklearn_sample_data(
        n_samples=200, n_features=6, n_informative=4, task='regression', random_state=123
    )
    X_reg_processed, preprocessor_reg = preprocess_data(X_reg_raw, num_cols_reg, cat_cols_reg, scaler_type='minmax')
    
    if X_reg_processed is not None and not X_reg_processed.empty:
        X_reg_selected, sel_reg_names = select_features_sklearn(X_reg_processed, y_reg, task='regression', k_best=8, n_rfe=4)

        # 演示1: Linear Regression with preprocessor_reg on raw data
        print("\n>>> 模型1: Linear Regression (使用ColumnTransformer Pipeline) <<<")
        linreg_pipeline, linreg_metrics = train_evaluate_model(X_reg_raw, y_reg, task='regression', 
                                                               model_type='linear_regression',
                                                               preprocessor_obj=preprocessor_reg,
                                                               random_state=123)
        # Linear Regression 通常没有太多超参数可调，或其调优不通过GridSearchCV

        # 演示2: Random Forest Regressor on ALREADY preprocessed and feature-selected data
        print("\n\n>>> 模型2: Random Forest Regressor (在已手动处理和选择的数据上) <<<")
        rf_reg_pipeline_manual, rf_reg_metrics_manual = train_evaluate_model(X_reg_selected, y_reg, task='regression',
                                                                             model_type='random_forest',
                                                                             preprocessor_obj=None, # 数据已处理
                                                                             random_state=123, model_params={'n_estimators': 80})
        if rf_reg_pipeline_manual:
            simple_rf_reg_pipeline = Pipeline([('model', rf_reg_pipeline_manual.named_steps['model'])])
            conceptual_hyperparameter_tuning(simple_rf_reg_pipeline, X_reg_selected, y_reg, task='regression')

        # PCA 演示 (回归)
        X_reg_pca, pca_model_reg = apply_pca_sklearn(X_reg_processed, n_components=3, random_state=123) # 固定3个主成分
        if X_reg_pca is not None:
            print(f"PCA降维后的数据形状 (回归): {X_reg_pca.shape}")

    print("\n\n========== Scikit-learn核心功能演示完成 ==========")
    print(f"所有图表和潜在输出已保存到 '{OUTPUT_DIR}' 目录 (如果适用)。")

if __name__ == '__main__':
    run_sklearn_demos() 