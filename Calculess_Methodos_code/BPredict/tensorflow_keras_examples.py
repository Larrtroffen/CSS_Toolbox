import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping # ModelCheckpoint can be added if h5py is standard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from sklearn.datasets import make_classification, make_regression # Replaced with custom
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, mean_squared_error, log_loss
import matplotlib.pyplot as plt
import os
import json # For saving metrics

# 确保TensorFlow日志级别 (减少冗余信息)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')

# 定义输出目录
OUTPUT_DIR = "tensorflow_keras_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建目录: {OUTPUT_DIR}")

def create_tf_keras_sample_data_api(
    n_samples: int = 500, 
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
    创建包含数值和类别特征（含NaN）的TensorFlow/Keras演示数据集。

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

    # 生成数值特征
    if task == 'classification':
        # 使用更受控的方式生成分类数据
        means = np.random.rand(n_classes, n_informative_num) * 3 # 调整均值范围
        covs = [np.eye(n_informative_num) * (np.random.rand()*0.5 + 0.3) for _ in range(n_classes)] #调整协方差
        X_list, y_list = [], []
        samples_per_class = n_samples // n_classes
        for i in range(n_classes):
            X_class = np.random.multivariate_normal(means[i], covs[i], samples_per_class)
            X_list.append(X_class)
            y_list.extend([i] * samples_per_class)
        X_num_informative = np.vstack(X_list)
        y = np.array(y_list)
        if len(y) < n_samples:
            remaining = n_samples - len(y)
            X_rem = np.random.multivariate_normal(means[0], covs[0], remaining)
            X_num_informative = np.vstack([X_num_informative, X_rem])
            y = np.concatenate([y, [0]*remaining])
        if n_num_features > n_informative_num:
            X_num_other = np.random.randn(n_samples, n_num_features - n_informative_num) * 0.5 #减少噪声特征影响
            X_num = np.hstack((X_num_informative, X_num_other))
        else:
            X_num = X_num_informative
    else: # 回归
        coeffs = np.random.rand(n_informative_num) * 8 - 4
        X_num = np.random.rand(n_samples, n_num_features) * 5
        y = np.dot(X_num[:, :n_informative_num], coeffs) + np.random.normal(0, 1.5, n_samples)

    df = pd.DataFrame(X_num, columns=[f'num_feat_{i}' for i in range(n_num_features)])
    
    # 类别特征
    cat_base_options = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta']
    for i in range(n_cat_features):
        cat_name = f'cat_feat_{chr(ord("A")+i)}'
        num_unique_this_cat = np.random.randint(2, 5) # 每列2-4个唯一类别
        choices = np.random.choice(cat_base_options, num_unique_this_cat, replace=False).tolist()
        df[cat_name] = np.random.choice(choices, n_samples)

    # 引入NaN
    if n_samples > 0:
        for col_idx in range(n_num_features):
            nan_indices = np.random.choice(df.index, size=int(n_samples * nan_percentage_num), replace=False)
            df.iloc[nan_indices, col_idx] = np.nan
        for col_name in df.select_dtypes(include='object').columns:
            nan_indices = np.random.choice(df.index, size=int(n_samples * nan_percentage_cat), replace=False)
            df.loc[nan_indices, col_name] = np.nan # Keras OneHotEncoder 通常能处理 NaN
            
    print(f"创建 TF/Keras {task} 数据集: {df.shape[0]} 样本, {df.shape[1]} 特征。")
    print(f"目标变量形状: {y.shape}, 总 NaN 数量: {df.isnull().sum().sum()}")
    return df, pd.Series(y, name='target')

def create_keras_preprocessor_api(X_df_train: pd.DataFrame) -> ColumnTransformer:
    """
    创建并拟合一个Scikit-learn ColumnTransformer用于预处理数值和类别特征。
    包含对NaN的插补。

    参数:
    - X_df_train (pd.DataFrame): 用于拟合预处理器的训练特征DataFrame。

    返回:
    - ColumnTransformer: 拟合好的预处理器。
    """
    numeric_features = X_df_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_df_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # 数值特征使用中位数插补
        ('scaler', StandardScaler()) # 然后进行标准化
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # 类别特征使用众数插补
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # 保留其他未指定列 (如果有)
    )
    
    print("创建预处理器...")
    preprocessor.fit(X_df_train) # 在训练数据上拟合
    print("预处理器拟合完成。")
    # 打印转换后的特征名 (如果需要调试)
    # try:
    #     feature_names_out = preprocessor.get_feature_names_out()
    #     print(f"  转换后的特征名 (部分): {feature_names_out[:5]}...")
    # except Exception as e:
    #     print(f"  获取转换后特征名失败: {e}")
    return preprocessor

def build_keras_sequential_model_api(
    input_dim: int, 
    task: str = 'classification', 
    num_classes: int | None = None,
    layers_config: list[dict] | None = None, # 例如: [{'units': 128, 'activation': 'relu', 'dropout': 0.3, 'batch_norm': True}, ...]
    optimizer_config: dict | None = None # 例如: {'name': 'adam', 'learning_rate': 0.001}
) -> keras.Model:
    """
    构建并编译一个Keras序贯模型。

    参数:
    - input_dim (int): 输入层维度 (预处理后特征数量)。
    - task (str): 'classification' 或 'regression'。
    - num_classes (int | None): (仅分类) 类别数。
    - layers_config (list[dict] | None): 模型层配置列表。如果None，使用默认配置。
    - optimizer_config (dict | None): 优化器配置。如果None，使用默认Adam。

    返回:
    - keras.Model: 编译好的Keras模型。
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,), name="input_layer")) 

    if layers_config is None:
        # 默认层配置
        layers_config = [
            {'units': 128, 'activation': 'relu', 'dropout': 0.3, 'batch_norm': True},
            {'units': 64, 'activation': 'relu', 'dropout': 0.2, 'batch_norm': True}
        ]

    for config in layers_config:
        model.add(layers.Dense(units=config['units'], activation=config['activation']))
        if config.get('batch_norm', False):
            model.add(layers.BatchNormalization())
        if config.get('dropout', 0) > 0:
            model.add(layers.Dropout(config['dropout']))

    # 输出层
    if task == 'classification':
        if num_classes is None:
            raise ValueError("分类任务需要指定 num_classes。")
        if num_classes == 2 or (num_classes == 1 and task == 'classification'): # 二分类
            model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else: # 多分类
            model.add(layers.Dense(num_classes, activation='softmax', name='output_layer'))
            loss = 'categorical_crossentropy' # 目标y需要是one-hot编码
            metrics = ['accuracy'] # AUC for multiclass often needs specific setup
    else: # 回归
        model.add(layers.Dense(1, name='output_layer')) # 线性输出
        loss = 'mean_squared_error'
        metrics = ['mae', 'mse', keras.metrics.RootMeanSquaredError(name='rmse')]

    # 优化器
    if optimizer_config is None:
        optimizer_config = {'name': 'adam', 'learning_rate': 0.001}
    
    opt_name = optimizer_config.get('name', 'adam').lower()
    lr = optimizer_config.get('learning_rate', 0.001)
    if opt_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    else:
        print(f"警告: 未知优化器 '{opt_name}'，使用默认Adam。")
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(f"Keras {task} 模型已构建并编译。")
    model.summary() # 打印模型结构
    return model

def train_evaluate_keras_model_api(
    model: keras.Model, 
    X_train_processed: np.ndarray, 
    y_train_prepared: np.ndarray, 
    X_test_processed: np.ndarray, 
    y_test_prepared: np.ndarray,
    task: str, # 'classification' or 'regression'
    epochs: int = 50, 
    batch_size: int = 32, 
    early_stopping_patience: int | None = 10,
    validation_split: float = 0.15, # 在训练集上划分验证集
    num_classes_for_eval: int | None = None # 仅用于分类评估时的类别数
) -> tuple[keras.callbacks.History, dict]:
    """
    训练并评估一个已编译的Keras模型。

    参数:
    - model (keras.Model): 已编译的Keras模型。
    - X_train_processed (np.ndarray): 预处理后的训练特征。
    - y_train_prepared (np.ndarray): 准备好的训练目标 (可能已one-hot编码)。
    - X_test_processed (np.ndarray): 预处理后的测试特征。
    - y_test_prepared (np.ndarray): 准备好的测试目标。
    - task (str): 'classification' 或 'regression'。
    - epochs (int): 训练轮数。
    - batch_size (int): 批大小。
    - early_stopping_patience (int | None): 早停的耐心轮数。如果None则不使用。
    - validation_split (float): 从训练数据中用于验证的比例。
    - num_classes_for_eval (int | None): 评估分类模型时使用的类别数。

    返回:
    - keras.callbacks.History: 训练历史对象。
    - dict: 在测试集上的评估指标字典。
    """
    callbacks = []
    if early_stopping_patience is not None:
        early_stop = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, 
                                   restore_best_weights=True, verbose=1)
        callbacks.append(early_stop)
    
    # 模型检查点 (可选, 如果需要保存中间模型)
    # checkpoint_path = os.path.join(OUTPUT_DIR, f"keras_{task}_best_model.weights.h5")
    # model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, 
    #                                  monitor='val_loss', mode='min', save_weights_only=True)
    # callbacks.append(model_checkpoint)

    print(f"开始训练 Keras {task} 模型 (共 {epochs} 轮)...")
    history = model.fit(
        X_train_processed,
        y_train_prepared,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1 # 0=silent, 1=progress bar, 2=one line per epoch
    )
    print("模型训练完成。")

    # 在测试集上评估
    print("\n在测试集上评估模型:")
    eval_results = model.evaluate(X_test_processed, y_test_prepared, verbose=0, return_dict=True)
    print(f"  测试集评估结果 (来自 model.evaluate): {eval_results}")

    # 计算并保存更详细的指标
    y_pred_raw = model.predict(X_test_processed, verbose=0)
    detailed_metrics = eval_results.copy() # 从eval_results开始

    if task == 'classification':
        if num_classes_for_eval is None:
            raise ValueError("分类任务评估需要 num_classes_for_eval。")
        
        # y_test_prepared 对于二分类可能是 (n_samples,), 多分类是 (n_samples, n_classes)
        # y_pred_raw 对于二分类是 (n_samples, 1), 多分类是 (n_samples, n_classes)
        if num_classes_for_eval == 2 or (num_classes_for_eval == 1 and y_pred_raw.shape[1] == 1):
            y_pred_classes_manual = (y_pred_raw > 0.5).astype(int).flatten()
            y_true_classes_manual = y_test_prepared.flatten() if y_test_prepared.ndim > 1 else y_test_prepared
            detailed_metrics['accuracy_manual'] = accuracy_score(y_true_classes_manual, y_pred_classes_manual)
            try:
                detailed_metrics['roc_auc_manual'] = roc_auc_score(y_true_classes_manual, y_pred_raw.flatten()) 
            except ValueError as e_auc:
                print(f"计算ROC AUC失败: {e_auc}") # 例如测试集中只有一个类别
            detailed_metrics['logloss_manual'] = log_loss(y_true_classes_manual, y_pred_raw.flatten())
        else: # 多分类
            y_pred_classes_manual = np.argmax(y_pred_raw, axis=1)
            y_true_classes_manual = np.argmax(y_test_prepared, axis=1) # 假设y_test_prepared是one-hot
            detailed_metrics['accuracy_manual'] = accuracy_score(y_true_classes_manual, y_pred_classes_manual)
            try:
                detailed_metrics['roc_auc_ovr_manual'] = roc_auc_score(y_true_classes_manual, y_pred_raw, multi_class='ovr')
            except ValueError as e_auc_mc:
                 print(f"计算多分类ROC AUC (OVR)失败: {e_auc_mc}")
            detailed_metrics['logloss_manual'] = log_loss(y_true_classes_manual, y_pred_raw)

    else: # 回归
        detailed_metrics['r2_score_manual'] = r2_score(y_test_prepared.flatten(), y_pred_raw.flatten())
        # MSE 和 RMSE 通常已在 eval_results 中
        if 'mse' not in detailed_metrics: detailed_metrics['mse_manual'] = mean_squared_error(y_test_prepared.flatten(), y_pred_raw.flatten())
        if 'rmse' not in detailed_metrics: detailed_metrics['rmse_manual'] = np.sqrt(detailed_metrics.get('mse_manual', mean_squared_error(y_test_prepared.flatten(), y_pred_raw.flatten())))

    print(f"  详细测试集指标: {detailed_metrics}")
    
    # 保存模型 (整个模型或仅权重)
    model_save_path = os.path.join(OUTPUT_DIR, f"keras_{task}_model.keras") # Keras v3推荐格式
    model.save(model_save_path)
    print(f"  Keras模型已保存到: {model_save_path}")
    
    # 保存指标到JSON
    metrics_path = os.path.join(OUTPUT_DIR, f"keras_{task}_metrics.json")
    with open(metrics_path, 'w') as f:
        # 转换numpy类型为Python原生类型以进行JSON序列化
        serializable_metrics = {k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) for k,v in detailed_metrics.items()}
        json.dump(serializable_metrics, f, indent=4)
    print(f"  评估指标已保存到: {metrics_path}")

    # 绘制和保存训练历史图
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练损失 (Train Loss)')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='验证损失 (Validation Loss)')
    plt.title(f'Keras 模型损失 ({task.capitalize()})')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    loss_plot_path = os.path.join(OUTPUT_DIR, f"keras_{task}_loss_history.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"  损失历史图已保存: {loss_plot_path}")

    if 'accuracy' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='训练准确率 (Train Accuracy)')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='验证准确率 (Validation Accuracy)')
        plt.title(f'Keras 模型准确率 ({task.capitalize()})')
        plt.xlabel('轮次 (Epoch)')
        plt.ylabel('准确率 (Accuracy)')
        plt.legend()
        acc_plot_path = os.path.join(OUTPUT_DIR, f"keras_{task}_accuracy_history.png")
        plt.savefig(acc_plot_path)
        plt.close()
        print(f"  准确率历史图已保存: {acc_plot_path}")

    return history, detailed_metrics

def explain_keras_functional_api_conceptual() -> None:
    """打印Keras Functional API的概念解释。"""
    print("\n--- 概念: Keras Functional API --- ")
    print("Keras Functional API 允许构建更灵活和复杂的模型结构，例如多输入、多输出模型，或共享层的模型。")
    print("它通过将层视为可调用对象并直接连接张量来工作。")
    print("\n基本结构示例:")
    print("  # 定义输入张量")
    print("  input_tensor = keras.Input(shape=(原始特征数量,), name='input')")
    print("")
    print("  # 第一个处理分支 (例如处理数值特征)")
    print("  # dense_branch_1 = layers.Dense(64, activation='relu')(input_tensor) # 假设输入已预处理")
    print("  # dense_branch_1 = layers.Dense(32, activation='relu')(dense_branch_1)")
    print("")
    print("  # 假设有另一个输入或分支 (例如处理文本或图像特征，或不同的数值特征子集)")
    print("  # input_tensor_aux = keras.Input(shape=(辅助特征数量,), name='aux_input')")
    print("  # dense_branch_aux = layers.Dense(32, activation='relu')(input_tensor_aux)")
    print("")
    print("  # 合并分支 (如果需要)")
    print("  # merged = layers.concatenate([dense_branch_1, dense_branch_aux])")
    print("")
    print("  # 在合并的特征上添加更多层")
    print("  # final_dense = layers.Dense(64, activation='relu')(merged if 'merged' in locals() else input_tensor)")
    print("  # final_dropout = layers.Dropout(0.5)(final_dense)")
    print("")
    print("  # 定义输出层")
    print("  # output_tensor = layers.Dense(1, activation='sigmoid', name='main_output')(final_dropout if 'final_dropout' in locals() else input_tensor)")
    print("")
    print("  # 创建模型，指定输入和输出")
    print("  # model = keras.Model(inputs=[input_tensor, input_tensor_aux] if 'input_tensor_aux' in locals() else input_tensor, outputs=output_tensor)")
    print("")
    print("  # 编译模型如常")
    print("  # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])")
    print("\nFunctional API 的强大之处在于其处理非线性拓扑、共享层以及多输入/输出的能力。")
    print("对于表格数据，如果需要对不同类型的特征（例如数值、类别、文本）应用非常不同的预处理和网络路径，它会非常有用。")


if __name__ == '__main__':
    print("===== TensorFlow/Keras API化功能演示 =====")
    main_seed = 888
    tf.random.set_seed(main_seed) # 设置TensorFlow的全局随机种子
    np.random.seed(main_seed)    # 设置Numpy的随机种子

    # --- 1. 分类任务演示 ---
    print("\n\n*** 1. Keras 分类任务演示 ***")
    X_df_clf, y_s_clf = create_tf_keras_sample_data_api(
        n_samples=600, n_features=12, n_informative_num=6, n_cat_features=3,
        n_classes=3, task='classification', nan_percentage_num=0.07, nan_percentage_cat=0.06,
        random_state=main_seed
    )
    
    # 1.1 数据准备与预处理
    num_unique_classes_clf = y_s_clf.nunique()
    if num_unique_classes_clf > 2: # 多分类，需要one-hot编码目标变量
        y_s_clf_prepared = keras.utils.to_categorical(y_s_clf, num_classes=num_unique_classes_clf)
    else: # 二分类
        y_s_clf_prepared = y_s_clf.copy() # 无需one-hot

    X_train_clf_df, X_test_clf_df, y_train_clf, y_test_clf = train_test_split(
        X_df_clf, y_s_clf_prepared, test_size=0.2, random_state=main_seed,
        stratify=y_s_clf if num_unique_classes_clf > 1 else None # Stratify on original y before one-hot
    )

    preprocessor_clf = create_keras_preprocessor_api(X_train_clf_df)
    X_train_clf_processed = preprocessor_clf.transform(X_train_clf_df)
    X_test_clf_processed = preprocessor_clf.transform(X_test_clf_df)
    print(f"  分类数据预处理后: 训练集形状 {X_train_clf_processed.shape}, 测试集形状 {X_test_clf_processed.shape}")

    # 1.2 构建并训练分类模型
    clf_model_layers = [
        {'units': 96, 'activation': 'relu', 'dropout': 0.25, 'batch_norm': True},
        {'units': 48, 'activation': 'relu', 'dropout': 0.15, 'batch_norm': True}
    ]
    clf_optimizer = {'name': 'adam', 'learning_rate': 0.0015}

    keras_clf_model = build_keras_sequential_model_api(
        input_dim=X_train_clf_processed.shape[1],
        task='classification',
        num_classes=num_unique_classes_clf,
        layers_config=clf_model_layers,
        optimizer_config=clf_optimizer
    )
    
    clf_history, clf_metrics = train_evaluate_keras_model_api(
        model=keras_clf_model,
        X_train_processed=X_train_clf_processed, y_train_prepared=y_train_clf,
        X_test_processed=X_test_clf_processed, y_test_prepared=y_test_clf,
        task='classification', 
        epochs=40, # 减少轮数以加速演示
        batch_size=64, 
        early_stopping_patience=7,
        num_classes_for_eval=num_unique_classes_clf
    )
    print(f"  Keras分类模型训练历史keys: {clf_history.history.keys()}")
    print(f"  Keras分类模型评估指标: {clf_metrics}")


    # --- 2. 回归任务演示 ---
    print("\n\n*** 2. Keras 回归任务演示 ***")
    X_df_reg, y_s_reg = create_tf_keras_sample_data_api(
        n_samples=550, n_features=10, n_informative_num=5, n_cat_features=2,
        task='regression', nan_percentage_num=0.04, nan_percentage_cat=0.05,
        random_state=main_seed + 1
    )
    
    # 2.1 回归数据准备与预处理
    X_train_reg_df, X_test_reg_df, y_train_reg, y_test_reg = train_test_split(
        X_df_reg, y_s_reg, test_size=0.2, random_state=main_seed + 1
    )
    preprocessor_reg = create_keras_preprocessor_api(X_train_reg_df)
    X_train_reg_processed = preprocessor_reg.transform(X_train_reg_df)
    X_test_reg_processed = preprocessor_reg.transform(X_test_reg_df)
    print(f"  回归数据预处理后: 训练集形状 {X_train_reg_processed.shape}, 测试集形状 {X_test_reg_processed.shape}")

    # 2.2 构建并训练回归模型
    reg_model_layers = [
        {'units': 80, 'activation': 'relu', 'dropout': 0.2, 'batch_norm': False},
        {'units': 40, 'activation': 'relu', 'dropout': 0.1, 'batch_norm': False}
    ]
    reg_optimizer = {'name': 'rmsprop', 'learning_rate': 0.001}

    keras_reg_model = build_keras_sequential_model_api(
        input_dim=X_train_reg_processed.shape[1],
        task='regression',
        layers_config=reg_model_layers,
        optimizer_config=reg_optimizer
    )

    reg_history, reg_metrics = train_evaluate_keras_model_api(
        model=keras_reg_model,
        X_train_processed=X_train_reg_processed, y_train_prepared=y_train_reg.values, # .values to ensure numpy array
        X_test_processed=X_test_reg_processed, y_test_prepared=y_test_reg.values,
        task='regression',
        epochs=45, # 减少轮数
        batch_size=32,
        early_stopping_patience=8
    )
    print(f"  Keras回归模型训练历史keys: {reg_history.history.keys()}")
    print(f"  Keras回归模型评估指标: {reg_metrics}")


    # --- 3. Keras Functional API 概念解释 ---
    explain_keras_functional_api_conceptual()

    print("\n\n===== TensorFlow/Keras API化功能演示完成 =====")
    print(f"所有输出 (模型, 指标JSON, 图表) 保存在 '{OUTPUT_DIR}' 目录中。")
