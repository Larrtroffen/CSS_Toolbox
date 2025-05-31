import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # OneHotEncoder can be complex with PyTorch without a library
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, mean_squared_error, log_loss
import os
import json

# 定义输出目录
OUTPUT_DIR = "pytorch_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建目录: {OUTPUT_DIR}")

# 设置PyTorch的默认设备 (如果可用，使用GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch 将使用设备: {device}")

def create_pytorch_sample_data_api(
    n_samples: int = 500, 
    n_features: int = 10, 
    n_informative_num: int = 5, 
    n_cat_features: int = 2,
    n_classes: int = 2, 
    task: str = 'classification', 
    nan_percentage_num: float = 0.05,
    nan_percentage_cat: float = 0.05, # Note: PyTorch basic handling of NaNs for cats is manual
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    创建包含数值和类别特征（含NaN）的PyTorch演示数据集。
    类别特征将是字符串，需要后续编码。

    参数与返回值同 tensorflow_keras_examples.py 中的对应函数。
    """
    np.random.seed(random_state)
    n_num_features = n_features - n_cat_features
    if n_num_features < 0: raise ValueError("总特征数需 >= 类别特征数。")
    if n_informative_num > n_num_features: n_informative_num = n_num_features

    if task == 'classification':
        means = np.random.rand(n_classes, n_informative_num) * 2.5
        covs = [np.eye(n_informative_num) * (np.random.rand()*0.4 + 0.2) for _ in range(n_classes)]
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
            X_num_other = np.random.randn(n_samples, n_num_features - n_informative_num) * 0.4
            X_num = np.hstack((X_num_informative, X_num_other))
        else: X_num = X_num_informative
    else: # 回归
        coeffs = np.random.rand(n_informative_num) * 7 - 3.5
        X_num = np.random.rand(n_samples, n_num_features) * 6
        y = np.dot(X_num[:, :n_informative_num], coeffs) + np.random.normal(0, 1.2, n_samples)

    df = pd.DataFrame(X_num, columns=[f'num_feat_{i}' for i in range(n_num_features)])
    cat_base_options = ['TypeX', 'TypeY', 'TypeZ', 'TypeW', 'TypeV']
    for i in range(n_cat_features):
        cat_name = f'cat_feat_str_{chr(ord("A")+i)}' # 明确为字符串类别
        num_unique_this_cat = np.random.randint(2, 4)
        choices = np.random.choice(cat_base_options, num_unique_this_cat, replace=False).tolist()
        df[cat_name] = np.random.choice(choices, n_samples)

    if n_samples > 0:
        for col_idx in range(n_num_features):
            if nan_percentage_num > 0:
                nan_indices = np.random.choice(df.index, size=int(n_samples * nan_percentage_num), replace=False)
                df.iloc[nan_indices, col_idx] = np.nan
        for col_name in df.select_dtypes(include='object').columns:
            if nan_percentage_cat > 0:
                nan_indices = np.random.choice(df.index, size=int(n_samples * nan_percentage_cat), replace=False)
                df.loc[nan_indices, col_name] = pd.NA # 使用pd.NA，后续imputer处理
            
    print(f"创建 PyTorch {task} 数据集: {df.shape[0]} 样本, {df.shape[1]} 特征。")
    print(f"目标变量形状: {y.shape}, 总 NaN/NA 数量: {df.isnull().sum().sum()}")
    return df, pd.Series(y, name='target')

class TabularDataset(Dataset):
    """自定义PyTorch Dataset用于处理表格数据 (数值特征)。"""
    def __init__(self, X_data: np.ndarray, y_data: np.ndarray | None = None, task: str = 'classification'):
        self.X_data = torch.tensor(X_data, dtype=torch.float32)
        self.task = task
        if y_data is not None:
            if task == 'classification':
                 # 假设y_data是类别索引 (0, 1, 2...)
                self.y_data = torch.tensor(y_data, dtype=torch.long) 
            else: # 回归
                self.y_data = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1) # 保持形状 [n_samples, 1]
        else:
            self.y_data = None

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.y_data is not None:
            return self.X_data[idx], self.y_data[idx]
        return self.X_data[idx]

class PyTorchTabularModel(nn.Module):
    """一个简单的PyTorch神经网络模型用于表格数据。"""
    def __init__(self, input_dim: int, output_dim: int, layers_config: list[dict] | None = None):
        """
        参数:
        - input_dim (int): 输入特征数量。
        - output_dim (int): 输出维度 (例如，1用于二分类/回归, n_classes用于多分类)。
        - layers_config (list[dict] | None): 层配置, e.g., [{'units': 100, 'dropout': 0.2}, ...]
                                            如果None，使用默认结构。
        """
        super().__init__()
        model_layers = []
        current_dim = input_dim

        if layers_config is None:
            layers_config = [
                {'units': 128, 'dropout': 0.25},
                {'units': 64, 'dropout': 0.15}
            ]
        
        for i, config in enumerate(layers_config):
            model_layers.append(nn.Linear(current_dim, config['units']))
            model_layers.append(nn.ReLU())
            # Batch Norm可以加在这里，但为了简化，暂不添加
            # model_layers.append(nn.BatchNorm1d(config['units'])) 
            if config.get('dropout', 0) > 0:
                model_layers.append(nn.Dropout(config['dropout']))
            current_dim = config['units']
        
        model_layers.append(nn.Linear(current_dim, output_dim))
        # 输出层的激活函数 (例如Sigmoid或Softmax) 通常在损失函数中处理 (如BCEWithLogitsLoss)
        # 或者在推理时手动应用。

        self.network = nn.Sequential(*model_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def train_pytorch_model_api(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader | None, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    task: str,
    num_classes: int | None = None # 仅用于分类任务的指标计算
) -> tuple[nn.Module, list[dict]]:
    """
    训练一个PyTorch模型。

    参数:
    - model (nn.Module): PyTorch模型。
    - train_loader (DataLoader): 训练数据加载器。
    - val_loader (DataLoader | None): 验证数据加载器 (可选)。
    - criterion (nn.Module): 损失函数。
    - optimizer (optim.Optimizer): 优化器。
    - epochs (int): 训练轮数。
    - task (str): 'classification' 或 'regression'。
    - num_classes (int | None): 分类任务的类别数，用于评估。

    返回:
    - nn.Module: 训练好的模型。
    - list[dict]: 每轮的训练和验证指标历史。
    """
    model.to(device) # 将模型移到设备
    history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 10 # 简单的早停

    print(f"开始训练PyTorch {task} 模型，共 {epochs} 轮...")
    for epoch in range(epochs):
        model.train() # 设置为训练模式
        running_loss = 0.0
        train_preds_all, train_targets_all = [], []

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if task == 'classification':
                train_preds_all.extend(outputs.detach().cpu().numpy())
                train_targets_all.extend(targets.cpu().numpy())
            # 对于回归，也可以收集outputs和targets来计算epoch结束时的R2/MSE

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_metrics = {'epoch': epoch + 1, 'train_loss': epoch_loss}

        # 计算训练集上的指标 (如果需要)
        if task == 'classification' and train_preds_all:
            train_preds_np = np.array(train_preds_all)
            train_targets_np = np.array(train_targets_all)
            if num_classes is not None and num_classes > 2: # 多分类
                train_acc = accuracy_score(train_targets_np, np.argmax(train_preds_np, axis=1))
            else: # 二分类 (假设输出是logits, 需要sigmoid)
                train_acc = accuracy_score(train_targets_np, (torch.sigmoid(torch.tensor(train_preds_np)).numpy() > 0.5).astype(int))
            epoch_metrics['train_accuracy'] = train_acc

        # 验证步骤
        val_loss = None
        if val_loader:
            model.eval() # 设置为评估模式
            running_val_loss = 0.0
            val_preds_all, val_targets_all = [], []
            with torch.no_grad():
                for inputs_val, targets_val in val_loader:
                    inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, targets_val)
                    running_val_loss += loss_val.item() * inputs_val.size(0)
                    if task == 'classification':
                        val_preds_all.extend(outputs_val.cpu().numpy())
                        val_targets_all.extend(targets_val.cpu().numpy())
            
            val_loss = running_val_loss / len(val_loader.dataset)
            epoch_metrics['val_loss'] = val_loss

            if task == 'classification' and val_preds_all:
                val_preds_np = np.array(val_preds_all)
                val_targets_np = np.array(val_targets_all)
                if num_classes is not None and num_classes > 2: # 多分类, Softmax output from model if criterion is CrossEntropyLoss
                    val_acc = accuracy_score(val_targets_np, np.argmax(val_preds_np, axis=1))
                else: # 二分类, Sigmoid output from model or apply sigmoid to logits
                    val_acc = accuracy_score(val_targets_np, (torch.sigmoid(torch.tensor(val_preds_np)).numpy() > 0.5).astype(int))
                epoch_metrics['val_accuracy'] = val_acc
            
            # 简单的早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'pytorch_{task}_best_model.pth')) # 保存最佳模型
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"早停触发于轮次 {epoch+1}。最佳验证损失: {best_val_loss:.4f}")
                    # model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'pytorch_{task}_best_model.pth'))) # 加载最佳模型
                    history.append(epoch_metrics)
                    break
        
        history.append(epoch_metrics)
        print(f"轮次 [{epoch+1}/{epochs}] - "
              f"训练损失: {epoch_loss:.4f}" +
              (f", 训练准确率: {epoch_metrics.get('train_accuracy', 0):.4f}" if 'train_accuracy' in epoch_metrics else "") +
              (f", 验证损失: {val_loss:.4f}" if val_loss is not None else "") +
              (f", 验证准确率: {epoch_metrics.get('val_accuracy', 0):.4f}" if 'val_accuracy' in epoch_metrics else ""))

    print("模型训练完成。")
    return model, history

def evaluate_pytorch_model_api(
    model: nn.Module, 
    test_loader: DataLoader, 
    criterion: nn.Module, 
    task: str, 
    num_classes: int | None = None
) -> dict:
    """
    在测试集上评估PyTorch模型。

    参数:
    - model (nn.Module): 训练好的PyTorch模型。
    - test_loader (DataLoader): 测试数据加载器。
    - criterion (nn.Module): 损失函数 (用于计算测试损失)。
    - task (str): 'classification' 或 'regression'。
    - num_classes (int | None): 分类任务的类别数。

    返回:
    - dict: 包含评估指标的字典。
    """
    model.to(device)
    model.eval() # 设置为评估模式
    test_loss = 0.0
    all_targets = []
    all_preds_proba = [] # 用于分类的概率或回归的直接输出

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            all_targets.extend(targets.cpu().numpy())
            if task == 'classification':
                # 对于BCEWithLogitsLoss，输出是logits，需要sigmoid
                # 对于CrossEntropyLoss，输出是logits，需要softmax (但通常argmax就够了)
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    all_preds_proba.extend(torch.sigmoid(outputs).cpu().numpy())
                elif isinstance(criterion, nn.CrossEntropyLoss):
                     all_preds_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy()) # 获取概率
                else: # 其他情况，直接使用输出 (可能需要调整)
                    all_preds_proba.extend(outputs.cpu().numpy())
            else: # 回归
                all_preds_proba.extend(outputs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    metrics = {'test_loss': test_loss}
    targets_np = np.array(all_targets)
    preds_proba_np = np.array(all_preds_proba)

    if task == 'classification':
        if num_classes is None: raise ValueError("分类评估需要 num_classes")
        
        if num_classes == 2 or (num_classes == 1 and preds_proba_np.shape[1] if preds_proba_np.ndim > 1 else True):
            # 二分类, preds_proba_np 已经是概率 (0-1)
            preds_classes_np = (preds_proba_np > 0.5).astype(int).flatten()
            metrics['accuracy'] = accuracy_score(targets_np.flatten(), preds_classes_np)
            try:
                metrics['roc_auc'] = roc_auc_score(targets_np.flatten(), preds_proba_np.flatten() if preds_proba_np.ndim ==1 else preds_proba_np[:,0] if preds_proba_np.shape[1]==1 else preds_proba_np[:,1])
            except ValueError as e_auc: print(f"计算ROC AUC失败: {e_auc}")
            metrics['logloss'] = log_loss(targets_np.flatten(), preds_proba_np.flatten() if preds_proba_np.ndim ==1 else preds_proba_np)
        else: # 多分类
            preds_classes_np = np.argmax(preds_proba_np, axis=1)
            metrics['accuracy'] = accuracy_score(targets_np, preds_classes_np)
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(targets_np, preds_proba_np, multi_class='ovr')
            except ValueError as e_auc_mc: print(f"计算多分类ROC AUC (OVR)失败: {e_auc_mc}")
            metrics['logloss'] = log_loss(targets_np, preds_proba_np)
    else: # 回归
        metrics['r2_score'] = r2_score(targets_np.flatten(), preds_proba_np.flatten())
        metrics['mse'] = mean_squared_error(targets_np.flatten(), preds_proba_np.flatten())
        metrics['rmse'] = np.sqrt(metrics['mse'])

    print(f"  测试集评估指标: {metrics}")
    return metrics

def explain_pytorch_ecosystem_conceptual_api() -> None:
    """打印PyTorch生态中其他相关工具和库的概念性解释。"""
    print("\n--- 概念: PyTorch 生态系统与扩展 --- ")
    print("PyTorch 本身是一个强大的深度学习框架，其生态系统包含许多有用的库和工具，可用于表格数据等任务：")
    print("1. PyTorch Lightning: 一个轻量级的PyTorch包装器，用于简化样板代码，使研究和生产更容易。")
    print("   - 优点: 结构化代码, 易于扩展 (如多GPU训练, 混合精度), 内置常用回调。")
    print("2. Captum: 用于模型可解释性的库 (模型理解)。")
    print("   - 功能: 集成梯度 (Integrated Gradients), Shapley值变体, 激活图等。")
    print("   - 用途: 理解哪些特征对模型的预测最重要。")
    print("3. TorchServe: 一个灵活且易于使用的工具，用于部署PyTorch模型到生产环境。")
    print("   - 特点: 支持模型版本控制, 批量预测, A/B测试, 指标监控。")
    print("4. PyTorch Tabular: 一个专注于表格数据的PyTorch库 (第三方)。")
    print("   - 提供: 易于使用的API来训练多种先进的表格数据模型 (如TabNet, GANDALF, FT-Transformer)。")
    print("   - 优点: 简化了特征工程和模型选择。")
    print("5. ONNX (Open Neural Network Exchange):")
    print("   - 用途: 将PyTorch模型导出为ONNX格式，使其可以在其他框架或硬件加速器上运行。")
    print("   - 增强互操作性。")
    print("\n这些工具可以显著增强使用PyTorch处理表格数据项目的功能、可维护性和部署能力。")


if __name__ == '__main__':
    print("===== PyTorch API化功能演示 =====")
    main_seed = 1010
    torch.manual_seed(main_seed)
    np.random.seed(main_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(main_seed)

    # --- 1. 分类任务演示 ---
    print("\n\n*** 1. PyTorch 分类任务演示 ***")
    X_df_clf_pt, y_s_clf_pt = create_pytorch_sample_data_api(
        n_samples=450, n_features=12, n_informative_num=6, n_cat_features=3,
        n_classes=3, task='classification', nan_percentage_num=0.06, nan_percentage_cat=0.04,
        random_state=main_seed
    )
    
    # 1.1 数据预处理 (手动处理类别特征，然后用ColumnTransformer)
    # 对于PyTorch，通常需要手动将类别特征转换为数值 (例如，标签编码或嵌入层)
    # 这里使用简单的标签编码 + 标准化，嵌入层更高级但复杂
    
    # 识别数值和类别列
    numeric_cols_clf = X_df_clf_pt.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_clf = X_df_clf_pt.select_dtypes(include='object').columns.tolist()

    # 创建预处理器: 数值用插补+标准化，类别用插补+标签编码 (注意：标签编码结果是单列)
    # 为了简单起见，这个示例的预处理器将所有特征输出为数值型，适合简单MLP
    # 如果使用嵌入层，类别特征的处理方式会不同 (直接传入编码后的整数索引)

    # 步骤1: 插补DataFrame中的NaN/pd.NA
    for col in numeric_cols_clf:
        X_df_clf_pt[col].fillna(X_df_clf_pt[col].median(), inplace=True)
    for col in categorical_cols_clf:
        X_df_clf_pt[col].fillna(X_df_clf_pt[col].mode()[0] if not X_df_clf_pt[col].mode().empty else 'Unknown', inplace=True)

    # 步骤2: 类别特征标签编码
    label_encoders_clf = {}
    X_df_clf_pt_encoded = X_df_clf_pt.copy()
    for col in categorical_cols_clf:
        le = LabelEncoder()
        X_df_clf_pt_encoded[col] = le.fit_transform(X_df_clf_pt_encoded[col])
        label_encoders_clf[col] = le
        print(f"  类别特征 '{col}' 已标签编码。类别数: {len(le.classes_)}")
    
    # 步骤3: 数值特征标准化 (现在所有特征都是数值型)
    all_numeric_cols_after_encoding = X_df_clf_pt_encoded.columns.tolist() # 所有列现在都是数值
    scaler_clf = StandardScaler()
    X_clf_pt_processed = scaler_clf.fit_transform(X_df_clf_pt_encoded)
    
    # 目标变量 (分类任务通常是long类型的整数)
    y_clf_pt_prepared = y_s_clf_pt.values # 直接使用numpy array
    num_unique_classes_clf_pt = len(np.unique(y_clf_pt_prepared))

    X_train_clf_pt, X_test_clf_pt, y_train_clf_pt, y_test_clf_pt = train_test_split(
        X_clf_pt_processed, y_clf_pt_prepared, test_size=0.2, random_state=main_seed,
        stratify=y_clf_pt_prepared if num_unique_classes_clf_pt > 1 else None
    )

    train_dataset_clf = TabularDataset(X_train_clf_pt, y_train_clf_pt, task='classification')
    test_dataset_clf = TabularDataset(X_test_clf_pt, y_test_clf_pt, task='classification')
    val_dataset_clf = TabularDataset(X_test_clf_pt, y_test_clf_pt, task='classification') # 用测试集做简单验证

    train_loader_clf = DataLoader(train_dataset_clf, batch_size=64, shuffle=True)
    test_loader_clf = DataLoader(test_dataset_clf, batch_size=64, shuffle=False)
    val_loader_clf = DataLoader(val_dataset_clf, batch_size=64, shuffle=False)

    print(f"  PyTorch分类数据准备完成: 训练集大小 {len(train_loader_clf.dataset)}, 测试集大小 {len(test_loader_clf.dataset)}")

    # 1.2 构建、训练和评估分类模型
    input_dim_clf = X_train_clf_pt.shape[1]
    output_dim_clf = num_unique_classes_clf_pt # 对于CrossEntropyLoss, 输出维度是类别数
    
    clf_model_layers_pt = [
        {'units': 96, 'dropout': 0.3},
        {'units': 48, 'dropout': 0.2}
    ]
    pytorch_clf_model = PyTorchTabularModel(input_dim_clf, output_dim_clf, layers_config=clf_model_layers_pt)
    
    # 损失函数和优化器
    # CrossEntropyLoss 适用于多分类，它内部包含了Softmax
    # 对于二分类，如果模型输出1个logit，用BCEWithLogitsLoss；如果输出2个logits，也用CrossEntropyLoss
    if output_dim_clf == 1 or (output_dim_clf==2 and num_unique_classes_clf_pt <=2): # 二分类的特殊处理 (假设模型输出1个logit)
        criterion_clf = nn.BCEWithLogitsLoss() # 需要模型输出维度为1
        # 如果用BCEWithLogitsLoss，模型输出维度应为1，此处需要调整模型或逻辑
        # 修正：如果num_unique_classes_clf_pt == 2, PyTorchTabularModel输出维度应为1，或用CrossEntropyLoss
        if num_unique_classes_clf_pt == 2 and output_dim_clf == 2:
             print("警告: 二分类但模型输出维度为2，建议使用CrossEntropyLoss或修改模型输出为1并用BCEWithLogitsLoss")
             criterion_clf = nn.CrossEntropyLoss()
        elif num_unique_classes_clf_pt == 2 and output_dim_clf ==1:
             criterion_clf = nn.BCEWithLogitsLoss()
        else: # 多分类
            criterion_clf = nn.CrossEntropyLoss()
    else: # 多分类 (num_classes > 2)
        criterion_clf = nn.CrossEntropyLoss()
        
    optimizer_clf = optim.Adam(pytorch_clf_model.parameters(), lr=0.0015)

    print(f"  分类模型 ({type(pytorch_clf_model).__name__}) 已构建。输入维度: {input_dim_clf}, 输出维度: {output_dim_clf}")
    
    trained_clf_model, clf_history_pt = train_pytorch_model_api(
        pytorch_clf_model, train_loader_clf, val_loader_clf, criterion_clf, optimizer_clf, 
        epochs=30, task='classification', num_classes=num_unique_classes_clf_pt
    )
    
    clf_eval_metrics = evaluate_pytorch_model_api(
        trained_clf_model, test_loader_clf, criterion_clf, task='classification', num_classes=num_unique_classes_clf_pt
    )
    torch.save(trained_clf_model.state_dict(), os.path.join(OUTPUT_DIR, 'pytorch_classification_model.pth'))
    with open(os.path.join(OUTPUT_DIR, 'pytorch_classification_metrics.json'), 'w') as f:
        json.dump(clf_eval_metrics, f, indent=4)
    print(f"  PyTorch分类模型和指标已保存。历史记录条数: {len(clf_history_pt)}")


    # --- 2. 回归任务演示 ---
    print("\n\n*** 2. PyTorch 回归任务演示 ***")
    X_df_reg_pt, y_s_reg_pt = create_pytorch_sample_data_api(
        n_samples=400, n_features=10, n_informative_num=5, n_cat_features=2,
        task='regression', nan_percentage_num=0.05, nan_percentage_cat=0.03,
        random_state=main_seed + 1
    )
    
    # 2.1 回归数据预处理
    numeric_cols_reg = X_df_reg_pt.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_reg = X_df_reg_pt.select_dtypes(include='object').columns.tolist()
    
    for col in numeric_cols_reg:
        X_df_reg_pt[col].fillna(X_df_reg_pt[col].median(), inplace=True)
    for col in categorical_cols_reg:
        X_df_reg_pt[col].fillna(X_df_reg_pt[col].mode()[0] if not X_df_reg_pt[col].mode().empty else 'Unknown', inplace=True)

    label_encoders_reg = {}
    X_df_reg_pt_encoded = X_df_reg_pt.copy()
    for col in categorical_cols_reg:
        le_reg = LabelEncoder()
        X_df_reg_pt_encoded[col] = le_reg.fit_transform(X_df_reg_pt_encoded[col])
        label_encoders_reg[col] = le_reg
    
    scaler_reg = StandardScaler()
    X_reg_pt_processed = scaler_reg.fit_transform(X_df_reg_pt_encoded)
    y_reg_pt_prepared = y_s_reg_pt.values # Numpy array

    X_train_reg_pt, X_test_reg_pt, y_train_reg_pt, y_test_reg_pt = train_test_split(
        X_reg_pt_processed, y_reg_pt_prepared, test_size=0.2, random_state=main_seed + 1
    )

    train_dataset_reg = TabularDataset(X_train_reg_pt, y_train_reg_pt, task='regression')
    test_dataset_reg = TabularDataset(X_test_reg_pt, y_test_reg_pt, task='regression')
    val_dataset_reg = TabularDataset(X_test_reg_pt, y_test_reg_pt, task='regression') # 用测试集做简单验证

    train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
    test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=32, shuffle=False)
    print(f"  PyTorch回归数据准备完成: 训练集大小 {len(train_loader_reg.dataset)}, 测试集大小 {len(test_loader_reg.dataset)}")

    # 2.2 构建、训练和评估回归模型
    input_dim_reg = X_train_reg_pt.shape[1]
    output_dim_reg = 1 # 回归任务输出单个值
    reg_model_layers_pt = [
        {'units': 80, 'dropout': 0.25},
        {'units': 40, 'dropout': 0.1}
    ]
    pytorch_reg_model = PyTorchTabularModel(input_dim_reg, output_dim_reg, layers_config=reg_model_layers_pt)
    criterion_reg = nn.MSELoss() # 均方误差损失
    optimizer_reg = optim.RMSprop(pytorch_reg_model.parameters(), lr=0.001)
    print(f"  回归模型 ({type(pytorch_reg_model).__name__}) 已构建。输入维度: {input_dim_reg}, 输出维度: {output_dim_reg}")

    trained_reg_model, reg_history_pt = train_pytorch_model_api(
        pytorch_reg_model, train_loader_reg, val_loader_reg, criterion_reg, optimizer_reg, 
        epochs=35, task='regression'
    )
    reg_eval_metrics = evaluate_pytorch_model_api(
        trained_reg_model, test_loader_reg, criterion_reg, task='regression'
    )
    torch.save(trained_reg_model.state_dict(), os.path.join(OUTPUT_DIR, 'pytorch_regression_model.pth'))
    with open(os.path.join(OUTPUT_DIR, 'pytorch_regression_metrics.json'), 'w') as f:
        json.dump(reg_eval_metrics, f, indent=4)
    print(f"  PyTorch回归模型和指标已保存。历史记录条数: {len(reg_history_pt)}")

    # --- 3. PyTorch 生态系统概念解释 ---
    explain_pytorch_ecosystem_conceptual_api()

    print("\n\n===== PyTorch API化功能演示完成 =====")
    print(f"所有输出 (模型, 指标JSON) 保存在 '{OUTPUT_DIR}' 目录中。") 