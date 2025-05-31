import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# 常量和输出目录设置 (Constants and Output Directory Setup)
# ==============================================================================
OUTPUT_DIR = "dimensionality_reduction_charts_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 示例数据生成 (Sample Data Generation)
# ==============================================================================

def generate_sample_high_dim_data(n_samples=200, n_features=20, n_classes=3, random_state=None):
    """
    生成用于降维演示的高维样本数据。

    参数:
    n_samples (int): 样本数量。
    n_features (int): 特征数量 (维度)。
    n_classes (int): 类别数量 (用于生成标签)。
    random_state (int, optional): 随机种子，用于可复现性。

    返回:
    tuple: (X, y)
        X (np.ndarray): 特征数据，形状为 (n_samples, n_features)。
        y (np.ndarray): 标签数据，形状为 (n_samples,)。
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 为每个类别生成一些聚类中心
    centroids = np.random.rand(n_classes, n_features) * 10
    X_list = []
    y_list = []
    
    for i in range(n_classes):
        # 为每个类别生成样本，围绕其聚类中心分布
        # 确保样本数量大致平均分配到每个类别
        samples_per_class = n_samples // n_classes
        if i == n_classes - 1: # 最后一个类别获取剩余所有样本
            samples_per_class = n_samples - (len(X_list))
            
        class_data = np.random.randn(samples_per_class, n_features) * 1.5 + centroids[i, :]
        X_list.append(class_data)
        y_list.extend([i] * samples_per_class)
        
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    # 打乱数据顺序
    shuffle_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffle_indices)
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    
    print(f"生成了样本数据: {X.shape[0]}个样本, {X.shape[1]}个特征, {len(np.unique(y))}个类别。")
    return X, y

# ==============================================================================
# 数据预处理 (Data Preprocessing)
# ==============================================================================

def scale_features(X):
    """
    对特征数据进行标准化 (均值为0，方差为1)。

    参数:
    X (np.ndarray or pd.DataFrame): 输入特征数据。

    返回:
    np.ndarray: 标准化后的特征数据。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("特征数据已进行标准化处理。")
    return X_scaled

# ==============================================================================
# PCA (Principal Component Analysis)
# ==============================================================================

def apply_pca(X_scaled, n_components=None, random_state=None):
    """
    对标准化后的数据应用PCA。

    参数:
    X_scaled (np.ndarray): 标准化后的特征数据。
    n_components (int, float, str or None, optional): 要保留的主成分数量。
        - 如果是int, 则为具体数量。
        - 如果是float (0到1之间), 则为保留的累积解释方差比例。
        - 如果是'mle', 则使用Minka's MLE自动选择维度。
        - 如果是None, 则保留 min(n_samples, n_features) 个成分。
        默认为None。
    random_state (int, optional): 随机种子，用于SVD求解器。

    返回:
    tuple: (pca_model, X_pca)
        pca_model (sklearn.decomposition.PCA): 拟合的PCA模型。
        X_pca (np.ndarray): 降维后的数据。
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA应用完成。保留了 {pca.n_components_} 个主成分。")
    print(f"  这些成分解释的总方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
    return pca, X_pca

def plot_pca_explained_variance(pca_model, save_filename_png=None):
    """
    绘制PCA各主成分解释的方差比例以及累积方差比例。

    参数:
    pca_model (sklearn.decomposition.PCA): 拟合的PCA模型。
    save_filename_png (str, optional): 图像保存文件名。如果提供则保存。
    """
    if not hasattr(pca_model, 'explained_variance_ratio_'):
        print("PCA模型未拟合或不包含explained_variance_ratio_属性。")
        return

    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    n_components_to_plot = len(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_components_to_plot + 1), explained_variance_ratio, alpha=0.7, align='center',
            label='单个主成分解释方差', color='skyblue')
    plt.step(range(1, n_components_to_plot + 1), cumulative_explained_variance, where='mid',
             label='累积解释方差', color='red')
    
    plt.ylabel('解释方差比例')
    plt.xlabel('主成分序号')
    plt.title('PCA解释方差比例图')
    plt.xticks(range(1, n_components_to_plot + 1))
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_filename_png:
        full_save_path = os.path.join(OUTPUT_DIR, save_filename_png)
        plt.savefig(full_save_path)
        print(f"PCA解释方差图已保存至: {full_save_path}")
    else:
        plt.show()
    plt.close()

def plot_pca_2d_scatter(X_pca_2d, y_labels, title='PCA降维结果 (前两个主成分)', save_filename_png=None):
    """
    绘制PCA降维到2D后的散点图，按类别着色。

    参数:
    X_pca_2d (np.ndarray): PCA降维后的前两个主成分数据 (形状: n_samples, 2)。
    y_labels (np.ndarray): 样本的类别标签 (形状: n_samples,)。
    title (str, optional): 图表标题。
    save_filename_png (str, optional): 图像保存文件名。如果提供则保存。
    """
    if X_pca_2d.shape[1] < 2:
        print("PCA结果不足二维，无法绘制2D散点图。")
        return
        
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(y_labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        plt.scatter(X_pca_2d[y_labels == label, 0], X_pca_2d[y_labels == label, 1], 
                    color=colors(i), label=f'类别 {label}', alpha=0.8, s=50)
    
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_filename_png:
        full_save_path = os.path.join(OUTPUT_DIR, save_filename_png)
        plt.savefig(full_save_path)
        print(f"PCA 2D散点图已保存至: {full_save_path}")
    else:
        plt.show()
    plt.close()

# ==============================================================================
# t-SNE (t-distributed Stochastic Neighbor Embedding)
# ==============================================================================

def apply_tsne(X_scaled, n_components=2, perplexity=30.0, learning_rate=200.0, 
                 n_iter=1000, random_state=None, **tsne_kwargs):
    """
    对标准化后的数据应用t-SNE进行降维。

    参数:
    X_scaled (np.ndarray): 标准化后的特征数据。
    n_components (int, optional): 嵌入空间的维度 (通常是2或3)。默认为2。
    perplexity (float, optional): 与近邻点的数量相关。典型值在5到50之间。
    learning_rate (float, optional): 学习率。典型值在10.0到1000.0之间。
    n_iter (int, optional): 优化的最大迭代次数。
    random_state (int, optional): 随机种子，用于可复现性。
    **tsne_kwargs: 传递给 sklearn.manifold.TSNE 的其他参数。

    返回:
    np.ndarray: 降维后的t-SNE嵌入数据。
    """
    print(f"正在应用t-SNE (perplexity={perplexity}, n_iter={n_iter})... 这可能需要一些时间。")
    tsne = TSNE(n_components=n_components, 
                  perplexity=perplexity, 
                  learning_rate=learning_rate, 
                  n_iter=n_iter, 
                  random_state=random_state, 
                  **tsne_kwargs)
    X_tsne = tsne.fit_transform(X_scaled)
    print(f"t-SNE降维完成，结果形状: {X_tsne.shape}")
    return X_tsne

def plot_tsne_2d_scatter(X_tsne_2d, y_labels, title='t-SNE降维结果 (2D)', save_filename_png=None):
    """
    绘制t-SNE降维到2D后的散点图，按类别着色。

    参数:
    X_tsne_2d (np.ndarray): t-SNE降维后的2D数据 (形状: n_samples, 2)。
    y_labels (np.ndarray): 样本的类别标签 (形状: n_samples,)。
    title (str, optional): 图表标题。
    save_filename_png (str, optional): 图像保存文件名。如果提供则保存。
    """
    if X_tsne_2d.shape[1] != 2:
        print("t-SNE结果不是二维，无法绘制2D散点图。")
        return

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(y_labels)
    # colors = sns.color_palette("husl", len(unique_labels))
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        plt.scatter(X_tsne_2d[y_labels == label, 0], X_tsne_2d[y_labels == label, 1], 
                    color=colors(i), label=f'类别 {label}', alpha=0.7, s=50)
    
    plt.xlabel('t-SNE 维度 1')
    plt.ylabel('t-SNE 维度 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.xticks([]) # t-SNE轴通常没有直接解释意义
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()

    if save_filename_png:
        full_save_path = os.path.join(OUTPUT_DIR, save_filename_png)
        plt.savefig(full_save_path)
        print(f"t-SNE 2D散点图已保存至: {full_save_path}")
    else:
        plt.show()
    plt.close()

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================

def run_dimensionality_reduction_demos():
    """运行所有降维与可视化演示函数。"""
    print(f"--- 降维与可视化接口化演示 (图表将保存到 '{OUTPUT_DIR}' 目录) ---")

    # 1. 生成并准备数据
    print("\n--- 1. 数据准备 ---")
    X_hd, y_hd = generate_sample_high_dim_data(n_samples=300, n_features=50, n_classes=4, random_state=42)
    X_hd_scaled = scale_features(X_hd)

    # 2. PCA演示
    print("\n--- 2. PCA 演示 ---")
    # 第一次PCA: 自动选择主成分数量 (例如，保留95%方差)
    pca_model_auto, X_pca_auto = apply_pca(X_hd_scaled, n_components=0.95, random_state=42)
    plot_pca_explained_variance(pca_model_auto, save_filename_png="demo_pca_explained_variance_auto.png")
    
    # 第二次PCA: 显式指定降到2维以便可视化
    pca_model_2d, X_pca_2d = apply_pca(X_hd_scaled, n_components=2, random_state=42)
    if X_pca_2d.shape[1] == 2:
        plot_pca_2d_scatter(X_pca_2d, y_hd, 
                              title='PCA降至2D散点图 (按类别着色)', 
                              save_filename_png="demo_pca_scatter_2d.png")

    # 3. t-SNE演示
    print("\n--- 3. t-SNE 演示 ---")
    # 注意: t-SNE对参数敏感，特别是perplexity。
    # 对于大型数据集，t-SNE计算成本较高。
    # 如果数据维度仍然很高，可以先用PCA降到中等维度 (例如50维)，再用t-SNE。
    # 这里我们直接在原始标准化数据上应用 (如果特征数不是过高)
    X_to_tsne = X_hd_scaled
    if X_hd_scaled.shape[1] > 50: # 如果原始特征过多，先用PCA粗略降维
        print("原始特征数较多，先用PCA预降维到50维再进行t-SNE。")
        pca_for_tsne, X_pca_for_tsne = apply_pca(X_hd_scaled, n_components=50, random_state=42)
        X_to_tsne = X_pca_for_tsne

    X_tsne_result = apply_tsne(X_to_tsne, n_components=2, perplexity=30, 
                               learning_rate='auto', # sklearn 1.2+ 推荐 'auto' for learning_rate
                               init='pca', # PCA初始化通常更好
                               n_iter=1000, random_state=42)
    plot_tsne_2d_scatter(X_tsne_result, y_hd, 
                         title='t-SNE降至2D散点图 (按类别着色)',
                         save_filename_png="demo_tsne_scatter_2d.png")

    print(f"\n--- 降维演示完成。输出图表已保存到 '{OUTPUT_DIR}' 目录。 ---")

if __name__ == '__main__':
    run_dimensionality_reduction_demos() 