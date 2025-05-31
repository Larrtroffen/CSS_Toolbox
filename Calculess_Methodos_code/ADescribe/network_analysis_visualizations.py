import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# For PyVis, it would be: from pyvis.network import Network
# For netwulf, it would be: import netwulf as nw (after installation)

# ==============================================================================
# 常量和输出目录设置 (Constants and Output Directory Setup)
# ==============================================================================
OUTPUT_DIR = "network_charts_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 图创建函数 (Graph Creation Functions)
# ==============================================================================

def create_graph_from_edgelist(edgelist, directed=False, add_weights=True):
    """
    从边列表创建NetworkX图。

    参数:
    edgelist (list of tuples): 边列表。每个元组可以是 (u, v) 或 (u, v, weight_dict)。
                                例如: [(1, 2), (2, 3, {'weight': 5})]
    directed (bool, optional): 是否创建有向图。默认为False (无向图)。
    add_weights (bool, optional): 如果边元组包含第三个元素 (字典), 
                                 是否将其作为边的属性添加。默认为True。
                                 如果为False，仅使用前两个元素创建边。

    返回:
    nx.Graph or nx.DiGraph: 创建的NetworkX图对象。
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    edges_to_add = []
    for edge in edgelist:
        if len(edge) == 2:
            edges_to_add.append(edge)
        elif len(edge) == 3 and isinstance(edge[2], dict) and add_weights:
            edges_to_add.append((edge[0], edge[1], edge[2]))
        elif len(edge) >= 2: # 忽略权重（如果有）
            edges_to_add.append((edge[0], edge[1]))
        else:
            print(f"警告: 边 '{edge}' 格式不正确，已跳过。")
            
    G.add_edges_from(edges_to_add)
    print(f"从边列表创建了图，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
    return G

def create_example_network(graph_type='karate', **kwargs):
    """
    使用NetworkX内置的图生成器创建示例网络。

    参数:
    graph_type (str, optional): 要生成的图的类型。
        可选值: 'karate', 'erdos_renyi', 'barabasi_albert', 'complete', 'cycle'.
        默认为 'karate'。
    **kwargs: 传递给相应图生成器的参数。
        - erdos_renyi: n (int), p (float), seed (int, optional)
        - barabasi_albert: n (int), m (int), seed (int, optional)
        - complete: n (int)
        - cycle: n (int)

    返回:
    nx.Graph: 生成的NetworkX图对象。
    """
    G = None
    if graph_type == 'karate':
        G = nx.karate_club_graph()
        print("加载了 Zachary's Karate Club 图。")
    elif graph_type == 'erdos_renyi':
        n = kwargs.get('n', 30)
        p = kwargs.get('p', 0.1)
        seed = kwargs.get('seed', None)
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        print(f"创建了 Erdos-Renyi 随机图 (n={n}, p={p}).")
    elif graph_type == 'barabasi_albert':
        n = kwargs.get('n', 30)
        m = kwargs.get('m', 2)
        seed = kwargs.get('seed', None)
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        print(f"创建了 Barabasi-Albert 无标度图 (n={n}, m={m}).")
    elif graph_type == 'complete':
        n = kwargs.get('n', 10)
        G = nx.complete_graph(n)
        print(f"创建了包含 {n} 个节点的完全图。")
    elif graph_type == 'cycle':
        n = kwargs.get('n', 10)
        G = nx.cycle_graph(n)
        print(f"创建了包含 {n} 个节点的循环图。")
    else:
        print(f"未知的图类型 '{graph_type}'。返回一个空图。")
        G = nx.Graph()
    return G

# ==============================================================================
# 图基本信息与操作 (Basic Graph Info and Operations)
# ==============================================================================

def get_graph_info(G):
    """
    打印并返回图的基本信息。

    参数:
    G (nx.Graph or nx.DiGraph): NetworkX图对象。

    返回:
    dict: 包含节点数、边数、是否为有向图等信息的字典。
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        print("输入不是有效的NetworkX图对象。")
        return {}
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    is_directed = G.is_directed()
    is_multigraph = G.is_multigraph()
    
    print("\n--- 图基本信息 ---")
    print(f"  节点数: {num_nodes}")
    print(f"  边数: {num_edges}")
    print(f"  是否为有向图: {is_directed}")
    print(f"  是否为多重图: {is_multigraph}")
    if num_nodes > 0:
        print(f"  节点示例 (前5个): {list(G.nodes(data=True))[:5]}...")
        print(f"  边示例 (前5个): {list(G.edges(data=True))[:5]}...")

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'is_directed': is_directed,
        'is_multigraph': is_multigraph
    }

def get_node_degree(G, node, weight=None):
    """
    获取指定节点的度 (或加权度)。

    参数:
    G (nx.Graph or nx.DiGraph): NetworkX图对象。
    node: 要查询的节点。
    weight (str, optional): 如果计算加权度，指定边属性的键名。默认为None (非加权度)。

    返回:
    int or float: 节点的度或加权度。如果节点不存在则返回None。
    """
    if node not in G:
        print(f"节点 {node} 不在图中。")
        return None
    return G.degree(node, weight=weight)

# ==============================================================================
# 网络度量计算 (Network Metrics Calculation)
# ==============================================================================

def calculate_network_metrics(G, robust=True):
    """
    计算常用的网络度量指标。

    参数:
    G (nx.Graph or nx.DiGraph): NetworkX图对象。
    robust (bool, optional): 是否稳健地处理计算错误 (例如，对于断开的图)。默认为True。

    返回:
    dict: 包含各种网络度量指标的字典。
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph)) or G.number_of_nodes() == 0:
        print("图为空或无效，无法计算度量指标。")
        return {}

    metrics = {}
    print("\n--- 网络度量指标计算 --- ")

    try:
        metrics['density'] = nx.density(G)
        print(f"  图密度: {metrics['density']:.4f}")
    except Exception as e:
        if robust: print(f"计算密度时出错: {e}")
        else: raise

    try:
        # 度中心性
        degree_centrality = nx.degree_centrality(G)
        metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
        metrics['top_5_degree_centrality'] = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)[:5]
        print(f"  平均度中心性: {metrics['avg_degree_centrality']:.4f}")
    except Exception as e:
        if robust: print(f"计算度中心性时出错: {e}")
        else: raise
        
    # 对于非连接图，以下某些指标可能只对最大连通分量有意义或引发错误
    # 确定是否在连通图上操作，或者在最大连通分量上操作
    effective_G = G
    if not G.is_directed() and not nx.is_connected(G):
        print("  图不连通。部分度量将针对最大连通分量计算。")
        largest_cc = max(nx.connected_components(G), key=len)
        effective_G = G.subgraph(largest_cc).copy()
    elif G.is_directed() and not nx.is_strongly_connected(G):
        print("  有向图非强连通。部分度量将针对最大强连通分量计算 (如果适用) 或在底层无向图上计算。")
        # 对于某些有向图指标，可能需要在其底层无向图上操作或处理各分量
        # effective_G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
        # 或者，有些指标如平均最短路径长度，本身可以处理非强连通图

    try:
        metrics['avg_clustering_coefficient'] = nx.average_clustering(G) # 用原图G
        print(f"  平均聚类系数 (全图): {metrics['avg_clustering_coefficient']:.4f}")
    except Exception as e:
        if robust: print(f"计算平均聚类系数时出错: {e}")
        else: raise

    try:
        betweenness_centrality = nx.betweenness_centrality(effective_G) # k参数可用于加速大型图
        metrics['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
        metrics['top_5_betweenness_centrality'] = sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)[:5]
        print(f"  平均介数中心性 (在有效图上): {metrics['avg_betweenness_centrality']:.4f}")
    except Exception as e:
        if robust: print(f"计算介数中心性时出错: {e}")
        else: raise

    try:
        closeness_centrality = nx.closeness_centrality(effective_G)
        metrics['avg_closeness_centrality'] = np.mean(list(closeness_centrality.values()))
        metrics['top_5_closeness_centrality'] = sorted(closeness_centrality.items(), key=lambda item: item[1], reverse=True)[:5]
        print(f"  平均接近度中心性 (在有效图上): {metrics['avg_closeness_centrality']:.4f}")
    except Exception as e:
        if robust: print(f"计算接近度中心性时出错: {e}")
        else: raise
    
    if not G.is_directed(): # 仅对无向图
        if nx.is_connected(effective_G):
            try:
                metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(effective_G)
                print(f"  平均最短路径长度 (在有效图上): {metrics['avg_shortest_path_length']:.4f}")
            except Exception as e:
                if robust: print(f"计算平均最短路径长度时出错: {e}")
                else: raise
        else:
             print("  有效图仍然不连通，无法计算全局平均最短路径长度。")
    elif G.is_directed():
        # 对于有向图，最短路径的定义和连通性更复杂
        # 可以考虑使用 nx.shortest_path_length 对特定节点对进行计算
        print("  对于有向图，平均最短路径长度的计算需要特定考虑 (如基于强连通性)。")

    return metrics

# ==============================================================================
# 图可视化函数 (Graph Visualization Functions)
# ==============================================================================

def visualize_network(G, layout_type='spring', title='NetworkX Graph', 
                      node_color_attr=None, node_size_attr=None, 
                      edge_width_attr=None, show_labels=True,
                      save_filename_png=None, **kwargs):
    """
    使用Matplotlib可视化NetworkX图。

    参数:
    G (nx.Graph or nx.DiGraph): NetworkX图对象。
    layout_type (str, optional): 布局算法。
        可选: 'spring', 'circular', 'kamada_kawai', 'random', 'spectral', 'shell'. 
        默认为 'spring'。
    title (str, optional): 图表标题。
    node_color_attr (str, optional): 用于节点颜色编码的节点属性名。
                                   如果属性是数值，则使用连续色条；如果是类别，则使用离散颜色。
    node_size_attr (str, optional): 用于节点大小编码的节点属性名 (应为数值)。
    edge_width_attr (str, optional): 用于边宽度编码的边属性名 (应为数值，例如 'weight')。
    show_labels (bool, optional): 是否显示节点标签。默认为True。
    save_filename_png (str, optional): PNG图像的保存名 (例如 'my_network.png')。
                                     如果None，则尝试plt.show()。
    **kwargs: 传递给 nx.draw_networkx 或特定布局函数的额外参数。
              例如: seed (int) 用于可复现布局, k (float) 用于spring_layout的间距。
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph)) or G.number_of_nodes() == 0:
        print("图为空或无效，无法可视化。")
        return

    plt.figure(figsize=kwargs.pop('figsize', (12, 10)))
    
    # 布局
    pos = None
    layout_seed = kwargs.pop('seed', 42)
    if layout_type == 'spring':
        pos = nx.spring_layout(G, seed=layout_seed, k=kwargs.pop('k', None), iterations=kwargs.pop('iterations', 50))
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, **kwargs.pop('kamada_kawai_kwargs', {}))
    elif layout_type == 'random':
        pos = nx.random_layout(G, seed=layout_seed)
    elif layout_type == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout_type == 'shell':
        # Shell布局可能需要 nlist 参数 (节点的层列表)
        pos = nx.shell_layout(G, **kwargs.pop('shell_kwargs', {}))
    else:
        print(f"未知的布局类型 '{layout_type}'。默认为 spring_layout。")
        pos = nx.spring_layout(G, seed=layout_seed)

    # 节点颜色
    node_colors_final = 'skyblue'
    node_cmap = plt.cm.viridis # 默认色条
    if node_color_attr and node_color_attr in list(G.nodes(data=True))[0][1]:
        attr_values = np.array([data.get(node_color_attr) for _, data in G.nodes(data=True)])
        if pd.api.types.is_numeric_dtype(attr_values) and not pd.api.types.is_bool_dtype(attr_values):
            node_colors_final = attr_values
            node_cmap = kwargs.pop('node_cmap', plt.cm.coolwarm)
        else: # 类别属性
            unique_cats = pd.Series(attr_values).unique()
            cat_map = {cat: i for i, cat in enumerate(unique_cats)}
            node_colors_final = [cat_map[val] for val in attr_values]
            node_cmap = kwargs.pop('node_cmap', plt.cm.get_cmap('tab10', len(unique_cats)))
    
    # 节点大小
    node_sizes_final = 300 # 默认大小
    if node_size_attr and node_size_attr in list(G.nodes(data=True))[0][1]:
        size_values = np.array([data.get(node_size_attr, 1) for _, data in G.nodes(data=True)])
        if pd.api.types.is_numeric_dtype(size_values):
            # 归一化或缩放大小值以获得合理的视觉效果
            min_size = kwargs.pop('min_node_size', 100)
            max_size = kwargs.pop('max_node_size', 1000)
            if np.ptp(size_values) > 0:
                node_sizes_final = min_size + (size_values - np.min(size_values)) / np.ptp(size_values) * (max_size - min_size)
            else:
                node_sizes_final = [min_size] * len(size_values)
        else:
            print(f"警告: 节点大小属性 '{node_size_attr}' 不是数值型，将使用默认大小。")

    # 边宽度
    edge_widths_final = 1.0
    if edge_width_attr:
        try:
            weights = [G.edges[u,v].get(edge_width_attr, 1.0) for u,v in G.edges()]
            # 归一化或缩放宽度
            min_width = kwargs.pop('min_edge_width', 0.5)
            max_width = kwargs.pop('max_edge_width', 5.0)
            if np.ptp(weights) > 0:
                 edge_widths_final = [min_width + (w - np.min(weights)) / np.ptp(weights) * (max_width - min_width) for w in weights]
            else:
                 edge_widths_final = [min_width] * len(weights)
        except Exception as e:
            print(f"处理边宽度属性 '{edge_width_attr}' 时出错: {e}。将使用默认宽度。")

    nx.draw_networkx(G, pos, 
                       with_labels=show_labels,
                       node_color=node_colors_final,
                       cmap=node_cmap if isinstance(node_colors_final, (list, np.ndarray)) and pd.api.types.is_numeric_dtype(node_colors_final) else None, # cmap仅用于数值颜色
                       node_size=node_sizes_final,
                       width=edge_widths_final,
                       font_size=kwargs.pop('font_size', 8),
                       font_color=kwargs.pop('font_color', 'black'),
                       edge_color=kwargs.pop('edge_color', 'gray'),
                       alpha=kwargs.pop('alpha', 0.8),
                       **kwargs)

    plt.title(f"{title} (布局: {layout_type})", fontsize=15)
    plt.axis('off') # 关闭坐标轴
    
    if save_filename_png:
        full_save_path = os.path.join(OUTPUT_DIR, save_filename_png if save_filename_png.lower().endswith('.png') else f"{save_filename_png}.png")
        plt.savefig(full_save_path, bbox_inches='tight')
        print(f"网络图已保存至: {full_save_path}")
    else:
        plt.show()
    plt.close()

# ==============================================================================
# 交互式可视化 (Conceptual Placeholder for PyVis)
# ==============================================================================

def conceptual_pyvis_visualization(G, filename="interactive_network.html"):
    """
    PyVis交互式网络可视化的概念性演示。
    实际执行需要安装PyVis库。

    参数:
    G (nx.Graph or nx.DiGraph): NetworkX图对象。
    filename (str, optional): 输出HTML文件的名称。
    """
    print("\n--- PyVis 交互式可视化 (概念性演示) ---")
    print(f"要使用PyVis生成交互式HTML可视化 (如果已安装PyVis):")
    print("  from pyvis.network import Network")
    net_name = G.name if hasattr(G, 'name') and G.name else "交互式网络图"
    print(f"  nt = Network(notebook=True, height='750px', width='100%', heading='{net_name}')")
    print("  nt.from_nx(G)")
    print(f"  # 可选: 添加节点颜色、大小等 PyVis 特定配置")
    print(f"  # nt.show_buttons(filter_=['physics'])")
    save_path = os.path.join(OUTPUT_DIR, filename)
    print(f"  nt.save_graph('{save_path}') # 或 nt.show('{save_path}') 直接在浏览器中打开")
    print(f"这将创建一个HTML文件: {save_path}")
    # 实际代码 (如果 PyVis 是依赖项):
    # try:
    #     from pyvis.network import Network
    #     nt = Network(notebook=True, height='750px', width='100%', heading=G.name or 'Interactive Network')
    #     nt.from_nx(G)
    #     # Example: Color nodes by degree
    #     # for node in nt.nodes:
    #     #     node['value'] = G.degree(node['id'])
    #     #     node['title'] = f"Degree: {G.degree(node['id'])}"
    #     # nt.show_buttons(filter_=['physics'])
    #     full_save_path = os.path.join(OUTPUT_DIR, filename)
    #     nt.save_graph(full_save_path)
    #     print(f"PyVis 交互式图已保存至: {full_save_path}")
    # except ImportError:
    #     print("未找到PyVis库。跳过PyVis交互式演示。")
    # except Exception as e:
    #     print(f"PyVis演示出错: {e}")

# ==============================================================================
# 演示函数 (用于独立运行和测试)
# ==============================================================================

def run_network_analysis_demos():
    """运行所有网络分析与可视化演示函数。"""
    print(f"--- 网络分析与可视化接口化演示 (图表将保存到 '{OUTPUT_DIR}' 目录) ---")

    # 1. 创建图
    print("\n=== 1. 创建图 ===")
    karate_club = create_example_network(graph_type='karate')
    er_graph = create_example_network(graph_type='erdos_renyi', n=50, p=0.08, seed=123)
    
    custom_edges = [
        ('A', 'B', {'weight': 3, 'type': 'friend'}),
        ('A', 'C', {'weight': 1, 'type': 'colleague'}),
        ('B', 'C', {'weight': 2, 'type': 'friend'}),
        ('B', 'D', {'weight': 5, 'type': 'family'}),
        ('C', 'D', {'weight': 1}),
        ('D', 'E')
    ]
    custom_graph = create_graph_from_edgelist(custom_edges)
    # 为自定义图添加一些节点属性
    nx.set_node_attributes(custom_graph, {
        'A': {'department': 'Sales', 'age': 30},
        'B': {'department': 'Marketing', 'age': 35},
        'C': {'department': 'Sales', 'age': 28},
        'D': {'department': 'HR', 'age': 42},
        'E': {'department': 'Marketing', 'age': 25},
    })


    # 2. 图基本信息
    print("\n=== 2. 图基本信息 ===")
    get_graph_info(karate_club)
    get_graph_info(custom_graph)
    print(f"  节点 'A' 在自定义图中的度: {get_node_degree(custom_graph, 'A')}")
    print(f"  节点 'B' 在自定义图中的加权度 ('weight'): {get_node_degree(custom_graph, 'B', weight='weight')}")

    # 3. 网络度量
    print("\n=== 3. 网络度量 ===")
    karate_metrics = calculate_network_metrics(karate_club)
    # print("空手道俱乐部图度量详情:", karate_metrics)
    custom_graph_metrics = calculate_network_metrics(custom_graph)
    # print("自定义图度量详情:", custom_graph_metrics)

    # 4. 网络可视化 (Matplotlib)
    print("\n=== 4. 网络可视化 (Matplotlib) ===")
    visualize_network(karate_club, layout_type='kamada_kawai', title='空手道俱乐部图 (Kamada-Kawai布局)',
                      save_filename_png="demo_karate_club_kamada.png")
    
    visualize_network(er_graph, layout_type='spring', title='Erdos-Renyi随机图 (Spring布局)',
                      node_size_attr=None, # ER图没有预设的size属性
                      save_filename_png="demo_er_graph_spring.png",
                      show_labels=False, figsize=(8,6)) # 对于稍大的图，可以不显示标签

    visualize_network(custom_graph, layout_type='spring', title='自定义社交网络图',
                      node_color_attr='department', # 按部门着色
                      node_size_attr='age',       # 按年龄调整大小
                      edge_width_attr='weight',    # 按关系权重调整边宽
                      save_filename_png="demo_custom_social_network.png",
                      k=0.8, iterations=100) # spring layout参数

    # 5. 概念性PyVis演示
    conceptual_pyvis_visualization(karate_club, filename="demo_pyvis_karate.html")

    print(f"--- 网络分析演示完成。输出已保存到 '{OUTPUT_DIR}' 目录。 ---")

if __name__ == '__main__':
    run_network_analysis_demos()

    # Mention other libraries like igraph, netwulf as per the outline
    print("\n--- Other Network Libraries (Conceptual) ---")
    print("python-igraph: Another powerful network analysis library, often faster for certain algorithms.")
    print("  - Usage involves creating an igraph.Graph object and using its methods for analysis and plotting.")
    print("netwulf: For interactive web-based visualizations, often used in Jupyter.")
    print("  - Usage: import netwulf as nw; nw.visualize(G_networkx) # G_networkx is a NetworkX graph") 