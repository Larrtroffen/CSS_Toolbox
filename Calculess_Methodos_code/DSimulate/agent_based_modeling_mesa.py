import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import random # random is part of mesa.Model via self.random, no need for separate import unless used elsewhere

# --- 辅助函数：创建输出目录 ---
def _ensure_output_directory_api(dir_name: str = "mesa_outputs_demo") -> str:
    """
    确保指定的输出目录存在，如果不存在则创建它。

    参数:
    - dir_name (str): 要创建的目录名。

    返回:
    - str: 目录的绝对或相对路径。
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print(f"  已创建目录: {dir_name}")
        except OSError as e:
            print(f"  创建目录 {dir_name} 失败: {e}. 图表可能不会保存。")
    return dir_name

# --- Schelling Segregation Model (API化) ---

class _SchellingAgent(mesa.Agent):
    """Schelling模型的个体代理。"""
    def __init__(self, unique_id, model, agent_type: int, homophily_threshold: float):
        """
        初始化一个Schelling代理。

        参数:
        - unique_id: 代理的唯一ID。
        - model: 代理所属的模型实例。
        - agent_type (int): 代理的类型 (例如, 0 或 1, 代表两个不同的群体)。
        - homophily_threshold (float): 同质性阈值，代理满意的最小同类型邻居比例。
        """
        super().__init__(unique_id, model)
        self.agent_type = agent_type
        self.homophily_threshold = homophily_threshold

    def step(self):
        """代理的行动步骤：检查满意度，如果不满意则移动。"""
        if self._is_unhappy():
            self.model.grid.move_to_empty(self)
            self.model.moved_agents_in_step += 1 # 记录本轮移动的代理数量

    def _is_unhappy(self) -> bool:
        """检查代理是否对其当前位置不满意。"""
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if not neighbors:
            return False # 没有邻居，不会因为邻居而不满意

        same_type_neighbors = sum(1 for neighbor in neighbors if neighbor.agent_type == self.agent_type)
        total_neighbors = len(neighbors)
        
        proportion_same_type = same_type_neighbors / total_neighbors if total_neighbors > 0 else 0
        
        return proportion_same_type < self.homophily_threshold

class _SchellingModel(mesa.Model):
    """Schelling种族隔离模型。"""
    def __init__(self, width: int = 20, height: int = 20, density: float = 0.8, 
                 proportion_group0: float = 0.5, homophily: float = 0.4, seed: int | None = None):
        """
        初始化Schelling模型。

        参数:
        - width (int): 网格宽度。
        - height (int): 网格高度。
        - density (float): 被占据单元格的比例。
        - proportion_group0 (float): 属于群体0的代理的比例。
        - homophily (float): 所有代理的同质性阈值。
        - seed (int | None): 随机数种子，用于可复现性。
        """
        super().__init__(seed=seed) # 将种子传递给父类Model
        self.width = width
        self.height = height
        self.density = density
        self.proportion_group0 = proportion_group0
        self.homophily = homophily
        self.moved_agents_in_step = 0 # 计数器：本轮移动的代理数量

        self.grid = mesa.space.SingleGrid(width, height, torus=True) # 单格网格，环形边界
        self.schedule = mesa.time.RandomActivation(self) # 每轮随机激活代理

        total_cells = width * height
        n_agents = int(total_cells * density)
        n_group0 = int(n_agents * proportion_group0)

        agent_id_counter = 0
        for i in range(n_agents):
            agent_type = 0 if i < n_group0 else 1
            agent = _SchellingAgent(agent_id_counter, self, agent_type, self.homophily)
            agent_id_counter += 1
            self.schedule.add(agent)
            self.grid.move_to_empty(agent) # 将代理放置到随机的空单元格

        # 数据收集器，用于模型级别和代理级别的变量
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "unhappy_agents": lambda m: sum(1 for a in m.schedule.agents if a._is_unhappy()),
                "moved_this_step": lambda m: m.moved_agents_in_step,
                "segregation_metric": self._calculate_segregation_metric
            },
            # agent_reporters={"agent_type": "agent_type", "is_unhappy": lambda a: a._is_unhappy(), "pos": "pos"} # 代理数据可能非常大
        )
        self.running = True # 用于基于无移动判断停止条件
        self.datacollector.collect(self) # 收集初始状态

    def step(self):
        """将模型推进一个时间步。"""
        self.moved_agents_in_step = 0 # 为新的一步重置计数器
        self.schedule.step() # 执行所有代理的step方法
        self.datacollector.collect(self) # 收集这一步的数据
        
        if self.moved_agents_in_step == 0:
            self.running = False # 如果本轮没有代理移动，则停止 (达到均衡)
            print(f"  模型在第 {self.schedule.steps} 步达到均衡状态。")

    def _calculate_segregation_metric(self) -> float:
        """一个简单的隔离度量：同类型邻居的平均比例。"""
        proportions = []
        for agent in self.schedule.agents:
            neighbors = self.grid.get_neighbors(agent.pos, moore=True, include_center=False)
            if not neighbors: continue
            same_type_neighbors = sum(1 for n in neighbors if n.agent_type == agent.agent_type)
            proportions.append(same_type_neighbors / len(neighbors))
        return np.mean(proportions) if proportions else 0

def run_schelling_model_api(
    width: int = 20, 
    height: int = 20, 
    density: float = 0.8, 
    proportion_group0: float = 0.5, 
    homophily_threshold: float = 0.4, 
    max_steps: int = 100, 
    seed_val: int | None = 42,
    output_dir: str = "mesa_outputs_demo",
    plot_results: bool = True
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """
    运行Schelling种族隔离模型。

    参数:
    - width, height (int): 网格尺寸。
    - density (float): 代理占据网格的密度 (0到1之间)。
    - proportion_group0 (float): 群体0的代理所占比例。
    - homophily_threshold (float): 代理满意的同质性阈值。
    - max_steps (int): 模型运行的最大步数。
    - seed_val (int | None): 随机种子。
    - output_dir (str): 保存图表的目录。
    - plot_results (bool): 是否生成并保存图表。

    返回:
    - tuple[pd.DataFrame | None, str | None, str | None]:
        - 模型级别数据的DataFrame。
        - 模型动态图的路径 (如果不绘图则为None)。
        - 最终网格分布图的路径 (如果不绘图则为None)。
    """
    print(f"\n--- (API) 运行Schelling模型 ---")
    print(f"  参数: 网格={width}x{height}, 密度={density}, 群体0比例={proportion_group0}, 同质性阈值={homophily_threshold}, 随机种子={seed_val}")
    
    model = _SchellingModel(width, height, density, proportion_group0, homophily_threshold, seed=seed_val)
    
    for i in range(max_steps):
        if not model.running:
            break
        model.step()
    
    print(f"  Schelling模型在 {model.schedule.steps} 步后结束 (或达到最大步数)。")
    model_data_df = model.datacollector.get_model_vars_dataframe()
    dynamics_plot_path = None
    grid_plot_path = None

    if not plot_results:
        return model_data_df, None, None

    target_dir = _ensure_output_directory_api(output_dir)
    if not os.path.exists(target_dir):
        print(f"  警告: 输出目录 {target_dir} 不存在，Schelling模型图表无法保存。")
        return model_data_df, None, None

    # 绘制模型级别数据图
    try:
        fig, ax1 = plt.subplots(figsize=(10,6))
        color = 'tab:red'
        ax1.set_xlabel('模型步数')
        ax1.set_ylabel('不满意的代理数量', color=color)
        ax1.plot(model_data_df.index, model_data_df['unhappy_agents'], color=color, marker='.', linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('隔离度量', color=color)
        ax2.plot(model_data_df.index, model_data_df['segregation_metric'], color=color, marker='x', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title(f'Schelling模型动态 (同质性阈值: {homophily_threshold})')
        dynamics_plot_filename = f"schelling_dynamics_h{int(homophily_threshold*100)}_s{seed_val if seed_val is not None else 'None'}.png"
        dynamics_plot_path = os.path.join(target_dir, dynamics_plot_filename)
        plt.savefig(dynamics_plot_path)
        print(f"  Schelling模型动态图已保存至: {dynamics_plot_path}")
        plt.close(fig)
    except Exception as e_dyn_plot:
        print(f"  绘制Schelling模型动态图失败: {e_dyn_plot}")
        dynamics_plot_path = None

    # 绘制最终代理分布图
    try:
        agent_counts = np.zeros((model.grid.width, model.grid.height))
        for cell_content, pos_x, pos_y in model.grid.coord_iter(): # Corrected iteration for SingleGrid
            if cell_content is not None:
                agent_counts[pos_x][pos_y] = cell_content.agent_type + 1
        
        plt.figure(figsize=(8, 8))
        plt.imshow(agent_counts.T, interpolation='nearest', cmap='viridis')
        plt.colorbar(ticks=[0, 1, 2], label="0: 空, 1: 类型0, 2: 类型1")
        plt.title(f'最终代理分布 (同质性阈值: {homophily_threshold})')
        grid_plot_filename = f"schelling_grid_h{int(homophily_threshold*100)}_s{seed_val if seed_val is not None else 'None'}.png"
        grid_plot_path = os.path.join(target_dir, grid_plot_filename)
        plt.savefig(grid_plot_path)
        print(f"  Schelling最终网格分布图已保存至: {grid_plot_path}")
        plt.close()
    except Exception as e_grid_plot:
        print(f"  绘制Schelling最终网格分布图失败: {e_grid_plot}")
        grid_plot_path = None
        
    return model_data_df, dynamics_plot_path, grid_plot_path


def explain_sir_model_conceptual_api():
    """提供关于使用Mesa构建基础SIR模型的概念性解释和代码片段。"""
    print("\n--- (API) Mesa基础SIR模型概念解释 ---")
    print("SIR模型 (易感-感染-恢复模型) 用于模拟传染病在人群中的传播。")
    print("代理可以处于三种状态之一: 易感 (S), 感染 (I), 或恢复 (R)。")
    
    sir_agent_code = (
        "# SIRAgent (SIR代理) 的概念代码片段:\n"
        "# class SIRAgent(mesa.Agent):\n"
        "#     def __init__(self, unique_id, model, initial_state=\"S\", infection_prob=0.1, recovery_duration=10, exposure_radius=1):\n"
        "#         super().__init__(unique_id, model)\n"
        "#         self.state = initial_state  # \"S\", \"I\", 或 \"R\"\n"
        "#         self.infection_prob = infection_prob\n"
        "#         self.recovery_duration = recovery_duration\n"
        "#         self.exposure_radius = exposure_radius # 代理能影响的邻居范围\n"
        "#         self.infection_timer = 0  # 该代理已被感染多久\n"
        "# \n"
        "#     def step(self):\n"
        "#         if self.state == \"I\":\n"
        "#             # 尝试感染邻居 (如果模型有网格和空间交互)\n"
        "#             # 对于非空间模型，这可能涉及随机交互或接触网络\n"
        "#             if hasattr(self.model, 'grid'): # 检查模型是否有grid属性\n"
        "#                 neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=self.exposure_radius)\n"
        "#                 for neighbor in neighbors:\n"
        "#                     if hasattr(neighbor, 'state') and neighbor.state == \"S\" and self.model.random.random() < self.infection_prob:\n"
        "#                         neighbor.state = \"I\"  # 邻居被感染\n"
        "#                         neighbor.infection_timer = 0 # 重置其计时器\n"
        "#             \n"
        "#             self.infection_timer += 1\n"
        "#             if self.infection_timer >= self.recovery_duration:\n"
        "#                 self.state = \"R\"  # 代理恢复\n"
    )
    print("SIRAgent概念代码片段:\n" + sir_agent_code)

    sir_model_code = (
        "# SIRModel (SIR模型) 的概念代码片段:\n"
        "# class SIRModel(mesa.Model):\n"
        "#     def __init__(self, n_agents=100, width=20, height=20, initial_infected=1, \n"
        "#                  infection_prob=0.05, recovery_duration=14, exposure_radius=1, seed=None):\n"
        "#         super().__init__(seed=seed)\n"
        "#         self.num_agents = n_agents\n"
        "#         # 如果允许每个单元格有多个代理，使用MultiGrid；如果每个单元格最多一个代理，使用SingleGrid\n"
        "#         self.grid = mesa.space.MultiGrid(width, height, torus=True) \n"
        "#         self.schedule = mesa.time.RandomActivation(self)\n"
        "#         self.infection_prob = infection_prob\n"
        "#         self.recovery_duration = recovery_duration\n"
        "#         self.exposure_radius = exposure_radius\n"
        "#         self.running = True\n"
        "# \n"
        "#         # 创建代理\n"
        "#         for i in range(self.num_agents):\n"
        "#             agent_state = \"I\" if i < initial_infected else \"S\"\n"
        "#             agent = SIRAgent(i, self, agent_state, self.infection_prob, self.recovery_duration, self.exposure_radius)\n"
        "#             self.schedule.add(agent)\n"
        "#             # 将代理放置到随机网格单元中\n"
        "#             x = self.random.randrange(self.grid.width)\n"
        "#             y = self.random.randrange(self.grid.height)\n"
        "#             self.grid.place_agent(agent, (x, y))\n"
        "# \n"
        "#         self.datacollector = mesa.DataCollector(\n"
        "#             model_reporters={\"Susceptible\": lambda m: sum(1 for a in m.schedule.agents if a.state == \"S\"),\n"
        "#                              \"Infected\":    lambda m: sum(1 for a in m.schedule.agents if a.state == \"I\"),\n"
        "#                              \"Recovered\":   lambda m: sum(1 for a in m.schedule.agents if a.state == \"R\")},\n"
        "#             # agent_reporters={\"state\": \"state\"} # 如果需要追踪每个agent的状态
"
        "#         )\n"
        "#         self.datacollector.collect(self)
"
        "# \n"
        "#     def step(self):\n"
        "#         self.schedule.step()\n"
        "#         self.datacollector.collect(self)\n"
        "#         # 可以在这里添加停止条件，例如当没有感染者时 self.running = False\n"
        "#         infected_count = self.datacollector.model_vars[\"Infected\"][-1]
"
        "#         if infected_count == 0:
"
        "#            self.running = False
"
    )
    print("\nSIRModel概念代码片段:\n" + sir_model_code)
    print("\n注意: 上述SIR代码是概念性的，需要根据具体场景进行调整和完整实现。例如，空间交互 (通过网格) 或非空间交互 (通过网络或随机混合) 的方式会影响感染逻辑。")

if __name__ == '__main__':
    print("========== (D 部分) Mesa基于代理的建模演示 ==========")
    main_mesa_output_dir = "mesa_outputs_main_demo"
    _ensure_output_directory_api(main_mesa_output_dir) # 主演示的输出目录

    # 1. 运行Schelling隔离模型
    print("\n--- 演示: Schelling隔离模型 (参数组1) ---")
    schelling_data_1, dynamics_plot_1, grid_plot_1 = run_schelling_model_api(
        width=25, height=25, 
        density=0.9, 
        proportion_group0=0.45, 
        homophily_threshold=0.35, 
        max_steps=150, 
        seed_val=101,
        output_dir=main_mesa_output_dir
    )
    if schelling_data_1 is not None:
        print(f"  Schelling模型(1)运行完成。动态图: {dynamics_plot_1}, 网格图: {grid_plot_1}")
        # print(schelling_data_1.tail())

    print("\n--- 演示: Schelling隔离模型 (参数组2, 更高同质性需求) ---")
    schelling_data_2, dynamics_plot_2, grid_plot_2 = run_schelling_model_api(
        width=25, height=25, 
        density=0.9, 
        proportion_group0=0.45, 
        homophily_threshold=0.6, # 更高的同质性需求
        max_steps=150, 
        seed_val=102,
        output_dir=main_mesa_output_dir
    )
    if schelling_data_2 is not None:
        print(f"  Schelling模型(2)运行完成。动态图: {dynamics_plot_2}, 网格图: {grid_plot_2}")
        # print(schelling_data_2.tail())

    # 2. SIR模型概念解释
    explain_sir_model_conceptual_api()

    print(f"\n\n========== (D 部分) Mesa基于代理的建模演示结束 ==========")
    print(f"注意: Mesa模拟图表 (如果生成) 已保存在 '{main_mesa_output_dir}' 目录中。")
