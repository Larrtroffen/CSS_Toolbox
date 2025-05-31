import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 辅助函数：创建输出目录 ---
def _ensure_output_directory_api(dir_name: str = "simpy_outputs_demo") -> str:
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

# --- API 函数 ---

def run_basic_clock_simulation_api(simulation_runtime: float = 2.1) -> list[str]:
    """
    运行一个基础的SimPy时钟模拟示例。
    包含两个时钟进程，以不同间隔打印时间。

    参数:
    - simulation_runtime (float): 模拟运行的总时长。

    返回:
    - list[str]: 模拟过程中打印到控制台的日志消息列表。
    """
    print(f"\n--- (API) 基础SimPy时钟过程示例 (运行 {simulation_runtime} 时间单位) ---")
    logs = []

    def _clock_process(env: simpy.Environment, name: str, tick_interval: float):
        """内部时钟进程函数"""
        while True:
            log_message = f"时间: {env.now:.2f} - {name} 说: Tick!"
            print(log_message)
            logs.append(log_message)
            yield env.timeout(tick_interval)

    env = simpy.Environment()
    env.process(_clock_process(env, name='快速时钟', tick_interval=0.5))
    env.process(_clock_process(env, name='慢速时钟', tick_interval=1.0))
    
    print(f"  开始基础时钟模拟，持续 {simulation_runtime} 时间单位。")
    env.run(until=simulation_runtime)
    final_log = "基础时钟模拟结束。"
    print(f"  {final_log}")
    logs.append(final_log)
    return logs

class MachineShopSimulator:
    """封装机加工车间模拟的逻辑和数据收集。"""
    def __init__(self, env: simpy.Environment, machine_capacity: int, mean_processing_time: float, random_seed: int):
        self.env = env
        self.machine = simpy.Resource(env, capacity=machine_capacity)
        self.mean_processing_time = mean_processing_time
        self.random_seed_processing = random_seed  # Seed for processing times
        # Data collection lists specific to this simulator instance
        self.data_records = [] # List of dictionaries, one per customer processed

    def _customer_process(self, customer_id: str):
        """单个顾客的到达、请求机器、使用和离开的过程。"""
        arrival_time = self.env.now
        # print(f"时间: {arrival_time:.2f} - {customer_id} 到达机加工车间。") # 内部调试信息

        with self.machine.request() as request:
            yield request
            
            service_start_time = self.env.now
            wait_time = service_start_time - arrival_time
            
            # 使用实例特定的随机种子生成器，如果需要更精细控制
            # For now, use np.random which can be seeded globally for the whole simulation run if needed
            # np.random.seed(self.random_seed_processing + customer_id_numeric_part) # More complex seeding
            processing_time_actual = np.random.exponential(self.mean_processing_time)
            
            yield self.env.timeout(processing_time_actual)
            departure_time = self.env.now
            
            self.data_records.append({
                'customer_id': customer_id,
                'arrival_time': arrival_time,
                'service_start_time': service_start_time,
                'wait_time': wait_time,
                'processing_time': processing_time_actual,
                'departure_time': departure_time,
                'system_time': departure_time - arrival_time
            })
            # print(f"时间: {departure_time:.2f} - {customer_id} 离开。等待: {wait_time:.2f}, 加工: {processing_time_actual:.2f}")

    def source_process(self, n_customers: int, mean_interarrival_time: float, random_seed_arrival: int):
        """生成顾客到达机加工车间的过程。"""
        np.random.seed(random_seed_arrival) # Seed for interarrival times
        for i in range(n_customers):
            self.env.process(self._customer_process(f'顾客_{i+1}'))
            interarrival_time_actual = np.random.exponential(mean_interarrival_time)
            yield self.env.timeout(interarrival_time_actual)

    def get_results_df(self) -> pd.DataFrame:
        """将收集到的数据转换为Pandas DataFrame。"""
        return pd.DataFrame(self.data_records)

def simulate_machine_shop_api(
    n_customers: int = 10, 
    mean_interarrival_time: float = 1.0, 
    mean_processing_time: float = 0.8, 
    machine_capacity: int = 1,
    simulation_duration_max: float | None = None, # 可选，最大模拟时长
    output_dir: str = "simpy_outputs_demo",
    plot_results: bool = True,
    random_seed: int = 42
) -> tuple[pd.DataFrame | None, str | None]:
    """
    模拟一个带有特定数量机器的机加工车间。
    顾客以指数分布的间隔时间到达，服务时间也服从指数分布。

    参数:
    - n_customers (int): 要生成的顾客总数。
    - mean_interarrival_time (float): 平均顾客到达间隔时间。
    - mean_processing_time (float): 平均服务时间。
    - machine_capacity (int): 车间中的机器数量 (服务台数量)。
    - simulation_duration_max (float | None): 模拟的最大运行时长。如果None，则运行直到所有顾客处理完毕。
    - output_dir (str): 保存图表的目录名。
    - plot_results (bool): 是否生成并保存等待时间直方图。
    - random_seed (int): 用于模拟中随机过程的种子 (到达和服务时间)。

    返回:
    - tuple[pd.DataFrame | None, str | None]: 
        包含模拟结果 (每个顾客一行) 的DataFrame，以及图表保存路径 (如果生成了图表)。
        如果模拟未产生数据或绘图失败，相应部分可能为None。
    """
    print(f"\n--- (API) SimPy机加工车间模拟 ({machine_capacity}台机器) ---")
    print(f"  参数: {n_customers}顾客, 到达间隔 Exp({mean_interarrival_time}), 服务时间 Exp({mean_processing_time})")
    
    np.random.seed(random_seed) # Set global seed for numpy for this simulation run for reproducibility of processing times inside customer if not seeded per customer
    env = simpy.Environment()
    shop_simulator = MachineShopSimulator(env, machine_capacity, mean_processing_time, random_seed)
    
    # 注意: source_process内部会为到达间隔时间设置其自己的随机种子
    # 如果希望加工时间也用不同于全局的、且可复现的种子，可以在MachineShopSimulator或_customer_process中进一步处理
    env.process(shop_simulator.source_process(n_customers, mean_interarrival_time, random_seed + 1)) # Use offset seed for source
    
    if simulation_duration_max:
        env.run(until=simulation_duration_max)
    else:
        env.run() # 运行直到所有事件处理完毕

    print("  机加工车间模拟结束。")
    results_df = shop_simulator.get_results_df()
    plot_path = None

    if results_df.empty:
        print("  警告: 模拟未收集到数据。")
        return None, None

    num_completed_customers = len(results_df)
    print(f"  收集到 {num_completed_customers} 位顾客的数据。")
    print(f"    平均等待时间: {results_df['wait_time'].mean():.3f}")
    print(f"    最大等待时间: {results_df['wait_time'].max():.3f}")
    print(f"    平均系统内时间: {results_df['system_time'].mean():.3f}")

    if plot_results:
        target_dir = _ensure_output_directory_api(output_dir)
        if not os.path.exists(target_dir):
            print(f"  警告: 输出目录 {target_dir} 不存在，无法保存机加工车间图。")
            return results_df, None
            
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['wait_time'], kde=True, bins=max(10, num_completed_customers//5 if num_completed_customers > 0 else 10))
        plt.title(f'顾客等待时间分布 (N={num_completed_customers}, {machine_capacity}台机器)')
        plt.xlabel('等待时间')
        plt.ylabel('频数')
        plt.grid(True, linestyle='--', alpha=0.7)
        plot_file_name = f"machine_shop_waits_cap{machine_capacity}_cust{n_customers}.png"
        plot_path = os.path.join(target_dir, plot_file_name)
        try:
            plt.savefig(plot_path)
            print(f"  等待时间直方图已保存至: {plot_path}")
        except Exception as e_plot:
            print(f"  保存等待时间直方图失败: {e_plot}")
            plot_path = None
        plt.close()
        
    return results_df, plot_path


def explain_advanced_simpy_features_api():
    """打印关于SimPy更高级功能的概念性解释。"""
    print("\n--- (API) SimPy高级特性概念解释 ---")
    print("- **容器 (Containers):** 用于模拟存储离散或连续数量物质的资源 (例如油箱、容量有限的仓库)。")
    print("  - `simpy.Store(env, capacity=N)`: 用于离散物品的存储 (例如零件、缓冲区中的作业)，支持put/get操作。还有`PriorityStore`。")
    print("  - `simpy.Container(env, capacity=C, init=I)`: 用于连续量 (例如罐中的液体)。支持`get(amount)`和`put(amount)`。")
    print("- **中断 (Interrupts):** 进程可以被其他进程中断 (例如机器故障、更高优先级的任务)。需要仔细处理状态和恢复逻辑。")
    print("  - `process_instance = env.process(...)`")
    print("  - `process_instance.interrupt('原因')` 来中断目标进程。")
    print("  - 在被中断的进程中: `try: yield env.timeout(duration) except simpy.Interrupt as interrupt: print(f'被 {interrupt.cause} 中断')`")
    print("- **监控与数据收集:** SimPy本身是最小化的；数据收集通常通过在进程中附加到Python列表/数组，或创建专用监控进程来观察共享变量或资源状态来完成。")
    print("  - 示例: `env.process(monitor_queue_length(env, resource, interval))` (自定义的监控函数)。")
    print("- **条件等待:** 使用 `env.any_of([event1, event2])` 或 `env.all_of([...])` 等待多个事件中的一个或全部发生。")
    print("- **共享事件 (Shared Events):** `event = env.event()` 可用于进程间的同步。`event.succeed()` 或 `event.fail()` 触发事件，进程通过 `yield event` 等待。")
    print("- **实时模拟 (Real-time Simulation):** SimPy可以通过 `RealtimeEnvironment` 与真实时钟同步运行，但这在典型的DES分析中较少见。")


if __name__ == '__main__':
    print("========== (D 部分) SimPy离散事件模拟演示 ==========")
    main_simpy_output_dir = "simpy_outputs_main_demo"
    _ensure_output_directory_api(main_simpy_output_dir) # 主演示的输出目录

    # 1. 基础时钟模拟
    run_basic_clock_simulation_api(simulation_runtime=1.8)

    # 2. 机加工车间模拟
    print("\n--- 演示: 机加工车间模拟 (1台机器) ---")
    results_df_1_server, plot_1_server_path = simulate_machine_shop_api(
        n_customers=75, 
        mean_interarrival_time=1.0, 
        mean_processing_time=0.8, 
        machine_capacity=1,
        output_dir=main_simpy_output_dir,
        random_seed=101
    )
    if results_df_1_server is not None:
        print(f"  1台机器模拟完成。图表: {plot_1_server_path if plot_1_server_path else '未生成'}")
        # print(results_df_1_server.tail()) # 打印部分数据

    print("\n--- 演示: 机加工车间模拟 (2台机器) ---")
    results_df_2_servers, plot_2_servers_path = simulate_machine_shop_api(
        n_customers=75, 
        mean_interarrival_time=1.0, 
        mean_processing_time=0.8, 
        machine_capacity=2,
        output_dir=main_simpy_output_dir,
        random_seed=202
    )
    if results_df_2_servers is not None:
        print(f"  2台机器模拟完成。图表: {plot_2_servers_path if plot_2_servers_path else '未生成'}")
        # print(results_df_2_servers.tail()) # 打印部分数据

    # 比较平均等待时间 (如果都有结果)
    if results_df_1_server is not None and not results_df_1_server.empty and \
       results_df_2_servers is not None and not results_df_2_servers.empty:
        avg_wait_1 = results_df_1_server['wait_time'].mean()
        avg_wait_2 = results_df_2_servers['wait_time'].mean()
        print(f"\n  结果比较:")
        print(f"    1台机器平均等待时间: {avg_wait_1:.3f}")
        print(f"    2台机器平均等待时间: {avg_wait_2:.3f}")

    # 3. SimPy高级特性概念解释
    explain_advanced_simpy_features_api()
    
    print(f"\n\n========== (D 部分) SimPy离散事件模拟演示结束 ==========")
    print(f"注意: SimPy模拟图表 (如果生成) 已保存在 '{main_simpy_output_dir}' 目录中。") 