import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 辅助函数：创建输出目录 ---
def _ensure_output_directory_api(dir_name: str = "monte_carlo_outputs_demo") -> str:
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
            # 如果目录创建失败，可以考虑返回 None 或抛出异常，取决于后续逻辑
    return dir_name

# --- API 函数 ---

def estimate_pi_monte_carlo_api(
    n_samples: int = 10000, 
    plot_if_samples_le: int = 2000, 
    output_dir: str = "monte_carlo_outputs_demo", 
    random_seed: int = 42
) -> tuple[float, str | None]:
    """
    使用蒙特卡洛方法估计圆周率 Pi。
    通过在一个1x1的正方形内随机投点，计算落在半径为1的四分之一圆内的点的比例。

    参数:
    - n_samples (int): 投点的总数量。
    - plot_if_samples_le (int): 如果样本数小于等于此值，则生成并保存散点图。
    - output_dir (str): 保存图表的目录名。
    - random_seed (int): 随机数种子。

    返回:
    - tuple[float, str | None]: 估计的Pi值，以及图表保存路径 (如果生成了图表)。
    """
    print(f"\n--- (API) 使用蒙特卡洛估计Pi ({n_samples} 个样本) ---")
    np.random.seed(random_seed)
    
    x_coords = np.random.uniform(0, 1, n_samples)
    y_coords = np.random.uniform(0, 1, n_samples)

    distance_squared = x_coords**2 + y_coords**2
    points_inside_circle = distance_squared <= 1
    n_inside_circle = np.sum(points_inside_circle)

    estimated_pi = 4 * (n_inside_circle / n_samples)
    print(f"  估计的Pi值: {estimated_pi:.6f}")
    print(f"  Numpy中的Pi值: {np.pi:.6f}")
    print(f"  绝对误差: {abs(np.pi - estimated_pi):.6f}")

    plot_path = None
    if n_samples <= plot_if_samples_le:
        target_dir = _ensure_output_directory_api(output_dir)
        if not os.path.exists(target_dir): # 再次检查，因为 _ensure_output_directory_api 可能失败但不抛异常
            print(f"  警告: 输出目录 {target_dir} 不存在，无法保存Pi估计图。")
            return estimated_pi, None
            
        plt.figure(figsize=(6, 6))
        plt.scatter(x_coords[points_inside_circle], y_coords[points_inside_circle], color='blue', s=5, label='圆内点')
        plt.scatter(x_coords[~points_inside_circle], y_coords[~points_inside_circle], color='red', s=5, label='圆外点')
        theta = np.linspace(0, np.pi/2, 100)
        plt.plot(np.cos(theta), np.sin(theta), color='black', linestyle='--', label='1/4圆边界 (r=1)')
        plt.title(f'蒙特卡洛估计Pi ({n_samples} 点)')
        plt.xlabel('X轴')
        plt.ylabel('Y轴')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plot_path = os.path.join(target_dir, f"monte_carlo_pi_estimation_{n_samples}.png")
        try:
            plt.savefig(plot_path)
            print(f"  Pi估计图已保存至: {plot_path}")
        except Exception as e_plot:
            print(f"  保存Pi估计图失败: {e_plot}")
            plot_path = None # 保存失败则路径无效
        plt.close()
        
    return estimated_pi, plot_path

# 用于蒙特卡洛积分示例的函数
def _func_x_squared_for_mc(x: np.ndarray) -> np.ndarray:
    """计算 x 的平方。"""
    return x**2

def monte_carlo_integration_api(
    func_to_integrate: callable, 
    lower_bound: float, 
    upper_bound: float, 
    n_samples: int = 10000, 
    random_seed: int = 123,
    known_integral_value: float | None = None # 可选，用于计算误差
) -> float:
    """
    使用蒙特卡洛方法估计一元函数在给定区间上的定积分。
    积分公式: ∫f(x)dx [a,b] ≈ (b-a)/N * Σf(x_i)，其中 x_i 在 [a,b] 上均匀采样。

    参数:
    - func_to_integrate (callable): 要积分的函数，接受一个numpy数组并返回一个numpy数组。
    - lower_bound (float): 积分下限。
    - upper_bound (float): 积分上限。
    - n_samples (int): 采样点数量。
    - random_seed (int): 随机数种子。
    - known_integral_value (float | None): 函数在该区间上的真实积分值 (如果已知)，用于比较。

    返回:
    - float: 估计的积分值。
    """
    print(f"\n--- (API) 蒙特卡洛积分 ({n_samples} 个样本) ---")
    print(f"  函数: {func_to_integrate.__name__ if hasattr(func_to_integrate, '__name__') else '匿名函数'}")
    print(f"  区间: [{lower_bound}, {upper_bound}]")
    np.random.seed(random_seed)
    
    random_samples_x = np.random.uniform(lower_bound, upper_bound, n_samples)
    function_values_at_samples = func_to_integrate(random_samples_x)
    
    integral_estimate = (upper_bound - lower_bound) * np.mean(function_values_at_samples)
    print(f"  估计的积分值: {integral_estimate:.6f}")

    if known_integral_value is not None:
        print(f"  已知的真实积分值: {known_integral_value:.6f}")
        print(f"  绝对误差: {abs(known_integral_value - integral_estimate):.6f}")
        
    return integral_estimate


def simulate_random_walk_api(
    n_steps: int = 100, 
    n_simulations: int = 5, 
    initial_value: float = 0,
    output_dir: str = "monte_carlo_outputs_demo",
    random_seed: int = 777 
) -> tuple[pd.DataFrame, str | None]:
    """
    模拟指定数量和步数的一维简单随机游走。
    每一步以等概率向正向或负向移动1个单位。

    参数:
    - n_steps (int): 每次模拟的步数。
    - n_simulations (int): 模拟的次数 (路径数量)。
    - initial_value (float): 随机游走的初始值。
    - output_dir (str): 保存图表的目录名。
    - random_seed (int): 随机数种子。

    返回:
    - tuple[pd.DataFrame, str | None]: 包含所有模拟路径的DataFrame (每列是一次模拟)，以及图表保存路径。
    """
    print(f"\n--- (API) 模拟简单一维随机游走 ({n_simulations} 次模拟, 每次 {n_steps} 步) ---")
    np.random.seed(random_seed)
    all_walks_data = {}

    for i in range(n_simulations):
        steps = np.random.choice([-1, 1], size=n_steps)
        walk = np.concatenate(([initial_value], initial_value + np.cumsum(steps)))
        all_walks_data[f'Walk_{i+1}'] = walk
    
    all_walks_df = pd.DataFrame(all_walks_data)
    plot_path = None
    
    target_dir = _ensure_output_directory_api(output_dir)
    if not os.path.exists(target_dir):
        print(f"  警告: 输出目录 {target_dir} 不存在，无法保存随机游走图。")
        return all_walks_df, None

    plt.figure(figsize=(10, 6))
    for col_name in all_walks_df.columns:
        plt.plot(all_walks_df.index, all_walks_df[col_name], label=col_name)
    
    plt.title(f'{n_simulations} 次简单一维随机游走 (每条 {n_steps} 步)')
    plt.xlabel('步数 (时间)')
    plt.ylabel('位置')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(target_dir, "simple_random_walks_demo.png")
    try:
        plt.savefig(plot_path)
        print(f"  随机游走图已保存至: {plot_path}")
    except Exception as e_plot:
        print(f"  保存随机游走图失败: {e_plot}")
        plot_path = None
    plt.close()
    
    return all_walks_df, plot_path


def explain_conceptual_queue_simulation_api():
    """打印关于简单 M/M/1 排队系统概念性模拟的解释。"""
    print("\n--- (API) M/M/1排队系统模拟概念解释 ---")
    print("M/M/1 排队系统特征:")
    print("  - M (Memoryless/Markovian): 顾客到达过程服从泊松分布 (到达间隔时间服从指数分布)。")
    print("  - M (Memoryless/Markovian): 服务时间服从指数分布。")
    print("  - 1: 系统中只有一个服务台。")
    print("关键参数:")
    print("  - λ (lambda): 平均到达率 (例如，每分钟到达的顾客数)。")
    print("  - μ (mu): 平均服务率 (例如，每个服务台每分钟能服务的顾客数)。")
    print("  - ρ (rho) = λ / μ: 系统负荷强度或服务台利用率。系统稳定的条件是 ρ < 1。")
    print("模拟目的: 理解队列长度、等待时间、服务台繁忙程度等性能指标。")
    
    queue_sim_code_conceptual = (
        "# M/M/1排队概念性Python代码片段 (仅为说明，非完整库实现):\n"
        "# def simulate_mm1_conceptual(lambda_arrival, mu_service, max_events=1000):\n"
        "#     np.random.seed(101)\n"
        "#     interarrivals = np.random.exponential(1.0/lambda_arrival, max_events)\n"
        "#     services = np.random.exponential(1.0/mu_service, max_events)\n"
        "#     \n"
        "#     arrival_times = np.cumsum(interarrivals)\n"
        "#     departure_times = np.zeros(max_events)\n"
        "#     server_free_at = 0.0 # 服务器何时空闲\n"
        "#     waits = []\n"
        "#     \n"
        "#     for i in range(max_events):\n"
        "#         arrival_time = arrival_times[i]\n"
        "#         service_time = services[i]\n"
        "#         \n"
        "#         start_service_time = max(arrival_time, server_free_at)\n"
        "#         waits.append(start_service_time - arrival_time)\n"
        "#         departure_times[i] = start_service_time + service_time\n"
        "#         server_free_at = departure_times[i]\n"
        "#     \n"
        "#     avg_wait = np.mean(waits)\n"
        "#     print(f\"  模拟得到的平均等待时间: {avg_wait:.4f}\")\n"
        "#     # 理论值 (M/M/1): Wq = (λ / (μ * (μ - λ))) if μ > λ else infinity\n"
        "#     if mu_service > lambda_arrival:\n"
        "#         rho_val = lambda_arrival / mu_service\n"
        "#         theoretical_wq = rho_val / (mu_service - lambda_arrival) # 另一种 Wq 公式
        "#         # theoretical_wq = (lambda_arrival / (mu_service * (mu_service - lambda_arrival)))
"
        "#         print(f\"  理论平均等待时间 (Wq): {theoretical_wq:.4f}\")\n"
        "#     return waits, departure_times\n"
        "# \n"
        "# # 示例调用:\n"
        "# # LAMBDA_RATE = 0.8 # 例如，每单位时间到达0.8个顾客\n"
        "# # MU_RATE = 1.0   # 例如，每单位时间服务1个顾客\n"
        "# # if LAMBDA_RATE < MU_RATE:\n"
        "# #     sim_waits, _ = simulate_mm1_conceptual(LAMBDA_RATE, MU_RATE, max_events=5000)\n"
        "# # else:\n"
        "# #     print(f\"  队列不稳定 (λ >= μ: {LAMBDA_RATE} >= {MU_RATE})，平均等待时间可能无意义。\")\n"
    )
    print(queue_sim_code_conceptual)
    print("对于更复杂的排队系统或离散事件模拟，推荐使用如 `SimPy` 这样的专用库。")


if __name__ == '__main__':
    print("========== (D 部分) 蒙特卡洛模拟演示 ==========")
    # 为演示创建一个特定的主输出目录
    main_demo_output_dir = "monte_carlo_outputs_main_demo"
    _ensure_output_directory_api(main_demo_output_dir) # 确保目录存在

    # 1. 估计Pi
    print("\n--- 演示: 估计Pi --- ")
    estimate_pi_monte_carlo_api(n_samples=1500, output_dir=main_demo_output_dir, random_seed=420) # 用于绘图
    estimate_pi_monte_carlo_api(n_samples=50000, output_dir=main_demo_output_dir, random_seed=421) # 不绘图，仅计算

    # 2. 蒙特卡洛积分
    print("\n--- 演示: 蒙特卡洛积分 --- ")
    # 示例1: 积分 f(x) = x^2 从 0 到 1 (真实值为 1/3)
    monte_carlo_integration_api(
        _func_x_squared_for_mc, 
        lower_bound=0, upper_bound=1, 
        n_samples=20000, random_seed=1230, 
        known_integral_value=1/3
    )
    # 示例2: 积分 f(x) = sin(x) 从 0 到 pi (真实值为 2)
    monte_carlo_integration_api(
        np.sin, 
        lower_bound=0, upper_bound=np.pi, 
        n_samples=20000, random_seed=1231,
        known_integral_value=2.0
    )

    # 3. 简单随机游走模拟
    print("\n--- 演示: 简单随机游走 --- ")
    simulate_random_walk_api(
        n_steps=150, n_simulations=6, initial_value=5, 
        output_dir=main_demo_output_dir, random_seed=7770
    )

    # 4. M/M/1排队系统概念解释
    explain_conceptual_queue_simulation_api()

    print(f"\n\n========== (D 部分) 蒙特卡洛模拟演示结束 ==========")
    print(f"注意: 蒙特卡洛模拟图表 (如果生成) 已保存在 '{main_demo_output_dir}' 目录中。") 