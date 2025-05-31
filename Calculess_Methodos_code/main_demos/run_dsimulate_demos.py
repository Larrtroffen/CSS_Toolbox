import pandas as pd
import numpy as np

# Placeholder for DSimulate module imports
from DSimulate.monte_carlo_simulations import (
    estimate_pi_monte_carlo_api,
    monte_carlo_integration_api,
    _func_x_squared_for_mc, # Helper for integration demo
    simulate_random_walk_api,
    explain_conceptual_queue_simulation_api,
    _ensure_output_directory_api # Utility for directory creation
)
from DSimulate.discrete_event_simulation_simpy import (
    run_basic_clock_simulation_api,
    simulate_machine_shop_api,
    explain_advanced_simpy_features_api,
    # _ensure_output_directory_api is already imported from monte_carlo, 
    # but it's good practice for each module to define/import its needs.
    # If it was different, we'd alias one.
)
from DSimulate.agent_based_modeling_mesa import (
    run_schelling_model_api,
    explain_sir_model_conceptual_api
    # _ensure_output_directory_api is also available from other modules if needed
)
from DSimulate.system_dynamics_pysd import (
    run_teacup_model_demo_api,
    explain_pysd_model_loading_api,
    explain_pysd_sensitivity_policy_api,
    explain_pysd_python_based_models_api
)
from DSimulate.simulation_tools_comparison import (
    generate_simulation_tools_comparison_api
)
# from DSimulate.simulation_tools_comparison import ...

def demo_monte_carlo_simulations():
    print("\n--- Simulation: Monte Carlo (演示) ---")
    # 为此演示部分设置主输出目录
    mc_output_dir = "monte_carlo_outputs_demo"
    _ensure_output_directory_api(mc_output_dir) # 确保目录存在

    # 1. 估计 Pi
    print("  (MC.1) 演示: 估计Pi")
    estimate_pi_monte_carlo_api(n_samples=1000, plot_if_samples_le=1000, output_dir=mc_output_dir, random_seed=42)
    estimate_pi_monte_carlo_api(n_samples=20000, plot_if_samples_le=1000, output_dir=mc_output_dir, random_seed=43)

    # 2. 蒙特卡洛积分
    print("\n  (MC.2) 演示: 蒙特卡洛积分")
    # 示例 a: 积分 f(x) = x^2 从 0 到 1 (真实积分 = 1/3)
    monte_carlo_integration_api(
        _func_x_squared_for_mc, 
        lower_bound=0, upper_bound=1, 
        n_samples=15000, random_seed=123, 
        known_integral_value=1/3
    )
    # 示例 b: 积分 f(x) = sin(x) 从 0 到 pi (真实积分 = 2)
    monte_carlo_integration_api(
        np.sin, 
        lower_bound=0, upper_bound=np.pi, 
        n_samples=15000, random_seed=124,
        known_integral_value=2.0
    )

    # 3. 简单随机游走模拟
    print("\n  (MC.3) 演示: 简单随机游走")
    simulate_random_walk_api(
        n_steps=100, n_simulations=4, initial_value=0, 
        output_dir=mc_output_dir, random_seed=777
    )

    # 4. M/M/1 排队系统概念解释
    explain_conceptual_queue_simulation_api()
    
    print(f"\n  Monte Carlo 模拟图表 (如果生成) 已保存在 '{mc_output_dir}' 目录中。")
    print("--- Monte Carlo 演示结束 ---")

def demo_des_simpy():
    print("\n--- Simulation: Discrete Event Simulation (SimPy) (演示) ---")
    des_output_dir = "simpy_outputs_demo" # Defined in run_all_dsimulate_demos global setup
    # _ensure_output_directory_api(des_output_dir) # Ensure it for this specific demo if not done globally

    # 1. 基础时钟模拟
    print("  (DES.1) 演示: 基础SimPy时钟过程")
    run_basic_clock_simulation_api(simulation_runtime=1.5)

    # 2. 机加工车间模拟
    print("\n  (DES.2) 演示: 机加工车间模拟 (1台机器)")
    df_shop_1_results, plot_1_path = simulate_machine_shop_api(
        n_customers=60,
        mean_interarrival_time=0.9,
        mean_processing_time=0.7,
        machine_capacity=1,
        output_dir=des_output_dir,
        random_seed=111
    )
    if df_shop_1_results is not None:
        print(f"    1台机器模拟数据已收集。图表: {plot_1_path if plot_1_path else '未生成'}")

    print("\n  (DES.3) 演示: 机加工车间模拟 (2台机器)")
    df_shop_2_results, plot_2_path = simulate_machine_shop_api(
        n_customers=60,
        mean_interarrival_time=0.9, 
        mean_processing_time=0.7, 
        machine_capacity=2, 
        output_dir=des_output_dir,
        random_seed=222
    )
    if df_shop_2_results is not None:
        print(f"    2台机器模拟数据已收集。图表: {plot_2_path if plot_2_path else '未生成'}")
    
    # 简单的结果比较
    if df_shop_1_results is not None and not df_shop_1_results.empty and \
       df_shop_2_results is not None and not df_shop_2_results.empty:
        avg_wait_1 = df_shop_1_results['wait_time'].mean()
        avg_wait_2 = df_shop_2_results['wait_time'].mean()
        print(f"\n    结果比较 (平均等待时间): 1台机器={avg_wait_1:.3f}, 2台机器={avg_wait_2:.3f}")

    # 3. SimPy高级特性概念解释
    explain_advanced_simpy_features_api()

    print(f"\n  SimPy 模拟图表 (如果生成) 已保存在 '{des_output_dir}' 目录中。")
    print("--- DES (SimPy) 演示结束 ---")

def demo_abm_mesa():
    print("\n--- Simulation: Agent-Based Modeling (Mesa) (演示) ---")
    abm_output_dir = "mesa_outputs_demo" # Defined in run_all_dsimulate_demos global setup

    # 1. 运行Schelling隔离模型
    print("  (ABM.1) 演示: Schelling隔离模型 (参数组 A)")
    schelling_data_A, dynamics_plot_A, grid_plot_A = run_schelling_model_api(
        width=20, height=20, 
        density=0.85, 
        proportion_group0=0.5, 
        homophily_threshold=0.3, 
        max_steps=100, 
        seed_val=201,
        output_dir=abm_output_dir
    )
    if schelling_data_A is not None:
        print(f"    Schelling模型(A)数据已收集。动态图: {dynamics_plot_A}, 网格图: {grid_plot_A}")

    print("\n  (ABM.2) 演示: Schelling隔离模型 (参数组 B, 更高同质性需求)")
    schelling_data_B, dynamics_plot_B, grid_plot_B = run_schelling_model_api(
        width=20, height=20, 
        density=0.85, 
        proportion_group0=0.5, 
        homophily_threshold=0.55, # 更高的同质性需求
        max_steps=100, 
        seed_val=202,
        output_dir=abm_output_dir
    )
    if schelling_data_B is not None:
        print(f"    Schelling模型(B)数据已收集。动态图: {dynamics_plot_B}, 网格图: {grid_plot_B}")

    # 2. SIR模型概念解释
    explain_sir_model_conceptual_api()
    
    print(f"\n  Mesa (ABM) 模拟图表 (如果生成) 已保存在 '{abm_output_dir}' 目录中。")
    print("--- ABM (Mesa) 演示结束 ---")

def demo_system_dynamics_pysd():
    print("\n--- Simulation: System Dynamics (PySD) (演示) ---")
    pysd_output_dir = "pysd_outputs_demo" # Defined in run_all_dsimulate_demos global setup

    # 1. PySD模型加载概念
    explain_pysd_model_loading_api()

    # 2. 运行Teacup演示模型
    print("\n  (PySD.1) 演示: 运行Teacup演示模型")
    # 注意: run_teacup_model_demo_api 内部会检查 PySD 是否可用
    teacup_results, teacup_plot_path = run_teacup_model_demo_api(output_dir=pysd_output_dir)
    if teacup_results is not None:
        print(f"    Teacup模型演示运行成功。图表: {teacup_plot_path if teacup_plot_path else '未生成'}")
    else:
        print("    Teacup模型演示未运行或失败 (可能PySD不可用或配置问题)。")

    # 3. PySD灵敏度分析与政策探索概念
    explain_pysd_sensitivity_policy_api()

    # 4. PySD原生Python建模概念
    explain_pysd_python_based_models_api()
    
    print(f"\n  PySD 模拟图表 (如果生成) 已保存在 '{pysd_output_dir}' 目录中。")
    print("--- System Dynamics (PySD) 演示结束 ---")

def demo_simulation_tools_comparison_conceptual():
    print("\n--- Simulation: 不同模拟工具对比概念解说 ---")
    # 此API函数会在内部处理其输出目录 (默认为 "simulation_docs_demo")
    # 但为了与此演示文件的结构保持一致，我们可以传递一个特定的目录名
    comparison_output_dir = "simulation_docs_demo" # 与 run_all_dsimulate_demos 中创建的目录对应
    # _ensure_output_directory_api(comparison_output_dir) # generate_simulation_tools_comparison_api 会自己创建
    
    saved_file = generate_simulation_tools_comparison_api(output_dir=comparison_output_dir)
    if saved_file:
        print(f"  模拟工具对比概念文档已生成并保存至: {saved_file}")
    else:
        print("  模拟工具对比概念文档生成失败。")
    print("--- 工具比较概念演示结束 ---")

def run_all_dsimulate_demos():
    """运行 DSimulate 部分的所有演示函数。"""
    print("========== 开始 D: 模拟 演示 ==========")
    
    # 确保输出目录存在 (如果需要)
    # output_dirs_d = ["monte_carlo_outputs_demo", "simpy_outputs_demo", "mesa_outputs_demo", "pysd_outputs_demo"] # 示例
    import os # 移到这里，因为 _ensure_output_directory_api 可能在各demo函数中被调用
    # monte_carlo_outputs_demo 会由其自己的demo函数中的 _ensure_output_directory_api 创建
    # 其他模块的演示函数也应该负责创建自己的输出目录，或统一在此创建
    
    # 创建 DSimulate 下所有演示可能需要的顶级输出目录
    # 对于 monte_carlo_outputs_demo，它会在 demo_monte_carlo_simulations 内部通过 _ensure_output_directory_api 创建
    # 其他演示也应遵循此模式或在此处统一创建
    demo_output_dirs_to_ensure = [
        "simpy_outputs_demo", 
        "mesa_outputs_demo", 
        "pysd_outputs_demo",
        "simulation_docs_demo" # 改名以匹配新的比较脚本输出
    ]
    for out_dir in demo_output_dirs_to_ensure:
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
                print(f"创建演示输出目录: {out_dir}")
            except OSError as e:
                print(f"创建目录 {out_dir} 失败: {e}")

    demo_monte_carlo_simulations()
    demo_des_simpy()
    demo_abm_mesa()
    demo_system_dynamics_pysd()
    demo_simulation_tools_comparison_conceptual()

    print("========== D: 模拟 演示结束 ==========\n\n")

if __name__ == '__main__':
    run_all_dsimulate_demos() 