import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# PySD will be imported if available, otherwise conceptual explanations will be dominant.
_PYSD_AVAILABLE = False
PYSD_MODEL_INSTANCE = None # To hold a loaded model for conceptual examples if needed

try:
    import pysd
    _PYSD_AVAILABLE = True
    print("提示: PySD库已找到。")
except ImportError:
    print("警告: PySD库未找到。此脚本中的PySD相关部分将主要是概念性的。")
    print("要运行PySD示例，请先安装: pip install pysd")

# --- 辅助函数：创建输出目录 ---
def _ensure_output_directory_api(dir_name: str = "pysd_outputs_demo") -> str:
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

def explain_pysd_model_loading_api():
    """提供关于使用PySD加载系统动力学模型的概念性解释。"""
    print("\n--- (API) PySD模型加载概念解释 ---")
    print("PySD允许加载在Vensim (.mdl), XMILE (.xmile), 或用Python定义的模型文件。")
    print("加载Vensim模型文件的示例 (假设有名为 'my_model.mdl' 的文件):")
    load_code = (
        "# if _PYSD_AVAILABLE:\n"
        "#     try:\n"
        "#         # model = pysd.read_vensim('path/to/your/vensim_model.mdl')\n"
        "#         # 或者对于XMILE文件:\n"
        "#         # model = pysd.read_xmile('path/to/your/xmile_model.xml') \n"
        "#         print(\"  概念示例: model = pysd.read_vensim('my_model.mdl')\")\n"
        "#         print(\"  请将 'my_model.mdl' 替换为您的模型文件的实际路径。\")\n"
        "#     except FileNotFoundError:\n"
        "#         print(\"  概念模型文件未找到。请提供有效的模型文件路径。\")\n"
        "#     except Exception as e:\n"
        "#         print(f\"  加载概念模型时出错: {e}\")\n"
        "# else:\n"
        "#     print(\"  PySD库不可用，无法演示模型加载。\")"
    )
    print(load_code)
    print("在本脚本的后续演示中，如果PySD已安装，我们将尝试使用一个临时的 'teacup.mdl' 模型。")

def run_teacup_model_demo_api(output_dir: str = "pysd_outputs_demo", plot_results: bool = True) -> tuple[pd.DataFrame | None, str | None]:
    """
    运行PySD的 "Teacup" (茶杯冷却) 演示模型并绘制结果。
    如果PySD不可用，则只打印概念性步骤。
    该函数会尝试在output_dir中创建一个临时的 'dummy_teacup.mdl' 文件用于演示。

    参数:
    - output_dir (str): 保存图表和临时模型文件的目录。
    - plot_results (bool): 是否绘制并保存结果图表。

    返回:
    - tuple[pd.DataFrame | None, str | None]:
        - 模拟结果的DataFrame (如果成功运行)。
        - 结果图表的路径 (如果成功绘制并保存)。
        如果PySD不可用或运行失败，相应部分可能为None。
    """
    global PYSD_MODEL_INSTANCE
    print("\n--- (API) 运行PySD模型并绘图 (Teacup演示) ---")
    target_dir = _ensure_output_directory_api(output_dir) # 确保主输出目录存在

    if not _PYSD_AVAILABLE:
        print("  PySD库不可用。跳过Teacup模型运行。")
        print("  概念步骤:")
        print("    1. `results = model.run()` 获取Pandas DataFrame格式的模拟结果。")
        print("    2. 使用matplotlib等库可视化 `results`。例如: `results['Teacup Temperature'].plot()`")
        return None, None

    dummy_teacup_mdl_path = os.path.join(target_dir, "dummy_teacup.mdl")
    teacup_vensim_content = (
        "Teacup Temperature = INTEG ( (Room Temperature - Teacup Temperature)/Characteristic Time , Initial Temperature)\n"
        "Units: Degrees F\n"
        "Initial Temperature = 180\n"
        "Units: Degrees F\n"
        "Room Temperature = 70\n"
        "Units: Degrees F\n"
        "Characteristic Time = 10\n"
        "Units: Minutes\n"
        "FINAL TIME = 30\n"
        "Units: Minute\n"
        "INITIAL TIME = 0\n"
        "Units: Minute\n"
        "SAVEPER = TIME STEP\n"
        "Units: Minute\n"
        "TIME STEP = 0.125\n"
        "Units: Minute\n"
        "~~\n"
        "|Sketch Information\n"
        "*View 1\n"
        "$UROOM TEMPERATURE|\n"
        "$NTEACUP TEMPERATURE|\n"
        "$CCHARACTERISTIC TIME|\n"
        "$FINITIAL TEMPERATURE|\n"
    )
    try:
        with open(dummy_teacup_mdl_path, 'w', encoding='utf-8') as f:
            f.write(teacup_vensim_content)
        print(f"  已在 '{target_dir}' 中创建演示用的 'dummy_teacup.mdl' 文件。")

        model = pysd.read_vensim(dummy_teacup_mdl_path)
        PYSD_MODEL_INSTANCE = model # Store for conceptual sensitivity if needed
        print("  Teacup演示模型加载成功。")

        print("  运行Teacup模型...")
        results_df = model.run() 
        print(f"  模拟完成。结果列: {results_df.columns.tolist()}")

        plot_path = None
        if plot_results:
            plt.figure(figsize=(10, 6))
            results_df['Teacup Temperature'].plot(label='茶杯温度')
            
            room_temp_val = model.param_dict().get('Room Temperature', 70) # 从参数获取或用默认值
            plt.axhline(room_temp_val, color='red', linestyle='--', label=f'室温 ({room_temp_val})')
            
            plt.title('茶杯冷却模拟 (PySD)')
            plt.xlabel('时间 (分钟)')
            plt.ylabel('温度 (华氏度)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plot_filename = "pysd_teacup_simulation_demo.png"
            plot_path = os.path.join(target_dir, plot_filename)
            plt.savefig(plot_path)
            print(f"  Teacup模拟图已保存至: {plot_path}")
            plt.close()
        return results_df, plot_path

    except FileNotFoundError:
        print("  错误: 无法找到或加载 (临时的) Teacup模型文件。")
    except Exception as e:
        print(f"  运行/绘制Teacup模型时发生错误: {e}")
    return None, None

def explain_pysd_sensitivity_policy_api():
    """提供关于使用PySD进行灵敏度分析和政策探索的概念性解释。"""
    print("\n--- (API) PySD灵敏度分析与政策探索概念 ---")
    print("PySD允许更改模型参数以探索不同情景或测试政策效果。")
    
    sensitivity_code = (
        "# global PYSD_MODEL_INSTANCE # 假设这个全局变量在之前的API调用中被设置了 (例如 run_teacup_model_demo_api)
"
        "# if _PYSD_AVAILABLE and PYSD_MODEL_INSTANCE is not None:\n"
        "#     model = PYSD_MODEL_INSTANCE
"
        "#     # 示例: 更改Teacup模型的 'Characteristic Time' 参数
"
        "#     original_char_time = model.param_dict().get('Characteristic Time', 10) 
"
        "#     # 情景1: 更快冷却 (特征时间减小)
"
        "#     results_faster_cooling = model.run(params={'Characteristic Time': original_char_time / 2})\n"
        "#     # 情景2: 更慢冷却 (特征时间增大)
"
        "#     results_slower_cooling = model.run(params={'Characteristic Time': original_char_time * 2})\n"
        "# \n"
        "#     # 绘制比较图 (此处仅为概念，实际绘图需定义 output_dir)
"
        "#     # output_dir_sens = _ensure_output_directory_api(\"pysd_outputs_demo\") # 确保目录存在
"
        "#     # plt.figure(figsize=(10, 6))\n"
        "#     # if results_faster_cooling is not None: results_faster_cooling['Teacup Temperature'].plot(label=f'快速冷却 (时间={original_char_time/2})')\n"
        "#     # model.run(params={'Characteristic Time': original_char_time})['Teacup Temperature'].plot(label=f'正常冷却 (时间={original_char_time})') # 重跑原始参数用于绘图
"
        "#     # if results_slower_cooling is not None: results_slower_cooling['Teacup Temperature'].plot(label=f'慢速冷却 (时间={original_char_time*2})')\n"
        "#     # plt.title('灵敏度分析: 特征时间对茶杯冷却的影响') ... plt.show() / plt.savefig(...) \n"
        "#     print(f\"  概念性灵敏度分析图将保存到 pysd_outputs_demo 目录中。\")\n"
        "# else:\n"
        "#     print(\"  PySD库或加载的模型实例不可用，无法演示灵敏度/政策分析。\")"
    )
    print(sensitivity_code)
    print("对于系统性的灵敏度分析 (例如对参数范围进行蒙特卡洛模拟) 或优化，PySD可以与其他Python库 (如 `SALib` (灵敏度) 或 `scipy.optimize`) 集成。")

def explain_pysd_python_based_models_api():
    """提供关于使用PySD在Python中原生构建系统动力学模型的概念性解释。"""
    print("\n--- (API) PySD原生Python建模概念 ---")
    print("PySD除了能运行Vensim/XMILE文件外，也支持直接在Python中定义模型。")
    print("这通常涉及创建一个遵循特定约定的Python文件 (例如 `my_python_model.py`)。")
    
    python_model_example = (
        "# 假设的 `my_python_model.py` 文件内容片段:\n"
        "# \n"
        "# from pysd.py_backend.functions import Integ # 用于存量/流量的示例导入\n"
        "# import numpy as np\n"
        "# \n"
        "# # 使用PySD可识别的特定函数名定义模型组件\n"
        "# _aux_人口增长率 = 0.02 # 示例常量 (如果需要唯一性，可使用非ASCII字符)\n"
        "# def 人口():\n"
        "#     return Integ(净增长(), 初始人口()) # 定义存量 '人口'\n"
        "# \n"
        "# def 初始人口():\n"
        "#     return 1000\n"
        "# \n"
        "# def 净增长():\n"
        "#     return 人口() * _aux_人口增长率 # 定义流量 '净增长'\n"
        "# \n"
        "# # 模型运行控制参数 (可选，通常在Vensim/XMILE中定义)\n"
        "# # FINAL_TIME = 100\n"
        "# # TIME_STEP = 1\n"
        "# \n"
        "# # 然后使用 pysd.load('path/to/my_python_model.py') 加载此模型。"
    )
    print(python_model_example)
    print("关键点是使用 `pysd.py_backend.functions` 中的函数 (如 `Integ`) 和遵循命名约定，以便PySD能够正确解析和翻译模型。")

if __name__ == '__main__':
    print("========== (D 部分) PySD系统动力学建模演示 ==========")
    main_pysd_output_dir = "pysd_outputs_main_demo"
    _ensure_output_directory_api(main_pysd_output_dir) # 主演示的输出目录

    # 1. PySD模型加载概念
    explain_pysd_model_loading_api()

    # 2. 运行Teacup演示模型 (如果PySD可用)
    print("\n--- 演示: 运行Teacup模型 ---")
    teacup_results, teacup_plot = run_teacup_model_demo_api(output_dir=main_pysd_output_dir)
    if teacup_results is not None:
        print(f"  Teacup模型运行成功。数据包含 {len(teacup_results)}行。图表: {teacup_plot}")
    else:
        print("  Teacup模型未运行或未生成结果 (可能是因为PySD不可用)。")

    # 3. PySD灵敏度分析与政策探索概念
    explain_pysd_sensitivity_policy_api()

    # 4. PySD原生Python建模概念
    explain_pysd_python_based_models_api()

    print(f"\n\n========== (D 部分) PySD系统动力学建模演示结束 ==========")
    print(f"注意: PySD模拟图表 (如果生成) 已保存在 '{main_pysd_output_dir}' 目录中。") 