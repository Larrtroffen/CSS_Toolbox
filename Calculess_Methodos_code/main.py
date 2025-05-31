# ==============================================================================
# CNCSS Toolbox - 主演示脚本 (Main Demonstration Script)
# ==============================================================================
# 本脚本旨在通过调用各个子模块的演示函数，来展示CNCSS工具箱中不同部分的
# 核心功能和API用法。每个主要部分（A, B, C, D）的演示函数被组织在
# 'main_demos' 子目录下的独立脚本中，以便于管理和维护。
# 
# 所有模块的输出（如图表、报告等）将保存在各自模块指定的输出目录中，
# 例如 'ADescribe/pandas_outputs/', 'BPredict/sklearn_outputs/' 等。
# 请确保在运行前，各个模块的依赖已正确安装。
# ==============================================================================

# 导入各个部分的演示运行器
from main_demos.run_adescribe_demos import run_all_adescribe_demos
from main_demos.run_bpredict_demos import run_all_bpredict_demos
from main_demos.run_ccausal_demos import run_all_ccausal_demos # 包含占位符
from main_demos.run_dsimulate_demos import run_all_dsimulate_demos # 包含占位符

# ==============================================================================
# 主执行块 (Main Execution Block)
# ==============================================================================
if __name__ == "__main__":
    print("欢迎来到 CNCSS Toolbox 演示!")
    print("本程序将依次演示工具箱的各个核心部分。")
    print("=============================================")

    # 依次运行各个部分的演示
    # 用户可以根据需要注释掉不想运行的部分

    # A: 描述性分析与EDA (Descriptive Analysis & EDA)
    run_all_adescribe_demos()

    # B: 预测性建模 (Predictive Modeling)
    run_all_bpredict_demos()

    # C: 因果推断 (Causal Inference)
    run_all_ccausal_demos()

    # D: 模拟 (Simulation)
    run_all_dsimulate_demos()

    print("\\n=============================================")
    print("CNCSS Toolbox 所有选定部分的演示已完成。")
    print("请检查各个模块的输出目录 (例如 'ADescribe/pandas_outputs/', 'BPredict/sklearn_outputs/'等) 查看结果。")
