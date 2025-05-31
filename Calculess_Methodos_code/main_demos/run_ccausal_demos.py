import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # For CausalML demo data split
import os # For creating output directories

# Placeholder for CCausal module imports
from CCausal.dowhy_example import (
    generate_dowhy_data_api,
    define_causal_model_api,
    identify_causal_effect_api,
    estimate_causal_effect_api,
    refute_causal_estimate_api
)
from CCausal.econml_example import (
    generate_heterogeneous_data_api,
    train_evaluate_lineardml_api,
    train_evaluate_causalforestdml_api,
    train_evaluate_xlearner_econml_api,
    plot_cate_comparison_api
)
from CCausal.causalml_example import (
    generate_uplift_data_api,
    train_evaluate_uplift_tree_api,
    train_evaluate_uplift_rf_api,
    train_evaluate_slearner_api,
    train_evaluate_tlearner_api,
    train_evaluate_xlearner_cml_api,
    evaluate_uplift_predictions_api
)
from CCausal.other_causal_methods_conceptual import (
    generate_did_data_api,
    run_did_example_api,
    explain_did_conceptual_api,
    explain_rdd_conceptual_api,
    explain_causal_discovery_conceptual_api,
    explain_mediation_analysis_conceptual_api
)

def demo_dowhy_functionalities():
    print("\n--- Causal Inference: DoWhy (演示) ---")
    output_directory_dowhy_demo = "dowhy_outputs_demo"
    # 确保输出目录存在 (统一在 run_all_ccausal_demos 中创建)

    # --- 示例1: 后门调整 (连续结果, 连续处理) ---
    print("\n  (DoWhy.1) 后门调整 (连续Y, 连续T)")
    df_ex1, w_ex1, z_ex1, t_ex1, y_ex1 = generate_dowhy_data_api(seed=42, treatment_type='continuous', outcome_type='continuous')
    
    graph_ex1_dot = f"digraph {{ {w_ex1}; {z_ex1}; {t_ex1}; {y_ex1}; {w_ex1} -> {t_ex1}; {w_ex1} -> {y_ex1}; {z_ex1} -> {t_ex1}; {t_ex1} -> {y_ex1}; }}"
    model_ex1 = define_causal_model_api(df_ex1, treatment_name=t_ex1, outcome_name=y_ex1, graph_dot_string=graph_ex1_dot)

    if model_ex1:
        estimand_ex1 = identify_causal_effect_api(model_ex1)
        if estimand_ex1:
            estimate_ex1 = estimate_causal_effect_api(model_ex1, estimand_ex1, method_name="backdoor.linear_regression")
            if estimate_ex1:
                print(f"    估计ATE ({t_ex1} -> {y_ex1}): {estimate_ex1.value:.4f} (预期~2.0)")
                refute_causal_estimate_api(model_ex1, estimand_ex1, estimate_ex1, method_name="random_common_cause")
                # 可以添加更多反驳器，例如 placebo_treatment_refuter，如果适用
                # refute_causal_estimate_api(model_ex1, estimand_ex1, estimate_ex1, method_name="placebo_treatment_refuter", placebo_type="permute")

    # --- 示例2: 工具变量法 (连续结果, 连续处理) ---
    print("\n  (DoWhy.2) 工具变量法 (连续Y, 连续T)")
    # 使用与示例1相同的数据集 df_ex1, t_ex1, y_ex1, z_ex1, w_ex1
    # 假设 W 是未观测的混杂，我们使用 Z 作为工具变量
    # 注意：IV方法通常需要较大的样本量和强工具变量才能获得可靠结果
    # graph_ex2_dot 与 graph_ex1_dot 相同，关键在于如何指定模型中的角色
    model_ex2 = define_causal_model_api(df_ex1, treatment_name=t_ex1, outcome_name=y_ex1, 
                                      instrument_names=[z_ex1], # 指定Z为工具变量
                                      graph_dot_string=graph_ex1_dot # 依然提供完整图
                                     )
    if model_ex2:
        estimand_ex2 = identify_causal_effect_api(model_ex2)
        if estimand_ex2:
            estimate_ex2 = estimate_causal_effect_api(model_ex2, estimand_ex2, method_name="iv.instrumental_variable")
            if estimate_ex2:
                print(f"    估计ATE ({t_ex1} -> {y_ex1} using IV {z_ex1}): {estimate_ex2.value:.4f} (预期~2.0)")
                refute_causal_estimate_api(model_ex2, estimand_ex2, estimate_ex2, method_name="dummy_outcome_refuter", effect_strength_on_outcome=0.01)

    # --- 示例3: 后门调整 (二元结果, 二元处理) ---
    print("\n  (DoWhy.3) 后门调整 (二元Y, 二元T) - 倾向性得分匹配")
    df_ex3, w_ex3, z_ex3_unused, t_ex3, y_ex3 = generate_dowhy_data_api(seed=43, treatment_type='binary', outcome_type='binary')
    graph_ex3_dot = f"digraph {{ {w_ex3}; {t_ex3}; {y_ex3}; {w_ex3} -> {t_ex3}; {w_ex3} -> {y_ex3}; {t_ex3} -> {y_ex3}; }}"
    model_ex3 = define_causal_model_api(df_ex3, treatment_name=t_ex3, outcome_name=y_ex3, graph_dot_string=graph_ex3_dot)

    if model_ex3:
        estimand_ex3 = identify_causal_effect_api(model_ex3)
        if estimand_ex3:
            estimate_ex3_psm = estimate_causal_effect_api(model_ex3, estimand_ex3, method_name="backdoor.propensity_score_matching")
            if estimate_ex3_psm:
                print(f"    估计ATE (PSM, {t_ex3} -> {y_ex3}): {estimate_ex3_psm.value:.4f}")
                refute_causal_estimate_api(model_ex3, estimand_ex3, estimate_ex3_psm, "data_subset_refuter", subset_fraction=0.8)
    
    print(f"\n  注意: DoWhy演示的图表 (如果生成并配置了保存) 可能在 '{output_directory_dowhy_demo}' 或工作目录。")
    print("--- DoWhy演示结束 ---")

def demo_econml_functionalities():
    print("\n--- Causal Inference: EconML (CATE Estimation) ---")
    
    # EconML.0: 设置和数据生成
    main_seed_econml_demo = 123
    num_samples_econml_demo = 1500 # 演示用较少样本
    num_features_econml_demo = 4
    num_instruments_econml_demo = 1
    output_directory_econml_demo = "econml_outputs_demo" # 演示特定的输出目录

    # 确保输出目录存在 (统一在 run_all_ccausal_demos 中创建)

    print("  (EconML.0) 生成异质效应数据...")
    df_econml_data, x_cols, w_cols, z_cols, t_col, y_col = generate_heterogeneous_data_api(
        n_samples=num_samples_econml_demo, 
        n_features=num_features_econml_demo,
        n_instruments=num_instruments_econml_demo,
        seed=main_seed_econml_demo,
        effect_type='nonlinear' # 使用非线性效应测试模型
    )
    print(f"  EconML演示数据已生成: {df_econml_data.shape}")
    print(f"    X特征: {x_cols}, W特征: {w_cols}, Z特征: {z_cols}, 处理: {t_col}, 结果: {y_col}")

    # EconML.1: 数据分割
    print("\n  (EconML.1) 数据分割...")
    df_train_econml, df_test_econml = train_test_split(
        df_econml_data, test_size=0.4, random_state=main_seed_econml_demo + 1
    )
    print(f"  训练集大小: {df_train_econml.shape}, 测试集大小: {df_test_econml.shape}")

    # EconML.2: 模型训练与评估
    print("\n  (EconML.2) 训练模型并收集预测...")
    df_test_preds_collected_econml = df_test_econml.copy()

    # 2.1 LinearDML
    print("    (EconML.2.1) 训练 LinearDML...")
    _, df_test_preds_collected_econml = train_evaluate_lineardml_api(
        df_train_econml, df_test_preds_collected_econml,
        Y_col=y_col, T_col=t_col, X_cols=x_cols, W_cols=w_cols,
        model_y_params={'n_estimators': 25, 'max_depth': 2, 'random_state': main_seed_econml_demo},
        model_t_params={'n_estimators': 25, 'max_depth': 2, 'random_state': main_seed_econml_demo},
        random_state=main_seed_econml_demo
    )

    # 2.2 CausalForestDML
    print("    (EconML.2.2) 训练 CausalForestDML...")
    _, df_test_preds_collected_econml = train_evaluate_causalforestdml_api(
        df_train_econml, df_test_preds_collected_econml,
        Y_col=y_col, T_col=t_col, X_cols=x_cols, W_cols=w_cols,
        model_y_params={'n_estimators': 30, 'max_depth': 3, 'random_state': main_seed_econml_demo},
        model_t_params={'n_estimators': 30, 'max_depth': 3, 'random_state': main_seed_econml_demo},
        cf_params={'n_estimators': 50, 'min_samples_leaf': 10, 'max_depth': 4, 'random_state': main_seed_econml_demo},
        random_state=main_seed_econml_demo
    )

    # 2.3 X-Learner (EconML)
    print("    (EconML.2.3) 训练 X-Learner (EconML)...")
    _, df_test_preds_collected_econml = train_evaluate_xlearner_econml_api(
        df_train_econml, df_test_preds_collected_econml,
        Y_col=y_col, T_col=t_col, X_cols=x_cols,
        model_regression_params={'n_estimators': 25, 'max_depth': 2, 'random_state': main_seed_econml_demo},
        model_propensity_params={'n_estimators': 25, 'max_depth': 2, 'random_state': main_seed_econml_demo}
    )
    
    # EconML.3: 绘制CATE比较图
    print("\n  (EconML.3) 绘制CATE图表...")
    econml_pred_cols_demo = [col for col in df_test_preds_collected_econml.columns if col.startswith('cate_') and col != 'true_CATE']
    
    if 'true_CATE' in df_test_preds_collected_econml.columns and econml_pred_cols_demo:
        plot_cate_comparison_api(
            df_test_preds_collected_econml,
            true_cate_col='true_CATE',
            predicted_cate_cols=econml_pred_cols_demo,
            output_dir=output_directory_econml_demo,
            prefix="demo_econml_cate_comparison"
        )
    else:
        print("    未能生成CATE比较图 (可能缺少 'true_CATE' 或有效的预测列)。")

    print(f"  注意: EconML演示评估图表 (如果成功) 应保存在 '{output_directory_econml_demo}' 目录中。")
    print("--- EconML (CATE Estimation) 演示结束 ---")

def demo_causalml_functionalities():
    print("\n--- Causal Inference: CausalML (Uplift Modeling) ---")
    
    # CML.0: 设置和数据生成
    main_seed_cml_demo = 42
    num_samples_cml_demo = 2000 # 减少样本量以便快速演示
    output_directory_cml_demo = "causalml_outputs_demo" # 演示特定的输出目录

    # 确保输出目录存在 (在run_all_ccausal_demos中更集中地创建)
    # if not os.path.exists(output_directory_cml_demo):
    #     os.makedirs(output_directory_cml_demo)
    #     print(f"  创建CausalML演示输出目录: {output_directory_cml_demo}")

    print("  (CML.0) 生成Uplift数据...")
    df_uplift_cml, feature_cols_cml = generate_uplift_data_api(
        n_samples=num_samples_cml_demo, random_seed=main_seed_cml_demo
    )
    print(f"  CausalML演示数据已生成: {df_uplift_cml.shape}")
    print(f"  特征列: {feature_cols_cml}")

    # CML.1: 数据分割
    print("\n  (CML.1) 数据分割...")
    df_train_cml, df_test_cml = train_test_split(
        df_uplift_cml, test_size=0.4, random_state=main_seed_cml_demo + 1
    )
    print(f"  训练集大小: {df_train_cml.shape}, 测试集大小: {df_test_cml.shape}")

    # CML.2: 模型训练与预测收集
    print("\n  (CML.2) 训练模型并收集预测...")
    df_test_preds_collected_cml = df_test_cml.copy()

    # 2.1 Uplift Tree
    print("    (CML.2.1) 训练 Uplift Tree...")
    _, df_test_preds_collected_cml = train_evaluate_uplift_tree_api(
        df_train_cml, df_test_preds_collected_cml, feature_cols_cml, random_state=main_seed_cml_demo
    )
    # 2.2 Uplift Random Forest
    print("    (CML.2.2) 训练 Uplift Random Forest...")
    _, df_test_preds_collected_cml = train_evaluate_uplift_rf_api(
        df_train_cml, df_test_preds_collected_cml, feature_cols_cml, random_state=main_seed_cml_demo, n_estimators=20 # 减少estimators
    )
    # 2.3 S-Learner
    print("    (CML.2.3) 训练 S-Learner...")
    _, df_test_preds_collected_cml = train_evaluate_slearner_api(
        df_train_cml, df_test_preds_collected_cml, feature_cols_cml, 
        base_learner_params={'n_estimators':20, 'max_depth':3, 'random_state':main_seed_cml_demo}
    )
    # 2.4 T-Learner
    print("    (CML.2.4) 训练 T-Learner...")
    _, df_test_preds_collected_cml = train_evaluate_tlearner_api(
        df_train_cml, df_test_preds_collected_cml, feature_cols_cml,
        base_learner_params={'n_estimators':20, 'max_depth':3, 'random_state':main_seed_cml_demo}
    )
    # 2.5 X-Learner (CausalML)
    print("    (CML.2.5) 训练 X-Learner (CausalML)...")
    _, df_test_preds_collected_cml = train_evaluate_xlearner_cml_api(
        df_train_cml, df_test_preds_collected_cml, feature_cols_cml,
        outcome_learner_params={'n_estimators':10, 'max_depth':2, 'random_state':main_seed_cml_demo},
        effect_learner_params={'n_estimators':10, 'max_depth':2, 'random_state':main_seed_cml_demo},
        propensity_learner_params={'random_state':main_seed_cml_demo, 'solver':'liblinear'}
    )
    
    # CML.3: 评估预测
    print("\n  (CML.3) 评估所有模型的预测...")
    col_for_plotting_cml = 'uplift_rf' if 'uplift_rf' in df_test_preds_collected_cml.columns else None
    if not col_for_plotting_cml:
        available_pred_cols_cml = [c for c in df_test_preds_collected_cml.columns if c.startswith('uplift_')]
        if available_pred_cols_cml: col_for_plotting_cml = available_pred_cols_cml[0]

    all_eval_metrics_cml = evaluate_uplift_predictions_api(
        df_test_preds_collected_cml, 
        output_dir=output_directory_cml_demo, # 使用演示特定的输出目录
        plot_metrics_for_col=col_for_plotting_cml 
    )

    print("\n  所有CausalML模型在演示数据上的评估指标汇总:")
    if all_eval_metrics_cml:
        for model_pred_col, metrics in all_eval_metrics_cml.items():
            print(f"    模型 ({model_pred_col}): {metrics}")
    else:
        print("    未能获取CausalML评估指标。")

    print(f"  注意: CausalML演示评估图表 (如果成功) 应保存在 '{output_directory_cml_demo}' 目录中。")
    print("--- CausalML (Uplift Modeling) 演示结束 ---")

def demo_other_causal_conceptual():
    print("\n--- Causal Inference: 其他方法概念与DiD示例 ---")
    output_directory_conceptual_demo = "other_causal_outputs_demo"
    # 确保输出目录存在 (统一在 run_all_ccausal_demos 中创建)

    # 1. DiD 演示
    explain_did_conceptual_api() # 先解释概念
    print("\n  --- 运行 DiD 实际示例 (API 调用) ---")
    df_did_data_demo = generate_did_data_api(n_per_group=120, seed=101) # 演示用数据
    print(f"    已生成DiD演示数据，形状: {df_did_data_demo.shape}")
    
    did_model_results, did_plot_path = run_did_example_api(
        df_did_data_demo, 
        output_dir=output_directory_conceptual_demo
    )
    if did_model_results:
        print(f"    DiD模型已成功拟合。图表 (如果生成) 保存于: {did_plot_path if did_plot_path else '未生成'}")
    else:
        print("    DiD示例运行失败或未生成模型/图表。")
    print("  --- DiD实际示例结束 ---")

    # 2. RDD 概念解释
    explain_rdd_conceptual_api()

    # 3. 因果发现 概念解释
    explain_causal_discovery_conceptual_api()

    # 4. 中介分析 概念解释
    explain_mediation_analysis_conceptual_api()
    
    print(f"\n  注意: 其他因果方法演示的图表 (如DiD图) (如果成功) 应保存在 '{output_directory_conceptual_demo}' 目录中。")
    print("--- 其他方法概念与DiD示例演示结束 ---")

def run_all_ccausal_demos():
    """运行 CCausal 部分的所有演示函数。"""
    print("========== 开始 C: 因果推断 演示 ==========")
    
    # 统一创建输出目录
    output_dirs_c = ["causalml_outputs_demo", "econml_outputs_demo", "dowhy_outputs_demo", "other_causal_outputs_demo"]
    
    for out_dir in output_dirs_c:
        # 最好是在项目根目录下创建这些目录，或者相对于 Calculess_Methodos_code
        # 为简单起见，这里我们假设在当前工作目录下创建
        # 更健壮的做法是使用 Calculess_Methodos_code/CCausal/causalml_outputs_demo
        # target_dir_path = os.path.join("CCausal", out_dir) # 假设脚本从 Calculess_Methodos_code 运行
        # 如果脚本从 main_demos 运行，路径需要调整
        if not os.path.exists(out_dir): # 为了简单，先在当前执行目录下创建
            try:
                os.makedirs(out_dir)
                print(f"创建输出目录: {out_dir}")
            except OSError as e:
                print(f"创建目录 {out_dir} 失败: {e}. 请检查权限或路径。")
            
    demo_dowhy_functionalities() 
    demo_econml_functionalities()
    demo_causalml_functionalities()
    demo_other_causal_conceptual()

    print("========== C: 因果推断 演示结束 ==========\n\n")

if __name__ == '__main__':
    run_all_ccausal_demos() 
 