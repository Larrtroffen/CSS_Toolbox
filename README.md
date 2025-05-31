![](https://549b-static-lowcode-3gkwb0vfd2ab3beb-1257068422.cos.ap-shanghai.myqcloud.com/%E6%B7%BB%E5%8A%A0%E6%A0%871%E9%A2%98.png)

![GitHub contributors](https://img.shields.io/github/contributors/Larrtroffen/Stata_Guidebook)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/Larrtroffen/Stata_Guidebook/main) ![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/Larrtroffen/Stata_Guidebook) ![GitHub all releases](https://img.shields.io/github/downloads/Larrtroffen/Stata_Guidebook/total) ![GitHub Repo stars](https://img.shields.io/github/stars/Larrtroffen/Stata_Guidebook) ![GitHub release (with filter)](https://img.shields.io/github/v/release/Larrtroffen/Stata_Guidebook)
<a href="https://mp.weixin.qq.com/mp/profile_ext?action=home&__biz=MzkzNzY4NTU5OA==&scene=124#wechat_redirect"><img src="https://img.shields.io/badge/WeChat-@回归不归-07c160" /></a>&emsp;

![Alt](https://repobeats.axiom.co/api/embed/f915fdf2050c14532e7d853be7d6059fc2eea71c.svg "Repobeats analytics image")

# 正在建设中，敬请期待 🏗️

### 项目结构

```
└── CNCSS_Toolbox：计算社会科学工具箱
    ├── Caculess_Methodos《无算之策》正文文本（GitBook链接）
    ├── Caculess_Methodos_code《无算之策》所用的环境与代码
    ├── DataSets 常用数据集（互联网上公开的）
    └── Assets 参考资料（互联网上公开的）
```

## 《无算之策》正文文本

### 第零部分——引言 <a href="#p0" id="p0"></a>

- [封面](README.md)
- [引言](Calculess_Methodos/p0/yy/README.md)
  - [第一节——时代变革下的社会科学：研究疆域的重塑与范式困境](Calculess_Methodos/p0/yy/j1.md)
  - [第二节——计算社会科学的缘起：智识汇流、领域界定与范式确立](Calculess_Methodos/p0/yy/j2.md)
  - [第三节——计算社会科学的四大支柱：核心任务、解决的问题与领域特性](Calculess_Methodos/p0/yy/j3.md)
  - [第四节——无算之策：编排思路与结构](Calculess_Methodos/p0/yy/j4.md)
  - [参考文献](Calculess_Methodos/p0/yy/ref.md)
- [问题回答与补充](Calculess_Methodos/p0/questions.md)

### 第一部分——计算社会科学研究方法基础 <a href="#p1" id="p1"></a>

- [第一章——认识论基础：计算社会科学的研究思维范式](Calculess_Methodos/p1/z1/README.md)
  - [第一节——问题思维: 计算社会科学探究的源起与罗盘](Calculess_Methodos/p1/z1/j1.md)
  - [第二节——数据思维: 理解与批判计算社会科学的经验基础](Calculess_Methodos/p1/z1/j2.md)
  - [第三节——模型思维：构建理解、预测与干预社会复杂性的抽象工具](Calculess_Methodos/p1/z1/j3.md)
- [第二章——方法论基础：通用的研究流程与核心概念](Calculess_Methodos/p1/z2/README.md)
  - [第一节——计算社会科学研究的迭代生命周期：核心阶段与循环反馈](Calculess_Methodos/p1/z2/j1.md)

## 《无算之策》代码部分

本部分包含了《无算之策》中各章节 Python 代码示例的实现。代码遵循“认识论 (Epistemology) - 方法论 (Methodology) - 操作化 (Operationalization)”的框架，旨在提供可直接运行和修改的材料。所有 Python 脚本均包含 `if __name__ == '__main__':` 块，使其可以独立执行以查看演示效果。同时，每个脚本都被设计为包含可导入的 API 式函数，方便用户在自己的项目中调用特定功能。所有注释和文档字符串均使用中文编写。

### **主演示脚本:**

- [`Calculess_Methodos_code/main.py`](Calculess_Methodos_code/main.py): 调用所有子演示脚本的总入口。

### **各部分演示集:**

- [`Calculess_Methodos_code/main_demos/run_adescribe_demos.py`](Calculess_Methodos_code/main_demos/run_adescribe_demos.py): 运行 A 部分所有描述性分析演示。
- [`Calculess_Methodos_code/main_demos/run_bpredict_demos.py`](Calculess_Methodos_code/main_demos/run_bpredict_demos.py): 运行 B 部分所有预测性建模演示。
- [`Calculess_Methodos_code/main_demos/run_ccausal_demos.py`](Calculess_Methodos_code/main_demos/run_ccausal_demos.py): 运行 C 部分所有因果推断演示。
- [`Calculess_Methodos_code/main_demos/run_dsimulate_demos.py`](Calculess_Methodos_code/main_demos/run_dsimulate_demos.py): 运行 D 部分所有模拟演示。

### **代码库结构 (Calculess_Methodos_code):**

```
Calculess_Methodos_code/
├── main.py
├── main_demos/
│ ├── run_adescribe_demos.py
│ ├── run_bpredict_demos.py
│ ├── run_ccausal_demos.py
│ └── run_dsimulate_demos.py
├── ADescribe/ (描述性分析与探索性数据分析 EDA)
│ ├── init.py
│ ├── altair_examples.py
│ ├── bokeh_examples.py
│ ├── geopandas_folium_examples.py
│ ├── lux_conceptual.py
│ ├── matplotlib_examples.py
│ ├── networkx_examples.py
│ ├── nltk_spacy_wordcloud_text_tools.py
│ ├── pandas_operations.py
│ ├── plotly_examples.py
│ ├── seaborn_examples.py
│ ├── sweetviz_autoviz_conceptual.py
│ ├── tsne_pca_examples.py
│ └── ydata_profiling_examples.py
├── BPredict/ (预测性建模)
│ ├── init.py
│ ├── catboost_examples.py
│ ├── hyperparameter_tuning_examples.py
│ ├── lightgbm_examples.py
│ ├── model_deployment_conceptual.py
│ ├── model_evaluation_comparison.py
│ ├── model_interpretability_examples.py
│ ├── pytorch_examples.py
│ ├── sklearn_examples.py
│ ├── tensorflow_keras_examples.py
│ └── xgboost_examples.py
├── CCausal/ (因果推断)
│ ├── init.py
│ ├── causalml_example.py
│ ├── dowhy_example.py
│ ├── econml_example.py
│ └── other_causal_methods_conceptual.py
└── DSimulate/ (模拟)
├── init.py
├── agent_based_modeling_mesa.py
├── discrete_event_simulation_simpy.py
├── monte_carlo_simulations.py
├── simulation_tools_comparison.py
└── system_dynamics_pysd.py
```

### **详细模块链接:**

- **A: 描述性分析与 EDA (Descriptive Analysis & EDA)**
  - [`ADescribe/pandas_operations.py`](Calculess_Methodos_code/ADescribe/pandas_operations.py)
  - [`ADescribe/matplotlib_examples.py`](Calculess_Methodos_code/ADescribe/matplotlib_examples.py)
  - [`ADescribe/seaborn_examples.py`](Calculess_Methodos_code/ADescribe/seaborn_examples.py)
  - [`ADescribe/plotly_examples.py`](Calculess_Methodos_code/ADescribe/plotly_examples.py)
  - [`ADescribe/bokeh_examples.py`](Calculess_Methodos_code/ADescribe/bokeh_examples.py)
  - [`ADescribe/altair_examples.py`](Calculess_Methodos_code/ADescribe/altair_examples.py)
  - [`ADescribe/geopandas_folium_examples.py`](Calculess_Methodos_code/ADescribe/geopandas_folium_examples.py)
  - [`ADescribe/networkx_examples.py`](Calculess_Methodos_code/ADescribe/networkx_examples.py)
  - [`ADescribe/nltk_spacy_wordcloud_text_tools.py`](Calculess_Methodos_code/ADescribe/nltk_spacy_wordcloud_text_tools.py)
  - [`ADescribe/tsne_pca_examples.py`](Calculess_Methodos_code/ADescribe/tsne_pca_examples.py)
  - [`ADescribe/ydata_profiling_examples.py`](Calculess_Methodos_code/ADescribe/ydata_profiling_examples.py)
  - [`ADescribe/sweetviz_autoviz_conceptual.py`](Calculess_Methodos_code/ADescribe/sweetviz_autoviz_conceptual.py)
  - [`ADescribe/lux_conceptual.py`](Calculess_Methodos_code/ADescribe/lux_conceptual.py)
- **B: 预测性建模 (Predictive Modeling)**
  - [`BPredict/sklearn_examples.py`](Calculess_Methodos_code/BPredict/sklearn_examples.py)
  - [`BPredict/xgboost_examples.py`](Calculess_Methodos_code/BPredict/xgboost_examples.py)
  - [`BPredict/lightgbm_examples.py`](Calculess_Methodos_code/BPredict/lightgbm_examples.py)
  - [`BPredict/catboost_examples.py`](Calculess_Methodos_code/BPredict/catboost_examples.py)
  - [`BPredict/tensorflow_keras_examples.py`](Calculess_Methodos_code/BPredict/tensorflow_keras_examples.py)
  - [`BPredict/pytorch_examples.py`](Calculess_Methodos_code/BPredict/pytorch_examples.py)
  - [`BPredict/model_evaluation_comparison.py`](Calculess_Methodos_code/BPredict/model_evaluation_comparison.py)
  - [`BPredict/hyperparameter_tuning_examples.py`](Calculess_Methodos_code/BPredict/hyperparameter_tuning_examples.py)
  - [`BPredict/model_interpretability_examples.py`](Calculess_Methodos_code/BPredict/model_interpretability_examples.py)
  - [`BPredict/model_deployment_conceptual.py`](Calculess_Methodos_code/BPredict/model_deployment_conceptual.py)
- **C: 因果推断 (Causal Inference)**
  - [`CCausal/dowhy_example.py`](Calculess_Methodos_code/CCausal/dowhy_example.py)
  - [`CCausal/econml_example.py`](Calculess_Methodos_code/CCausal/econml_example.py)
  - [`CCausal/causalml_example.py`](Calculess_Methodos_code/CCausal/causalml_example.py)
  - [`CCausal/other_causal_methods_conceptual.py`](Calculess_Methodos_code/CCausal/other_causal_methods_conceptual.py)
- **D: 模拟 (Simulation)**
  - [`DSimulate/monte_carlo_simulations.py`](Calculess_Methodos_code/DSimulate/monte_carlo_simulations.py)
  - [`DSimulate/discrete_event_simulation_simpy.py`](Calculess_Methodos_code/DSimulate/discrete_event_simulation_simpy.py)
  - [`DSimulate/agent_based_modeling_mesa.py`](Calculess_Methodos_code/DSimulate/agent_based_modeling_mesa.py)
  - [`DSimulate/system_dynamics_pysd.py`](Calculess_Methodos_code/DSimulate/system_dynamics_pysd.py)
  - [`DSimulate/simulation_tools_comparison.py`](Calculess_Methodos_code/DSimulate/simulation_tools_comparison.py)

## 项目推介——回归不归往期推文

### 描述

- [Neo4jGraphConstruction——自动提取图数据](https://github.com/neo4j-labs/llm-graph-builder)
  - 介绍推文：[项目推介丨文本 →LLM→ 图——非结构化数据整理方案](https://mp.weixin.qq.com/s/5YWeCIoNSrGhf9co3YSn5g)

### 预测

- [Autogluon——自动机器学习](https://github.com/autogluon/autogluon)
  - 介绍推文：[项目推介丨自动调参、自动变量选择、自动训练、自动编程——工具推介与 Agent 时代的能动迷思
    ](https://mp.weixin.qq.com/s/G4SbYscBADKvxhjOkSafnA)

### 因果

- [Oneclick——自动筛选控制变量](https://shutterzor.cn/stata/)
  - 介绍推文：[项目推介丨自动调参、自动变量选择、自动训练、自动编程——工具推介与 Agent 时代的能动迷思
    ](https://mp.weixin.qq.com/s/G4SbYscBADKvxhjOkSafnA)
- [CausalML——因果机器学习](https://causalml.readthedocs.io/en/latest/about.html)
  - 介绍推文：[项目推介丨向计量方法引入机器学习，向计算方法介绍因果推断 —— CasualML](https://mp.weixin.qq.com/s/5iHwpxGeaGFoAGJgH-RfAQ)
- [EconML——计量经济学的机器学习模型](https://github.com/py-why/EconML)
  - 介绍推文：[项目推介丨计量经济学的机器学习工具箱 —— EconML](https://mp.weixin.qq.com/s/EXImSaOfBCZ5qaqKxKwTEg)

### 模拟

- [PlatformWar——基于社交平台的观点生成](https://github.com/LYiHub/platform-war-public)
  - 介绍推文：[项目推介丨探索社交平台差异：一个 Graph-RAG 驱动的 LLM 对话框架](https://mp.weixin.qq.com/s/GSeelpWGsky10gWwmM4-LA)
- [Concordia——基于生成式 LLM 的 ABM 库](https://github.com/google-deepmind/concordia/tree/main)
  - 介绍推文：[项目推介丨最早的模拟社会开源库：谷歌 Concordia](https://mp.weixin.qq.com/s/6VNPHBY9bOle19h37eVilQ)
- [AgentSociety——基于大模型的社会模拟器](https://agentsociety.readthedocs.io/en/latest/)
  - 介绍推文：[项目推介丨低门槛社会模拟器！清华大学发布 AgentSociety 1.0，社会模拟引擎+社科研究与治理工具箱=范式变革！](https://mp.weixin.qq.com/s/4sqPQd5hkYrFxL-ylGQPTA)
- [OASIS——社交媒体模拟器](https://github.com/camel-ai/oasis)
  - 介绍推文：[项目推介丨如何使用模拟社会框架辅助社会科学研究？](https://mp.weixin.qq.com/s/RYGBEjyzeYTONi3-AwlIDA)
- [YuLan-Onesim——新一代大模型社会模拟平台](https://github.com/RUC-GSAI/YuLan-OneSim)
  - 介绍推文：[项目推介丨人大高瓴人工智能学院推出新一代社会模拟平台玉兰-万象](https://mp.weixin.qq.com/s/7EQtcmd9IEdy6qxZLPkvyw)

---

CNCSS-Toolbox © 2024- by Larrtroffen(Xinhua Li, 李欣桦) is licensed under CC BY-NC-SA 4.0
