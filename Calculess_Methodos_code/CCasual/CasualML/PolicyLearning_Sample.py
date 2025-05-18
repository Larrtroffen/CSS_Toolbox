import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold # 用于交叉验证
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier # 梯度提升模型，可作为CausalML元学习器的基础模型
from sklearn.tree import DecisionTreeClassifier # 决策树分类器，可用于策略学习
from causalml.optimize import PolicyLearner # CausalML中的策略学习器核心类
from sklearn.tree import plot_tree # 用于可视化决策树
from lightgbm import LGBMRegressor # LightGBM回归器，一种高效的梯度提升框架
from causalml.inference.meta import BaseXRegressor # CausalML中的X-learner元学习器

# 设置随机种子以保证结果可复现
np.random.seed(1234)

# 数据集参数
n = 10000  # 样本数量
p = 10     # 特征数量

# 生成特征 X：从正态分布中随机抽取
X = np.random.normal(size=(n, p))

# 生成处理分配的倾向得分 ee (propensity score for treatment)
# 这里倾向得分 ee 依赖于特征 X[:, 2]
ee = 1 / (1 + np.exp(X[:, 2]))

# 生成真实的个体处理效应 (Individual Treatment Effect, ITE) tt
# 真实的ITE依赖于特征 X[:, 0] 和 X[:, 1]，并且是异质的
tt = 1 / (1 + np.exp(X[:, 0] + X[:, 1])/2) - 0.5

# 生成二元处理变量 W (0: 控制组, 1: 处理组)
# W 是基于倾向得分 ee 的伯努利分布随机变量
W = np.random.binomial(1, ee, n)

# 生成结果变量 Y
# Y 依赖于特征 X[:, 2] (基础效应), 处理 W, 真实的ITE tt, 以及一些随机噪声
Y = X[:, 2] + W * tt + np.random.normal(size=n)

# 初始化CausalML的PolicyLearner
# model_mu (结果模型) 和 model_w (处理分配模型/倾向得分模型) 默认使用梯度提升机
# policy_learner 指定用于学习最终分配策略的模型，这里使用一个简单的决策树
# calibration=True 表示在内部对倾向得分模型进行校准
policy_learner = PolicyLearner(policy_learner=DecisionTreeClassifier(max_depth=2), calibration=True)

# 使用生成的特征X, 处理W, 和结果Y来拟合策略学习器
# fit方法会训练内部的结果模型、倾向得分模型，并最终训练策略决策模型
policy_learner.fit(X, W, Y)

# 可视化学习到的策略决策树
# policy_learner.model_pi 是拟合好的策略模型 (这里是DecisionTreeClassifier)
plt.figure(figsize=(15,7)) # 设置图形大小
plot_tree(policy_learner.model_pi, feature_names=[f'X[{i}]' for i in range(p)], class_names=['No Treat', 'Treat'], filled=True) # 使用sklearn的plot_tree函数
plt.show()

# 初始化X-learner，使用LGBMRegressor作为基础模型
learner_x = BaseXRegressor(LGBMRegressor())

# 使用X-learner拟合数据并预测个体处理效应 (ITE)
# X: 特征, treatment: 处理变量W, y: 结果变量Y
# fit_predict 方法会直接返回估计的ITE值
ite_x = learner_x.fit_predict(X=X, treatment=W, y=Y)

# 创建一个DataFrame来比较不同策略下的平均处理效应
# 我们关注的是“实际被处理的个体所获得的真实ITE的平均值”

# PolicyLearner (DR-DT) 策略:
# policy_learner.predict(X) 返回 -1 (不处理) 或 1 (处理)
# (policy_learner.predict(X) + 1) / 2 将其转换为 0 (不处理) 或 1 (处理)
# 乘以真实的ITE (tt)，然后求平均
dr_dt_optimal_effect = np.mean(((policy_learner.predict(X) + 1) / 2) * tt)

# 真实最优策略:
# np.sign(tt) 返回 -1 (ITE<0), 0 (ITE=0), 1 (ITE>0)
# (np.sign(tt) + 1) / 2 将其转换为 0 (ITE<=0, 不处理) 或 1 (ITE>0, 处理)
# 乘以真实的ITE (tt)，然后求平均
true_optimal_effect = np.mean(((np.sign(tt) + 1) / 2) * tt)

# 基于X-learner ITE估计的策略:
# (np.sign(ite_x) + 1) / 2 将估计的ITE符号转换为 0 (不处理) 或 1 (处理)
# 乘以真实的ITE (tt)，然后求平均
x_learner_effect = np.mean(((np.sign(ite_x) + 1) / 2) * tt)

results_df = pd.DataFrame({
    'DR-DT Optimal': [dr_dt_optimal_effect],
    'True Optimal': [true_optimal_effect],
    'X Learner': [x_learner_effect],
})

print(results_df)


