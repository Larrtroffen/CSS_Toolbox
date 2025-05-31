import os
import pandas as pd
import numpy as np

# Scikit-learn for model training example
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# For saving/loading models
import joblib # For scikit-learn models
# import tensorflow as tf # For Keras/TF models (conceptual)
# import torch # For PyTorch models (conceptual)

# For Flask API (conceptual)
# from flask import Flask, request, jsonify

# --- API Functions ---

def create_sample_sklearn_pipeline_for_deployment_api(
    n_samples: int = 120, 
    n_features: int = 7, 
    random_state: int = 420
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """
    创建一个简单的Scikit-learn Pipeline及其训练数据，用于部署演示。

    参数:
    - n_samples (int): 生成的样本数量。
    - n_features (int): 生成的特征数量。
    - random_state (int): 随机种子。

    返回:
    - tuple[Pipeline, pd.DataFrame, pd.Series]: 训练好的Pipeline, 特征数据 (X_train), 目标数据 (y_train)。
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, 
        random_state=random_state, n_informative=max(2, n_features-2)
    )
    X_train_arr, _, y_train_arr, _ = train_test_split(X, y, test_size=0.25, random_state=random_state)
    
    # 将numpy数组转换为DataFrame，以便更好地模拟真实场景
    feature_names = [f'feat_{i}' for i in range(X_train_arr.shape[1])]
    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names)
    y_train_series = pd.Series(y_train_arr, name='target')

    pipeline = Pipeline([
        ('scaler', StandardScaler()), # 假设所有特征都是数值型且需要标准化
        ('model', RandomForestClassifier(n_estimators=15, random_state=random_state, max_depth=4))
    ])
    pipeline.fit(X_train_df, y_train_series)
    # print("API: 示例Scikit-learn Pipeline已训练。")
    return pipeline, X_train_df, y_train_series

def explain_sklearn_model_persistence_api(
    model_pipeline: Pipeline, 
    model_name: str = "sklearn_generic_pipeline.joblib",
    output_dir: str = "model_deployment_outputs",
    sample_data_for_test: pd.DataFrame | None = None
) -> bool:
    """
    演示使用joblib保存和加载Scikit-learn模型（通常是Pipeline）。

    参数:
    - model_pipeline (Pipeline): 要保存的Scikit-learn Pipeline。
    - model_name (str): 模型保存的文件名。
    - output_dir (str): 模型保存的目录。
    - sample_data_for_test (pd.DataFrame | None): 用于测试加载后模型的样本数据 (可选)。

    返回:
    - bool: 如果保存和加载都成功，则返回True，否则返回False。
    """
    print("\n--- 1. Scikit-learn 模型持久化 (使用 joblib) (API) ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    model_path = os.path.join(output_dir, model_name)

    # 保存模型
    try:
        joblib.dump(model_pipeline, model_path)
        print(f"Scikit-learn Pipeline已保存至: {model_path}")
    except Exception as e:
        print(f"保存Scikit-learn模型时出错 (joblib): {e}")
        return False

    # 加载模型
    try:
        loaded_pipeline = joblib.load(model_path)
        print(f"Scikit-learn Pipeline已成功从 {model_path} 加载。")
        if sample_data_for_test is not None and not sample_data_for_test.empty:
            try:
                sample_prediction = loaded_pipeline.predict(sample_data_for_test.head(1))
                print(f"  使用加载的模型进行预测 (对提供的样本数据的第一行): {sample_prediction}")
            except Exception as e_pred:
                print(f"  使用加载的模型预测时出错: {e_pred} (请确保样本数据与模型训练时格式一致)")
        return True
    except Exception as e:
        print(f"加载Scikit-learn模型时出错 (joblib): {e}")
        return False

def explain_keras_model_persistence_conceptual_api():
    """提供Keras/TensorFlow模型持久化的概念性解释。"""
    print("\n--- 2. Keras/TensorFlow 模型持久化 - 概念性解释 (API) ---")
    print("Keras模型通常使用 `model.save()` 保存，并使用 `keras.models.load_model()` 加载。")
    print("  保存命令示例 (Keras v3+ 原生格式):")
    print("    `import tensorflow as tf`")
    print("    `model.save('./model_deployment_outputs/keras_model.keras')`")
    print("  保存命令示例 (旧版 TensorFlow SavedModel 格式):")
    print("    `model.save('./model_deployment_outputs/keras_model_tf_dir', save_format='tf')`")
    print("  加载命令示例:")
    print("    `loaded_model = tf.keras.models.load_model('./model_deployment_outputs/keras_model.keras')`")
    print("此操作会保存模型的架构、权重以及训练配置（如优化器状态）。")
    print("对于仅权重，可以使用 `model.save_weights()` 和 `model.load_weights()`。")

def explain_pytorch_model_persistence_conceptual_api():
    """提供PyTorch模型持久化的概念性解释。"""
    print("\n--- 3. PyTorch 模型持久化 - 概念性解释 (API) ---")
    print("PyTorch模型通常通过保存其 `state_dict` (包含所有权重和偏差) 来进行持久化。")
    print("  保存state_dict命令示例:")
    print("    `import torch`")
    print("    `torch.save(model.state_dict(), './model_deployment_outputs/pytorch_model_state.pth')`")
    print("  加载state_dict的步骤:")
    print("    1. 实例化模型类: `model_instance = YourModelClass(*args, **kwargs)`")
    print("    2. 加载state_dict: `model_instance.load_state_dict(torch.load('./model_deployment_outputs/pytorch_model_state.pth'))`")
    print("    3. 设置为评估模式: `model_instance.eval()` (这对于失活Dropout和BatchNorm等层很重要)")
    print("虽然也可以保存整个模型 (`torch.save(model, PATH)`), 但保存state_dict更为推荐，因为它更灵活且不易出错。")

def explain_flask_api_conceptual_api():
    """提供使用Flask创建简单REST API的概念性解释和示例代码结构。"""
    print("\n--- 4. 使用Flask创建REST API - 概念性解释 (API) ---")
    print("Flask是一个轻量级的Python Web框架，常用于为机器学习模型创建简单的API服务。")
    print("以下是一个用于模型服务的Flask应用的概念性结构 (通常保存为如 `api_server.py` 的文件):")
    flask_app_code = """
# api_server.py (示例)
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd # 假设输入可能是JSON，需要转换为DataFrame

# 1. 加载你训练好的模型 (确保部署时路径正确)
# MODEL_PATH = './model_deployment_outputs/your_model_pipeline.joblib' 
# try:
#     model = joblib.load(MODEL_PATH)
#     print(f"模型已从 {MODEL_PATH} 成功加载。")
# except FileNotFoundError:
#     print(f"错误: 模型文件 {MODEL_PATH} 未找到。请确保路径正确且模型已放置。")
#     model = None
# except Exception as e:
#     print(f"加载模型时发生错误: {e}")
#     model = None

# # 为了演示，我们在这里用一个虚拟模型代替实际加载
class DummyModel:
    def predict(self, data):
        # 假设data是一个DataFrame
        print(f"虚拟模型收到数据条数: {len(data)}")
        return [f"prediction_for_row_{i}" for i in range(len(data))]
    def predict_proba(self, data):
        return [[np.random.rand(), 1-np.random.rand()] for _ in range(len(data))]

model = DummyModel() # 使用虚拟模型进行演示
print("警告: 正在使用虚拟模型进行API演示。请替换为实际加载的模型。")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    global model # 声明我们要使用全局的model变量
    if model is None:
        return jsonify({'error': '模型未加载或加载失败，无法进行预测。'}), 500
    try:
        # 获取JSON数据
        json_data = request.get_json(force=True)
        print(f"收到请求数据: {json_data}")
        
        # --- 关键步骤: 将JSON数据转换为模型期望的格式 ---
        # 这一步高度依赖于你的模型是如何训练的以及它期望的输入是什么。
        # 示例: 假设JSON数据是一个包含特征的字典，或者一个字典列表。
        # 如果是单个预测实例的字典: e.g., {'feature1': val1, 'feature2': val2, ...}
        # input_df = pd.DataFrame([json_data])
        # 如果是多个预测实例的列表: e.g., [{'f1':v1,...}, {'f1':v2,...}]
        # input_df = pd.DataFrame(json_data)
        
        # **此处需要根据您的具体模型进行适配**
        # 例如，如果模型期望一个包含特定列的DataFrame:
        # required_features = ['age', 'income', 'education_level'] # 假设的特征
        # try:
        #     input_df = pd.DataFrame(json_data, columns=required_features) # 如果json_data是字典列表
        # except Exception as df_error:
        #     return jsonify({'error': f'数据格式转换错误: {df_error}. 请确保JSON包含所需特征。'}), 400

        # 假设对于虚拟模型，我们期望一个字典列表，每个字典代表一行数据
        if not isinstance(json_data, list) or not all(isinstance(item, dict) for item in json_data):
            return jsonify({'error': '输入数据格式错误。期望一个JSON对象列表。'}), 400
        
        input_df = pd.DataFrame(json_data) # 转换为DataFrame
        if input_df.empty:
            return jsonify({'error': '输入数据为空。'}), 400

        # 进行预测
        predictions = model.predict(input_df)
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)
            # 确保概率是可JSON序列化的 (例如，转换为列表)
            probabilities = [p.tolist() for p in probabilities] if probabilities is not None else None

        return jsonify({
            'predictions': predictions, # 通常是个列表
            'probabilities': probabilities # 通常是列表的列表
        })

    except Exception as e:
        # 记录异常 e
        print(f"预测过程中发生错误: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    # 用于开发环境: app.run(debug=True, host='0.0.0.0', port=5001) # 使用不同端口避免冲突
    # 用于生产环境: 使用生产级的WSGI服务器，如Gunicorn或uWSGI
    # 示例: gunicorn -w 4 -b 0.0.0.0:5001 api_server:app
    print("要运行此Flask API (如果已保存为 api_server.py):")
    print("  1. 确保Flask, joblib, pandas, numpy等已安装: pip install Flask joblib pandas numpy scikit-learn")
    print("  2. 将你的模型 (例如 .joblib 文件) 放在 MODEL_PATH 指定的位置，并取消注释模型加载部分。")
    print("  3. (当前为演示模式，使用虚拟模型) 运行: python api_server.py")
    print("  4. API将在 http://localhost:5001 (或配置的端口) 上可用。")
    print("  5. 发送POST请求到 http://localhost:5001/predict，附带JSON数据。")
    print("     示例 (curl): curl -X POST -H \"Content-Type: application/json\" -d '[{"feature1": 1, "feature2": "A"}]' http://localhost:5001/predict")
    # app.run(debug=False, host='0.0.0.0', port=5001) # 取消注释以实际运行 (确保没有其他服务占用端口5001)
"""
    print(flask_app_code)
    print("\n请注意: 以上Flask代码是一个模板。您需要根据模型的实际输入/输出和预处理需求进行修改。")

def explain_docker_containerization_conceptual_api():
    """提供使用Docker进行容器化的概念性解释和示例Dockerfile。"""
    print("\n--- 5. 使用Docker进行容器化 - 概念性解释 (API) ---")
    print("Docker能够将应用程序及其所有依赖（库、运行时环境等）打包到一个可移植的镜像中，并在容器中运行。")
    print("这确保了在不同环境（开发、测试、生产）中的一致性。")
    print("\n一个用于Flask API的Dockerfile概念性示例:")
    dockerfile_content = """
# Dockerfile (示例)

# 1. 选择一个基础镜像 (例如，官方Python镜像)
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 复制项目的依赖文件 (例如 requirements.txt)
COPY requirements.txt ./

# 4. 安装依赖
# (在构建镜像前，确保 requirements.txt 文件包含所有必要的包: Flask, gunicorn, joblib, scikit-learn, pandas, numpy等)
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制应用程序代码和模型到镜像中
COPY ./app /app/app # 假设你的Flask应用在 ./app/api_server.py
COPY ./model_deployment_outputs /app/model_deployment_outputs # 假设模型在此目录

# 6. 暴露应用程序运行的端口 (例如，Gunicorn可能运行在8000端口)
EXPOSE 8000

# 7. (可选) 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP_PATH=/app/app/api_server.py # 示例: 指向你的Flask应用文件
ENV MODEL_FILE_PATH=/app/model_deployment_outputs/your_model_pipeline.joblib # 示例: 模型路径

# 8. 定义容器启动时运行的命令 (使用Gunicorn作为生产WSGI服务器)
# CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8000", "app.api_server:app"] # 假设api_server.py在app目录下，且Flask实例名为app
# 或者，如果api_server.py在根目录 (WORKDIR /app)，且实例名为app:
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8000", "api_server:app"] # 确保api_server.py中有 `app = Flask(__name__)`
"""
    print(dockerfile_content)
    print("\n构建和运行Docker镜像 (概念性命令):")
    print("  1. 创建 `requirements.txt` 文件，列出所有Python依赖。")
    print("  2. (可选) 创建一个 `./app` 目录并将你的Flask API脚本 (如 `api_server.py`) 放入其中。")
    print("  3. 在包含Dockerfile的项目根目录下运行构建命令: `docker build -t my_ml_api .`")
    print("  4. 运行容器: `docker run -p 8000:8000 -e MODEL_FILE_PATH=/app/model_deployment_outputs/your_model.joblib my_ml_api`")
    print("     (将宿主机的8000端口映射到容器的8000端口，并可传递环境变量覆盖Dockerfile中的默认值)")

def explain_cloud_deployment_options_conceptual_api():
    """提供云平台部署ML模型的概念性概述。"""
    print("\n--- 6. 云平台部署选项 - 概念性解释 (API) ---")
    print("主流云平台提供了多种托管和扩展机器学习模型推理服务的方式:")
    print("- **Google Cloud Vertex AI Endpoints:**")
    print("  - 功能: 上传训练好的模型（或自定义容器），创建可公开访问或私有的预测端点。自动处理扩展、版本控制。")
    print("  - 适用: TensorFlow, Scikit-learn, XGBoost, PyTorch, 自定义容器。")
    print("- **Amazon SageMaker Endpoints:**")
    print("  - 功能: 部署在SageMaker中训练的模型或自定义模型。提供推理管道、自动扩展、A/B测试、监控等。")
    print("  - 适用: 多种框架，支持自定义容器和预构建容器。")
    print("- **Azure Machine Learning Endpoints (Online Endpoints):**")
    print("  - 功能: 将模型部署为Web服务。支持基于Docker容器的部署，可扩展到Azure Kubernetes Service (AKS) 或托管计算。")
    print("  - 适用: MLflow模型, Python/R模型, 自定义容器。")
    print("- **Serverless Functions (如 AWS Lambda, Google Cloud Functions, Azure Functions):**")
    print("  - 功能: 对于延迟要求不高、调用频率较低或模型较小的场景。通常与API Gateway结合使用。")
    print("  - 挑战: 冷启动时间，包大小限制，模型加载策略。")
    print("- **Kubernetes-based Solutions (如 Kubeflow, Seldon Core):**")
    print("  - 功能: 提供更细致的控制和可移植性，但需要更多的运维管理。")
    print("  - 适用: 需要高度定制化部署流程的复杂场景。")
    print("\n选择云服务时，需考虑成本、易用性、扩展需求、现有技术栈以及特定功能（如模型监控、漂移检测等）。")

def explain_production_considerations_conceptual_api():
    """概述在生产环境中部署和维护ML模型时需要考虑的关键因素。"""
    print("\n--- 7. 生产环境部署考量 - 概念性解释 (API) ---")
    print("- **模型监控 (Model Monitoring):**")
    print("  - 性能跟踪: 持续监控模型的关键性能指标（如准确率、F1分数、RMSE）。")
    print("  - 数据漂移 (Data Drift): 检测输入数据的分布变化，可能导致模型性能下降。")
    print("  - 概念漂移 (Concept Drift): 检测目标变量与特征之间关系的变化。")
    print("- **日志记录 (Logging):** 全面记录API请求、模型输入、预测输出、错误信息及系统指标，用于调试、审计和分析。")
    print("- **版本控制 (Versioning):** 对模型、代码、数据和环境进行严格版本控制，确保可复现性、可追溯性及安全回滚。")
    print("- **CI/CD/CT (持续集成/持续部署/持续训练):**")
    print("  - CI/CD: 自动化模型训练、测试、打包和部署流程。")
    print("  - CT: 建立模型自动或半自动重新训练和更新的机制，以应对数据漂移或性能衰退。")
    print("- **安全性 (Security):** 保护API端点（认证、授权），数据传输加密，防止未授权访问和潜在攻击。")
    print("- **可扩展性与可用性 (Scalability & Availability):** 设计部署方案以处理预期负载，具备故障恢复能力，确保服务高可用。")
    print("- **成本管理 (Cost Management):** 优化资源使用（如计算实例、存储），选择合适的定价模型，控制部署成本。")
    print("- **A/B 测试与灰度发布 (A/B Testing & Canary Releases):** 安全地推出新模型版本，通过在部分流量上进行测试来评估其表现，再全面推广。")
    print("- **文档与交接 (Documentation & Handoff):** 保持清晰的文档，便于团队协作和后续维护。")

if __name__ == '__main__':
    print("========== (B 部分) 模型部署概念演示 ==========")
    main_deploy_seed = 420 # 与创建模型时一致
    output_dir_deploy = "model_deployment_outputs" # 主输出目录

    # 演示 Scikit-learn 模型持久化
    sample_pipeline, X_train_sample, _ = create_sample_sklearn_pipeline_for_deployment_api(
        random_state=main_deploy_seed
    )
    explain_sklearn_model_persistence_api(
        model_pipeline=sample_pipeline, 
        model_name="sklearn_ RandomForestClassifier_pipeline.joblib", 
        output_dir=output_dir_deploy,
        sample_data_for_test=X_train_sample # 使用部分训练数据测试加载的模型
    )
    
    # 概念性解释
    explain_keras_model_persistence_conceptual_api()
    explain_pytorch_model_persistence_conceptual_api()
    explain_flask_api_conceptual_api()
    explain_docker_containerization_conceptual_api()
    explain_cloud_deployment_options_conceptual_api()
    explain_production_considerations_conceptual_api()
    
    print("\n\n========== (B 部分) 模型部署概念演示结束 ==========")
    print("提示: 此脚本主要为概念性演示。实际部署涉及环境配置、服务器设置和云服务集成等复杂步骤。")
    print(f"示例 Scikit-learn 模型已保存在 '{output_dir_deploy}' 目录中 (如果成功)。") 