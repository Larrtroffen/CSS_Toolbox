import os

# --- 辅助函数：创建输出目录 ---
def _ensure_output_directory_api(dir_name: str) -> str:
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
            print(f"  创建目录 {dir_name} 失败: {e}. 文件可能不会保存。")
    return dir_name

# --- API 函数 ---
def generate_simulation_tools_comparison_api(output_dir: str = "simulation_docs_demo") -> str | None:
    """
    生成关于不同模拟工具和范式的概念性比较文本，并将其保存到文件。
    比较内容包括蒙特卡洛方法、离散事件模拟(SimPy)、基于代理的建模(Mesa)和系统动力学(PySD)。

    参数:
    - output_dir (str): 保存比较文本文件的目录名。

    返回:
    - str | None: 成功保存的文件的路径，如果保存失败则为None。
    """
    print("\n--- (API) 生成模拟工具和范式对比 --- ")
    
    # 注意: comparison_text 的内容保持英文，因为它非常详细，
    # 用户可以根据需要自行翻译。此处的重点是API结构和中文文档。
    comparison_text = """
    This document provides a conceptual comparison of the simulation tools and paradigms
    covered in this section: Monte Carlo methods, Discrete-Event Simulation (SimPy),
    Agent-Based Modeling (Mesa), and System Dynamics (PySD).

    --- 1. Monte Carlo (MC) Simulations (using NumPy/SciPy) ---
    - **Paradigm:**
        - Uses repeated random sampling to obtain numerical results.
        - Often applied to problems that are deterministic in principle but where direct calculation is too complex (e.g., high-dimensional integrals) or to model systems with inherent randomness.
        - Focuses on understanding the distribution of outcomes from processes involving uncertainty.
    - **Typical Use Cases:**
        - Estimating quantities (e.g., Pi, integrals).
        - Risk analysis and financial modeling (e.g., option pricing, Value at Risk).
        - Simulating physical processes with random components (e.g., particle transport).
        - Uncertainty quantification in complex models.
        - Brute-force search or optimization in large state spaces.
    - **Strengths:**
        - Conceptually simple for many problems.
        - Easy to implement for basic cases using standard libraries like NumPy.
        - Highly parallelizable, as individual simulation runs are often independent.
        - Useful when the underlying analytical model is unknown or intractable.
    - **Weaknesses:**
        - Convergence can be slow (error decreases as 1/sqrt(N) for N samples typically).
        - May not be efficient for problems with very rare events without variance reduction techniques.
        - Does not inherently model time-dependent interactions or system structure in the same way as DES, ABM, or SD.
        - Output is typically a distribution or expectation, not a time-series trajectory of a structured system (unless simulating a stochastic process step-by-step).
    - **Scalability & Performance:**
        - Scales well with the number of independent runs (easily parallelized).
        - Performance per run depends on the complexity of the function being sampled or the process being simulated.
    - **Community & Ecosystem:**
        - Leverages the vast Python scientific computing ecosystem (NumPy, SciPy, Matplotlib).
        - Abundant resources and examples available due to its fundamental nature.

    --- 2. Discrete-Event Simulation (DES) with SimPy ---
    - **Paradigm:**
        - Models systems as a sequence of events occurring at discrete points in time.
        - Focuses on processes, resources, and entities (e.g., customers, jobs, packets).
        - State changes only occur at event times.
        - Manages a future event list and advances simulation time to the next event.
    - **Typical Use Cases:**
        - Queueing systems (e.g., call centers, banks, manufacturing lines, network traffic).
        - Logistics and supply chain modeling.
        - Healthcare systems (e.g., patient flow, hospital resource allocation).
        - Transportation systems.
        - Performance analysis of computer and communication systems.
    - **Strengths:**
        - Efficient for systems where changes are event-driven and periods of inactivity exist.
        - Natural way to model resource contention, queues, and complex workflows.
        - SimPy is lightweight, flexible, and integrates well with Python.
        - Allows detailed modeling of process logic and entity behavior.
    - **Weaknesses:**
        - Can become complex to define all event types and interactions for very large systems.
        - Not well-suited for continuous processes unless they are discretized (though SimPy can handle `timeout` with float values).
        - Debugging can sometimes be tricky due to the event-driven, asynchronous-like nature of process execution.
    - **Scalability & Performance:**
        - Performance depends on the number of events processed, not directly on the length of simulated time.
        - Large numbers of active processes or frequent events can slow down simulation.
        - SimPy itself is single-threaded but can be part of larger parallel experiment setups (running multiple independent simulations).
    - **Community & Ecosystem:**
        - SimPy has a dedicated and active user base.
        - Good documentation and examples available.
        - Being pure Python, it integrates well with other data analysis and visualization libraries.

    --- 3. Agent-Based Modeling (ABM) with Mesa ---
    - **Paradigm:**
        - Models systems as collections of autonomous, interacting agents.
        - Focuses on emergent behavior arising from local agent interactions.
        - Agents have states and rules governing their behavior and interactions with each other and their environment.
    - **Typical Use Cases:**
        - Social simulations (e.g., opinion dynamics, segregation, cooperation, spread of information/disease).
        - Ecological modeling (e.g., predator-prey dynamics, species competition).
        - Economic modeling (e.g., artificial stock markets, consumer behavior).
        - Crowd behavior and urban dynamics.
        - Biological systems (e.g., immune system response, flocking/swarming).
    - **Strengths:**
        - Captures heterogeneity and individual agent behavior effectively.
        - Excellent for studying emergent phenomena and complex adaptive systems.
        - Mesa provides a good framework for defining agents, environments (grids, networks), and schedulers.
        - Modular design, allowing for easy extension and modification.
        - Built-in tools for data collection and visualization (e.g., server for web-based interactive viz).
    - **Weaknesses:**
        - Can be computationally intensive, especially with many agents or complex interactions.
        - Model validation and calibration can be challenging due to the many degrees of freedom.
        - Aggregating agent behavior to macro-level insights requires careful analysis.
        - Theoretical understanding of emergent behavior can be difficult to derive formally.
    - **Scalability & Performance:**
        - Performance scales with the number of agents and the complexity of their step functions/interactions.
        - Mesa is Python-based; for very large-scale ABM, compiled languages or specialized frameworks (e.g., Repast HPC, NetLogo on clusters) might be needed, though Mesa is continually improving.
        - Parallelization can be complex due to inter-agent dependencies, but Mesa is exploring options.
    - **Community & Ecosystem:**
        - Mesa has a growing and active community.
        - Good documentation, tutorials, and a model zoo.
        - Benefits from Python's data science stack for analysis of simulation outputs.

    --- 4. System Dynamics (SD) with PySD ---
    - **Paradigm:**
        - Models systems using stocks (accumulations), flows (rates of change), and feedback loops.
        - Focuses on understanding the dynamic behavior of complex systems over time based on their aggregate structure.
        - Uses differential/difference equations, often represented visually as stock-and-flow diagrams.
    - **Typical Use Cases:**
        - Business strategy and policy analysis (e.g., market growth, resource management).
        - Environmental modeling (e.g., climate change, population dynamics, resource depletion).
        - Public policy (e.g., healthcare systems, urban planning, economic development).
        - Understanding long-term trends and the impact of feedback loops.
    - **Strengths:**
        - Excellent for modeling feedback mechanisms and their impact on system behavior.
        - Provides a high-level, aggregate view of system structure and dynamics.
        - PySD allows running models from established SD software (Vensim, XMILE) or defining them in Python.
        - Good for qualitative insights into system behavior patterns (e.g., growth, oscillation, collapse).
        - Facilitates policy testing and scenario analysis by changing parameters or model structure.
    - **Weaknesses:**
        - Aggregation can mask important individual-level heterogeneity (contrast with ABM).
        - Calibration of parameters (especially soft variables or behavioral aspects) can be difficult.
        - Spatial explicitness is often not a primary focus (though can be incorporated with array variables).
        - Structural validation (ensuring the model accurately represents the real system's causal structure) is crucial and challenging.
    - **Scalability & Performance:**
        - PySD translates SD models into Python code that solves ODEs/difference equations using SciPy solvers.
        - Performance depends on the size of the model (number of variables, complexity of equations) and the stiffness of the equations.
        - Generally efficient for typical SD model sizes.
    - **Community & Ecosystem:**
        - PySD bridges the traditional SD community (often using Vensim, Stella) with the Python ecosystem.
        - Active development and good documentation.
        - Allows leveraging Python's analytical and visualization tools with SD models.
        - System Dynamics Society provides a broader community and resources.

    --- Summary: Choosing the Right Tool/Paradigm ---
    - **Nature of the problem:** Is it about random sampling (MC), process flow and resources (DES), individual interactions and emergence (ABM), or aggregate feedback structures (SD)?
    - **Level of detail:** Do you need to model individual entities (ABM, DES) or aggregate quantities (SD)?
    - **Type of questions:** Are you interested in distributions (MC), throughput/wait times (DES), emergent patterns (ABM), or long-term dynamic behavior and policy impacts (SD)?
    - **Data availability:** Calibration and validation approaches differ significantly.
    - **Computational resources:** ABM can be demanding; MC can be if high precision is needed.

    Often, these methods can also be complementary. For instance, parameters within an ABM or SD model might be informed by Monte Carlo analysis, or a DES model might feed into a higher-level SD model.
    """ # End of comparison_text
    print(comparison_text) # 打印到控制台

    target_dir = _ensure_output_directory_api(output_dir)
    if not os.path.exists(target_dir):
        print(f"  警告: 输出目录 {target_dir} 不存在，比较文件无法保存。")
        return None

    file_path = os.path.join(target_dir, "simulation_tools_comparison_conceptual.txt")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(comparison_text)
        print(f"\n  模拟工具概念比较已保存至: {file_path}")
        return file_path
    except IOError as e:
        print(f"  保存比较文件时出错: {e}")
        return None

if __name__ == '__main__':
    print("========== (D 部分) 模拟工具概念比较演示 ==========")
    # 为主演示创建一个特定的输出目录
    main_comparison_output_dir = "simulation_docs_main_demo"
    # _ensure_output_directory_api(main_comparison_output_dir) # generate_simulation_tools_comparison_api会处理其内部目录创建

    saved_file_path = generate_simulation_tools_comparison_api(output_dir=main_comparison_output_dir)
    
    if saved_file_path:
        print(f"  比较文档已生成并保存于: {saved_file_path}")
    else:
        print("  比较文档未能生成或保存。")

    print(f"\n\n========== (D 部分) 模拟工具概念比较演示结束 ==========") 