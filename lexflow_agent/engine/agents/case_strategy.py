# 模块文档字符串：案件策略智能体，负责证据矩阵分析和诉讼策略生成
"""Case Strategy Agent: evidence matrix and litigation strategy."""

# 从 langgraph.graph 导入图构建所需的 END 常量和 StateGraph 类
from langgraph.graph import END, StateGraph

# 从证据节点模块导入证据分析节点函数
from lexflow_agent.engine.nodes.evidence import analyze_evidence_node
# 从策略节点模块导入策略生成节点函数
from lexflow_agent.engine.nodes.strategy import generate_strategy_node
# 从状态模块导入案件智能体状态类
from lexflow_agent.engine.state import CaseAgentState


# 定义构建案件策略智能体的工厂函数
def build_case_strategy_agent():
    # 创建基于 CaseAgentState 的状态图构建器
    builder = StateGraph(CaseAgentState)
    # 添加起始节点，设置当前智能体标识为案件策略智能体
    builder.add_node("start", lambda state: {"current_agent": "case_strategy_agent"})
    # 添加证据分析节点，用于分析案件证据
    builder.add_node("analyze_evidence", analyze_evidence_node)
    # 添加策略生成节点，用于生成诉讼策略
    builder.add_node("generate_strategy", generate_strategy_node)
    # 设置图的入口点为起始节点
    builder.set_entry_point("start")
    # 添加从起始节点到证据分析节点的有向边
    builder.add_edge("start", "analyze_evidence")
    # 添加从证据分析节点到策略生成节点的有向边
    builder.add_edge("analyze_evidence", "generate_strategy")
    # 添加从策略生成节点到结束节点的有向边
    builder.add_edge("generate_strategy", END)
    # 编译并返回构建好的状态图
    return builder.compile()