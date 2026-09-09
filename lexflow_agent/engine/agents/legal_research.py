# 模块文档字符串：法律研究智能体，负责问题分解和基于 RAG 的法律检索
"""Legal Research Agent: issue decomposition and grounded legal RAG."""

# 从 langgraph.graph 导入图构建所需的 END 常量和 StateGraph 类
from langgraph.graph import END, StateGraph

# 从问题节点模块导入问题识别节点函数
from lexflow_agent.engine.nodes.issues import identify_issues_node
# 从法律研究节点模块导入法律研究节点函数
from lexflow_agent.engine.nodes.legal_research import research_law_node
# 从状态模块导入案件智能体状态类
from lexflow_agent.engine.state import CaseAgentState


# 定义构建法律研究智能体的工厂函数
def build_legal_research_agent():
    # 创建基于 CaseAgentState 的状态图构建器
    builder = StateGraph(CaseAgentState)
    # 添加起始节点，设置当前智能体标识为法律研究智能体
    builder.add_node("start", lambda state: {"current_agent": "legal_research_agent"})
    # 添加问题识别节点，用于识别和分解案件中的法律问题
    builder.add_node("identify_issues", identify_issues_node)
    # 添加法律研究节点，用于进行法律条文和案例的检索研究
    builder.add_node("research_law", research_law_node)
    # 设置图的入口点为起始节点
    builder.set_entry_point("start")
    # 添加从起始节点到问题识别节点的有向边
    builder.add_edge("start", "identify_issues")
    # 添加从问题识别节点到法律研究节点的有向边
    builder.add_edge("identify_issues", "research_law")
    # 添加从法律研究节点到结束节点的有向边
    builder.add_edge("research_law", END)
    # 编译并返回构建好的状态图
    return builder.compile()